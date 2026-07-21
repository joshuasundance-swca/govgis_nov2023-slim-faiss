"""Acceptance tests for `govgis.providers.openai` (Stage 4).

Mocks `openai.OpenAI` entirely -- these tests never make a real network
call -- so they can assert this module's own contract, mirroring
`tests/test_providers_huggingface.py`/`tests/test_providers_anthropic.py`'s
shape for the sibling providers:

- every `openai.OpenAIError` this module can encounter is translated to the
  correct `govgis.providers.base.ProviderError` subclass, never re-raised or
  leaked raw (see `docs/modernization-plan.md`'s BYOK non-negotiable);
- retries happen only for `ProviderTransientError`/`ProviderRateLimitError`
  (this module's `_RETRYABLE_ERRORS` -- deliberately narrower than
  `govgis.providers.anthropic`'s retry set, which also retries
  `ProviderTimeoutError`; see `test_generate_does_not_retry_timeout` below
  and this lane's final report for that cross-provider inconsistency),
  bounded to the shared contract's 1-2 attempts, and never for auth or
  invalid-request failures;
- `store=False` is always passed (the plan's explicit instruction);
- no test-supplied key substring ever appears in a raised exception's
  message, across every error path this module handles.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)
from govgis.providers.openai import OpenAIProvider

_SECRET_KEY = "sk-test-key-should-never-leak-00112233"  # noqa: S105
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")
_MODEL = "gpt-5.6-terra"


def _status_error(
    error_type: type[openai.APIStatusError],
    status_code: int,
    *,
    message: str = "",
) -> openai.APIStatusError:
    response = httpx.Response(status_code, request=_REQUEST, text=message)
    return error_type(message=message, response=response, body=None)


def _response(text: str = "a grounded answer", *, status: str | None = "completed") -> Any:
    return SimpleNamespace(output_text=text, status=status)


def _install_scripted_client(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> list[dict[str, Any]]:
    """Replace `openai.OpenAI` with a fake driven by `script`.

    `_generate_once` constructs a fresh client per call (see
    `govgis/providers/openai.py`), so this records one entry in `calls` per
    `client.responses.create` invocation, across however many client
    instances tenacity's retry loop constructs.
    """
    calls: list[dict[str, Any]] = []

    class _ScriptedResponses:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            outcome = script.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _ScriptedOpenAIClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.responses = _ScriptedResponses()

    monkeypatch.setattr("govgis.providers.openai.openai.OpenAI", _ScriptedOpenAIClient)
    return calls


def test_generate_returns_output_text_and_passes_store_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(monkeypatch, [_response("a grounded answer")])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate(
        "what floods are mapped?",
        model=_MODEL,
        temperature=0.2,
        max_tokens=128,
        timeout=20.0,
    )

    assert result == "a grounded answer"
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == _MODEL
    assert call["input"] == "what floods are mapped?"
    assert call["max_output_tokens"] == 128
    assert call["store"] is False
    assert call["timeout"] == 20.0
    assert call["temperature"] == 0.2


def test_generate_omits_temperature_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(monkeypatch, [_response("ok")])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert "temperature" not in calls[0]


def test_generate_returns_partial_text_when_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_scripted_client(monkeypatch, [_response("partial...", status="incomplete")])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert result == "partial..."


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued", "in_progress"])
def test_generate_raises_transient_for_non_completed_status(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    _install_scripted_client(monkeypatch, [_response("", status=status), _response("ok")])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    # ProviderTransientError from a bad `status` is retried like any other
    # transient failure (it is in `_RETRYABLE_ERRORS`), so a subsequent
    # success is returned rather than the failure surfacing to the caller.
    assert result == "ok"


def test_generate_translates_authentication_error_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.AuthenticationError, 401, message=f"bad key {_SECRET_KEY}")],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_translates_permission_denied_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.PermissionDeniedError, 403)],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


@pytest.mark.parametrize(
    "error_type",
    [
        openai.BadRequestError,
        openai.NotFoundError,
        openai.ConflictError,
        openai.UnprocessableEntityError,
    ],
)
def test_generate_translates_invalid_request_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[openai.APIStatusError],
) -> None:
    status_map = {
        openai.BadRequestError: 400,
        openai.NotFoundError: 404,
        openai.ConflictError: 409,
        openai.UnprocessableEntityError: 422,
    }
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(error_type, status_map[error_type])],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.RateLimitError, 429), _response("ok after retry")],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert result == "ok after retry"
    assert len(calls) == 2


def test_generate_retries_rate_limit_up_to_shared_contract_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.RateLimitError, 429), _status_error(openai.RateLimitError, 429)],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderRateLimitError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    # Shared contract: "retries limited to 1-2 attempts" -- this module's
    # `_MAX_ATTEMPTS = 2` means 2 TOTAL calls (1 initial + 1 retry), the low
    # end of that range.
    assert len(calls) == 2


def test_generate_retries_internal_server_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.InternalServerError, 503), _response("recovered")],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_retries_connection_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [
            openai.APIConnectionError(message="unreachable", request=_REQUEST),
            _response("recovered"),
        ],
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_does_not_retry_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately narrower than `govgis.providers.anthropic`: `ProviderTimeoutError`
    is not in `_RETRYABLE_ERRORS`, so a single timeout raises immediately.
    See this module's docstring / this lane's final report for the
    cross-provider inconsistency this documents rather than resolves.
    """
    calls = _install_scripted_client(monkeypatch, [openai.APITimeoutError(request=_REQUEST)])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderTimeoutError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_status_code_fallback_maps_by_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # A status-coded failure not matched by any specific `except` clause
    # above (e.g. an SDK version skew): classified via `APIStatusError`'s
    # generic branch by raw status code.
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(openai.APIStatusError, 418)],  # teapot: nothing else matches this
    )
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_catch_all_openai_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class _WeirdOpenAIError(openai.OpenAIError):
        pass

    calls = _install_scripted_client(monkeypatch, [_WeirdOpenAIError("unexpected shape")])
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderError):
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


@pytest.mark.parametrize(
    "script",
    [
        [_status_error(openai.AuthenticationError, 401, message=f"bad key {_SECRET_KEY}")],
        [_status_error(openai.PermissionDeniedError, 403, message=_SECRET_KEY)],
        [_status_error(openai.RateLimitError, 429, message=_SECRET_KEY)] * 2,
        [_status_error(openai.InternalServerError, 503, message=_SECRET_KEY)] * 2,
        [openai.APIConnectionError(message=f"failed for {_SECRET_KEY}", request=_REQUEST)] * 2,
        [openai.APITimeoutError(request=_REQUEST)],
        [_status_error(openai.BadRequestError, 400, message=_SECRET_KEY)],
    ],
)
def test_no_error_message_ever_contains_the_key(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> None:
    _install_scripted_client(monkeypatch, list(script))
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderError) as excinfo:
        provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)

    assert _SECRET_KEY not in str(excinfo.value)


def test_provider_repr_never_includes_the_key() -> None:
    provider = OpenAIProvider(api_key=_SECRET_KEY)
    assert _SECRET_KEY not in repr(provider)


def test_concurrent_call_is_rejected_not_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-instance lock backstop (Stage 4 shared contract: 1 in-flight).

    Mirrors `test_providers_anthropic.py`'s
    `test_concurrent_call_is_rejected_not_queued` -- this module's own
    module-level lock closes the gap `test_provider_timeout_retry_concurrency.py`
    previously documented (`OpenAIProvider` had neither this lock nor an
    app-level guard).
    """
    entered = threading.Event()
    release = threading.Event()

    class _BlockingResponses:
        def create(self, **_kwargs: Any) -> Any:
            entered.set()
            release.wait(timeout=5.0)
            return _response("slow answer")

    class _BlockingOpenAIClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.responses = _BlockingResponses()

    monkeypatch.setattr("govgis.providers.openai.openai.OpenAI", _BlockingOpenAIClient)
    provider = OpenAIProvider(api_key=_SECRET_KEY)

    results: list[str | BaseException] = []

    def _call() -> None:
        try:
            answer = provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)
            results.append(answer)
        except BaseException as exc:
            results.append(exc)

    first = threading.Thread(target=_call)
    first.start()
    try:
        assert entered.wait(timeout=5.0), "first call never reached the (blocking) SDK call"

        # Second call must be rejected immediately, not queued behind the first.
        with pytest.raises(ProviderTransientError):
            provider.generate("prompt", model=_MODEL, max_tokens=64, timeout=15.0)
    finally:
        release.set()
        first.join(timeout=5.0)
