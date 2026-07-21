"""Acceptance tests for `govgis.providers.anthropic` (Stage 4).

Mocks `anthropic.Anthropic` entirely -- these tests never make a real
network call -- so they can assert this module's own contract, mirroring
`tests/test_providers_huggingface.py`'s shape for the sibling provider:

- every `anthropic.AnthropicError` this module can encounter is translated
  to the correct `govgis.providers.base.ProviderError` subclass, never
  re-raised or leaked raw (see `docs/modernization-plan.md`'s BYOK
  non-negotiable: a raw provider exception can carry the request, including
  the caller's API key);
- retries happen only for `ProviderTransientError`/`ProviderTimeoutError`/
  `ProviderRateLimitError` (this module's own retry set -- see
  `govgis/providers/anthropic.py`'s `_generate_with_retry`), bounded to the
  shared contract's 1-2 retries (3 total attempts), and never for auth or
  invalid-request failures;
- the per-instance concurrency guard (`AnthropicProvider`'s lock) rejects a
  second concurrent call rather than queuing it;
- no test-supplied key substring ever appears in a raised exception's
  message, across every error path this module handles -- the same property
  Stage 4's dedicated BYOK secret-leak test (`test_provider_secret_leak.py`)
  checks at a shallower, cross-provider level.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from govgis.providers.anthropic import DEFAULT_QUALITY_MODEL, AnthropicProvider
from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)

_SECRET_KEY = "sk-ant-api03-test-key-should-never-leak-00112233"  # noqa: S105
_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(
    error_type: type[anthropic.APIStatusError],
    status_code: int,
    *,
    message: str = "",
) -> anthropic.APIStatusError:
    response = httpx.Response(status_code, request=_REQUEST, text=message)
    return error_type(message=message, response=response, body=None)


def _text_message(text: str = "a grounded answer", *, stop_reason: str = "end_turn") -> Any:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


def _install_scripted_client(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> list[dict[str, Any]]:
    """Replace `anthropic.Anthropic` with a fake driven by `script`.

    Each `messages.create` call pops and either raises (if the popped value
    is a `BaseException`) or returns (otherwise) the next scripted outcome.
    """
    calls: list[dict[str, Any]] = []

    class _ScriptedMessages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            outcome = script.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _ScriptedAnthropicClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _ScriptedMessages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _ScriptedAnthropicClient)
    return calls


def _fails_if_called(monkeypatch: pytest.MonkeyPatch) -> None:
    class _UnexpectedAnthropicClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("anthropic.Anthropic must not be constructed for this case")

    monkeypatch.setattr(
        "govgis.providers.anthropic.anthropic.Anthropic",
        _UnexpectedAnthropicClient,
    )


def test_constructor_rejects_empty_key() -> None:
    with pytest.raises(ProviderAuthError):
        AnthropicProvider(api_key="")


def test_constructor_rejects_empty_key_does_not_construct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fails_if_called(monkeypatch)
    with pytest.raises(ProviderAuthError):
        AnthropicProvider(api_key="   ")


def test_generate_returns_extracted_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(monkeypatch, [_text_message("a grounded answer")])
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    result = provider.generate(
        "what floods are mapped?",
        model=DEFAULT_QUALITY_MODEL,
        temperature=0.2,
        max_tokens=128,
        timeout=20.0,
    )

    assert result == "a grounded answer"
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == DEFAULT_QUALITY_MODEL
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.2
    assert call["timeout"] == 20.0
    assert call["messages"] == [{"role": "user", "content": "what floods are mapped?"}]


def test_generate_omits_temperature_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(monkeypatch, [_text_message("ok")])
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert "temperature" not in calls[0]


def test_generate_raises_on_refusal_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_scripted_client(monkeypatch, [_text_message("", stop_reason="refusal")])
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)


def test_generate_translates_authentication_error_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.AuthenticationError, 401, message=f"bad key {_SECRET_KEY}")],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_translates_permission_denied_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.PermissionDeniedError, 403)],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_translates_not_found_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.NotFoundError, 404)],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model="not-a-real-model", max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_translates_bad_request_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.BadRequestError, 400)],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.RateLimitError, 429), _text_message("ok after retry")],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert result == "ok after retry"
    assert len(calls) == 2


def test_generate_retries_rate_limit_up_to_shared_contract_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [
            _status_error(anthropic.RateLimitError, 429),
            _status_error(anthropic.RateLimitError, 429),
            _status_error(anthropic.RateLimitError, 429),
        ],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderRateLimitError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    # Shared contract: "retries limited to 1-2 attempts" -- this module
    # counts retries after the initial call, so 3 total calls (1 initial +
    # 2 retries), never an unbounded retry loop.
    assert len(calls) == 3


def test_generate_retries_server_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_status_error(anthropic.InternalServerError, 503), _text_message("recovered")],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_retries_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Note: this module's retry set includes ProviderTimeoutError (see
    # `_generate_with_retry`) -- unlike `govgis.providers.openai`/
    # `.huggingface`, which deliberately do NOT retry timeouts (only
    # transient/rate-limit failures). This is a real, observed inconsistency
    # across providers, not asserted here as "correct" -- see this lane's
    # final report.
    calls = _install_scripted_client(
        monkeypatch,
        [anthropic.APITimeoutError(request=_REQUEST), _text_message("recovered")],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_raises_provider_timeout_error_when_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [
            anthropic.APITimeoutError(request=_REQUEST),
            anthropic.APITimeoutError(request=_REQUEST),
            anthropic.APITimeoutError(request=_REQUEST),
        ],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderTimeoutError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 3


def test_generate_retries_connection_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [
            anthropic.APIConnectionError(message="unreachable", request=_REQUEST),
            _text_message("recovered"),
        ],
    )
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    result = provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_concurrent_call_is_rejected_not_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-instance lock backstop (Stage 4 shared contract: 1 in-flight)."""
    entered = threading.Event()
    release = threading.Event()

    class _BlockingMessages:
        def create(self, **_kwargs: Any) -> Any:
            entered.set()
            release.wait(timeout=5.0)
            return _text_message("slow answer")

    class _BlockingAnthropicClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _BlockingMessages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _BlockingAnthropicClient)
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    results: list[str | BaseException] = []

    def _call() -> None:
        try:
            answer = provider.generate(
                "prompt",
                model=DEFAULT_QUALITY_MODEL,
                max_tokens=64,
                timeout=15.0,
            )
            results.append(answer)
        except BaseException as exc:
            results.append(exc)

    first = threading.Thread(target=_call)
    first.start()
    assert entered.wait(timeout=5.0), "first call never reached the (blocking) SDK call"

    # Second call must be rejected immediately, not queued behind the first.
    with pytest.raises(ProviderTransientError):
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    release.set()
    first.join(timeout=5.0)
    assert results == ["slow answer"]


@pytest.mark.parametrize(
    "script",
    [
        [_status_error(anthropic.AuthenticationError, 401, message=f"bad key {_SECRET_KEY}")],
        [_status_error(anthropic.PermissionDeniedError, 403, message=_SECRET_KEY)],
        [
            _status_error(anthropic.RateLimitError, 429, message=_SECRET_KEY),
            _status_error(anthropic.RateLimitError, 429, message=_SECRET_KEY),
            _status_error(anthropic.RateLimitError, 429, message=_SECRET_KEY),
        ],
        [
            _status_error(anthropic.InternalServerError, 503, message=_SECRET_KEY),
            _status_error(anthropic.InternalServerError, 503, message=_SECRET_KEY),
            _status_error(anthropic.InternalServerError, 503, message=_SECRET_KEY),
        ],
        [anthropic.APIConnectionError(message=f"failed for {_SECRET_KEY}", request=_REQUEST)] * 3,
        [_status_error(anthropic.NotFoundError, 404, message=_SECRET_KEY)],
    ],
)
def test_no_error_message_ever_contains_the_key(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> None:
    _install_scripted_client(monkeypatch, list(script))
    provider = AnthropicProvider(api_key=_SECRET_KEY)

    with pytest.raises(ProviderError) as excinfo:
        provider.generate("prompt", model=DEFAULT_QUALITY_MODEL, max_tokens=64, timeout=15.0)

    assert _SECRET_KEY not in str(excinfo.value)
