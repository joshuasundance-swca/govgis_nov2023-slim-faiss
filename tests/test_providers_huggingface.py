"""Acceptance tests for `govgis.providers.huggingface` (Stage 4).

Mocks `huggingface_hub.InferenceClient` entirely -- these tests never make a
real network call -- so they can assert this module's own contract:

- only `CURATED_MODELS` are accepted, and every other model is rejected
  before any client is constructed;
- every `huggingface_hub`/`httpx` exception this module can encounter is
  translated to the correct `govgis.providers.base.ProviderError` subclass,
  never re-raised or leaked raw (see `docs/modernization-plan.md`'s BYOK
  non-negotiable: a raw provider exception can carry the request, including
  the caller's access token);
- retries happen only for `ProviderTransientError`/`ProviderRateLimitError`,
  bounded to the shared contract's 1-2 attempts, and never for auth or
  invalid-request failures;
- no test-supplied token substring ever appears in a raised exception's
  message, across every error path this module handles -- the same property
  Stage 4's dedicated BYOK secret-leak test (a separate lane's deliverable)
  checks at the app level, verified here at the unit level for this module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

import govgis.providers.huggingface as hf_provider
from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from govgis.providers.huggingface import CURATED_MODELS, DEFAULT_MODEL, HuggingFaceProvider

_SECRET_TOKEN = "hf_oauth_test_token_should_never_leak_00112233"  # noqa: S105


def _http_error(status_code: int, body: str = "") -> HfHubHTTPError:
    request = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=body)
    return HfHubHTTPError(f"HTTP {status_code} error", response=response)


def _completion(content: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _install_scripted_client(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> list[dict[str, Any]]:
    """Replace `hf_provider.InferenceClient` with a fake driven by `script`.

    Each `chat_completion` call pops and either raises (if the popped value
    is a `BaseException`) or returns (otherwise) the next scripted outcome.
    Returns the shared `calls` list so tests can assert call count and
    arguments -- including proving a retry actually happened.
    """
    calls: list[dict[str, Any]] = []

    class _ScriptedInferenceClient:
        def __init__(self, *, model: str, token: str, timeout: float, provider: str) -> None:
            self._model = model
            self._token = token
            self._timeout = timeout
            self._provider = provider

        def chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            model: str,
            max_tokens: int,
            temperature: float | None,
        ) -> Any:
            calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "token": self._token,
                    "timeout": self._timeout,
                    "provider": self._provider,
                },
            )
            outcome = script.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(hf_provider, "InferenceClient", _ScriptedInferenceClient)
    return calls


def _fails_if_called(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a client that fails the test if it is ever constructed."""

    class _UnexpectedInferenceClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("InferenceClient must not be constructed for this case")

    monkeypatch.setattr(hf_provider, "InferenceClient", _UnexpectedInferenceClient)
    return []


def test_curated_models_are_nonempty_and_default_is_curated() -> None:
    assert CURATED_MODELS
    assert DEFAULT_MODEL in CURATED_MODELS


def test_constructor_rejects_empty_token() -> None:
    with pytest.raises(ProviderAuthError):
        HuggingFaceProvider(token="")


def test_generate_rejects_uncurated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _fails_if_called(monkeypatch)
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)
    with pytest.raises(ProviderInvalidRequestError):
        provider.generate(
            "prompt",
            model="not-a-curated-model",
            max_tokens=64,
            timeout=15.0,
        )


def test_generate_returns_completion_content(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(monkeypatch, [_completion("a grounded answer")])
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    result = provider.generate(
        "what floods are mapped?",
        model=DEFAULT_MODEL,
        temperature=0.2,
        max_tokens=128,
        timeout=20.0,
    )

    assert result == "a grounded answer"
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.2
    assert call["token"] == _SECRET_TOKEN
    assert call["timeout"] == 20.0
    assert call["provider"] == "auto"
    assert call["messages"] == [{"role": "user", "content": "what floods are mapped?"}]


def test_generate_translates_timeout_and_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [InferenceTimeoutError("timed out")],
    )
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderTimeoutError):
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


@pytest.mark.parametrize("status_code", [401, 403])
def test_generate_translates_auth_errors_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    calls = _install_scripted_client(monkeypatch, [_http_error(status_code)])
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_translates_invalid_request_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(monkeypatch, [_http_error(400, body="bad request")])
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderInvalidRequestError):
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_generate_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_http_error(429), _completion("ok after retry")],
    )
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    result = provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert result == "ok after retry"
    assert len(calls) == 2


def test_generate_retries_rate_limit_up_to_shared_contract_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_client(monkeypatch, [_http_error(429), _http_error(429)])
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderRateLimitError):
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    # Shared contract: "retries limited to 1-2 attempts" -- 2 total calls,
    # not an unbounded retry loop.
    assert len(calls) == 2


def test_generate_retries_server_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [_http_error(503), _completion("recovered")],
    )
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    result = provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_retries_network_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(
        monkeypatch,
        [httpx.ConnectError("connection reset"), _completion("recovered")],
    )
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    result = provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert result == "recovered"
    assert len(calls) == 2


def test_generate_raises_on_empty_completion_content(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_client(monkeypatch, [_completion(None)])
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderError):
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    # Not a retryable category (ProviderTransientError/ProviderRateLimitError) --
    # a model returning no content is not a transport-level failure.
    assert len(calls) == 1


@pytest.mark.parametrize(
    "script",
    [
        [_http_error(401)],
        [_http_error(403)],
        [_http_error(400, body=f"echoing back {_SECRET_TOKEN}")],
        [_http_error(429), _http_error(429)],
        [_http_error(503), _http_error(503)],
        [InferenceTimeoutError("timed out")],
        [_completion(None)],
    ],
)
def test_no_error_message_ever_contains_the_token(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> None:
    _install_scripted_client(monkeypatch, script)
    provider = HuggingFaceProvider(token=_SECRET_TOKEN)

    with pytest.raises(ProviderError) as excinfo:
        provider.generate("prompt", model=DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    assert _SECRET_TOKEN not in str(excinfo.value)
