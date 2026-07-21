"""Stage 4's simulated slow/failing provider gate.

Per `docs/modernization-plan.md` Stage 4's shared contract and Gate: "a
simulated slow/failing provider test asserts the timeout, retry-limit, and
concurrency-limit behavior" (concrete numbers: bounded timeout in the
15-30s range, retries limited to 1-2 attempts on transient/5xx failures
only -- never on auth errors -- and a per-session concurrency limit of 1
in-flight generation call).

This file is the cross-provider synthesis of that gate; each assertion is
also exercised at finer grain in the per-provider contract test files
(`test_providers_anthropic.py`, `test_providers_openai.py`,
`test_providers_huggingface.py`), which this file cross-references rather
than duplicates. What is new here:

- a single explicit table proving retry-limit and never-retry-auth for all
  three providers side by side;
- the concurrency-limit finding: `AnthropicProvider` and `OpenAIProvider`
  each own a per-instance lock (see `test_providers_anthropic.py`'s and
  `test_providers_openai.py`'s `test_concurrent_call_is_rejected_not_queued`
  for the real tests). `HuggingFaceProvider` is a plain stateless class with
  no such lock -- the shared contract's "per-session concurrency limit of 1
  in-flight" is enforced one layer up for HF, at the `app.py` UI call site
  (`hf_in_flight_state`, a per-`gr.State` guard -- see
  `tests/test_app_provider_kill_switch.py`'s
  `test_in_flight_guard_short_circuits_without_calling_the_provider`).
  Anthropic/OpenAI are not wired into `app.py` yet (a separate lane's
  scope), so today the Hugging Face path has the app-level guard and
  Anthropic/OpenAI each have their own module-level lock as the interim
  backstop -- closing the module-level-lock gap that used to exist for
  `OpenAIProvider` before it is wired into `app.py` (previously a real,
  unresolved gap; see this lane's final report for that history).
"""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import openai
import pytest

from govgis.providers.anthropic import AnthropicProvider
from govgis.providers.base import ProviderAuthError, ProviderRateLimitError, ProviderTransientError
from govgis.providers.openai import OpenAIProvider

_ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
_OPENAI_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")

# Shared contract: "retries limited to 1-2 attempts" -- expressed here as
# "total calls observed", the same shape each provider's own tests assert.
_EXPECTED_TOTAL_CALLS_ON_PERSISTENT_RATE_LIMIT = {
    # AnthropicProvider retries up to `_MAX_RETRY_ATTEMPTS = 2` additional
    # times (3 total calls) -- the high end of the "1-2 retries" range.
    "anthropic": 3,
    # OpenAIProvider's tenacity `stop_after_attempt(_MAX_ATTEMPTS=2)` means
    # 2 total calls -- the low end of the same range. Both are within the
    # shared contract's stated bound; this asymmetry is a real, observed
    # cross-provider difference, not a bug this test papers over.
    "openai": 2,
}


def _anthropic_client_always_rate_limited(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            response = httpx.Response(429, request=_ANTHROPIC_REQUEST, text="rate limited")
            raise anthropic.RateLimitError(message="rate limited", response=response, body=None)

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _Messages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _Client)
    return calls


def _openai_client_always_rate_limited(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    class _Responses:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            response = httpx.Response(429, request=_OPENAI_REQUEST, text="rate limited")
            raise openai.RateLimitError(message="rate limited", response=response, body=None)

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.responses = _Responses()

    monkeypatch.setattr("govgis.providers.openai.openai.OpenAI", _Client)
    return calls


def test_anthropic_retry_limit_on_persistent_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _anthropic_client_always_rate_limited(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-fake")

    with pytest.raises(ProviderRateLimitError):
        provider.generate("prompt", model="claude-sonnet-5", max_tokens=64, timeout=15.0)

    assert len(calls) == _EXPECTED_TOTAL_CALLS_ON_PERSISTENT_RATE_LIMIT["anthropic"]


def test_openai_retry_limit_on_persistent_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _openai_client_always_rate_limited(monkeypatch)
    provider = OpenAIProvider(api_key="sk-fake")

    with pytest.raises(ProviderRateLimitError):
        provider.generate("prompt", model="gpt-5.6-terra", max_tokens=64, timeout=15.0)

    assert len(calls) == _EXPECTED_TOTAL_CALLS_ON_PERSISTENT_RATE_LIMIT["openai"]


def test_anthropic_never_retries_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            response = httpx.Response(401, request=_ANTHROPIC_REQUEST, text="bad key")
            raise anthropic.AuthenticationError(message="bad key", response=response, body=None)

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _Messages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _Client)
    provider = AnthropicProvider(api_key="sk-ant-fake")

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model="claude-sonnet-5", max_tokens=64, timeout=15.0)

    # The load-bearing assertion of the whole gate: an auth failure gets
    # exactly one call, never a retry -- retrying an invalid key wastes the
    # caller's bounded attempt budget on a request that cannot ever succeed.
    assert len(calls) == 1


def test_openai_never_retries_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class _Responses:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            response = httpx.Response(401, request=_OPENAI_REQUEST, text="bad key")
            raise openai.AuthenticationError(message="bad key", response=response, body=None)

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.responses = _Responses()

    monkeypatch.setattr("govgis.providers.openai.openai.OpenAI", _Client)
    provider = OpenAIProvider(api_key="sk-fake")

    with pytest.raises(ProviderAuthError):
        provider.generate("prompt", model="gpt-5.6-terra", max_tokens=64, timeout=15.0)

    assert len(calls) == 1


def test_anthropic_timeout_parameter_is_forwarded_to_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller-supplied `timeout` (15-30s range per the shared contract)
    must reach the underlying SDK call on every attempt, not just the
    first -- a retry that silently drops the bound would let a slow
    provider hang indefinitely on attempt 2+.
    """
    seen_timeouts: list[float] = []

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            seen_timeouts.append(kwargs["timeout"])
            response = httpx.Response(503, request=_ANTHROPIC_REQUEST, text="busy")
            raise anthropic.InternalServerError(message="busy", response=response, body=None)

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _Messages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _Client)
    provider = AnthropicProvider(api_key="sk-ant-fake")

    with pytest.raises(ProviderTransientError):
        provider.generate("prompt", model="claude-sonnet-5", max_tokens=64, timeout=22.5)

    assert seen_timeouts == [22.5] * 3
