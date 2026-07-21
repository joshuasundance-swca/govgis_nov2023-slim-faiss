"""Stage 4's dedicated BYOK secret-leak gate.

Per `docs/modernization-plan.md`'s "Secrets, cost, and public abuse" and
Stage 4's Gate: "a forced-failure test (e.g. an invalid test key) asserts no
substring of the test key appears in captured logs, exceptions, or
telemetry."

This is the canonical, cross-provider version of that gate: parametrized
over all three `govgis.providers.*` implementations, each forced to fail via
a mocked SDK/HTTP error whose message and response body *deliberately*
contain the test key (worst case: a naive fix that mishandled an SDK error
message would echo it straight into `str(exc)`), then asserts the key
substring appears nowhere in:

- the raised exception's `str()`;
- the exception's `repr()` (covers a leak via `args`/`__dict__` that a plain
  `str()` might not surface);
- anything written to stdout/stderr during the call (`capsys`);
- anything captured by the logging framework during the call (`caplog`),
  when a caller does exactly what a "standard caller" would do on catching
  the raised `ProviderError`: call `logging.exception(...)` from the
  `except` block. Python's default logging formatter (and pytest's
  `caplog`) renders that call's `exc_info` by walking and printing the full
  `__cause__` chain -- so this is the vector a provider module using
  `raise ... from exc` (preserving the raw SDK exception as `__cause__`,
  which can carry the key) would leak through even though `str()`/`repr()`
  on the wrapped exception alone stay clean;
- `traceback.format_exception(...)` on the caught exception directly (the
  same `__cause__`-chain-walking property as `logging.exception()` above,
  asserted independently of the logging framework -- covers an uncaught
  exception reaching a default traceback handler, not only a caller that
  logs).

Deeper, per-error-path leak coverage for each provider lives in that
provider's own contract test file (`test_providers_anthropic.py`,
`test_providers_openai.py`, `test_providers_huggingface.py`); this file is
the single, explicitly-Stage-4-Gate-shaped deliverable, not a replacement
for that depth.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import openai
import pytest
from huggingface_hub.errors import HfHubHTTPError

import govgis.providers.huggingface as huggingface_provider
from govgis.providers.anthropic import AnthropicProvider
from govgis.providers.base import ProviderError
from govgis.providers.huggingface import DEFAULT_MODEL as HF_DEFAULT_MODEL
from govgis.providers.huggingface import HuggingFaceProvider
from govgis.providers.openai import OpenAIProvider

_TEST_KEY = "SECRET-test-key-11223344-should-never-appear-anywhere"


def _make_anthropic_case(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, text=f"invalid key: {_TEST_KEY}")
    error = anthropic.AuthenticationError(
        message=f"authentication failed for key {_TEST_KEY}",
        response=response,
        body=None,
    )

    class _FailingMessages:
        def create(self, **_kwargs: Any) -> Any:
            raise error

    class _FailingAnthropicClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.messages = _FailingMessages()

    monkeypatch.setattr("govgis.providers.anthropic.anthropic.Anthropic", _FailingAnthropicClient)

    def _run() -> None:
        provider = AnthropicProvider(api_key=_TEST_KEY)
        provider.generate("prompt", model="claude-sonnet-5", max_tokens=64, timeout=15.0)

    return _run


def _make_openai_case(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(401, request=request, text=f"Incorrect API key provided: {_TEST_KEY}")
    error = openai.AuthenticationError(
        message=f"Incorrect API key provided: {_TEST_KEY}",
        response=response,
        body=None,
    )

    class _FailingResponses:
        def create(self, **_kwargs: Any) -> Any:
            raise error

    class _FailingOpenAIClient:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            del api_key, max_retries
            self.responses = _FailingResponses()

    monkeypatch.setattr("govgis.providers.openai.openai.OpenAI", _FailingOpenAIClient)

    def _run() -> None:
        provider = OpenAIProvider(api_key=_TEST_KEY)
        provider.generate("prompt", model="gpt-5.6-terra", max_tokens=64, timeout=15.0)

    return _run


def _make_huggingface_case(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    request = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    response = httpx.Response(
        401,
        request=request,
        text=f"Authorization header contained invalid token {_TEST_KEY}",
    )
    error = HfHubHTTPError(f"401 error, token {_TEST_KEY}", response=response)

    class _FailingInferenceClient:
        def __init__(self, *, model: str, token: str, timeout: float, provider: str) -> None:
            del model, token, timeout, provider

        def chat_completion(self, **_kwargs: Any) -> Any:
            raise error

    monkeypatch.setattr(huggingface_provider, "InferenceClient", _FailingInferenceClient)

    def _run() -> None:
        provider = HuggingFaceProvider(token=_TEST_KEY)
        provider.generate("prompt", model=HF_DEFAULT_MODEL, max_tokens=64, timeout=15.0)

    return _run


_CASES: dict[str, Callable[[pytest.MonkeyPatch], Callable[[], None]]] = {
    "anthropic": _make_anthropic_case,
    "openai": _make_openai_case,
    "huggingface": _make_huggingface_case,
}


@pytest.mark.parametrize("provider_name", sorted(_CASES))
def test_invalid_key_failure_never_leaks_the_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    provider_name: str,
) -> None:
    run = _CASES[provider_name](monkeypatch)
    logger = logging.getLogger(f"test_provider_secret_leak.{provider_name}")

    caught: ProviderError | None = None
    with caplog.at_level(logging.DEBUG):
        try:
            run()
        except ProviderError as exc:
            caught = exc
            # Simulate a standard caller's `logging.exception(...)` /
            # uncaught-propagation path -- see module docstring. Python's
            # default traceback formatting walks and prints the FULL
            # `__cause__` chain, so a provider module that preserves the raw
            # SDK exception as `__cause__` (`raise ... from exc`) would leak
            # the key here even though `str(exc)`/`repr(exc)` on the wrapped
            # exception alone stay clean -- this is the vector `from None`
            # exists to close.
            logger.exception("provider call failed")

    assert caught is not None, f"{provider_name} provider case did not raise a ProviderError"
    formatted_traceback = "".join(traceback.format_exception(caught))

    captured = capsys.readouterr()

    assert _TEST_KEY not in str(caught), "key substring leaked via exception str()"
    assert _TEST_KEY not in repr(caught), "key substring leaked via exception repr()"
    assert _TEST_KEY not in captured.out, "key substring leaked via stdout"
    assert _TEST_KEY not in captured.err, "key substring leaked via stderr"
    assert _TEST_KEY not in caplog.text, (
        "key substring leaked via the logging framework's logging.exception() capture"
    )
    assert _TEST_KEY not in formatted_traceback, (
        "key substring leaked via traceback.format_exception()'s __cause__ chain"
    )
