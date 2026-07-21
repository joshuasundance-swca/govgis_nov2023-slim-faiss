"""Stage 4's "retrieval survives every provider failure" gate, at `app.py`.

Per `docs/modernization-plan.md` Stage 4's Gate: "retrieval survives every
provider failure." `app.py` wires in all three `govgis.providers.*` modules
(Hugging Face via OAuth, Anthropic/OpenAI via BYOK API key); the
architectural claim this gate rests on -- that `govgis.retrieval` has no
dependency on any provider module at all, so a provider failure literally
cannot reach retrieval code -- is proven directly by
`test_retrieval_module_has_no_provider_dependency` below and holds
regardless of which providers are wired into `app.py`. Provider-level
failure isolation (each `generate` call raising only sanitized
`ProviderError` subclasses, never corrupting shared state) is covered by
`test_providers_anthropic.py`/`test_providers_openai.py`/
`test_providers_huggingface.py`.

Also covers, at the same `app.py` layer, for every wired provider:

- the per-session concurrency guard (`{hf,anthropic,openai}_in_flight_state`)
  short-circuits before ever constructing a provider, per the Stage 4 shared
  contract's "per-session concurrency limit of 1 in-flight generation call";
- a provider failure whose message deliberately contains a secret-shaped
  substring never reaches the rendered answer text -- `app.py`'s blanket
  `except ProviderError` always substitutes its own generic message (see
  `_HF_ANSWER_ERROR_MESSAGE`/`_ANTHROPIC_ANSWER_ERROR_MESSAGE`/
  `_OPENAI_ANSWER_ERROR_MESSAGE`), never the provider's own text;
- for Anthropic/OpenAI specifically, the BYOK API key the user typed into
  the request never reaches the rendered answer text either, on a
  provider-failure path where a naive implementation could plausibly echo
  it (see `test_byok_api_key_never_reaches_the_rendered_answer_on_failure`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import gradio as gr
import pytest

import app
from govgis.models import GisRecord, SearchResult
from govgis.providers.anthropic import DEFAULT_FAST_MODEL as ANTHROPIC_DEFAULT_MODEL
from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)
from govgis.providers.huggingface import DEFAULT_MODEL
from govgis.providers.openai import DEFAULT_FAST_MODEL as OPENAI_DEFAULT_MODEL
from govgis.retrieval import RetrievalIndex

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_RECORD = GisRecord(
    id="rec-flood",
    name="Statewide Flood Hazard Zones",
    type="FeatureServer",
    url="https://gis.example.gov/arcgis/rest/services/Flood/FeatureServer/0",
    description="FEMA flood hazard zone polygons.",
    metadata_text="name: Statewide Flood Hazard Zones\ntype: FeatureServer",
)
_RESULTS = [SearchResult(record=_RECORD, score=0.91)]

_FAKE_OAUTH_TOKEN = gr.OAuthToken(
    token="fake-hf-oauth-token-should-never-leak",  # noqa: S106
    scope="inference-api",
    expires_at=9999999999,
)


class _StubRetrievalIndex:
    """Duck-types `RetrievalIndex.search` without loading real FAISS/model artifacts."""

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        del top_k
        if not query.strip():
            return []
        return _RESULTS


def _install_stub_retrieval_index(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = cast("RetrievalIndex", _StubRetrievalIndex())
    monkeypatch.setattr(app, "_get_retrieval_index", lambda: stub)


def test_retrieval_module_has_no_provider_dependency() -> None:
    """Structural proof backing the kill-switch claim: `govgis.retrieval`
    never imports or references any `govgis.providers.*` symbol, so a
    provider failure -- of any shape, for any provider -- literally cannot
    execute inside retrieval code. This is the property that makes the
    per-provider `app.py`-level test below ("HF fails, search still works")
    a fact about the whole architecture, not a coincidence of the current
    wiring.
    """
    source = (REPOSITORY_ROOT / "govgis" / "retrieval.py").read_text(encoding="utf-8").lower()
    assert "provider" not in source


def test_handle_search_returns_results_with_stub_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_retrieval_index(monkeypatch)

    outputs = list(app._handle_search("flood zones", 3))

    assert outputs[-1] != app._IDLE_MESSAGE
    assert outputs[-1] != app._ERROR_MESSAGE
    assert "Statewide Flood Hazard Zones" in outputs[-1]


@pytest.mark.parametrize(
    "provider_error",
    [
        ProviderTimeoutError("timed out"),
        ProviderAuthError("bad token"),
        ProviderRateLimitError("rate limited"),
        ProviderTransientError("transient failure"),
        ProviderInvalidRequestError("bad model"),
        ProviderError("unexpected"),
    ],
)
def test_search_still_returns_results_after_every_provider_failure_shape(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: ProviderError,
) -> None:
    """The literal Stage 4 Gate assertion: for every `ProviderError` shape
    the HF answer-synthesis path can raise, (1) answer generation degrades
    to the generic error message and (2) retrieval -- called again, as a
    user would after an answer-generation failure -- still returns real
    results. Steps 1/2 share no state (`_handle_search` never touches
    `HuggingFaceProvider`), so this also demonstrates the kill-switch
    end-to-end, not just architecturally.
    """
    _install_stub_retrieval_index(monkeypatch)

    class _FailingProvider:
        def __init__(self, *, token: str) -> None:
            del token

        def generate(self, *_args: Any, **_kwargs: Any) -> str:
            raise provider_error

    monkeypatch.setattr(app, "HuggingFaceProvider", _FailingProvider)

    answer_outputs = list(
        app._handle_generate_hf_answer("flood zones", DEFAULT_MODEL, False, _FAKE_OAUTH_TOKEN),
    )
    final_answer_text, final_in_flight = answer_outputs[-1]
    assert final_answer_text == app._HF_ANSWER_ERROR_MESSAGE
    assert final_in_flight is False

    search_outputs = list(app._handle_search("flood zones", 3))
    assert "Statewide Flood Hazard Zones" in search_outputs[-1]


def test_in_flight_guard_short_circuits_without_calling_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnexpectedProvider:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("HuggingFaceProvider must not be constructed while in_flight=True")

    monkeypatch.setattr(app, "HuggingFaceProvider", _UnexpectedProvider)

    outputs = list(
        app._handle_generate_hf_answer("flood zones", DEFAULT_MODEL, True, _FAKE_OAUTH_TOKEN),
    )

    text, in_flight = outputs[-1]
    assert text == app._HF_ALREADY_GENERATING_MESSAGE
    assert in_flight is True


def test_provider_failure_message_never_reaches_the_rendered_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_retrieval_index(monkeypatch)
    secret = "hf_oauth_leak_probe_should_never_render_00998877"  # noqa: S105

    class _LeakyProvider:
        def __init__(self, *, token: str) -> None:
            del token

        def generate(self, *_args: Any, **_kwargs: Any) -> str:
            raise ProviderTransientError(f"upstream said: {secret}")

    monkeypatch.setattr(app, "HuggingFaceProvider", _LeakyProvider)

    outputs = list(
        app._handle_generate_hf_answer("flood zones", DEFAULT_MODEL, False, _FAKE_OAUTH_TOKEN),
    )
    final_text, _final_in_flight = outputs[-1]

    assert secret not in final_text
    assert final_text == app._HF_ANSWER_ERROR_MESSAGE


# --- BYOK (Anthropic/OpenAI) coverage ---------------------------------------
#
# `_handle_generate_anthropic_answer`/`_handle_generate_openai_answer` share
# `_handle_generate_hf_answer`'s shape (concurrency guard, retrieval-then-
# generate, sanitized error substitution) but take a user-typed `api_key: str`
# instead of a `gr.OAuthToken`, so the fixtures below are BYOK-specific rather
# than a straight parametrization of the HF tests above.
_TEST_BYOK_API_KEY = "byok-test-key-should-never-leak-99887766"

_BYOK_HANDLERS: dict[str, Callable[[str, str, str, bool], Iterator[tuple[str, bool]]]] = {
    "anthropic": app._handle_generate_anthropic_answer,
    "openai": app._handle_generate_openai_answer,
}
_BYOK_PROVIDER_CLASS_ATTRS = {"anthropic": "AnthropicProvider", "openai": "OpenAIProvider"}
_BYOK_DEFAULT_MODELS = {"anthropic": ANTHROPIC_DEFAULT_MODEL, "openai": OPENAI_DEFAULT_MODEL}
_BYOK_ERROR_MESSAGES = {
    "anthropic": app._ANTHROPIC_ANSWER_ERROR_MESSAGE,
    "openai": app._OPENAI_ANSWER_ERROR_MESSAGE,
}
_BYOK_ALREADY_GENERATING_MESSAGES = {
    "anthropic": app._ANTHROPIC_ALREADY_GENERATING_MESSAGE,
    "openai": app._OPENAI_ALREADY_GENERATING_MESSAGE,
}
_BYOK_NO_KEY_MESSAGES = {
    "anthropic": app._ANTHROPIC_NO_KEY_MESSAGE,
    "openai": app._OPENAI_NO_KEY_MESSAGE,
}


@pytest.mark.parametrize("provider_name", sorted(_BYOK_HANDLERS))
@pytest.mark.parametrize(
    "provider_error",
    [
        ProviderTimeoutError("timed out"),
        ProviderAuthError("bad key"),
        ProviderRateLimitError("rate limited"),
        ProviderTransientError("transient failure"),
        ProviderInvalidRequestError("bad model"),
        ProviderError("unexpected"),
    ],
)
def test_search_still_returns_results_after_every_byok_provider_failure_shape(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    provider_error: ProviderError,
) -> None:
    """BYOK counterpart of
    `test_search_still_returns_results_after_every_provider_failure_shape`
    above: for every `ProviderError` shape an Anthropic/OpenAI answer call can
    raise, (1) answer generation degrades to the generic error message and
    (2) retrieval still returns real results afterward.
    """
    _install_stub_retrieval_index(monkeypatch)

    class _FailingProvider:
        def __init__(self, *, api_key: str) -> None:
            del api_key

        def generate(self, *_args: Any, **_kwargs: Any) -> str:
            raise provider_error

    monkeypatch.setattr(app, _BYOK_PROVIDER_CLASS_ATTRS[provider_name], _FailingProvider)

    handler = _BYOK_HANDLERS[provider_name]
    default_model = _BYOK_DEFAULT_MODELS[provider_name]
    answer_outputs = list(
        handler("flood zones", _TEST_BYOK_API_KEY, default_model, False),
    )
    final_answer_text, final_in_flight = answer_outputs[-1]
    assert final_answer_text == _BYOK_ERROR_MESSAGES[provider_name]
    assert final_in_flight is False

    search_outputs = list(app._handle_search("flood zones", 3))
    assert "Statewide Flood Hazard Zones" in search_outputs[-1]


@pytest.mark.parametrize("provider_name", sorted(_BYOK_HANDLERS))
def test_byok_in_flight_guard_short_circuits_without_calling_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    class _UnexpectedProvider:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError(
                f"{_BYOK_PROVIDER_CLASS_ATTRS[provider_name]} must not be constructed "
                "while in_flight=True",
            )

    monkeypatch.setattr(app, _BYOK_PROVIDER_CLASS_ATTRS[provider_name], _UnexpectedProvider)

    handler = _BYOK_HANDLERS[provider_name]
    default_model = _BYOK_DEFAULT_MODELS[provider_name]
    outputs = list(handler("flood zones", _TEST_BYOK_API_KEY, default_model, True))

    text, in_flight = outputs[-1]
    assert text == _BYOK_ALREADY_GENERATING_MESSAGES[provider_name]
    assert in_flight is True


@pytest.mark.parametrize("provider_name", sorted(_BYOK_HANDLERS))
def test_missing_byok_api_key_yields_generic_message_without_calling_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    class _UnexpectedProvider:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError(
                f"{_BYOK_PROVIDER_CLASS_ATTRS[provider_name]} must not be constructed "
                "with an empty API key",
            )

    monkeypatch.setattr(app, _BYOK_PROVIDER_CLASS_ATTRS[provider_name], _UnexpectedProvider)

    handler = _BYOK_HANDLERS[provider_name]
    default_model = _BYOK_DEFAULT_MODELS[provider_name]
    outputs = list(handler("flood zones", "   ", default_model, False))

    text, in_flight = outputs[-1]
    assert text == _BYOK_NO_KEY_MESSAGES[provider_name]
    assert in_flight is False


@pytest.mark.parametrize("provider_name", sorted(_BYOK_HANDLERS))
def test_byok_api_key_never_reaches_the_rendered_answer_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    """A provider failure whose message deliberately echoes back the caller's
    own API key (the exact shape Anthropic's/OpenAI's real SDKs have been
    observed to produce for a rejected key -- see
    `govgis/providers/openai.py`'s module docstring) must never reach the
    rendered answer text: `app.py`'s blanket `except ProviderError`
    substitutes a fixed generic message regardless of what the caught
    exception's own `str()` contains.
    """
    _install_stub_retrieval_index(monkeypatch)

    class _LeakyProvider:
        def __init__(self, *, api_key: str) -> None:
            del api_key

        def generate(self, *_args: Any, **_kwargs: Any) -> str:
            raise ProviderAuthError(f"rejected key: {_TEST_BYOK_API_KEY}")

    monkeypatch.setattr(app, _BYOK_PROVIDER_CLASS_ATTRS[provider_name], _LeakyProvider)

    handler = _BYOK_HANDLERS[provider_name]
    default_model = _BYOK_DEFAULT_MODELS[provider_name]
    outputs = list(handler("flood zones", _TEST_BYOK_API_KEY, default_model, False))
    final_text, _final_in_flight = outputs[-1]

    assert _TEST_BYOK_API_KEY not in final_text
    assert final_text == _BYOK_ERROR_MESSAGES[provider_name]
