"""Stage 4's opt-in live smoke tests: "opt-in live smoke tests pass without
exposing credentials" (`docs/modernization-plan.md` Stage 4 Gate).

Every other provider test file (`test_providers_anthropic.py`,
`test_providers_openai.py`, `test_providers_huggingface.py`,
`test_provider_secret_leak.py`, `test_provider_timeout_retry_concurrency.py`)
mocks the SDK client entirely and never makes a real network call. Stage 4
verification round 1 correctly found that gap: `docs/stage4/
build_provider_comparison.py` makes real network calls but only exercises the
auth-*rejection* path (see its own docstring) and is a manually-run script,
not part of the pytest suite -- so no real, successful `generate()` call had
ever actually run under this repo's test tooling.

This file closes that gap as real `pytest` tests, each skipped unless BOTH:

1. the provider's real credential env var is set
   (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `HF_TOKEN`), AND
2. `GOVGIS_RUN_LIVE_PROVIDER_SMOKE_TESTS=1` is also set.

Condition 2 is deliberate and load-bearing, not redundant with condition 1:
gating on the credential env var alone would mean a developer with, say, an
ambient `ANTHROPIC_API_KEY` set in their shell for an unrelated CLI tool
would silently spend real money against a real account the moment they ran
the plain `pytest` suite -- exactly the "never use owner-funded ... keys ...
without an explicit ... decision" non-negotiable
(`docs/modernization-plan.md`'s "Secrets, cost, and public abuse") applied to
a developer's own key, not just the Space owner's. Requiring a second,
purpose-specific opt-in makes running these tests a deliberate choice.

Run (example, Anthropic only):
    GOVGIS_RUN_LIVE_PROVIDER_SMOKE_TESTS=1 ANTHROPIC_API_KEY=sk-ant-... \
        .venv/Scripts/python.exe -m pytest tests/test_provider_live_smoke.py

Each test asserts only that a real `generate()` call returns a non-empty
string and that the real credential does not appear in that returned text --
it does not assert anything about answer quality (see
`docs/stage4/provider_comparison.md`'s "What would close the quality-score
gap" for that separate, larger, not-yet-funded effort).
"""

from __future__ import annotations

import os

import pytest

from govgis.providers.anthropic import DEFAULT_FAST_MODEL as ANTHROPIC_DEFAULT_MODEL
from govgis.providers.anthropic import AnthropicProvider
from govgis.providers.huggingface import DEFAULT_MODEL as HF_DEFAULT_MODEL
from govgis.providers.huggingface import HuggingFaceProvider
from govgis.providers.openai import OpenAIProvider

_OPT_IN_ENV_VAR = "GOVGIS_RUN_LIVE_PROVIDER_SMOKE_TESTS"

# Matches `docs/stage4/build_provider_comparison.py`'s OpenAI candidate set
# (`gpt-5.6-luna` is the efficient tier) -- kept as a literal here rather than
# importing that script, since it is a manually-run evidence tool, not a
# reusable module this test suite should depend on.
_OPENAI_DEFAULT_MODEL = "gpt-5.6-luna"

_LIVE_PROMPT = "Reply with exactly the single word: OK"
_LIVE_TIMEOUT_SECONDS = 30.0
_LIVE_MAX_TOKENS = 16


def _opted_in() -> bool:
    return os.environ.get(_OPT_IN_ENV_VAR) == "1"


def _skip_reason(credential_env_var: str) -> str | None:
    if not _opted_in():
        return (
            f"live provider smoke tests are opt-in; set {_OPT_IN_ENV_VAR}=1 and a funded "
            f"{credential_env_var} to run this test (see module docstring)"
        )
    if not os.environ.get(credential_env_var):
        return f"{_OPT_IN_ENV_VAR}=1 but {credential_env_var} is not set"
    return None


_ANTHROPIC_SKIP_REASON = _skip_reason("ANTHROPIC_API_KEY")
_OPENAI_SKIP_REASON = _skip_reason("OPENAI_API_KEY")
_HF_SKIP_REASON = _skip_reason("HF_TOKEN")


@pytest.mark.skipif(_ANTHROPIC_SKIP_REASON is not None, reason=_ANTHROPIC_SKIP_REASON or "")
def test_anthropic_live_generate_returns_text() -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    provider = AnthropicProvider(api_key=api_key)

    result = provider.generate(
        _LIVE_PROMPT,
        model=ANTHROPIC_DEFAULT_MODEL,
        max_tokens=_LIVE_MAX_TOKENS,
        timeout=_LIVE_TIMEOUT_SECONDS,
    )

    assert isinstance(result, str)
    assert result.strip()
    assert api_key not in result


@pytest.mark.skipif(_OPENAI_SKIP_REASON is not None, reason=_OPENAI_SKIP_REASON or "")
def test_openai_live_generate_returns_text() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    provider = OpenAIProvider(api_key=api_key)

    result = provider.generate(
        _LIVE_PROMPT,
        model=_OPENAI_DEFAULT_MODEL,
        max_tokens=_LIVE_MAX_TOKENS,
        timeout=_LIVE_TIMEOUT_SECONDS,
    )

    assert isinstance(result, str)
    assert result.strip()
    assert api_key not in result


@pytest.mark.skipif(_HF_SKIP_REASON is not None, reason=_HF_SKIP_REASON or "")
def test_huggingface_live_generate_returns_text() -> None:
    token = os.environ["HF_TOKEN"]
    provider = HuggingFaceProvider(token=token)

    result = provider.generate(
        _LIVE_PROMPT,
        model=HF_DEFAULT_MODEL,
        max_tokens=_LIVE_MAX_TOKENS,
        timeout=_LIVE_TIMEOUT_SECONDS,
    )

    assert isinstance(result, str)
    assert result.strip()
    assert token not in result
