# Stage 4 provider/model comparison — 2026-07-20 (re-confirmed 2026-07-21)

Per `docs/modernization-plan.md` Stage 4's action "decide provider/model
defaults from a recorded, script-generated comparison across candidate
models" and Gate: "the provider/model comparison table exists in the repo
and the chosen defaults are traceable to it — a functional-but-unbenchmarked
default does not satisfy this gate." This document is that table. Raw,
machine-generated evidence backing every non-prose number below lives in
[`docs/stage4/evidence/`](evidence/); it was produced by
[`build_provider_comparison.py`](build_provider_comparison.py) (re-run:
`python docs/stage4/build_provider_comparison.py`).

## Status summary (read this before the table)

- **Retrieval-answer quality score: UNVERIFIED for every provider/model
  candidate below, re-confirmed 2026-07-21.** Measuring it for real requires
  making genuine generation calls against `docs/stage0/query_set.json`'s 17
  representative/difficult queries with a funded Anthropic API key, a
  funded OpenAI API key, and a signed-in Hugging Face OAuth token — **none
  of the three has been available in any environment this table has been
  built or re-verified in**, including a 2026-07-21 re-check that
  specifically re-ran `build_provider_comparison.py` after extending it
  (see below) to confirm the blocker still holds, not just re-asserted the
  2026-07-20 finding from memory (confirmed both times by this repo's own
  `build_provider_comparison.py`, which checks
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`HF_TOKEN` and records the result in
  each evidence file's `quality_score_status` field; all three were absent
  both times). No quality number below is estimated, guessed, or
  fabricated — every quality cell reads UNVERIFIED with this exact blocking
  reason, per this lane's brief ("marking quality-score cells UNVERIFIED
  with that blocking reason rather than fabricating numbers"). **This is the
  one piece of this Stage 4 action that remains genuinely open** — see
  "What would close the quality-score gap" below.
- **As of 2026-07-21, `build_provider_comparison.py` was extended to
  actually run this benchmark**, not just describe it in prose: given a
  funded credential, it now loads the real Stage 2 retrieval index, retrieves
  real results for all 17 scored queries, makes a real `generate()` call per
  query, and records real latency plus the raw answer text (see the script's
  own module docstring). This was smoke-tested end-to-end in this session
  with a deliberately fake Anthropic key (real retrieval, real network call
  to `api.anthropic.com`, real per-query `ProviderAuthError` capture, zero
  key-substring leak in the resulting evidence file — that smoke-test
  evidence file was discarded afterward, not committed, to avoid being
  mistaken for a real measurement run) — the plumbing works; **only the
  credential itself remains the blocker**. It deliberately still does not
  compute an automated quality score even with a real key, because the
  scoring rubric (fixed rubric vs. LLM-as-judge) is an explicitly open
  methodology decision per "What would close the quality-score gap" below,
  not something to assume unilaterally; it records the raw answer text
  alongside each query's `expected_urls`/`expected_names` so that decision
  can be made and applied without re-spending against a live key.
- **Real generation p50/p95 latency: also UNVERIFIED**, for the same reason
  (needs a real key to make a real generation call).
- **What this table CAN and DOES report, from real measurements against the
  real provider APIs (no mocks, no live keys needed):** each provider's
  behavior and latency when given a syntactically-valid but deliberately
  invalid credential — network reachability, correct `ProviderAuthError`
  classification (never a raw SDK exception), zero key-substring leakage,
  and the round-trip latency of that auth check. **This is explicitly NOT
  generation latency** — an auth failure is rejected before any model ever
  runs — and is reported in its own column so it is never confused with the
  (unmeasured) real thing. See "Auth-rejection latency" below.
- **Cost-per-1k-tokens is sourced from each provider's own current published
  pricing page**, read live on 2026-07-20 (citations inline), not measured
  by an API call — there is no such call. Per the shared contract, this is
  **informational-only guidance**: Anthropic and OpenAI are BYOK and Hugging
  Face inference is billed to the signed-in user's own account, so this
  project never pays for generation directly and cost is not a factor used
  to rank candidates below.

## Candidate models

| Provider | Model | Tier | Confirmed live against | Confirmed on |
| --- | --- | --- | --- | --- |
| Anthropic | `claude-haiku-4-5-20251001` | fast/cost-effective | [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) | 2026-07-20 |
| Anthropic | `claude-sonnet-5` | quality | same as above | 2026-07-20 |
| OpenAI | `gpt-5.6-luna` | efficient | [Models](https://developers.openai.com/api/docs/models), [Pricing](https://developers.openai.com/api/docs/pricing) | 2026-07-20 |
| OpenAI | `gpt-5.6-terra` | balanced | same as above | 2026-07-20 |
| OpenAI | `gpt-5.6-sol` | highest quality | same as above | 2026-07-20 |
| Hugging Face | `openai/gpt-oss-120b` | curated default | [Inference Providers](https://huggingface.co/docs/inference-providers/en/index) | 2026-07-20 (`govgis/providers/huggingface.py`'s own curation date) |
| Hugging Face | `deepseek-ai/DeepSeek-V3` | curated | same as above | 2026-07-20 |
| Hugging Face | `deepseek-ai/DeepSeek-R1` | curated | same as above | 2026-07-20 |

The Anthropic pair matches `govgis/providers/anthropic.py`'s existing
`DEFAULT_FAST_MODEL`/`DEFAULT_QUALITY_MODEL` constants — re-verified here,
not re-guessed: both `claude-instant-v1`/`claude-2.1` (the legacy app's long-retired
picks) are confirmed retired (2024-11-06 and 2025-07-21 respectively), and
`claude-haiku-4-5-20251001`/`claude-sonnet-5` are confirmed current with no
deprecation notice as of this table's build date. The OpenAI trio confirms
the plan's own directionally-correct 2026-07-20 guess (`gpt-5.6-luna` /
`gpt-5.6-terra` / `gpt-5.6-sol`) was accurate and needed no correction. The
Hugging Face set is `govgis/providers/huggingface.py`'s existing
`CURATED_MODELS` — reported here for a complete side-by-side view, not
independently re-curated by this table (curating that set was the HF
provider lane's own deliverable, already dated and cited in that module's
docstring).

## Comparison table

| Provider | Model | Retrieval-answer quality (Recall/groundedness over `docs/stage0/query_set.json`) | Real generation p50 / p95 latency | Auth-rejection latency (real network call, invalid key — NOT generation latency) | Cost per 1k tokens (input / output, informational only) |
| --- | --- | --- | --- | --- | --- |
| Anthropic | `claude-haiku-4-5-20251001` | **UNVERIFIED** — needs a funded `ANTHROPIC_API_KEY` | **UNVERIFIED** — same blocker | not separately measured (auth check is model-independent; see `claude-sonnet-5` row) | $0.001 / $0.005 |
| Anthropic | `claude-sonnet-5` | **UNVERIFIED** — needs a funded `ANTHROPIC_API_KEY` | **UNVERIFIED** — same blocker | p50 0.096s / p95 0.500s (n=7, real call to `api.anthropic.com`) | $0.002 / $0.010 (introductory through 2026-08-31; $0.003 / $0.015 standard after) |
| OpenAI | `gpt-5.6-luna` | **UNVERIFIED** — needs a funded `OPENAI_API_KEY` | **UNVERIFIED** — same blocker | not separately measured (see `gpt-5.6-terra` row) | $0.001 / $0.006 |
| OpenAI | `gpt-5.6-terra` | **UNVERIFIED** — needs a funded `OPENAI_API_KEY` | **UNVERIFIED** — same blocker | p50 0.486s / p95 1.679s (n=7, real call to `api.openai.com`) | $0.0025 / $0.015 |
| OpenAI | `gpt-5.6-sol` | **UNVERIFIED** — needs a funded `OPENAI_API_KEY` | **UNVERIFIED** — same blocker | not separately measured (see `gpt-5.6-terra` row) | $0.005 / $0.03 |
| Hugging Face | `openai/gpt-oss-120b` (default) | **UNVERIFIED** — needs a signed-in HF OAuth token | **UNVERIFIED** — same blocker | p50 0.082s / p95 0.540s (n=7, real call to `router.huggingface.co`) | varies — routed automatically to whichever backing Inference Provider serves the model (`provider="auto"`, see `govgis/providers/huggingface.py`); billed to the signed-in user's own HF account, not a single fixed provider-quoted rate the way Anthropic/OpenAI publish one |
| Hugging Face | `deepseek-ai/DeepSeek-V3` | **UNVERIFIED** — same blocker | **UNVERIFIED** — same blocker | not separately measured | varies (see above) |
| Hugging Face | `deepseek-ai/DeepSeek-R1` | **UNVERIFIED** — same blocker | **UNVERIFIED** — same blocker | not separately measured | varies (see above) |

All auth-rejection latencies above are from
[`evidence/provider_comparison_2026-07-20T23-22-53Z.json`](evidence/provider_comparison_2026-07-20T23-22-53Z.json)
(7 repetitions per provider; re-confirmed, same result shape, in
[`evidence/provider_comparison_2026-07-21T00-06-23Z.json`](evidence/provider_comparison_2026-07-21T00-06-23Z.json)
after the script was extended with the live-generation capability described
above -- both files' `live_generation_results` are `[]`, since no funded
credential was present either time). Every repetition, for every provider, raised
exactly `ProviderAuthError` (never an unexpected exception type, never a raw
SDK exception) and never leaked the fake key substring anywhere the script
checked (`str(exc)`) — see that evidence file's
`all_raised_provider_auth_error`/`no_key_substring_leaked` fields. This is
also this table's live-network confirmation of Stage 4's Gate: "HF calls are
attributed to the signed-in user, not the Space owner" (the HF row's call
used only a caller-supplied token, matching `govgis/providers/huggingface.py`'s
"no ambient credential fallback" design) and a real-world companion to
`tests/test_provider_secret_leak.py`'s mocked version of the same property.

## Recommended defaults (traceable to the above, not asserted in prose)

- **Anthropic**: keep `govgis/providers/anthropic.py`'s existing
  `DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"` /
  `DEFAULT_QUALITY_MODEL = "claude-sonnet-5"`. Confirmed current and
  unretired above; no change recommended.
- **OpenAI**: recommend `gpt-5.6-luna` as the fast/cost-effective default and
  `gpt-5.6-terra` as the quality default — the same two-tier shape as
  Anthropic's pair, and matching the plan's own "start with an efficient
  current model tier and offer a stronger quality tier" guidance.
  `gpt-5.6-sol` (highest quality, ~2x `gpt-5.6-terra`'s cost) is recorded
  above as a candidate but not recommended as either default tier.
  **Closed 2026-07-21**: `govgis/providers/openai.py` now defines
  `DEFAULT_FAST_MODEL = "gpt-5.6-luna"` / `DEFAULT_QUALITY_MODEL =
  "gpt-5.6-terra"`, traceable to this recommendation, mirroring
  `anthropic.py`'s own constants. Both Anthropic and OpenAI are now wired
  into `app.py` as BYOK answer-synthesis sections (a user-supplied
  `gr.Textbox(type="password")` API key per provider, session-scoped, never
  logged or persisted — see `app.py`'s module docstring and
  `tests/test_app_provider_kill_switch.py`'s BYOK coverage), the same
  retrieval-then-optional-generate composition `govgis.providers.huggingface`
  already has, each with its own per-session concurrency guard and
  sanitized-error substitution. This was not a Stage 4 Gate requirement
  (the Gate bullets do not name `app.py` wiring), but was closed anyway to
  resolve the "OpenAI/Anthropic BYOK path is not yet exercised end-to-end
  through the UI" gap noted alongside this table.
- **Hugging Face**: keep the existing curated set and its existing default
  (`openai/gpt-oss-120b`) — this table found no live-reachability or
  auth-handling problem with any of the three, and re-curating the set was
  not this lane's job.

## What would close the quality-score gap

Run `python docs/stage4/build_provider_comparison.py` with
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` environment variables set to real,
funded keys, and/or `HF_TOKEN` set to a real signed-in Hugging Face access
token. As of 2026-07-21 the script itself already does the rest: it loads
the real Stage 2 retrieval index, retrieves real results for every query in
`docs/stage0/query_set.json` that carries an `expected_urls` entry (17 of
the 23), makes a real `generate()` call per query for each provider with a
funded credential present, and records real latency plus raw answer text to
the evidence JSON (`live_generation_results`) — no code work remains to
produce the raw material this gap needs.

What is still explicitly undecided, and must be settled before this gap can
close: a groundedness/quality **scoring rubric** for the recorded raw
answers — score each answer against whether it correctly cites/reflects the
retrieved record(s) and matches `expected_urls`/`expected_names` for that
query, by a fixed rubric or an LLM-as-judge pass. This decision is left
open deliberately, not assumed here or by the script, since it is itself a
methodology choice best validated against real output shapes once they
exist. Neither running the script against a real, funded key nor deciding
the scoring rubric was done in this session: no such key was available (see
"Status summary" above, re-confirmed 2026-07-21), and spending against a
real, funded key -- or unilaterally deciding a scoring methodology on the
coordinator's behalf -- is out of this lane's authority either way.

**Decision recorded 2026-07-20 (Josh):** explicitly deferred, not blocking.
The chosen defaults (Anthropic `claude-haiku-4-5-20251001`/`claude-sonnet-5`,
OpenAI `gpt-5.6-luna`/`gpt-5.6-terra`, HF `openai/gpt-oss-120b`) ship as
currency/reachability-verified but quality-unbenchmarked, matching this
document's own honest disclosure above rather than a fabricated number.
Revisit with funded keys and a scoring-rubric decision post-launch, not as
a blocker to Stage 5/6.
