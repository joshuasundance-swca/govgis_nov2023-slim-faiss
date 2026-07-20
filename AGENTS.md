# govgis_nov2023-slim-faiss — agent guidance

## What this is

A public Hugging Face Space: semantic search over `govgis_nov2023` GIS metadata (1,684 government
ArcGIS servers, ~1M layers) using FAISS + BAAI/bge-large-en-v1.5 embeddings, with optional LLM
answer synthesis. **Currently deployed** (`app.py`, production baseline `5b3caca`): a single-file
Streamlit app on Python 3.11, using LangChain (`ChatAnthropic` with the long-retired
`claude-instant-v1`/`claude-2.1` model IDs) and `FAISS.deserialize_from_bytes` on a Hub-hosted
pickle-like artifact. **Do not extend `app.py`** — it is the legacy app being replaced, not a
starting point.

A modernization is planned but **not started**. Status, decisions, risks, and the 8-stage
implementation plan live in `docs/modernization-plan.md` — read it before doing any modernization
work. It is the durable planning authority; this file does not duplicate it.

## Commands (current legacy app only — no modernization tooling exists yet)

There is no `pyproject.toml`, no lockfile, no `.venv`, and no CI job that runs tests, lint, or type
checks. The only way to run the test suite today:

```
python -m unittest tests/test_space_build_contract.py
```

(or `python -m pytest tests/` if `pytest` happens to be installed locally — it is not a declared
dependency anywhere in this repo). `requirements.txt` is exact-version-pinned (`==`), runtime-only,
with no dev/test tooling in it (anthropic, faiss-cpu, huggingface-hub, langchain, langsmith, openai,
pydantic, PyYAML, sentence-transformers, torch). `.pre-commit-config.yaml` exists (ruff, black,
mypy, bandit, shellcheck, misc hygiene hooks)
but **is not invoked by any GitHub Actions workflow** — it only runs if a contributor runs
`pre-commit run --all-files` locally.

Modernization work (Stage 1 onward) will introduce `pyproject.toml` + a locked uv environment +
package layout under `govgis/` — see `docs/modernization-plan.md`'s "Target architecture" and
"Staged implementation and gates" sections for the target commands and quality gate. Do not invent
commands ahead of that work landing.

## CI/CD (as it exists today — verify before relying on it)

Three GitHub Actions workflows, none of which run tests/lint/type-checks:
- `.github/workflows/hf-space.yml` — on push to `main`, runs `git push` (no `--force`) to the HF
  Space remote. Does **not** wait for the Space to build, verify the deployed SHA, or exercise
  retrieval.
- `.github/workflows/check-file-size-limit.yml` — runs on PRs against `main`, enforces a 10MB file
  size limit (for HF Space LFS sync).
- `.github/workflows/bumpver.yml` — manual version bump + tag + push.

`main` currently has **no branch protection** — none of the above are configured as required status
checks, so nothing actually blocks a merge or a direct push today. `docs/modernization-plan.md`
Stage 1 now owns closing this (add branch protection with the new CI gate as a required check,
sequenced after that gate exists) — treat it as planned, not yet done, until Stage 1 actually lands.

## Non-negotiables (from `docs/modernization-plan.md` — do not weaken without recording a new decision there)

- Anthropic/OpenAI access is BYOK, session-scoped. Keys must never enter logs, caches, telemetry,
  URLs, persisted state, or exception messages.
- HF Inference Providers access uses the signed-in user's OAuth token (`inference-api` scope) —
  requests are billed to and act on behalf of that user, never the Space owner. This is a real
  behavioral distinction, not just phrasing: Anthropic/OpenAI stay BYOK regardless of HF sign-in.
- Never use owner-funded provider keys in the public Space without an explicit later decision adding
  authentication, quotas, spending limits, and abuse controls.
- Dataset-, user-, and model-controlled content is untrusted. Do not render it through an
  unsanitized HTML path (Gradio's `gr.HTML` performs **no** sanitization — confirmed against
  Gradio's own docs).
- Keep production (`5b3caca`) working and unchanged until a modernized branch passes every local,
  CI, staging, and live observable gate in the plan.

## Gotchas / pitfalls (verified against live systems, 2026-07-20)

- The Space's `README.md` YAML front matter is the **single source of truth** for the runtime
  Python/Streamlit version (`python_version: 3.11`, `sdk_version: 1.29.0`) — `requirements.txt`
  intentionally does not list `streamlit` (it comes from the SDK, not pip).
  `tests/test_space_build_contract.py` asserts this contract; keep it green if you touch either file.
- Hugging Face's current `spaces-config-reference` docs only list `sdk: gradio | docker | static` as
  valid values — Streamlit is absent from the current documented options, even though this Space's
  existing `sdk: streamlit` config is still `RUNNING` today (grandfathered, not actively broken).
- HF Space "persistent storage" (the old paid tier) no longer exists. Current persistence options are
  ephemeral local disk, Storage Buckets (read-write, explicitly mounted), or read-only repo volumes
  (datasets/models/other Spaces — **always** read-only when mounted). `preload_from_hub` runs at
  build time only and does **not** honor a custom `HF_HOME`.
- `git rev-parse main` / GitHub `main` / the live Space `sha` were all confirmed identical
  (`5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9`) as of 2026-07-20 — re-verify before assuming this is
  still the baseline; it will drift as work lands.

## Modernization plan status (2026-07-20)

Claude Code completed the plan's own "Review mandate for Claude Code": an 8-axis adversarial audit
(31 confirmed findings, 0 critical, 1 high) is applied directly to `docs/modernization-plan.md` on
branch `docs/modernization-plan`. Full evidence and methodology: `audit-recommendations/`. Nothing
has been implemented — this was a plan review, not a code change. Before starting Stage 0, read the
plan's "Claude Code review — 2026-07-20" section (near the top) and the "Implementation orchestration
strategy" subsection (under "Staged implementation and gates") for the verdict and execution plan.

## Implementation approach: orchestrate Stages 1–4, gate Stages 5–6 manually

Decided 2026-07-20 (full rationale in the plan's "Implementation orchestration strategy"
subsection): **Stage 0 runs inline** (small, partly bottlenecked on authenticated-log access, not
compute). **Stages 1–4 run as one continuous multi-agent orchestration effort** — parallel
disjoint-file build lanes within each stage, adversarial verification against that stage's actual
Gate bullets (not a self-report), bounded fix loops, and the coordinator re-running the stage's real
Gate commands before advancing. If this session is Claude Code with dynamic Workflow orchestration
available, that's the mechanism (see the `workflow-orchestration` skill — author the script fresh
per that skill's guidance, using the plan's lane breakdown as the spec, not a script to replay). In
any other harness, the same PRINCIPLE applies even without that specific tool: parallelize disjoint
files within a stage, verify against the stage's Gate before calling it done, and do not skip or
merge stage-gate checkpoints for speed.

Two hard boundaries, regardless of harness or orchestration mode: (1) orchestration accelerates the
*work* inside a stage, never the *stage-gate sequence* — Stage 3 cannot meaningfully start before
Stage 2's Gate passes, because it needs Stage 2's actual retrieval core and types; (2) **Stages 5
(staging) and 6 (production) are not orchestrated at all** — creating the real staging Space,
merging to `main`, and deploying to the public production Space are external, hard-to-reverse,
outward-facing actions that require the user's explicit go-ahead at the time, never a blanket
up-front authorization.

## For a cold session (Codex, Claude Code, or otherwise) picking this up

`docs/modernization-plan.md` plus this file plus `audit-recommendations/` (if present) are the
complete tracked context — there is no memory-only state this repo depends on.
`checkpoint.local.md`, if present, is ephemeral/git-ignored operational scratch, not authoritative;
treat its claims as hints and reconcile against `git status`/the plan/this file before acting on it.
