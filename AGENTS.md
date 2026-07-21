# govgis_nov2023-slim-faiss — agent guidance

## What this is

A public Hugging Face Space: semantic search over `govgis_nov2023` GIS metadata (865,304 records
across 1,684 government ArcGIS servers) using native FAISS + BAAI/bge-large-en-v1.5 embeddings, with
optional LLM answer synthesis. **As of 2026-07-21, the modernized app is live in production**
(`joshuasundance/govgis_nov2023-slim-faiss`, SHA re-verify via `HfApi().space_info(...)` before
trusting any SHA recorded here — `main` moves): a native Gradio 6.20.0 Blocks app on Python 3.14,
`govgis/` package doing direct Sentence Transformers + native FAISS retrieval (no LangChain, no
pickle), search that works with no login/key, and optional BYOK Anthropic/OpenAI + curated HF
Inference Providers (signed-in user's own `inference-api` OAuth token, never the Space owner's) answer
synthesis. Full history and evidence: `docs/modernization-plan.md`'s Stage 0-6 results.

The **legacy app is gone from `main`** (it lived at `app.py`/`requirements.txt` on Python 3.11 with
Streamlit + LangChain, deserializing a 4.28 GB Hub-hosted LangChain FAISS pickle-like artifact,
rendering descriptions via unsanitized raw HTML). Its known-good state is tagged
`pre-modernization-baseline` (local tag, not pushed — push it if a rollback is ever needed; see the
plan's Stage 6 rollback recipe).

## Commands

```
.venv\Scripts\python.exe -m pytest tests/ -q          # test suite
.venv\Scripts\python.exe -m ruff check .               # lint
.venv\Scripts\python.exe -m ruff format --check .       # format check
.venv\Scripts\python.exe -m mypy .                      # type check (strict)
.venv\Scripts\python.exe -m pre_commit run --all-files  # full local gate
uv sync --locked --all-groups                           # bootstrap/refresh the venv
```

`pyproject.toml` + `uv.lock` govern the dev/CI Python environment. `requirements.txt` is a
**generated artifact**, never hand-edited — see the Gotchas section below for the exact `uv export`
invocation, which is non-obvious and load-bearing (a plainer invocation silently produces a file the
real HF Space Dockerfile cannot install).

## CI/CD

Seven required GitHub Actions checks on every PR against `main` (branch protection enforces this,
`strict: true`, `enforce_admins: false` deliberately so `bumpver.yml`'s direct-push-to-`main` path
keeps working): `check-file-sizes`, `test`, `lint`, `typecheck`, `pre-commit`, `gr-html-safety`,
`requirements-drift`. `requirements-drift` only regenerates-and-diffs `requirements.txt` against
`uv.lock` — it does **not** `pip install` it, so it cannot catch a real Space-build incompatibility
(see the Gotchas entry below; this is a known, accepted gap, not an oversight).

`.github/workflows/hf-space.yml` pushes `main` to the HF Space git remote on every push, then
**polls the Space API until it reaches `RUNNING` at the pushed SHA** (added 2026-07-21; previously it
reported success the instant `git push` completed regardless of whether the Space actually built) —
fails the job on `BUILD_ERROR`/`RUNTIME_ERROR`/`CONFIG_ERROR` or a 20-minute timeout instead of
silently lying green.

## Non-negotiables (do not weaken without recording a new decision in `docs/modernization-plan.md`)

- Anthropic/OpenAI access is BYOK, session-scoped. Keys must never enter logs, caches, telemetry,
  URLs, persisted state, or exception messages.
- HF Inference Providers access uses the signed-in user's OAuth token (`inference-api` scope) —
  requests are billed to and act on behalf of that user, never the Space owner. Anthropic/OpenAI stay
  BYOK regardless of HF sign-in.
- Never use owner-funded provider keys in the public Space without an explicit later decision adding
  authentication, quotas, spending limits, and abuse controls.
- Dataset-, user-, and model-controlled content is untrusted. Never bind it through an unsanitized
  HTML path (`gr.HTML` performs no sanitization) — CI's `gr-html-safety` job enforces this.
  `SafePresentationRecord` is the only type any rendering code may consume; only
  `govgis/presentation.py` may construct one.

## Gotchas / pitfalls

- **`requirements.txt` must be directly `pip install`-able exactly as HF's Gradio Space Dockerfile
  invokes it** (confirmed live 2026-07-21, Stage 5's first staging `BUILD_ERROR` — full account in
  `docs/modernization-plan.md`'s "Stage 5 results"): the Dockerfile runs
  `pip install -r requirements.txt gradio[oauth,mcp]==<sdk_version> uvicorn>=0.14.0 websockets>=10.4
  spaces` verbatim, with no other flags. Regenerate with exactly:
  `uv export --locked --no-dev --no-hashes --no-emit-project --emit-index-url --format
  requirements.txt -o requirements.txt` — not the plainer form — because: `-e .` (a bare `uv
  export`'s self-install line) can't carry a hash and pip refuses it once any requirement has one
  (`--no-emit-project`; the `govgis` package doesn't need installing — it's already on `sys.path`
  next to `app.py` when the Space runs); pip's hash-checking mode is invocation-global, so it also
  breaks on HF's own unhashed appended packages (`--no-hashes`); and plain pip has no knowledge of
  the `pytorch-cpu` index redirect in `[tool.uv.sources]`, so `torch==...+cpu` is otherwise
  unresolvable (`--emit-index-url`). Before trusting any `requirements.txt` change, simulate the real
  Dockerfile command: `uv pip install --dry-run --python-version 3.14 --python-platform linux
  --index-strategy unsafe-best-match -r requirements.txt "gradio[oauth,mcp]==<sdk_version>"
  "uvicorn>=0.14.0" "websockets>=10.4" spaces` (the `--index-strategy` flag matters: `uv pip`'s
  stricter dependency-confusion-safe default rejects a resolution real `pip` accepts).
- **`pyproject.toml`'s `pydantic` dependency is deliberately upper-bounded
  (`>=2.11.10,<=2.12.5`)** to stay inside gradio 6.20.0's own `mcp`-extra ceiling (HF's Dockerfile
  appends `gradio[oauth,mcp]==<sdk_version>` unconditionally, regardless of which gradio extras this
  project itself declares — confirmed against gradio's own PyPI `requires_dist` metadata). **Do not
  merge Dependabot PR #49 ("Bump the app group...") as-is** — opened 2026-07-21, it reverts this
  ceiling back to a bare `pydantic==2.13.4`, which would silently reintroduce the exact `BUILD_ERROR`
  this fix resolved on the next Space deploy. CI's `requirements-drift`/`check-file-sizes` checks
  were both failing on that PR as of this writing, which structurally blocks a normal merge, but that
  is not a substitute for knowing why — re-check gradio's `mcp`-extra `requires_dist` range on any
  future gradio version bump (this project's own version, or Dependabot's) before accepting a
  pydantic bump.
- **Also do not merge Dependabot PR #45** ("Bump the app group... 10 updates", open since
  2024-06-24) — it targets the pre-modernization `langchain`/`streamlit`-era dependency set and would
  revert the migration.
- The Space's `README.md` YAML front matter is the **single source of truth** for the runtime
  Python/SDK version (`python_version: "3.14"`, `sdk: gradio`, `sdk_version: 6.20.0`,
  `hf_oauth: true`, `hf_oauth_scopes: [inference-api]`) — `tests/test_space_build_contract.py`
  asserts this contract; keep it green if you touch either file or `requirements.txt`.
- HF Space "persistent storage" (the old paid tier) no longer exists. Current persistence options are
  ephemeral local disk, Storage Buckets (read-write, explicitly mounted), or read-only repo volumes.
  `preload_from_hub` runs at build time only, does not honor a custom `HF_HOME`, and was **never
  tested** in this project — the retrieval artifact is instead fetched via `hf_hub_download` at
  request time from a pinned dataset repo (see `app.py`'s `_hub_artifact_paths()`), which is simpler
  and was sufficient; revisit `preload_from_hub` only if cold-start latency becomes a real problem.
- The Space's build/run log endpoints (`GET /api/spaces/<id>/logs/{build,run}`) require
  authentication (the Space owner's own `huggingface_hub` token) — an unauthenticated check gets
  HTTP 401. The run-log endpoint only retains logs for the currently running instance.
- `client.predict(..., api_name=...)` via `gradio_client.Client(space_id)` is the fastest way to
  smoke-test a live Space's actual behavior end-to-end (retrieval, provider error paths) without a
  browser — call `.view_api()` first to discover the exact endpoint names (they're derived from
  handler function names, e.g. `/_handle_search`, not necessarily what you'd guess). First query
  after any cold start pays ~50-65s to load the ~3.5GB FAISS index + embedding model; subsequent
  queries in the same container are ~1-2s. This is cache-once-per-container behavior by design
  (`app.py`'s `_get_retrieval_index()`), not a bug.

## Modernization: COMPLETE (2026-07-21)

`docs/modernization-plan.md`'s Stages 0-6 are all gate-passed with real evidence recorded in each
stage's own "Stage N results" subsection — read those, not just the plan's original intent, for what
actually happened (several real bugs were found and fixed during Stage 5 staging validation that no
local/CI gate could have caught; see the `requirements.txt` gotcha above). Production is live and
independently re-verified (not just trusted from CI's self-report) as of this writing. Explicitly
deferred, not forgotten: Stage 4's provider quality/latency benchmarking (the user's decision — "get
it up and running... then look at other govgis ideas"), Stage 7 (footprint reduction via smaller
embedding models, blocked on an absolute Recall floor that doesn't exist yet), Peak RSS and provider
latency (Stage 0 itself never measured a comparable baseline for either, so both stay
"not-yet-gatable" rather than silently passed).

**Orchestration pattern that worked well** (for future reference, including on the rebuild-plan
below): Stage 0 ran inline; Stages 1-4 ran as one continuous multi-agent Workflow orchestration
(parallel disjoint-file build lanes per stage, adversarial verification against that stage's actual
Gate bullets, bounded fix loops, coordinator re-running the real Gate commands before advancing);
Stages 5-6 (real staging Space, production merge+deploy) were deliberately NOT orchestrated and each
required the user's explicit go-ahead at the time it happened, never a blanket up-front
authorization — this caught real, external-system-only bugs that no amount of agent orchestration
inside a sandbox would have found.

## Separate initiative: govgis data-pipeline rebuild (planned, not started)

`docs/rebuild-plan.md` (1,847 lines) + its adversarial audit in `docs/rebuild-plan-audit/` (16
files, `00-executive-summary.md` first) propose a **from-scratch rebuild of the data pipeline that
produces the govgis dataset** — a stateless, matrix-sharded batch pipeline on free GitHub Actions
that turns completely off between runs, replacing the legacy pipeline's un-committed notebooks,
unstable `hash()`-based IDs, silent row drops, and unmanifested 4.28 GB pickle blob. This is a
**separate initiative from the Gradio-migration modernization above** — it targets the data that
feeds the Space, not the Space's UI, and would live in a **new, separate repo** (`govgis-pipeline`),
not this one. The plan was produced by 4 independent draft lenses → 3 judges → a 7-axis adversarial
audit + completeness critic (refute-as-default, every surviving finding applied as a diff): 38 axis
findings (30 confirmed, 7 adjusted, 1 refuted) + 6 completeness gaps, zero critical, seven high, all
applied. Verdict: **direction sound, not overturned.**

**Not started. The load-bearing blocker before any public crawl**: the plan's "Open Decision #1" —
Joseph Elfelt (the sole source for the ~7,500-server seed list) has been non-responsive since
2025-12-15, and the surviving `.txt` seed is the text of a PDF whose scraping his terms prohibit; a
human must pick a no-reply default (proceed under a free/non-commercial posture after a defined
window, or block) before Stage 0 completes. Seven other open decisions exist (facet precision, crawl
runner, budget tier, redistribution scope, embedding dimension, seed diversification, snapshot
cadence) but don't block *starting* Stages 0-2 (code-only, nothing published). Dominant cost/risk is
~2-4 focused solo months of build effort before the first snapshot, not recurring dollars (~$0
orchestration, low-tens-of-dollars per GPU re-embed). Full detail: `docs/rebuild-plan.md` section on
Open Decisions, and `docs/rebuild-plan-audit/00-executive-summary.md` for the audit verdict.

Do not start implementation on this without the user's explicit go-ahead — it's a real multi-month
commitment with a live legal-exposure question (SWCA is a commercial consultancy;
`commercial_use_authorized` defaults `false`), not a quick follow-up task.

## For a cold session (Codex, Claude Code, or otherwise) picking this up

`docs/modernization-plan.md`, `docs/rebuild-plan.md` (+ its audit), and this file are the complete
tracked context for this repo's two initiatives — there is no memory-only state either depends on.
`checkpoint.local.md`, if present, is ephemeral/git-ignored operational scratch, not authoritative;
treat its claims as hints and reconcile against `git status`/live system state/this file before
acting on it. Two open Dependabot PRs (#45, #49) exist and must **not** be merged as-is — see
Gotchas above.
