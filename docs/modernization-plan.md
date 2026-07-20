# GIS semantic-search Space modernization plan

Status: approved for planning; implementation has not begun
Recorded: 2026-07-20
Production baseline: `5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9`

## Purpose

Modernize the public Hugging Face Space without disrupting the known-good
deployment. The target is a secure, reproducible, provider-neutral GIS
semantic-search application with:

- a native Gradio interface;
- search that works without an LLM;
- current Anthropic and OpenAI support through user-provided API keys;
- a curated Hugging Face Inference Providers option authorized as the signed-in
  Hugging Face user;
- safe, versioned retrieval artifacts;
- measured startup improvements;
- observable tests, CI, deployment validation, and rollback.

This document is the durable planning authority. Written claims remain
hypotheses until their corresponding validation gate passes.

## Claude Code review — 2026-07-20

Per this document's own "Review mandate for Claude Code" (below), Claude Code
adversarially reviewed this plan the same day it was recorded: live-reconciled
every baseline claim and artifact count against Git, GitHub, the Hugging Face
Hub/Space APIs, and current official documentation; ran an 8-axis, 56-agent
adversarial audit (auditor → verifier per finding, plus a completeness
critic) targeting each of the mandate's 8 points; and applied the surviving
findings directly to this document as the diff you are reading, rather than
only listing them separately.

**Verdict: the plan was sound. Zero critical findings; one high finding (BYOK
secret-handling had no verification mechanism, now gated in Stage 4); 29
other findings, all wording/gate/citation precision fixes, no direction
changes.** The single most load-bearing gap, found independently four
different ways, was that `main` has no branch protection today — Stage 6's
"merge only after required checks pass" had nothing to attach to. Stage 1 now
owns closing that, sequenced after its own CI gate lands.

Full findings, evidence, and methodology: [`audit-recommendations/`](../audit-recommendations/)
(`00-executive-summary.md` for the verdict and priorities; `findings.json` for
the machine-readable index; one file per axis for full evidence).
Newly opened decisions from this review (all applied above, not left as
prose-only intent): configure branch protection with required CI checks after
Stage 1 (and decide `bumpver.yml`'s exemption); record Stage 0's
retrieval-quality threshold as an artifact Stage 2/7 gate on, not an adjective;
capture the actual rollback/SHA-verification commands instead of describing
them in prose; add a BYOK secret-leak test and concrete timeout/retry/
concurrency numbers to Stage 4; preserve MIT license/attribution in the
README rewrite; and record a lightweight review/enforcement note for this
single-maintainer repo.

**Addendum, same day:** the user and Claude Code agreed implementation should
use multi-agent Workflow orchestration to accelerate Stages 1–4, while keeping
Stage 0 inline and Stages 5–7 gated/manual. See "Implementation orchestration
strategy" under "Staged implementation and gates" below.

## Confirmed decisions

1. Migrate the UI from the built-in Streamlit Space SDK — no longer offered
   or documented as a configurable `sdk` value for new or reconfigured Spaces
   (the current Hugging Face Spaces configuration reference enumerates only
   `gradio`, `docker`, or `static`; this Space's existing `sdk: streamlit`
   config keeps running only because it predates that change, not because
   Streamlit is actively supported going forward) — to the native Gradio
   Space SDK. No dated, formal Hugging Face deprecation announcement for
   Streamlit Spaces was located; "deprecated" here means "absent from the
   current documented options," not a scheduled retirement.
2. Use BYOK for Anthropic and OpenAI. Keys must remain session-scoped and must
   never enter logs, caches, telemetry, URLs, persisted state, or exception
   messages.
3. Include Hugging Face Inference Providers as a curated "Open models via
   Hugging Face" option, not as an unrestricted provider/model browser.
4. Use Hugging Face OAuth with the least-privilege `inference-api` scope so an
   authenticated user's token can make Inference Provider requests on that
   user's behalf. This does not grant Anthropic or OpenAI access.
5. A temporary staging Space is authorized when platform-level validation is
   needed. Create it only when required, keep it separate from production, and
   remove it after the production deployment and rollback window are verified.
   Record its exact ID before creation and verify that exact target before
   cleanup.
6. Modernize the Python project and dependency workflow according to current
   Hugging Face and repository standards. The recommended target is Python 3.14
   with a `pyproject.toml` and a locked uv environment, subject to a clean Linux
   install and Hugging Face staging build. Fall back to Python 3.13 only if a
   real compatibility failure is reproduced.
7. Keep production working until a modernized branch passes all local, CI,
   staging, and live observable gates.
8. Do not merge Dependabot PR #45 as-is. Its dependency set and Streamlit
   assumptions are already stale and do not represent the intended architecture.

## Reconciled baseline

Verified on 2026-07-20:

- Local `main`, fetched `origin/main`, GitHub `main`, and the public Space SHA
  were all
  `5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9`.
- GitHub PR #46 was merged and deployment workflow run `29759385794`
  completed successfully.
- The public Space reported `RUNNING` on `cpu-basic` and returned HTTP 200.
- The Space API's JSON has no top-level `storage` key at all (it exposes
  `usedStorage: 0` instead) and no `volumes` field, consistent with current
  Hugging Face docs that legacy persistent Space storage is no longer
  offered. (The response shape does not literally contain `storage: null`;
  the substantive conclusion — no persistent storage configured — is
  unchanged.)
- Public build and run log endpoints returned HTTP 401. Build duration,
  artifact-load duration, peak RSS, and cold-start time are therefore
  **UNVERIFIED**, not inferred from the healthy endpoint.
- `main` has **no branch protection today**
  (`branches/main/protection` returns HTTP 404 "Branch not protected").
  `.github/workflows/check-file-size-limit.yml` runs on pull requests but is
  not a required status check, so nothing currently blocks a merge or a
  direct push to `main`. Stage 1 records a follow-up action to close this
  once a CI gate exists to make required (see Stage 1).
- The local worktree was clean before this documentation branch was created.

Known-good rollback points:

- GitHub and Space revision:
  `5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9`
- Recovery implementation commit:
  `3e85082` (tree-identical to `5b3caca`; the PR #46 merge added no further
  changes beyond this commit's own content — both SHAs restore the exact
  same working tree, so either is an equally valid rollback target).
- Existing runtime contract: Python 3.11 and Streamlit 1.29.0, with the
  Streamlit version controlled only by Space README metadata. Confirmed
  exact against README.md's YAML front matter and already backed by an
  executable gate (`tests/test_space_build_contract.py`).
- This baseline snapshot is dated 2026-07-20 and `main` is not frozen while
  the modernization branch is in progress; re-run the reconciliation
  commands in this section at Stage 0 kickoff rather than trusting this
  snapshot to still be current.

## Artifact inventory

Counts below were calculated by script from the live Hub repository trees; they
are not model arithmetic.

| Artifact | Bytes | GiB |
| --- | ---: | ---: |
| Serialized FAISS artifact | 4,280,523,962 | 3.987 |
| Embedding files needed by the current model | 1,341,561,506 | 1.249 |
| Current startup artifact working set | 5,622,085,468 | 5.236 |
| Entire dataset repository | 10,153,589,456 | 9.456 |
| Entire embedding-model repository | 4,019,210,262 | 3.743 |

Pinned source revisions:

- Dataset `joshuasundance/govgis_nov2023-slim-spatial`:
  `ab1220e6823732093a1c8a0122af98f7da1f4217`
- Model `BAAI/bge-large-en-v1.5`:
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`

The model repository contains duplicate PyTorch, Safetensors, and ONNX weight
formats. Do not download or preload the entire model repository. The current
Sentence Transformers load requires 10 files totaling 1,341,561,506 bytes,
dominated by `model.safetensors` at 1,340,616,616 bytes.

## Current risks

### Critical trust boundary: serialized FAISS data

The runtime currently deserializes a Hub-hosted LangChain FAISS byte artifact.
That representation includes Python-serialized docstore state. A remote
multi-gigabyte pickle-like artifact must not remain part of the public runtime
trust path.

Target artifact layout:

- native `index.faiss`;
- typed `documents.parquet` or JSONL keyed by integer vector ID;
- a manifest with schema version, record count, embedding model and revision,
  vector dimensions, distance metric, source revision, creation command, and
  SHA-256 checksums.

The existing artifact may be converted once only after verifying its pinned
revision and checksum. Run conversion in an isolated, no-network process, then
validate vector/document counts and retrieval parity. The deployed application
must never opt into dangerous deserialization. Stage 2's gate must include a
check that the conversion process actually ran with no outbound network
access (e.g. run it under a sandboxed/network-disabled process and assert
that; do not simply document the intent) — otherwise nothing verifies "run in
an isolated, no-network process" beyond prose.

**Rollback caveat**: rolling back to the production baseline (`5b3caca`)
after Stage 2 has shipped re-exposes the legacy `FAISS.deserialize_from_bytes`
pickle-like path and the raw-HTML XSS path below — a rollback is a return to
a *previously accepted* risk level, not a risk-free action. Note this
explicitly at the point of any rollback decision made after Stage 2/3 ship
(see Stage 6).

### Unsafe rendering

The current app passes dataset-controlled descriptions directly to a raw HTML
component (`app.py:225,227`, `st.components.v1.html()` — unsanitized).
This is an XSS path. Names, URLs, fields, model output, and dataset
descriptions must all be treated as untrusted.

Target policy:

- parse records into strict Pydantic v2 models;
- accept external links only with `http` or `https` schemes, and re-apply
  that allowlist to any URL appearing in LLM-*generated* answer text, not
  only to URLs in retrieved records — a successful prompt injection could
  cause the model to emit a link that never passed through record
  validation;
- convert source HTML descriptions to plain text for the first release;
- escape generated Markdown and retain Gradio HTML sanitization
  (`gr.Markdown`'s `sanitize_html=True` default; confirmed via Gradio's own
  docs that disabling it is explicitly not recommended);
- never use `gr.HTML` for dataset, user, or model-controlled content — Gradio's
  own docs confirm `gr.HTML` performs no sanitization at all. This must be a
  durable, CI-enforced rule (e.g. a grep/lint check or a code-review checklist
  item that fails the gate if `gr.HTML` appears bound to any
  dataset/user/model-sourced value), not only the one-time Stage 3 browser
  test — a later change could silently reintroduce it with no regression
  signal.
- verify the painted browser result with adversarial fixtures.

### Prompt injection

Retrieved ArcGIS metadata is untrusted content supplied to an LLM. Delimit it as
data, state that instructions inside it are not authoritative, request claims
grounded in returned records, and test malicious record content. Prompt text
alone is not a security boundary; output rendering and link validation still
apply — including to LLM-generated links, per the Unsafe rendering policy
above.

### Secrets, cost, and public abuse

- Anthropic and OpenAI keys are BYOK and session-scoped. This policy needs a
  verification mechanism, not only a statement: Stage 4's gate must include a
  test that forces a provider-call failure path (e.g. an invalid key) and
  asserts no substring of the test key appears in captured logs, exceptions,
  or telemetry. Without this, a future logging change could leak a key with
  no gate catching it.
- Hugging Face inference uses the signed-in user's OAuth token with only the
  `inference-api` scope.
- Never use owner-funded production provider keys in the public Space unless a
  later explicit decision adds authentication, quotas, spending limits, and
  abuse controls.
- Provider errors must be typed and sanitized.
- Add bounded timeouts, retries only for transient failures, concurrency
  limits, and user-visible rate/cost guidance, with concrete starting numbers
  recorded at Stage 4 (e.g. a provider-call timeout in the 15-30s range,
  retries limited to 1-2 attempts on transient/5xx errors only, and a
  per-session concurrency limit of 1 in-flight generation call) and a
  corresponding Stage 4 gate asserting the timeout/retry behavior under a
  simulated slow/failing provider. Treat the specific numbers as a starting
  point to be revised from Stage 4/5 measurements, not as final.

### Reliability and observability gaps

The current deployment workflow (`hf-space.yml`) reports success when Git push
succeeds; it does not wait for the Space build, verify the deployed SHA, or
exercise retrieval. No GitHub Actions workflow currently runs any test, lint,
or type-check tier at all. `main` has no branch protection, so even the
existing file-size-limit workflow is not a *required* check — nothing today
actually blocks a merge or a direct push. These proxies are not sufficient
release gates. Separately, `bumpver.yml` pushes version-bump commits directly
to `main` on manual dispatch; any branch-protection scheme added later must
explicitly account for this path (exempt it, or route version bumps through
the same gate).

## Target architecture

Keep the application small and explicit:

```text
Gradio UI
  -> validated search request
  -> embedding + FAISS retrieval
  -> typed GIS search results
  -> optional provider-neutral query rewrite or answer synthesis
  -> safe presentation model
  -> sanitized rendered output
```

Recommended modules:

- `app.py`: Gradio composition and event wiring only. This includes owning the
  "optional provider-neutral query rewrite or answer synthesis" pipeline step
  in the flow diagram above: `app.py` composes `retrieval.py` output with an
  optional `providers/*.py` call, in that order, and passes the result to
  `presentation.py`. No separate orchestration module is needed at this size;
  naming the owner here closes the gap between the flow diagram and this list.
- `govgis/config.py`: validated environment and artifact configuration.
- `govgis/models.py`: Pydantic request, GIS record, result, and provider
  models, **and the "safe presentation model" type** referenced in the flow
  diagram — the typed, validated shape that `presentation.py` produces and
  that `app.py` is the only thing permitted to render.
- `govgis/artifacts.py`: manifest/checksum validation and artifact loading.
- `govgis/retrieval.py`: query embedding and direct FAISS search.
- `govgis/providers/base.py`: narrow provider protocol and common errors.
- `govgis/providers/anthropic.py`: official Anthropic Messages client.
- `govgis/providers/openai.py`: official OpenAI Responses client.
- `govgis/providers/huggingface.py`: `InferenceClient` using HF OAuth.
- `govgis/presentation.py`: converts raw GIS records and LLM output into the
  safe presentation model defined in `models.py` — plain-text conversion,
  link validation, and escaped Markdown. This is the only module permitted to
  construct that type; `app.py` may render it but must not construct or
  bypass it. This is the seam where the current app's XSS bug
  (`app.py:225,227`, raw HTML via `st.components.v1.html()`) lives — naming
  its owner here is a deliberate guard against silently reintroducing it.

Remove LangChain and LangSmith unless a measured requirement appears that the
small direct architecture cannot meet. The current chain adds dependency churn,
conceals serialization risk, and is not needed for one retrieval step plus one
optional generation step. Removing LangSmith does drop its tracing/observability
capability; the "Reliability and observability gaps" fixes above (CI gate,
branch protection, deployment-SHA verification) are the intended replacement
for the release-process half of that gap, not a tracing replacement — if
per-request tracing is later found necessary, that is a new, separately
justified decision, not an assumed default.

`pyproject.toml`/uv govern the Python *dependency* environment; they do not
control the Space's runtime SDK contract. The Space's `python_version` and
`sdk_version` are set by the `README.md` YAML front matter (see
`tests/test_space_build_contract.py`, which already asserts this), independent
of whatever Python version uv resolves locally or in CI. Stage 1 must keep
both in sync explicitly — updating `pyproject.toml`'s Python requirement does
not update the Space's front matter, and vice versa.

## Product behavior

### Retrieval-first experience

- Search must work without signing in or entering an API key.
- The user's original query is the default retrieval query.
- Apply the embedding model's documented query instruction consistently.
- Query rewriting is optional and off by default until an evaluation shows a
  material retrieval gain worth an additional provider call.
- Show retrieved GIS records even if answer synthesis fails.
- Preserve record name, type, description, parent-service description, fields,
  and validated source URL.

### Optional answer synthesis

Provider choices:

- Anthropic: start with a current fast/cost-effective model and offer a current
  Sonnet-tier quality option. Do not retain `claude-instant-v1` or
  `claude-2.1`; both are retired (confirmed 2026-07-20 against Anthropic's
  model-deprecations page: `claude-instant-*` retired 2024-11-06,
  `claude-2.1` retired 2025-07-21). As of 2026-07-20 the directionally correct
  current picks are `claude-haiku-4-5-20251001` (fast/cost-effective) and
  `claude-sonnet-5` (quality) — do not treat these IDs as final; re-verify
  against the model-deprecations page at implementation time per this
  document's Review mandate point 5.
- OpenAI: use the Responses API with `store=False`; start with an efficient
  current model tier and offer a stronger quality tier. As of 2026-07-20 the
  current flagship family is GPT-5.6 (`gpt-5.6-luna` efficient /
  `gpt-5.6-terra` balanced / `gpt-5.6-sol` highest quality) — re-verify at
  implementation time; this family did not exist as of this document's
  authors' training data, which is exactly why this section defers to a
  live check rather than pinning a name now.
- Hugging Face: curate a small set of available open chat models and use
  Inference Provider routing. Do not expose arbitrary provider-specific
  parameters in the first release.

Shared controls should be limited to portable concepts such as provider, model,
temperature where supported, and maximum output tokens. Provider-specific
capabilities should not leak into the common interface without a tested need.

Timeouts, retries, and concurrency limits (see "Secrets, cost, and public
abuse" above) must ship with concrete numbers and a corresponding Stage 4
gate, not only the policy statement above — see Stage 4's gate list.

### Attribution and licensing

The current README credits Joseph Elfelt and the creators of the `restgdf`
library, and links to the MIT license (the link target is currently broken —
`README.md` points to `LICENSE.md`, but the file is `LICENSE`; fix this as
part of the rewrite). The Gradio migration requires rewriting the Space's
README YAML front matter and body; when doing so:

- keep `license: mit` in the front matter and fix the license link to point at
  the actual `LICENSE` file;
- retain the Joseph Elfelt / `restgdf` acknowledgment and the
  `govgis_nov2023` dataset attribution as provenance (restgdf is not a current
  runtime dependency — this is historical credit, not a live attribution
  requirement);
- drop the stale "written by GPT-4" and "Claude-Instant / Claude-2.1" copy,
  which no longer describes the target architecture.

MIT compliance itself rests on the untouched `LICENSE` file, which this
migration does not remove — the above is documentation hygiene and
provenance preservation, not a licensing-compliance blocker.

### Review, enforcement, and sign-off

This is a single-maintainer, pre-implementation repo with no formal review or
approval process today (`approv|reviewer|maintainer|sign-off` do not appear
anywhere in this document outside staging-Space pre-approval). For a
single-maintainer repo, "review authority" is largely ceremony — approval is
inherently self-approval — but the concrete, non-ceremonial substance is the
same enforcement gap already recorded in "Reliability and observability gaps"
and Stage 1: without branch protection and required status checks on `main`,
"gate passed" is a self-assertion with nothing checking it. Once Stage 1's CI
gate exists and Stage 1's branch-protection follow-up action lands (see
Stage 1), each stage's own Gate bullets serve as the sign-off checklist —
"approved" becomes a checkable record (the CI run against the PR) rather than
an assertion in this document.

## Startup and storage strategy

Current Hugging Face behavior:

- Space-local disk is ephemeral.
- Legacy persistent Space storage is no longer available.
- Models, datasets, Spaces, and storage buckets can be mounted as managed
  volumes.
- Repository mounts are read-only and fetch bytes lazily.
- Buckets are mutable, non-versioned storage and are inappropriate as the
  primary authority for immutable release artifacts.
- `preload_from_hub` moves selected public Hub files into the default Hub cache
  during the image build. It does not honor a custom `HF_HOME`.

Experiment in this order on staging:

1. Establish current cold-start and load metrics from authenticated logs or
   explicit structured startup telemetry.
2. Load the safe split artifact from pinned public repositories using the
   existing Hub cache path.
3. Measure a read-only dataset/model volume configuration. Lazy mounting avoids
   an eager full-repository snapshot, but the application will still read all
   bytes needed by the index and embedding weights.
4. Measure `preload_from_hub` for exact files and pinned revisions. It may trade
   longer builds for faster restarts.
5. Verify whether `preload_from_hub` correctly resolves the current dataset repo.
   Official wording mentions datasets, but the documented syntax has no
   `repo_type`; treat this as an experiment, not a fact.
6. Prefer a dedicated versioned Hub repository for the safe search-index release
   if that makes preloading and lifecycle management clearer.
7. Do not create a storage bucket unless a future feature needs mutable runtime
   data.

Select the winner using cold-start time, rebuild time, failure rate, disk use,
and operational complexity. Do not optimize only one restart.

## Staged implementation and gates

### Implementation orchestration strategy (added 2026-07-20)

Decided with the user after the Claude Code review above: use multi-agent
Workflow orchestration ("ultracode") to accelerate the *buildable* stages,
but do not let orchestration compress the stage-gate sequence itself or
touch anything external/irreversible without an explicit human go/no-go.
Orchestration speeds up the work inside a stage (parallel disjoint-file
build lanes, fan-out adversarial verification against that stage's actual
Gate bullets, bounded fix loops); it does not replace a stage's Gate as the
advance/no-advance decision, and it does not shrink the real dependency
chain between stages — Stage 3 still cannot start meaningfully before
Stage 2's Gate passes, because Stage 3 needs Stage 2's actual retrieval
core and types to exist.

**Grouping:**

- **Stage 0** — run inline/coordinator-led, not as a fan-out. It's small
  (a query set, a few API calls, a few recorded values) and partly bottlenecked
  on authenticated-log access, which more agents can't accelerate.
- **Stages 1–4 — one continuous orchestrated effort.** These are genuinely
  buildable, verifiable, disjoint-file work. Suggested lane shape (a future
  session should author the actual Workflow script fresh, per the
  `workflow-orchestration` skill — this is the spec, not the script):
  - *Stage 1*: 2 lanes — packaging (`pyproject.toml`, lockfile,
    `.python-version`, package layout) and tooling (pre-commit, Ruff, mypy,
    CI workflow creation). Verify against Stage 1's Gate. **Then**, as a
    separate, deliberate, non-parallel step: configure GitHub branch
    protection against the now-existing CI check — this mutates shared
    repo state and should get an explicit go-ahead before applying, same
    as any other GitHub-visible change.
  - *Stage 2*: parallel lanes for `govgis/models.py` (GIS record/result/safe
    presentation types), `govgis/artifacts.py` (manifest/checksum
    validation), `govgis/retrieval.py` (embedding + FAISS search), and the
    failing-tests-first lane — plus **one single, coordinator-run, non-parallel
    conversion of the actual pinned legacy artifact** (isolated, no-network,
    once). Do not let multiple agents race on the real conversion step.
    Verify against Stage 2's Gate, including the Stage 0-recorded threshold
    and the no-network proof.
  - *Stage 3*: lanes for `govgis/presentation.py` (link validation, escaping,
    the durable "no `gr.HTML` on untrusted content" lint check) and `app.py`'s
    Gradio Blocks composition. Verify with real browser-level tests.
  - *Stage 4*: `govgis/providers/base.py` first (small, fast, others depend on
    it), then true parallel lanes for `anthropic.py`, `openai.py`,
    `huggingface.py` (fully disjoint files), plus the provider/model
    comparison-table action, the BYOK secret-leak test, and the
    timeout/retry/concurrency test. Verify against Stage 4's Gate.
  - Verifiers get an explicit top-tier model (never left on inherit); mechanical
    scaffolding lanes can tier down. Every stage's fan-out is followed by the
    coordinator re-running that stage's actual Gate commands before declaring
    it done — a green build report is not the gate.
- **Stage 5 (staging)** — narrow and mostly sequential: creating the actual
  staging Space is a real external resource with a real (if temporary) cost
  and footprint; get an explicit go-ahead before creating it, even though its
  ID was already decided in Stage 0. Waiting for real cold builds/restarts is
  calendar-bound, not something a larger fan-out speeds up.
- **Stage 6 (production rollout)** — no autonomous orchestration. This merges
  to `main` and deploys to a public Space with real users; per standing policy,
  it requires the user's explicit go-ahead at the time, not a blanket
  authorization granted now for "as much work as possible."
- **Stage 7 (footprint reduction)** — deferred; only after Stage 6 ships with
  parity confirmed. Can reuse the same orchestration shape as Stages 1–4 when
  picked up.

### Stage 0: baseline and test oracle

Actions:

- create 15–25 representative GIS queries with expected relevant URLs;
- include difficult, empty, malformed, and adversarial records;
- capture cold-start, artifact-load, RSS, first-query, warm-query, and provider
  latency, via authenticated Space build/run logs — the only mechanism
  available for the CURRENT (legacy) baseline, since adding structured
  startup telemetry would modify the running production app and violate this
  stage's own "production Space remains unchanged" gate; telemetry can only
  measure the modernized build in later stages;
- record a retrieval-quality threshold (numeric or procedural) bound to this
  stage's query set — Stage 2's "agreed threshold" and Stage 7's "predeclared
  quality floor" gates both depend on a value produced here; until this
  action runs, that threshold is an open question (see "Open
  implementation-time questions"), not yet a fact this document can gate on;
- tag the known-good release and document the exact rollback command (see
  "Reconciled baseline" for the currently-known-good SHA and the actual
  redeploy mechanism: `git push` to the HF Space remote per
  `.github/workflows/hf-space.yml`);
- decide the exact staging Space ID before creating it, and record it in this
  document (or a linked file) before creation — this record is itself the
  gate for Stage 6's "confirming production health" and staging-cleanup step.

Gate:

- baseline data and counts are generated by scripts;
- cold-start, artifact-load, RSS, first-query, warm-query, and provider
  latency are captured as non-placeholder values (or explicitly recorded as
  UNVERIFIED with the blocking reason, e.g. authenticated-log access
  unavailable) — a script that runs without producing these specific values
  does not satisfy this gate;
- the retrieval-quality threshold and staging Space ID are recorded in this
  document (or a linked, versioned file) before Stage 2 and Stage 5
  respectively begin;
- raw outputs are retained for review;
- the current production Space remains unchanged.

#### Stage 0 results (recorded 2026-07-20) — Gate passed

All five actions above are complete; full detail, scripts, and raw evidence
live under [`docs/stage0/`](../stage0/) (not duplicated here in full):

- **Query set**: 23 queries in
  [`docs/stage0/query_set.json`](../stage0/query_set.json), generated by
  [`docs/stage0/build_query_set.py`](../stage0/build_query_set.py) — 14
  representative + 3 difficult (paraphrased, no literal keyword overlap with
  the target) queries, each grounded by script against a real row of the
  pinned dataset revision (`joshuasundance/govgis_nov2023-slim-spatial@ab1220e`),
  plus 1 empty-input, 2 malformed (long gibberish; mixed CJK/emoji/zero-width),
  2 adversarial (prompt-injection; HTML/script-injection), and 1 out-of-domain
  query. The dataset itself independently confirms two of this document's
  risk claims with real counts: 702,078 of 865,304 records (81%) have an
  empty `description`, and 29,283 (3.4%) have raw HTML markup in
  `description` (real example:
  [`docs/stage0/evidence/html_in_description_example.md`](../stage0/evidence/html_in_description_example.md)) —
  concrete grounding for the "Unsafe rendering" risk section, not a
  hypothetical. No literal `<script>` or `javascript:` content exists
  anywhere in the corpus today — the risk is architectural (what the target
  design must guard against), not an active exploit in the current data.
- **Timings**: [`docs/stage0/baseline_timings.md`](../stage0/baseline_timings.md),
  backed by raw authenticated build/run log captures in
  `docs/stage0/evidence/`. Measured: build duration 2m26s (queued to build
  complete; `pip install` alone is 93.1s of that), cold-start (container
  start to Streamlit ready) 70s, end-to-end (queued to ready) 3m43s, and an
  *inferred* ~56s artifact-deserialize/model-construct interval (not directly
  logged — app.py logs nothing at that point; see the file for the exact
  timestamp gap this is derived from). Restart sample size is n=1 — the
  run-log endpoint only retains the currently-running instance, so
  historical cold-start samples for the *current* baseline don't exist and
  can't be produced without restarting production, which this stage's own
  gate forbids. **Correction**: the earlier "build/run log endpoints return
  401" finding (Reconciled baseline, above) was an unauthenticated check;
  with the Space owner's own token both return HTTP 200 (see
  `baseline_timings.md` and the updated `AGENTS.md` gotcha). Recorded
  UNVERIFIED, each with a blocking reason: peak RSS (no memory
  instrumentation anywhere in the current app or the log streams), first-query
  and warm-query latency (no request-level logging in `app.py`; needs an
  interactive browser session against production, which no tool in this
  session could drive), and provider latency (exercising the legacy
  `claude-instant-v1`/`claude-2.1` path needs a real user-supplied Anthropic
  key that isn't available or appropriate to spend here — also moot, since
  Stage 4 benchmarks current models fresh rather than reusing a number from
  two long-retired model IDs).
- **Retrieval-quality threshold (procedural, not a pre-guessed number)**:
  Recall@3 (k=3, matching `app.py`'s `DEFAULT_SEARCH_RESULT_LIMIT`) over the
  17 representative+difficult queries that carry an `expected_urls` entry — a
  query counts as a hit if any of its expected URLs appears in the top 3
  results. **The threshold is parity, not an absolute floor**: Stage 2's
  converted (native FAISS + Sentence Transformers) index's Recall@3 on this
  query set must be `>=` the legacy `FAISS.deserialize_from_bytes` index's
  own Recall@3 on the same query set, both computed by script in the *same*
  Stage 2 run. Rationale for not computing the legacy number here: Stage 2
  must load the same 4.28 GB legacy artifact anyway (for the one-time
  conversion + its own parity check), and this machine had 16 GB free of
  953 GB total (99% used) at decision time — loading it twice to produce a
  number Stage 2 recomputes regardless wasn't worth the disk/bandwidth. The
  6 empty/malformed/adversarial/out-of-domain queries are behavioral checks,
  not Recall inputs — they feed later gates instead: `empty_input` and the
  two `malformed_*` entries into Stage 2's "corrupt/mismatched input fails
  closed" gate, `adversarial_html_injection` into Stage 3's `gr.HTML` gate,
  and `adversarial_prompt_injection` into Stage 4's secret-leak gate. Stage 7
  ("predeclared quality floor") is where an absolute Recall number gets set,
  once a real measured baseline exists from Stage 2.
- **Rollback**: local annotated tag `pre-modernization-baseline` at
  `5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9` (tree-identical to `3e85082`) —
  not yet pushed (tag pushes are a GitHub-visible change like any other; ask
  before pushing). Preferred rollback is a forward `git revert` on `main`,
  letting `hf-space.yml`'s existing on-push automation redeploy it normally.
  Emergency direct rollback (bypasses GitHub, mirrors `hf-space.yml`'s own
  push but forced against the HF Space remote specifically, not GitHub's
  `main`):
  `git push https://joshuasundance:$HF_TOKEN@huggingface.co/spaces/joshuasundance/govgis_nov2023-slim-faiss pre-modernization-baseline:main --force`.
  Full text is in the tag's annotation (`git show pre-modernization-baseline`).
- **Staging Space ID**: `joshuasundance/govgis_nov2023-slim-faiss-staging`
  (confirmed unclaimed via the Space API at decision time). Not created —
  Stage 5 creates it, with an explicit go-ahead at that time.

### Stage 1: project foundation

Actions:

- add `pyproject.toml`, package layout, lockfile, and `.python-version`;
- target Python 3.14 and validate binary wheels on the same Linux/Python
  combination used by the Space;
- separate runtime and development dependencies;
- update `.github/dependabot.yml`'s `package-ecosystem` from `pip` (which
  targets `requirements.txt`) to `uv` (which targets `pyproject.toml`/the
  lockfile) — nothing currently schedules this, and leaving it as `pip` would
  silently stop producing dependency-update PRs once `requirements.txt` is
  removed;
- update pre-commit, Ruff, mypy, pytest, and security checks;
- create GitHub Actions CI to run the complete gate — no workflow today runs
  any test, lint, or type-check tier, so this is a new job, not an update to
  an existing one;
- once that CI gate is green, configure GitHub branch protection on `main`
  requiring the new CI check(s) as required status checks, with
  admin-enforcement on, and record the exact required-check name(s) here once
  created; decide how `bumpver.yml`'s direct-push-to-`main` path is handled
  under the new protection (exempt it explicitly, or route version bumps
  through the same PR gate) — this closes the gap named in "Reliability and
  observability gaps" and is what Stage 6's "merge only after required checks
  pass" gate actually attaches to. Sequence this after the CI gate lands; a
  check cannot be required before it exists.

Gate:

- fresh locked install succeeds;
- `pytest`, Ruff, mypy, and full pre-commit are green;
- CI logs prove that each expected tier actually ran;
- branch protection on `main` is confirmed active with the CI gate as a
  required status check (verify via the same `branches/main/protection` API
  call used to establish this document's baseline — it must no longer return
  404).

Rollback:

- documentation and tooling commit can be reverted independently; production is
  still untouched.

### Stage 2: safe artifact conversion and retrieval core

Actions:

- write failing tests for manifest validation, checksum mismatch, count
  mismatch, malformed metadata, and retrieval expectations;
- convert the pinned legacy artifact in isolation;
- use Sentence Transformers and FAISS directly;
- remove runtime LangChain deserialization;
- verify retrieval parity and memory/latency.

Gate:

- no pickle or dangerous-deserialization path is reachable;
- vector count equals metadata count through a script;
- the converted index's Recall@3 on the Stage 0 query set
  (`docs/stage0/query_set.json`) is `>=` the legacy `FAISS.deserialize_from_bytes`
  index's own Recall@3 on the same query set, both computed by script in this
  same stage — the parity procedure Stage 0 recorded (see "Stage 0 results");
- the conversion process is verified to have run with no outbound network
  access (assert this programmatically — e.g. run under a sandboxed/
  network-disabled process and check — not only documented as an intent);
- corrupt or mismatched artifacts fail closed with a useful message.

Rollback:

- keep legacy production at its known-good SHA; do not deploy this stage alone.
  Note: rolling back after this stage ships re-exposes the pickle-like
  deserialization risk this stage closes (see "Critical trust boundary:
  serialized FAISS data").

### Stage 3: Gradio and secure presentation

Actions:

- implement a native Gradio Blocks app;
- retain retrieval-only operation;
- add loading, empty, error, and retry states;
- render only validated presentation models — the safe presentation model
  type owned by `govgis/models.py` and produced only by
  `govgis/presentation.py` (see "Target architecture");
- test hostile HTML, Markdown, URLs, and oversized fields in a real browser.

Gate:

- browser-level tests prove scripts and unsafe links do not execute;
- a durable, CI-enforced check (not only the one-time browser test above)
  confirms `gr.HTML` is never bound to dataset-, user-, or model-sourced
  content anywhere in the codebase — e.g. a grep/lint rule that fails the
  gate on any such usage;
- local Gradio API and browser smoke tests pass;
- required GIS fields and links are visible and usable.

Rollback:

- staging only until this and provider gates pass. Note: rolling back after
  this stage ships re-exposes the raw-HTML XSS path this stage closes (see
  "Unsafe rendering").

### Stage 4: providers and authentication

Actions:

- implement the narrow provider protocol;
- add direct official Anthropic and OpenAI clients;
- add HF OAuth with only `inference-api`;
- add the curated HF open-model lane;
- sanitize exceptions and instrument latency/usage without secrets;
- decide provider/model defaults from a recorded, script-generated comparison
  across candidate models (retrieval-answer quality score, p50/p95 latency,
  and per-model cost-per-1k-tokens as user-facing guidance — cost is
  informational since Anthropic/OpenAI are BYOK and HF uses the signed-in
  user's token, so quality and latency are the load-bearing axes); check the
  raw comparison table into the repo alongside the Stage 0 oracle outputs so
  the decision is auditable, not asserted in prose;
- implement bounded timeouts, retries limited to transient/5xx failures, and
  a per-session concurrency limit for provider calls (starting numbers: see
  "Secrets, cost, and public abuse").

Gate:

- the provider/model comparison table exists in the repo and the chosen
  defaults are traceable to it — a functional-but-unbenchmarked default does
  not satisfy this gate;
- a forced-failure test (e.g. an invalid test key) asserts no substring of the
  test key appears in captured logs, exceptions, or telemetry;
- a simulated slow/failing provider test asserts the timeout, retry-limit, and
  concurrency-limit behavior;
- provider contract tests pass with recorded fixtures or mocks;
- opt-in live smoke tests pass without exposing credentials;
- retrieval survives every provider failure;
- HF calls are attributed to the signed-in user, not the Space owner.

Rollback:

- providers are optional; disable a failing provider without disabling search.

### Stage 5: staging deployment

Actions:

- create the exact pre-approved temporary staging Space (the ID recorded in
  Stage 0);
- apply storage/preload experiments only to staging;
- deploy the branch;
- wait for `RUNNING`, verify its SHA via the Space API (`GET
  https://huggingface.co/api/spaces/<id>` → `.sha` / `.runtime.sha`, the same
  call used to reconcile this document's own baseline), and run retrieval and
  provider smoke tests;
- measure multiple (at minimum 2-3) cold builds/restarts rather than a single
  success — a single successful build does not satisfy the Gate below.

Gate:

- all local and CI gates remain green;
- staging reports the intended SHA, verified via the Space API call above and
  recorded in the deployment log — not asserted from a single manual check;
- at least 2-3 independent cold builds/restarts were measured and their
  results (not just the best one) are recorded;
- observable browser/API results meet the oracle;
- startup and memory fit `cpu-basic`, or a hardware/cost decision is recorded.

Rollback:

- stop staging and keep production on `5b3caca`.

### Stage 6: production rollout

Actions:

- merge only after required checks pass — enforced by the branch protection
  configured in Stage 1, not by convention;
- update `hf-space.yml` itself to wait for the Space to reach `RUNNING` and
  verify the deployed SHA via the Space API before treating the deploy step
  as successful, instead of the current behavior (success reported as soon as
  `git push` to the HF remote completes, regardless of whether the Space
  actually builds and starts);
- monitor GitHub deployment, HF build, runtime stage, and deployed SHA;
- execute retrieval-only and optional-provider smoke tests;
- observe through the agreed rollback window (duration: see "Open
  implementation-time questions" — record the chosen value here once decided,
  before this stage first executes);
- remove the temporary staging Space after resolving its exact ID (recorded in
  Stage 0) and confirming production health.

Gate:

- production is `RUNNING` at the merge SHA;
- endpoint and retrieval output are correct;
- no new security, latency, memory, or provider regression is observed,
  compared explicitly against the Stage 0 baseline values (this gate cannot
  be evaluated for a metric Stage 0 recorded as UNVERIFIED — treat that metric
  as not-yet-gatable rather than silently passing);
- staging cleanup is confirmed and recorded.

Rollback:

- redeploy the tagged known-good revision: `git push
  https://<user>:<token>@huggingface.co/spaces/joshuasundance/govgis_nov2023-slim-faiss
  5b3caca:main --force` (the same mechanism `.github/workflows/hf-space.yml`
  uses for forward deploys), then verify it reaches `RUNNING` and returns the
  known-good observable result via the Space API. This re-exposes the risks
  named in the Stage 2/3 rollback notes above — treat a post-Stage-2/3
  rollback as returning to a previously accepted risk level, not a
  risk-free action.

### Stage 7: evaluated footprint reduction

Only after parity, benchmark:

- smaller current embedding models;
- scalar or product quantization;
- IVF/HNSW alternatives;
- compact metadata;
- supported memory mapping.

Gate:

- compare the same query set for retrieval quality, index size, build time,
  cold-start, warm latency, memory, and cost;
- ship a smaller representation only if it meets the quality floor recorded
  in Stage 0 (this gate cannot pass until that value exists).

Rollback:

- revert to the Stage 6 shipped artifact/manifest revision and redeploy; do
  not delete the prior artifact until the new one clears the observation
  window from Stage 6.

## Central quality gate

Definition of done for implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy .
.\.venv\Scripts\python.exe -m pre_commit run --all-files
```

The final command set may be wrapped by repository scripts, but CI and local
verification must prove which test tiers were actually invoked. Add browser and
deployed-Space smoke gates separately; unit-test success is not a proxy for the
observable UI.

## Evidence sources

- Hugging Face Space configuration (also the real evidence for the Streamlit
  claim below: its `sdk` value enumeration lists only `gradio`, `docker`, or
  `static` — Streamlit is absent):
  <https://huggingface.co/docs/hub/main/spaces-config-reference>
- Streamlit SDK tutorial page (carries **no** deprecation language itself as
  of 2026-07-20; kept for reference to the Streamlit-Space authoring flow,
  not as evidence that Streamlit is deprecated — see the config-reference
  entry above for that):
  <https://huggingface.co/docs/hub/main/spaces-sdks-streamlit>
- Gradio Spaces:
  <https://huggingface.co/docs/hub/main/spaces-sdks-gradio>
- Space storage and mounts:
  <https://huggingface.co/docs/hub/main/spaces-storage>
- Space volume API:
  <https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space>
- Storage access patterns:
  <https://huggingface.co/docs/hub/main/storage-buckets-access>
- Hugging Face OAuth:
  <https://huggingface.co/docs/hub/en/spaces-oauth>
- Inference Providers:
  <https://huggingface.co/docs/inference-providers/en/index>
- Inference billing:
  <https://huggingface.co/docs/inference-providers/en/pricing>
- Gradio Markdown sanitization:
  <https://www.gradio.app/docs/gradio/markdown>
- Gradio custom HTML security:
  <https://www.gradio.app/guides/custom-HTML-components>
- Anthropic model lifecycle:
  <https://platform.claude.com/docs/en/about-claude/model-deprecations>
- Anthropic model selection:
  <https://platform.claude.com/docs/en/about-claude/models/choosing-a-model>
- OpenAI Responses migration:
  <https://developers.openai.com/api/docs/guides/migrate-to-responses>
- OpenAI current model guidance:
  <https://developers.openai.com/api/docs/guides/latest-model>
- Python 3.14.6:
  <https://www.python.org/downloads/release/python-3146/>

These links are time-sensitive. Recheck them at implementation and release
milestones.

## Review mandate for Claude Code

Before implementation, review this plan adversarially with refute as the default:

1. Reconcile every baseline claim against Git, GitHub, the live Space API, and
   current official documentation.
2. Re-run artifact counts by script and inspect real source artifacts rather
   than relying on this table.
3. Identify assumptions that lack an executable validation gate.
4. Challenge whether Gradio, Python 3.14, uv, direct provider clients, the safe
   artifact split, and HF OAuth are still the smallest reliable choices.
5. Verify that the provider defaults are active and appropriate at implementation
   time; do not preserve a stale model name merely because it is written here.
6. Verify the real producer/consumer format for the converted FAISS and metadata
   artifacts.
7. Strengthen rollback and observable tests before authorizing production work.
8. Record proposed changes as a diff to this document, including evidence and
   any newly opened decisions.

## Open implementation-time questions

These are validation questions, not blockers to recording the plan:

- ~~What are the measured current cold-start, load, and peak-RSS values?~~
  Answered by Stage 0 for cold-start/load (with one inferred figure); peak
  RSS stays UNVERIFIED — see "Stage 0 results".
- Does the current Space platform successfully build the selected Python 3.14
  dependency set?
- Does `preload_from_hub` resolve the dataset repository and pinned file as
  expected, or should the safe index live in a dedicated model repository?
- What is the exact document/vector count in the legacy artifact?
- ~~What retrieval-quality threshold and query set should govern index
  changes?~~ Answered by Stage 0: the query set is
  `docs/stage0/query_set.json`; the threshold is the Recall@3 parity
  procedure in "Stage 0 results" (the absolute number is still open until
  Stage 2 measures it).
- Which current Anthropic, OpenAI, and HF models offer the best measured
  quality/latency/cost for grounded GIS result descriptions?
- How long should the production rollback observation window remain open before
  deleting staging?
