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

## Confirmed decisions

1. Migrate the UI from the deprecated built-in Streamlit Space SDK to the
   native Gradio Space SDK.
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
- The Space API reported `storage: null` and did not expose a `volumes` field.
- Public build and run log endpoints returned HTTP 401. Build duration,
  artifact-load duration, peak RSS, and cold-start time are therefore
  **UNVERIFIED**, not inferred from the healthy endpoint.
- The local worktree was clean before this documentation branch was created.

Known-good rollback points:

- GitHub and Space revision:
  `5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9`
- Recovery implementation commit:
  `3e85082`
- Existing runtime contract: Python 3.11 and Streamlit 1.29.0, with the
  Streamlit version controlled only by Space README metadata.

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
must never opt into dangerous deserialization.

### Unsafe rendering

The current app passes dataset-controlled descriptions directly to a raw HTML
component. This is an XSS path. Names, URLs, fields, model output, and dataset
descriptions must all be treated as untrusted.

Target policy:

- parse records into strict Pydantic v2 models;
- accept external links only with `http` or `https` schemes;
- convert source HTML descriptions to plain text for the first release;
- escape generated Markdown and retain Gradio HTML sanitization;
- do not use `gr.HTML` for dataset, user, or model-controlled content;
- verify the painted browser result with adversarial fixtures.

### Prompt injection

Retrieved ArcGIS metadata is untrusted content supplied to an LLM. Delimit it as
data, state that instructions inside it are not authoritative, request claims
grounded in returned records, and test malicious record content. Prompt text
alone is not a security boundary; output rendering and link validation still
apply.

### Secrets, cost, and public abuse

- Anthropic and OpenAI keys are BYOK and session-scoped.
- Hugging Face inference uses the signed-in user's OAuth token with only the
  `inference-api` scope.
- Never use owner-funded production provider keys in the public Space unless a
  later explicit decision adds authentication, quotas, spending limits, and
  abuse controls.
- Provider errors must be typed and sanitized.
- Add bounded timeouts, retries only for transient failures, concurrency limits,
  and user-visible rate/cost guidance.

### Reliability and observability gaps

The current deployment workflow reports success when Git push succeeds; it does
not wait for the Space build, verify the deployed SHA, or exercise retrieval.
The current required PR check only enforces a file-size limit. These proxies are
not sufficient release gates.

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

- `app.py`: Gradio composition and event wiring only.
- `govgis/config.py`: validated environment and artifact configuration.
- `govgis/models.py`: Pydantic request, GIS record, result, and provider models.
- `govgis/artifacts.py`: manifest/checksum validation and artifact loading.
- `govgis/retrieval.py`: query embedding and direct FAISS search.
- `govgis/providers/base.py`: narrow provider protocol and common errors.
- `govgis/providers/anthropic.py`: official Anthropic Messages client.
- `govgis/providers/openai.py`: official OpenAI Responses client.
- `govgis/providers/huggingface.py`: `InferenceClient` using HF OAuth.
- `govgis/presentation.py`: plain-text conversion, link validation, and escaped
  Markdown.

Remove LangChain and LangSmith unless a measured requirement appears that the
small direct architecture cannot meet. The current chain adds dependency churn,
conceals serialization risk, and is not needed for one retrieval step plus one
optional generation step.

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
  Sonnet quality option. Do not retain `claude-instant-v1` or `claude-2.1`;
  both are retired.
- OpenAI: use the Responses API with `store=False`; start with an efficient
  current model tier and offer a stronger quality tier.
- Hugging Face: curate a small set of available open chat models and use
  Inference Provider routing. Do not expose arbitrary provider-specific
  parameters in the first release.

Shared controls should be limited to portable concepts such as provider, model,
temperature where supported, and maximum output tokens. Provider-specific
capabilities should not leak into the common interface without a tested need.

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

### Stage 0: baseline and test oracle

Actions:

- create 15–25 representative GIS queries with expected relevant URLs;
- include difficult, empty, malformed, and adversarial records;
- capture cold-start, artifact-load, RSS, first-query, warm-query, and provider
  latency;
- tag the known-good release and document the exact rollback command;
- decide the exact staging Space ID before creating it.

Gate:

- baseline data and counts are generated by scripts;
- raw outputs are retained for review;
- the current production Space remains unchanged.

### Stage 1: project foundation

Actions:

- add `pyproject.toml`, package layout, lockfile, and `.python-version`;
- target Python 3.14 and validate binary wheels on the same Linux/Python
  combination used by the Space;
- separate runtime and development dependencies;
- update pre-commit, Ruff, mypy, pytest, and security checks;
- update GitHub Actions to invoke the complete gate.

Gate:

- fresh locked install succeeds;
- `pytest`, Ruff, mypy, and full pre-commit are green;
- CI logs prove that each expected tier actually ran.

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
- baseline retrieval quality meets the agreed threshold;
- corrupt or mismatched artifacts fail closed with a useful message.

Rollback:

- keep legacy production at its known-good SHA; do not deploy this stage alone.

### Stage 3: Gradio and secure presentation

Actions:

- implement a native Gradio Blocks app;
- retain retrieval-only operation;
- add loading, empty, error, and retry states;
- render only validated presentation models;
- test hostile HTML, Markdown, URLs, and oversized fields in a real browser.

Gate:

- browser-level tests prove scripts and unsafe links do not execute;
- local Gradio API and browser smoke tests pass;
- required GIS fields and links are visible and usable.

Rollback:

- staging only until this and provider gates pass.

### Stage 4: providers and authentication

Actions:

- implement the narrow provider protocol;
- add direct official Anthropic and OpenAI clients;
- add HF OAuth with only `inference-api`;
- add the curated HF open-model lane;
- sanitize exceptions and instrument latency/usage without secrets;
- decide provider/model defaults from representative quality, cost, and latency
  measurements.

Gate:

- provider contract tests pass with recorded fixtures or mocks;
- opt-in live smoke tests pass without exposing credentials;
- retrieval survives every provider failure;
- HF calls are attributed to the signed-in user, not the Space owner.

Rollback:

- providers are optional; disable a failing provider without disabling search.

### Stage 5: staging deployment

Actions:

- create the exact pre-approved temporary staging Space;
- apply storage/preload experiments only to staging;
- deploy the branch;
- wait for `RUNNING`, verify its SHA, and run retrieval and provider smoke tests;
- measure multiple cold builds/restarts rather than a single success.

Gate:

- all local and CI gates remain green;
- staging reports the intended SHA;
- observable browser/API results meet the oracle;
- startup and memory fit `cpu-basic`, or a hardware/cost decision is recorded.

Rollback:

- stop staging and keep production on `5b3caca`.

### Stage 6: production rollout

Actions:

- merge only after required checks pass;
- monitor GitHub deployment, HF build, runtime stage, and deployed SHA;
- execute retrieval-only and optional-provider smoke tests;
- observe through the agreed rollback window;
- remove the temporary staging Space after resolving its exact ID and confirming
  production health.

Gate:

- production is `RUNNING` at the merge SHA;
- endpoint and retrieval output are correct;
- no new security, latency, memory, or provider regression is observed;
- staging cleanup is confirmed and recorded.

Rollback:

- redeploy the tagged known-good revision and verify it reaches `RUNNING` and
  returns the known-good observable result.

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
- ship a smaller representation only if it meets the predeclared quality floor.

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

- Hugging Face Space configuration:
  <https://huggingface.co/docs/hub/main/spaces-config-reference>
- Streamlit SDK deprecation:
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

- What are the measured current cold-start, load, and peak-RSS values?
- Does the current Space platform successfully build the selected Python 3.14
  dependency set?
- Does `preload_from_hub` resolve the dataset repository and pinned file as
  expected, or should the safe index live in a dedicated model repository?
- What is the exact document/vector count in the legacy artifact?
- What retrieval-quality threshold and query set should govern index changes?
- Which current Anthropic, OpenAI, and HF models offer the best measured
  quality/latency/cost for grounded GIS result descriptions?
- How long should the production rollback observation window remain open before
  deleting staging?
