# Stage 0 baseline timings — 2026-07-20

Legacy production Space: `joshuasundance/govgis_nov2023-slim-faiss`, SHA
`5b3cacaf27fc4c75cb4e6e4c3d86dc1796ece5c9` (confirmed live via authenticated
`get_space_runtime` at the time of this capture — matches the known-good
baseline in `docs/modernization-plan.md`'s "Reconciled baseline").

## Correction to a prior finding

`docs/modernization-plan.md`'s "Reconciled baseline" and `AGENTS.md`'s
gotchas record that the public build/run log endpoints returned HTTP 401
during the 2026-07-20 adversarial review. That check was **unauthenticated**.
Retried today with the Space owner's (`joshuasundance`) cached
`huggingface_hub` token — the same identity that owns this dataset/model/Space
— both `GET /api/spaces/<id>/logs/build` and `.../logs/run` returned **HTTP
200** as `text/event-stream`. Authenticated log access to one's own Space is
available; it was only the anonymous check that failed. `AGENTS.md` is
updated accordingly.

Method: `huggingface_hub.get_token()` (the CLI's cached token) as a bearer
token against both endpoints, streamed and capped (8 MB / 90s) to avoid
hanging on the run log's live tail. Script: not checked in (one-off `requests`
call); raw captured output is retained below.

## Raw evidence retained

- `docs/stage0/evidence/space_runtime_2026-07-20.json` — `get_space_runtime` dump
- `docs/stage0/evidence/build_log_2026-07-20.jsonl` — full build log (865 SSE lines; the stream terminates naturally once historical build output is exhausted)
- `docs/stage0/evidence/run_log_2026-07-20.jsonl` — run log, capped at 8MB/90s (this stream tails live and does not terminate on its own; 186 non-empty lines were captured, spanning 16:26:09Z-19:28:33Z)

## Build timeline (one observed build: today's PR #46 deploy of `5b3caca`)

| Event | Timestamp (UTC) |
| --- | --- |
| Build queued (commit `5b3caca`) | 16:23:36.000Z |
| `pip install` step completes | 16:25:24.518Z (step itself: **93.1s**) |
| Final build step (`DONE 0.5s`) | 16:26:02.026Z |

**Build duration (queued -> build complete): 2m26s.** `pip install`-ing
`requirements.txt` is the dominant cost (93.1s of the 146s total) — worth
re-measuring once Stage 1 lands a locked `uv` environment, since `uv` install
is typically far faster than pip for an equivalent dependency set.

## Cold-start timeline (container start -> serving)

| Event | Timestamp (UTC) |
| --- | --- |
| `===== Application Startup =====` (container process start) | 16:26:09Z |
| Streamlit ready (`You can now view your Streamlit app in your browser`) | 16:27:19.064Z |

**Cold-start (container start -> Streamlit ready): 70s.**
**End-to-end (build queued -> Streamlit ready): 3m43s (223s).**

## Artifact-load timeline (first real session only — `@st.cache_resource` means this pays once per process)

`app.py` has no explicit "artifact loaded" log line, so this section
distinguishes what is **directly logged** from what is **inferred** from a
timestamp gap.

| Event | Timestamp (UTC) | Directly logged? |
| --- | --- | --- |
| First script execution begins (first `LangChainDeprecationWarning` burst = first real visitor/session since Streamlit came up) | 16:50:59.114Z | yes |
| FAISS artifact download starts (4.28 GB) | 16:51:01.143Z | yes (progress bar) |
| FAISS artifact download ends | 16:51:07.365Z (~6.2s, up to 605 MB/s) | yes |
| Embedding model files download (dominant: `model.safetensors`, 1.34 GB) | 16:51:16.713Z -> 16:51:18.801Z (~2.1s, up to 378 MB/s) | yes |
| Last small config/tokenizer file downloaded | 16:51:19.685Z | yes |
| Next session's first script execution begins | 16:52:15.525Z | yes |

Note: **16:27:19Z -> 16:50:59Z is a 23m40s gap with zero log activity** —
this is idle wait for the first real visitor/session after the container was
ready, not part of cold-start or artifact-load. It is excluded from the
figures below.

**Inferred in-memory load time:** the 56s between the last download-progress
line (16:51:19.685Z) and the next session's warning burst (16:52:15.525Z) is
the best available proxy for `FAISS.deserialize_from_bytes` (docstore
deserialization) plus `HuggingFaceBgeEmbeddings` model construction — neither
is logged directly. Treat this as an **estimate**, not a measured value.

**Combined artifact-load estimate (first visitor to ready-to-serve): ~67s**
(~11s download + ~56s inferred deserialize/construct), on top of whatever
idle time elapsed before that first visitor.

Confirms `@st.cache_resource` works as intended: subsequent sessions
(16:52:15Z, 16:54:46Z, 17:02:31Z, 17:31:49Z, 17:36:54Z, 19:27:50Z, 19:28:33Z —
8 sessions total in the captured window) show only the import-time
deprecation-warning burst, never another download-progress burst.

## Restart sample size: n=1

The run-log endpoint only retains logs for the **currently running**
instance/process — there is no API path to pull historical cold-start
samples from prior restarts of the current baseline. The one restart
captured here is today's PR #46 deploy. Stage 5's "at least 2-3 independent
cold builds/restarts" requirement targets the **staging** Space (restarted
deliberately for that purpose); it is not achievable for the current
production baseline without deliberately restarting production, which this
stage's own gate forbids ("current production Space remains unchanged").

## UNVERIFIED values (with blocking reason, per this stage's Gate)

- **Peak RSS**: UNVERIFIED. `app.py` has no memory instrumentation, and the
  build/run log streams do not surface container-level memory. Getting this
  would require either HF's hardware/usage dashboards (not reachable via the
  API endpoints checked here) or adding telemetry to the running app, which
  this stage's gate forbids.
- **First-query / warm-query latency**: UNVERIFIED. `app.py` performs no
  request-level logging, and Streamlit's script-execution model does not log
  per-session elapsed time. Measuring this needs a live interactive browser
  session against production; no browser-automation tool was available in
  this session. A future session with one could close this gap with a
  read-only timed page load + query submission (using the app exactly as any
  public visitor would, not modifying it).
- **Provider (Anthropic) latency**: UNVERIFIED. Exercising `claude-instant-v1`
  / `claude-2.1` requires a real user-supplied Anthropic key in the BYOK
  sidebar field; no key is available or appropriate to spend against in this
  session. Also moot as a forward-looking baseline: both model IDs are
  long-retired (see `AGENTS.md`), and Stage 4 already calls for benchmarking
  *current* models fresh rather than reusing this number.
