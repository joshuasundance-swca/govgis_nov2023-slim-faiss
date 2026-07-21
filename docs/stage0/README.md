# Stage 0 — baseline and test oracle

Supporting artifacts for `docs/modernization-plan.md`'s Stage 0 (see that
document's "Stage 0 results" subsection for the recorded decisions this
directory backs). Nothing here is authoritative on its own — the plan is.

- `build_query_set.py` — deterministic script that grounds every
  representative/difficult query in `query_set.json` against a real row of
  the pinned dataset revision (`joshuasundance/govgis_nov2023-slim-spatial@ab1220e`).
  Re-run with `python docs/stage0/build_query_set.py`; it downloads only the
  216 MB metadata geoparquet, never the 4.28 GB FAISS artifact or the 5.6 GB
  embeddings parquet — loading either is Stage 2's job, which needs them
  anyway for the real conversion and parity check.
- `query_set.json` — the 23-query test oracle: 14 representative + 3
  difficult (grounded, real expected URLs) + 1 empty-input + 2 malformed + 2
  adversarial + 1 out-of-domain (behavioral checks, `expected_urls: []`).
- `baseline_timings.md` — cold-start/build/artifact-load timings measured
  from authenticated Space build/run logs, with UNVERIFIED values (RSS,
  first-query/warm-query latency, provider latency) and their blocking
  reasons spelled out.
- `evidence/` — raw log/API captures the timings above were derived from
  (`build_log_2026-07-20.jsonl`, `run_log_2026-07-20.jsonl`,
  `space_runtime_2026-07-20.json`), retained per Stage 0's gate ("raw
  outputs are retained for review").

Retrieval-quality threshold, rollback tag/command, and the staging Space ID
are recorded directly in the plan's "Stage 0 results" subsection, not
duplicated here.
