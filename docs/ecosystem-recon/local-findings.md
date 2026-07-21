# Local filesystem recon — govgis ecosystem

Recorded 2026-07-20 by the coordinator (done inline rather than via a workflow
subagent, to avoid permission stalls on paths outside this repo's working
directory). Scope: `C:\Users\Joshua.Bailey\PycharmProjects\` and this repo.

## Sibling repo: `govgis_nov2023-slim-spatial-server`

Path: `C:\Users\Joshua.Bailey\PycharmProjects\govgis_nov2023-slim-spatial-server`
GitHub: `joshuasundance-swca/govgis_nov2023-slim-spatial-server` (public)

A **much more advanced, actively-developed** sibling project — a Dockerized
Postgres+PostGIS+pgvector backend serving the govgis dataset, with a FastAPI
REST API and a FastMCP (Model Context Protocol) server on top, so AI agents
can call `gis_layer_search` directly. This is real prior art for "dramatically
improved data engineering" — worth studying before designing whatever comes
after this Space.

- **Components** (`docker-compose.yml`): `postgres` (custom image
  `joshuasundance/postgis_pgvector:1.0.0`), `postgres-init` (loads the
  geoparquet into Postgres via `backend/load_data.py`), `backend` (FastAPI,
  `uvicorn app:app`), `mcp` (FastMCP server wrapping the backend's `/search`
  endpoint as an MCP tool `gis_layer_search`), `inspector` (MCP Inspector UI),
  `pgadmin`.
- **Data source it consumes**: NOT the same file this repo (`slim-faiss`)
  uses. It reads
  `govgis_nov2023_slim_spatial_embs.geoparquet` (5.66 GB, embeddings
  pre-baked into the parquet — same file whose size I recorded in Stage 0's
  artifact-download check) directly into Postgres/pgvector, no FAISS
  involved at all. Confirmed present locally at
  `govgis_nov2023-slim-spatial-server/govgis-nov2023/govgis_nov2023_slim_spatial_embs.geoparquet`
  (5,656,518,817 bytes, matches the Hub API size exactly).
- **`backend/app.py`**: FastAPI `/search` endpoint. Builds a query via
  `SemanticSearchRequest.build_query(embedding_model)` (see `backend/models.py`
  — not fully read, just referenced) against a `layers` table, using the SAME
  `BAAI/bge-large-en-v1.5` embedding model as this repo, loaded via
  **`langchain_community.embeddings.HuggingFaceBgeEmbeddings`** (still
  LangChain-based, unlike this repo's Stage 2 plan to go LangChain-free).
- **`backend/mcp.py`**: wraps `/search` as an MCP tool, markdownifies string
  fields in the response (interesting pattern for LLM-friendly output).
- **`agent.ipynb`** (47 KB notebook, not read cell-by-cell): per the README's
  "UPDATES (09/30/2025)" section, demonstrates using `deepagents` (a Python
  agent framework) against this MCP server.
- **README explicitly links two HF datasets**:
  - `https://huggingface.co/datasets/joshuasundance/govgis_nov2023` — the
    **full, non-slim** dataset (this repo only uses the "slim" variant; the
    full one has not been examined yet — see the Hugging Face recon lane).
  - `https://huggingface.co/datasets/joshuasundance/govgis_nov2023-slim-spatial`
    — same dataset this repo (`slim-faiss`) is pinned to.
- **Stack**: Python 3.13 (this Space still targets 3.11 pre-Stage-1, 3.14
  post-Stage-1), pre-commit/ruff/mypy/black/bandit (same tooling lineage as
  this repo's pre-Stage-1 config), `bumpver.toml` (same versioning tool),
  Snyk vulnerability badge (not present in this repo).
- **Recent git history** (`git log --oneline -15`) shows active, current
  work: async Postgres driver migration (asyncpg, with a noted TODO to move
  to psycopg3), ingestion/query performance tuning, vector index work, a
  60-second API timeout fix, and GitHub Copilot-authored commits — this
  project is not dormant.
- **`.env`** exists locally with real values (not read/exposed here — only
  `.env-example`'s variable names, which are safe placeholders, were read).
  Do not commit or display its contents.
- **`models/` dir**: a local HuggingFace hub cache for `BAAI/bge-large-en-v1.5`
  (mounted into the `backend`/`postgres-init` containers) — not source code,
  just a cache.

## No other local govgis-adjacent projects found

`ls ../` (siblings of this repo under `PycharmProjects/`) surfaced only these
two govgis-named directories; no other local clone, notebook, or data file
elsewhere under this repo or its immediate siblings referenced "govgis" by
name. (Recon did not do a full-disk search outside `PycharmProjects/` and
this repo's own working directories — see the parallel GitHub/HF/web recon
lanes for anything not on this machine.)

## This repo's own `.gitignore` hints, now explained

This repo's `.gitignore` has long carried `hf_cache/` and `govgis-nov2023/`
as ignored patterns even though neither exists in *this* repo's working
tree — they're almost certainly leftover/copied from the sibling
`govgis_nov2023-slim-spatial-server` repo's `.gitignore` (which legitimately
uses a `govgis-nov2023/` directory for its mounted raw data), or added
defensively in case a contributor mirrors that pattern here. Not a bug, but
worth knowing the provenance if it ever looks like dead config.
