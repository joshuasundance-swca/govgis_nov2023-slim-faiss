# GitHub Ecosystem Recon: govgis Dataset Pipeline

Read-only research into every GitHub resource adjacent to the `govgis_nov2023` dataset
ecosystem, gathered via `gh` CLI and the GitHub API on 2026-07-20. Purpose: inform a future
rebuild/modernization of the underlying **govgis dataset pipeline** (not just this Space).
Nothing here was invented; every claim is traceable to a `gh api`/`gh search` call or a
live HTTP fetch made during this session.

## Identity map

- **`joshuasundance-swca`** — GitHub `type: User` (not an org). Bio: "Data Science Architect
  at SWCA Environmental Consultants (@swca)". Owns 65 repos (`gh repo list --limit 200`).
  This is where essentially all govgis-adjacent work lives.
- **`joshuasundance`** — GitHub `type: User`, no bio. **Owns zero repos** (`gh api
  users/joshuasundance/repos` returns an empty list). It does own one unrelated gist
  (Jenks Natural Breaks arcpy script, 2020) and is separately the **Hugging Face**
  username/namespace (`huggingface.co/joshuasundance/...`, `huggingface.co/datasets/
  joshuasundance/...`) — a different platform's identity, not a second GitHub account
  with repos. No org account exists at either handle.
- A broad `gh search repos govgis` and `gh search code govgis_nov2023` across all of
  GitHub turned up no third-party fork, collaborator repo, or org repo that mirrors or
  extends this ecosystem. The only non-`joshuasundance-swca` hits were (a) an unrelated
  Chinese OpenLayers demo literally named `govgis` (`niuzhendong/govgis`, no connection),
  and (b) `Lexicom7/EO_Datasets`, a third-party repo whose `docs/DATASET_CATEGORIES.md`
  cites `govgis_nov2023` by name as an example of a "survey data / metadata compilation"
  dataset type — i.e. an external project is aware of and references the dataset, but
  does not fork or build on it. Everything else matching the string "govgis" in GitHub
  code search was pure textual coincidence (security-research notes, expired-domain
  lists, etc.) with no relation to this project.

---

## Core repos (govgis pipeline)

### `govgis_nov2023-slim-faiss` (this repo)
- **URL**: https://github.com/joshuasundance-swca/govgis_nov2023-slim-faiss
- **Purpose**: The Streamlit-based Hugging Face Space that is the subject of this
  research — semantic search UI over the `govgis_nov2023` dataset using a FAISS
  index (`govgis_nov2023-slim-nospatial.faiss.bytes`, pulled at runtime from HF repo
  `joshuasundance/govgis_nov2023-slim-spatial`) and BAAI/bge-large-en-v1.5 embeddings.
- **Tech stack**: Python 3.11 (Space) / 3.14 (declared in pyproject), Streamlit 1.29,
  FAISS, Claude (Anthropic) for optional query rephrasing/response generation.
- **Status**: Active. Latest commit 2026-07-20 (same day as this recon) — "Fix Hugging
  Face Space Python build contract" — merged just hours before this research ran. Before
  that, the repo was dormant since 2024-02-05 (a run of dependabot bumps), so the
  modernization-plan work landing this month is the first real activity in ~2.5 years.
  Workflows: `bumpver.yml`, `check-file-size-limit.yml`, `hf-space.yml` (push-to-HF-Space
  on change) — none are scheduled/cron; there is no data-refresh automation here.
- **Relationship to pipeline**: Terminal consumer. Reads a pre-built FAISS index off
  Hugging Face; does not crawl or build data itself.
- **Notable**: README explicitly credits "Joseph Elfelt and the creators of the `restgdf`
  library" (see Data Lineage section below). No open GitHub issues. TODO list in README
  is UI/feature scope (open-source model option, hybrid BM25 search, geospatial
  filtering), not data-pipeline related.

### `govgis_nov2023-slim-spatial-server`
- **URL**: https://github.com/joshuasundance-swca/govgis_nov2023-slim-spatial-server
- **Purpose**: The "more advanced sibling" — Dockerized Postgres+PostGIS+pgvector stack
  that loads the same `govgis_nov2023` data (as a geoparquet file) into a spatially- and
  vector-indexed `layers` table, serves it via FastAPI, and (added 2025-09-30) exposes a
  **FastMCP** server (`backend/mcp.py`, `FastMCP(name="govgis_nov2023")`) with a single
  `gis_layer_search` tool that proxies to the FastAPI `/search` endpoint and markdownifies
  the response. `agent.ipynb` demonstrates driving it with `deepagents`.
- **Tech stack**: Python 3.13, asyncpg, geopandas, shapely, pgvector, FastAPI, FastMCP,
  Docker Compose (postgres + postgres-init + pgadmin), uses the `joshuasundance/
  postgis_pgvector:1.0.0` Docker image (see below).
- **Status**: Active-ish. Last real commit 2025-10-09 ("add 60 second timeout calling
  vector db api"); before that, an ingestion-throughput optimization (PR #63, "5-10x /
  25x" speedup via vectorized loading + native pgvector adapter) landed 2025-10-02. As of
  2026-07-20 there are **5 open PRs**, mostly dependabot/Snyk dependency and base-image
  bumps (#75 "Bump the app group... 17 updates", #76/#73/#71 Snyk Python base-image
  upgrades) — i.e. dependency maintenance is lagging by ~9-10 months of unmerged bumps,
  though no functional issues are open.
- **Relationship to pipeline**: Downstream consumer + a step up in sophistication from
  this repo — adds spatial querying, vector similarity in Postgres, and an MCP tool
  surface. `backend/load_data.py` **loads a pre-built `.geoparquet` file** downloaded
  from HF (`GEOPARQUET_PATH` env var) — it does **not** crawl ArcGIS servers itself.
  Confirms: the actual dataset-build/crawl step happens somewhere outside this repo (not
  found in any of the GitHub repos surveyed — see Gaps below).
- **Notable**: Two HF dataset dependencies: `joshuasundance/govgis_nov2023` and
  `joshuasundance/govgis_nov2023-slim-spatial`. `.env-example` shows the geoparquet is
  expected at `postgres-init/govgis_nov2023_slim_spatial_embs.geoparquet`.

### `restgdf`
- **URL**: https://github.com/joshuasundance-swca/restgdf
- **Purpose**: "Async, typed Python client for Esri ArcGIS REST services — turn
  FeatureServer/MapServer layers into GeoDataFrames." This is the **core crawling/client
  library** almost certainly used (directly or via a predecessor) to build the
  `govgis_nov2023` dataset from live ArcGIS REST endpoints.
- **Tech stack**: Python ≥3.9, asyncio/aiohttp, Pydantic v2 (typed response models —
  `LayerMetadata`, `CrawlReport`, etc.), optional GeoPandas extra, optional
  `restgdf[resilience]` extra (stamina + aiolimiter for retry/rate-limiting) and
  `restgdf[telemetry]` extra (OpenTelemetry).
- **Maturity / PyPI**: Published on PyPI (`pip install restgdf`), currently on
  **v3.0.0** (released 2026-05-03; v2.0.0 released 2026-04-21; v1.0.0 released
  2023-12-15; earliest tagged release 0.9.7, 2023-11-14 — i.e. it existed in
  pre-1.0 form right around the "nov2023" snapshot date). Has readthedocs docs
  (including `llms.txt`/`llms-full.txt` for LLM consumption) and a DeepWiki page. CI
  (pytest), coverage, and PyPI-publish GitHub Actions are all green/wired up.
  `Directory.crawl` / `safe_crawl` helpers exist for bulk crawling ArcGIS service
  directories, and a `CrawlReport` typed model — strong evidence this library's crawl
  path is what built the dataset.
- **Status**: Actively developed. Most recent commits are from 2026-05-05, but the repo
  has **6 open PRs as of 2026-07-20**, all dependency bumps (dependabot: aiohttp,
  urllib3, pydantic-settings, soupsieve, plus a GH Actions group bump) except one real
  feature PR, **#175 "Add spatial filter geometry payload helper"**, still open. Recent
  closed PRs show a v3 release push: #172 "docs: comprehensive documentation overhaul for
  v3 release", #170 "Tighten release-readiness checks", #169 "Harden gate-3 follow-up
  fixes" — i.e. restgdf just went through a major typed-model rewrite (2.0) and a
  further release (3.0), and is the most actively maintained repo in this whole
  ecosystem.
- **No open non-PR issues.**
- **Relationship to pipeline**: The client library. Used by `restgdf_api` and cited
  directly in its own README's "Uses" section alongside the `govgis_nov2023` HF dataset.
- **Data-lineage note (README does NOT explain server-list assembly)**: restgdf's README
  and docs describe *how to crawl a given ArcGIS server/layer* but do not explain how
  the *list* of ~1,684 government servers was originally assembled. That provenance is
  documented in a sibling tool, `restgdf_api` (see below), and in the older `dataripper`
  repo — not in restgdf itself. No file in restgdf's tree (`scripts/` contains only
  `bumpver_stamp_date.py`; `docs/` is API reference only) or its git history references
  "govgis" or "1684" — confirmed via `gh search code` scoped to the repo and a full
  `git/trees` listing.

### `restgdf_api`
- **URL**: https://github.com/joshuasundance-swca/restgdf_api
- **Purpose**: "openapi-documented arcgis proxy & geospatial data discovery server" — a
  FastAPI proxy in front of `restgdf`, with a **`mappingsupport` router** that is the
  single most important data-lineage artifact found in this recon (see below).
- **Tech stack**: FastAPI, restgdf, pandas, Docker (published to Docker Hub as
  `joshuasundance/restgdf_api`), Kubernetes manifests included.
- **Status**: **Stale.** Last commit 2024-04-17 (over 2 years old as of this recon).
  No CI/pytest workflow visible in its workflow list (`bumpver.yml`, `docker-hub.yml`
  only). Has **2 open, unaddressed real issues** (not PRs — see below), both about the
  same upstream data source breaking, oldest dating to 2024-04-17 and the newer one
  open since 2025-12-15 with zero comments/response.
- **Relationship to pipeline**: **This is the data-lineage smoking gun.**
  `restgdf_api/mappingsupport.py` implements a live proxy over:
  ```
  https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.csv
  ```
  — Joseph Elfelt's "surf_gis" CSV of federal/state/county/city government GIS server
  URLs (columns: Line-number, Type, State, County, Town, FIPS, Server-owner, ArcGIS-url,
  https, Show-contents, SSL, Open, Comment, ...). It caches the CSV for a week
  (`604800` seconds) and exposes `/mappingsupport/`, `/mappingsupport/state/`,
  `/mappingsupport/county/`, `/mappingsupport/town/` endpoints to filter it. This is
  almost certainly the **origin list of government ArcGIS server URLs** that were then
  crawled (with `restgdf`) to build `govgis_nov2023`, directly matching the README
  acknowledgment of "Joseph Elfelt and the creators of the restgdf library."
- **Known open issues (critical for a rebuild)**:
  - **#74 "Mapping Support CSV"** (opened 2025-12-15, still open, 0 comments): *"The
    CSV at <https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-
    servers.csv> is no longer available, would you by chance happen to have a copy of
    this?"* — **This session independently confirmed via live fetch (2026-07-20) that
    this exact CSV URL now returns HTTP 404.** This is a direct threat to any plan to
    re-crawl a newer snapshot using the same pipeline shape.
  - **#31** ("`get_df` ParserError: Error tokenizing data. C error: Expected 15 fields
    in line 3569, saw 25", opened 2024-04-17, still open): a malformed-row / schema-drift
    problem in the same CSV (extra unescaped commas or embedded delimiters on some rows)
    — i.e. this source was already known to be dirty/inconsistently formatted **before**
    it went missing entirely.
- **Live-fetch addendum (this session, not from GitHub)**: `mappingsupport.com` itself
  (root domain) is still online and Elfelt's project page is still active — it states
  "An updated list is usually posted each Wednesday morning" and now advertises **"7,500+
  addresses for government ArcGIS servers"** (up substantially from the ~1,684 servers
  actually present in the Nov 2023 snapshot — consistent with either significant list
  growth since 2023, or with 1,684 representing only the subset that were successfully
  reachable/crawled). The specific `.csv` path is dead (404); a `.pdf` variant of the
  list at the same `/p/surf_gis/` path (referenced in `dataripper`'s README, see below)
  did not error but timed out while fetching (large file) rather than confirming
  content — so the safest documented conclusion is: **the CSV endpoint this codebase
  depends on is gone, but Elfelt's underlying list/project is still alive and actively
  maintained in some format**, and re-establishing a working ingestion path (CSV, PDF,
  or contacting Elfelt per the issue's own suggestion) is a prerequisite for any refresh.

### `geospatial-data-converter`
- **URL**: https://github.com/joshuasundance-swca/geospatial-data-converter
- **Purpose**: Standalone Streamlit + GeoPandas app for converting between geospatial
  formats (KML/KMZ/GeoJSON/EsriJSON/WKT/GPX/Shapefile/OpenFileGDB/TopoJSON/CSV), with
  direct "ArcGIS feature layer URL" input support. Deployed both as a Docker image
  (Docker Hub `joshuasundance/geospatial-data-converter`) and an HF Space.
- **Tech stack**: Python 3.14, Streamlit, GeoPandas, pydeck (map preview), Docker.
  Notably more mature CI/release engineering than the govgis repos: has a real `ci.yml`
  (pre-commit + pytest 3.14 + wheel/sdist build + `twine check` + Docker smoke tests),
  a `release-assets.yml`, and a documented dry-run release process in the README.
- **Status**: Active. Last commit 2026-05-05.
- **Relationship to pipeline**: Tangential/utility — not part of the govgis dataset
  build or serving chain, but directly reuses `restgdf`-adjacent skills (ArcGIS layer
  fetch) and is topically tagged `restgdf` in its GitHub topics. Best understood as a
  generic tool from the same author/toolkit, useful as a UX/release-engineering
  reference for modernizing the other Spaces (its CI and release workflow are
  noticeably more rigorous than `govgis_nov2023-slim-faiss`'s).
- **Notable**: One closed issue, #54 "Zipped GDB to KML Error" — resolved, no open
  issues.

---

## Adjacent / predecessor repos

### `dataripper`
- **URL**: https://github.com/joshuasundance-swca/dataripper
- **Purpose**: "Mini-library to pull GIS data from ArcGIS REST service layers" —
  Josh's **original, pre-`restgdf` tool** for the same job. README: "Introduces the
  ripper class. Inspired by [DataPillager](https://github.com/gdherbert/DataPillager)
  but has no arcpy dependency." Also bundles `gdftoge`, Google Earth ↔ GeoPandas ↔ Esri
  interop helpers.
- **Tech stack**: Jupyter Notebook / conda (`environment.yml`), no async, no Pydantic —
  a simpler, pre-restgdf-era tool.
- **Status**: **Dormant since 2020-07-01** (last commit). Effectively an archived
  predecessor.
- **Relationship to pipeline**: Direct tooling ancestor of `restgdf`. Its README contains
  the *same* Joseph Elfelt acknowledgment, dated 2020, well before the govgis_nov2023
  snapshot: *"Special shout-out to Joseph Elfelt and his [list of GIS servers]
  (https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.pdf),
  which I used for testing."* This establishes that Josh has been using Elfelt's server
  list continuously since at least 2020 — confirming the mappingsupport.com list is the
  long-standing, single source of truth for "which government ArcGIS servers exist" across
  this entire body of work, not something adopted only for the Nov 2023 build.
- **Also notable**: its README image link points to `github.com/joshuasundance/...`
  (the personal, no-longer-repo-owning handle) — evidence the repo was originally created
  under the personal account and later consolidated under `joshuasundance-swca` (or the
  link is simply stale; GitHub redirects renamed-owner links automatically so this isn't
  fully conclusive).

### `arcgis-api-python-docker`
- **URL**: https://github.com/joshuasundance-swca/arcgis-api-python-docker
- **Purpose**: "docker image for the arcgis api for python (maintained substitute for
  deprecated esridocker/arcgis-api-python-notebook)" — a Docker image wrapping Esri's
  official (non-REST, SDK-based) `arcgis` Python API, as a maintained replacement for
  Esri's own now-deprecated Docker image.
- **Status**: Stale — last commit 2024-01-29 (a dependabot merge).
- **Relationship to pipeline**: Tangential. Different technology path (Esri's official
  SDK vs. raw REST via restgdf) — likely a separate concern from the govgis crawl
  pipeline, more useful for ArcGIS Pro/AGOL-authenticated workflows.

### `postgis_pgvector`
- **URL**: https://github.com/joshuasundance-swca/postgis_pgvector
- **Purpose**: "docker image for postgresql w/ postgis & pgvector" — the base Docker
  image (`joshuasundance/postgis_pgvector:1.0.0`) that `govgis_nov2023-slim-spatial-
  server`'s `docker-compose.yml` depends on directly.
- **Status**: Stale — last commit 2023-11-20 (version bump to 1.0.0), essentially frozen
  since release, right around the original nov2023 snapshot date.
- **Relationship to pipeline**: Direct infrastructure dependency of the spatial-server
  sibling repo. Small surface area, low maintenance need (it's a thin Dockerfile wrapper
  around upstream postgis/pgvector), so staleness here is lower-risk than elsewhere.

### `restgdf_api` docker/pipeline note
(Covered in Core repos above — included here only as a cross-reference: it is the
data-source proxy, not the pipeline's compute layer.)

### `MCP-Server-ArcGIS-Pro-AddIn` (fork)
- **URL**: https://github.com/joshuasundance-swca/MCP-Server-ArcGIS-Pro-AddIn
- **Fork of**: `nicogis/MCP-Server-ArcGIS-Pro-AddIn` (external author, not Josh's own
  work).
- **Purpose**: An MCP server paired with an ArcGIS Pro Add-In (C#), i.e. exposing ArcGIS
  Pro desktop functionality to MCP-speaking agents.
- **Status**: Last commit 2025-08-28 (a merge from upstream `master`) — kept roughly in
  sync with upstream, not independently developed.
- **Relationship to pipeline**: Not part of the govgis data pipeline, but directly
  relevant as **prior art for MCP-over-GIS** patterns — worth a look given
  `govgis_nov2023-slim-spatial-server` just added its own FastMCP server. This fork
  shows a different flavor (desktop ArcGIS Pro integration vs. server-side dataset
  search) of the same "MCP + GIS" idea Josh is already pursuing.

### `aprx_explorer`
- **URL**: https://github.com/joshuasundance-swca/aprx_explorer
- **Purpose**: "tool for creating a tabular history of an ArcGIS Pro project, with
  optional LLM-generated text summaries."
- **Status**: Stale — last commit 2024-02-08.
- **Relationship to pipeline**: Tangential — ArcGIS Pro project (`.aprx`) introspection,
  not REST-server crawling. Not part of the govgis dataset chain.

### `perona_malik`
- **URL**: https://github.com/joshuasundance-swca/perona_malik
- **Purpose**: "Perona Malik raster smoothing" — a geospatial raster-processing
  algorithm implementation.
- **Status**: Stale — single initial commit, 2022-03-14.
- **Relationship to pipeline**: Tangential GIS (raster processing), unrelated to the
  vector-layer govgis dataset.

### `usgs_natmap_old`
- **URL**: https://github.com/joshuasundance-swca/usgs_natmap_old
- **Purpose**: "Download USGS data using input geometry and CSV output from USGS TNM
  Download V1.0" — a different government-GIS-data-source tool (USGS National Map, not
  ArcGIS REST servers).
- **Status**: Stale — last commit 2020-06-26.
- **Relationship to pipeline**: Tangential — same broad "government GIS data acquisition"
  space as govgis, but a different agency/API (USGS TNM, not Esri ArcGIS REST) and a
  much older, unrelated tool.

### `paper-chaser-mcp` (fork, non-GIS — noted for completeness only)
- **URL**: https://github.com/joshuasundance-swca/paper-chaser-mcp
- **Fork of**: `Silung/scholar-search-mcp` (external).
- **Purpose**: FastMCP server for academic literature search/citation chasing — not
  GIS-related.
- **Relationship to pipeline**: None directly. Flagged only because it's a second,
  independent example of Josh working with **FastMCP** server patterns, which is the
  same framework the spatial-server sibling just adopted for its GIS MCP tool — a
  possible reference point for MCP-server conventions, not a data source.

---

## Gists

Checked both identities' gists (`gh api users/<login>/gists`). None relate to the govgis
pipeline, data crawling, or ArcGIS server lists. Notable but unrelated:
`joshuasundance-swca` has a gist "Use LangChain to summarize ArcGIS Pro geoprocessing
history" (2024-02-08, pairs with the `aprx_explorer` repo) and one on compiling
`mistral.rs` with CUDA. `joshuasundance` (personal) has one gist, an arcpy Jenks Natural
Breaks script (2020), unrelated to govgis.

---

## Data lineage summary (synthesized)

1. **Server list source**: Joseph Elfelt's `mappingsupport.com` "surf_gis" project — a
   volunteer-maintained directory of federal/state/county/city government ArcGIS/GIS
   server URLs, updated roughly weekly. Used continuously by Josh's tooling since at
   least 2020 (`dataripper`), through the exact CSV endpoint proxied by `restgdf_api`
   (`mappingsupport.py`), up to the `govgis_nov2023` build.
2. **Crawl/extraction tooling**: `restgdf` (formerly `dataripper`) — an async, typed
   Python client that turns ArcGIS FeatureServer/MapServer REST endpoints into
   GeoDataFrames/typed metadata, with directory-crawl and `CrawlReport` support built
   for exactly this kind of bulk-server sweep.
3. **The actual "build govgis_nov2023" script/notebook was not found in any GitHub repo
   surveyed.** `govgis_nov2023-slim-spatial-server/backend/load_data.py` only *loads* a
   pre-built `.geoparquet` from Hugging Face; it does not crawl. No repo in this account
   contains a script that (a) pulls the mappingsupport list, (b) crawls it with restgdf,
   and (c) emits the `govgis_nov2023` dataset artifacts. That step happened outside of
   what's visible on GitHub (local script, notebook never committed, or logic living
   only inside the Hugging Face dataset repo itself, which is out of scope for this
   GitHub-only recon).
4. **Consumers**: `govgis_nov2023-slim-faiss` (this repo, FAISS/Streamlit search) and
   `govgis_nov2023-slim-spatial-server` (Postgres/PostGIS/pgvector + FastAPI + FastMCP)
   both consume pre-built HF dataset artifacts rather than rebuilding them.

## Gaps / risks for a future rebuild (factual, not speculative beyond what's cited above)

- **The upstream CSV endpoint the pipeline used is currently dead** (HTTP 404, confirmed
  live 2026-07-20), and this has been an open, unanswered issue in `restgdf_api` (#74)
  since 2025-12-15. The root `mappingsupport.com` site is still active and claims
  "7,500+" server addresses maintained on a roughly weekly cadence, but the exact
  machine-readable path/format currently in use was not confirmed in this session (the
  `.csv` path 404s; a `.pdf` variant did not error but the fetch timed out before
  content could be confirmed).
- **The CSV format has known drift/quality issues independent of availability** — issue
  #31 in `restgdf_api` (2024) shows inconsistent field counts row-to-row, meaning any
  rebuild ingestion needs a tolerant parser, not a fixed-width `pandas.read_csv(names=...)`
  call like the current `mappingsupport.py` uses.
- **No scheduled/cron GitHub Actions workflow exists anywhere in this ecosystem** for
  re-crawling or refreshing the dataset. All `.github/workflows/*.yml` files found across
  the five core repos are CI, version-bump (`bumpver.yml`), Docker Hub publish, or
  HF-Space-push automations — none are data-refresh pipelines. A "nov2023 → newer
  snapshot" refresh would need new automation built from scratch.
- **`restgdf_api` itself (the mappingsupport-proxying service) is 2+ years stale** and
  would likely need a refresh/rewrite to match `restgdf`'s current v3 API before reuse.
- **Dependency maintenance is uneven across the ecosystem**: `restgdf` is actively
  released (v3.0.0, May 2026) with only routine dependabot PRs pending; the two govgis
  serving repos have several-month backlogs of unmerged dependency/security PRs;
  `restgdf_api`, `postgis_pgvector`, `arcgis-api-python-docker`, `dataripper`,
  `aprx_explorer`, `perona_malik`, and `usgs_natmap_old` are all effectively frozen.
