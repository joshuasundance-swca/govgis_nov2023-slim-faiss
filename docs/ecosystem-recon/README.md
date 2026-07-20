# govgis ecosystem recon — synthesis

Recorded 2026-07-20, in parallel with this repo's Stage 1-4 modernization
build (see `docs/modernization-plan.md`). Pure information-gathering, no code
changes, requested to inform a **future, separate effort**: rebuilding the
underlying `govgis` dataset pipeline itself with better data engineering, not
just this one Space's UI. Four sources, each written independently and cross-
checked against each other where they overlap:

- [`local-findings.md`](local-findings.md) — coordinator, local filesystem
- [`github-findings.md`](github-findings.md) — subagent, `gh` CLI/API
- [`huggingface-findings.md`](huggingface-findings.md) — subagent, `huggingface_hub`/Hub API
- [`upstream-source-findings.md`](upstream-source-findings.md) — subagent, WebSearch/WebFetch

## The full picture, in one pass

**Data lineage (now fully reconstructed, converging from three independent
angles — GitHub code, HF notebooks, and live web research all agree):**

1. **Root source**: Joseph Elfelt's `mappingsupport.com` "surf_gis" list —
   a volunteer-maintained, weekly-refreshed catalog of government ArcGIS REST
   server roots, running since ~2018. Confirmed **still active** (last
   updated June 18, 2026) and has grown from ~2,200 (2020) to **7,500+
   servers today** — roughly 4-5x the ~1,684 servers behind `govgis_nov2023`
   (Nov 2023). Continuously used by this maintainer's tooling since at least
   2020 (`dataripper`'s README carries the same acknowledgment).
2. **Crawl tooling**: `restgdf` (formerly `dataripper`, rewritten as an
   async/typed client) — pulls the seed list via a proxy service
   (`restgdf_api`'s `mappingsupport.py` router), then crawls each server root
   with concurrency-limited, retried async requests. The actual "build
   govgis_nov2023" script was **not found in any GitHub repo** — it exists
   only as notebooks inside the `govgis_nov2023` Hugging Face dataset repo
   itself (`scrape_2_11142023.ipynb`, `to_parquet.ipynb`, etc.), never
   committed to GitHub.
3. **Two HF dataset repos, one snapshot, no versioning**: `govgis_nov2023`
   (3.18GB, full 205-column raw metadata, 1,684 servers / 195,479 services /
   865,864 layers) and `govgis_nov2023-slim-spatial` (9.46GB, the flattened
   7-column + embeddings + legacy FAISS variant this Space consumes). Both
   are single-burst, single-branch, zero-tag, frozen since Nov 2023 — no
   precedent anywhere in this ecosystem for publishing successive dataset
   snapshots.
4. **Consumers**: this Space (FAISS/Streamlit→Gradio) and
   `govgis_nov2023-slim-spatial-server` (Postgres+PostGIS+pgvector+FastAPI+
   FastMCP, actively developed, real prior art for "better data
   engineering"). Both load pre-built artifacts; neither crawls.

## The two hardest blockers for any rebuild

1. **The exact upstream feed this pipeline depended on is dead.** The CSV
   endpoint `restgdf_api`'s `mappingsupport.py` proxies
   (`mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.csv`)
   returns HTTP 404 — confirmed by two independent live fetches this session
   (the GitHub-recon agent and the coordinator's earlier session both hit
   the same 404). An open, unanswered `restgdf_api` issue (#74, since
   2025-12-15) reports the same thing. Elfelt's underlying list is alive and
   growing (7,500+, updated weekly) and the `.txt` mirror still works — a
   rebuild needs to re-establish a working ingestion path (the `.txt`
   mirror, the `.pdf`, or contacting Elfelt directly) before anything else.
2. **Licensing has real teeth, not just boilerplate.** Elfelt's list itself:
   scraping the PDF is explicitly prohibited, commercial use needs his
   written permission, free derivative works are explicitly welcomed. Below
   that: federal-level government GIS data is public domain (17 U.S.C.
   §105), but **state/local is not uniformly public domain** — cited,
   directly-conflicting court rulings exist (NY/SC permit state copyright
   assertions over GIS data; FL/CA hold the opposite). A rebuild that
   redistributes scraped metadata at scale, not just links to it, should not
   assume blanket public-domain status below the federal level, and should
   get Elfelt's explicit sign-off before any commercial use.

## Other findings worth carrying into a rebuild's design

- **No comparable server-root-level catalog exists.** FGDC/GeoPlatform.gov/
  data.gov all operate at the dataset-metadata level, not the raw
  server-endpoint level Elfelt's list uniquely fills.
- **`restgdf` is the most actively maintained repo in the whole ecosystem**
  (v3.0.0 as of May 2026, real CI, PyPI-published, ReadTheDocs) — the
  crawling tool itself is not the bottleneck; the seed-list feed and the
  never-automated build pipeline are.
- **No scheduled/cron refresh workflow exists anywhere** — a "nov2023 →
  current" refresh needs new automation built from scratch, on both the
  crawl side and (per this Space's own modernization plan) the artifact/
  manifest/versioning side.
- **A real architecture reference already exists and is under active
  development**: `govgis_nov2023-slim-spatial-server`'s Postgres+pgvector+
  FastAPI+FastMCP stack is a concrete, working example of "GIS metadata as
  an agent-callable search tool" — worth reading in full before designing
  a rebuild's serving layer, rather than starting from this Space's
  FAISS-file approach.
- **A real CI/CD reference also already exists**: `geospatial-data-converter`
  (same author) has meaningfully more mature release engineering (wheel/
  sdist build, `twine check`, Docker smoke tests, a documented dry-run
  release process, GitHub→HF-Space mirroring) than any of the govgis repos —
  a good pattern to borrow for whatever ships the rebuilt pipeline.
- **Format gotcha**: Hugging Face's Dataset Viewer does not support
  `.geoparquet` (confirmed via a still-open upstream issue,
  `huggingface/datasets#6438`, filed by this same maintainer in 2023, no
  resolution since Feb 2024). Any rebuilt dataset repo should ship a plain
  `.parquet` sibling (WKB/WKT or split lat/lon columns) if a working preview
  matters.
- **Storage**: both existing dataset repos use plain Git LFS, no Xet — worth
  a deliberate choice for a rebuild that would otherwise re-upload
  near-duplicate multi-GB files on every refresh.

## Confidence

Every claim above traces to a live API call, a `gh`/Hub fetch, or a directly
quoted web source in the four linked files — nothing here was inferred
without being labeled as such in the source file it came from. Where a
sub-agent could not verify something (e.g. the DataPillager→restgdf lineage
claim, GeoPlatform.gov's bulk-API capability), it says so explicitly in
`upstream-source-findings.md` rather than asserting it.
