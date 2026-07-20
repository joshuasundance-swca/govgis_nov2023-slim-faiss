# Hugging Face Hub recon — govgis ecosystem

Recorded 2026-07-20. Scope: every dataset/model/Space under the `joshuasundance`
namespace, anything matching `govgis` anywhere on the Hub, and the six orgs the
authenticated user (`joshuasundance`) belongs to (`LangChainDatasets`, `swca`,
`Major-TOM`, `ml-intern-explorers`, `build-small-hackathon`,
`ICML-2026-agent-repro`). All data pulled live via `huggingface_hub.HfApi`
(`list_datasets`/`list_models`/`list_spaces`, `repo_info(..., files_metadata=True)`)
and the raw Hub REST API (`/api/datasets/<repo>/commits`, `/refs`,
`/discussions`) using the cached token. Nothing below is inferred — sizes, row
counts, and commit SHAs are as returned by the API; schema/methodology details
come from reading the actual notebooks and README source in each repo.

See also `docs/ecosystem-recon/local-findings.md` (written separately) for the
sibling GitHub project `govgis_nov2023-slim-spatial-server` — a Postgres +
pgvector + FastAPI + MCP server that already consumes
`govgis_nov2023_slim_spatial_embs.geoparquet` directly (no FAISS), and is
actively developed. That is real prior art for "better data engineering" and
should be read alongside this file.

## 1. Full namespace inventory

### `joshuasundance/*` datasets (15 total; 2 are govgis-related)

| repo_id | downloads | last modified |
|---|---|---|
| **`joshuasundance/govgis_nov2023`** | 63 | 2023-11-17 |
| **`joshuasundance/govgis_nov2023-slim-spatial`** | 36 | 2023-11-23 |
| `joshuasundance/wikiquote_tv` | 15 | 2023-12-11 |
| `joshuasundance/mtg-coloridentity-multilabel-classification` | 28 | 2024-01-31 |
| `joshuasundance/mypo-4k-rfc` | 8 | 2024-07-14 |
| `joshuasundance/mypo-4k-rfc-val-phi3test` | 10 | 2024-07-14 |
| `joshuasundance/python-code-instructions-85k-mypo` | 16 | 2026-04-27 |
| `joshuasundance/python-code-instructions-85k-mypo-qaqc` | 28 | 2026-04-27 |
| `joshuasundance/codex-7m-probe-stats` | 9 | 2026-04-27 |
| `joshuasundance/codex-7m-qaqc-smoke` | 14 | 2026-04-28 |
| `joshuasundance/codex-7m-qaqc-raw` | 22 | 2026-04-29 |
| `joshuasundance/codex-7m-sft-strict-intermediate` | 6 | 2026-04-29 |
| `joshuasundance/codex-7m-dpo-strict-intermediate` | 25 | 2026-04-29 |
| `joshuasundance/codex-7m-sft-strict-intermediate-v2` | 5 | 2026-04-30 |
| `joshuasundance/codex-7m-dpo-strict-intermediate-v2` | 5 | 2026-04-30 |

None of the other 13 datasets are GIS-related (MTG card classification, a
Python-code-instructions/DPO/SFT training-data lineage, a TV-quote corpus).

### `joshuasundance/*` models (13 total; 0 govgis-related)

`setfit-absa-*` (2), `mtg-coloridentity-multilabel-classification`,
`phi3-mini-4k-qlora-python-code-*` (3), `mypo-training`,
`mypo-qwen2.5-coder-1.5b-{sft,dpo-v2,dpo-v3}`, `myponline`,
`myponline-checkpoint`, `myponline-sft-q15`. None reference GIS/geospatial
data.

### `joshuasundance/*` Spaces (8 total; 1 govgis, 1 GIS-adjacent)

| repo_id | sdk | private | notes |
|---|---|---|---|
| **`joshuasundance/govgis_nov2023-slim-faiss`** | streamlit | no | this repo |
| **`joshuasundance/geospatial-data-converter`** | docker | no | GIS-adjacent, see §5 |
| `joshuasundance/langchain-streamlit-demo` | docker | no | unrelated |
| `joshuasundance/mtg-coloridentity` | streamlit | no | unrelated |
| `joshuasundance/streamlit-gpt4o` | docker | no | unrelated |
| `joshuasundance/trackio` | gradio | **yes** | unrelated |
| `joshuasundance/mypo-live` | gradio | **yes** | unrelated |
| `joshuasundance/myponline-dashboard` | gradio | **yes** | unrelated |

### Hub-wide search for `govgis` (no author filter)

- Datasets: exactly `joshuasundance/govgis_nov2023` and
  `joshuasundance/govgis_nov2023-slim-spatial` — no third-party forks, mirrors,
  or unofficial copies exist on the Hub.
- Models: **zero** results.
- Spaces: exactly `joshuasundance/govgis_nov2023-slim-faiss` — no other Space
  built on this data exists on the Hub (the Postgres/MCP server from
  `local-findings.md` is a local/GitHub-only project, not deployed as a Space).

### Org sweep (the 6 orgs `joshuasundance` belongs to)

| org | datasets | models | spaces | GIS/govgis-related? |
|---|---|---|---|---|
| `LangChainDatasets` | 11 | 0 | 0 | No — early LangChain example datasets (Paul Graham QA, SQL-QA-Chinook, agent-vectordb-qa, etc.), unrelated to govgis. |
| `swca` | 0 | 0 | 0 | Empty org, nothing to check. |
| `Major-TOM` | 25 | 0 | 1 | **No govgis overlap**, but this is a legitimate large-scale geospatial ML org (Sentinel-1/2, DEM, land-cover, embeddings, a "Spatial-Reference-Grid" dataset, a Core Viewer Space) — worth studying as a pattern reference for how a mature geospatial-dataset org structures multiple companion datasets + a viewer Space, even though it's Earth-observation imagery, not ArcGIS service metadata. |
| `ml-intern-explorers` | 2 | 15 | 9 | No — unrelated ML experiments (music, vision, agent traces). |
| `build-small-hackathon` | 132 | 149 | 854 | No — a large hackathon org (agent traces, game/companion apps, LoRA fine-tunes). Grepped every repo name in all three listings; zero GIS/geo/govgis matches. |
| `ICML-2026-agent-repro` | ~23 | 0 | 0 | No — ICML reproducibility-challenge repos (paper reproductions, judge/verdict logbooks). |

**Conclusion: no govgis-related or otherwise GIS-related repo exists in any of
the six orgs.** The entire govgis footprint on the Hub is the two datasets and
one Space already known, all under the personal `joshuasundance` namespace.

## 2. `joshuasundance/govgis_nov2023` — the full, non-slim dataset

- **repo_id**: `joshuasundance/govgis_nov2023`, type `dataset`, public, license MIT.
- **Latest revision (== `main` HEAD)**: `4f59df9cb8903367ddeea75e5ad916efe7de746b`
- **DOI**: `10.57967/hf/1368` (auto-registered by the Hub)
- **Tags**: `language:en`, `license:mit`, `size_categories:100K<n<1M`,
  `format:parquet`, `modality:text`, `modality:geospatial`, `library:datasets`,
  `library:dask`, `library:mlcroissant`, `library:polars`, `gis`, `geospatial`
- **Total size**: 3.18 GB (3,419,631,298 bytes) across 11 files.

### File breakdown

| file | size | contents |
|---|---|---|
| `govgis-nov2023-mongodb.gz` | 1.05 GB (1,129,192,237 B) | `mongodump` archive of DB `govgis-nov2023`: a `services` collection plus one collection **per layer type** (`feature_layer`, `raster_layer`, `group_layer`, etc. — 18 distinct ArcGIS layer types seen in the notebook). No CRS reprojection applied here. |
| `jsonfiles.tar.gz` | 898.61 MB (942,264,569 B) | The raw per-server scrape output — one JSON file per successfully-scraped ArcGIS root URL (server + nested services + layers, straight from `restgdf`), before any flattening. |
| `layers.parquet` | 1.19 GB (1,275,703,456 B) | **865,864 rows** (verified via parquet metadata — 560 more rows than the slim-spatial dataset's 865,304, see §4) × **205 raw columns** — essentially every field ArcGIS REST can return for a layer: `fields`, `drawingInfo`, `capabilities`, `extent`, `subtypes`, `domains` (via `types`/`templates`), `relationships`, `editFieldsInfo`, `timeInfo`, `archivingInfo`, versioning flags, etc. All columns are cast to `str` (see `to_parquet.ipynb`), so nested JSON structures survive as stringified dicts/lists rather than typed columns. |
| `servers.parquet` | 73.99 KB (75,769 B) | **1,684 rows** × 4 columns (`currentVersion`, `url`, `hash`, `error`) — this is the authoritative count of the "~1,684 government ArcGIS servers" already known; confirmed directly from row count, not just the README's prose claim. |
| `services.parquet` | 68.99 MB (72,344,852 B) | **195,479 rows** × ~200 raw columns — service-level metadata (one row per ArcGIS "service", i.e. the container above individual layers). |
| `to_mongo_4326reproj.py` | 7.87 KB | Script (described in §6) that reprojects layer extents to EPSG:4326 and writes into MongoDB DB `govgis-nov2023-slim-spatial` — the actual production source of the slim-spatial dataset. |
| `to_mongodb.ipynb` | 6.33 KB | Notebook version of the (non-reprojected) Mongo load into DB `govgis-nov2023`, split by layer type — this produced `govgis-nov2023-mongodb.gz`. |
| `to_parquet.ipynb` | 9.84 KB | Notebook that flattens the raw per-server JSON into the three `*.parquet` files above (all columns `.astype(str)`, empty strings dropped as all-null columns). |
| `scrape_2_11142023.ipynb` | 10.57 KB | The actual scrape notebook (see §6 methodology). |
| `failed_scrape_11142023.ipynb` | 9.31 KB | An earlier/exploratory scrape pass — filters out JSON files lacking a `metadata` key or with empty content (1,684 valid files survive out of an initial 2,038 candidate root URLs). |
| `README.md` | 3.06 KB | Dataset card (GPT-4-authored per its own disclosure). |

### Hub metadata / freshness

- **12 commits**, all between 2023-11-15 04:54 and 2023-11-17 16:16 — a single
  ~2.5-day creation burst. **No commits since 2023-11-17** (≈2 years 8 months
  stale as of 2026-07-20).
- **Refs**: only a `main` branch; **no tags** (no semantic versioning of any
  kind). An automatic Hub bot (`refs/convert/parquet`) generated a
  parquet-conversion ref at commit `9399c017...` on 2023-11-16 — this succeeds
  because the data files are already plain `.parquet`, so the **Dataset
  Viewer works normally** for this repo.
- **1 open, unaddressed discussion**: "[bot] Conversion to Parquet" from the
  `parquet-converter` bot (opened 2023-11-16, 1 comment, never closed —
  informational only, not an issue).
- No tags/branches beyond `main`; no releases; no model card widget.

### How it differs from `govgis_nov2023-slim-spatial`

This is the **raw, wide** version: full per-layer/service/server metadata as
returned by ArcGIS REST, split into three relational-ish parquet tables
(servers → services → layers) plus the raw JSON and two different MongoDB
exports (one reprojected/deduped into a single `layers` collection, one
per-layer-type and un-reprojected). It has **~28x more columns** on the layer
table than slim-spatial (205 vs. 7) and **560 more rows** (865,864 vs.
865,304 — likely layers whose extent failed CRS reprojection/validation and
were dropped when slim-spatial's geometry column was built; see §6). It has
**no embeddings** and **no FAISS index** — those only exist in slim-spatial.
It is genuinely the dataset a rebuilt pipeline should start from, since it is
the only artifact that retains full field schemas, renderer/`drawingInfo`,
capabilities, subtypes/domains, and versioning info per layer.

## 3. `joshuasundance/govgis_nov2023-slim-spatial` — the dataset this Space consumes

- **repo_id**: `joshuasundance/govgis_nov2023-slim-spatial`, type `dataset`, public, license MIT.
- **Latest revision (== `main` HEAD)**: `ab1220e6823732093a1c8a0122af98f7da1f4217`
- **DOI**: `10.57967/hf/1369`
- **Tags**: `language:en`, `license:mit`, `size_categories:100K<n<1M`,
  `modality:geospatial`, `gis`, `geospatial` (notably **no** `format:parquet`
  or `library:*` tags — the Hub does not recognize `.geoparquet` as a known
  format, see below).
- **Total size**: 9.46 GB (10,153,589,456 bytes) across 5 files.

### File breakdown

| file | size | contents |
|---|---|---|
| `govgis_nov2023-slim-nospatial.faiss.bytes` | 3.99 GB (4,280,523,962 B) | Legacy LangChain-serialized FAISS blob (already known). |
| `govgis_nov2023_slim_spatial_embs.geoparquet` | 5.27 GB (5,656,518,817 B) | Same 865,304 rows plus `BAAI/bge-large-en-v1.5` embedding vectors (already known). |
| `govgis_nov2023_slim_spatial.geoparquet` | 206.50 MB (216,533,216 B) | **865,304 rows × 7 columns**: `id`, `name`, `type`, `description`, `url`, `metadata_text`, `geometry` — confirmed by reading `govgis_nov2023_slim-spatial_gpd.ipynb`'s own `gdf.head()` output. `geometry` is a `POLYGON` — the layer's **bounding-box extent reprojected to EPSG:4326**, not the actual feature geometry (there is no true feature-level spatial data anywhere in this ecosystem; "spatial" here means "each row has a georeferenced bbox for map display/filtering", not vector features). |
| `govgis_nov2023_slim-spatial_gpd.ipynb` | 7.15 KB | The tiny build notebook — just loads the geoparquet with GeoPandas and inspects shape/head; does not show how `metadata_text`/embeddings were generated (that logic lives elsewhere, presumably in this Space's own history, not in the dataset repo). |
| `README.md` | 3.52 KB | Dataset card (Zephyr-7B-authored per its own disclosure). |

### Hub metadata / freshness

- **8 commits**, all between 2023-11-19 20:53 and 2023-11-23 00:18 (another
  single creation burst, ~3 days). **No commits since 2023-11-23** (≈2 years
  8 months stale).
- **Refs**: only `main`; **no tags**; **no `refs/convert/parquet`** (unlike
  the full dataset) — confirming the Hub's parquet-conversion bot does not
  recognize `.geoparquet` files at all.
- **1 open discussion**: "Dataset Viewer issue" (opened by `joshuasundance`
  himself, 2023-11-20, 3 comments). The Dataset Viewer fails with
  `ConfigNamesError`/`FileNotFoundError` because `datasets` doesn't treat
  `.geoparquet` as a supported data file extension. The discussion links to
  an upstream issue the user filed himself:
  [`huggingface/datasets#6438` "Support GeoParquet"](https://github.com/huggingface/datasets/issues/6438)
  — verified live via `gh issue view`: **state is still `OPEN`**, created
  2023-11-20, last activity 2024-02-07 (a `weiji14`/CONTRIBUTOR comment
  proposing to allowlist `.geoparquet`/`.gpq` extensions to reuse the Parquet
  reader), no further activity since. This is a real, still-unresolved Hub
  limitation directly relevant to modernization: **any future dataset repo
  that ships `.geoparquet` will have the same broken preview** unless it also
  ships a plain `.parquet` sibling (WKB/WKT geometry as a string column) or
  the upstream issue lands.

### Revision-pin check against this repo's `docs/modernization-plan.md`

`docs/modernization-plan.md` pins `joshuasundance/govgis_nov2023-slim-spatial`
at `ab1220e6823732093a1c8a0122af98f7da1f4217` (lines 155-156, 561). That SHA
is **exactly the current `main` HEAD** returned by the Hub API — this repo is
**not** pinned to a stale/outdated snapshot; there simply is no newer revision
to be behind. The dataset has had zero commits since that pin was presumably
made in 2023-11-23. (The full `govgis_nov2023` dataset is likewise just sitting
at its one and only post-creation-burst HEAD, `4f59df9c...`.)

## 4. `joshuasundance/govgis_nov2023-slim-faiss` — this Space

Already fully known per the task brief; not re-derived. Confirmed via
`list_spaces`: `sdk: streamlit`, public, no `lastModified` reported by the
list API (Spaces don't expose it the same way repos do).

## 5. `joshuasundance/geospatial-data-converter` — adjacent GIS Space (not govgis-branded, but same author/ecosystem)

- **repo_id**: `joshuasundance/geospatial-data-converter`, type `space`, `sdk: docker`, public, pinned.
- **Latest revision**: `5071d691bab51f84fb9c82136d0ffc1e38170730`
- Not part of the govgis data pipeline (it doesn't touch the govgis dataset),
  but it is the **same author's actively-maintained GIS tooling**, GitHub-mirrored
  (`joshuasundance-swca/geospatial-data-converter`) with a full CI/CD stack that
  is directly relevant as a modernization pattern reference:
  - GitHub Actions workflows: `ci.yml` (pre-commit + pytest on Python 3.14 +
    wheel/sdist build + `twine check` + Docker smoke test), `docker-hub.yml`
    (publishes to Docker Hub), `hf-space.yml` (**mirrors the GitHub repo to
    this HF Space on push** — a concrete, working example of the
    GitHub→HF-Space CI pattern this repo's modernization plan would need),
    `release-assets.yml`, `bumpver.yml` (version bump automation), and a
    file-size-limit check.
  - `pyproject.toml` + `.pre-commit-config.yaml` present (ruff/mypy/black per
    badges), `Dockerfile` based on `python:3.14-slim-bookworm`.
  - Functionally: a Streamlit app for converting between geospatial formats
    (KML/KMZ/GeoJSON/EsriJSON/WKT/GPX/Shapefile-zip/FileGDB), including an
    **"ArcGIS feature layer URL" input source** (`arcgis_loader.py`) that
    fetches directly from ArcGIS REST services — i.e., the same class of data
    source govgis harvests from, but as a live single-layer fetch rather than
    a bulk crawl.
  - Has a real test suite (`tests/test_arcgis_loader.py`,
    `tests/test_conversions.py`) with fixture data — a level of test coverage
    the govgis dataset-build notebooks themselves never had.

## 6. Data-collection methodology (reconstructed from the notebooks/scripts in `govgis_nov2023`)

Read in full: `scrape_2_11142023.ipynb`, `failed_scrape_11142023.ipynb`,
`to_parquet.ipynb`, `to_mongodb.ipynb`, `to_mongo_4326reproj.py`.

1. **Seed list**: a `restgdf_api` Docker service exposes
   `/mappingsupport/{state,county,town,root}` endpoints that scrape
   [mappingsupport.com](https://mappingsupport.com) (Joseph Elfelt's public,
   manually-curated list of government ArcGIS servers — **not** an
   Esri/ArcGIS-official index). The notebook regexes out unique
   `.../rest/services` root URLs from an `ArcGIS-url` column and dedupes them
   — this produced **2,038 unique candidate root URLs**, saved to
   `unique_roots.csv`.
2. **Crawl**: `restgdf.utils.crawl.fetch_all_data` (the user's own
   [`restgdf`](https://github.com/joshuasundance-swca/restgdf) library, an
   async ArcGIS REST client) is called per root URL through an `aiohttp`
   session, concurrency-limited via `asyncio.Semaphore(10)`, retried with
   `tenacity` (`stop_after_attempt(3)`, exponential backoff), writing one JSON
   file per server to disk. The final production run (`scrape_2_...`, output
   dir `output_tryagain`) took **2h16m wall-clock** for 2,038 URLs; an earlier
   pass without the concurrency semaphore (`failed_scrape_...`, output dir
   `output`) took 24m38s for the same count but is explicitly named "failed"
   in the filename, implying data-quality issues untangled by the
   semaphore-throttled rerun.
3. **Filtering**: of files produced, any missing a `metadata` key or with
   empty content are deleted — **1,684 of the 2,038 candidate servers survive**
   as the final server count (confirmed exactly by `servers.parquet`'s 1,684
   rows).
4. **Two parallel downstream pipelines from the same raw JSON**:
   - `to_parquet.ipynb`: flattens server→service→layer into three tables,
     everything cast to `str`, IDs from Python's non-deterministic built-in
     `hash()` (session-local, not stable across runs) → `servers.parquet`,
     `services.parquet`, `layers.parquet` (the full dataset).
   - `to_mongodb.ipynb` (DB `govgis-nov2023`) and `to_mongo_4326reproj.py`
     (DB `govgis-nov2023-slim-spatial`): both use a **deterministic UUID**
     scheme (`random.seed(url); uuid.UUID(bytes=..., version=4)`) instead of
     `hash()` — a meaningfully different/better ID strategy than the parquet
     pipeline. The `_4326reproj` variant additionally reprojects each layer's
     `extent` from its native `spatialReference` (via `pyproj`, trying
     `EPSG`/`ESRI` authority codes, falling back to raw WKT) into an EPSG:4326
     GeoJSON polygon, clamping out-of-range coordinates and discarding any
     `extent` that fails reprojection or is degenerate (`GEOSException`,
     `CRSError`, collinear points). **This is the exact source of
     slim-spatial's `geometry` column** and almost certainly explains the
     560-row shortfall vs. the full dataset's `layers.parquet`.
5. **Snapshot date**: "as of November 15, 2023" per the README; the actual
   scrape (per notebook timestamps/filenames) ran 2023-11-14 through
   2023-11-17. It is explicitly a **one-time static snapshot**, not a
   maintained/re-crawled index — the README says so directly ("This is a
   static snapshot and not actively maintained like Joseph Elfelt's ongoing
   listings. However, this foundation may evolve into a maintained index.").
   Nearly 2 years 8 months have passed since with no re-crawl.

## 7. Model: `BAAI/bge-large-en-v1.5`

Not owned by `joshuasundance`; the embedding model used to produce
`govgis_nov2023_slim_spatial_embs.geoparquet`'s vectors. Confirmed as the only
embedding-model reference anywhere in this ecosystem (both dataset READMEs
name it explicitly; no other embedding model appears in any repo, commit, or
README examined). No govgis-specific fine-tune of it exists on the Hub (the
`models search=govgis` query returned zero results).

## 8. Storage/versioning implications for a rebuilt pipeline

- **No dedicated versioning strategy exists today.** Both dataset repos are
  single-burst, single-branch, zero-tag repos — every file was uploaded once
  and never touched again. There is no precedent in this ecosystem for
  publishing successive snapshots (e.g. `govgis_2024`, `govgis_2025`) as
  either new repos or new branches/tags of the same repo — a rebuild aiming
  at "actively maintained" (as the slim-spatial README itself speculates)
  would need to establish that pattern from scratch.
- **Plain Git LFS throughout, no Xet.** Checked the raw Hub API
  (`?blobs=true`) for both dataset repos: `xetEnabled` is absent/`False` on
  both, and every large file carries a standard `BlobLfsInfo`
  (sha256 + pointer_size). Both repos predate Hugging Face's Xet rollout, so
  this is expected, but it means a from-scratch rebuild has a live choice to
  make (Xet gives dedup'd, delta-friendly storage — plausibly attractive for
  a pipeline that would otherwise re-upload a near-duplicate multi-GB
  geoparquet on every refresh).
- **Repo topology to reconsider**: the current split — one repo with raw
  JSON/Mongo dumps + 3 flat parquet tables (`govgis_nov2023`), a second repo
  with a geoparquet + embeddings geoparquet + legacy FAISS blob
  (`govgis_nov2023-slim-spatial`) — mixes "raw crawl output," "relational
  metadata," "search-ready flattened text/geometry," and "a specific vector
  index format" all as flat files in two repos with no manifest tying
  versions together. The Space's own `docs/modernization-plan.md` (already
  read) plans exactly this kind of manifest (schema version, record count,
  embedding model + revision, vector dims, distance metric, source revision) —
  that plan is consistent with what's genuinely missing from the current Hub
  artifacts today; there is no such manifest published anywhere for either
  existing dataset repo.
- **Format-compatibility gotcha to carry forward**: ship a plain
  `.parquet` (WKB/WKT geometry as string, or split lat/lon/bbox columns)
  alongside any `.geoparquet` if Hub Dataset Viewer / auto-conversion /
  `datasets`-library loading matters for the rebuild — `.geoparquet` support
  is still an open, stalled upstream issue (`huggingface/datasets#6438`,
  filed by this same user in 2023, no resolution as of last activity
  2024-02-07).
- **DOIs already exist** (`10.57967/hf/1368`, `10.57967/hf/1369`) — the Hub
  auto-assigned these on first publish; a rebuild publishing new repos would
  get new DOIs, so any citation trail from the old ones would need an
  explicit "superseded by" note in the new dataset cards.
