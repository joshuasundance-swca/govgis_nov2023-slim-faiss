# Rebuild plan — geospatial-native rethink

Draft, 2026-07-20. Lens: **geospatial-native rethink**. This is one of several
parallel drafts of a from-scratch rebuild plan for the underlying `govgis`
**dataset pipeline** — the data engineering, not this repo's in-progress
Gradio UI migration (that effort is separate and untouched here). It is
grounded entirely in `docs/ecosystem-recon/` (four cited recon files read in
full) and borrows the *methodology* — staged gates, explicit test oracles,
adversarial review before implementation — from `docs/modernization-plan.md`,
not its architecture. Per the maintainer: no prior architectural decision
(single-repo/single-Space/FAISS, or the sibling's Postgres/pgvector/MCP) is a
default to preserve; each is an option to be argued from first principles.

## Thesis: the product is a spatial catalog, not a metadata search box

The current shape is *semantic search over stringified metadata*: a 4.28 GB
LangChain-serialized FAISS blob plus BAAI/bge-large-en-v1.5 embeddings, queried
by cosine similarity over a `metadata_text` field, returning a ranked list of
GIS layers. Every recon file confirms this. This draft argues, from first
principles, that this is the **wrong primary shape** for what the data actually
is — and that the right shape is a **spatially-and-administratively queryable
catalog** in which real spatial predicates and structured jurisdiction filters
are first-class, full-text is a peer, and dense-vector similarity is a
re-ranking *facet*, not the index.

Three grounded observations drive that conclusion:

1. **There is no true feature geometry anywhere, and there should not be.** The
   recon confirms `geometry` in `govgis_nov2023-slim-spatial` is each layer's
   *bounding-box extent reprojected to EPSG:4326* — a rectangle, not the
   layer's features (`huggingface-findings.md` §3). This is not a defect to fix
   by materializing feature geometry: doing so would be terabytes, would
   duplicate live servers, and would walk straight into the state/local
   copyright ambiguity the recon documents. It *is*, however, a genuine spatial
   signal the current retrieval mechanism cannot use at all. FAISS-over-a-vector
   cannot answer "which layers' extents intersect this county?" — the single
   most valuable question a government-GIS-server catalog exists to answer.

2. **81% of `description` values are empty** (702,078 of 865,304;
   `modernization-plan.md` Stage 0 results). A dense-vector index over a field
   that is blank four times out of five has very little text to embed. The
   reliable signal in this corpus is *structured*: layer `name`, `type`,
   `geometryType`, field names, the parent service, and — critically — the
   jurisdiction of the server. Structured and full-text retrieval extract that
   signal directly; a single embedding of a mostly-empty `metadata_text`
   blurs it.

3. **The seed list already carries the spatial-administrative key, and the
   pipeline threw it away.** Elfelt's `surf_gis` list is columnar —
   `Type, State, County, Town, FIPS, Server-owner, ArcGIS-url, ...`
   (`github-findings.md`, `restgdf_api/mappingsupport.py`). Those
   jurisdiction/FIPS columns never reached the dataset schema; the slim variant
   is 7 columns (`id, name, type, description, url, metadata_text, geometry`)
   with no jurisdiction, no FIPS, no agency. Recovering them turns "layers
   covering Travis County, TX" from a fuzzy bbox guess into a precise FIPS join
   plus an extent refinement.

Taken together: the highest-value queries are *place + jurisdiction + topic*
("polygon layers covering this county, about flood zones, from a county or
state agency"). That is a hybrid of spatial predicate, structured filter,
full-text, and — optionally — semantic re-ranking. It is not what a FAISS blob
does. So this draft keeps semantic search as a real but *secondary* capability
and rebuilds the product around a spatial-native catalog. It is fine to have
concluded the opposite; the argument above is why I did not.

The rest of this document works through all nine required dimensions from that
stance.

## 1. Seed source and licensing

### Re-establishing ingestion

The exact feed the pipeline used — the `list-federal-state-county-city-GIS-servers.csv`
endpoint proxied by `restgdf_api/mappingsupport.py` — returns HTTP 404,
confirmed by two independent live fetches and an unanswered upstream issue
(`restgdf_api#74`, open since 2025-12-15). Elfelt's underlying project is
alive and growing: the recon fetched the **`.txt` mirror** directly and it
returned current content ("7,500+ ArcGIS server addresses for the USA", last
updated June 18, 2026, refreshed "each Wednesday"; `upstream-source-findings.md`
§1). The `.pdf` variant exists but **scraping the PDF is explicitly prohibited**.

Recommended ingestion path, in priority order:

1. **Parse the `.txt` mirror as the machine-readable primary.** It is published,
   it works today, it carries the same columnar schema as the dead CSV, and it
   is not the prohibited PDF. Build a *tolerant* parser: `restgdf_api#31`
   already documented ragged rows ("Expected 15 fields... saw 25"), so a
   fixed-width `read_csv(names=...)` is known to break. Parse defensively
   (quote/delimiter-aware, per-row field-count validation, quarantine
   malformed rows with a report rather than aborting or silently dropping).
2. **Contact Elfelt directly** — both to confirm terms for a re-published
   derivative (see below) and to ask whether a stable machine-readable feed or
   the CSV can be restored. Issue #74 is his own project's unanswered
   request; a direct email is the honest move and costs one message.
3. **Do not scrape the PDF.** Prohibited in the source's own terms; the `.txt`
   makes it unnecessary.

The seed step must **preserve the full columnar schema**, not just the
`ArcGIS-url` column the original regex extracted. `Type, State, County, Town,
FIPS, Server-owner` are load-bearing for everything spatial-administrative
downstream (dimensions 3 and 5). This is the single cheapest high-leverage
change in the whole rebuild.

### Honoring the licensing terms *in the design*, not just in a note

Elfelt's stated terms (`upstream-source-findings.md` §1, quoted verbatim there):
PDF scraping prohibited; commercial products prohibited without written
permission; free derivative works explicitly permitted; sharing welcomed. The
downstream government servers are a separate, more nuanced question: federal is
public domain (17 U.S.C. §105), but state/local is **not uniformly** so —
directly conflicting court rulings exist (NY/SC permit state copyright over GIS
data; FL/CA hold the opposite). Design decisions that make these terms
operative:

- **Two-tier licensing on the published artifact.** Separate the license of
  *the server-URL list* (Elfelt-derived — inherits his terms, carries a
  machine-readable `NOTICE` that commercial use requires his written permission)
  from *our crawled metadata contribution* (which can be MIT/CC0 as our own
  work). Do not stamp one blanket permissive license across the whole dataset
  as the current repos implicitly do. This is a field/column-level provenance
  concern, so it belongs in the manifest and the dataset card, not just the
  LICENSE file.
- **Capture each service's own license, per record.** ArcGIS REST service and
  layer metadata expose `copyrightText`, `licenseInfo`, and `accessInformation`
  fields. The rebuild must *harvest and store these as first-class columns*.
  This converts "state/local copyright is ambiguous" from a blanket legal worry
  into per-record evidence: a layer that declares CC0 or "public domain" in its
  own `licenseInfo` is safe to treat as such; a silent one is flagged for the
  conservative default below. Nothing in the current 7-column or 205-column
  schema captures this, even though ArcGIS returns it.
- **Index-and-link, do not bulk-redistribute feature data.** Keep the catalog
  at *metadata + bbox extent + a link to the live service*. Do not materialize
  feature geometry into the published dataset. This is the geospatial-native
  choice (below) *and* the licensing-safe choice *and* the cost-sane choice —
  a rare three-way convergence. Redistributing a rectangle-extent + metadata +
  link is defensible; redistributing scraped county parcel geometry at scale is
  the case the FL/NY split is actually about.
- **Politeness and ToS hygiene in the crawler:** honor `robots.txt`, rate-limit
  per host, identify the crawler's user-agent, and record per-server any
  explicit terms encountered. These are cheap and they are what "good citizen
  of someone else's infrastructure" looks like in code.

## 2. Crawl / ETL architecture

### Tooling: consume `restgdf`, do not reinvent — but build the missing pipeline

`restgdf` v3.0.0 (May 2026) is the most actively maintained repo in the whole
ecosystem: async, typed (Pydantic v2 `LayerMetadata`/`CrawlReport`), MIT,
PyPI-published, with `Directory.crawl`/`safe_crawl`, an optional
`restgdf[resilience]` extra (stamina + aiolimiter) and `restgdf[telemetry]`
(OpenTelemetry), and an open PR #175 adding a spatial-filter geometry payload
helper. The crawling *client* is not the bottleneck. The bottleneck, confirmed
across all four recon files, is that **the actual build pipeline never existed
as committed, tested code** — it lived only in un-versioned notebooks inside the
HF dataset repo (`scrape_2_11142023.ipynb`, `to_parquet.ipynb`,
`to_mongo_4326reproj.py`). The rebuild's real work is to turn that into a
first-class, tested, CI'd `govgis-pipeline` codebase built on `restgdf`.

Do **not** resurrect `restgdf_api` as the seed proxy: it is 2+ years stale, has
no CI, and its whole `mappingsupport.py` router depends on the dead CSV. Replace
it with a small typed seed-ingestion module that reads the `.txt` mirror
directly (dimension 1). Reconsider, but do not assume, the official **ArcGIS API
for Python** as an alternative crawl client: it is heavier and
org-administration-oriented; for "crawl thousands of arbitrary public REST
roots," `restgdf`'s narrow async design is the better fit, and the recon's
comparable-tools survey (restapi is synchronous/GPL; esri2gpd is
smaller/synchronous) confirms no better async/MIT option exists.

### Concurrency, retry, rate-limiting

The original run used `asyncio.Semaphore(10)` + `tenacity` (3 attempts,
exponential backoff) and took 2h16m for 2,038 roots; the un-throttled pass was
faster (24m38s) but was explicitly the "failed" run — i.e. politeness was
already learned to matter. For 7,500+ servers:

- **Per-host rate limiting**, not a single global semaphore. Many government
  servers share infrastructure; the unit of politeness is the hostname. Use
  `aiolimiter` per host (via `restgdf[resilience]`).
- **Adaptive backoff and per-host circuit breaking:** after N consecutive
  failures or a 429/503 from a host, quarantine it for the run rather than
  hammering it. Record the reason.
- **Bounded global concurrency** on top of per-host limits, tuned to the
  runner's network, with a hard wall-clock budget per run (crawls must be
  resumable — see automation).

### Validation and failure handling

- **Typed models end to end** (Pydantic v2, reusing `restgdf`'s where
  possible). Validate CRS presence, extent sanity, and geometry validity at
  ingest — do not defer to a downstream reprojection step that silently drops
  rows (the current pipeline lost 560 layers to reprojection failure with no
  record of which or why; `huggingface-findings.md` §2/§6).
- **A `CrawlReport`/manifest per run**: servers attempted / reachable / failed
  (with error class), services and layers discovered, per-host timing. The
  current `servers.parquet` already has an `error` column — elevate that to a
  structured, queried, gated output.
- **Fail loud, quarantine, never silently drop.** Every dropped record gets a
  reason code retained in the snapshot (see geometry handling, dimension 3).

### Incremental vs full re-crawl

The one-shot 2h16m burst is the wrong model for a maintained catalog. Design
**incremental by default**:

- **Cheap change-detection poll:** ArcGIS server roots expose `currentVersion`.
  Poll it (and a content hash of the service directory) to detect changed/new
  servers without a full crawl.
- **Targeted re-crawl** of only new-in-seed-list or changed servers on the
  frequent cadence.
- **Full re-crawl** on a slow cadence (quarterly/semi-annual) or on demand, to
  catch silent changes and re-baseline.

This makes steady-state crawl cost a function of *drift*, not of catalog size —
essential as the seed list quadruples toward 7,500+.

## 3. Data model and schema

### Keep the tiering concept; redesign every tier

The current split — raw JSON + mongodump + three all-`str` parquet tables
(`govgis_nov2023`), and a separate 7-column geoparquet + embeddings + FAISS blob
(`govgis_nov2023-slim-spatial`) — mixes four different concerns (raw crawl,
relational metadata, search-ready view, a specific index format) across two
repos with no manifest tying versions together. Keep the *idea* of tiers;
rebuild each:

- **Raw tier (provenance/replay):** per-server JSON as it comes off `restgdf`,
  stored as compressed JSONL or Parquet (not a monolithic tarball). This is the
  replay/audit source; never queried directly.
- **Rich tier (typed, queryable):** replace the 205-column *all-`str`*
  `layers.parquet` with a **typed schema**. This is a real, cited defect fix:
  `to_parquet.ipynb` casts every column — including nested `fields`,
  `drawingInfo`, `capabilities`, `extent`, `domains`, `types` — to `str`,
  destroying queryability. Store nested structures as Parquet nested/struct
  columns or typed JSON, not stringified dicts. Preserve the full schema
  fidelity that makes the *full* dataset (not the slim one) the right rebuild
  starting point (`huggingface-findings.md` §2).
- **Catalog tier (search/discovery-ready):** a GeoParquet with a real geometry
  column plus the recovered jurisdiction/FIPS attributes, captured license
  fields, a compact typed field summary, `geometryType`, the live-service link,
  `metadata_text` for embedding, and an optional embedding vector reference.
  This is the tier the serving layer (dimension 5) is built on.

### Geometry: keep the bbox extent, but be honest and lossless about it

Should "geometry = reprojected bbox extent" change? **No — but its handling
must.** First-principles: materializing real feature geometry is out of scope
(cost, licensing, live-server duplication). The bbox extent is the honest,
affordable spatial signal. The changes are about honesty and losslessness:

- **Name it truthfully.** Carry a STAC-style `bbox` (minx, miny, maxx, maxy) and
  an `extent` polygon, documented explicitly as *the layer's declared extent,
  not its features*. Do not let a column called `geometry` imply feature-level
  data.
- **Store native CRS and EPSG:4326.** Keep the server's declared
  `spatialReference` alongside the reprojected 4326 extent, so nothing is lost
  and reprojection is auditable.
- **Never silently drop reprojection failures.** The 560 rows lost in the
  current build (865,864 → 865,304) vanished with no record. In the rebuild, a
  layer whose extent fails reprojection or is degenerate keeps its row with
  `geometry_valid = false` and its raw native-CRS extent retained. Losslessness
  is a hard requirement for successive-snapshot diffing (dimension 4).
- **Capture `geometryType`** (`esriGeometryPoint`/`Polyline`/`Polygon`/…) as an
  attribute. It tells a user *what kind* of features a layer holds — high-value
  for discovery ("polygon layers covering X") — at zero geometry-storage cost.

The honest limitation to state plainly: bbox extents are **coarse**. A statewide
layer's extent intersects every county query. That is exactly why the
jurisdiction/FIPS attributes (dimension 1) are not optional — the precise answer
to "layers covering this county" is *FIPS join first, extent-intersection to
refine*, not extent alone. The two together are precise; either alone
over-matches.

### Deterministic IDs — a real, cited defect

`to_parquet.ipynb` derives IDs from **Python's built-in `hash()`**, which is
salted per process and **not stable across runs** (`huggingface-findings.md`
§6). This makes successive-snapshot diffing impossible: the "same" layer gets a
different ID next crawl. Fix: adopt a **deterministic natural-key hash** —
`blake2b`/`sha256` of the normalized `(server_url, service_path, layer_id)`
tuple, namespaced (UUIDv5-style). Notably, the *Mongo* pipelines in the same
dataset already used a deterministic scheme (`random.seed(url); uuid.UUID(...)`)
— so the better strategy already existed in-house and simply was not used for
the parquet artifacts. Deterministic IDs are the prerequisite for everything in
dimension 4; they are not a nice-to-have.

### Typed over stringified, throughout

State the rule once and enforce it: no `astype(str)` firehose. Every field has a
type; nested structures stay nested; numbers stay numbers; the schema is
versioned in the manifest. This is what makes the rich tier queryable and the
catalog tier trustworthy.

## 4. Storage and versioning

### Repo topology: purpose-specific, not the current entangled two

Recommend a small set of purpose-specific repos, replacing the current
two-repo split whose boundaries do not match how the data changes:

- **`govgis-pipeline`** (GitHub code repo): the crawler/ETL/build code that
  never existed on GitHub. `restgdf`-based, typed, tested, CI'd, cron'd.
- **`govgis-catalog`** (HF dataset repo): the canonical versioned catalog —
  GeoParquet (real geometry) + a plain `.parquet` sibling + STAC JSON +
  manifest. Metadata only; changes at crawl cadence.
- **`govgis-embeddings`** (HF dataset repo): vectors + index artifacts kept
  *separate* from metadata, so a re-embed (new model) does not churn the
  metadata repo and a metadata refresh does not force a full re-upload of 5 GB
  of vectors. This directly fixes the current 9.46 GB monolith that is
  re-uploaded whole.

`restgdf` stays an upstream *dependency*, not a fork target. The sibling
`govgis_nov2023-slim-spatial-server` becomes the serving home (dimension 5),
extended, not duplicated.

### A genuine successive-snapshot strategy (nothing like this exists today)

Both current dataset repos are single-burst, single-branch, zero-tag, frozen
since Nov 2023 — there is no precedent anywhere in this ecosystem for
publishing successive snapshots (`huggingface-findings.md` §8). Build one:

- **Dated, immutable snapshots** (`snapshot=2026-07`) published as Hub
  tags/branches, plus a moving `latest` ref. Partition the catalog GeoParquet by
  `snapshot_date` and `state` for incremental append and query pushdown.
- **A diff artifact per snapshot** — added / removed / changed servers, services,
  and layers — made possible *only* by the deterministic IDs from dimension 3.
  This is genuinely new and independently valuable: it *is* the drift-detection
  signal (dimension 6) and a first-class product ("what changed since the last
  crawl") that no one in this space currently offers.

### Manifest design

Every snapshot ships a manifest (JSON) recording: schema version; crawl date;
seed-list source URL, fetch date, and row count; servers attempted / reachable /
failed; service and layer counts; embedding model + revision + dimensions +
distance metric; geometry CRS(s) and valid-rate; `restgdf` version; per-file
SHA-256 checksums; and a diff-vs-previous summary. This extends the
manifest the modernization plan already specifies for the search index, applied
to the whole dataset — and it is the thing the recon explicitly notes is missing
everywhere today.

### LFS vs Xet, and Dataset-Viewer compatibility

- **Choose Xet.** The current repos are plain Git LFS (no Xet;
  `huggingface-findings.md` §8). A successive-snapshot pipeline re-uploads
  near-duplicate multi-GB files every refresh; Xet's content-defined chunking
  dedups across snapshots, which is exactly this workload. Separating embeddings
  into their own repo compounds the saving.
- **Ship a plain `.parquet` sibling** with geometry as a WKB/WKT string column
  (or split `minx/miny/maxx/maxy` + centroid lat/lon). The Hub Dataset Viewer
  still does not support `.geoparquet` — `huggingface/datasets#6438`, filed by
  this same maintainer in 2023, remains OPEN (last activity 2024-02-07). Keeping
  the `.geoparquet` as the spatial-native artifact *and* a plain-parquet sibling
  is the only way to have both working previews and real geometry until #6438
  lands.

### DOI and citation / superseding the old repos

The old repos carry auto-registered DOIs (`10.57967/hf/1368`, `1369`). New repos
get new DOIs. Handle the transition explicitly: add "supersedes / superseded by"
cross-references in both old and new dataset cards; **freeze, do not delete** the
old repos (they are cited externally — e.g. `Lexicom7/EO_Datasets` references
`govgis_nov2023` by name — and DOIs must resolve permanently); add a deprecation
banner pointing at the new catalog; and make citations snapshot-specific
(cite a dated tag + its DOI, not a moving `latest`).

## 5. Serving / access architecture

### First principles: one canonical store, several consumer-specific views

The current anti-pattern is *one shape (a FAISS blob) serving every need*. A
government-GIS catalog has genuinely different consumers — a human doing spatial
discovery, an agent calling a tool, a data engineer bulk-downloading, a GIS
analyst wanting interop — and they want different query surfaces over the *same*
data. Build the canonical store once; project views.

**Is FAISS-in-a-Space still right?** As the *primary* index, no. FAISS cannot
pre-filter by geometry or by exact attribute (jurisdiction, type, license) before
ranking — which are precisely the catalog's highest-value operations. A
vector-only index is a special case of the hybrid store below, minus its most
important capabilities. Keep a lightweight vector option only for the zero-infra
static demo (below), not as the retrieval spine.

**Canonical store: extend the sibling Postgres + PostGIS + pgvector server.** It
already exists, is Dockerized, actively developed, and exposes `gis_layer_search`
via FastMCP (`local-findings.md`; `github-findings.md`). Extend rather than start
fresh — but upgrade it: it is still LangChain-based
(`langchain_community.embeddings`), carries a ~9-month backlog of unmerged
dependency/security PRs, and today exposes only `/search` + one MCP tool.
PostGIS + pgvector + Postgres full-text in one engine is exactly the hybrid
substrate the thesis calls for.

**Views over that store:**

- **Hybrid search API** — the core. Query pipeline: spatial predicate (PostGIS
  `ST_Intersects` on the bbox extent) → structured filter (FIPS/jurisdiction,
  `type`, `geometryType`, license) → full-text (Postgres `tsvector`/BM25 over
  name + description + field names + parent service) → **optional** pgvector
  re-rank of the survivors. Vector similarity refines an already-relevant,
  already-in-region candidate set instead of being the first and only filter.
- **STAC API** (`stac-fastapi` + `pgstac`) for spatial + temporal + CQL2
  attribute discovery and ecosystem interop. A govgis catalog maps cleanly onto
  STAC: server → Catalog, service → Collection, layer → Item (bbox + properties
  + a link to the live REST endpoint as the "asset"). Red-teaming this honestly:
  STAC's assets are usually static downloadables, and a layer is a live service,
  so this is a slightly unconventional STAC use — but the core Item shape (bbox +
  properties + links) fits, and STAC's tooling (pystac, QGIS STAC plugin, CQL2)
  gives spatial+attribute filtering essentially for free. Worth adopting as the
  interop face; not worth contorting the data to fit if a specific extension
  proves painful.
- **MCP tool** (extend the sibling's `gis_layer_search`): add spatial and
  jurisdiction parameters so an agent can ask "polygon layers covering FIPS
  48453 about flood zones." This is the agent-native view and the sibling
  already proves the pattern.
- **Bulk / zero-infra view: GeoParquet + DuckDB.** The versioned catalog
  GeoParquet is directly queryable with **DuckDB spatial** (plus its FTS and
  `vss` vector extensions) with no server at all — and, importantly,
  **DuckDB-WASM in the browser** can run spatial + full-text + vector queries
  over the catalog client-side. That is a credible, near-zero-cost replacement
  for the public search Space: no 5 GB FAISS load, no always-on backend. This is
  the geospatial-native, modern-tooling reframe the lens is meant to surface.
- **On-demand feature fetch** (not materialized): when a user actually wants a
  layer's real features, fetch them live via `restgdf` /
  `geospatial-data-converter` (which already does single-layer ArcGIS-URL →
  GeoDataFrame) and hand back GeoParquet. The catalog stays light; real geometry
  is fetched from the authoritative live source on demand — respecting both cost
  and the licensing stance.

**Is a hybrid needed? Yes.** The honest architecture is two serving tiers over
one data model: a rich Postgres/PostGIS/pgvector API (+ STAC + MCP) for heavy,
agent, and interop use, and a static GeoParquet + DuckDB-WASM path for the cheap
public demo and bulk users. The separately-in-progress Gradio Space's eventual
target is to be a *thin client* over one of these, not a monolith that loads a
multi-GB index — but that migration is out of scope here and untouched.

## 6. Refresh cadence and automation

Nothing like this exists anywhere in the ecosystem today (confirmed in every
recon file). Build it as a first-class deliverable.

- **Seed refresh (weekly):** Elfelt posts on Wednesdays. Poll the `.txt` mirror
  weekly, diff against the last seed snapshot, and queue new/removed servers.
- **Change-detection poll (weekly):** hit each server root's `currentVersion` /
  directory hash (cheap) and queue only changed servers for re-crawl.
- **Incremental metadata crawl (weekly/biweekly):** crawl only the queued
  changed/new servers.
- **Full re-crawl (quarterly / on demand):** re-baseline and catch silent
  changes.

**Automation shape.** GitHub Actions provides the *trigger* and the gates, but a
full crawl of 7,500 servers will exceed GitHub-hosted runners' 6-hour wall-clock
limit (the original 2,038-server crawl was 2h16m; ~8h+ scaled up is a real
risk). Design: a cron-scheduled GHA workflow that *dispatches* the crawl to a
longer-lived executor — an HF Job, a self-hosted runner, or a small VM spun up
per run — then, on completion, validates → builds artifacts → publishes the
snapshot + manifest + diff to the Hub → opens a PR (or auto-commits behind the
gates below). Borrow the release engineering from `geospatial-data-converter`,
the ecosystem's most mature CI/CD (pre-commit + pytest + build + `twine check` +
Docker smoke + GitHub→HF mirror).

**Data-quality gates on every refresh (fail closed).** Before a snapshot is
published, it must pass — mirroring the modernization plan's staged-gate method,
applied to data:

- row-count sanity (within a declared band of the previous snapshot, or the
  deviation is explained and acknowledged);
- schema conformance (typed schema unchanged, or the schema version is bumped
  deliberately);
- geometry validity rate ≥ threshold; reprojection-failure rate ≤ threshold;
- deterministic-ID uniqueness *and* stability (no collisions; unchanged layers
  keep their IDs across snapshots — directly testable given dimension 3);
- embedding count == catalog record count;
- manifest checksums verified;
- a **golden-query oracle** check (dimension 7): retrieval/spatial recall on a
  fixed query set does not regress vs the previous snapshot.

**Drift detection.** The per-snapshot diff artifact *is* the drift signal. Alert
on anomalies that indicate a pipeline regression rather than real-world change —
e.g. a sudden >X% collapse in reachable servers usually means the seed feed or
the crawler broke, not that government GIS vanished. Fail closed and open an
issue; do not publish a corrupt snapshot over a good one.

## 7. Staged implementation plan

Reuse the modernization plan's *method* — staged gates, explicit oracles,
adversarial review before implementation, rollback thinking — not its
architecture. Each phase has an advance/no-advance gate; nothing external or
irreversible happens without an explicit human go/no-go (publishing repos,
running a full public crawl).

**Phase 0 — baseline, oracle, and legal footing.**
Recover seed ingestion (prove the `.txt` mirror parses tolerantly; email Elfelt
re: terms and a stable feed). Build the **test oracle**: a fixed set of
*topic + place + jurisdiction* queries with expected layer IDs, grounded against
the **existing** Nov 2023 data — which is available locally (the sibling repo
holds `govgis_nov2023_slim_spatial_embs.geoparquet`) — so the oracle is built on
the real data shape, not a fixture (per the fable-posture real-data rule). Record
parity thresholds (semantic Recall@k; spatial precision/recall for bbox+FIPS
queries). Record the licensing decision (two-tier license, per-record license
capture, index-don't-redistribute).
*Gate:* oracle exists and runs against real data; seed ingestion works and is
tolerant; licensing stance recorded.

**Phase 1 — pipeline foundation (`govgis-pipeline`).**
`restgdf`-based crawler + typed Pydantic models + tolerant seed parser +
deterministic IDs + per-host rate limiting + `CrawlReport`. CI borrowed from
`geospatial-data-converter`.
*Gate:* a small sample (~20 servers) crawls end to end with typed output;
deterministic IDs are byte-identical across two independent runs; tests, ruff,
mypy green.

**Phase 2 — canonical data model and storage.**
Typed rich schema + catalog GeoParquet (real geometry, native+4326 CRS,
`geometry_valid` flag, `geometryType`, recovered FIPS/jurisdiction, captured
license fields) + plain-parquet sibling + manifest + Xet repos. **Convert the
existing Nov 2023 raw JSON into the new schema as the first snapshot without
re-crawling** — this proves the schema and yields an immediate parity target.
*Gate:* new-schema snapshot reproduces Nov 2023 counts (865,864 layers /
195,479 services / 1,684 servers) within the *documented, explained* 560-row
reprojection delta — and, unlike the original, those 560 are *retained with
reason codes*, not dropped; IDs deterministic; geometry valid-rate measured;
Dataset Viewer previews the plain-parquet sibling.

**Phase 3 — serving (hybrid store).**
Extend the sibling Postgres/PostGIS/pgvector server; add full-text + spatial +
jurisdiction filters, the hybrid query pipeline, a STAC API, the upgraded MCP
tool, and the DuckDB-WASM static path; retire runtime LangChain deserialization.
*Gate:* hybrid retrieval beats vector-only on the spatial-administrative oracle
queries; semantic Recall@k ≥ the legacy FAISS index on the semantic queries
(parity, computed in-run — the same parity discipline the modernization plan's
Stage 2 uses); STAC API validates against the STAC spec; MCP tool callable with
spatial+jurisdiction params.

**Phase 4 — fresh crawl and automation.**
Re-crawl the current 7,500-server list; build the 2026 snapshot; wire cron +
change-detection + drift detection + the full data-quality gate set.
*Gate:* the 2026 snapshot passes every data-quality gate; the diff vs Nov 2023
is sane and published; one full automated cron run completes green end to end.

**Phase 5 — cutover and supersede.**
Publish the new catalog/embedding repos; add supersede + DOI cross-references;
point consumers (the eventual Gradio Space target, the MCP server) at the new
store; freeze the old repos with deprecation banners.
*Gate:* consumers work against the new store; old repos annotated and still
resolving their DOIs; observation window elapsed before anything old is retired
(nothing is deleted).

**Rollback thinking.** Every snapshot is immutable and tagged; consumers pin to a
snapshot revision. A bad snapshot → repin to the previous tag; never delete an
old artifact until the new one clears its observation window. The frozen Nov 2023
repos remain the ultimate fallback. Adversarial review (below) runs *before*
Phase 1 implementation, per the method.

**Test oracle summary.** The golden *topic + place + jurisdiction* query set with
expected layer IDs is the durable oracle, measured every snapshot: Recall@k for
semantic queries, spatial precision/recall for bbox+FIPS queries, ID-stability
checks, and geometry validity rates — all grounded against real data, never a
hand-built fixture.

## 8. Ownership and repo structure

Recommend **three new purpose-specific repos + extending the sibling server**,
rather than consolidating into one mega-repo or perpetuating the current
entangled two-dataset split:

- `govgis-pipeline` (code) — the crawler/ETL/build system that never existed.
- `govgis-catalog` (dataset) — canonical versioned metadata catalog.
- `govgis-embeddings` (dataset) — vectors/index, decoupled from metadata cadence.
- **Extend `govgis_nov2023-slim-spatial-server`** as the serving home — it
  already has the Postgres/PostGIS/pgvector/FastAPI/FastMCP stack; adopting it
  means also clearing its dependency backlog and de-LangChaining it, which is
  net-positive maintenance.

Rationale: crawl code, rich metadata, vectors, and serving each have a different
change cadence and size profile; entangling them (as today) is the maintenance
liability the recon documents. `restgdf` stays an upstream dependency. Given a
**single maintainer**, keep the repo count minimal-but-separated and let the
automation + gates (dimension 6) carry the operational load that a team would
otherwise carry — fail-closed refreshes need no babysitting while green.

## 9. Cost and ops burden

Honest tradeoffs, tiered by ambition:

- **Crawl compute:** 7,500 servers ≈ many hours; exceeds GitHub-hosted runner
  limits, so a longer-lived executor (HF Job / self-hosted runner / per-run VM)
  is required — a small, bounded, per-run cost. Incremental crawling keeps
  steady-state cost proportional to *drift*, not catalog size.
- **Storage:** raw + rich + catalog + embeddings across successive snapshots is
  multi-GB × N. Xet dedup + separating embeddings from metadata are the two
  mitigations; realistic footprint is tens of GB/year with Xet, materially more
  without.
- **Embedding compute:** re-embedding a growing corpus (865k layers today,
  scaling with a quadrupling seed list) on any model change is a GPU cost. Keep
  embeddings incremental (embed only new/changed `metadata_text`) and isolated
  in their own repo.
- **Serving:** the Postgres/PostGIS/pgvector API needs a persistent host — the
  main recurring cost. The DuckDB-WASM static path has ~zero serving cost (Hub
  file hosting only), which is the concrete reason to offer it as the cheap
  default and reserve the Postgres API for heavy/agent/interop use.
- **Ops / maintenance:** one maintainer. The automation + gates *are* the
  toil-reduction, but they are also code to maintain. The honest tradeoff is
  meaningfully more upfront engineering (a real pipeline, gates, drift detection)
  in exchange for far lower steady-state toil — versus today's
  zero-automation / zero-maintenance-but-permanently-frozen state.
- **Legal:** per-record license capture + index-don't-redistribute keeps
  exposure low without commissioning 50-state legal review; a single email to
  Elfelt is the one-time relationship cost.

**Recommended tiers, so the maintainer can pick a budget:**

- **Minimal:** static catalog GeoParquet + DuckDB-WASM search + quarterly manual
  re-crawl. Near-zero serving cost, no Postgres. Delivers the spatial + full-text
  + jurisdiction filtering the thesis is built on, without an always-on backend.
- **Standard:** + Postgres/PostGIS/pgvector hybrid API + STAC + MCP + weekly
  incremental cron. One small managed Postgres or VM.
- **Full:** + FIPS-precise spatial-admin queries, drift alerting, on-demand
  live feature fetch, and the full successive-snapshot/diff product.

Even the Minimal tier changes the product's shape from "semantic search box"
to "queryable spatial catalog" — which is the point of this lens. The higher
tiers add reach and freshness, not a different thesis.

## What I would red-team before building

Consistent with the borrowed method, the highest-risk assumptions to
adversarially verify *before* implementation: (a) that the `.txt` mirror is a
durable ingestion path and Elfelt is amenable — get it in writing; (b) that STAC
models a live-service catalog without contortion — prototype one Collection
before committing the API face to it; (c) that bbox + FIPS filtering actually
delivers the spatial precision claimed, given how coarse statewide extents are —
measure it on the real Nov 2023 data in Phase 0; (d) that DuckDB-WASM can carry a
865k-row (and growing) catalog in-browser at acceptable latency — benchmark
before betting the cheap public path on it; and (e) the per-record license
stance with counsel before any re-publication at the new scale.
