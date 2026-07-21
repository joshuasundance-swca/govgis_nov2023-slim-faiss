# Rebuilding the govgis dataset pipeline — an agentic-serving-first plan

Status: draft for review. Planning only — no code, no repo creation, no deployment.
Recorded 2026-07-20. Grounded in `docs/ecosystem-recon/` (the four cited recon files),
`docs/modernization-plan.md` (borrowed as a **method**, not an architecture), and a direct
read of the sibling `govgis_nov2023-slim-spatial-server` source
(`backend/app.py`, `models.py`, `mcp.py`, `load_data.py`).

## Core thesis

The govgis dataset is not, at bottom, a file to download or a search box to type into. It is a
**machine-readable index of where US government GIS lives** — 7,500+ ArcGIS REST server roots,
their services, and their hundreds of thousands of layers — and the highest-value consumer of an
index like that is an *agent* that can ask "what county-level parcel layers cover Travis County,
Texas, and give me the live endpoint I can pull features from." Everything downstream — a human
search UI, a bulk parquet download, a REST API — is a projection of that same agent-shaped need.

So this plan inverts the historical build order. The current ecosystem crawled first, flattened to
seven stringly-typed columns second, and bolted search on last; the agent surface (the sibling's
lone `gis_layer_search` MCP tool) was an afterthought grafted onto a schema that had already thrown
away everything an agent needs to filter on. This plan designs the **agent tool surface first**,
derives the **schema** from what those tools must filter and chain on, and treats the crawl, the
storage, and the UI as suppliers to that contract. The organizing architectural principle is a hard
split between **the data** (one versioned, immutable, snapshot-tagged artifact set on the Hugging
Face Hub) and **the service** (one query core that loads a *pinned* snapshot and presents it through
four coordinated faces: MCP, REST, a thin human UI, and bulk download). No consumer keeps its own
copy of the data ever again.

This is a deliberate, justified supersession of the two prior designs — the FAISS-file-on-a-Space
pattern and the always-on Postgres+pgvector sibling — not a dismissal of them. The sibling's stack
demonstrably works and its Pydantic-typed, parameterized query builder is genuinely good prior art I
build on directly. What I change, and why, is argued per dimension below.

---

## Dimension 1 — Seed source and licensing

### The seed feed, and unblocking it

The pipeline's original ingestion path is dead: the CSV at
`mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.csv` returns HTTP 404
(confirmed by two independent live fetches this session and by the unanswered `restgdf_api` issue
#74 open since 2025-12-15). But Elfelt's underlying list is alive, growing, and actively maintained:
the `.txt` mirror at the same path fetched cleanly this session, reporting **"7,500+ ArcGIS server
addresses for the USA," last updated June 18, 2026, refreshed most Wednesdays.** That is a 4–5×
larger seed than the ~1,684 servers behind `govgis_nov2023`.

**Decision: ingest the `.txt` mirror as the primary seed, with a manual-CSV and direct-contact
fallback, and never scrape the PDF.** The `.txt` mirror is a stable, machine-readable, currently-live
endpoint. The ingestion step downloads it once per refresh cycle (not a high-frequency scrape),
parses it with a **tolerant** parser — `restgdf_api` issue #31 documents that even the CSV had
inconsistent field counts row-to-row, so a fixed-width `read_csv(names=...)` is out; use a
line-oriented parser that regexes `.../rest/services` roots and preserves the jurisdiction columns
(Type / State / County / Town / FIPS / Server-owner) that the CSV historically carried. Each
downloaded seed is itself archived as a dated **seed snapshot artifact** with its source URL, fetch
timestamp, and row count, so the crawl is always reproducible from a recorded input, not a
now-mutated live page.

Because the `.txt` path could 404 the way the `.csv` did, the pipeline treats the seed fetch as a
**gated, fail-closed** step: if the expected format or a plausible row count (say, within ±30% of the
last snapshot) is not returned, the refresh halts and alerts rather than crawling a truncated or
malformed list. A documented secondary path — email Elfelt directly per issue #74's own suggestion,
or manually download the current file — covers a hard outage. This is not merely noted; it is the
first executable gate in the pipeline (Stage 1 below).

### Honoring Elfelt's terms in the design, not just the README

Elfelt's stated terms, quoted verbatim from the `.txt` mirror in the recon:

> "Scrapping data from the PDF file is prohibited."
> "Commercial products based on this list are prohibited unless specific written permission is
> obtained from Joseph Elfelt authorizing that commercial use."
> "Permission is given for anyone to make a derivative work based on this list as long as the
> derivative work is available to everyone for free."

These translate into concrete design constraints, not disclaimers:

1. **Never touch the PDF programmatically.** The ingestion module has exactly one Elfelt endpoint
   coded into it — the `.txt` mirror — and a comment plus a test asserting the PDF URL appears
   nowhere in the codebase. Prohibited-surface avoidance is enforced, not trusted.
2. **Keep the derivative work free and open, forever.** The rebuilt dataset ships under an open
   license (MIT or CC-BY-4.0 for the data card), with **no gated, paid, or login-walled tier**. This
   is what makes the rebuild squarely inside the "derivative work available to everyone for free"
   permission. The dataset card and the `manifest.json` both carry an explicit Elfelt attribution and
   a link to mappingsupport.com as the seed provenance.
3. **Gate commercial use behind written permission — as a pipeline fact, not a footnote.** SWCA is a
   commercial consultancy; any internal-productization or client-facing commercial use of this data
   trips Elfelt's commercial clause. The plan's Stage 0 makes **obtaining (or explicitly declining
   and staying non-commercial) Elfelt's written permission a launch gate**, recorded in the repo. The
   default posture is free/non-commercial, which needs no permission; a commercial pivot is a
   separate, gated decision with a paper trail.

### The state/local public-domain nuance, made a schema field

The recon establishes that federal GIS data is public domain (17 U.S.C. §105) but **state and local
GIS data is not uniformly so** — directly-conflicting court rulings exist (NY/SC permit state
copyright assertions over GIS data; FL/CA hold the opposite). This is the single most-cited legal
risk and it is invisible in the current 7-column schema. My design **captures license provenance
per record**: the crawl already receives each ArcGIS service's `copyrightText` and `licenseInfo`
fields (standard ArcGIS REST metadata), and those become typed columns
(`source_copyright_text`, `source_license_info`, `jurisdiction_level`). A record with no declared
license and a non-federal jurisdiction is flagged `license_status = "unverified-nonfederal"`.

Crucially, the rebuilt dataset stores **metadata plus a bounding-box extent, never real feature
geometry** (see Dimension 3) — it is an *index that points at* live government data, not a
redistribution of that data. That posture materially lowers the copyright-redistribution exposure
the recon warns about: we redistribute descriptions and a coverage box, and hand agents the live URL
to fetch the actual features under the source's own terms. The per-record license fields let a
downstream consumer (or an agent) make a defensible reuse decision instead of assuming blanket
public domain. Legal review is still warranted before any commercial redistribution scale-up; the
schema makes that review tractable instead of impossible.

---

## Dimension 2 — Crawl / ETL architecture

**Tooling: `restgdf` v3.0.0, unambiguously.** It is the most actively maintained repo in the whole
ecosystem (typed Pydantic-v2 response models, `Directory.crawl`/`safe_crawl`, a `CrawlReport` model,
an optional `restgdf[resilience]` extra bundling `stamina` + `aiolimiter` for retry and rate
limiting, and an optional telemetry extra). It is the direct descendant of `dataripper`, purpose-built
for exactly this bulk-server sweep, and pinning it removes the need to reinvent an async ArcGIS
client. The recon's survey of alternatives confirms the choice: Esri's official ArcGIS API for Python
is heavier and org-administration-oriented; `restapi` is synchronous and GPL-2.0; `esri2gpd` is
lighter but synchronous and single-layer. restgdf's async + typed + MIT profile is the best fit, and
it is the maintainer's own tool. Keep a thin raw-`httpx` escape hatch for the minority of servers
restgdf chokes on, recording those as `crawl_method = "fallback"`.

**Concurrency, retry, rate-limiting — as a good citizen of government infrastructure.** The original
build used `asyncio.Semaphore(10)` + `tenacity` (3 attempts, exponential backoff) and took 2h16m for
2,038 servers; at 7,500+ servers the naive scale-up is ~8+ hours. The rebuild adds **per-host rate
limiting** (aiolimiter, a low request-per-second cap keyed on server hostname) on top of the global
semaphore, because hammering a county's single ArcGIS box is both rude and a fast path to getting the
crawler IP-blocked. A polite, slower crawl that completes is worth more than a fast one that gets
throttled or blocked mid-run. Retries stay bounded (transient/5xx and timeouts only, never on 4xx),
and each server has a wall-clock budget so one hanging endpoint cannot stall the run.

**Validation happens at crawl time, typed.** restgdf already emits typed metadata; the pipeline
validates each server's response against Pydantic v2 models on ingest and **quarantines** — does not
silently drop — failures into a `crawl_errors` table with the exception and the raw payload. The
original build's "filter out files missing a `metadata` key" (1,684 of 2,038 survived) becomes an
explicit, counted, inspectable quarantine rather than a silent deletion, so a sudden spike in
unreachable servers is a visible signal, not invisible attrition.

**Incremental vs full re-crawl.** This is where deterministic IDs (Dimension 3) pay off. Each
server's crawl produces a content hash of its normalized service/layer metadata; the pipeline stores
the prior snapshot's per-server hashes and, on a refresh, **only fully re-processes and re-embeds
servers whose hash changed or that are newly seeded**, while still *pinging* every server to record
reachability. A periodic (e.g. quarterly) full re-crawl guards against hash-collision drift and
schema evolution. Steady-state refreshes are therefore cheap: most of the expensive work
(embedding 865K+ layers) is skipped for the unchanged majority.

**Failure handling and the run record.** Every crawl emits a `CrawlReport`-derived **run manifest**:
seed size, servers attempted / reachable / quarantined, layers discovered / validated / embedded,
restgdf version, wall-clock, and per-error detail. A run that loses more than a configured fraction
of previously-reachable servers is treated as an upstream anomaly and **does not publish** (drift
gate, Dimension 6). The crawl never mutates the published dataset directly; it produces candidate
artifacts that only a passing gate promotes.

---

## Dimension 3 — Data model and schema

### What to keep, what to fix, what to add

**Keep the tiered split, but fix its types.** The current split is sound in spirit — raw crawl output,
wide relational metadata, and a slim search view are three legitimately different products — but its
execution has two real defects the recon documents. First, the 205-column `layers.parquet` casts
**every column to `str`** (`to_parquet.ipynb` does `.astype(str)`), so nested ArcGIS structures
survive only as stringified dicts, unqueryable without re-parsing. Second, that parquet pipeline
mints row IDs with **Python's built-in `hash()`**, which is process-seed-randomized and therefore
**not stable across runs** — a genuine, cited defect that makes cross-snapshot diffing impossible.

The rebuilt artifact set, from raw to search-ready:

1. **`raw/` — archival crawl JSON**, one document per server, exactly as restgdf returns it. Never
   downloaded by normal consumers; the fidelity floor and the input to any future re-flatten.
2. **Typed relational parquet — `servers` / `services` / `layers`** with *real* types. `fields`,
   `extent`, `capabilities`, `drawingInfo`, `timeInfo` become typed nested structures or, where a
   column is genuinely free-form, a typed `JSON`/`struct` column — not blanket `str`. This is the
   analyst-facing product and the source the search view is derived from.
3. **`layers_search` — the slim, agent-ready view**: the current seven columns
   (`id, name, type, description, url, metadata_text, geometry`) **plus the enrichment facets that
   make agentic filtering possible** (below), plus embeddings in a companion file.

**Fix deterministic IDs — adopt content-addressed `uuid5`.** Replace `hash()` with a deterministic
`uuid5(NAMESPACE, canonical_key)` where `canonical_key` is the normalized
`server_url + service_path + layer_id`. This is stable across runs and machines (unlike `hash()`),
standard (unlike the seed pipeline's `random.seed(url); uuid4` trick, which works but is idiosyncratic),
and — being content-addressed on the layer's stable identity — lets the diff engine recognize the
same layer across snapshots even as its description or extent changes. Stable IDs are the linchpin of
both incremental crawl and successive-snapshot diffing; they are not a nicety.

**Keep geometry = reprojected bbox extent — do not chase real feature geometry.** This design choice,
which one might reflexively want to "upgrade," is actually correct for the product and should be kept
and documented honestly. The index answers *"which layers cover this area?"* — a question a
bounding-box extent in EPSG:4326 answers perfectly, cheaply, and with a small storage footprint.
Pulling real feature geometry for 865K+ layers would be enormous, would duplicate data that lives
authoritatively on the source servers, and would drift the moment those servers update. The index's
job is to *point at* live geometry, not mirror it. I keep the existing reprojection logic
(`to_mongo_4326reproj.py`'s pyproj-based extent reprojection with out-of-range clamping and
degenerate-extent discard — which is exactly why slim-spatial has 560 fewer rows than the full
dataset), add a stored **centroid** and a coarse **geohash** for cheap pre-filtering, and label the
column honestly in the schema and card as `extent_4326` (a coverage box), never `geometry` implying
features.

### The enrichment that unlocks agentic filtering — the most consequential schema change

The current slim schema throws away the single most useful thing the seed list carries: **who owns
the server and where its jurisdiction sits.** Elfelt's list has Type / State / County / Town / FIPS /
Server-owner columns; the crawl discards them and the search schema has no place for them. The
result is that the sibling's MCP tool can filter by layer `type` and by an intersecting point, but an
agent **cannot** ask for "state-level layers in Oregon" or "layers owned by the City of Austin,"
because those facets do not exist in the served data. That is a product-defining gap for agentic
access.

The rebuilt `layers_search` schema therefore carries typed enrichment facets, derived by **joining
the crawl output back to the seed snapshot** (matched on server URL) and by parsing the ArcGIS
service metadata:

| facet | source | why an agent needs it |
| --- | --- | --- |
| `jurisdiction_level` (federal/state/county/city/tribal/other) | seed `Type` + URL heuristics | "only county-level parcel data" |
| `state`, `county`, `place`, `fips` | seed columns | "layers covering Travis County, TX" |
| `agency` / `server_owner` | seed `Server-owner` | "layers published by USGS" |
| `service_type`, `geometry_type` | ArcGIS metadata | "point layers I can address-match against" |
| `source_copyright_text`, `source_license_info`, `license_status` | ArcGIS metadata (Dim. 1) | reuse decisions |
| `extent_4326`, `centroid`, `geohash` | reprojection | spatial filtering |
| `last_crawled`, `snapshot_id`, `source_server_url` | pipeline | provenance / citation |

Everything is a strict Pydantic v2 model shared between pipeline and service (Dimension 8's
`govgis-core` library), so the same typed contract validates on write and on read. Free-form text
fields (`description`, `metadata_text`) are stored as-is but flagged for the HTML the corpus is known
to contain (the modernization plan's Stage 0 measured 3.4% of descriptions carrying raw HTML markup);
sanitization is a **serving-time** concern (Dimension 5), not baked destructively into the stored
data, so the raw source text stays faithful for analysts while agents/humans get a safe rendering.

---

## Dimension 4 — Storage and versioning

### Repo topology

Replace the accidental current split (one repo of raw JSON + Mongo dumps + three flat parquets; a
second repo of a geoparquet + embeddings geoparquet + a legacy FAISS blob) with a **purpose-designed
small set**:

- **`govgis` (versioned dataset repo)** — the typed relational parquets, the `layers_search` view,
  the embeddings companion, a plain-`.parquet` sibling for the Viewer (below), and `manifest.json`.
  This is the source of truth every consumer pins.
- **`govgis-raw` (archival repo)** — the per-server raw crawl JSON. Large, rarely fetched, kept for
  fidelity and re-flattening. Split out so the main repo stays lean and previewable.

Embeddings live in the `govgis` repo but as a **separate file/config** from the non-embedding tables,
so the metadata tables preview in the Dataset Viewer even though the multi-GB embedding file does not.

### A genuine successive-snapshot strategy — the thing that exists nowhere in this ecosystem today

Both current dataset repos are single-burst, single-branch, zero-tag, frozen since November 2023.
There is **no precedent anywhere in this ecosystem for publishing successive snapshots**. This plan
establishes one:

- Each refresh publishes an **immutable, dated snapshot** as a git **tag** on the `govgis` repo:
  `snapshot-2026-07`, `snapshot-2026-10`, … Tags are immutable pins; a serving layer that references
  a tag can never be silently shifted under it.
- A moving **`latest`** branch points at the newest passing snapshot for consumers who want current
  data without pinning.
- Every snapshot ships a **`manifest.json`** recording: schema version, seed source + fetch date +
  seed row count, servers attempted/reachable/quarantined, layer counts (discovered/validated/embedded),
  embedding model **and revision**, vector dimensions and distance metric, restgdf version, crawl
  date range, `supersedes` (prior snapshot tag), and **SHA-256 checksums** for every artifact. This
  is exactly the manifest the modernization plan designs for the *serving* side — reused here as the
  *dataset's* contract, which is where it structurally belongs.
- A **`diff/` artifact per snapshot** — added / removed / changed layers versus the prior snapshot,
  made computable *because* IDs are deterministic (Dimension 3). This is genuinely novel for this
  ecosystem and is the substrate for both drift detection (Dimension 6) and an agent-facing
  "what changed" capability (Dimension 5). Nobody in this space ships a dataset changelog you can
  query; this design does.

### LFS vs Xet

**Choose Xet.** Both current repos use plain Git LFS (they predate Xet). A snapshot pipeline that
re-uploads a near-duplicate multi-GB embeddings parquet on every refresh is the textbook case for
Xet's content-defined chunking and dedup: successive snapshots differ only in the changed minority of
rows, so Xet stores deltas rather than full re-uploads. This is the concrete storage-cost lever that
makes a *maintained* (vs one-shot) dataset economically sane, and it directly enables the incremental
philosophy of Dimensions 2 and 6.

### Dataset Viewer / `.geoparquet` compatibility

`.geoparquet` is unsupported by the Hub Dataset Viewer — a still-open upstream issue
(`huggingface/datasets#6438`) the maintainer filed himself in 2023, no resolution since Feb 2024.
The rebuild does not wait on upstream: the **primary** served table is **plain `.parquet` with
geometry as a WKB (or WKT) string column plus split `minx/miny/maxx/maxy` float columns**, which the
Viewer renders, `datasets`/`polars`/DuckDB all load natively, and which needs no GeoParquet reader. A
`.geoparquet` sibling is kept as a convenience for GeoPandas users. Shipping plain parquet first is
also what makes the DuckDB serving path (Dimension 5) trivial — it reads the parquet directly with
zero conversion.

### DOI and citation handoff

The old repos carry auto-assigned DOIs (`10.57967/hf/1368`, `10.57967/hf/1369`). New repos get new
DOIs. Both old dataset cards get an explicit **"superseded by → `govgis`"** banner; the new card
carries a **"supersedes → …"** note and a citation block naming Elfelt's list and restgdf as
provenance. Old repos are frozen, not deleted (their DOIs and any external citations stay resolvable),
with a pointer forward. Each snapshot tag is independently citable via the manifest.

---

## Dimension 5 — Serving / access architecture (the center of this lens)

### One backend, four faces — and no consumer keeps its own copy

The defining decision: **the data and the service are separated, and every consumer surface reads one
pinned snapshot through one shared query core.** Today the Gradio Space maintains a FAISS blob and the
sibling server maintains a Postgres copy — two independent ingestions of the same November-2023 data,
each with its own drift and its own ops burden. The rebuild ends that. There is:

- **`govgis-core`** — a shared library (Pydantic schema models + embedding + retrieval logic),
  imported by both the pipeline and the service so the write contract and the read contract are
  literally the same code.
- **`govgis-service`** — one deployable that loads a pinned `govgis` snapshot and exposes it through
  **four coordinated faces over one query engine**:
  1. **MCP server (primary surface)** — the agent contract, designed first (below).
  2. **REST API** — the same query core with an OpenAPI schema, for programmatic/non-MCP clients.
  3. **A thin human UI** — the search box, as a *client* of the REST/MCP core, holding no data of its
     own. (The Gradio Space's own in-progress migration is a separate effort and is not disturbed by
     this plan; the rebuild simply offers it a REST/MCP endpoint it can later point at instead of
     shipping its own FAISS blob.)
  4. **Bulk download** — the versioned HF dataset itself, for data scientists who want the parquet.

### Query engine: recommend embedded DuckDB (VSS + spatial), keep Postgres+pgvector as the proven fallback

The sibling proves Postgres+PostGIS+pgvector works, and its parameterized query builder is good code
I would reuse. But three facts push the *recommended* default to an **embedded DuckDB** read path,
and I argue this as a real tradeoff rather than novelty-seeking:

1. **Ops burden.** The sibling runs a five-container compose (postgres + init-loader + backend + mcp
   + inspector + pgadmin) that must stay *on* to answer a query. For a single-maintainer, low-QPS,
   read-mostly index, an always-on PostGIS instance is more machine to babysit than the workload
   justifies. DuckDB is a library: the service process opens the pinned parquet and serves — nothing
   else to keep alive.
2. **The artifact is already parquet.** The sibling's `load_data.py` reads a 5.66 GB geoparquet into
   memory via GeoPandas and `COPY`s it into Postgres on init — a slow, memory-heavy step that runs
   every fresh deploy. DuckDB reads the pinned parquet snapshot **directly**, with zero load/ETL step;
   promoting a new snapshot is just re-pinning a revision.
3. **Both current stacks lack an ANN index.** The sibling's search is `ORDER BY embeddings <=> $1` —
   a **full sequential scan** over 865K vectors on every query (no HNSW/IVFFlat index exists in
   `load_data.py`). Whatever engine is chosen must add an approximate-nearest-neighbor index for
   agent-acceptable latency. DuckDB's VSS extension and pgvector both offer HNSW; on a read-only
   pinned snapshot, DuckDB's build-once HNSW is a clean fit and its weaker write/persistence story
   simply does not bite.

So: **DuckDB + VSS (HNSW) + the spatial extension, embedded, reading the pinned plain-parquet
snapshot** is the recommended serving read path — lightest ops, native artifact format, real ANN.
**Postgres+pgvector remains the documented fallback** for the day concurrency, write paths, or QPS
outgrow an embedded engine; the sibling already demonstrates that path, so escalating to it is
low-risk and mostly a redeploy, not a rewrite (the `govgis-core` retrieval interface abstracts the
engine).

### The MCP tool surface an agent actually wants — designed, not grafted

The sibling exposes **one** tool, `gis_layer_search(SemanticSearchRequest) -> SearchResponse`, that
takes a query string, an optional `type_filter`, an optional intersecting point, and skip/limit, and
markdownifies every string in the response. That is a reasonable v0 but three things make it a weak
agent surface: it cannot filter on jurisdiction/agency/geography-by-name (the facets don't exist in
its schema), it offers no drill-down or facet-enumeration for chaining, and flattening all fields to
markdown destroys the structure an agent reasons over. The rebuilt surface is a **small, composable,
chainable tool set** returning **typed structured results with provenance**, not markdown blobs:

- **`search_gis_layers(query, filters?, k?)`** — semantic search plus *structured* filters
  (`jurisdiction_level`, `state`, `county`, `agency`, `service_type`, `geometry_type`, `bbox` or
  `point`, `license_status`). Returns ranked, typed `LayerResult`s each carrying a **stable `id`**,
  the enrichment facets, the reprojected extent, and a **`citation`** (source server URL + snapshot
  id) so an agent's answer is grounded and attributable.
- **`get_layer(id)`** — full typed drill-down for a layer a prior search surfaced. This is the
  chaining primitive: search → pick → expand → act. The sibling has no such tool.
- **`list_facets(facet)`** — enumerate valid values for `state`, `county`, `agency`,
  `jurisdiction_level`, `service_type`. This exists specifically to **stop agents hallucinating
  filter values**: an agent lists the real agencies, then filters on one that provably exists, rather
  than guessing "City of Austin GIS Dept" and getting zero hits. Facet enumeration is a
  correctness feature for agents, not a convenience.
- **`get_live_endpoint(id)`** — returns the validated, `https`-only live ArcGIS REST URL (and the
  service's declared license) so the agent can go fetch **real features** from the authoritative
  source under that source's terms. This is the bridge that makes the index a launchpad into live
  government data rather than a dead-end description. It is also where the "we index, we don't
  redistribute" posture of Dimension 1 pays off: the agent gets pointed at the source.
- **`whats_changed(since_snapshot)`** — surfaces the `diff/` artifact (Dimension 4): layers added /
  removed / changed since a prior snapshot. A genuinely new capability enabled by deterministic IDs
  and successive snapshots, letting an agent answer "what new county parcel servers appeared this
  quarter."

**Output safety is a first-class part of the MCP contract, because retrieved government metadata is
untrusted content fed to an LLM.** The modernization plan already establishes this threat model for
the Space; it applies with equal force to the MCP surface. Concretely: retrieved text is delimited
and labeled as *data, not instructions*; all URLs (in records **and** in any generated text) pass an
`http`/`https` allowlist; HTML in descriptions (3.4% of the corpus) is converted to plain text at
serving time; results are returned as **typed JSON with explicit fields**, not markdown-flattened —
the agent gets structure to reason over plus an explicit `citation`, rather than the sibling's
markdownify-everything which both loses structure and would happily pass injected markup through. The
one-tool → tool-set expansion and the typed-with-provenance output are the two highest-leverage
agentic-serving decisions in this plan.

### Should the Space and the sibling merge, split, or share?

**Share a backend; the sibling grows into (or is superseded by) `govgis-service`; the Space becomes a
thin client.** The sibling is already the more advanced consumer and is the closest thing to the
target, so the pragmatic path is to **extend it into the unified `govgis-service`** (add the tool
set, the enrichment-aware schema, the ANN index, and — if adopting the recommended read path — the
DuckDB engine behind the `govgis-core` retrieval interface), and reduce the Gradio Space to a UI that
calls that service. Neither keeps its own data copy. If extending the sibling proves more friction
than a clean build, retiring it in favor of a fresh `govgis-service` is the acceptable alternative;
either way the end state is one service, one pinned data source, four faces.

---

## Dimension 6 — Refresh cadence and automation

Nothing like this exists anywhere in the ecosystem today — there is no scheduled/cron workflow on any
repo. The rebuild builds it from scratch, and the design is shaped by the single-maintainer reality:
**low-touch, fail-closed, trivially reversible.**

**Cadence and compute split.** A scheduled monthly (tunable) job orchestrated by **GitHub Actions
cron**, but with the heavy crawl+embed step delegated to a **Hugging Face Job** (the maintainer
already uses `hf jobs`) or a self-hosted runner — a crawl of 7,500+ servers taking many hours will
blow past a hosted GH runner's limits, and the embedding of changed rows wants a GPU. GH Actions owns
orchestration, gating, and publish; HF Jobs owns the compute. The pipeline is:

1. Fetch + validate the seed snapshot (fail-closed on 404/format/row-count anomaly — Dimension 1).
2. Incremental crawl with restgdf (per-host rate-limited; only changed/new servers fully processed —
   Dimension 2), emitting a run manifest.
3. Build typed artifacts + enrichment + deterministic IDs + the diff vs the prior snapshot.
4. Embed only changed rows.
5. **Run the data-quality gate suite** (below).
6. Publish a new immutable snapshot tag + manifest to HF (Xet) **only if every gate passes**; else
   quarantine the candidate artifacts and alert. The live dataset is never mutated in place.
7. **Promote** the serving layer to the new snapshot as a *separate, explicit* pin bump — data
   publish and serving promote are decoupled, so a bad snapshot never auto-reaches production.

**Drift detection** compares the new run's manifest against the prior snapshot's: a >N% drop in
reachable servers, a large unexpected swing in layer counts, an embedding-dimension or normalization
change, or a seed-size anomaly halts the publish and alerts. This is exactly the "green gate can lie"
discipline from the maintainer's own posture — a crawl that "succeeds" while quietly losing a third of
its servers must not publish.

**Data-quality gates on every refresh** (borrowing the modernization plan's gate *method* — explicit,
scripted, fail-closed — not its content):

- Row-count and reachable-server sanity vs the prior snapshot (drift bounds).
- Schema conformance: Pydantic validation pass-rate above a floor; quarantine the rest.
- Deterministic-ID uniqueness and stability (no collisions; known layers keep their IDs).
- Geometry validity: extent-reprojection success rate; degenerate extents discarded and counted.
- Embedding integrity: correct dimension (1024 for bge-large), normalized, no NaNs.
- **Retrieval regression: Recall@K on a frozen golden query set must be `>=` the prior snapshot's** —
  the same parity-gate idea the modernization plan's Stage 2 uses, applied to every refresh so a
  crawl or embedding-model change can't silently degrade search.
- Checksum manifest generated and verified.

Publish is the *only* action gated behind all of these passing; everything upstream is reproducible
from recorded inputs.

---

## Dimension 7 — Staged implementation plan

The method is borrowed directly from `docs/modernization-plan.md` — **staged, each stage with an
explicit executable Gate, a rollback, and a test oracle; adversarial review before implementation** —
but the stages are the pipeline's, not the Space's. Immutable snapshots make rollback nearly free:
"roll back" means "re-pin the serving layer to the previous snapshot tag," which is always available.

- **Stage 0 — Oracle, baseline, and legal/seed unblock.** Build the frozen **golden query set** (a
  set of GIS queries with expected relevant layer URLs, plus adversarial/malformed/empty cases —
  reuse and extend the modernization plan's `docs/stage0/query_set.json` shape). Confirm a working
  **seed ingestion path** (the top blocker — resolve the dead `.csv` by validating the `.txt` mirror
  end to end). **Obtain or explicitly decline Elfelt's written commercial permission** and record the
  posture. *Gate:* golden set exists and is scriptable; the `.txt` seed fetch+parse produces a
  plausible row count; the licensing decision is recorded. *Rollback:* n/a (no artifacts published).
- **Stage 1 — Seed ingestion + provenance.** Tolerant parser, jurisdiction columns preserved, dated
  seed-snapshot artifact with source/timestamp/count; the PDF-avoidance test. *Gate:* seed snapshot
  reproducible, fail-closed on anomaly, PDF URL absent from code. *Oracle:* parsed row count vs the
  live `.txt` header claim.
- **Stage 2 — Crawl core.** restgdf integration, per-host rate limiting, typed validation +
  quarantine, incremental hashing, run manifest. *Gate:* a bounded crawl of a sample of servers
  produces a valid run manifest with reachable/quarantined counts and no unbounded hangs. *Rollback:*
  crawl writes only candidate artifacts, never the dataset.
- **Stage 3 — Typed schema + deterministic IDs + enrichment + geometry.** The `uuid5` scheme,
  seed-join enrichment facets, license fields, `extent_4326`/centroid/geohash, typed relational
  parquet + `layers_search` view + plain-parquet Viewer sibling. *Gate:* ID uniqueness/stability
  across two runs; enrichment facets populated; schema validates. *Oracle:* known layers keep IDs
  run-to-run.
- **Stage 4 — Embeddings + search index + retrieval gate.** Embed `layers_search`, build the HNSW
  index, compute Recall@K on the golden set. *Gate:* embeddings correct dimension/normalized; Recall@K
  meets or beats the recorded baseline. *Rollback:* re-embed is idempotent from the typed tables.
- **Stage 5 — `govgis-core` + `govgis-service` (MCP + REST + engine).** The shared library, the
  DuckDB-VSS read path, the tool set, output-safety, contract tests, and **adversarial tests**
  (prompt-injection records, filter-value hallucination, malformed input, oversized fields). *Gate:*
  every tool returns typed validated output; injection/malformed fixtures fail closed; latency
  acceptable with the ANN index. *Oracle:* the golden set answered through the live MCP/REST path.
- **Stage 6 — Versioned publish + thin UI.** Publish snapshot tag + manifest + diff to HF over Xet
  with old-repo supersede banners; wire the human UI as a client. *Gate:* Viewer renders the plain
  parquet; manifest checksums verify; a pinned snapshot serves end to end. *Rollback:* old repos
  stay frozen and resolvable; serving stays on the prior pin until promote.
- **Stage 7 — Automation + drift.** The scheduled crawl→gate→publish→promote workflow, drift
  detection, and the per-refresh quality-gate suite. *Gate:* a dry-run refresh halts on an injected
  drift anomaly and publishes on a clean run; promote is a separate explicit step. *Rollback:*
  re-pin to any prior snapshot tag.

Each stage's Gate is the advance/no-advance decision; a green build report is not the gate. Stages
that mutate external state (HF publish, serving promote, and anything touching Elfelt) get an explicit
human go-ahead, per standing policy.

---

## Dimension 8 — Ownership and repo structure

Consolidate the current sprawl (this Space, the sibling server, and an unfindable build notebook that
lives only inside the HF dataset repo) into a coherent, minimal set with clear ownership:

- **`restgdf`** — unchanged; the crawl library, pinned as a dependency. Already the ecosystem's
  healthiest repo.
- **`govgis-pipeline` (new)** — the crawl → validate → build → gate → publish ETL and its scheduled
  automation. This is the artifact that **never existed on GitHub** — the original build lived only
  as uncommitted notebooks inside the HF dataset repo. Making it a real, tested, CI'd repo (borrowing
  `geospatial-data-converter`'s more-mature release engineering as the CI pattern) is itself a
  major closure of ecosystem risk.
- **`govgis-core` (new)** — shared Pydantic schema + embedding + retrieval, depended on by both the
  pipeline and the service so write and read share one contract.
- **`govgis-service` (extend the sibling, or fresh)** — the unified MCP + REST + engine deployable.
  Preferred path: grow `govgis_nov2023-slim-spatial-server` into this (it's closest to target); the
  fallback is a clean build with the sibling retired.
- **`govgis` + `govgis-raw` (HF dataset repos)** — the versioned artifacts and the archival raw crawl.
- **The Gradio Space** — reduced to a thin UI client of `govgis-service`; its own in-flight
  modernization is left alone and simply offered the endpoint to adopt later.

Recommendation on the two existing consumers: **neither keeps its own data copy going forward.** The
Space and the sibling both become clients of one versioned backend — which is the whole point of the
data/service split.

---

## Dimension 9 — Cost and ops burden

The honest framing: this is a **single-maintainer, low-QPS, read-mostly** project, and the dominant
design constraint is *sustainable operating burden*, not peak performance. Every recommendation above
is tilted toward "a bad day is a re-pin, and steady state runs itself."

- **Crawl compute** is the largest recurring cost: hours of async crawling plus a GPU embed of
  changed rows. **Incremental refresh makes steady state cheap** — most of the 865K+ layers are
  unchanged month to month, so only the changed minority is re-embedded. The initial full embed is a
  one-time GPU job (HF Jobs). Bandwidth is essentially free (public government servers), and polite
  per-host rate limiting trades a longer wall-clock for not getting blocked — the right trade for a
  job that runs unattended.
- **Storage** is multi-GB per snapshot (embeddings dominate), but **Xet dedup means the marginal cost
  of each new snapshot is roughly the changed delta, not a full re-upload** — which is exactly what
  makes a *maintained* dataset (vs a one-shot) affordable. Old snapshots are cheap immutable tags.
- **Serving** is where the biggest ops lever sits, and it is why the recommended read path is embedded
  DuckDB rather than the sibling's always-on five-container PostGIS compose. An embedded engine
  reading a pinned parquet snapshot has **no always-on database to operate, patch, back up, or pay
  for while idle**, and no load-on-init step; a new snapshot is a pin bump. The sibling's stack is the
  documented fallback for when real concurrency arrives — a deliberate, reversible escalation, not the
  default a hobby-scale index should carry.
- **Automation ops** are minimized by the fail-closed, drift-gated, promote-decoupled design: the
  scheduled job either publishes a clean snapshot or halts and alerts, and promotion to serving is a
  separate explicit action. The maintainer's recurring burden is reviewing an occasional drift alert
  and approving a promote — not firefighting a live database or a silently-degraded index.
- **The cost of *not* doing this** is the status quo: two divergent hand-built copies of a
  2.5-year-stale snapshot, a dead seed feed, no automation, and an agent surface that can't filter by
  geography. The rebuild's ongoing burden is modest and mostly automated; the current trajectory's
  burden is unbounded staleness.

The tradeoff is stated plainly rather than sold: the DuckDB-first recommendation trades the sibling's
proven, concurrency-ready Postgres stack for lower idle cost and zero-ETL snapshot promotion, on the
bet that this workload is read-mostly and low-QPS. If that bet proves wrong, the `govgis-core`
abstraction makes falling back to the sibling's pgvector path a redeploy, not a rewrite.
