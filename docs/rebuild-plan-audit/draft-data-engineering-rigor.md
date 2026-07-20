# Rebuilding the govgis Dataset Pipeline — a data-engineering-rigor plan

Draft, 2026-07-20. Scope: a from-scratch rebuild of the **data pipeline** that
produces the `govgis` dataset — the crawl, the transforms, the published
artifacts, and their refresh — **not** the Gradio Space migration, which is a
separate, already-in-progress effort (`docs/modernization-plan.md`) and is not
touched here.

This document is grounded entirely in the recon under
`docs/ecosystem-recon/` (four cited source files plus their synthesis) and
borrows the *method* — staged implementation with explicit gates, test oracles,
and adversarial review before build — from `docs/modernization-plan.md`. It
does **not** inherit that plan's architecture (FAISS files on a Space), and per
the maintainer's explicit instruction it treats none of the existing
architectural decisions (single-repo/FAISS, or the sibling's Postgres+pgvector)
as defaults.

## Thesis

The `govgis` dataset is not a research artifact that happened to get published;
it is a **derived data product with a live upstream, real licensing
constraints, and a documented history of silent correctness defects** —
non-deterministic IDs, blanket string-casting that discards every type, a
"geometry" column that is actually a reprojected bounding box, no manifest, no
checksums, no schema, no versioning, and a build script that exists only as
uncommitted notebooks. Treated as a warehouse pipeline instead of a one-off
scrape, almost every hard problem here is a *data-contract* problem, and the
right response is to make correctness **checkable at every hop** rather than
trusted.

The rebuild therefore adopts a **medallion pipeline** (immutable Bronze raw
landing → typed, validated Silver → consumer-specific Gold), a **versioned
schema/contract registry**, **content-addressed deterministic IDs**, a
**checksummed manifest with a snapshot lineage chain**, and a **gate at every
stage boundary** (crawl → raw storage → transform → publish) such that a bad
refresh is refused *before* it is published, not discovered after. Ease of
setup and operational simplicity are explicitly subordinated to this: where a
more rigorous approach costs more infrastructure or more up-front work, this
plan takes it.

Everything below is a hypothesis until its gate passes. That is the point.

## Guiding principles (the rigor tenets this plan is accountable to)

1. **Every stage boundary is a validation gate.** No data advances to the next
   layer, and nothing is published, until a scripted, versioned set of
   assertions passes. Gates fail *closed*.
2. **Schemas are code, versioned, and enforced.** Typed Pydantic v2 record
   contracts plus dataframe/columnar schema validation, both checked in, both
   CI-tested, both stamped into every manifest. No `astype(str)`.
3. **Determinism is a tested property, not an aspiration.** IDs, ordering, and
   transform outputs are reproducible; re-running the same input produces
   byte-identical typed output. The current `hash()`-based IDs are a cited
   defect this plan removes on day one.
4. **Everything shipped is checksummed and manifested.** Record counts, vector
   counts, file digests, source revisions, and the tool/model versions that
   produced them travel *with* the data. A consumer can verify what it loaded.
5. **Provenance and licensing are columns, not footnotes.** The legal nuance
   (federal PD vs. unverified state/local; Elfelt's terms) is encoded as typed
   fields that *gate what may be redistributed*, so the pipeline enforces the
   license posture instead of merely documenting it.
6. **The pipeline is the artifact.** The single most important thing missing
   from this ecosystem today is a committed, tested, runnable build. That is
   the primary deliverable; the datasets are its output.
7. **Snapshots are first-class and chained.** Successive snapshots
   (`2026Q3`, `2026Q4`, …) each reference their predecessor by manifest hash,
   giving a verifiable lineage no repo in this ecosystem has today.

---

## 1. Seed source and licensing

### 1.1 The seed problem, precisely

The pipeline's original seed feed —
`mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.csv` —
returns HTTP 404 (confirmed twice by live fetch on 2026-07-20; open, unanswered
`restgdf_api` issue #74 reports the same since 2025-12-15). Two facts from recon
change the design:

- **The list is alive and larger than ever.** The `.txt` mirror at the same
  path returns successfully: "7,500+ ArcGIS server addresses for the USA," last
  updated "June 18, 2026," refreshed "each Wednesday," each entry containing
  `rest/services` in the address. This is ~4–5× the ~1,684 servers in the
  Nov 2023 snapshot (which was itself a filtered subset of ~2,038 crawled
  candidate roots).
- **The format was already dirty before it disappeared.** `restgdf_api` issue
  #31 shows the CSV had row-to-row field-count drift (a `ParserError`:
  "Expected 15 fields in line 3569, saw 25"). Any ingestion must be a *tolerant,
  validated parser*, never a fixed-width `read_csv(names=...)`.

### 1.2 Ingestion design that honors Elfelt's terms *in the pipeline*

Elfelt's stated terms (quoted verbatim in
`upstream-source-findings.md` §1 from the `.txt` mirror):

> "Scrapping data from the PDF file is prohibited."
> "Commercial products based on this list are prohibited unless specific
> written permission is obtained from Joseph Elfelt…"
> "Permission is given for anyone to make a derivative work based on this list
> as long as the derivative work is available to everyone for free."

These are honored as **design constraints**, not annotations:

- **Ingest the `.txt` mirror, never the PDF.** The PDF path is explicitly
  off-limits; the `.txt` mirror is the same content in a permitted,
  machine-readable form. The seed loader targets the `.txt` (with the `.csv`
  as a fallback if Elfelt restores it) and has a hard guard that refuses to
  fetch or parse the `.pdf` URL — a unit-tested assertion, so a future
  well-meaning change cannot reintroduce PDF scraping.
- **Fetch politely, on his cadence.** One fetch per refresh cycle, cached for
  ≥7 days (matching his weekly Wednesday update), with a descriptive
  `User-Agent` identifying the project and a contact address. The seed feed is
  low-volume (one document); there is no reason to hit it hard.
- **Keep the published derivative free.** The rebuild's public outputs stay
  free-to-everyone, which is exactly the condition under which Elfelt welcomes
  derivative works. **Any commercial use is gated on written permission** —
  represented in the pipeline as a config flag `commercial_use_authorized`
  that defaults to `false` and, if ever set true, requires a checked-in
  reference to the permission grant. This makes "did we get Elfelt's sign-off?"
  a fail-closed precondition rather than a memory.
- **Record seed provenance.** Every snapshot's manifest stores the seed source
  URL, the fetch timestamp, the SHA-256 of the exact seed document ingested,
  and the parsed server count. The seed is itself a versioned, checksummed
  input.

**Recommended human action before the first production crawl:** contact Elfelt
directly (he is reachable and, per the GeoHipster interview, actively engaged),
confirm the `.txt` mirror is an acceptable machine-readable ingestion path,
and — even for the free/non-commercial case — get an explicit acknowledgment.
This is cheap insurance and consistent with his own issue-thread suggestion.

### 1.3 The downstream-server licensing nuance, encoded as a typed gate

Recon (`upstream-source-findings.md` §5) establishes that the crawled servers
themselves are **not uniformly public domain**: federal works are PD under
17 U.S.C. §105, but state/local GIS data has directly-conflicting court
rulings (NY/SC permit copyright assertions; FL/CA reject them; many states have
no clear guidance). A rigorous pipeline cannot flatten this into "it's
government data, it's public."

The rebuild encodes it as a typed, per-record **`license_status`** enum derived
during transform:

- `federal_public_domain` — server owner classified federal (from the seed
  list's `Type`/`Server-owner` fields and URL heuristics).
- `declared_open:<spdx-or-text>` — the ArcGIS service metadata itself declared
  a license. ArcGIS REST commonly exposes `licenseInfo` and `copyrightText`
  fields at the service/layer level; when present and open, capture and
  normalize them.
- `declared_restricted:<text>` — service metadata declared terms that restrict
  reuse.
- `unverified_state_local` — non-federal, no declared license found. The
  legally ambiguous default.

This column then **gates redistribution scope** at publish time:
`federal_public_domain` and `declared_open` rows may be redistributed in full
(all metadata). `unverified_state_local` and `declared_restricted` rows are, by
default, published in a **link/index-only** form (identifying fields + source
URL, so a consumer can go to the authoritative source) rather than a full-metadata
redistribution — a conservative posture that a later, deliberate legal review
can widen per-jurisdiction. The point: the pipeline's *default behavior* is the
legally-safe one, and widening it is an explicit, recorded decision. This is the
concrete answer to "honor the licensing in the design, not just note it."

---

## 2. Crawl / ETL architecture

### 2.1 Tooling: `restgdf` v3, with the resilience extras made mandatory

`restgdf` is the most actively maintained repo in the ecosystem (v3.0.0,
May 2026; typed Pydantic v2 models — `LayerMetadata`, `CrawlReport`; PyPI;
ReadTheDocs; green CI) and is purpose-built for exactly this bulk-server sweep
(`Directory.crawl`/`safe_crawl`). It is the correct crawl client; the original
pipeline's problems were never in the client.

The rebuild pins `restgdf[resilience,telemetry]` explicitly:

- **`[resilience]`** (stamina + aiolimiter) gives principled retry and rate
  limiting instead of the ad-hoc `asyncio.Semaphore(10)` + `tenacity` in the
  original notebook. Concurrency, per-host rate caps, and backoff become
  configured, recorded parameters.
- **`[telemetry]`** (OpenTelemetry) gives per-crawl spans/metrics so success
  rates, latencies, and error classes are observable and can feed drift gates,
  not reconstructed from filenames after the fact.

`restgdf`'s own `CrawlReport` typed model is adopted as the **crawl-stage
contract**: every crawl emits a typed report (attempted / succeeded / failed /
per-error-class counts) that the crawl gate asserts against.

The official Esri **ArcGIS API for Python** was evaluated (per recon §3) and
**rejected as the primary crawler**: it is heavier, org/administration-oriented,
and adds an Esri-SDK dependency for a job that is fundamentally "hit thousands
of arbitrary public REST roots and read JSON." It is kept in mind only as a
fallback for the rare server that speaks a dialect `restgdf` mishandles. The
GPL-licensed `restapi` and the lighter `esri2gpd` are noted as live
alternatives but offer no advantage over a maintained, async, MIT, in-house
client the maintainer already controls.

### 2.2 Concurrency, retry, rate-limiting, politeness

- **Bounded global concurrency** and **per-host rate limits** (a single agency
  server must not be hammered; the seed spans thousands of independent hosts, so
  global concurrency can be high while per-host stays low — e.g. global 20–50
  in-flight, per-host ≤2 concurrent, ≤N req/sec). Exact numbers are recorded in
  the run config and stamped into the crawl report; they start conservative and
  are tuned from Stage-measured data, not guessed once.
- **Retry only transient failures** (timeouts, 5xx, connection resets) with
  exponential backoff and a hard attempt cap; **never retry** deterministic 4xx
  (404/403 = the server or service is gone/forbidden — that is a *finding*, not
  a transient error, and it is recorded as such).
- **Per-request timeout budget** and an overall crawl wall-clock budget. Recon
  measured 2h16m for 2,038 roots with a semaphore of 10; 7,500 roots at similar
  per-host politeness is plausibly ~6–10h — which drives the automation-host
  decision in §6/§9 (GitHub Actions' ~6h job cap is a real constraint).
- **A descriptive `User-Agent`** identifying the project and a contact, so
  server operators can reach the maintainer rather than block an anonymous
  scraper.

### 2.3 Failure handling and what "failure" means

The original pipeline deleted any JSON lacking a `metadata` key and kept the
survivors (1,684 of 2,038). That silently conflates "server unreachable,"
"server returned garbage," and "server returned valid-but-empty" — and loses
the record of *why* a server dropped out. The rebuild instead **records every
outcome**:

- Every seed server gets a row in a **`crawl_outcomes`** table for the snapshot:
  `server_url`, `http_status`, `outcome` (`ok` / `unreachable` / `forbidden` /
  `invalid_response` / `empty` / `timeout`), `attempts`, `latency_ms`,
  `response_sha256`, `error_class`. Nothing is deleted; exclusion from the
  typed layer tables is a *derived* consequence of a recorded outcome, and the
  count reconciliation (§2.5) proves it.
- This makes **drift detection possible across snapshots**: "servers that were
  `ok` last quarter and are `unreachable` this quarter" is a queryable,
  gate-able signal, not an invisible gap.

### 2.4 Incremental vs. full re-crawl

Both are supported and used for different cadences (see §6):

- **Full snapshot crawl** — every server in the current seed list is crawled;
  produces a complete, self-contained snapshot. This is the quarterly/periodic
  authoritative build.
- **Incremental refresh** — the weekly seed delta (new/removed servers, cheap to
  compute from the checksummed seed) plus a rotating staleness sample (a
  deterministic slice of previously-`ok` servers, e.g. by `server_id` modulo, so
  every server is re-checked within a bounded window). Incremental output is
  *staged*, never auto-published; it feeds drift detection and decides whether a
  full snapshot is warranted early.

Incremental correctness depends entirely on deterministic IDs (§3.3): you can
only diff two crawls if "the same server/service/layer" has the same identity
across runs. The original `hash()` IDs made this impossible; the rebuild's
content-addressed IDs make it the default.

### 2.5 The crawl → raw-storage gate

Before any crawl output is accepted into the Bronze layer:

- The **`CrawlReport`** validates (typed, complete) and the **success rate is
  within bounds** — both an absolute floor (e.g. ≥X% of seed servers reachable)
  and a **delta bound vs. the prior snapshot** (a sudden 20-point drop in
  reachability means the crawler or the network is broken, not that a fifth of
  US government GIS went offline in a quarter — fail closed and investigate).
- **Count reconciliation**: `seed_server_count == sum(crawl_outcomes by
  outcome)`. Every seed server is accounted for by exactly one outcome. No
  silent drops.
- Every landed raw response has a recorded **`response_sha256`**; partial/truncated
  responses (content-length mismatch, JSON parse failure) are classified
  `invalid_response`, not silently written.

---

## 3. Data model and schema

### 3.1 Medallion layers, explicit contracts at each

- **Bronze (raw, immutable).** The exact JSON returned per server, plus HTTP
  metadata, content-addressed and append-only. **Never transformed, never
  overwritten.** This is the audit trail: any Silver/Gold row can be traced back
  to the exact bytes it came from. Preserving this is the single biggest schema
  improvement over the original pipeline, which kept raw JSON only as a
  `jsonfiles.tar.gz` blob with no index and no per-file checksum.
- **Silver (typed, normalized, validated).** Three relational tables —
  `servers`, `services`, `layers` — with **explicit typed schemas** and
  referential integrity, produced by parsing Bronze through Pydantic v2 models.
  This is where the original pipeline's worst defect is fixed: no `astype(str)`.
- **Gold (consumer-specific, published).** Derived views built *from validated
  Silver*: a bulk-download typed parquet set, a Dataset-Viewer-compatible plain
  parquet, a spatial geoparquet, and an embeddings/index artifact — each with
  its own manifest and each provably a function of the same Silver revision.

### 3.2 What to keep, drop, and change from the current split

Recon documents the current shapes: the full `govgis_nov2023` (205-column raw
layer table, everything stringified; plus servers/services parquet, raw JSON,
two MongoDB dumps) and the `slim-spatial` 7-column + embeddings + FAISS variant.

Decisions:

- **Keep the raw wide metadata** — but in Bronze (untyped JSON, by definition)
  and in a **typed** Silver `layers` table, not a 205-column `astype(str)`
  parquet. The recon is right that the full dataset "is genuinely the dataset a
  rebuilt pipeline should start from, since it is the only artifact that retains
  full field schemas, renderer/`drawingInfo`, capabilities, subtypes/domains."
  The rebuild preserves that richness *with types*.
- **Type the nested structures deliberately.** ArcGIS layer metadata is deeply
  nested (`fields`, `drawingInfo`, `capabilities`, `extent`, `types`,
  `relationships`, `timeInfo`). Blanket-stringifying them (the original sin)
  destroys queryability. The rebuild uses Arrow **struct/list types** for the
  nested objects that consumers actually query (notably `fields` — the field
  schema per layer is high-value), and for genuinely open-ended sub-objects that
  no consumer queries structurally, stores them as a **typed JSON string with a
  documented, versioned sub-schema** (a JSON column with a contract, not an
  accidental one). The split between "promote to struct" and "keep as
  contracted JSON" is itself recorded in the schema registry.
- **Keep a slim search view**, but as a *derived Gold view* of Silver, not a
  separate hand-built lineage. The 7 slim columns (`id`, `name`, `type`,
  `description`, `url`, `metadata_text`, `geometry`) become a `SELECT` over
  typed Silver plus a documented `metadata_text` construction rule — so the slim
  view can never drift from the wide table (the original pipeline built them
  from two independent notebook paths, which is exactly how the 560-row
  discrepancy between the full and slim datasets arose).
- **Do not bake embeddings into the primary table.** Embeddings are a separate,
  regenerable Gold artifact keyed by `layer_id`, with the embedding model +
  revision + dimension recorded in the manifest. Baking 5.27 GB of vectors into
  the geoparquet (as the sibling's `_embs.geoparquet` does) couples the data to
  one model choice and bloats every download; keeping them separate lets the
  embedding model change without re-publishing the metadata, and lets a
  metadata-only consumer skip 5 GB.

### 3.3 Deterministic, content-addressed IDs (a cited defect, fixed)

The current parquet pipeline uses Python's built-in `hash()` for IDs —
**non-deterministic across processes** (`PYTHONHASHSEED` randomization), so IDs
are not stable across runs and cannot be used to diff snapshots. (Notably the
*Mongo* pipeline already used a better scheme — `random.seed(url);
uuid.UUID(...)` — so even the original author had the right instinct in one of
two parallel paths.)

The rebuild uses **content-addressed IDs** derived from stable natural keys via
a fixed, documented, platform-independent hash (BLAKE2b, fixed digest size, hex):

- `server_id  = blake2b(normalize(server_url))`
- `service_id = blake2b(server_id + "/" + service_path)`
- `layer_id   = blake2b(service_id + "/" + str(layer_index))`

Properties, all **tested as gates**:

- **Deterministic** — same input → same ID, on any platform, in any process,
  across years. A dedicated test crawls a frozen fixture twice and asserts
  identical IDs.
- **Stable across snapshots** — a server whose URL is unchanged keeps its
  `server_id` in 2026Q4 that it had in 2026Q3, which is what makes cross-snapshot
  diffing and incremental refresh correct.
- **Collision-checked** — the transform gate asserts ID uniqueness per table
  (no two distinct natural keys share an ID) and referential integrity (every
  `layer.service_id` exists in `services`, every `service.server_id` exists in
  `servers`).
- **URL-normalization is versioned** — the `normalize()` rule (lowercasing host,
  stripping trailing slashes, canonicalizing scheme/port) is part of the schema
  registry, because changing it changes identities and must therefore bump a
  version and be a deliberate act.

### 3.4 The "geometry is a reprojected bbox" question

The current `geometry` column is a `POLYGON` that is the layer's **bounding-box
extent reprojected to EPSG:4326** — not real feature geometry. It is honestly
described in recon but *dishonestly named* in the data: a consumer sees
`geometry: POLYGON(...)` and reasonably assumes feature geometry.

The rebuild fixes the naming and the provenance, and does **not** attempt to
fetch true feature geometry (that is a different, far larger project — pulling
actual features from 865k+ layers is orders of magnitude more data and hits the
licensing redistribution question hard; out of scope):

- Rename to **`extent_bbox_4326`**, typed explicitly as a bounding box
  (`xmin, ymin, xmax, ymax` doubles) with a derived convenience polygon.
- **Preserve the native extent and native CRS** as separate typed columns
  (`extent_native`, `spatial_reference`) so the reprojection is reproducible and
  auditable — the original pipeline discarded the native extent after
  reprojecting, making the transform impossible to verify.
- **Validate reprojection** as a transform gate: coordinates within valid
  EPSG:4326 bounds, non-degenerate box (positive area, non-collinear), and
  record reprojection failures as a typed outcome (mirroring the original's
  discard-on-`GEOSException`/`CRSError`, but *counted and reconciled*, not
  silently dropped — this is the likely source of the original 560-row
  full-vs-slim gap, and here it becomes a visible, gated number).

### 3.5 Schema/contract registry

A versioned **schema registry** lives in the pipeline repo: Pydantic v2 models
for record-level contracts, and companion columnar schemas (Arrow schema files
or `pandera` DataFrameSchemas) for table-level validation. Each carries a
`schema_version`. Every manifest records the `schema_version` of every table it
published. A breaking schema change bumps the version and is a deliberate,
reviewed act. This is how a consumer six snapshots from now knows exactly what
shape it is loading — something no current artifact provides.

---

## 4. Storage and versioning

### 4.1 Repo topology (see also §8)

Separation of concerns, bounded to a manageable number of repos:

- **Bronze (raw) storage** — immutable, potentially private. Options: a private
  HF dataset repo, or object storage (S3/R2). Large, append-only, per-snapshot;
  primarily an audit/rebuild source, not a consumer surface. **Recommendation:
  a private, per-snapshot-tagged HF dataset repo** to keep everything on one
  platform with checksums and DOIs available, unless bandwidth/cost pushes it to
  object storage.
- **Gold (published) dataset** — a single public HF dataset repo,
  **`govgis` (unversioned name)**, that supersedes both existing frozen repos
  and holds successive snapshots as **tags/branches** with manifests.
- **Pipeline code** — a new repo, the missing artifact (§8).

### 4.2 A genuine successive-snapshot strategy (the ecosystem's biggest gap)

Recon is explicit: "there is no precedent anywhere in this ecosystem for
publishing successive dataset snapshots." Both existing dataset repos are
single-burst, single-branch, zero-tag. The rebuild establishes the pattern:

- **Snapshot identity**: a `snapshot_id` (`2026Q3`, or `20260815` for
  date-based), assigned once, immutable.
- **Publication as a git tag** on the Gold repo — `snapshot/2026Q3` — pointing at
  an immutable tree of that snapshot's Gold artifacts + manifest. `main` tracks
  "latest published snapshot" for convenience, but the tags are the canonical,
  citable releases.
- **Manifest lineage chain**: each snapshot's manifest records
  `prior_snapshot: {snapshot_id, manifest_sha256}`. This makes the sequence of
  snapshots a verifiable chain — you can prove 2026Q4 was built after and
  references 2026Q3, and detect a missing or tampered link. No such lineage
  exists anywhere in this ecosystem today.

### 4.3 Manifest design (concrete)

A `manifest.json` ships with every snapshot and is itself checksummed; the
manifest's own SHA-256 is the snapshot's release identity. Fields:

```
schema_version           # registry version governing this snapshot's tables
snapshot_id              # e.g. "2026Q3"
created_utc
pipeline_git_sha         # exact code that produced this
restgdf_version
embedding_model          # e.g. "BAAI/bge-large-en-v1.5"
embedding_model_revision # pinned commit
embedding_dim
distance_metric          # e.g. "cosine"
seed:
  source_url             # the .txt mirror
  fetched_utc
  sha256                 # checksum of the exact seed document ingested
  server_count
crawl:
  attempted, succeeded, failed, success_rate
  crawl_report_sha256
counts:
  servers, services, layers            # per-table row counts
license:
  elfelt_terms_ref
  commercial_use_authorized            # false unless a grant is referenced
  status_distribution                  # counts per license_status enum
files:
  - path, bytes, sha256, rows, schema_ref
prior_snapshot:
  snapshot_id, manifest_sha256         # the lineage chain link
validation:
  gate_results { crawl, raw, transform, publish }   # each pass/fail + evidence
  all_passed: true                     # publish is impossible if false
```

A consumer (this repo's Space, the sibling server, a bulk downloader) loads the
manifest first, verifies file checksums and counts before trusting a byte, and
fails closed on mismatch. This is the direct replacement for the current
"deserialize a 4.28 GB pickle-like blob and hope" posture.

### 4.4 LFS vs. Xet

Recon confirms both existing repos use plain Git LFS, no Xet, only because they
predate Xet. A snapshotting pipeline **re-uploads near-duplicate multi-GB files
every refresh** — precisely the case Xet's content-defined chunking/dedup is
built for. **Recommendation: enable Xet on the Gold repo** so quarter-over-quarter
snapshots (where most layers are unchanged) store deltas rather than full copies.
This is a real cost lever at 5–10 GB/snapshot × 4+ snapshots/year and is a
deliberate day-one choice, not a default inherited from the old repos.

### 4.5 Dataset-Viewer compatibility (the `.geoparquet` gap)

`.geoparquet` is not recognized by the HF Dataset Viewer or the parquet-conversion
bot (upstream `huggingface/datasets#6438`, filed by this maintainer in 2023, still
open, last activity 2024-02-07). The current slim-spatial repo has a broken
preview and lost its `format:parquet`/`library:*` tags as a result.

The rebuild **always ships a plain `.parquet` sibling** for every geospatial
artifact — geometry encoded as **WKB (or WKT) in a typed string/binary column**,
or as split `xmin/ymin/xmax/ymax` columns — so the Dataset Viewer renders and
`datasets`/`polars`/`duckdb` load it natively. The `.geoparquet` remains for
GeoPandas/PostGIS consumers. This is a publish-gate requirement, not a
nice-to-have: a snapshot without a viewer-compatible plain-parquet sibling does
not pass publish.

### 4.6 DOI / citation and superseding the old repos

The existing repos carry auto-assigned DOIs (`10.57967/hf/1368`,
`10.57967/hf/1369`). New repos get new DOIs. The rebuild:

- Adds a **"superseded by"** note and link to the new dataset in both old repos'
  cards (they stay published for citation stability — never deleted).
- Adds a **"supersedes"** note (with the old DOIs) in the new dataset card.
- Registers a DOI per canonical snapshot tag where the Hub supports it, so a
  paper can cite a *specific, immutable, checksummed* snapshot rather than "the
  dataset, whatever it says today."

---

## 5. Serving / access architecture

### 5.1 Principle: one validated Gold, many consumer-specific views

The current ecosystem has one shape (FAISS-in-a-Space) serving one need, and a
sibling that independently loads a geoparquet into Postgres. The rigor lens says:
**the source of truth is the typed, manifested Gold layer; serving surfaces are
thin, independently-swappable views built from it**, each shaped for its
consumer, each verifying the manifest before serving. This directly answers the
task's question — yes, prefer multiple consumer-specific views over one shape for
everything.

Three views, each justified:

- **Bulk-download dataset (HF Gold repo).** The typed parquet + geoparquet +
  plain-parquet-sibling + manifest. Serves data scientists, the sibling server's
  loader, and reproducibility. This is the primary, lowest-cost, highest-value
  surface and it exists whether or not any live service runs.
- **Search API + MCP tool (extend the sibling `-spatial-server`).** The sibling
  already implements Postgres+PostGIS+pgvector+FastAPI+FastMCP — real prior art
  for "GIS metadata as an agent-callable search tool," and better data
  engineering than a FAISS file. The rebuild **extends it to consume the new
  manifested Gold** (loading typed parquet with checksum verification instead of
  an opaque `_embs.geoparquet`), rather than starting a new serving stack. One
  note: the sibling currently loads embeddings via LangChain
  (`HuggingFaceBgeEmbeddings`); the rebuild should keep embeddings as a manifested
  artifact and let the server load them directly, decoupling from LangChain as
  this Space's own Stage 2 already plans to.
- **Semantic-search Space (this repo, post-Gradio-migration).** Becomes a thin
  consumer of a **published, manifest-validated native FAISS index** (built as a
  Gold artifact, checksummed, no pickle) — exactly the safe artifact layout the
  modernization plan already specifies. The Space stops being the place the index
  is mysteriously produced; it consumes a versioned Gold artifact like any other
  consumer.

### 5.2 Is FAISS-in-a-Space still right?

For a free, public, single-node semantic-search demo: yes, as *one* view — but
only when it consumes a **native, checksummed, non-pickle** index produced and
validated by the pipeline (the modernization plan's target), not the legacy
serialized blob. It is the wrong choice for the *authoritative* serving path
(no filtering, no spatial queries, single-node memory ceiling), which is why the
pgvector view exists alongside it. Neither is "the" architecture; both are views
of Gold.

### 5.3 The embedding model as a versioned, swappable input

Embeddings are a Gold artifact keyed by `layer_id`, with model + revision +
dimension in the manifest. This lets Stage-7-style experiments (smaller models,
quantization) happen without touching the metadata pipeline, and lets a consumer
that wants a different embedding model regenerate just that artifact from Silver.
`BAAI/bge-large-en-v1.5` is the current, well-understood choice (used by both
existing consumers); the rebuild keeps it as the default but does not hard-wire
it into the schema.

---

## 6. Refresh cadence and automation

### 6.1 Cadence, sized to the real cost

Elfelt refreshes weekly (Wednesdays); a full crawl of 7,500 servers is a
multi-hour job (extrapolated from the measured 2h16m for 2,038). Weekly full
re-crawls are neither necessary (the underlying GIS servers change slowly) nor
cheap. The rebuild splits the cadence:

- **Weekly seed-delta check (cheap, automated).** Fetch the `.txt` mirror,
  checksum it, diff the parsed server list against the last snapshot's seed.
  Emit a *seed-drift report* (new servers, removed servers, count change). This
  is minutes of work and needs no crawl. A large delta (e.g. a spike of new
  servers) can trigger an early full snapshot.
- **Quarterly full snapshot (automated, gated, human-approved to publish).**
  Full crawl → Bronze → Silver → Gold → all gates → **staged**. Publication of
  the snapshot tag is a human go/no-go (per standing autonomy policy: crawling
  and building are reversible and automatable; *publishing a public, DOI'd
  artifact* is an outward-facing act that gets an explicit sign-off).
- **On-demand incremental refresh (automated, staged only).** Re-crawl the seed
  delta + a rotating staleness sample; feed drift detection; never auto-publish.

### 6.2 Concrete automation design

- **Scheduler**: the weekly seed-delta job fits GitHub Actions `schedule:` cron
  comfortably (minutes, well under the job cap). The **full quarterly crawl does
  not** — a ~6–10h crawl exceeds GitHub Actions' ~6h job limit and is a poor fit
  for a shared runner. It runs on a **dedicated runner**: a self-hosted GitHub
  Actions runner, an HF Job, or a cheap ephemeral VM, triggered on a quarterly
  `schedule:` (or manual dispatch) and reporting back. This is the reason the
  automation is split, not a convenience.
- **`geospatial-data-converter` is the CI/release reference.** Recon flags it as
  the most mature release engineering in the ecosystem (pre-commit + pytest +
  wheel/sdist + `twine check` + Docker smoke tests + documented dry-run release +
  GitHub→HF mirroring). The pipeline repo borrows that shape for its own CI and
  for the publish workflow.
- **Idempotent, resumable stages.** Each stage checkpoints (Bronze is
  content-addressed and append-only, so a re-run skips already-landed servers;
  Silver/Gold are deterministic functions of their inputs). A crawl that dies at
  hour 7 resumes rather than restarting — essential at this scale.

### 6.3 Drift detection and data-quality gates on every refresh (new to this ecosystem)

Nothing like this exists today. Every refresh — incremental or full — runs a
**drift panel** comparing the new candidate snapshot against the prior published
one, computed by DuckDB/pandera over the typed tables:

- **Volume drift**: row counts per table within a configured band vs. prior
  (a 40% collapse in layers is a broken crawl, not a real-world event → fail).
- **Reachability drift**: crawl success-rate delta bounded (§2.5).
- **Null-rate / schema drift**: per-column null-rate change bounded;
  `description` empty-rate is a known 81% baseline (recon confirmed 702,078 of
  865,304 rows) — a jump to 99% empty means a parse regression.
- **Referential integrity**: 100% (every layer→service→server link resolves).
- **Determinism**: ID uniqueness and stable-ID overlap with prior snapshot
  within expectation (most `server_id`s should persist; a near-total ID churn
  means the normalization or hashing changed unexpectedly).
- **Geometry validity**: reprojected-bbox validity rate bounded.
- **License distribution**: `license_status` mix within band (a sudden swing in
  `federal_public_domain` share signals a seed-classification regression).

A refresh that trips any drift gate is **quarantined for human review**, not
published. This is the mechanism that catches a bad refresh *before it ships* —
the explicit ask of this lens.

---

## 7. Staged implementation plan

The method mirrors `docs/modernization-plan.md`'s staged-gates-with-oracle
pattern (proven on this same data, today). Each stage has actions, a **gate**
(scripted assertions, fail-closed), rollback thinking, and the **test oracle**
that stage is accountable to. Nothing advances on a green build report alone —
the coordinator re-runs the actual gate commands.

### The test oracle (defined once, used by every stage)

- **Golden fixtures**: a small set of **real, checked-in** ArcGIS server JSON
  responses (captured from live servers spanning the tricky cases — a healthy
  FeatureServer, a MapServer, a server with `licenseInfo`, one with a
  reprojection-failing extent, one with HTML in `description`, an empty one).
  These are the ground truth the transform is tested against — and per the
  fixture-honesty lesson, they are captured from the *real producer* (live
  `restgdf` output), never hand-authored, so a green test cannot pass on a
  fictional shape.
- **Determinism oracle**: transform the golden fixtures twice; assert
  byte-identical typed output and identical IDs.
- **Retrieval oracle**: reuse the Stage-0 query-set methodology already built in
  this repo (`docs/stage0/query_set.json`, 23 grounded queries, Recall@k parity
  procedure) as the search-quality oracle for any Gold index artifact.
- **Manifest/consumer oracle**: a consumer stub loads a published snapshot,
  verifies every checksum and count against the manifest, and fails closed on a
  deliberately corrupted artifact.

### Stage 0 — foundations, contracts, and the seed path

Actions: stand up the pipeline repo (uv, ruff, mypy strict-leaning, pytest,
pre-commit, CI — borrowing `geospatial-data-converter`'s shape); write the
schema registry (Pydantic v2 + columnar schemas) for `servers`/`services`/
`layers`/`crawl_outcomes`/`manifest`; implement and **unit-test the seed loader
against the `.txt` mirror with the hard no-PDF guard**; capture the golden
fixtures from live servers; contact Elfelt (§1.2).

Gate: schema registry validates round-trip on golden fixtures; seed loader
parses a real `.txt` fetch and refuses the PDF URL (tested); ID functions pass
the determinism oracle on fixtures; CI green (pytest/ruff/mypy/pre-commit);
fixtures checked in and documented.

Rollback: nothing published; pure code.

### Stage 1 — crawl to Bronze

Actions: implement the `restgdf[resilience,telemetry]` crawl with recorded
concurrency/rate/timeout config; write `crawl_outcomes` for every seed server;
land raw JSON content-addressed with per-file checksums; emit and validate the
`CrawlReport`. First run against a **bounded subset** (e.g. one state) to
validate the machinery before a full 7,500-server sweep.

Gate (crawl → raw): `CrawlReport` validates; success rate within absolute +
delta bounds; `seed_server_count == Σ crawl_outcomes`; every landed file
checksummed; no `invalid_response` written as if valid.

Rollback: Bronze is append-only and immutable; a bad crawl is a discarded
staging batch, never an overwrite of a prior snapshot.

### Stage 2 — Bronze → Silver transform

Actions: parse Bronze through Pydantic models into typed `servers`/`services`/
`layers`; assign content-addressed IDs; promote high-value nested fields to Arrow
structs, contract the rest as versioned JSON; reproject extents with native
extent + CRS preserved; classify `license_status`.

Gate (transform): 100% record schema-conformance (every row validates);
ID uniqueness + referential integrity 100%; determinism oracle passes;
reprojection validity within bounds with failures counted and reconciled;
count reconciliation Bronze→Silver (every included/excluded row explained);
null-rate/volume drift vs. prior snapshot within band.

Rollback: Silver is a deterministic function of Bronze; re-run, don't repair.

### Stage 3 — Silver → Gold artifacts

Actions: build the typed bulk parquet, the geoparquet, the **plain-parquet
Dataset-Viewer sibling** (WKB/split-bbox), the slim search view (as a derived
`SELECT`, not a parallel build), and the embeddings + native FAISS index (Gold);
assemble the `manifest.json` with checksums, counts, model/revision, lineage
link, license distribution.

Gate (publish-readiness): manifest complete and self-consistent (file checksums
and counts match the actual files); plain-parquet sibling present and
Dataset-Viewer-loadable; slim view row-count == its source projection (no
independent-build drift); index vector count == metadata count; retrieval oracle
Recall@k ≥ prior snapshot (or ≥ documented floor for the first snapshot);
manifest/consumer oracle passes on the real artifacts and fails closed on a
corrupted copy.

Rollback: Gold is regenerable from Silver; nothing is published yet.

### Stage 4 — publish (human go/no-go)

Actions: run the full **drift panel** (§6.3) vs. the prior published snapshot;
on all-green, publish the Gold repo snapshot **tag** with the manifest; enable
Xet; add supersede/superseded-by notes and DOIs; verify the published snapshot
by re-downloading and re-verifying the manifest end-to-end from a clean
environment.

Gate: drift panel all-green (or quarantined with explicit human override
recorded); published snapshot re-verifies from scratch (checksums, counts,
viewer preview renders); lineage chain link to prior snapshot resolves.

Rollback: prior snapshot tag remains the canonical "latest"; an unpublished or
quarantined candidate is discarded. Publishing is additive (a new tag) — it
never mutates or deletes a prior snapshot, so "rollback" is simply "don't move
`main` to the new tag."

### Stage 5 — serving views + automation

Actions: point the sibling `-spatial-server` and this repo's Space at the new
manifested Gold (both verify the manifest on load); wire the weekly seed-delta
cron and the quarterly full-crawl job on a dedicated runner; wire the drift panel
into every refresh.

Gate: both consumers load the new Gold and pass their own smoke tests against a
verified manifest; the weekly cron produces a seed-drift report on schedule;
a deliberately-injected bad refresh is caught and quarantined by the drift panel
(the automation is proven to *fail correctly*, not just to run).

Rollback: consumers can pin to the prior snapshot tag; automation can be disabled
without affecting published data.

### Stage 6 — footprint / quality evaluation (deferred, optional)

Only after a full snapshot ships with parity: evaluate smaller embedding models,
quantization, IVF/HNSW, compact metadata — each measured against the retrieval
oracle and the manifest's recorded quality floor, shipped only on a gate pass.
(Mirrors the modernization plan's Stage 7; reuses the same oracle.)

---

## 8. Ownership and repo structure

Recommendation: **three repos, clean separation, supersede rather than delete.**

1. **`govgis-pipeline` (new, the primary deliverable).** The crawl + transform +
   publish code, schema registry, golden fixtures, gates, drift panel, CI, and
   automation. This is the artifact the entire ecosystem is missing — recon
   confirms the original build "was not found in any GitHub repo… it exists only
   as notebooks inside the Hugging Face dataset repo itself, never committed."
   Making the build a first-class, tested, versioned repo is the single highest-
   leverage structural change.
2. **`govgis` Gold dataset (new HF dataset repo).** Versioned successive
   snapshots (tags), manifests, Xet, plain-parquet siblings. Supersedes the two
   frozen 2023 repos (which stay up, with cross-links, for citation stability).
   Bronze raw lands in a private companion (a private HF dataset repo or object
   store).
3. **Extend `govgis_nov2023-slim-spatial-server` (existing) for serving.** Do not
   start a new serving stack; the sibling's Postgres+pgvector+FastAPI+FastMCP is
   real, working, better-engineered prior art. Point it at the new Gold, decouple
   its embedding load from LangChain, and let its MCP tool surface the refreshed
   data.

Why not consolidate everything into the sibling server repo? Because the crawl/
transform/publish pipeline and the serving layer have different cadences, blast
radii, and dependency sets (a heavy async-crawl + geospatial-transform toolchain
vs. a Dockerized DB service). Coupling them makes the pipeline's CI hostage to
the server's, and vice versa. Separation is the rigor-favoring choice; the
bounded count (three, not a sprawl) keeps ops manageable.

`restgdf` and `restgdf_api` stay as-is: `restgdf` is a healthy dependency
(pin it); `restgdf_api`'s dead `mappingsupport.py` CSV proxy is **not** revived —
the rebuild's seed loader targets the `.txt` mirror directly and does not
reintroduce the 2-years-stale proxy service (issue #74/#31). If a hosted seed
proxy is ever wanted, it is a new, tested component in `govgis-pipeline`, not a
resurrection of stale code.

---

## 9. Cost and ops burden (honest tradeoffs)

This plan deliberately costs more than the status quo. The tradeoffs, named:

- **Crawl compute.** The expensive line item. ~6–10h per full snapshot on a
  dedicated runner, ~4×/year. A self-hosted runner or ephemeral cloud VM is
  low-dollar (hours of a small instance quarterly) but is real setup and
  maintenance the current zero-automation state does not have. The weekly
  seed-delta job is effectively free (GitHub Actions minutes). **Verdict: worth
  it — automation + drift detection is the entire point; a manual re-crawl is how
  you get another 2.5-year-stale snapshot.**
- **Storage.** ~5–10 GB per snapshot × 4+/year. **Xet dedup is the mitigation**
  (most layers are unchanged quarter-to-quarter, so deltas are small). Bronze
  raw adds storage but is the audit trail that makes the whole thing verifiable;
  it can live in cheaper/private storage and be lifecycle-expired after N
  snapshots if cost bites. **Verdict: Xet makes this affordable; Bronze is
  non-negotiable for rigor but its retention is tunable.**
- **Embedding compute.** Re-embedding at the 7,500-server scale (likely
  millions of layers, up from 865k) is a GPU job per snapshot. Mitigated by
  keeping embeddings a *separate, regenerable* artifact (only re-embed changed
  layers on incremental; full re-embed only on model change or full snapshot) —
  content-addressed IDs make "which layers changed" a cheap diff. **Verdict:
  incremental embedding keeps this bounded; the separation of embeddings from
  metadata is what makes it possible.**
- **Serving.** The pgvector view needs a hosted Postgres (real monthly cost);
  the FAISS Space and the bulk-download dataset are free-tier. Offering all three
  is more surface to maintain than one FAISS file. **Verdict: the bulk-download
  Gold + manifest is the cheap, always-on backbone; the live services are
  optional views the maintainer can run or pause independently without losing the
  data product.**
- **Maintenance surface.** Three repos, a schema registry, a gate suite, and a
  drift panel are more to maintain than two notebooks. But the notebooks produced
  a non-deterministic, unversioned, unchecked artifact that has been frozen and
  unverifiable for 2.5 years. **The rigor is the deliverable**; the cost buys
  correctness that is *checkable by anyone*, refreshes that *can't silently ship
  broken*, and snapshots that *can be cited and diffed*. For a dataset that other
  projects already reference by name (recon found `Lexicom7/EO_Datasets` citing
  it), that is the right trade.

---

## Validation-gate summary — how a bad refresh is caught before it ships

The end-to-end guarantee, one gate per hop, all fail-closed, all scripted:

| Hop | Gate asserts | A bad refresh is caught because… |
|---|---|---|
| **seed → parse** | `.txt` fetched (not PDF), checksummed; tolerant parse; server count sane | a mangled/empty seed or a format change fails the parse + count band |
| **crawl → Bronze** | `CrawlReport` valid; success-rate absolute + delta bounds; every seed server has exactly one recorded outcome; every file checksummed | a broken crawler or network shows as a reachability-delta breach, not a silent 40% row drop |
| **Bronze → Silver** | 100% schema conformance; ID uniqueness + referential integrity; determinism oracle; reprojection validity; volume/null drift bands | a parse regression, an ID-scheme change, or a type break fails conformance/determinism before it propagates |
| **Silver → Gold** | manifest self-consistency; plain-parquet viewer sibling present; slim view == its projection; vector count == metadata count; retrieval Recall@k parity | a mismatched index, a missing sibling, or a quality regression fails publish-readiness |
| **Gold → publish** | full drift panel vs. prior snapshot; re-download + re-verify from clean env; lineage link resolves | any cross-snapshot anomaly quarantines the candidate for human review — publish is impossible while `all_passed: false` |

No hop trusts the previous one; each re-checks. This is the difference between a
warehouse pipeline and a scrape.

---

## Open decisions for the maintainer

1. **Contact Elfelt** to confirm the `.txt` mirror as an acceptable
   machine-readable ingestion path and acknowledge the free-derivative use
   (recommended before the first full crawl; required before any commercial use).
2. **Bronze storage location** — private HF dataset repo vs. object storage
   (cost/bandwidth vs. one-platform simplicity). Recommendation: private HF
   dataset repo unless bandwidth pushes otherwise.
3. **Snapshot cadence** — quarterly full is the recommended default; confirm, or
   pick monthly (cost) / semi-annual (staleness) per how fast the underlying
   servers actually drift (the weekly seed-delta report will inform this after a
   cycle or two).
4. **Full-crawl runner** — self-hosted GitHub runner vs. HF Job vs. ephemeral VM
   for the ~6–10h quarterly crawl.
5. **Redistribution scope for `unverified_state_local` rows** — the plan defaults
   to link/index-only; a per-jurisdiction legal review could widen this. Confirm
   the conservative default is acceptable for the first snapshot.
6. **Embedding model** — keep `BAAI/bge-large-en-v1.5` (default) or evaluate a
   smaller/current model in Stage 6; the architecture makes this swappable
   either way.

These are validation-time decisions, not blockers to executing Stages 0–2.
