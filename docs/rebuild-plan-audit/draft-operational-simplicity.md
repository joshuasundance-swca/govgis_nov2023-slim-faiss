# govgis dataset pipeline — rebuild plan (operational-simplicity lens)

Status: draft for synthesis. One of several parallel lens drafts feeding a
combined rebuild plan. This draft argues the whole problem from a single
priority: **lowest possible maintenance burden for one person working in their
spare time.** Where that priority forces a tradeoff another lens might reject,
this draft accepts the tradeoff explicitly and says why the simpler path is
still good enough.

Grounded in: `docs/ecosystem-recon/` (all five files) and
`docs/modernization-plan.md` (borrowed as a *method* — staged gates, explicit
test oracles, adversarial review before implementation — not as an architecture
to copy). This is planning only: no code, no repo creation, no deployment.

Scope boundary: the *separate*, already-in-progress Gradio migration of this
Space (`docs/modernization-plan.md`, Stages 0–7) is not touched or altered here.
This plan produces the *data* that migration will eventually consume; the two
efforts meet at exactly one seam (Dimension 5), and this draft is careful not to
reach across it.

---

## Core thesis: the dataset is the product; everything else is off by default

The single most important fact in the recon is not technical — it is that
**every repo in this ecosystem has exactly one committer, and the most
sophisticated piece of infrastructure in it (the Postgres+PostGIS+pgvector+
FastMCP sibling) is carrying nine-to-ten months of unmerged dependency and
security PRs.** That backlog is not a criticism of the sibling; it is the
clearest possible measurement of what always-on infrastructure costs a
single spare-time maintainer. Any rebuild that adds a second thing-that-must-
be-patched has already failed the only test that matters here.

So the design goal is not "the best GIS data platform." It is **a pipeline that
produces a good dataset when told to, and then turns completely off, breaking
nothing while it sits untouched for six months.** Concretely, the whole rebuild
reduces to one sentence: *a scheduled GitHub Actions workflow crawls the current
government-server list with restgdf, builds typed, deterministically-keyed
Parquet artifacts, and publishes them as a versioned, manifest-described,
Dataset-Viewer-compatible snapshot to a single Hugging Face dataset repo — with
no server, no database, and no always-on process anywhere in the design.*

Three properties fall out of that sentence and drive every decision below:

1. **Stateless and self-contained.** Each snapshot is built from scratch and
   depends on nothing a previous run left running. There is no incremental
   state to corrupt, no migration to run, no drift to accumulate. Neglect is
   survivable by construction: if the cron doesn't fire for two quarters,
   nothing degrades — the next run just produces a fresh snapshot.
2. **Managed and ephemeral over self-hosted and always-on.** GitHub Actions
   (free for public repos) for orchestration and CPU work; Hugging Face Jobs
   (pay-per-use, serverless GPU) for the one genuinely heavy step (embeddings);
   the Hugging Face Hub for storage, versioning, and the "API." Nothing to
   patch, nothing with an uptime SLA, nothing with a public network surface the
   maintainer owns.
3. **The Hub *is* the API.** The lowest-maintenance data service is a file
   somebody else hosts. A well-formed Parquet dataset on the Hub is queryable
   today with `datasets`, `polars`, and `duckdb` by anyone, with zero code the
   maintainer has to keep alive. Search and agent access are thin, optional,
   sleep-when-idle layers on top — never the primary product.

---

## Dimension 1 — Seed source and licensing

### Getting the server list, given the dead CSV

The recon settles this cleanly. The `.csv` endpoint the old pipeline proxied
(`restgdf_api/mappingsupport.py`) returns HTTP 404 and has since at least
2025-12-15 (issue #74, unanswered). But Elfelt's list is alive, weekly-refreshed,
and — critically — the **plain-text mirror still works** and was fetched
verbatim this recon:
`https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.txt`,
"7,500+ ArcGIS server addresses for the USA," last updated 2026-06-18. Every
entry contains `rest/services`, i.e. it is exactly the server-root list the
pipeline needs.

The simplicity decision is to **ingest the `.txt` mirror and nothing else.** Not
the CSV (dead), not the PDF (explicitly prohibited to scrape, and a fragile
binary format), not a bespoke re-derivation of Elfelt's list (there is no
comparable server-root-level catalog anywhere — data.gov and GeoPlatform.gov
index datasets, not server roots; building our own would be a permanent
second job). The `.txt` mirror is a published, machine-readable representation
of the same list, which sidesteps the PDF-scraping prohibition and is trivially
parseable.

Two hard-won details from the recon are baked in:

- **Tolerant parsing, not fixed-width.** Issue #31 showed the source already had
  row-to-row field-count drift *before* it went missing. The parser extracts
  unique `.../rest/services` roots with a permissive regex and dedupes, exactly
  as the original notebook did — it never assumes a fixed column count.
- **Pin the seed into the repo every run.** The dead-CSV episode is the whole
  lesson: an upstream feed can vanish silently. So each pipeline run commits the
  exact seed list it used (`seeds/YYYY-MM-DD.txt` + a checksum) into the pipeline
  repo *before* crawling. If the mirror 404s on a future run, the build fails a
  gate loudly (never silently produces an empty snapshot) and the last committed
  seed remains a usable fallback. This is one file per run — negligible storage,
  large resilience payoff, zero standing infrastructure.

### Honoring the licensing, in the design and not just the notes

The recon establishes real terms with teeth, quoted from primary sources:
scraping the PDF is prohibited; commercial products need Elfelt's written
permission; free derivative works are explicitly welcomed and sharing is
encouraged. Below that, state/local government GIS data is **not** uniformly
public domain — directly conflicting court rulings exist (NY/SC permit state
copyright assertions; FL/CA hold the opposite). The simplest design that
actually honors all of this — rather than noting it and moving on — makes five
concrete choices:

1. **Stay in the free/non-commercial lane the terms already bless.** The rebuild
   is MIT-licensed on its own code and compilation, published free to everyone,
   a "derivative work available to everyone for free" — precisely the case
   Elfelt explicitly permits. Commercial use is out of scope for this plan, so
   the written-permission requirement is a boundary we simply do not cross,
   not a blocker to route around.
2. **One documented human action: email Elfelt.** Before the first public
   snapshot, the maintainer sends a short note confirming the free public
   rebuild and confirming the `.txt` mirror is an acceptable ingestion path.
   This is a five-minute, one-time act that converts "are we allowed to do this"
   from an assumption into a recorded permission, and it costs nothing to
   maintain. It is a gate in Stage 0, not a nicety.
3. **Attribution travels with the data.** Elfelt and restgdf are credited in the
   manifest, the dataset card, and each snapshot's provenance — carrying forward
   the acknowledgment this ecosystem has used since 2020 (`dataripper`).
4. **Redistribute the light footprint we already redistribute.** We keep
   publishing *metadata plus a bounding-box extent*, never real feature
   geometry (see Dimension 3). This is a deliberate licensing simplification:
   indexing/describing services and linking back to source is a far lighter
   redistribution posture than mirroring the actual government data, which is
   where the state/local copyright ambiguity bites hardest. Every record keeps
   its live `url`, so consumers reach the authoritative source themselves.
5. **Capture each source's own license, and keep a denylist.** Where an ArcGIS
   service exposes a license/terms field in its metadata (many do), it is stored
   as a typed column rather than discarded — so a downstream consumer can filter
   on it. And a small committed `excluded_servers.txt` denylist gives a trivial,
   no-infrastructure takedown/opt-out mechanism: add a root, it is skipped on the
   next crawl. We do not attempt to adjudicate fifty states' copyright law in
   code — that is the opposite of simple; we make the source terms visible and
   provide a cheap exclusion lever, and flag "get real legal review before any
   redistribution of actual feature data or any commercial use" as a standing
   note, not a task this pipeline resolves.

---

## Dimension 2 — Crawl / ETL architecture

**Use restgdf v3.0.0 directly; write no crawler.** The recon is unambiguous that
restgdf is the most actively maintained repo in the ecosystem (v3.0.0, real CI,
PyPI, ReadTheDocs, typed Pydantic v2 models, `Directory.crawl`/`safe_crawl`,
`CrawlReport`), and that the crawl tool is *not* the bottleneck — the seed feed
and the never-automated build were. The lowest-maintenance choice is to depend
on restgdf as a normal pinned PyPI dependency and use its resilience extra
(`restgdf[resilience]` = stamina + aiolimiter) rather than re-implementing
retry/rate-limiting. The official Esri ArcGIS API for Python was considered and
rejected for this lens: it is heavier, org/administration-oriented, and a much
larger dependency surface for a job that is "grab public REST metadata at scale,"
which is exactly what restgdf was purpose-built for.

**Run it in GitHub Actions, sharded by a matrix — because of one real
constraint.** The 2023 production crawl took 2h16m for 2,038 candidate roots.
The list is now 7,500+, roughly 3.7× larger, so a single crawl plausibly exceeds
GitHub Actions' 6-hour per-job ceiling. The simple, still-serverless answer is a
**matrix of shard jobs** (e.g. partition roots into N buckets by a stable hash,
or by state), each comfortably under 6 hours and each writing its shard's raw
output as a job artifact, followed by one lightweight combine job. This keeps
everything inside free managed CI with no self-hosted runner, and it makes the
crawl embarrassingly parallel without introducing any coordinator process.

Concurrency, retry, rate-limiting, and politeness are restgdf's job
(per-shard `asyncio.Semaphore`, `stamina` backoff, `aiolimiter` per-host), mirroring
the original notebook's `Semaphore(10)` + `tenacity(stop_after_attempt(3))`
approach but with a maintained library instead of hand-rolled code.

**Failure handling treats a down server as data, not as a pipeline failure.**
Servers go offline constantly; that is expected and must never fail the run. As
the original did, each server's outcome (success, error, `currentVersion`) is
recorded in a `servers` table with an `error` column. The *pipeline* fails only
if a global threshold is breached — e.g. fewer than X% of seed roots crawled
successfully, which signals a systemic problem (network, auth, a restgdf
regression) rather than the normal long tail of dead endpoints. That threshold is
a data-quality gate (Dimension 6), not a per-server exception.

**Full snapshots, not incremental — deliberately, at least to start.** Incremental
crawling (only re-fetch servers whose `currentVersion` changed) is attractive on
paper but requires carrying per-server state across runs and trusting it to stay
consistent — the exact kind of stateful complexity this lens avoids. Because the
cadence is low (Dimension 6) and each snapshot is self-contained, a full re-crawl
per run is the simpler and more robust default: nothing to reconcile, nothing to
corrupt, trivially reproducible. Incremental crawl is named explicitly in
"What this plan deliberately does not build" as a future option to add *only if*
crawl cost ever becomes the binding constraint.

---

## Dimension 3 — Data model and schema

**Keep the two-tier split, but fix its two real defects and type only what is
queried.**

- **Raw tier: keep everything, type almost nothing.** The 205-column
  `layers.parquet` exists because a snapshot is a moment in time you cannot
  re-fetch — throwing away fields would be irreversible data loss. But fully
  typing 205 ArcGIS fields that drift server-to-server is a permanent schema-
  maintenance job. The simplicity resolution: store the raw crawl output with a
  **small set of typed top-level columns** (stable identity + join keys +
  crawl outcome) plus **one nested/JSON column preserving the complete original
  record** (Arrow struct or a JSON string). This keeps the full escape hatch with
  zero brittle wide-schema maintenance — you can always reach any field later
  without having promised a column for it today. This directly replaces the
  original's "cast all 205 columns to `str`" approach, which was both lossy in
  type and wide in maintenance.
- **Slim tier: the search-facing product, properly typed.** Keep the 7-column
  slim shape that consumers actually use (`id`, `name`, `type`, `description`,
  `url`, `metadata_text`, `geometry`), but define it as a strict Pydantic v2 /
  Arrow schema with real types and validation, not stringified columns. The
  recon's Stage-0 findings (81% of records have empty `description`, 3.4% carry
  raw HTML) are encoded as schema expectations and quality-gate bands, so the
  data's known shape is asserted rather than assumed.

**Fix the non-deterministic `hash()` ID — this is load-bearing.** The current
Parquet pipeline keys rows with Python's built-in `hash()`, which is
session-randomized and therefore *not stable across runs* — a real, cited defect
that makes any cross-snapshot diff impossible. The fix already exists in this
same ecosystem: the Mongo path used a **deterministic UUID** derived from the
record's URL. The rebuild adopts a deterministic scheme for real —
`uuid5(NAMESPACE, canonical_key)` where the canonical key is a normalized
(server root, service path, layer id) tuple. Stable IDs are not a nice-to-have
here: they are the *enabling primitive* for cheap versioning (Dimension 4),
snapshot diffing, and drift detection (Dimension 6). A deterministic-ID
reproducibility test — build twice, assert identical IDs — is part of the test
oracle (Dimension 7).

**Geometry stays as the bounding-box extent — for both simplicity and
licensing.** The recon confirms the current `geometry` is the layer's extent
reprojected to EPSG:4326, not real feature geometry, and that no true
feature-level spatial data exists anywhere in this ecosystem. Changing this to
real geometry would explode dataset size, multiply crawl time, and move us from
"redistributing lightweight metadata + a bbox" into "redistributing the actual
government data" — squarely into the state/local copyright ambiguity. So the
bbox stays, with one storage improvement for Dataset-Viewer compatibility
(Dimension 4): store the extent as plain `minx/miny/maxx/maxy` float columns
plus a WKT string in the primary `.parquet`, and keep a real `geometry` column
only in an optional `.geoparquet` sibling. The reprojection/clamp/discard logic
that already exists in `to_mongo_4326reproj.py` is the reference implementation —
reuse its behavior, don't reinvent it.

---

## Dimension 4 — Storage and versioning

**One dataset repo, snapshots as tags, described by a manifest.** The recon shows
the current state is two frozen, single-branch, zero-tag repos with no manifest
tying anything together, and *no precedent anywhere for successive snapshots*.
The lowest-maintenance way to establish that precedent is **not** a new repo per
year (`govgis_2024`, `govgis_2025` — that fragments DOIs, discovery, and consumer
pins) and **not** a snapshots/ subdirectory that grows without bound. It is a
single repo where each refresh overwrites the canonical artifact files and the
commit is **tagged `snapshot-YYYY-MM-DD`**. Tags give immutable, citable,
revision-pinnable points at zero conceptual cost — and revision pinning already
demonstrably works (this very Space pins the dataset at a SHA today). One repo,
one card, one DOI, one thing to reason about.

**Ship a manifest — nothing like it exists today.** A single `manifest.json` per
snapshot records: schema version, snapshot date, seed-list source URL + fetch
date + seed checksum, server/service/layer counts, embedding model + pinned
revision, vector dimensions, distance metric, pinned restgdf version, the
pipeline repo's git SHA, SHA-256 checksums of every artifact, and a **diff
summary vs. the previous snapshot** (servers/services/layers added, removed,
changed — cheaply computable *because* IDs are now deterministic). This manifest
is the connective tissue the ecosystem has never had, and it is also what makes
the quality gates (Dimension 6) and consumer pinning trustworthy.

**Use Xet, not plain LFS.** Both current repos are plain Git LFS because they
predate Xet. A pipeline that re-publishes near-duplicate multi-GB Parquet on
every refresh is the textbook case for Xet's dedup/delta storage — smaller
pushes, faster CI, less bandwidth. Enabling Xet on a new repo is a one-time
setting with an ongoing maintenance payoff, so it is an easy call for this lens.

**Solve the Dataset-Viewer gap by shipping plain Parquet as primary.** The recon
confirms the Hub's Viewer still cannot read `.geoparquet` (issue #6438, filed by
this maintainer in 2023, unresolved). Rather than wait on upstream, the primary
search-facing artifact is a plain `.parquet` (bbox as float columns + WKT, per
Dimension 3), which the Viewer and the auto-Parquet-conversion bot handle
normally. The `.geoparquet` ships as an *optional sibling* for geo-native tools
(and keeps the Postgres sibling working — Dimension 5). This closes a
years-open papercut with a file-layout choice, no code and no upstream
dependency.

**Citation/DOI continuity for the superseded repos.** The two existing repos have
Hub-assigned DOIs (`10.57967/hf/1368`, `1369`). They are never deleted; instead
their dataset cards get a "superseded by <new repo>" note, and the new repo's
card carries a "supersedes / continues" back-reference. New repo gets its own
DOI; each tagged snapshot is independently citable by revision. This preserves
every existing citation trail at the cost of two one-line card edits.

---

## Dimension 5 — Serving and access architecture

This is where the simplicity lens most sharply overrules the existing prior art,
so the reasoning is spelled out rather than asserted.

**The primary "serving layer" is the dataset repo itself.** A well-formed Parquet
dataset on the Hub is already a queryable API: `datasets.load_dataset(...)`,
`polars.read_parquet(...)`, `duckdb` over the Hub URL, or a plain file download
all work today with zero code the maintainer keeps alive. For a bulk-download or
analytical consumer, that is the entire answer — and it is the lowest-maintenance
"service" possible, because someone else operates it.

**Do NOT adopt or extend the Postgres+PostGIS+pgvector+FastAPI+FastMCP sibling as
core infrastructure.** This is the central serving decision of this lens, and the
recon supplies its own justification: that stack is Dockerized, always-on,
needs a database and image patched, and is *currently* nine-to-ten months behind
on dependency/security PRs under exactly the single-maintainer conditions this
plan is designed for. Signing the rebuild up to run and patch that indefinitely
is the single highest-burden thing available, for a capability (spatial +
vector query in a relational DB) that most consumers of a *metadata + bbox*
catalog do not need. So it is deliberately excluded from the critical path.

Crucially, excluding it costs almost nothing, because the geoparquet it consumes
is a byproduct this pipeline publishes anyway (Dimension 4). The sibling **keeps
working for free as an optional, user-run, self-hosted deployment** — a power
user who wants in-database spatial+vector queries can `docker compose up` against
the published geoparquet exactly as today. We keep the *minimal slice* (publish
the geoparquet it eats) and drop the *burdensome slice* (the maintainer running
the server). That is the justified partial-keep the brief asks for.

**Search and agent access are thin layers that sleep when idle.** Two
consumer-specific views beyond bulk download, both zero-always-on:

- **Semantic search** is the existing Gradio Space (being modernized *separately*
  — this plan does not touch it), which is itself a managed, sleep-when-idle HF
  Space with no server for the maintainer to patch. FAISS-in-a-Space is, under
  this lens, perfectly fine: it is serverless from the maintainer's perspective.
  The rebuild's only obligation to it is to **publish artifacts in the shape that
  migration already targets** — a native `index.faiss` + typed `documents.parquet`
  + manifest with checksums (exactly the Stage-2 target layout in
  `modernization-plan.md`). By having the pipeline emit that safe layout on every
  snapshot, we eliminate the modernization plan's one-time manual conversion step
  for all *future* snapshots — the two efforts align at this single seam without
  coupling.
- **Agent/MCP access**, if wanted, is a thin MCP surface on that *same* Gradio
  Space (Gradio can expose an MCP endpoint natively) — not a second Postgres+
  FastMCP deployment. One sleep-when-idle Space serves both the human UI and the
  agent tool, versus a standing database and two FastAPI/FastMCP processes.

So the serving picture is: **bulk = the Hub files (zero infra); search + agent =
one managed Space (zero always-on); heavyweight spatial DB = optional, user-run,
untouched.** Three consumer shapes, all fed from one pipeline build, none adding
a thing the maintainer must keep patched.

---

## Dimension 6 — Refresh cadence and automation

**A quarterly cron, plus manual dispatch.** GitHub Actions `schedule:` runs the
full crawl→build→publish pipeline once a quarter, with `workflow_dispatch` for
on-demand runs. Quarterly (not weekly) is a deliberate burden choice: the source
list changes on a scale of months, each run is a self-contained full snapshot,
and a lower cadence means fewer runs to glance at and a smaller GPU-embedding
bill (Dimension 9). Everything about the cadence is tuned so that *doing nothing*
between runs is safe.

**Cheap drift detection that notifies, never auto-acts.** A separate, very light
weekly job fetches the current seed `.txt`, compares its server count/roster
against the last published snapshot's committed seed, and — if drift exceeds a
threshold (e.g. >10% new roots) — **opens a GitHub Issue** saying "the upstream
list has grown materially; a re-crawl may be worthwhile." It does not crawl, does
not publish, does not page anyone. It converts "is it time to refresh?" from a
thing the maintainer has to remember into a thing that shows up in their inbox
when it is actually true. This is the entire monitoring system, and it has no
standing infrastructure.

**Data-quality gates block publish — the safety net that lets neglect be safe.**
Nothing like this exists in the ecosystem today, and it is what makes an
automated pipeline trustworthy without supervision. Before any snapshot is
tagged and published, the run must pass, all computed by script and recorded in
the manifest:

- **crawl-coverage floor** (≥ X% of seed roots crawled successfully — the
  systemic-failure detector from Dimension 2);
- **row-count sanity** (layer/service/server counts within a band of the prior
  snapshot — catches a truncated crawl);
- **schema validation** (every slim record satisfies the Pydantic schema);
- **known-shape bands** (e.g. empty-description share near the ~81% baseline;
  HTML-in-description share near ~3.4% — a wild deviation signals a parsing
  regression, per the Stage-0 evidence);
- **deterministic-ID reproducibility** (a re-key of a sample yields identical
  IDs);
- **embedding integrity** (vector count == row count, correct dimensions);
- **retrieval parity** (Recall@k on a small committed query set — borrowing the
  Stage-0 oracle idea — stays ≥ the prior snapshot's);
- **checksums generated and written to the manifest.**

A failed gate means the snapshot is **not published**, and an Issue is opened
with the failing gate. The last good published snapshot stays live and pinned.
This is the mechanism by which a six-month-unattended pipeline cannot silently
ship garbage: the worst case is "no new snapshot + one Issue," never "a broken
snapshot consumers pinned to."

---

## Dimension 7 — Staged implementation plan

Borrowing the *method* from `modernization-plan.md` — explicit stages, a real
gate per stage, a pre-declared test oracle, adversarial review before building —
without borrowing its architecture. Each stage ends at a scripted, checkable
gate; rollback throughout rests on two facts: **old repos are never deleted, and
every snapshot is an immutable tag consumers pin to**, so "roll back" always
means "point at the previous tag" and is never destructive.

- **Stage 0 — ground and permission.** Confirm the `.txt` mirror parses to clean
  roots; send the Elfelt permission email (Dimension 1); freeze the current
  baseline (the two existing repos + this Space's pinned revision); define the
  test oracle: a small committed query set with expected URLs (reuse/extend the
  Stage-0 set already in `docs/stage0/`), golden schema fixtures, and a
  deterministic-ID reproducibility spec. *Gate:* mirror parses, permission
  recorded, oracle committed.
- **Stage 1 — pipeline skeleton, proven end-to-end on ten servers.** New
  `govgis-pipeline` repo scaffolded to current conventions (uv, pyproject,
  pinned restgdf, pre-commit/ruff/mypy/pytest, CI modeled on
  `geospatial-data-converter`'s mature `ci.yml`). Implement the deterministic-ID
  and schema modules with tests, and run the full crawl→type→slim→manifest path
  against a **10-server seed**. *Gate:* typed raw + slim + manifest produced;
  IDs identical across two runs; CI green.
- **Stage 2 — full-scale crawl and raw/slim artifacts.** Matrix-sharded crawl of
  the full 7,500+ list; combine; emit typed raw (escape-hatch JSON column) and
  slim Parquet; write the manifest with counts and checksums; wire the
  data-quality gates (Dimension 6). *Gate:* coverage floor met, counts/schema/
  known-shape bands pass, checksums written, seed pinned into the repo.
- **Stage 3 — embeddings and search artifacts.** Generate `bge-large-en-v1.5`
  embeddings via **HF Jobs (serverless GPU, pay-per-use)**, build the native
  `index.faiss` + typed `documents.parquet`, and the Dataset-Viewer-compatible
  plain `.parquet` + optional `.geoparquet`. *Gate:* vector count == row count,
  correct dims, Recall@k parity vs. the legacy index on the committed query set.
- **Stage 4 — publish the first snapshot.** New single HF dataset repo, **Xet
  enabled**; upload artifacts + manifest; tag `snapshot-YYYY-MM-DD`; add
  superseded-by/supersedes notes on the old and new cards. *Gate:* Dataset Viewer
  renders the plain Parquet; the snapshot is revision-pinnable; checksums verify
  after download.
- **Stage 5 — automate.** Add the quarterly cron, the weekly drift-detection
  Issue bot, and make the quality gates block publish. *Gate:* a dry-run
  `workflow_dispatch` produces a snapshot unattended and correctly *refuses* to
  publish a deliberately-corrupted test build.
- **Stage 6 — connect consumers (coordinated, not forced).** When the separate
  Gradio migration reaches the point of consuming artifacts, it points at the new
  snapshot's safe layout; the Postgres sibling continues to work off the
  published geoparquet with no change. *Gate:* each consumer resolves and pins a
  snapshot revision; none is broken by the cutover.

**Test oracle, stated up front (not discovered later):** deterministic-ID
reproducibility (build-twice-assert-equal), golden schema fixtures, the
committed query set with Recall@k parity, and the known-shape statistical bands
from Stage-0 evidence. A green pipeline that violates any of these is a false
green and does not advance.

---

## Dimension 8 — Ownership and repo structure

The simplicity lens wants the *smallest number of surfaces the maintainer must
keep alive*, so the target footprint is deliberately minimal:

- **New: `govgis-pipeline` (GitHub).** The build script that never existed on
  GitHub — crawl, type, key, embed, manifest, publish, plus CI and the cron/
  drift automation. This is the one genuinely new thing to maintain, and it is
  a stateless batch job, the cheapest kind.
- **New: one HF dataset repo (e.g. `govgis`).** Snapshots as tags, Xet-backed,
  one manifest, one card, one DOI. Supersedes both `govgis_nov2023` and
  `govgis_nov2023-slim-spatial`, which stay frozen with superseded-by notes.
- **Unchanged: this Space.** Modernized on its own track; repointed at new
  artifacts at Stage 6; not otherwise touched by this plan.
- **Unchanged and un-adopted: the Postgres sibling.** Left exactly as-is as an
  optional user-run deployment fed by the published geoparquet. Explicitly *not*
  extended, because extending it means owning its patch treadmill.

Deliberately rejected structures: **do not fold the pipeline into the Space repo**
(build and serve should stay separable; the Space is on its own migration), and
**do not extend the sibling server repo** (that couples the low-maintenance data
job to a high-maintenance always-on service). Net new standing surface for the
maintainer: one batch-job repo and one dataset repo. Everything else either
stays frozen or keeps working for free.

---

## Dimension 9 — Cost and ops burden

Realistic accounting, with the tradeoffs named rather than hidden:

- **Orchestration & CPU (crawl, typing, slimming, manifest, gates): ~$0.**
  GitHub Actions is free for public repos; a matrix-sharded quarterly crawl and
  a light weekly drift check sit well within free limits.
- **GPU embeddings: small, occasional, pay-per-use.** Embedding ~865k texts
  through `bge-large-en-v1.5` on HF Jobs is a few GPU-hours per *full* re-embed —
  single-digit-to-low-tens of dollars — and, crucially, **only runs when the
  embedding model or the `metadata_text` actually changes**, gated by the
  manifest. Most quarterly refreshes that only add/remove servers re-embed just
  the changed rows, not all 865k. This is the one non-zero recurring cost, and it
  is bounded and infrequent by design. (Doing embeddings on free Actions CPU was
  considered and rejected: 865k × bge-large on CPU would blow the 6-hour job
  ceiling — paying a few dollars of serverless GPU is the lower-*burden* choice
  even though it is not the lower-dollar one.)
- **Storage: cheap and shrinking-per-refresh.** Xet dedup means each new snapshot
  uploads deltas, not a fresh ~10 GB, so storage growth is sublinear in snapshot
  count.
- **No always-on anything: $0 and, more importantly, no patch/uptime/security
  burden.** This is the real savings versus the Postgres path. There is no
  database to patch, no image CVE backlog, no public endpoint the maintainer is
  on the hook for. The recon's nine-to-ten-month unmerged-PR backlog on the
  sibling is the concrete number this avoids.
- **Human burden: on the order of one to two hours per quarter.** Read the drift
  Issue, kick off (or approve) a snapshot, glance at the gate results and the
  manifest diff, done. There is no daily or weekly obligation, and skipping a
  quarter is safe. That is the entire point of the design.

One-time costs: the Elfelt permission email (minutes), and — *only if* commercial
use or real-feature-data redistribution is ever contemplated — a real legal
review. Neither is on the recurring path.

---

## What this plan deliberately does NOT build (and why the simpler path holds)

- **No Postgres / PostGIS / pgvector / always-on API or MCP server as core
  infra.** The recon's own evidence — a nine-to-ten-month security-PR backlog on
  exactly that stack under exactly these single-maintainer conditions — is the
  argument. The capability is preserved as an optional user-run deployment fed by
  a byproduct we publish anyway; the *burden* is dropped.
- **No real feature geometry.** Bbox-extent only, for both size and the
  state/local copyright ambiguity. Real geometry would multiply size, crawl time,
  and legal exposure.
- **No fully-typed 205-column schema.** A JSON/struct escape-hatch column keeps
  every field without a brittle wide schema to maintain.
- **No incremental-crawl state (initially).** Full self-contained snapshots have
  nothing to corrupt; incremental is a later option *only if* crawl cost becomes
  binding.
- **No repo-per-snapshot and no unbounded snapshots/ directory.** One repo, tags —
  which keeps DOIs, discovery, and consumer pins from fragmenting.
- **No `.geoparquet` as the primary artifact.** Plain Parquet primary so the
  Dataset Viewer works today; `.geoparquet` an optional sibling.
- **No bespoke crawler and no PDF scraping.** restgdf for crawling; the `.txt`
  mirror for ingestion (respecting the PDF-scraping prohibition and dodging the
  fragility that already killed the CSV path).

The through-line: every rejected item is something that would demand attention on
a schedule the maintainer cannot reliably keep. What remains is a stateless batch
job and a pile of files on someone else's storage — a system whose failure mode
under neglect is "no new data," never "broken data" or "a compromised server."

---

## Open questions for synthesis (honest gaps in this lens)

- **Re-embed granularity.** The plan assumes changed-rows-only re-embedding is
  feasible via manifest-tracked text hashes; the exact mechanism (and its
  correctness under schema evolution) needs the implementation-time detail that
  Stage 3 would produce.
- **Matrix shard sizing.** The 6-hour ceiling argument is sound directionally,
  but the real per-shard crawl time at 7,500 servers is unmeasured; Stage 2
  should measure a single shard before fixing N.
- **Whether search artifacts belong in the pipeline at all.** This draft has the
  pipeline emit the FAISS/documents layout to serve the modernization plan; a
  competing view is that search-index construction belongs to the *consumer*
  (the Space), keeping the dataset repo purely data. Synthesis should weigh
  coupling-cost against the convenience of every snapshot shipping serve-ready.
- **The Elfelt relationship is a single point of dependency.** The entire
  ingestion path rests on one volunteer's weekly-maintained list with no SLA. The
  seed-pinning mitigation softens a sudden disappearance but does not replace the
  source. No comparable catalog exists; this is a real, un-eliminable risk this
  lens can only mitigate, not solve.
