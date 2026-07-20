# Rebuilding the govgis dataset pipeline — synthesized rebuild plan

Status: **final revised plan**, 2026-07-20. This rebuild plan was assembled from
four independent lens drafts (`docs/rebuild-plan-audit/draft-*.md`) and three
judges' verdicts (`docs/rebuild-plan-audit/judge-{a,b,c}.md`), then **adversarially
audited across seven axes plus a completeness critic and revised against every
surviving finding** (executive summary and per-axis evidence in
`docs/rebuild-plan-audit/`). It remains deliberately explicit about the decisions
it makes and the evidence behind them. Where the audit refuted a finding (notably
the claim that public-repo GitHub Actions artifact storage is quota-limited — it
is free for public repos), the plan was left unchanged and the reason recorded.

Scope: a from-scratch rebuild of the **data pipeline** that produces the
`govgis` dataset — the seed ingestion, crawl, transforms, published artifacts,
their versioning, and their refresh automation. It is **not** the Gradio Space
UI migration, which is a separate, already-in-progress effort
(`docs/modernization-plan.md`) and is not touched here. This plan produces the
data that migration will eventually consume; the two efforts meet at exactly one
seam (Dimension 5) and this document is careful not to reach across it.

Grounding: `docs/ecosystem-recon/` (five cited recon files, read in full), the
sibling `govgis_nov2023-slim-spatial-server` source, and **two live fetches of
Elfelt's seed `.txt` mirror made while writing this synthesis** (see Grounding
Correction 1). Method — staged implementation with executable gates, explicit
test oracles, and adversarial review before implementation — is borrowed from
`docs/modernization-plan.md`; its architecture (FAISS files on a Space) is not.
Per the maintainer's explicit instruction, no prior architectural decision
(single-repo/FAISS, or the sibling's Postgres+pgvector+MCP) is treated as a
default; each is evaluated as an option and argued.

---

## Base lens and how this synthesis was built

The **base lens is operational-simplicity.** Two of the three judges (a, c)
recommended it, and the recon supplies the decisive reason: every repo in this
ecosystem has exactly one committer, and the most sophisticated piece of
infrastructure in it — the Postgres+PostGIS+pgvector+FastMCP sibling — is
carrying nine-to-ten months of unmerged dependency/security PRs. That backlog is
the clearest available measurement of what always-on infrastructure costs a
single spare-time maintainer. The design goal is therefore **a stateless batch
pipeline that produces a good dataset when told to and then turns completely
off, breaking nothing while it sits untouched for six months.**

The third judge (b) recommended data-engineering-rigor as the base, and that
recommendation is honored not by switching the spine but by treating rigor's
correctness machinery as **non-negotiable on top of the simplicity spine.** This
resolves what looks like a tension but is not: fail-closed gates, deterministic
IDs, a checksummed manifest, and a drift panel are exactly what *make* an
unattended pipeline safe to neglect. Simplicity gets you "turns off cleanly";
rigor gets you "the worst failure is *no new data*, never *broken data*." You
need both, and they do not fight. The rigor draft's own thesis — "make
correctness checkable at every hop instead of trusted" — is adopted wholesale;
what changes is that the checkable pipeline is a stateless batch job, not a
medallion warehouse with standing infrastructure.

Onto that spine this synthesis grafts the best ideas the judges identified from
the agentic-serving and geospatial-native drafts — hybrid retrieval as the
primary query model, a composable typed MCP tool set, embedded-DuckDB
spatial/FTS/structured serving with a real quantized FAISS vector index,
jurisdiction/facet recovery, lossless honest geometry, and licensing that binds
Elfelt's terms to the whole product — as **optional, consumer-specific tiers**
that never become always-on burden. The conflicts between drafts (build Postgres vs. avoid
it; full vs. incremental crawl; type-everything vs. type-nothing) are resolved
explicitly in "Conflicts resolved," not hedged.

---

## Two grounding corrections this synthesis makes (the judges' shared gaps)

All three judges converged on two shared gaps common to every draft. This
synthesis closes both with primary-source grounding, and both change the design.

### Grounding Correction 1 — the surviving seed mirror is NOT a columnar table (refutes the single most-grafted idea as written)

Every one of the four drafts stakes seed ingestion — and the "single cheapest
high-leverage change," jurisdiction/FIPS facet recovery — on the assumption that
the live `.txt` mirror carries the dead CSV's columnar schema
(`Type, State, County, Town, FIPS, Server-owner, ArcGIS-url`). The geospatial
draft states it outright: "it carries the same columnar schema as the dead CSV."
No draft read a real sample. The recon itself only ever confirmed the `.txt`
returns *prose* header content (size line, license terms, cadence), never its
row format. This is precisely the maintainer's own posture trap: a format-reader
is a *claim* about a format; read a real sample before concluding a field is
present.

**I fetched the live `.txt` twice while writing this.** It is
**Elfelt's human-readable report rendered as plain text — the text of the PDF,
not a columnar CSV.** Verbatim, its structure is a title block, a byline
("By Joseph Elfelt, https://mappingsupport.com", "June 18, 2026"), a numbered
table of contents (1. Copyright and terms … 6. Federal GIS Servers p.12 …
7. State, Regional, County and City GIS Servers p.33 … 8. Washington D.C. p.463
… 9. Native American Tribes p.464 … 10. U.S. Territories p.465 …), and then
jurisdiction-sectioned entries whose real format is:

```
* USDA - Forest Service - Geospatial Data Discovery
     Website: https://data-usfs.hub.arcgis.com
     GIS: https://apps.fs.usda.gov/arcx/rest/services
          2-22-2025 interesting wildfire data at RDW_Wildfire
```

```
     Alabama Department of Conservation and Natural Resources
     Website: https://www.outdooralabama.com
     GIS: https://conservationgis.alabama.gov/adcnrweb/rest/services
          7-30-2023 No tiled data
```

There are **no `FIPS`, `County`, `Town`, or `Server-owner` columns.** There is a
free-text **agency/owner name** (the heading line), a `Website:` URL, a `GIS:`
`rest/services` URL, and free-text **dated annotation** lines. Jurisdiction is
implied by **section** (Federal / State-Regional-County-City / DC / Tribal /
Territory / Misc / Canada) and, within the giant State section, by **sub-headings
grouped by state**. This directly **refutes** the columnar-facet-recovery thesis
*as written* in the agentic and geospatial drafts, and it means the seed loader
cannot be "regex the ArcGIS-url column plus preserve the FIPS columns." There are
no FIPS columns to preserve.

The constructive resolution (Dimensions 1 and 3) is that facet recovery is
*still* the highest-leverage change — but the **source is corrected and made to
degrade gracefully** rather than depend on an unverified columnar feed:

- **Jurisdiction level** (federal / state / regional / county / city / tribal /
  territory / misc) — parsed from the `.txt` **section structure**. Reliable.
- **State** — parsed from the state **sub-headings** within the State section.
  Parseable with moderate effort.
- **Agency / owner name** — the free-text heading line above each `GIS:` URL.
  Available, unnormalized (a string, not a clean facet).
- **County / place / FIPS (precise)** — **not present in the `.txt`.** Recovered
  by *deriving* them: spatial-join each layer's bbox extent (centroid) against
  Census TIGER county/place polygons to assign FIPS from geometry, plus host and
  agency-name heuristics (`gis.traviscountytx.gov`, "Travis County …"). Elfelt's
  original **CSV** — which did carry those columns — becomes an **enrichment that
  upgrades precision if obtained** (see the elevated Stage-0 contact gate),
  never a blocker on the critical path.

This is a more robust design than any single draft's, and it is the strongest
possible way to close the gap the judges flagged: I read the real sample and it
changed the plan.

There is a second consequence, on licensing. The **sanctioned machine-readable
feed (the CSV) is dead**, and the surviving mirror is the **text of the same list
whose PDF form Elfelt's terms explicitly prohibit scraping** ("Scrapping data
from the PDF file is prohibited."). The strongest *available* argument that
parsing the `.txt` is acceptable is evidentiary, not rhetorical: Elfelt publishes
the list in **three parallel representations at the same path** — `.pdf`, `.csv`,
and `.txt` (`upstream-source-findings.md` §1) — himself hosts the `.txt`, the
prohibition names the **PDF specifically**, and the mappingsupport.com index page
states **"Everyone is welcome to share this list"** (`upstream-source-findings.md`
§1). Parsing a maintainer-hosted plain-text representation whose sibling CSV was
explicitly the sanctioned machine-readable form is therefore materially different
from "scraping the PDF." **But** the sanctioned form is gone and the terms do
single out the PDF, so **the permissibility of the ingestion path is a genuine
open question this plan does not presume to have settled** — it is a recorded open
decision, gated, not an assumption. This elevates "contact Elfelt to confirm an
acceptable machine-readable path (ideally restore/share the CSV)" from a
recommended nicety to a **Stage-0 launch gate** (Dimensions 1, 6, and 7), and it
ties the two hardest problems together: the same email that secures the ingestion
path can recover the CSV that upgrades the facets. Crucially, the gate must
authorize not just **parsing** the `.txt` but the **standing weekly automated
fetch** the refresh design depends on (Dimension 6) — automated repeated
retrieval is a distinct question from a one-time read, and the recon cautioned
specifically against bulk programmatic access (`upstream-source-findings.md` §1).

### Grounding Correction 2 — re-baseline the entire cost/feasibility envelope on ~3–4M layers, not 865k

Every draft sizes embedding cost, index memory, storage, and in-browser
feasibility against the **stale Nov-2023 numbers** (865,864 layers from 1,684
servers) even though the shared premise is that the seed list quadrupled-to-
quintupled to 7,500+ servers. That is the "headline number's scope/inclusion
basis is a claim the arithmetic gate can't check" failure mode — every cost
section is internally consistent but anchored to a corpus its own premise has
outgrown.

**Re-baselined estimate (explicitly an estimate, to be measured — see the
Stage-1 measure-first gate):**

- Nov 2023 ratios (from `servers.parquet` / `services.parquet` /
  `layers.parquet`): 1,684 crawled servers → 195,479 services → 865,864 layers,
  i.e. **~514 layers/server**, ~116 services/server, ~4.4 layers/service. The
  1,684 were the survivors of ~2,038 crawled candidates (~83% yield).
- 7,500 seed servers × ~83% crawl yield ≈ **~6,200 crawled servers**; at ~514
  layers/server ≈ **~3.2 million layers**. Defensible working range: **~3–4M
  layers** — roughly **4× the 865k baseline.** Treat every 865k-derived figure
  in the drafts as a **lower bound.**

Propagating that into the numbers the drafts got wrong:

- **Crawl wall-clock.** 2h16m for 2,038 roots at `Semaphore(10)` → **~8–9h** for
  7,500 at similar politeness. This **exceeds GitHub Actions' ~6h per-job
  ceiling** — so the crawl must be **matrix-sharded** across parallel free
  runners (Dimension 6), and shard sizing must be **measured**, not guessed
  (Stage-1 gate: crawl one representative state shard, record wall-clock and
  layers/server, then compute N).
- **Embedding compute.** ~3.5M texts through `bge-large-en-v1.5` (1024-dim) at a
  realistic GPU throughput of ~500–2,000 texts/s is **~0.5–2 GPU-hours of pure
  inference** for a full re-embed (3.5M ÷ 2,000 ≈ 0.5 h; 3.5M ÷ 500 ≈ 2 h). Add
  model-load, tokenization, batching, and IO overhead and the realistic
  wall-clock is **~1–3 hours** — still **single-digit-to-low-tens of dollars** on
  pay-per-use HF Jobs GPU (e.g. L4 ≈ $0.80/h, A10G ≈ $1.3/h, A100 ≈ $4/h). Bounded
  and infrequent, and **only a full re-embed pays it** (see the incremental-embed
  decision). (The earlier drafts' "~2–7 GPU-hours" did not follow from this
  throughput and is corrected here; the dollar conclusion is unchanged because
  the number was already conservative.)
- **Vector footprint, scoped per serving surface — the decisive feasibility
  finding.** 3.5M × 1024 dims × 4 bytes (fp32) ≈ **~14 GB** of raw vectors. The
  ANN requirement is real but it is **surface-specific**, and the plan must not
  conflate the two surfaces it serves:
    - **Vector-first surface — this repo's Gradio Space.** A semantic-search UI
      that scores a query against *all* vectors needs a resident ANN index. A flat
      fp32 index at ~14 GB **does not fit a free `cpu-basic` HF Space** (16 GB,
      minus ~1.3 GB for the bge-large model and app overhead). fp16 ≈ 7 GB (still
      tight); int8 scalar-quantized ≈ 3.5 GB (fits); PQ ≈ 0.3–0.9 GB (comfortable).
      **On this surface a quantized ANN index is MANDATORY at true scale** — and
      IVF/PQ/SQ8 quantization is **FAISS vocabulary**, so this index is built with
      **FAISS** (already in-house, already what the Space consumes), *not* DuckDB
      VSS. DuckDB VSS builds an in-memory full-precision float32 HNSW over
      `FLOAT[N]` arrays, exposes **no** SQ8/PQ knob, has no IVF, and its persistent
      HNSW historically requires an experimental flag and loads wholly into RAM —
      i.e. a DuckDB VSS index over 3.5M × 1024 reproduces the exact ~14 GB
      footprint this bullet says will not fit. The quantized index therefore comes
      from FAISS (see Dimension 5 and Conflict #7).
    - **Hybrid-search surface — the optional DuckDB search/MCP tier.** Here the
      query model is spatial → structured/FIPS → full-text → *optional* dense
      re-rank of an already-filtered candidate set (typically hundreds-to-low-
      thousands of survivors). A brute-force dot-product over that small set needs
      **no resident ANN index and no quantization** — the survivors' vectors are
      fetched by id and scored on the fly. This surface needs the vectors reachable
      by id, not a 14 GB in-RAM HNSW.
  So there is **one embedding artifact** and, where an ANN index is genuinely
  required, **one FAISS index** — never two independent ANN structures over the
  same vectors. DuckDB owns spatial/FTS/structured filtering; FAISS owns the
  quantized vector index for the vector-first Space; the hybrid tier re-ranks a
  small candidate set by brute force and delegates any at-scale vector search to
  the same FAISS index. **The embedding *dimension* is the root of the whole
  footprint cascade:** a 384-dim model (e.g. `bge-small-en-v1.5`) cuts raw vectors
  to ~5 GB fp32 — small enough to fit `cpu-basic` with *no* quantization — so model
  choice is a first-class **Stage-3** decision weighed on retrieval quality vs.
  footprint, not a Stage-6 afterthought (Open Decision 7).
- **Index build-time memory, and where the build runs.** Building/training a
  quantized FAISS index does *not* need all 14 GB resident at once: IVF/PQ trains
  on a representative sample (a few hundred thousand vectors) and then *adds* the
  remainder in memmapped batches, so peak build RAM is a fraction of the flat
  footprint. The index is built **on the same HF Jobs GPU step that produced the
  embeddings** — the vectors are already resident there, so co-locating the build
  is zero marginal data movement — **not** on the 16 GiB standard GitHub runner,
  whose ~14 GB disk / ~16 GB RAM cannot hold the flat vector set. Peak build memory
  is a **measured Stage-3 gate** (parallel to the crawl's Stage-1 measure-first
  gate), not an assumed-free step.
- **DuckDB-WASM in-browser feasibility.** At 3.5M rows the typed metadata parquet
  (~1–2 GB) is queryable in-browser via **HTTP range requests over a
  state-partitioned parquet** — DuckDB-WASM does partial reads, so structured +
  spatial + full-text filtering *is* feasible without loading the whole file. But
  ~14 GB of vectors is **not** browser-loadable, so **in-browser vector re-rank
  is off the table; the WASM path is structured/spatial/FTS filtering only, with
  any vector re-rank staying server-side.** This refines the geospatial draft's
  "DuckDB-WASM runs the full hybrid incl. vss in-browser," which does not hold at
  this scale.
- **Storage per snapshot.** Typed rich parquet (~4–8 GB) + slim/plain-parquet
  (~1–2 GB) + geoparquet sibling + embeddings (~4–14 GB depending on precision) +
  index ≈ **~15–30 GB per full snapshot**, embeddings dominating. At ~4
  snapshots/year that is 60–120 GB/yr **without dedup** — which is exactly why
  **Xet** (content-defined chunking, cross-snapshot dedup) and **keeping
  embeddings in their own file** (so a re-embed does not churn metadata) are
  load-bearing storage choices, not conveniences (Dimension 4).

The execution design below is built around the **~3–4M** figure, and the first
real crawl gate is a **measurement** that replaces this estimate with a number.

---

## Architecture in one paragraph

A scheduled, matrix-sharded **GitHub Actions** workflow fetches Elfelt's current
`.txt` mirror, parses it tolerantly into a checked-in, checksummed seed with a
recorded per-server `crawl_outcomes` ledger; crawls every seed root with a pinned
**restgdf v3** (`[resilience]` extras) across parallel shards; lands the raw
per-server JSON **content-addressed and immutable**; transforms it through typed
Pydantic v2 / Arrow schemas into a validated table set with **deterministic,
content-addressed BLAKE2b IDs**, honest lossless geometry, recovered
jurisdiction facets, and a per-record `license_status`; embeds **only the layers
whose text changed** on a pay-per-use **HF Jobs GPU**; builds a quantized
**FAISS ANN index** on that same GPU step; and publishes a **single Hugging Face
dataset repo** whose
successive **snapshots are immutable tags**, each described by a checksummed
**manifest with a lineage chain** and each gated by a **fail-closed data-quality
drift panel** before publish. There is **no always-on server or database in the
critical path.** Serving is a set of **optional, swappable views** over the
pinned snapshot — a bulk-download dataset (the Hub itself), an embedded-DuckDB
hybrid search + typed **MCP tool set** for agents, and this repo's Gradio Space
as a thin consumer of a checksummed native index — with the sibling's
Postgres+pgvector stack kept as a documented, user-run fallback, never operated
as core infrastructure.

---

## Dimension 1 — Seed source and licensing

### Getting the list, given the dead CSV and the prose-only mirror

The `.csv` endpoint the old pipeline proxied (`restgdf_api/mappingsupport.py`)
returns HTTP 404 and has since at least 2025-12-15 (issue #74, unanswered). Per
Grounding Correction 1, the surviving `.txt` mirror is **Elfelt's human-readable
report** ("7,500+ ArcGIS server addresses for the USA," updated 2026-06-18,
refreshed most Wednesdays), not a columnar table. The design consequences:

1. **Ingest the `.txt` mirror as the primary seed, parsed as a structured
   document, not a CSV.** The seed loader is a **section-and-entry parser**, not
   `read_csv`: it walks the report's sections to assign a coarse
   `jurisdiction_level`, walks state sub-headings within the State section to
   assign `state`, and extracts, per entry, the agency/owner **name** (heading
   line), the `Website:` URL, every `GIS:` `.../rest/services` root, and the
   dated annotation text. It is tolerant by construction (the source was already
   ragged — issue #31's "Expected 15 fields, saw 25"), quarantines
   un-parseable blocks into a report rather than aborting or silently dropping,
   and dedupes roots. It **never fetches the PDF** — the PDF URL appears nowhere
   in the codebase, enforced by a unit test.
2. **Commit the exact seed every run.** Each run writes `seeds/YYYY-MM-DD.txt`
   plus a SHA-256 into the pipeline repo *before* crawling. The dead-CSV episode
   is the whole lesson: an upstream feed can vanish silently. If the mirror 404s
   on a future run, the build fails a gate loudly and the last committed seed is
   a usable fallback. One small file per run — negligible storage, large
   resilience payoff, zero standing infrastructure.
3. **Contact Elfelt — as a Stage-0 gate, not a nicety.** Because the sanctioned
   machine-readable form (CSV) is dead and the surviving mirror is the text of
   the scraping-prohibited PDF, the *permissibility of the ingestion path itself*
   is now an open question. A single email (issue #74 is his own project's
   unanswered request) does three things at once: confirms parsing the `.txt` is
   acceptable; asks whether the **CSV can be restored or shared** (which would
   recover the precise FIPS/county facets and upgrade Dimension 3); and secures
   the free-derivative acknowledgment below. Recorded in the repo, it is the
   Stage-0 launch gate.
4. **Decision branches for the Elfelt gate — because he is non-responsive.**
   Issue #74 has sat unanswered since 2025-12-15, so the gate must specify what
   happens on each outcome, not merely require that contact was attempted (see
   Stage 0). Three branches: **(a) Elfelt confirms** the `.txt` path (and ideally
   restores the CSV) — proceed to automated ingestion, record the confirmation
   reference. **(b) Elfelt declines automation** or asks for manual-only access —
   fall back to the committed-seed model (point 2) operated *manually*: a human
   downloads the `.txt` on the cadence he permits and commits it, and the weekly
   automated fetch cron is disabled; the pipeline still runs from the committed
   seed. **(c) No reply within a defined window (e.g. 60 days).** This is the
   expected case. The default is to **proceed under the free/non-commercial
   posture only** — parsing a maintainer-hosted, share-welcomed plain-text
   representation for a free public derivative, with the decision, its date, its
   rationale, and the unanswered-contact record checked into the repo, and with a
   standing commitment to honor any later objection immediately (the
   `excluded_servers`/takedown lever and the ability to unpublish). Proceeding is
   a **recorded, reversible decision with a named default**, not an unbounded wait
   and not a silent assumption. Commercial use stays hard-blocked regardless
   (`commercial_use_authorized` defaults `false`).

### Source diversification: complementary and fallback seed sources

The plan's founding lesson is that **an upstream feed can vanish silently** — it
is why this whole rebuild exists. Answering that lesson with only "commit the last
seed" hardens, but does not remove, a single-point dependency on one non-responsive
volunteer's scraping-restricted feed. Dimension 1 therefore explicitly evaluates
complementary sources (the recon surfaced concrete candidates in
`upstream-source-findings.md` §2), and the conclusion is deliberately staged, not
hand-waved:

- **Elfelt's list stays the primary seed.** The recon is unambiguous that **no
  other actively-maintained, comparably-comprehensive, server-root-level catalog
  exists** (`upstream-source-findings.md` §2 bottom line); the alternatives
  operate one level up (dataset-metadata, not server roots). So diversification
  *complements* Elfelt, it does not replace him.
- **data.gov / GeoPlatform.gov (NGDA) — a complementary, licensing-clean federal
  layer.** These index *datasets*, many of whose distribution URLs are ArcGIS
  REST/MapServer/FeatureServer endpoints. Harvesting those distribution URLs and
  reducing them to `rest/services` roots yields an **independent, federal,
  unambiguously-public-domain** seed slice that (a) does not inherit Elfelt's
  terms and (b) survives if his feed dies. Adoption is a **Stage-6 option**:
  useful as resilience and as a cross-check on Elfelt's federal coverage, not
  required for the first snapshot.
- **FGDC Clearinghouse Registry — a follow-up federal source to scope.** The
  recon flags it as a legacy CSW/ISO-19115 clearinghouse network worth a look for
  an "official" federal complement; its current technical form is unverified
  (`upstream-source-findings.md` §2), so it is a **Stage-6 investigation item**,
  not a committed source.
- **Shodan/Censys ArcGIS-REST fingerprinting — a genuine independent-discovery
  technique, gated on licensing and trust.** Fingerprinting the ArcGIS REST API's
  characteristic JSON at internet scale is a source with **no dependency on
  Elfelt at all** and is the strongest true diversification available. But it has
  no prior art the recon could cite, it discovers untrusted hosts with **no
  provenance or curation** (raising exactly the crawl-security concerns of
  Dimension 2), and it carries its own terms-of-service constraints. It is
  therefore a **deliberately-deferred Stage-6 research option** — the escape hatch
  if Elfelt's list ever dies for good — evaluated before adoption, never a silent
  default.

The design consequence today is small but real: the seed loader and
`crawl_outcomes` ledger are built to **union multiple seed sources keyed by
normalized root URL** (dedup is already required for Elfelt's own duplicates), so
adding a federal-portal slice later is a new source row, not a re-architecture.
The single-source risk is named as a first-class **Open Decision**, not left
implicit.

### Honoring the licensing in the design — Elfelt's terms bind the whole product, enforced not noted

Elfelt's terms, quoted verbatim from the mirror
(`upstream-source-findings.md` §1):

> "Scrapping data from the PDF file is prohibited."
> "Commercial products based on this list are prohibited unless specific written
> permission is obtained from Joseph Elfelt authorizing that commercial use."
> "Permission is given for anyone to make a derivative work based on this list as
> long as the derivative work is available to everyone for free."

Below the list, the crawled servers themselves are legally non-uniform: federal
GIS data is public domain (17 U.S.C. §105), but state/local is **not** — directly
conflicting court rulings exist (NY/SC permit state copyright assertions over GIS
data; FL/CA reject them; many states have no clear guidance;
`upstream-source-findings.md` §5). The design makes all of this operative:

- **Licensing that treats Elfelt's terms as binding on the whole product — not a
  permissive stamp that quietly overrides them.** The earlier framing (an MIT/CC0
  "own crawled-metadata contribution" tier beside an Elfelt-terms "server-URL list"
  tier) is **corrected here**, because it contained a real contradiction: the whole
  catalog is a **derivative work "based on this list"** — Elfelt's list has been
  this maintainer's single seed source since 2020 (`github-findings.md`) — so his
  conditions (**free availability**, **no commercial use without written
  permission**) plausibly reach the *entire* product, not just the URL column.
  Stamping the metadata MIT/CC0 would grant downstream parties **unrestricted
  commercial use** over Elfelt-derived content, directly contradicting the plan's
  own commitment (Dimension 1, upstream §5) to treat those terms as binding — and
  a commercial vendor relying on that stamp would breach Elfelt's terms with the
  maintainer having mislicensed the material. The design instead:
    - **A single dataset-level license posture governs the published product: a
      `NOTICE` carrying Elfelt's verbatim terms** (free-derivative-welcomed,
      commercial-use-needs-written-permission) **applies to the dataset as a
      whole**, machine-readable in the manifest (`license.elfelt_terms_ref`) and
      restated on the card. There is no tier that authorizes downstream commercial
      reuse of the catalog.
    - **MIT/CC0 is scoped only to the genuinely separable, non-Elfelt-derived
      artifacts** — the pipeline *code*, the schema registry, and independently
      authored documentation — which are not "based on the list." It is **not**
      applied to the crawled catalog rows.
    - **The legal softener is noted, not leaned on.** Under *Feist*, facts (a
      server's URL, its declared extent) are uncopyrightable, so Elfelt's terms may
      not be legally *enforceable* over the bare facts. The plan nonetheless treats
      his stated terms as **binding by choice** (they are the social license that
      keeps this ecosystem healthy), and defaults to the conservative posture
      rather than litigating *Feist* in a dataset card.
  The dataset does not stamp one blanket permissive license across the whole thing,
  as the current repos implicitly do (and which Dimension 4 corrects on those old
  repos too).
- **`license_status` as a typed enum that GATES redistribution scope.** Derived
  per record: `federal_public_domain` (owner classified federal from section +
  URL/name heuristics); `declared_open:<spdx-or-text>` and
  `declared_restricted:<text>` (from the service/layer's own `licenseInfo` /
  `copyrightText` / `accessInformation`, which ArcGIS REST exposes and which
  every current schema discards); `unverified_state_local` (non-federal, no
  declared license — the legally ambiguous default). At publish time this column
  gates scope: `federal_public_domain` and `declared_open` rows may be
  redistributed in full; `unverified_state_local` and `declared_restricted` rows
  default to **link/index-only** (identifying fields + live source URL, no
  full-metadata redistribution). The pipeline's default behavior is the
  legally-safe one; widening it per-jurisdiction is a deliberate, recorded
  decision, not a silent assumption.
- **`commercial_use_authorized` — a fail-closed precondition flag.** Defaults
  `false`; if ever set `true` it requires a checked-in reference to the specific
  permission grant. "Did we get Elfelt's sign-off?" becomes a build-time
  precondition, not a memory. SWCA is a commercial consultancy, so this clause is
  live, not hypothetical; the default posture is free/non-commercial, which needs
  no permission.
- **Index-and-link, never bulk-redistribute feature data.** The catalog stays at
  *metadata + a bbox extent + a link to the live service* and does not
  materialize feature geometry. This is simultaneously the geospatially-honest,
  licensing-safe, and cheapest choice — redistributing a rectangle-extent +
  description + link is defensible; mirroring scraped county parcel geometry at
  scale is exactly the case the FL/NY split is about.
- **Cheap, no-infrastructure control levers.** Honor `robots.txt`; identify a
  descriptive `User-Agent` with a contact; rate-limit per host; and keep a
  committed `excluded_servers.txt` denylist so a takedown/opt-out is "add a root,
  it is skipped next crawl" — a trivial lever with no standing service. Real
  50-state legal review is flagged as a standing note required *only* before any
  commercial use or any redistribution of actual feature data — not a task this
  pipeline resolves.

---

## Dimension 2 — Crawl / ETL architecture

**Tooling: `restgdf` v3.0.0, pinned, with resilience extras.** It is the
ecosystem's healthiest repo (typed Pydantic v2 `LayerMetadata`/`CrawlReport`,
`Directory.crawl`/`safe_crawl`, MIT, PyPI, ReadTheDocs, green CI) and is
purpose-built for this bulk-server sweep. Depend on it as a normal pinned
dependency and use `restgdf[resilience]` (stamina + aiolimiter) rather than
re-implementing retry/rate-limiting. The recon's survey confirms the choice:
Esri's official ArcGIS API for Python is heavier and org-administration-oriented;
`restapi` is synchronous/GPL; `esri2gpd` is synchronous/single-layer. Keep a thin
raw-`httpx` fallback for the minority of servers restgdf chokes on, recorded as
`crawl_method="fallback"`. Do **not** resurrect `restgdf_api`'s dead CSV proxy;
the seed loader reads the `.txt` directly.

**Concurrency, retry, rate-limiting, politeness.** Bounded global concurrency
(the seed spans thousands of independent hosts, so global concurrency can be high
— e.g. 20–50 in-flight) with **per-host rate limits** (aiolimiter, keyed on
hostname; a county's single ArcGIS box must not be hammered — it is both rude and
a fast path to an IP block). Retry only transient failures (timeouts, 5xx,
connection resets) with bounded exponential backoff and a hard attempt cap;
**never** retry deterministic 4xx (404/403 is a *finding*, not a transient
error). Per-request timeout budget plus an overall wall-clock budget; a hanging
endpoint cannot stall the run. All parameters are recorded in the run config and
stamped into the crawl report — they start conservative and are tuned from
measured data, not guessed once.

**The crawl is a security boundary, not just a politeness problem — this is the
largest new trust surface the rebuild introduces.** The current Space only
*consumes* artifacts; a rebuild that fetches from **~6,200–7,500 untrusted
third-party hosts** — many found "with simple Google searches," serving "draft
and/or temporary" data (`upstream-source-findings.md` §1) — is making thousands of
outbound requests to endpoints it does not control, some of which can be hijacked
or planted. Liveness controls (timeouts, rate limits, `robots.txt`) do not address
this. The crawl fetcher therefore enforces:
- **SSRF / egress restriction.** Before any request, the resolved destination IP
  is checked against a **deny-list of private and link-local ranges** — RFC-1918
  (`10/8`, `172.16/12`, `192.168/16`), loopback (`127/8`, `::1`), link-local
  (`169.254/16`, including the `169.254.169.254` cloud-metadata endpoint), and
  IPv6 ULA/link-local. A crawl target that resolves into those ranges is refused
  and recorded as a `crawl_outcomes` outcome, not followed. A scheme allowlist
  (`http`/`https` only) is **necessary but not sufficient** — `http://169.254.169.254/`
  passes a scheme check — so the IP check is the actual control.
- **Redirect pinning.** Redirects are followed only to hosts that pass the same
  IP/scheme checks (re-validated on each hop, defeating DNS-rebinding and
  redirect-to-metadata), with a bounded redirect count; a cross-scheme or
  into-private-range redirect terminates the fetch.
- **Response-size and decompression caps.** A hard **byte cap per response** and a
  **decompression-ratio cap** (defeating zip/gzip bombs from a hostile or
  compromised host) — the wall-clock budget bounds *time*, but nothing today bounds
  *bytes*, and a single malicious response could exhaust runner disk/RAM.
- **Self-hosted-runner fallback carries a stated security cost.** The documented
  fallback (self-hosted runner / HF Job / ephemeral VM, Dimension 6) runs
  untrusted-host fetches on infrastructure that may hold cloud credentials — the
  textbook metadata-credential-theft posture. It is therefore **only** authorized
  on an **ephemeral, credential-stripped** VM with egress limited to the crawl,
  never a persistent runner sharing an account's secrets. The default remains
  serverless free CI precisely because it avoids this exposure.

**Failure is data, recorded per server — never a silent drop.** The original
pipeline deleted any JSON lacking a `metadata` key (1,684 of 2,038 survived),
conflating "unreachable," "garbage," and "valid-but-empty" and losing the record
of *why* a server dropped out. The rebuild writes a **`crawl_outcomes`** row for
**every** seed server: `server_url`, `http_status`, `outcome`
(`ok`/`unreachable`/`forbidden`/`invalid_response`/`empty`/`timeout`),
`attempts`, `latency_ms`, `response_sha256`, `error_class`. Nothing is deleted;
exclusion from the typed tables is a *derived consequence* of a recorded outcome.
The crawl→raw gate then asserts **count reconciliation**:
`seed_server_count == Σ(crawl_outcomes by outcome)` — silent attrition becomes a
reconciled, drift-detectable number. This is the single highest-value crawl-stage
graft from the rigor draft (all three judges named it).

**Full crawl per published snapshot; incremental only where it is cheap and
stateless.** This is the resolved conflict (see "Conflicts resolved"). Each
*published* snapshot is a **full, self-contained crawl** — stateless, robust,
nothing to reconcile or corrupt, trivially reproducible (the simplicity spine).
The full crawl is embarrassingly parallel and, matrix-sharded, is cheap in
wall-clock. What is made incremental is the **one genuinely expensive step,
embedding**: because IDs are deterministic and content-addressed (Dimension 3), a
snapshot's layer-text hashes diff cleanly against the prior snapshot's, so only
changed/new layers are re-embedded — and that diff is computed *between two
immutable snapshots*, carrying **no mutable standing crawl state.** Stateful
incremental *crawling* (carrying per-server `currentVersion` across runs) is named
as a future option only if crawl cost ever becomes the binding constraint; it is
deliberately not built first, because its state is the exact kind of thing that
rots unattended.

**Every crawl emits a typed `CrawlReport`-derived run manifest** (attempted /
reachable / quarantined, layers discovered / validated / embedded, restgdf
version, wall-clock, per-error detail). A run that loses more than a configured
fraction of previously-reachable servers is an upstream anomaly and **does not
publish** (Dimension 6). The crawl never mutates published data; it produces
candidate artifacts a passing gate promotes.

---

## Dimension 3 — Data model and schema

### A light medallion: immutable raw → typed core → consumer views

Keep the tiering idea; execute it as a **stateless three-layer transform**, not a
standing warehouse:

- **Raw (immutable, content-addressed).** The exact JSON returned per server plus
  HTTP metadata, stored as **per-file content-addressed objects with checksums**
  (not a monolithic `jsonfiles.tar.gz` with no index, as today). This is the
  replay/audit floor: any typed row traces to the exact bytes it came from, and
  any future re-flatten is possible without re-crawling. This is a light Bronze
  without warehouse apparatus (judge c's graft).
- **Typed core (validated).** `servers` / `services` / `layers` with **real
  types** and referential integrity, produced by parsing raw through Pydantic v2
  models. This is where the original pipeline's worst defect — `astype(str)` on
  all 205 columns — is fixed.
- **Consumer views (published Gold).** Derived *from validated core*: a typed
  bulk parquet, a Dataset-Viewer-compatible plain parquet, a geoparquet sibling,
  a slim search view, and the embeddings + ANN index — each a provable function
  of the same core revision, each manifested.

### What to type, and how deep (the resolved typing conflict)

Not "type almost nothing" (simplicity draft) and not "type all 205 columns"
(implied burden). The resolution (judge a's graft): **type the load-bearing
columns and promote the high-value nested structures; keep a complete
escape-hatch for the genuinely open-ended remainder.**

- **Typed, first-class:** the identity/join keys; the recovered facets
  (`jurisdiction_level`, `state`, `county`, `place`, `fips`, `agency`); geometry
  columns (below); license columns (Dimension 1); `service_type`, `geometryType`;
  and provenance (`last_crawled`, `snapshot_id`, `source_server_url`).
- **Promoted to typed Arrow structs:** the high-value nested objects consumers
  actually query — above all the **per-layer `fields` schema** (what columns a
  layer exposes — essential for `get_layer` and for discovery), plus `extent`.
- **Escape-hatch:** genuinely open-ended sub-objects no consumer queries
  structurally (`drawingInfo`, `timeInfo`, `templates`, …) are preserved as a
  **typed JSON string / struct column with a documented, versioned sub-schema** —
  a contracted JSON column, not an accidental one. The split between "promote"
  and "contract as JSON" is itself recorded in the schema registry. This keeps
  the full fidelity that makes the *full* dataset the right starting point,
  without a brittle 205-wide schema to maintain.

The **slim search view** (`id, name, type, description, url, metadata_text`,
geometry, plus the recovered facets) is a *derived `SELECT` over the typed core*,
never a parallel hand-built lineage — the original's two independent notebook
paths are exactly how the 560-row full-vs-slim discrepancy arose. This repo's own
Stage-0 corpus analysis (`docs/stage0/`, part of the separate modernization
effort — **not** the ecosystem recon) found **81% of `description` empty
(702,078 / 865,304)** and **3.4% carrying raw HTML (29,283 / 865,304)**; those are
encoded here as schema expectations and quality-gate bands.

**`metadata_text` is a first-class, specified, tested field — not an inherited
name.** It is the text every semantic query ranks over, and the recon found the
*original* construction logic could not be located in any repo
(`huggingface-findings.md` §6). Because 81% of `description` is empty, the
composition of `metadata_text` **is** retrieval quality, so the rebuild re-specifies
it explicitly rather than inheriting a black box:
- **Definition (versioned in the schema registry):** a deterministic function of
  named typed columns — `name`, `service_type`/`geometryType`, the parent service's
  name/description, the recovered `agency` and `jurisdiction`/`state` facets, and
  the promoted `fields` names — joined in a fixed order with a documented separator,
  with `description` appended **only when non-empty and after HTML→plaintext
  normalization** (Dimension 5's sanitization, applied at *build* time for the
  embedded/searched text; see the raw-HTML companion-column note there). The exact
  field list and join rule carry a `metadata_text_version` stamped in the manifest,
  because changing them changes every embedding and every ranking.
- **Oracle (Stage 2/3):** a golden fixture pins `metadata_text` byte-exactly for a
  set of representative layers (empty-description, HTML-description, rich-fields,
  minimal), so a silent change to the construction is a **failing test**, not an
  invisible retrieval shift. The retrieval oracle (Recall@k) ranks over this
  contracted text, and `metadata_text_version` is part of what the drift panel
  checks across snapshots.

### Deterministic, content-addressed IDs (a cited defect, fixed and tested)

The current parquet pipeline keys rows with Python's built-in `hash()` —
process-salted (`PYTHONHASHSEED`), so IDs are not stable across runs and cannot
diff snapshots. (Notably the *Mongo* path already used a better deterministic
`random.seed(url); uuid.UUID(...)` scheme — the right instinct existed in-house
and simply was not used for parquet.) The rebuild uses **content-addressed IDs
from a fixed, platform-independent BLAKE2b hash** over normalized natural keys
(stronger than bare `uuid5` — judge a/b/c all preferred it):

```
server_id  = blake2b(normalize(server_url))
service_id = blake2b(server_id + "/" + service_path)
layer_id   = blake2b(service_id + "/" + str(layer_index))
```

Tested as gates: **deterministic** (build-twice-assert-byte-identical on frozen
fixtures); **stable across snapshots** (an unchanged server keeps its `server_id`
quarter to quarter — the property that makes diffing and incremental embedding
correct); **collision-checked** (ID uniqueness per table); **referential
integrity 100%** (every `layer.service_id` and `service.server_id` resolves). The
`normalize()` rule (lowercase host, strip trailing slash, canonicalize
scheme/port) is **versioned in the schema registry and stamped in the manifest**,
because changing it changes identities and must be a deliberate, version-bumping
act.

### Geometry: keep the bbox extent, but honest and lossless

The current `geometry` is the layer's **bounding-box extent reprojected to
EPSG:4326** — a rectangle, not features — honestly described in recon but
dishonestly *named* in the data. Do **not** chase real feature geometry (it would
be terabytes, duplicate live servers, and walk into the redistribution/copyright
question). Fix the naming and the losslessness instead (geospatial draft's
graft, endorsed by all judges):

- **Name it truthfully:** a STAC-style `bbox` (`minx, miny, maxx, maxy` doubles)
  plus a derived `extent_4326` polygon, documented as *the layer's declared
  extent, not its features*.
- **Preserve native extent + native CRS** (`extent_native`, `spatial_reference`)
  beside the 4326 extent, so reprojection is reproducible and auditable — the
  original discarded the native extent, making the transform unverifiable.
- **Losslessness — retain, don't drop.** A layer whose extent fails reprojection
  or is degenerate keeps its row with `geometry_valid=false`, a **reason code**,
  and its retained native-CRS extent — instead of the silent discard that lost
  560 rows (865,864 → 865,304) with no record of which or why. Losslessness is a
  hard requirement for honest cross-snapshot diffing.
- **Capture `geometryType`** (`esriGeometryPoint`/`Polyline`/`Polygon`/…) as a
  typed attribute at zero storage cost — high-value for discovery ("polygon
  layers covering X") — and store a **centroid + geohash** for cheap spatial
  pre-filtering.

Honest limitation, stated plainly: bbox extents are **coarse** — a statewide
layer's extent intersects every county query. That is exactly why the recovered
jurisdiction/FIPS facets (Dimension 1) are not optional: the precise answer to
"layers covering this county" is **FIPS/administrative filter first, extent
intersection to refine**, never extent alone.

### TIGER is a pinned, versioned external data dependency (not a free primitive)

FIPS/county recovery — the highest-leverage new facet — is a spatial join of each
layer's bbox centroid against **Census TIGER county/place polygons**. TIGER is a
real, versioned, external geospatial dependency and the plan treats it as one:
- **Pinned and manifested.** A specific **TIGER vintage** (e.g. `tiger_2024`) is
  pinned and recorded in the manifest (`geometry.tiger_vintage`), exactly like
  `restgdf_version` and `embedding_model + revision`. TIGER re-releases annually
  and place/county boundaries change, so two snapshots built against different
  vintages can assign a **different `fips`/`county` to an unchanged layer** —
  silently breaking cross-snapshot determinism *at the facet layer* with no
  content-ID change to signal it. Pinning the vintage and stamping it makes a facet
  change a **deliberate, manifested act**, and the drift panel flags a vintage
  change as a known cause of facet churn.
- **Covered by the determinism oracle.** The build-twice-identical oracle
  (Dimension 7) pins the TIGER vintage as a frozen input, so "same core + same
  TIGER vintage ⇒ byte-identical `fips`/`county`" is actually asserted, not assumed.
- **Costed, not absorbed into "$0."** The TIGER download (national county+place is
  a bounded, cacheable few-GB) and the **~3.5M point-in-polygon joins** are a real
  transform step with a real memory/time footprint; they are part of the Stage-2
  transform costed in Dimension 9, run on the same job that produces the typed
  core, and are a candidate for the state-partitioned parallelism the transform
  already uses.

### Schema/contract registry

A versioned schema registry lives in the pipeline repo: Pydantic v2 record
contracts plus columnar schemas (Arrow / `pandera`), each carrying a
`schema_version` stamped into every manifest. A breaking change bumps the version
and is a deliberate, reviewed act. A consumer six snapshots out knows exactly
what shape it is loading — something no current artifact provides.

---

## Dimension 4 — Storage and versioning

**One public dataset repo, snapshots as tags.** Not repo-per-year (fragments
DOIs, discovery, consumer pins) and not an unbounded `snapshots/` directory
(judge b's graft). A single `govgis` HF dataset repo; each refresh publishes an
**immutable, dated snapshot as a git tag** (`snapshot-2026-10`), with a moving
`latest` ref for consumers who want current data without pinning. Revision
pinning already demonstrably works (this Space pins the dataset at a SHA today).
One repo, one card, one DOI lineage, one thing to reason about.

**Precedent check — Major-TOM.** The recon surfaced `Major-TOM`
(`huggingface-findings.md` §1) as a mature geospatial-dataset org the maintainer
already belongs to, structured as **multiple companion datasets + a viewer Space**
— exactly the companion-artifact topology being designed here. Two patterns are
worth borrowing and one is not: borrow its **consistent grid/partition convention
across companion datasets** (mirrored here by the state-partitioned parquet and a
shared schema registry) and its **separate viewer Space over the data** (mirrored
by keeping the Gradio Space a thin consumer). Do **not** borrow its
**repo-per-modality sprawl** — Major-TOM is a large multi-contributor EO-imagery
org whose many-repo split is justified by scale this single-maintainer,
metadata-only project does not have; here the same "companion artifacts" live as
**separate files/tags within one repo** (Dimension 8's three-standing-surface
count), which is the simplicity-spine choice. The precedent validates the
companion-dataset-plus-viewer shape while its scale difference explains why the
repo *count* stays low.

**Embeddings in their own file (or companion repo), separate from metadata, in a
named non-pickle format.** So a re-embed (new model) does not churn the metadata
tables, a metadata-only consumer skips multi-GB of vectors, and the metadata tables
preview in the Dataset Viewer even though the embedding file does not. Given the
~4–14 GB vector footprint at true scale (Grounding Correction 2), this separation
is a real cost lever, not tidiness. The **format is specified, not left implicit**,
so the plan's "no pickle" posture actually holds through the vector path: vectors
ship as **Arrow/parquet `FLOAT` arrays or `.safetensors`** (or `.npy` explicitly
loaded with `allow_pickle=False`) — never a pickle-backed `.npz`/`allow_pickle=True`
path that would re-introduce arbitrary-code-execution on load. Row order is the
stable content-addressed-`id` order (the Xet-dedup requirement above), so the
embedding file is join-aligned to the metadata by position and stable across
snapshots.

**A checksummed manifest with a lineage chain — nothing like it exists today.**
Every snapshot ships a `manifest.json`, itself checksummed (its SHA-256 is the
snapshot's release identity):

```
schema_version
schema_registry_version         # the Pydantic/Arrow contract revision
id_normalize_version            # the URL-normalization rule feeding the BLAKE2b IDs
metadata_text_version           # the metadata_text construction rule (Dimension 3)
snapshot_id                     # e.g. "2026-10"
created_utc
pipeline_git_sha                # exact code that produced this
restgdf_version
tiger_vintage                   # pinned Census TIGER release feeding FIPS/county
embedding_model + revision + dim + distance_metric
seed:      { sources: [ { name, source_url, fetched_utc, sha256, server_count } ],
             total_unique_roots }              # a list, so a federal complement adds a row
crawl:     { attempted, reachable, quarantined, success_rate, report_sha256 }
counts:    { servers, services, layers }
license:   { elfelt_terms_ref, dataset_notice_ref, commercial_use_authorized,
             status_distribution }             # NOTICE governs the whole product
geometry:  { valid_rate, native_crs_distribution, tiger_vintage }
files:     [ { path, bytes, sha256, rows, schema_ref } ]
citation:  { snapshot_tag, manifest_sha256, repo_doi, citation_cff_ref }  # see below
prior_snapshot: { snapshot_id, manifest_sha256 }   # the tamper-evident chain link
diff_summary:   { servers_added/removed, layers_added/removed/changed }
validation:     { gate_results{...}, all_passed: true }   # publish impossible if false
```

The `prior_snapshot` link makes the snapshot sequence a **verifiable, tamper-
evident chain** (judge a/c graft). A consumer loads the manifest first, verifies
checksums and counts before trusting a byte, and fails closed on mismatch — the
direct replacement for today's "deserialize a 4.28 GB pickle-like blob and hope."

**A per-snapshot `diff/` artifact** (added/removed/changed servers, services,
layers vs. the prior snapshot) — computable *only because* IDs are deterministic
— is the substrate for both drift detection (Dimension 6) and an agent-facing
`whats_changed` capability (Dimension 5). No one in this space ships a queryable
dataset changelog; this does.

**Xet, not plain LFS — and a stable file layout that makes its dedup actually
fire.** Both current repos are plain LFS only because they predate Xet. A snapshot
pipeline re-uploads near-duplicate multi-GB files every refresh — the textbook
case for Xet's content-defined chunking/dedup — so Xet is the right choice over
LFS regardless. But the *headline* "each new snapshot's marginal cost ≈ the changed
delta" (Dimension 9) is not free: content-defined chunking only dedups a re-written
parquet where **unchanged rows stay chunk-aligned**, which requires the pipeline to
**write rows in a deterministic, stable order** (sort by content-addressed `id`)
and **hold row-group boundaries stable** across snapshots, so a re-embed or a
metadata edit that touches a minority of rows does not reorder the file and defeat
alignment. This ordering is a **build requirement**, not an incidental property.
Two honest caveats the earlier framing skipped: **float embedding vectors do not
dedup like text** (a changed vector is fully novel bytes), so the dominant
embeddings file dedups mainly by *unchanged rows staying byte-identical and
aligned*, not by near-match compression; and the **actual dedup ratio is
unquantified until measured** on real successive snapshots — the Stage-4 gate
records observed post-Xet incremental upload size so the cost model is grounded in
a number, not an assumption.

**Dataset-Viewer compatibility (the `.geoparquet` gap).** `.geoparquet` is
unsupported by the Viewer and the parquet-conversion bot (`huggingface/datasets#6438`,
filed by this maintainer in 2023, still open). Ship the **primary** table as
**plain `.parquet`** — geometry as WKB/WKT string plus split
`minx/miny/maxx/maxy` float columns — so the Viewer renders it and
`datasets`/`polars`/`duckdb` load it natively. A `.geoparquet` sibling is kept
for GeoPandas/PostGIS users. This is a **publish-gate requirement**: a snapshot
without a Viewer-loadable plain-parquet sibling does not pass publish. Shipping
plain parquet first is also what makes the DuckDB serving path (Dimension 5)
zero-conversion.

**DOI, citation, and superseding the old repos.** The existing repos carry
auto-assigned DOIs (`10.57967/hf/1368`, `1369`). The old repos are **frozen, never
deleted** (they are cited externally — recon found `Lexicom7/EO_Datasets`
referencing `govgis_nov2023` by name — and DOIs must resolve permanently). When
their cards are edited to add the **"superseded by → `govgis`"** banner (a change
already being made, so the incremental cost is near zero), the **license/terms
metadata is corrected in the same edit**: add the Elfelt-derived-content `NOTICE`
(commercial use needs written permission; free derivatives welcomed) that the
original blanket `license: mit` omitted. An irrevocable MIT grant on already-
downloaded copies cannot be retracted, but the corrected card stops *new*
consumers from relying on the wrong terms and records the maintainer's actual
position — the honest, low-cost half of a fix that is otherwise impossible to make
fully retroactive.

Per-snapshot citation is designed, not assumed, because the earlier "each snapshot
tag is independently citable via its manifest" glossed a real mechanism gap: **HF
DOIs are repo-level and auto-assigned, not minted per git tag**
(`huggingface-findings.md` §2/§8). The concrete scheme:
- **The `govgis` repo gets one repo-level DOI** (the stable "cite the dataset"
  identifier), with a "supersedes `10.57967/hf/1368`, `1369`" note on the card.
- **A specific snapshot is cited by (repo DOI + immutable snapshot tag +
  `manifest_sha256`)**, and a **`CITATION.cff`** in the repo carries the citation
  template; each snapshot's manifest records this triple in its `citation` block
  (above). This is a real, resolvable, verifiable per-snapshot citation without
  pretending HF mints a DOI per tag.
- **If true per-snapshot DOIs are ever required** (e.g. a paper cites an exact
  snapshot and a reviewer demands a DOI), the escape hatch is a **Zenodo/DataCite
  deposit of that snapshot**, which does mint a versioned DOI — a recorded option,
  scoped as needed, not standing infrastructure.

---

## Dimension 5 — Serving / access architecture

**Principle: one validated snapshot, several optional consumer-specific views;
no consumer keeps its own copy again.** The source of truth is the pinned,
manifested dataset; serving surfaces are thin, independently swappable views over
it, each verifying the manifest before serving. This is where the biggest ops
lever sits, and it is the sharpest resolved conflict.

**The primary "serving layer" is the dataset repo itself.** A well-formed parquet
snapshot on the Hub is already a queryable API: `datasets.load_dataset`,
`polars.read_parquet`, `duckdb` over the Hub URL, or a plain download all work
today with zero code the maintainer keeps alive. For bulk/analytical consumers
that is the entire answer, and it is the lowest-maintenance service possible
because someone else operates it.

**Search default: hybrid retrieval, vector demoted to a re-ranker.** Grounded in
the 81%-empty-`description` fact, a vector-only index over a mostly-blank field is
weak. Wherever a live search tier runs, the query model is **spatial predicate
(bbox intersect) → structured/FIPS/jurisdiction/license filter → full-text (over
name + description + field names + parent service) → optional dense-vector
re-rank of the survivors.** Vector similarity refines an already-relevant,
already-in-region candidate set instead of being the first and only filter. This
is the geospatial draft's thesis, grafted by all three judges, and it serves
ops-minimalism because the *filtering* front — the bulk of the work — is one
embedded library (DuckDB), with the vector step delegated to a single quantized
FAISS index rather than a second standing service. Because vectors only ever
re-rank an already-narrowed candidate set (hundreds-to-low-thousands of rows),
this surface never needs a resident 14 GB ANN index at all — that requirement is
specific to the vector-first Space, not the hybrid tier (Grounding Correction 2).

**Recommended read engine: embedded DuckDB for spatial + FTS + structured
filtering, with FAISS owning the quantized vector index.** The hybrid query model
splits cleanly by engine, and the split is deliberate — one embedded engine does
*not* do everything, because the one thing DuckDB cannot do at this scale is the
quantized ANN index the feasibility analysis makes mandatory (Grounding
Correction 2). Justified as a real tradeoff, not novelty:
- **DuckDB owns spatial + full-text + structured filtering.** It is a library:
  the service process opens the pinned plain-parquet directly and serves, with **no
  always-on database to patch/back-up/pay-for-while-idle** and no load-into-Postgres
  step on every deploy (the sibling's `load_data.py` reads a 5.66 GB geoparquet into
  memory and `COPY`s it on init); promoting a snapshot is just re-pinning a
  revision. Its spatial extension gives `ST_Intersects` bbox pre-filtering and its
  FTS extension the keyword layer — exactly the front of the hybrid pipeline, which
  narrows ~3–4M rows to a candidate set before any vector work.
- **FAISS owns the vector index — not DuckDB VSS.** DuckDB's VSS builds a
  full-precision float32 HNSW over `FLOAT[N]` columns and supports **neither SQ8/PQ
  nor IVF**; a VSS index over 3.5M × 1024 is the ~14 GB in-RAM footprint that does
  not fit `cpu-basic` (Grounding Correction 2), and its persistent HNSW historically
  needs an experimental flag. The **quantized** ANN index the plan requires is
  therefore a **FAISS** artifact (IVF/PQ/SQ8), built as a snapshot file. This keeps
  the engine count honest: **one FAISS index serves the vector-first Space; the
  DuckDB hybrid tier delegates any at-scale vector step to that same FAISS index**
  and re-ranks small already-filtered candidate sets by brute force — never a second
  ANN structure over the same vectors.
- **On the "both current stacks lack an ANN index" claim — retracted as
  unsourced.** Earlier drafts asserted the sibling does a full sequential scan with
  no index. The recon actually characterizes the sibling as loading into a
  "**spatially- and vector-indexed** `layers` table" (`github-findings.md`), and
  pgvector supports both exact and ANN (ivfflat/HNSW) search — the sibling's actual
  index DDL is not in the recon, so this plan does **not** assert its absence. The
  motivation for a first-class quantized ANN index here stands on its own (the
  `cpu-basic` fit at 3–4M, Grounding Correction 2), not on a claim about the
  sibling that the recon contradicts.

**The agent tier: a composable, typed, provenance-carrying MCP tool set — as an
*optional* consumer, not always-on core.** The sibling exposes one
`gis_layer_search` tool that markdownifies every field. Replace it (all judges'
graft) with a small chainable set returning **structured JSON, not markdown
blobs**:

- `search(query, filters?, k?)` — semantic + *structured* filters
  (`jurisdiction_level`, `state`, `county`, `agency`, `service_type`,
  `geometry_type`, `bbox`/`point`, `license_status`), returning ranked typed
  `LayerResult`s each with a stable `id`, the facets, the extent, and a
  `citation` (source URL + snapshot id).
- `get_layer(id)` — full typed drill-down (the chaining primitive; the sibling
  has none).
- `list_facets(facet)` — enumerate real values for `state`/`county`/`agency`/…,
  specifically to **stop agents hallucinating filter values** (an agent lists
  real agencies, then filters on one that provably exists). A correctness
  feature, not a convenience.
- `get_live_endpoint(id)` — the `https`-only live ArcGIS URL (validated against the
  SSRF controls below, not merely the scheme) plus the service's declared license,
  so the agent fetches **real features from the authoritative source under that
  source's terms.** This is where "we index, we don't redistribute" (Dimension 1)
  pays off. Note the returned URL is untrusted crawl data: if the *server itself*
  dereferences it (the Full tier's live-feature fetch), the SSRF controls below are
  mandatory, not optional.
- `whats_changed(since_snapshot)` — surfaces the `diff/` artifact; a genuinely
  new capability enabled by deterministic IDs and successive snapshots.

**Serving-tier safety and abuse controls are in the contract**, because retrieved
government metadata is untrusted LLM input and the serving tier makes outbound
requests to untrusted URLs. This mirrors the modernization plan's threat model,
extended to the surfaces the rebuild adds:
- **Output safety.** Retrieved text is delimited and labeled *data, not
  instructions*; all URLs (in records **and** in generated text) pass an
  `http`/`https` allowlist; results are typed JSON with explicit fields, not
  markdown-flattened.
- **Raw HTML is neutralized at *publish* time, not pushed onto every consumer.**
  3.4% of `description` values carry raw HTML; if the published parquet redistributes
  it unmodified it flows into `metadata_text`, the embeddings, and every
  downstream `load_dataset`/polars/DuckDB user — re-creating the exact XSS class the
  modernization plan fixed, now with no regression signal. So the pipeline ships a
  **`description_plaintext` companion column** produced at build time (HTML→text),
  and it is `description_plaintext` (not raw `description`) that feeds
  `metadata_text` and the embeddings. The raw `description` is retained for
  fidelity but is never the field a consumer is expected to render, and the card
  documents that. The drift panel's HTML-rate band (near 3.4%) is a *parse-regression*
  check, explicitly **not** a sanitization control — the companion column is the
  control.
- **Serve-time outbound fetches get full SSRF protection, not just a scheme
  check.** `get_live_endpoint` returns a URL sourced from the untrusted crawl, and
  the Full tier's on-demand live-feature fetch actually *requests* such URLs
  server-side. A hijacked-since-crawl domain or planted seed can yield
  `https://169.254.169.254/`, `https://[::1]/`, or `https://internal.corp/` that a
  scheme allowlist passes. The serving tier therefore applies the **same IP
  deny-list, redirect pinning, and DNS-rebinding protection as the crawl
  (Dimension 2)** to every outbound request, plus a response-size cap — the
  metadata endpoint and private ranges are blocked, re-validated per redirect hop.
- **Abuse / availability controls (when a live tier runs).** "Optional and
  sleep-when-idle" is a *cost* property, not an abuse control. The DuckDB
  search + MCP endpoint over ~3–4M rows is a compute-heavy target, so the live tier
  ships a **per-request timeout, a concurrency cap, and a simple rate limit** on the
  ANN+spatial+FTS path (the read-only, no-per-call-dollar nature keeps this low-risk
  — the worst case is degraded availability that self-heals on idle, not data or
  cost exposure — but the cap is present, not assumed away by "it sleeps").
- **Deserialization/supply-chain surfaces are named, not left latent.** The
  **embeddings file format is specified** (Arrow/parquet `FLOAT` arrays or
  `.safetensors`/`.npy` loaded with `allow_pickle=False`) so the "no pickle"
  headline actually holds through the vector path, and the DuckDB `spatial`/`fts`
  extensions are **version-pinned and checksum-verified** at load (INSTALL/LOAD
  pulls a binary), not taken from a floating latest.

**Decouple data-publish from serving-promote** (all judges' graft). Publishing a
snapshot tag and *promoting* the serving layer to it are **two explicit steps**:
a bad snapshot never auto-reaches production; promote is a separate pin bump with
its own go-ahead.

**This repo's Gradio Space** becomes a **thin consumer of a checksummed native
index** (the modernization plan's Stage-2 target layout: `index.faiss` + typed
`documents.parquet` + manifest, no pickle) built as a snapshot artifact — not the
place the index is mysteriously produced. If an MCP surface is wanted with
minimal infra, Gradio can expose one natively on that same sleep-when-idle Space,
rather than a second standing deployment.

**The one seam, sequenced explicitly — because both efforts touch the Space's
index.** This plan meets the separate modernization effort at exactly this point,
and the ordering must be stated rather than left to collide. The modernization
effort produces its *own* `index.faiss` at its Stage 2 from the **existing
Nov-2023 data**, to ship the Gradio migration on its own track *before* any
rebuild snapshot exists. This rebuild's Stage 5 later re-points the migrated Space
at the **rebuild's** checksummed native index once a rebuilt snapshot is published.
The contract: **the modernization effort owns the index format and the Space's
consumption contract** (`index.faiss` + `documents.parquet` + manifest); this
rebuild **conforms to that format** and only swaps the *source artifact* from the
Nov-2023-derived index to a rebuilt-snapshot index. Rebuild Stage 5 therefore
**presupposes the migrated Gradio app has shipped** and is a no-op if it has not
(it degrades to "publish the index in the agreed format; the Space adopts it
whenever it migrates"). If both a modernization index and a rebuild index are
ready simultaneously, the **rebuild snapshot wins** (it is the fresher data in the
same format), adopted via the normal decoupled publish→promote step — never two
indices contending silently. Since both efforts are the same maintainer, this is a
sequencing note, not a cross-team negotiation, but it is written down so Stage 5
is not ambiguous.

**The sibling Postgres+pgvector+FastMCP stack: kept as a documented, user-run
fallback — not operated as core.** This is the resolved build-vs-avoid conflict.
The 9–10-month security-PR backlog under exactly these single-maintainer
conditions is disqualifying evidence for making it core infrastructure. But
excluding it costs almost nothing: the geoparquet it consumes is a byproduct this
pipeline publishes anyway, so a power user who wants in-database spatial+vector
queries can `docker compose up` against the published snapshot exactly as today.
It is also the documented **escalation path** for the day real concurrency/QPS
outgrows an embedded engine — a reversible redeploy, not a rewrite, because a thin
`govgis-core` retrieval interface abstracts the engine. We keep the capability
and drop the standing burden.

**STAC** is a worthwhile *optional* interop face (server → Catalog, service →
Collection, layer → Item with bbox + properties + a live-endpoint link) but is
not core; prototype one Collection before committing (judge b's caveat), since a
live service is a slightly unconventional STAC asset.

---

## Dimension 6 — Refresh cadence and automation

Nothing like this exists anywhere in the ecosystem today. It is built from
scratch, shaped by the single-maintainer reality: **low-touch, fail-closed,
trivially reversible, safe to neglect.**

**Cadence and compute split.**

- **Weekly seed-delta check (near-free) — and it must be loud on the founding
  failure mode.** A light GitHub Actions cron fetches the `.txt` mirror, checksums
  it, parses it, and diffs the server roster against the last snapshot's committed
  seed. (This automated *repeated* fetch is one of the things the Stage-0 Elfelt
  gate must authorize — see Dimension 1's decision branches; if Elfelt declines
  automation, this cron is disabled and the delta check becomes a manual step.)
  Outcomes and the **notification each produces**:
    - **Growth** (drift > a threshold, e.g. >10% new roots): **opens a GitHub
      Issue** — "the upstream list grew materially; a re-crawl may be worthwhile."
    - **Upstream death or breakage — the exact failure that founded this project**
      (the `.txt` 404s, moves, is truncated, changes format so the parser yields an
      implausible roster, or returns byte-identical for many weeks implying a stale
      mirror): **also opens a GitHub Issue** (not just a red Actions run), because a
      silently-red job on a repo checked every six months is exactly how a dead feed
      goes unnoticed. The check treats "I could not get a sane, fresh seed" as a
      first-class alert, distinct from "the seed grew." A red run alone is not the
      signal; the Issue is.
  It does not crawl, publish, or page anyone. This is the primary monitoring
  system: near-zero standing infrastructure, converting "is it time to refresh?"
  and "is the upstream still alive?" from things the maintainer must remember into
  things that appear in their inbox when true (judge b's graft) — provided the cron
  itself is still alive, which is the liveness problem addressed below.
- **Quarterly full snapshot (automated crawl+build, human go/no-go to publish).**
  The heavy pipeline. Crawling and building are reversible and automatable;
  publishing a public, DOI'd artifact is an outward-facing act that gets an
  explicit sign-off.

**Automation liveness — the design is neglect-*safe* only if the automation
survives neglect.** Two failure modes would silently defeat the whole monitoring
story, and both are addressed explicitly:
- **GitHub auto-disables `schedule` workflows after 60 days of repository
  inactivity.** The spine of this plan is "untouched for six months," and the
  weekly cron never commits anything (it only opens Issues), so with no push
  activity GitHub would **disable the cron at ~day 60** — one-third into the very
  window it exists to cover. This is a documented, non-optional GitHub behavior and
  it is fatal to a "cron is the monitoring system" design left as-is. The fix is a
  **keepalive that produces real repo activity**: the weekly job **commits a small
  heartbeat** (a `last_check.json` recording the fetch timestamp, seed checksum, and
  roster count) on every run. That commit is itself repo activity that re-arms the
  schedule, *and* it doubles as the state below. (A `workflow_dispatch` re-arm or an
  external scheduler is the fallback if committing is undesirable, but the heartbeat
  commit is the simplest self-sustaining choice.)
- **Nothing today fires on the *absence* of a run — a dead-man's-switch is
  required.** All other notifications come from inside a run that reaches a gate; if
  the cron is disabled, a runner OOMs before the gate, an HF Job wedges, or the
  workflow simply never triggers, **no signal is emitted at all**, and GitHub's
  default scheduled-failure email (to the last committer only) stops once the cron
  auto-disables — the fallback degrades in lockstep with the primary. So the design
  adds a **last-successful-check / last-successful-snapshot age signal**: the
  heartbeat commit's timestamp is the ground truth, and a **cheap external uptime
  ping** (a free dead-man's-switch service, or a second lightweight scheduled check
  hosted independently) alerts the maintainer if the heartbeat is older than, e.g.,
  two weeks. "No new data" must be *observable as* "the machine stopped," not
  indistinguishable from "nothing changed." This is the one piece of standing
  external dependency the design accepts, because a monitoring system that cannot
  report its own death is not a monitoring system.

**The crawl execution, concretely (closing the judges' shared execution gap).**
A ~8–9h full crawl exceeds GitHub Actions' ~6h job ceiling, so the crawl is
**matrix-sharded across parallel free runners** — partition seed roots by
`state` (natural, and aligns with per-host politeness) or by a stable hash into N
shards, each comfortably under the ceiling, each writing its shard's raw output
as a job artifact, followed by one lightweight combine job. This keeps everything
inside free managed CI with **no self-hosted runner.** N is **measured, not
guessed**: the Stage-1 gate crawls one representative state shard, records
wall-clock and layers/server at *current* scale, and computes N to keep each
shard under ~2–3h with margin. If a single shard proves pathological (a few
enormous servers), those roots move to their own shard. A self-hosted runner /
HF Job / ephemeral VM remains the documented fallback if sharding proves
insufficient (subject to Dimension 2's ephemeral/credential-stripped requirement),
but the serverless matrix is the default.

**Shard loss is expected and handled — a single failed leg is not a full re-crawl.**
Across N shards touching thousands of flaky government hosts over ~8–9h, individual
shards *will* fail, so the design specifies the behavior GitHub's defaults get
wrong:
- **`fail-fast: false` on the matrix**, so one shard's failure does not cancel its
  siblings (the default `fail-fast: true` would throw away hours of good work).
- **Immutable, content-addressed raw makes a lost shard a targeted re-run.** Each
  shard's raw output is content-addressed (Dimension 3), so re-running only the
  failed shard is idempotent and cheap — the completed shards' artifacts are reused,
  not recrawled. This is the shard-level resume the ~8–9h runtime demands; without
  it a single dropped leg would force a full non-idempotent re-run.
- **The combine job proceeds but records the gap; the gate decides.** A missing
  shard is not silently absorbed: the combine job records which shards are present,
  and the **`seed_count == Σ crawl_outcomes` reconciliation gate** (Dimension 2)
  catches the shortfall at publish-readiness — so a dropped shard **blocks publish**
  rather than shipping a quietly-truncated snapshot. The maintainer then re-runs
  just the missing shard(s) and re-combines.
- **Combine-job disk fits the runner.** Raw output is multi-GB (~3–4 GB compressed
  at true scale) and the standard runner has only ~14 GB disk, so the combine job
  **streams each shard's raw artifact straight into the private raw archive
  (Dimension 8)** and carries forward only the checksums/index it needs, rather than
  materializing every shard on local disk at once. Peak combine-job disk is a
  Stage-1-measured quantity, like the shard timing.

**The embedding execution, concretely — including the CI→GPU boundary the earlier
drafts treated as a black box.** Embeddings run on **pay-per-use HF Jobs GPU** (the
maintainer already uses `hf jobs`), **only for changed/new layers** (diffed via
deterministic text hashes against the prior snapshot). A full re-embed (~0.5–2
GPU-hours of inference, ~1–3h wall-clock, low-tens of dollars) runs only on a model
change or the first snapshot; steady-state refreshes re-embed the changed minority
for cents-to-dollars. CPU embedding on free Actions was considered and rejected —
3.5M × bge-large on CPU blows the 6h ceiling; a few dollars of serverless GPU is
the lower-*burden* choice. Crossing the boundary is designed, not assumed:
- **Auth.** The Actions job launches the HF Job with an **`HF_TOKEN` GitHub secret
  scoped to job submission** (see "Secrets and the CI trust boundary" below), never
  the publish-write token — the embedding step does not need write access to the
  public dataset repo.
- **Async, not a blocking wait.** The Actions job **launches the HF Job, records its
  job id** (the maintainer's own standing rule for `hf jobs`), and **does not block**
  for the 1–3h run — a blocking wait would reconsume the very 6h Actions ceiling the
  crawl was sharded to avoid. It either polls with a short-lived watcher or, cleaner,
  a **follow-up workflow re-triggers on job completion** to pick up the embeddings
  artifact and continue to index build.
- **Failure/retry across the boundary.** A crashed, OOM, or partially-completed GPU
  job is a **recorded failure that blocks the snapshot**, not a silent gap: the
  embedding-integrity gate (vector count == row count) catches a partial run, and
  the changed-rows diff makes a **re-run idempotent** (only the still-missing rows
  re-embed). The job id is retained so a wedged job can be cancelled — never left
  running unmentioned.
- **Index build co-locates here.** As Grounding Correction 2 notes, the quantized
  FAISS index is built on this same GPU step where the vectors are already resident
  — no second data movement, and the ~14 GB flat vectors never touch the 16 GiB
  Actions runner.

**Secrets and the CI trust boundary — least-privilege, isolated, and leak-gated.**
The pipeline provably needs credentials in CI: an **HF write token** to publish
snapshot tags and Xet uploads, a **private-archive write credential**, and an
**HF Jobs GPU credential** to run embeddings. The earlier drafts never mentioned
them; a CI-automated public *write* path with an unscoped token is a larger, less
hardened surface than the session-scoped BYOK model the modernization plan already
fixed. The model here:
- **Least-privilege, distinct scopes.** The publish-write token is scoped to the
  `govgis` dataset repo only; the archive credential to the private raw archive
  only; the HF Jobs credential to job submission only. No single account-wide token.
- **Crawl/publish isolation — the load-bearing one.** The **untrusted-content crawl
  job runs restgdf/stamina/aiolimiter against thousands of hostile-capable hosts and
  must NOT share a scope with the publish-write token.** Crawl output lands in the
  archive via a crawl-scoped credential; publishing the *public, externally-cited*
  dataset happens in a separate, later job holding the write token, after gates pass.
  A compromised dependency in the crawl job then cannot tamper with the published
  dataset.
- **A secret-leak gate, ported from the modernization plan's Stage-4.** A
  forced-failure test asserts no substring of a test credential appears in captured
  logs, exceptions, or telemetry — the rebuild keeps *both* halves of the
  modernization plan's output-safety story (URL/HTML safety **and** secret-leak),
  not just the first.

**Data-quality gates block publish — the safety net that makes neglect safe.**
Before any snapshot is tagged and published, all computed by script and recorded
in the manifest (the fail-closed **drift panel**, compared against the prior
published snapshot):

- **Crawl-coverage floor + reachability delta** (≥ X% of seed roots crawled;
  a sudden 20-point reachability drop is a broken crawler/network, not a
  real-world event → fail).
- **Count reconciliation** (`seed_count == Σ crawl_outcomes`) and **volume drift,
  as an *asymmetric* band with a growth-vs-runaway discriminator.** A symmetric
  "within X% of prior" band would false-positive-block exactly the case this
  project exists to capture: the **first fresh crawl is expected to be ~4× the 865k
  anchor** (Grounding Correction 2), and even steady-state the list grows most
  weeks. So the band is asymmetric — a **collapse** (e.g. a 40% *drop*, a truncated
  crawl) fails hard, while **growth is checked for *plausibility*, not bounded to
  the prior count**: layer growth must **track seed-roster growth** (the weekly
  seed-delta already measures new roots; layers/server should stay within its
  historical range), and a jump in row count **without** a corresponding rise in
  crawled servers is the runaway/duplication signal that fails. Because the gate is
  **fail-closed it is also fail-safe**: a false positive blocks *publish* only,
  which is a human go/no-go step anyway (below), with the documented
  quarantine-and-recorded-override path — the worst outcome is a human-cleared false
  alarm, never a silent block or bad data. The first snapshot, having no prior,
  gates against the Nov-2023 anchor with the growth expectation stated, exactly like
  the retrieval gate's first-snapshot floor. Concrete thresholds are `e.g.`
  starting points, tuned from the first measured snapshots, not final.
- **Schema conformance** (every row validates; Pydantic pass-rate above a floor;
  the rest quarantined) and **null-rate/known-shape bands** (empty-`description`
  near the ~81% baseline, HTML near ~3.4%; a wild deviation is a parse
  regression).
- **Determinism** (ID uniqueness; stable-ID overlap with prior within
  expectation — near-total ID churn means the normalization/hash changed
  unexpectedly) and **referential integrity 100%**.
- **Geometry validity rate** within bounds (reprojection failures counted and
  retained, not dropped).
- **License distribution** within band (a sudden swing in `federal_public_domain`
  share is a seed-classification regression). This is a *drift* check and is
  **necessary but not sufficient** — a classifier that is wrong-from-day-one
  produces a *stable* distribution and sails through it. The correctness of the
  `license_status` enum is pinned separately by a **known-answer oracle** at Stage 2
  (below), because this column gates redistribution scope and a silent misclassification
  leaks restricted data.
- **Embedding integrity** (vector count == row count, correct dims, normalized,
  no NaNs) and **retrieval parity** (Recall@k on the frozen golden query set
  ≥ the prior snapshot's, computed in-run).
- **Checksums** generated and written to the manifest.

A failed gate means the snapshot is **not published**, an Issue is opened with
the failing gate, and the last good snapshot stays live and pinned. The worst
case for a six-month-unattended pipeline is "no new snapshot + one Issue," never
"a broken snapshot consumers pinned to." **Publish and serving-promote are
separate explicit steps**, so even a published-but-imperfect snapshot never
auto-reaches production.

---

## Dimension 7 — Staged implementation plan

The method mirrors `docs/modernization-plan.md`'s staged-gates-with-oracle
pattern (proven on this same data, today). Each stage has actions, a **gate**
(scripted, fail-closed assertions), rollback thinking, and the test oracle it is
accountable to. Nothing advances on a green build report alone — the coordinator
re-runs the actual gate commands. Immutable snapshots make rollback nearly free:
"roll back" always means "re-pin to the previous tag," and old repos are never
deleted. Stages that mutate external state (HF publish, serving promote, anything
touching Elfelt) get an explicit human go-ahead.

### The test oracle (defined once, used by every stage)

- **Golden fixtures — captured from the real producer, never hand-authored.** A
  small set of real ArcGIS server JSON responses spanning the tricky cases (a
  healthy FeatureServer, a MapServer, one declaring `licenseInfo`, one with a
  reprojection-failing extent, one with HTML in `description`, an empty one),
  captured from live `restgdf` output so a green test cannot pass on a fictional
  shape. Each fixture carries its **known-answer expectations** — including the
  correct `license_status` enum and the byte-exact `metadata_text` — not just a
  valid *shape*.
- **`license_status` known-answer oracle (a legally load-bearing classifier gets a
  correctness test, not just a distribution band).** Because `license_status` gates
  redistribution scope — `federal_public_domain`/`declared_open` redistributed in
  full, `unverified_state_local`/`declared_restricted` link-only — a systematically
  *wrong-but-stable* classifier would pass every schema and drift check while leaking
  restricted data. So a labeled fixture set with **hand-verified correct enums**
  (a federal server, a state server with no declared license, one with an explicit
  `licenseInfo` open license, one with restrictive `copyrightText`) asserts the
  classifier assigns the *right value*, not merely a valid one. Misclassification is
  a **failing Stage-2 gate**, not an invisible distribution.
- **`metadata_text` construction oracle.** A golden fixture pins the byte-exact
  `metadata_text` for representative layers (empty-description, HTML-description,
  rich-fields, minimal), so a silent change to the field every semantic query ranks
  over is a failing test (Dimension 3).
- **Real-data parity anchor — pinned to the *actual* dropped rows, not a "likely"
  cause.** The existing Nov-2023 raw JSON is available (`jsonfiles.tar.gz` in
  `govgis_nov2023`; the sibling holds the geoparquet locally). Stage 2 **converts
  that existing raw JSON into the new schema without re-crawling** as the first
  snapshot — reproducing the 865,864 / 195,479 / 1,684 counts. The recon calls the
  560-row full-vs-slim delta a *likely* reprojection failure, so Stage 2 **inspects
  and enumerates the specific 560 dropped layer IDs** and pins the expected delta to
  *those exact rows*, rather than importing a ±560 tolerance band (a band wide enough
  to pass a pipeline that still silently drops those exact rows would not actually
  enforce the losslessness requirement it sits beside). Under the rebuild's
  losslessness rule those rows are **retained with `geometry_valid=false`**, so the
  new pipeline's expected behavior is "same 865,864 layers, 560 flagged," not "560
  fewer" — the gate asserts the *retention*, closing the parity and losslessness
  requirements together.
- **Determinism oracle** (build-twice-assert-identical IDs and typed output on
  fixtures, with the **TIGER vintage pinned as a frozen input** so facet derivation
  is actually covered — Dimension 3) and **manifest/consumer oracle** (a consumer
  stub verifies every checksum/count and fails closed on a deliberately corrupted
  artifact).
- **Retrieval + spatial oracle, with an *independent* spatial ground truth.** Extend
  this repo's existing Stage-0 query set (`docs/stage0/query_set.json`, 23 grounded
  queries, Recall@k parity procedure) with *topic + place + jurisdiction* queries.
  Recall@k for semantic queries is straightforward. The **administrative (FIPS/county)
  precision/recall is the trap the maintainer's own "fixture-is-fiction" rule warns
  about**: Nov-2023 carries no ground-truth FIPS (the CSV is dead), so "expected
  county" cannot be derived by the *same* centroid-in-TIGER heuristic the oracle is
  meant to test — that would be marking your own homework. The independent ground
  truth is a **small hand-verified set (~30–50 layers) whose jurisdiction is
  confirmed from an authority the derivation does not use** — the agency name and
  hosting portal (`gis.traviscountytx.gov` → Travis County, FIPS 48453), a
  known-single-county municipal server, an unambiguous statewide server — labeled by
  a human against those external signals. The derived-FIPS pipeline is then scored
  against *that* set. If an independent ground-truth set of adequate size cannot be
  assembled, the honest fallback is Open Decision 2: **hold the administrative-query
  tier** until Elfelt's CSV (which carried real FIPS) is recovered, rather than ship
  a facet feature validated only against its own heuristic.

### Stage 0 — foundations, contracts, seed path, and permission

Actions: stand up the `govgis-pipeline` repo (uv, ruff, mypy strict-leaning,
pytest, pre-commit, CI modeled on `geospatial-data-converter`'s mature
`ci.yml`); write the schema registry; implement and unit-test the **section-and-
entry `.txt` seed parser** (Grounding Correction 1) with the hard no-PDF guard;
capture golden fixtures from live servers; build the oracle query set; **email
Elfelt** to confirm the `.txt` ingestion path is acceptable, ask about restoring
the CSV, and record the free-derivative acknowledgment and the commercial-use
posture.
**Gate:** schema registry round-trips on golden fixtures; the seed parser
produces a plausible roster with `jurisdiction_level` + `state` + agency-name +
roots from a real `.txt` fetch, and refuses the PDF URL (tested); ID functions
pass the determinism oracle; and the **licensing question is *closed to a recorded
decision with a named default*, not merely "we emailed him"** (the launch gate).
Because Elfelt is non-responsive (issue #74 unanswered since 2025-12-15), the gate
passes on one of Dimension 1's three branches being **selected and recorded**:
Elfelt confirms → automated ingestion; Elfelt declines automation → manual
committed-seed mode; or **no reply within the defined window → proceed under the
free/non-commercial default, with the decision, date, rationale, and
unanswered-contact record checked in** and commercial use hard-blocked. What the
gate forbids is an *open* question with no default and no timebox — an indefinite
"waiting to hear back" is not a passing state. `commercial_use_authorized` remains
`false` regardless. CI green.
**Rollback:** pure code, nothing published. **But note the outward-facing residue:**
the Elfelt email cannot be un-sent and the recorded licensing decision is a
commitment, not a re-pinnable artifact — see the rollback note on irreversible
actions in the Stage-4/staging discussion below.

### Stage 1 — crawl to raw, measured

Actions: implement the `restgdf[resilience]` crawl with recorded
concurrency/rate/timeout config; write `crawl_outcomes` for every seed server;
land raw JSON content-addressed with per-file checksums; emit and validate the
run manifest. **Run first against one representative state shard** to **measure**
wall-clock and layers/server at current scale (this is the number that replaces
Grounding Correction 2's estimate and sets the shard count N).
**Gate:** run manifest validates; `seed_count == Σ crawl_outcomes`; success rate
within absolute + delta bounds; every landed file checksummed; no
`invalid_response` written as valid; **measured shard timing recorded and N
computed** so a full crawl fits the sharded free-runner budget.
**Rollback:** raw is append-only and immutable; a bad crawl is a discarded
staging batch, never an overwrite.

### Stage 2 — transform to typed core + geometry + facets + license (first snapshot from existing data)

Actions: parse raw through Pydantic into typed `servers`/`services`/`layers`;
assign content-addressed BLAKE2b IDs; promote `fields`/`extent` to Arrow structs,
contract the rest as versioned JSON; reproject extents with native extent+CRS
preserved and failures retained (`geometry_valid=false` + reason code); capture
`geometryType`/centroid/geohash; recover facets (level+state from the seed;
FIPS/county derived via TIGER spatial join + heuristics; CSV enrichment if
obtained); classify `license_status`. **Do this first over the existing Nov-2023
raw JSON** (the parity anchor).
**Gate:** 100% record schema-conformance; ID uniqueness + referential integrity;
determinism oracle passes (TIGER vintage pinned); reprojection validity within
bounds with failures **retained and reconciled**; the **`license_status`
known-answer oracle passes** (correct enum on the labeled fixtures, not just a
valid one); the **`metadata_text` construction oracle passes** (byte-exact on its
fixtures); **Nov-2023 parity reproduces the 865,864 layers with the specific 560
extent-failure rows enumerated and *retained* (`geometry_valid=false`), not dropped**
— pinned to the actual dropped IDs, not a tolerance band; slim view row-count == its
projection (no independent-build drift).
**Rollback:** the typed core is a deterministic function of raw; re-run, don't
repair. A wrong `license_status` *default* has no clean rollback once shipped — only
mitigations (the `excluded_servers` denylist, the link-only default) — which is
exactly why its correctness is gated *here*, before publish, not just drift-checked
after.

### Stage 3 — consumer views: parquet, geoparquet, viewer sibling, embeddings, ANN index

Actions: build the typed bulk parquet, the geoparquet, the **plain-parquet
Viewer sibling** (WKB + split bbox), the slim search view (derived `SELECT`), the
embeddings (HF Jobs GPU, changed-rows-only after the first full embed), and the
**quantized ANN index** (HNSW/IVF + SQ8/PQ, sized to fit `cpu-basic`); assemble
the manifest with checksums, counts, model/revision, lineage link, license
distribution.
**Gate:** manifest complete and self-consistent; plain-parquet sibling present
and Viewer-loadable; index vector count == metadata count; **index memory fits
the target serving envelope** (measured, per Grounding Correction 2); **peak
index-build memory measured and recorded** (the build-time gate parallel to the
crawl's Stage-1 measure); retrieval oracle Recall@k ≥ prior (or ≥ documented floor
for the first snapshot); manifest/consumer oracle passes on real artifacts and
fails closed on a corrupted copy; **plus an explicit projected-scale probe.** The
decisive "quantized ANN is mandatory / fits `cpu-basic`" claim was derived for
~3–4M vectors, but Stages 2–3 run on the **865k Nov-2023 parity corpus (~4× smaller,
~3.5 GB)** — so a Stage-3 pass at 865k does *not* by itself validate the requirement.
Stage 3 therefore also **builds the quantized FAISS index at *projected* scale**
(replicate/perturb the 865k vectors up to ~3–4M, or subsample a representative
fraction and extrapolate) and records **build memory, serve-time resident size, and
query latency at that scale** against the `cpu-basic` envelope. This exercises the
FAISS quantized-index path and the DuckDB spatial/FTS latency at the size the
feasibility argument is about, **before** Stage 6's real crawl — so a "14 GB blows
the envelope" surprise is caught at the index stage, not discovered when the fresh
snapshot lands.
**Rollback:** views are regenerable from the core; nothing published yet.

### Stage 4 — publish the first snapshot (human go/no-go)

Actions: run the full drift panel vs. the prior published snapshot (for the first
snapshot, vs. the Nov-2023 parity anchor); on all-green, publish the `govgis`
repo snapshot **tag** + manifest + `diff/`; enable **Xet**; add
supersede/superseded-by notes and DOIs; re-download from a clean environment and
re-verify the manifest end-to-end.
**Gate:** drift panel all-green (or quarantined with a recorded human override);
Viewer renders the plain parquet; the snapshot re-verifies from scratch; the
lineage-chain link resolves.
**Rollback:** publishing *data* is additive (a new tag); it never mutates or
deletes a prior snapshot, so the data rollback is simply "don't move `latest` to
the new tag." **But not every Stage-0/4 action is a re-pin, and the plan says so
plainly rather than implying the data model covers everything.** The
**irreversible/outward-facing actions have only forward remedies, not rollback**:
a **newly minted DOI resolves permanently** (it cannot be un-minted; the remedy is
a corrective "superseded/withdrawn" note); a **supersede banner posted to an old
DOI-bearing card** is a *manual* reversal, not an automatic one; and the **Elfelt
contact and the recorded licensing decision** (Stage 0) cannot be un-sent or
un-decided — they are commitments with reputational, not technical, reversal.
Because these are irreversible, they sit **last** in the stage sequence and behind
the **human go/no-go** precisely so they are entered deliberately; a wrong
`license_status` default that ships restricted data is mitigated (denylist,
link-only), never cleanly rolled back, which is why its correctness is gated at
Stage 2. The rule of thumb: *data* worst-case is "re-pin"; *outward-facing*
worst-case is "issue a correction and honor any objection immediately."

### Stage 5 — serving views + automation

Actions: stand up the **embedded-DuckDB hybrid search + MCP tool set** as an
optional service reading the pinned snapshot (with output-safety and adversarial
tests — prompt-injection records, filter-value hallucination, malformed/oversized
input, **and serve-time SSRF fixtures against `get_live_endpoint`/the live-fetch
path**); wire this repo's Space to consume the checksummed native index (per the
seam-sequencing contract in Dimension 5); wire the weekly seed-delta Issue bot
(**with the heartbeat-commit keepalive and the external dead-man's-switch**, per
"Automation liveness") and the quarterly matrix-sharded full-crawl workflow (**with
`fail-fast: false`, content-addressed shard resume, and the async HF-Jobs
embedding hand-off**); wire the drift panel into every refresh; wire the
**secret-leak gate and crawl/publish secret isolation**; keep **publish and
promote decoupled.**
**Gate:** every MCP tool returns typed validated output and injection/malformed/**SSRF**
fixtures fail closed; latency acceptable with the FAISS ANN index and **a
concurrency cap holds under simulated abusive load** (not just single-query
latency); consumers load the new snapshot and pass smoke tests against a verified
manifest; the weekly cron produces a seed-drift Issue on schedule **and produces
a distinct loud alert when a `.txt` fetch 404s/format-breaks** (both paths tested);
the **keepalive commit lands and the dead-man's-switch fires on a simulated missed
run**; a **lost shard re-runs idempotently** without a full re-crawl; the
**secret-leak forced-failure test passes**; a **deliberately-injected bad refresh is
caught and quarantined** by the drift panel (the automation is proven to *fail
correctly*, not just to run); promote is a separate explicit step.
**Rollback:** consumers pin to the prior snapshot tag; automation can be disabled
without affecting published data.

### Stage 6 — fresh full crawl + supersede, then footprint evaluation (deferred)

Actions: run the first **fresh full 7,500-server crawl** through the proven
pipeline (the growth-aware volume-drift gate is what keeps the expected ~4×
jump from false-blocking here); publish the 2026 snapshot; point consumers at it;
freeze the old repos with deprecation banners *and the corrected license NOTICE*
(never deleted). Only after a full snapshot ships with parity: evaluate **deeper**
model/quantization options beyond the Stage-3 dimension choice (IVF/HNSW variants,
deeper PQ, compact metadata), and evaluate the **complementary/independent seed
sources** held as options in Dimension 1 (a data.gov/GeoPlatform federal slice; a
scoping look at FGDC Clearinghouse or Shodan/Censys fingerprinting) as
diversification against the single-Elfelt-source risk — each measured against the
retrieval oracle and the manifest's recorded quality floor, shipped only on a gate
pass (mirrors the modernization plan's Stage 7).
**Gate:** the 2026 snapshot passes every data-quality gate at true 3–4M scale
(including index-fit and build-memory, fail-closed — the projected-scale probe from
Stage 3 is now exercised for real); its diff vs. Nov-2023 is sane and published;
one full automated cron run completes green end to end, **including the keepalive
and dead-man's-switch**; any footprint or seed-source change meets the quality
floor.
**Rollback:** re-pin to any prior snapshot tag; the frozen Nov-2023 repos remain
the ultimate fallback.

---

## Dimension 8 — Ownership and repo structure

**Bounded and separated by change-cadence, not sprawling.** Net new standing
surface for the maintainer is deliberately small:

1. **`govgis-pipeline` (new, GitHub) — the primary deliverable.** The crawl +
   transform + publish code, schema registry, golden fixtures, gates, drift
   panel, CI, and cron automation. This is the artifact the whole ecosystem is
   missing — recon confirms the original build "was not found in any GitHub
   repo… it exists only as notebooks inside the Hugging Face dataset repo,
   never committed." Making it a first-class, tested, versioned repo is the
   single highest-leverage structural change. It is a **stateless batch job** —
   the cheapest kind to own.
2. **`govgis` (new, HF dataset repo) — the published product.** Snapshots as
   tags, Xet-backed, manifests, plain-parquet Viewer siblings, embeddings in
   their own file. Supersedes both frozen 2023 repos (which stay up, cross-linked,
   for citation stability). Raw archival JSON lands in a **private companion**
   (a private HF dataset repo or object store) — the audit/replay floor. This is
   consistent with Elfelt's "derivative works must be available to everyone for
   free" condition: the private archive is an **internal audit/replay floor, not a
   published derivative** — an undistributed archive is not a "derivative work made
   available," so the free-availability condition does not bite it, and the
   **published derivative that the condition *does* govern (the `govgis` repo) is
   always public and free.** The private/public split is an operational choice
   (raw fidelity + storage cost management), not a way to gate access to the
   derivative.
3. **Unchanged: this Space.** Modernized on its own separate track; repointed at
   the new checksummed native index at Stage 5; not otherwise touched here.
4. **Unchanged and un-adopted: the sibling `-spatial-server`.** Left exactly
   as-is as an optional user-run deployment fed by the published geoparquet, and
   documented as the escalation path behind a `govgis-core` retrieval interface.
   **Explicitly not extended or operated as core**, because extending it means
   owning its patch treadmill — the exact burden the 9–10-month PR backlog
   measures.

`restgdf` stays an upstream dependency (pin it); `restgdf_api`'s dead CSV proxy
is **not** revived — the seed loader reads the `.txt` directly. Deliberately
rejected structures: **do not fold the pipeline into the Space repo** (build and
serve stay separable; the Space is on its own migration), and **do not extend the
sibling server repo** (couples a low-maintenance batch job to a high-maintenance
always-on service).

Why three repos rather than the agentic/geospatial drafts' four-to-five
(`govgis-core`, `govgis-service`, `govgis-embeddings`, `govgis-catalog`)? Because
under the simplicity spine those are *optional views*, not standing services:
`govgis-core` (the thin retrieval interface) and the MCP service are code that
lives *inside* the optional serving tier, deployed only if a live service is
wanted; embeddings are a *file/config* in the `govgis` repo, not a separate
always-maintained repo. Keeping the standing count at three (pipeline + public
dataset + private raw) is the simplicity-favoring choice; the separation that
remains is the minimum the different change-cadences justify.

**Bus factor and succession — named, because this product is built to be cited.**
The whole design concentrates on one person: three new standing surfaces, a
sole-source human relationship with Elfelt, and a load-bearing crawl dependency
(**restgdf**) that is *also* the same solo maintainer (and itself carries open
dependency PRs). The product is meant to be **DOI'd, cited, and depended on** by
others (`Lexicom7/EO_Datasets` already cites the old one), so "safe to neglect for
six months" is not the same as "safe if the one maintainer stops." The plan does
not pretend to solve single-maintainer concentration, but it refuses to leave it
implicit:
- **restgdf risk has a named fallback beyond the thin httpx shim.** The crawl is
  pinned to a specific restgdf version, and the plan already keeps a raw-`httpx`
  fallback for servers restgdf chokes on; the succession note adds that **restgdf
  is MIT and vendorable** — if it breaks against a future ArcGIS change inside a
  neglect window and its maintainer is unavailable, the crawl can pin the last-good
  version and the pipeline still runs from committed seeds, because nothing in the
  critical path requires restgdf's *latest*.
- **The pipeline is designed to be handed off.** Everything that matters is
  **in-repo and reproducible from disk** — the schema registry, the gates, the
  golden fixtures, the manifest chain, the committed seeds, and the private raw
  archive — so a successor inherits a *stateless, documented, gate-checked batch
  job*, not tribal knowledge in un-committed notebooks (the exact failure the recon
  found in the original build). A short `SUCCESSION.md` records the one genuinely
  personal asset — the Elfelt relationship and any permission grant — and the
  minimal credentials/handoff steps. This is the cheapest possible succession
  posture, but it is *stated*, which the current ecosystem never did.

---

## Dimension 9 — Cost and ops burden

Realistic accounting, tradeoffs named. Framed by the dominant constraint:
**single-maintainer, low-QPS, read-mostly, must survive neglect.**

- **Upfront build effort — the largest cost by far, and the real risk.** The
  recurring-dollar figures below are rounding error next to the **multi-person-month
  solo build** of Stages 0–5: a schema/contract registry, a tolerant section-and-
  entry `.txt` parser, the SSRF-hardened crawl + `crawl_outcomes` ledger, the typed
  transform with BLAKE2b IDs and TIGER FIPS derivation, `license_status`
  classification, the changed-rows embedding + quantized-FAISS build, the
  checksummed manifest + lineage + diff, and the ~10-gate drift panel — all before
  the first snapshot ships. Honest estimate for one spare-time maintainer: **roughly
  2–4 focused months of evenings/weekends to first published snapshot**, front-loaded
  in Stages 0–3. This is stated plainly because it is the project's **most likely
  failure point** — the same revealed-capacity signal the simplicity spine rests on
  (the sibling's 9–10-month unmerged-PR backlog) predicts that *finishing the build*,
  not *running it*, is where a solo effort stalls. Two mitigations are structural,
  not wishful: the **Nov-2023 parity anchor** (Stage 2 builds against existing real
  data with no crawl, yielding a working, demonstrable, valuable artifact early), and
  the fact that **each stage's gate produces something usable** (a validated typed
  core, then a manifested snapshot) so a stall leaves a real deliverable, not a
  half-integrated system. The **Minimal budget tier still requires all of Stages
  0–4** — the tiers scale *runtime ambition and ops*, not build effort, so there is
  no genuinely weekend-sized first increment; the honest smallest first milestone is
  "Stage 2: the Nov-2023 data, re-typed with deterministic IDs and a manifest,"
  which is itself weeks of work but delivers a strictly better artifact than anything
  in the ecosystem today.
- **Orchestration & CPU (crawl shards, transform, TIGER join, slimming, manifest,
  gates): ~$0 — because the resource-bounded steps are placed where they fit.**
  GitHub Actions is free for public repos (public-repo Actions *minutes* are free
  and public-repo build *artifacts* are not counted against storage — so the
  shard→combine artifact hand-off is genuinely free at this scale). The "~$0" holds
  only because each heavy step has a home that fits its free envelope: the **crawl
  is sharded** to stay under the 6h/job ceiling; the **combine job streams shards to
  the archive** rather than holding multi-GB on the ~14 GB runner disk; the
  **transform + TIGER point-in-polygon join** run on the standard runner and are
  **state-partitionable** if a single job's ~4–8 GB output or memory gets tight; and
  the **index build is deliberately *not* here** — it co-locates with the GPU
  embedding step (next bullet), because a flat 14 GB vector set does not fit the
  16 GiB runner. Runner disk/RAM headroom for combine and transform is a
  Stage-1-measured quantity, not an assumption.
- **GPU embeddings + index build: small, occasional, pay-per-use.** Re-baselined at
  ~3.5M layers: a full re-embed is **~0.5–2 GPU-hours of inference (~1–3h
  wall-clock) ≈ single-digit-to-low-tens of dollars** on HF Jobs, and runs **only**
  on a model change or first snapshot; steady-state changed-rows-only re-embeds cost
  cents-to-dollars. The quantized FAISS index builds on this same GPU job at zero
  marginal data movement. The one non-zero recurring compute cost, bounded and
  infrequent by design.
- **Storage: multi-GB per snapshot, sublinear in count *if the layout cooperates*.**
  ~15–30 GB per full snapshot (embeddings dominate). **Xet dedup** makes each new
  snapshot's marginal cost roughly the changed delta **provided the stable
  row-order / stable row-group-boundary requirement of Dimension 4 holds** — float
  vectors do not dedup by near-match, so the saving comes from unchanged rows
  staying byte-identical and chunk-aligned, and the **actual dedup ratio is a
  Stage-4-measured number, not an assumed one**. **Separated embeddings** avoid
  re-uploading vectors on a metadata-only refresh. Private raw is the audit trail;
  its retention is tunable (lifecycle-expire after N snapshots if cost bites).
- **Serving: $0 always-on by default.** Bulk = the Hub files (someone else's
  infra). The embedded-DuckDB hybrid search + MCP tier has **no always-on
  database to operate, patch, back up, or pay for while idle** and no
  load-on-init step — a new snapshot is a pin bump. The Space is a sleep-when-idle
  managed Space. The sibling's Postgres stack is the documented, reversible
  escalation for real concurrency — a deliberate future cost, not the default a
  hobby-scale index carries. **This is the concrete savings the design buys:** the
  9–10-month unmerged-security-PR backlog on the always-on sibling is the number
  this avoids.
- **Human burden (steady state): ~1–2 hours per quarter on the happy path,
  plus a bounded-but-real residual.** The happy path is: read the drift Issue,
  approve a snapshot, glance at the gate results and the manifest diff, approve the
  promote. But the honest steady-state number must also carry two items the happy
  path omits: **(a) gate-failure triage** — a fired gate (e.g. the 20-point
  reachability-drop gate) needs actual diagnosis (network blip vs. restgdf
  regression vs. shared-runner IP block vs. a real upstream change), not a glance,
  which is *the whole point* of the drift panel; and **(b) the pipeline repo's own
  dependency/security-PR trickle** — `govgis-pipeline` pins restgdf, pydantic,
  duckdb, faiss, geopandas, and CI actions, and will accrue Dependabot/security PRs,
  the same *class* of burden the sibling's backlog measures. The design **reduces**
  this residual (it does not "drop" it): because the repo is an **always-off batch
  job**, an unpatched dependency carries no idle attack surface, a failing gate's
  worst case is "no new data" so triage is **deferrable, not urgent**, and skipping
  a quarter is safe by construction. So the residual is real but low-stakes and
  self-throttling — call steady-state **~1–2 hours/quarter happy path, a few hours
  more in a quarter that fails a gate or when the maintainer chooses to clear the
  dependency queue** — not zero, and not the sibling's always-on treadmill.
- **One-time costs:** the Elfelt email (minutes; also potentially recovers the
  CSV and its facets); and — *only if* commercial use or real-feature-data
  redistribution is ever contemplated — a real legal review. Neither is on the
  recurring path.

**Budget tiers, so the maintainer explicitly picks the ambition** (geospatial
draft's framing):

- **Minimal:** the stateless pipeline + one Xet dataset repo + the Hub-as-API +
  the plain-parquet Viewer + quarterly manual re-crawl. Near-zero recurring cost,
  no live service. Already delivers versioned, manifested, drift-gated snapshots
  with recovered facets and hybrid-queryable parquet (via DuckDB locally / DuckDB-
  WASM in-browser for structured+spatial+FTS).
- **Standard:** + the embedded-DuckDB hybrid search + typed MCP tool set as an
  optional sleep-when-idle service + the weekly drift-Issue bot + the quarterly
  automated crawl. Bounded pay-per-use GPU; no always-on DB.
- **Full:** + FIPS-precise administrative queries (CSV recovered from Elfelt),
  the successive-snapshot `diff/` + `whats_changed` product, on-demand live
  feature fetch, an optional STAC face, and — only if concurrency demands it —
  the Postgres+pgvector escalation.

The through-line: every recurring cost is bounded and pay-per-use, and every
optional tier sleeps when idle. The pipeline's failure mode under neglect is
"no new data," never "broken data" or "a compromised server."

---

## Conflicts resolved (real decisions, not hedges)

1. **Build/extend Postgres+pgvector+FastMCP as core (rigor, geospatial) vs. avoid
   it (simplicity, judges a & b).** **Resolved: do NOT operate it as core
   infrastructure.** The 9–10-month security-PR backlog under exactly these
   single-maintainer conditions is disqualifying. The serving *read path*, when a
   live tier runs, is **embedded DuckDB for spatial + FTS + structured filtering
   with a quantized FAISS index owning the vector step** over the pinned parquet —
   zero always-on DB, zero ETL. (DuckDB VSS is *not* used for the vector index: it
   cannot quantize and would reproduce the ~14 GB footprint that does not fit free
   tier — see Grounding Correction 2 and Dimension 5.) The plan does **not** claim
   the sibling lacks an ANN index — the recon says it loads a "vector-indexed"
   table and that assertion is retracted as unsourced. Postgres+pgvector stays the
   **documented, reversible escalation** behind a `govgis-core` interface, and the
   sibling keeps working for free as an optional user-run deployment fed by the
   geoparquet byproduct. Capability kept; burden dropped.
2. **Full vs. incremental crawl.** **Resolved: full, stateless crawl per
   *published* snapshot** (simplicity's robustness — nothing to reconcile or
   corrupt), **but the expensive step (embedding) is made incremental** via
   deterministic-ID text-hash diffing *between two immutable snapshots* (no
   mutable standing crawl state). Best of both; stateful incremental *crawling* is
   a named future option only if crawl cost becomes binding.
3. **Type everything vs. type almost nothing.** **Resolved: type the load-bearing
   columns and promote high-value nested structures (`fields`, `extent`) to Arrow
   structs; keep a contracted JSON escape-hatch for the open-ended remainder.**
   Neither a brittle 205-wide schema nor a stringly-typed firehose.
4. **Facet-recovery source.** All four drafts assumed the seed columns survive in
   the `.txt`; **a real read refuted that.** **Resolved: recover level+state from
   the `.txt` sections, agency from the heading text, FIPS/county by *deriving*
   them (TIGER spatial join + heuristics + crawled metadata), with Elfelt's CSV as
   a precision-upgrade enrichment if obtained.** Graceful degradation, no
   dependency on an unverified feed.
5. **Repo topology (3 vs. 4–5 repos).** **Resolved: three standing surfaces**
   (`govgis-pipeline`, the `govgis` public dataset, a private raw archive); the
   `govgis-core` interface and MCP service are code inside the *optional* serving
   tier, and embeddings are a *file* in the dataset repo, not separate standing
   repos.
6. **Search shape (vector-primary vs. hybrid-primary).** **Resolved: hybrid
   primary** (spatial → structured/FIPS → full-text → optional vector re-rank),
   grounded in the 81%-empty-`description` fact; FAISS/vector demoted from spine
   to re-ranker facet.
7. **Serving engine at scale.** **Resolved by re-baselining, and split by
   engine:** at ~3–4M layers a flat fp32 index (~14 GB) does not fit free tier, so a
   **quantized ANN index is mandatory on the vector-first Space** — and quantization
   (IVF/PQ/SQ8) is FAISS, not DuckDB VSS, so **FAISS owns the vector index while
   DuckDB does spatial + FTS + structured filtering**; the hybrid tier re-ranks
   small filtered candidate sets by brute force. In-browser vector search is
   infeasible — the DuckDB-WASM path is structured+spatial+FTS only, vector re-rank
   server-side against the FAISS index. One embedding artifact, one FAISS index, no
   duplicate ANN structures.

---

## Open decisions for the maintainer

These are validation-time decisions, not blockers to executing Stages 0–2:

1. **Contact Elfelt, and pick the no-reply default (Stage-0 gate).** Confirm the
   `.txt` is an acceptable machine-readable ingestion path given the dead CSV and
   the PDF-scraping prohibition; confirm the **standing weekly automated fetch**
   cadence is acceptable, not just a one-time read; ask whether the **CSV can be
   restored/shared** (recovers precise FIPS/county facets); record the
   free-derivative acknowledgment. Because he is non-responsive, the decision the
   maintainer must actually make is **the no-reply default**: proceed under the
   free/non-commercial posture after a defined window (recommended), or block until
   he answers. Required before the first public crawl; required in writing before
   any commercial use regardless of the default.
2. **Facet precision without the CSV.** Confirm the derive-FIPS-from-geometry
   (TIGER spatial join) + heuristics approach is acceptable for the first
   snapshot, with CSV-based precision as a later enrichment — or hold the
   administrative-query tier until the CSV is recovered.
3. **Snapshot cadence.** Quarterly full is the recommended default; the weekly
   seed-delta Issue will inform whether monthly (cost) or semi-annual (staleness)
   fits how fast the servers actually drift.
4. **Crawl runner.** Matrix-sharded free GitHub Actions is the default; confirm
   after the Stage-1 shard-timing measurement, or pre-authorize a self-hosted
   runner / HF Job / ephemeral VM fallback if the measured full crawl won't shard
   under the ceiling.
5. **Serving ambition (the budget tier).** Minimal (Hub-as-API only), Standard
   (+ embedded-DuckDB hybrid search + MCP), or Full (+ Postgres escalation, STAC,
   on-demand fetch). Pick the tier.
6. **Redistribution scope for `unverified_state_local` rows.** The plan defaults
   to link/index-only; confirm that conservative default for the first snapshot,
   or commission the per-jurisdiction legal review to widen it.
7. **Embedding model and dimension — a Stage-3 choice, not a Stage-6 afterthought.**
   `BAAI/bge-large-en-v1.5` (1024-dim) is the migration-continuity default (both
   current consumers use it, and it is a genuine retrieval-quality choice), but the
   **dimension is the root of the whole footprint/quantization cascade**: a 384-dim
   model (`bge-small-en-v1.5`) cuts raw vectors to ~5 GB fp32 — fitting `cpu-basic`
   with *no* quantization — so a from-scratch rebuild should weigh
   retrieval-quality-vs-footprint and **pick the dimension at Stage 3**, when the
   index format and sizing are decided, not defer it past the stage that depends on
   it. The separated-embeddings design keeps a later switch a cheap isolated
   regenerate (~low-tens of dollars) either way; the point is to *choose*
   deliberately up front rather than inherit 1024 by default. Deeper model/quant
   optimization beyond that first choice stays a Stage-6 evaluation.
8. **Seed-source diversification.** The plan keeps Elfelt primary (no comparable
   catalog exists) but the whole pipeline is bottlenecked on one non-responsive
   volunteer's restricted feed. Decide whether to add a **complementary
   data.gov/GeoPlatform federal slice** (licensing-clean, Elfelt-independent) as a
   Stage-6 resilience source, and whether to scope the **FGDC Clearinghouse
   Registry** or **Shodan/Censys ArcGIS fingerprinting** as the deeper
   independent-discovery fallback if Elfelt's list ever dies for good (Dimension 1).
   Defaulting to single-source is a *recorded* choice here, not an oversight.

---

## Validation-gate summary — how a bad refresh is caught before it ships

One gate per hop, all fail-closed, all scripted; no hop trusts the previous one:

| Hop | Gate asserts | A bad refresh is caught because… |
|---|---|---|
| **seed → parse** | `.txt` fetched (not PDF), checksummed, committed; section/entry parse yields level+state+agency+roots; roster count sane | a mangled/empty seed or a format change fails the parse + count band |
| **crawl → raw** | run manifest valid; success-rate absolute + delta bounds; `seed_count == Σ crawl_outcomes`; every file checksummed; **shard timing measured** | a broken crawler/network shows as a reachability-delta breach, not a silent row drop |
| **raw → typed core** | 100% schema conformance; ID uniqueness + referential integrity; determinism oracle; reprojection validity (failures retained); Nov-2023 parity within the explained 560-row delta | a parse/ID/type regression fails conformance/determinism before it propagates |
| **core → views** | manifest self-consistency; Viewer sibling present; slim == its projection; vector count == metadata count; **index fits the serving envelope**; Recall@k parity | a mismatched index, missing sibling, oversized index, or quality regression fails publish-readiness |
| **views → publish** | full drift panel vs. prior; re-download + re-verify from clean env; lineage link resolves; publish/promote decoupled | any cross-snapshot anomaly quarantines the candidate; publish is impossible while `all_passed: false`; a bad snapshot never auto-promotes |

This is the difference between a maintained data product and a one-off scrape —
and it is what makes a stateless, single-maintainer, sleep-when-idle pipeline
safe to leave alone for six months.
