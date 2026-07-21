# Judge B — critique of the four govgis rebuild drafts

Recorded 2026-07-20. This is an independent adjudication of four parallel,
single-lens rebuild plans for the **govgis dataset pipeline** (the data
engineering — crawl, transform, publish, refresh — not this repo's separate,
in-progress Gradio migration). All four drafts, all five ecosystem-recon files,
and `docs/modernization-plan.md` were read in full before scoring. I am a judge,
not a drafter: the job is to score, pick a base, name the best graftable ideas
from the non-base drafts, and name the single biggest gap all four share.

The four lenses: **data-engineering-rigor**, **operational-simplicity**,
**agentic-serving**, **geospatial-native-rethink**.

---

## Scores (1–10; for technicalRisk, 10 = lowest risk / safest)

| Axis | rigor | simplicity | agentic | geospatial |
|---|:--:|:--:|:--:|:--:|
| dataQuality | 10 | 7 | 8 | 8 |
| licensingCompliance | 9 | 8 | 8 | 9 |
| maintainability | 5 | 10 | 6 | 5 |
| servingUtility | 7 | 5 | 10 | 9 |
| costRealism | 8 | 9 | 8 | 8 |
| technicalRisk (10=safe) | 6 | 9 | 6 | 5 |
| extensibility | 9 | 6 | 9 | 9 |

The scores are deliberately spread. Each lens is genuinely best-in-class on the
axis it was written to optimize, and genuinely weakest where its thesis forced a
tradeoff it accepted on purpose. That is the drafts working as intended, and it
is why a synthesis — not a winner — is the right output.

---

## Per-draft assessment

### data-engineering-rigor — the strongest spine

**Why it leads on dataQuality (10) and extensibility (9).** This is the only
draft that treats the dataset as a *warehouse pipeline with data contracts*
rather than a scrape with better hygiene. The medallion split (immutable Bronze
raw → typed/validated Silver → consumer-specific Gold), the versioned
schema/contract registry, content-addressed BLAKE2b IDs *tested as a gate*, the
determinism oracle (transform golden fixtures twice, assert byte-identical
output and identical IDs), the crawl-outcome reconciliation
(`seed_server_count == Σ crawl_outcomes`, nothing silently dropped), and the
cross-snapshot drift panel are collectively the ecosystem's single biggest
missing piece. Its validation-gate summary table (one fail-closed gate per hop,
seed→publish) is the clearest articulation of "a bad refresh is refused before
it ships" any of the four produced. Crucially, this rigor is not gold-plating:
the manifest + determinism + drift spine is *exactly what makes an unattended
refresh safe*, which is the same goal the simplicity lens reaches by a different
road.

**Where it is weakest (maintainability 5, technicalRisk 6).** It explicitly
subordinates operational simplicity, and it shows: three repos, a schema
registry, a Bronze medallion tier, a drift panel, and — its one real
architectural misstep — a recommendation to **extend the sibling
Postgres+pgvector+FastMCP as the authoritative serving view**. The
operational-simplicity draft refutes exactly this with the recon's own evidence
(that sibling is carrying 9–10 months of unmerged security PRs under precisely
these single-maintainer conditions). Rigor's own cost section actually leans the
other way — "the bulk-download Gold + manifest is the cheap, always-on backbone;
the live services are optional views the maintainer can run or pause" — so its
frame *accommodates* the simpler serving posture even though its Dimension 5
prose over-commits to the sibling. For a spare-time maintainer this is the plan
most at risk of being too heavy to finish, not because any single piece is
wrong but because the sum is large.

**Verdict:** the correct base for the pipeline spine, provided its serving/ops
layer is replaced wholesale by the simplicity lens.

### operational-simplicity — the right maintainability instinct and the sharpest single refutation

**Why it leads on maintainability (10) and technicalRisk (9).** Its central
fact is not technical: every repo here has one committer, and the most
sophisticated piece of infrastructure is 9–10 months behind on security PRs.
That is the clearest possible measurement of what always-on infra costs this
maintainer, and this draft is the only one that treats it as *the* binding
constraint rather than a footnote. The stateless-batch design ("produces a good
dataset when told to, then turns completely off, breaking nothing while it sits
untouched for six months") is genuinely neglect-survivable by construction, and
its failure mode under neglect is "no new data," never "broken data." The
matrix-sharded crawl on free GitHub Actions (to beat the 6-hour ceiling without
a self-hosted runner), HF Jobs pay-per-use GPU for embeddings, the
commit-the-seed-file-every-run fallback, and the weekly drift job that just
*opens a GitHub Issue* are all lower-risk, lower-burden, and more likely to
actually ship than any competing choice.

**Where it is weakest (servingUtility 5, dataQuality 7).** It deliberately
under-serves two mandatory dimensions. Serving is "the Hub is the API" plus one
Space; there is no facet filtering, no spatial predicate, no agent tool design —
which is a defensible *minimum* but leaves the agentic and spatial value on the
table. On data model it types "almost nothing" (one JSON escape-hatch column),
skips the medallion/registry, and does not preserve an immutable Bronze tier.
This is coherent within its thesis but is exactly what a synthesis must import
rigor to fix. Its cost section is the most concrete of the four, but it sizes
the *full* embedding job at "~865k texts → single-digit-to-low-tens of dollars"
— an under-count discussed below under the shared gap.

**Verdict:** not the base (its frame actively resists the rigor and serving the
task mandates), but the source of the synthesis's entire serving/ops posture.

### agentic-serving — the best serving design

**Why it leads on servingUtility (10).** It is the only draft that designs the
*consumer contract first* and derives the schema from it, and the payoff is
concrete. Three things stand out. (1) The schema-enrichment insight: the current
slim schema discards the jurisdiction/agency/FIPS columns the seed list carries,
so the sibling's MCP tool *cannot* answer "county-level parcel layers in Travis
County"; re-joining that seed metadata into typed facets is correctly named the
single highest-leverage change for agentic access. (2) Replacing the one
markdownify-everything `gis_layer_search` tool with a composable, typed,
provenance-carrying tool set — `search`, `get_layer`, `list_facets` (explicitly
to stop agents *hallucinating filter values*), `get_live_endpoint` (the bridge
to live features), `whats_changed` — returning structured JSON, not markdown
blobs, with output-safety in the contract because retrieved government metadata
is untrusted LLM input. (3) The sharp, source-grounded catch that *both* current
stacks lack any ANN index (the sibling does a full sequential scan
`ORDER BY embeddings <=> $1` over 865k vectors per query), so whatever serves
must add HNSW/IVF. Its data model (uuid5 content-addressed IDs, typed relational
parquet, per-record license fields, honest `extent_4326` + centroid + geohash)
is strong (dataQuality 8).

**Where it is weakest (maintainability 6, technicalRisk 6).** It recommends
embedded DuckDB (VSS/HNSW) as the *default* serving engine over pgvector. The
ops argument is good (no always-on DB, zero-ETL snapshot promotion, native
parquet reads) and it honestly keeps Postgres as a documented fallback behind a
`govgis-core` abstraction — but DuckDB VSS/HNSW at multi-million-vector scale is
less proven than pgvector, and building a `govgis-core` + `govgis-service` +
four-face surface is real software to build and keep alive. It is heavier than
simplicity while lighter than a full PostGIS stack.

**Verdict:** the serving *design* to graft — facets, tool set, output-safety,
ANN — onto a rigorous pipeline base, delivered through the Space's own MCP
surface rather than a new standing service.

### geospatial-native-rethink — the best product reframe, the riskiest surface

**Why it leads on servingUtility (9), licensingCompliance (9), extensibility
(9).** Its thesis is the most intellectually honest challenge to the status quo:
with 81% of `description` values empty and only bounding-box extents (no true
feature geometry), *semantic-search-over-a-vector is the wrong primary shape*.
The highest-value query is "which layers cover my area, in my jurisdiction,
about my topic" — a hybrid of spatial predicate + FIPS/structured filter +
full-text, with dense-vector similarity demoted to an optional re-ranker. This
is correct and well-argued, and it is deliverable cheaply (DuckDB spatial + FTS
+ vss). Two more standout, grafts-worthy ideas: the **losslessness** discipline
(retain reprojection-failed rows with `geometry_valid=false` and native CRS
rather than silently dropping the 560 rows the original build lost with no
record — a hard requirement for correct snapshot diffing), and the **two-tier
licensing** insight (separate the license of the Elfelt-derived server-URL *list*
from our own crawled-metadata *contribution*, machine-readable in the manifest)
which is sharper than any other draft's blanket-license handling. Capturing
`geometryType` at zero storage cost, the FIPS-join-first / extent-refine-second
precision argument, and separating embeddings into their own repo are all clean.
It is also the only draft that offers a tiered budget (Minimal/Standard/Full)
and the only one to red-team its own assumptions in a closing section.

**Where it is weakest (technicalRisk 5, maintainability 5).** It carries the
largest, least-proven serving surface: Postgres+PostGIS+pgvector *and* STAC
*and* MCP *and* DuckDB-WASM-in-browser. The author honestly flags the two
riskiest bets (STAC modeling live services is unconventional; DuckDB-WASM
carrying an 865k-and-growing catalog in-browser is unbenchmarked), which is to
its credit, but flagging risk is not retiring it. The Minimal tier rescues the
maintainability story only if the maintainer actually stops there — and the
draft's center of gravity is the Standard/Full hybrid stack.

**Verdict:** graft its query *model* (hybrid, vector-demoted) and its
geometry/license *honesty*; do not adopt its full serving surface.

---

## Recommended base for synthesis: **data-engineering-rigor**

The base should be the draft whose *structure* the synthesis inherits, and
data-engineering-rigor provides the most complete and load-bearing skeleton: the
gate-per-hop validation spine, the checksummed manifest with a snapshot-lineage
chain, determinism-as-a-tested-property, and the cross-snapshot drift panel are
precisely the "nothing like this exists in the ecosystem today" gaps the recon
identifies, and they are what make an unattended pipeline *trustworthy* — which
is the real prerequisite for the low-touch operation the maintainer needs. Its
staged plan already mirrors the proven `modernization-plan.md` method, and its
frame welcomes the simpler serving/ops posture (its own cost section already
leans that way).

This recommendation comes with one non-negotiable amendment, and it is the whole
reason the synthesis is not just "rigor as written": **replace rigor's serving
and ops layer wholesale with operational-simplicity's.** Specifically — do *not*
adopt the always-on Postgres+pgvector sibling as core/authoritative
infrastructure (the 9–10-month PR backlog is the disqualifying evidence); make
the bulk-download Gold + manifest the always-on backbone; run the crawl as a
matrix-sharded free-GHA job with HF Jobs GPU for embeddings; and keep any
Postgres path as an optional, user-run deployment fed by a byproduct the
pipeline publishes anyway. The synthesized plan is therefore **rigor's data
spine + simplicity's infrastructure + agentic's serving design + geospatial's
query model and honesty**.

I considered basing on operational-simplicity instead — it is the safest to
ship and the most maintainable. I rejected it as the *base* because its thesis
("everything else is off by default; type almost nothing") actively resists the
schema rigor, versioning discipline, and serving utility the task mandates as
required dimensions; a synthesis built on it would spend its length fighting its
own frame to add back what rigor already structures cleanly. Simplicity is the
base's indispensable *corrective*, not its foundation.

---

## Best specific ideas to graft from the non-base drafts

**From operational-simplicity:**
1. Reject the always-on Postgres+pgvector+FastMCP stack as *core* infrastructure;
   preserve it as an optional user-run `docker compose` deployment fed by the
   geoparquet byproduct the pipeline publishes anyway. Keep the capability, drop
   the standing burden.
2. Matrix-sharded crawl on free GitHub Actions (partition roots by stable hash or
   by state, each shard well under the 6-hour job ceiling, one lightweight combine
   job) — keeps the crawl serverless without a self-hosted runner. This is a
   better answer than rigor's "punt the full crawl to a dedicated runner."
3. HF Jobs pay-per-use serverless GPU for embeddings, run only when the model or
   `metadata_text` actually changes.
4. Weekly drift-detection job that **opens a GitHub Issue** ("the upstream list
   grew materially; a re-crawl may be worthwhile") — the entire monitoring system
   with zero standing infrastructure.
5. Commit the exact seed file into the pipeline repo every run
   (`seeds/YYYY-MM-DD.txt` + checksum) as a durable fallback — the dead-CSV
   episode is the lesson; an upstream feed can vanish silently.
6. A committed `excluded_servers.txt` denylist as a trivial, no-infrastructure
   takedown/opt-out lever.
7. One dataset repo with snapshots as **tags** (not repo-per-year, not an
   unbounded `snapshots/` directory) — the crisp argument that this prevents DOI,
   discovery, and consumer-pin fragmentation.

**From agentic-serving:**
1. Re-join the seed's jurisdiction/agency/FIPS metadata (Type/State/County/Town/
   FIPS/Server-owner) into typed facet columns — the single highest-leverage
   schema change for any filtered or agentic access.
2. The composable, typed, provenance-carrying MCP tool set (`search`,
   `get_layer`, `list_facets`, `get_live_endpoint`, `whats_changed`) returning
   structured JSON, replacing one markdownify-everything tool — with `list_facets`
   specifically to stop filter-value hallucination and `get_live_endpoint` as the
   bridge to live features under the source's own terms.
3. Output-safety as a first-class part of the serving/tool contract (retrieved
   government metadata is untrusted LLM input) — carry the modernization plan's
   threat model to the tool surface, not just the Space UI.
4. The catch that both current stacks lack any ANN index — any serving path must
   add HNSW/IVF, not full sequential scan.
5. Store centroid + geohash for cheap spatial pre-filtering.
6. Decouple data-publish from serving-promote as two explicit steps, so a bad
   snapshot can never auto-reach production.

**From geospatial-native-rethink:**
1. The hybrid retrieval model as the primary query shape — spatial predicate
   (bbox `ST_Intersects`) → structured/FIPS filter → full-text → *optional*
   vector re-rank — grounded in the 81%-empty-description fact; demote FAISS/vector
   from spine to re-ranking facet. Deliverable cheaply via DuckDB (spatial + FTS +
   vss extensions), which also serves simplicity's ops goal.
2. FIPS-join-first, extent-refine-second for "layers covering this county,"
   because coarse statewide bbox extents over-match every county query.
3. Losslessness: retain reprojection-failed/degenerate rows with
   `geometry_valid=false` and native CRS retained — never silently drop (the
   560-row lesson); required for correct snapshot diffing.
4. Capture `geometryType` (point/polyline/polygon/…) as an attribute at zero
   geometry-storage cost — high discovery value.
5. Two-tier licensing: separate the Elfelt-derived server-URL *list* license from
   our own crawled-metadata *contribution* license, machine-readable in the
   manifest and card.
6. Separate embeddings into their own repo/file so a re-embed does not churn the
   metadata repo and a metadata refresh does not re-upload multi-GB vectors.
7. Offer a DuckDB-WASM in-browser static path as a near-zero-cost public search
   option (no 5 GB FAISS load) — *pending a benchmark at true scale* (see below).
8. STAC as an optional interop face (server→Catalog, service→Collection,
   layer→Item) — worth prototyping one Collection before committing, per the
   draft's own honest red-team.

---

## The single biggest gap all four drafts share

**None of the four re-derives the pipeline's cost and serving-feasibility
envelope at the true target scale — they reason about downstream sizing using
the stale Nov-2023 numbers (865,864 layers from 1,684 servers) even though their
own shared premise is that the seed list has quadrupled to 7,500+ servers, which
implies a corpus on the order of ~3–4 million layers.**

Every draft correctly extrapolates *crawl wall-clock* (2h16m for 2,038 roots →
~8–10h at 7,500, and each adds per-host rate limiting and a dedicated/ sharded
executor to handle it). But the layer explosion is dropped everywhere the number
actually drives cost and feasibility downstream:

- **Embedding cost.** operational-simplicity sizes the *full* embed at "~865k
  texts → single-digit-to-low-tens of dollars"; agentic-serving says "embedding
  865K+ layers." The initial full re-embed at 7,500 servers is likely ~4× that
  corpus, so the first-snapshot GPU bill (and its wall-clock) is materially
  under-sized in two drafts. rigor and geospatial *acknowledge* "likely millions
  of layers" / "scaling with a quadrupling seed list" verbally, but neither
  quantifies it or propagates it into a revised cost.
- **Serving feasibility.** geospatial's DuckDB-WASM-in-browser proposal is
  benchmarked in its own red-team against "865k rows"; the real target is ~4M
  rows, which is a different feasibility question for an in-browser engine.
  Index memory on `cpu-basic`, HNSW build time, and the Dataset Viewer's practical
  row/preview limits are all sized (implicitly) against the old count.
- **Storage per snapshot.** ~5–10 GB/snapshot is quoted from the current
  artifacts; at ~4M layers plus embeddings the per-snapshot footprint is larger,
  which changes the Xet-dedup and Bronze-retention math each draft leans on.

This matters because it is precisely the "a headline number's semantics —
units, scope/inclusion basis — are claims the arithmetic gate cannot check"
failure mode: every draft's cost and feasibility section is internally
consistent but anchored to a corpus size its own premise has already outgrown.
The synthesis must **re-baseline the entire cost/feasibility envelope on a
defensible ~3–4M-layer estimate** (ideally bounded early by crawling one state
shard and measuring real layers-per-server at current scale), and treat every
865k-derived figure in these drafts as a lower bound to be revised, not a
starting point to inherit.

*Secondary shared gap worth flagging in the synthesis:* three of the four
(agentic and geospatial heavily, rigor lightly) build their jurisdiction/FIPS
facet-recovery — the "single highest-leverage change" — on the assumption that
the still-live `.txt` mirror carries the same columnar
`Type/State/County/Town/FIPS/Server-owner` schema as the dead `.csv`. The recon
confirms the `.txt` returns *server addresses* and the terms header, but does
**not** confirm it is a parseable columnar table with those fields (those columns
are attributed to the CSV and the prohibited PDF). Per the posture's
format-reader rule, a real sample of the actual `.txt` must be read and its
schema verified before the FIPS-recovery thesis is treated as executable — if the
`.txt` is a formatted report rather than delimited columns, the highest-leverage
feature needs a different recovery path (URL-derived jurisdiction heuristics, or
asking Elfelt to restore the CSV).
