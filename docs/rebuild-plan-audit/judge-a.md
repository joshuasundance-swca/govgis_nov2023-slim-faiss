# Judge A — adjudication of the four govgis-pipeline rebuild drafts

Recorded 2026-07-20. Inputs read in full before writing: all five
`docs/ecosystem-recon/` files, `docs/modernization-plan.md` (as a *method*
reference), and all four lens drafts under `docs/rebuild-plan-audit/`
(`draft-data-engineering-rigor.md`, `draft-operational-simplicity.md`,
`draft-agentic-serving.md`, `draft-geospatial-native-rethink.md`).

This is a judgment, not a re-draft. Verdict first, then the scored dimensions,
then the base-lens choice, the concrete grafts, and the one gap all four share.

## Verdict

All four drafts converge on the correct **core** — restgdf v3 as the crawl
client, the `.txt` mirror as the seed (never the PDF), deterministic
`uuid5`/`blake2b` IDs to kill the `hash()` defect, a checksummed `manifest.json`,
Xet storage, a plain-`.parquet` Viewer sibling for the `#6438` gap, immutable
snapshot tags, fail-closed data-quality gates, a new committed `govgis-pipeline`
repo, capture-per-record license fields, and an index-don't-redistribute posture.
That convergence is real signal: it is the spine of the rebuild and no draft
disputes it. The four differ almost entirely in **where they put their weight** —
correctness (rigor), sustainability (simplicity), agent-serving (agentic), or
retrieval-shape (geospatial) — and the differences are largely *additive*, which
is what makes a synthesis clean.

The load-bearing grounded fact that should decide the base is not technical: it
is that **every repo in this ecosystem has exactly one committer, and the single
most sophisticated always-on stack in it (the Postgres+pgvector+FastMCP sibling)
is carrying nine-to-ten months of unmerged dependency/security PRs.** That is the
measured cost of standing infrastructure under exactly these conditions. A
rebuild that repeats the over-build fails on the axis the ecosystem is already
visibly failing on. So the base must be the most *sustainable* spine, hardened to
the *correctness* bar the rebuild exists to reach, with the serving lenses
delivered as optional tiers rather than as the core.

## Scores (1–10; technicalRisk: 10 = safest / lowest-risk)

| Lens | dataQuality | licensing | maintainability | servingUtility | costRealism | techRisk(10=safe) | extensibility |
|---|---|---|---|---|---|---|---|
| data-engineering-rigor | 10 | 9 | 5 | 6 | 9 | 7 | 9 |
| operational-simplicity | 7 | 8 | 10 | 4 | 9 | 9 | 6 |
| agentic-serving | 8 | 8 | 6 | 10 | 8 | 7 | 9 |
| geospatial-native-rethink | 9 | 9 | 5 | 9 | 9 | 6 | 9 |

### data-engineering-rigor

- **dataQuality 10.** The strongest correctness treatment of the four: immutable
  content-addressed Bronze, typed Silver with Arrow structs, BLAKE2b
  content-addressed IDs with a *versioned* URL-normalization rule, collision +
  referential-integrity + determinism all asserted as gates, `crawl_outcomes`
  reconciliation (`seed_count == Σ outcomes`), native-extent-and-CRS preservation
  so reprojection is auditable, and a versioned schema/contract registry stamped
  into every manifest. Nothing else comes close on pure correctness.
- **licensing 9.** The only draft that turns license into *enforcement*: a
  `license_status` enum that gates redistribution scope (full vs. link/index-only)
  and a fail-closed `commercial_use_authorized` flag requiring a checked-in
  permission reference, plus a unit-tested no-PDF guard. Teeth, not annotation.
- **maintainability 5.** The weakness, admitted openly. Medallion (three layers,
  immutable Bronze), a schema registry, a gate suite, a drift panel, three repos
  *and* extending the backlogged Postgres sibling is the heaviest surface here.
  It keeps serving optional (bulk-download backbone), which softens the blow, but
  the pipeline itself is a lot for one spare-time maintainer to build and sustain.
- **servingUtility 6.** Correct in principle (one Gold, many thin views) but it
  keeps the sibling's single `gis_layer_search` tool and does not add
  jurisdiction facets, an ANN index, or a composable tool surface.
- **costRealism 9.** Honest, per-line-item verdicts; explicitly admits it costs
  more and says why that is the right trade.
- **techRisk 7.** The *design* is very safe (fail-closed at every hop). The
  *delivery* is the risk: the scope is large enough that a solo maintainer may
  never finish it, which is its own failure mode.
- **extensibility 9.** Registry + Gold-with-many-views + lineage chain +
  decoupled embeddings is built to evolve.

### operational-simplicity

- **dataQuality 7.** Fixes the load-bearing defect (`uuid5` over a canonical key,
  determinism-repro as a gate), types the slim search tier strictly, and ships a
  strong publish-gate set (coverage floor, row-count band, known-shape bands at
  the measured 81% empty / 3.4% HTML baselines, retrieval parity, checksums). But
  it deliberately keeps the raw tier as "small typed top-level + one JSON
  escape-hatch column" rather than promoting high-value nested structures to
  types, and its outcome recording is lighter than rigor's reconciliation. Good,
  not maximal — and every gap here is *additive*.
- **licensing 8.** Practical and real: an Elfelt-email Stage-0 gate, stay-in-the-
  free-lane, per-source license capture as a filterable column, and a genuinely
  clever cheap lever — a committed `excluded_servers.txt` denylist as a
  no-infrastructure takedown/opt-out. Lacks rigor's redistribution-scope
  enforcement, but the denylist is an idea worth keeping.
- **maintainability 10.** The entire design optimizes for the one constraint that
  matters most here. Stateless, self-contained snapshots; no server, no DB, no
  always-on process; neglect survivable *by construction* ("worst failure is no
  new data, never broken data"); one batch repo + one dataset repo; ~1–2
  hours/quarter. This is the property the ecosystem's evidence says the rebuild
  most needs.
- **servingUtility 4.** The deliberate cost of the lens. "The Hub is the API" is
  correct for bulk/analytical consumers but leaves the agentic and spatial
  consumers thin — no ANN index, no spatial predicate, no jurisdiction facets,
  and the MCP surface is hand-waved onto the Gradio Space. Fine as a floor,
  insufficient as the ceiling.
- **costRealism 9.** The most concrete: ~$0 CI, single-digit-dollar HF Jobs GPU
  gated to when text/model changes, Xet, and it even considers-and-rejects CPU
  embedding with reasoning. Honestly flags the unmeasured matrix-shard sizing.
- **techRisk 9.** Fewest moving parts, fail-closed, stateless — and, crucially,
  the one plan a single maintainer can actually finish and keep alive. Lowest
  delivery risk.
- **extensibility 6.** The foundation (deterministic IDs, manifest, tags) enables
  later growth, but it defers incremental crawl and richer serving and uses a
  JSON escape-hatch instead of typed structs, so growth requires adding structure
  later rather than extending it.

### agentic-serving

- **dataQuality 8.** Typed relational parquet with real nested types (not
  `str`), `uuid5` content-addressed IDs, honest `extent_4326` naming + centroid +
  geohash, a raw archival tier, and — the standout — re-joining the seed list's
  jurisdiction/agency columns into typed facets, the single highest-leverage
  schema change for its use case. Slightly less formalized than rigor's registry,
  but the enrichment adds real query value.
- **licensing 8.** Per-record `copyrightText`/`licenseInfo`/`jurisdiction_level`
  capture, a `license_status="unverified-nonfederal"` flag, a launch-gate on
  Elfelt permission with explicit SWCA-is-commercial awareness, and
  `get_live_endpoint` returning the declared license at serve time. Captures and
  flags well; does not gate scope as hard as rigor.
- **maintainability 6.** Picks embedded DuckDB precisely to avoid the sibling's
  always-on Postgres treadmill, and decouples publish from promote — both real
  ops wins. But it still stands up `govgis-core` + a four-face `govgis-service` +
  two dataset repos + a thin UI: a deployed service the maintainer must run and
  patch, heavier than simplicity though lighter than the Postgres path.
- **servingUtility 10.** The best serving design, and the point of the lens: a
  composable, typed, provenance-carrying tool set (`search`, `get_layer`,
  `list_facets` to stop filter-value hallucination, `get_live_endpoint` to bridge
  to live features, `whats_changed`), structured JSON over markdown, output
  safety in the contract, an actual ANN index (correctly catching that *both*
  current stacks do a full sequential scan), and four coordinated faces with no
  consumer keeping its own copy.
- **costRealism 8.** Honest single-maintainer/low-QPS framing, DuckDB to cut idle
  cost, incremental refresh, the DuckDB-vs-Postgres bet stated plainly rather than
  sold.
- **techRisk 7.** DuckDB VSS/HNSW is newer than pgvector (mitigated by a
  documented Postgres fallback behind the `govgis-core` abstraction), and there is
  more to build; the promote-decoupled design keeps a bad snapshot out of prod.
- **extensibility 9.** The `govgis-core` engine abstraction, four faces, and
  composable toolset are built to grow.

### geospatial-native-rethink

- **dataQuality 9.** Typed rich tier (no `astype(str)` firehose), FIPS/
  jurisdiction recovery, `blake2b`/UUIDv5 IDs, honest STAC-`bbox`-plus-extent
  naming with native+4326 CRS preserved, `geometryType` captured, and the
  distinctive **losslessness** move — retain reprojection-failed/degenerate rows
  with `geometry_valid=false` and a reason code rather than silently dropping the
  560 the original build lost. Its Phase-2 "convert the existing Nov 2023 raw JSON
  into the new schema without re-crawling and reproduce the counts" is a genuinely
  strong real-data parity target.
- **licensing 9.** The most legally-precise framing: a **two-tier license**
  separating the Elfelt-derived server-URL list (his terms) from our own crawled-
  metadata contribution (MIT/CC0), recorded per-field in the manifest/card; plus
  per-record capture, robots.txt/per-host/UA politeness, and the "three-way
  convergence" insight (index-don't-redistribute is simultaneously the
  geospatially-honest, licensing-safe, and cheapest choice).
- **maintainability 5.** The heaviest serving footprint of all four in its
  *recommended* form: extend the backlogged Postgres+PostGIS+pgvector sibling,
  add a STAC API (stac-fastapi + pgstac), a DuckDB-WASM path, and on-demand
  fetch. The tiered budget (Minimal = static GeoParquet + DuckDB-WASM, no server)
  is a real escape hatch, but "recommended" points at the most services.
- **servingUtility 9.** The most *correct* retrieval reframe: hybrid
  spatial-predicate + FIPS/structured filter + full-text first, dense vector
  demoted to an optional re-ranker — grounded in the fact that 81% of
  descriptions are empty, so a vector-only index is embedding mostly-blank text.
  Plus STAC interop (honestly red-teamed as an unconventional live-service use),
  an MCP tool with spatial+jurisdiction params, on-demand live-feature fetch, and
  the DuckDB-WASM zero-infra browser path. Slightly less developed on MCP-toolset
  composability than agentic, but the retrieval model is more defensible.
- **costRealism 9.** The tiered Minimal/Standard/Full budget is the most
  cost-conscious presentation, letting the maintainer pick an envelope;
  DuckDB-WASM is ~zero serving cost.
- **techRisk 6.** The most novel and therefore the highest delivery risk:
  STAC-for-live-services and DuckDB-WASM carrying an 865k+ (and growing) row
  catalog in-browser are both unproven and flagged for benchmarking; extending
  the backlogged sibling inherits its risk. Redeemed partly by the best explicit
  self-red-team section of the four and a low-risk Minimal starting tier.
- **extensibility 9.** Canonical store + projected views + STAC interop + tiered
  budgets + separate embeddings repo is highly extensible.

## Recommended base lens: operational-simplicity

Base = **operational-simplicity**, hardened to data-engineering-rigor's
correctness bar and enriched with the serving lenses' schema/views as optional
tiers. The reasoning:

1. **The dominant grounded constraint is sustainability, not sophistication.**
   The recon's clearest measurement is a solo maintainer already drowning in an
   always-on stack's patch backlog. Simplicity's "stateless batch job whose worst
   failure is *no new data, never broken data*" is precisely the fail-closed
   guarantee rigor wants — achieved more cheaply, and the only one of the four a
   single spare-time maintainer can reliably finish and keep alive.
2. **Simplicity's gaps are additive; rigor's gap is subtractive.** You can add
   typed Silver structs, `crawl_outcomes` reconciliation, a license enum, and a
   manifest lineage chain *onto* a stateless one-pipeline/one-repo skeleton
   without dismantling it. You cannot easily lighten rigor's medallion + Bronze +
   registry without gutting its structure. The additive direction is the clean
   synthesis; the subtractive one is a rewrite.
3. **The serving lenses are best delivered as tiers, not as the core.** Agentic's
   four-face service and geospatial's Postgres+STAC hybrid are the highest-burden
   things for a solo maintainer and exactly what the recon warns against as
   *core* infra. Their value (facet enrichment, composable MCP toolset, hybrid
   retrieval, DuckDB-WASM) grafts cleanly as optional, sleep-when-idle or
   zero-infra tiers on a data-first spine — which is how simplicity already frames
   serving, and how geospatial's own Minimal/Standard/Full budget proves it can
   be staged.

Rigor is the primary graft source, not the base: the synthesis is "simplicity's
sustainable spine, hardened to rigor's correctness bar." A reasonable judge could
invert this and base on rigor — the task does say "better data engineering" — but
basing on rigor forces you to carry the medallion/Bronze/extend-Postgres weight
that is the likeliest thing to sink a solo effort, and to fight the design to
lighten it. Basing on simplicity and hardening it is the lower-risk path to the
same correctness.

## Best specific ideas to graft from the non-base drafts

From **data-engineering-rigor**:
1. Content-addressed IDs via a fixed platform-independent hash (BLAKE2b over
   normalized natural keys) with the **URL-normalization rule versioned in the
   manifest**, and determinism tested as a gate (build-twice-assert-identical),
   plus collision + referential-integrity checks — stronger than bare `uuid5`
   because the normalization rule becomes a deliberate, versioned act.
2. The `crawl_outcomes` table: record *every* seed server's outcome (ok /
   unreachable / forbidden / invalid_response / empty / timeout) with
   `response_sha256`, gated on `seed_server_count == Σ outcomes` — turning silent
   attrition into a queryable, drift-detectable, reconciled signal.
3. The `license_status` typed enum that **gates redistribution scope**
   (federal_public_domain/declared_open in full; unverified_state_local/
   declared_restricted link/index-only by default) plus a fail-closed
   `commercial_use_authorized` flag requiring a checked-in permission reference.
4. The manifest **lineage chain** (`prior_snapshot: {snapshot_id,
   manifest_sha256}`) making the snapshot sequence tamper-evident.
5. Promote high-value nested fields (especially `fields`) to **typed Arrow
   structs**, keeping the JSON escape-hatch only for genuinely open-ended
   sub-objects — upgrading simplicity's "type almost nothing" raw tier where it
   matters.
6. Preserve **native extent + native CRS** as separate typed columns beside the
   reprojected bbox so the reprojection is auditable and reproducible.

From **agentic-serving**:
7. **Re-join the seed list's jurisdiction/agency columns** (Type/State/County/
   Town/FIPS/Server-owner) into typed facets in the search schema — the single
   cheapest high-leverage change; simplicity's slim schema drops them.
8. The **composable, typed, provenance-carrying MCP tool set** (`search`,
   `get_layer`, `list_facets` to stop filter-value hallucination,
   `get_live_endpoint`, `whats_changed`) returning structured JSON not markdown,
   with output safety in the contract — delivered as the optional agent tier.
9. **`get_live_endpoint`** (validated https-only URL + declared license) as the
   serving affordance that realizes index-don't-redistribute.
10. **Decouple publish from serving-promote** so a bad snapshot never
    auto-reaches production; promote is a separate explicit pin bump.
11. **Embedded DuckDB (VSS/HNSW + spatial) reading the pinned parquet directly**
    as the serving read path when a service tier is wanted — zero-ETL snapshot
    promotion, a real ANN index (both current stacks full-scan), far lighter than
    Postgres and aligned with the simplicity ethos.

From **geospatial-native-rethink**:
12. **Hybrid retrieval as the default when a search tier exists** — spatial +
    structured/FIPS + full-text first, dense vector as an optional re-ranker —
    grounded in the 81%-empty-description fact that makes vector-only weak.
13. **DuckDB-WASM in-browser static search** over the published parquet — a
    near-zero-cost, no-always-on public search that fits the simplicity thesis
    better than a hosted Space (subject to the flagged in-browser-at-scale
    benchmark).
14. **Losslessness**: retain reprojection-failed/degenerate rows with
    `geometry_valid=false` + native-CRS extent + reason code rather than dropping
    them (the source of the silent 560-row loss) — required for honest
    cross-snapshot diffing.
15. Capture **`geometryType`** as a typed attribute at zero geometry-storage cost
    (high discovery value: "polygon layers covering X").
16. The **two-tier licensing framing** (Elfelt-list terms vs. our own
    crawled-metadata contribution) recorded per-field — the most legally-precise
    framing; pairs with rigor's enforcement enum.
17. The **Phase-2 real-data parity move**: convert the existing Nov 2023 raw JSON
    into the new schema as the first snapshot *without re-crawling*, reproducing
    the 865,864 / 195,479 / 1,684 counts within the now-explained (and retained)
    560-row delta — grounds the oracle on real data before betting on a fresh
    crawl.
18. **STAC** as an optional interop face (server→Catalog, service→Collection,
    layer→Item), prototyped on one Collection first per its own honest red-team.
19. The **tiered budget presentation** (Minimal / Standard / Full) so the
    maintainer picks an ops/cost envelope.

## The single biggest gap all four share

**None of the four resolves the heavy-compute execution at real 2026 scale — the
~6–10h+ full crawl of 7,500 servers *and* the initial full embedding of the
resulting millions of layers — with anything more concrete than an open
decision.** All four correctly identify that the crawl exceeds GitHub Actions' ~6h
job ceiling, and all four then defer it to the same unresolved menu — "a
self-hosted runner, an HF Job, or an ephemeral VM, TBD." Simplicity gets closest
with matrix-sharding but *itself flags shard-sizing as unmeasured*; nobody
benchmarks even a single shard. Worse, the embedding side is under-analyzed by
all four: the corpus is 865k layers at 1,684 servers; at 7,500 servers it is
plausibly 4–5× (millions of layers), and while every draft says "embed only
changed rows / HF Jobs GPU," **none sizes the *initial* full embed, index build,
or serving memory at that new scale** (millions of 1024-dim bge-large vectors is
many GB of vectors plus a non-trivial FAISS/HNSW build and a real cold-start
memory footprint). This matters more than any schema or serving detail because it
is (a) the one genuinely hard, expensive, un-de-riskable step; (b) the true
gating dependency — you cannot produce a *single* new snapshot without solving
it; and (c) left as prose in all four while the easier problems (IDs, manifest,
gates, views) are handled well. The synthesis must close it with a **concrete,
measured, costed execution design** — benchmark one crawl shard at current scale,
size the initial embed/index-build/serving memory at the 7,500-server projection,
and pick-and-test an actual runner — rather than carrying it forward as yet
another open decision. It is the collective blind spot precisely because it is the
part more agents and better schemas cannot make go away; only measurement can.
