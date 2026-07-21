# Judge C — critique of the four govgis rebuild-plan drafts

Recorded 2026-07-20. Role: judge, not drafter. I read all four drafts in full,
the four recon files + their synthesis, and `docs/modernization-plan.md` (as a
method reference). Scores are 1–10; for `technicalRisk`, higher = safer /
lower-risk. This file is the written critique; the machine-readable scores are
in the accompanying `StructuredOutput` call.

## Scores at a glance

| lens | dataQ | licensing | maint | serving | cost | risk(safe) | extens |
|---|---|---|---|---|---|---|---|
| data-engineering-rigor | 10 | 9 | 4 | 6 | 8 | 7 | 9 |
| operational-simplicity | 7 | 8 | 9 | 5 | 9 | 8 | 6 |
| agentic-serving | 8 | 8 | 5 | 10 | 8 | 6 | 9 |
| geospatial-native-rethink | 8 | 8 | 4 | 9 | 8 | 5 | 9 |

All four are genuinely good, cover all nine mandated dimensions in real prose,
and converge on a striking amount (uuid5/blake2b deterministic IDs to kill the
`hash()` defect; ingest the `.txt` mirror, never the PDF; keep bbox-extent not
real feature geometry; Xet; plain-parquet sibling for the Viewer gap;
supersede-don't-delete the old DOIs; per-snapshot manifest + drift gates;
`restgdf` v3 as the crawler; a new committed `govgis-pipeline` repo as the
single missing artifact). The divergence is entirely about **what the product
is for** and **how much standing burden the single spare-time maintainer should
carry** — which is exactly where the lenses earn their keep.

---

## data-engineering-rigor — the correctness gold standard, heaviest to run

**dataQuality 10.** This is the clear leader. The medallion split (immutable
content-addressed Bronze → typed/validated Silver → consumer Gold) is the only
draft that treats raw crawl output as a first-class, checksummed, per-file audit
trail rather than a `jsonfiles.tar.gz` blob. The `crawl_outcomes` table records
*every* seed server's fate (ok/unreachable/forbidden/invalid_response/empty/
timeout) with count reconciliation (`seed_count == Σ outcomes`) — which converts
the original pipeline's silent 560-row attrition into a gated, queryable,
drift-detectable number. Content-addressed BLAKE2b IDs with tested determinism +
collision-checking + referential integrity, a versioned schema registry, Arrow
struct promotion for high-value nested fields (not `astype(str)`), preserved
native extent + native CRS for auditable reprojection, and a gate at every hop
that fails closed. Nothing else comes close on correctness.

**licensingCompliance 9.** The only draft that makes license status *operative*,
not merely captured: a per-record `license_status` enum
(`federal_public_domain` / `declared_open` / `declared_restricted` /
`unverified_state_local`) that **gates redistribution scope at publish time** —
ambiguous rows default to link/index-only, widening is a recorded decision.
Plus the unit-tested no-PDF guard, checksummed seed provenance, and a
fail-closed `commercial_use_authorized` flag. Docked one point only because the
enum's federal-vs-nonfederal classification leans on seed `Type`/URL heuristics
that are themselves an error source (mis-classify federal → over-redistribute).

**maintainability 4.** Its Achilles heel, and it says so ("ease of setup is
explicitly subordinated"). Three code/serving repos + a private Bronze store
(4 storage surfaces), a schema registry, a gate suite, a drift panel, medallion
transforms, *and* extending the sibling Postgres+pgvector+MCP server (inheriting
its documented 9–10-month patch backlog). For a single spare-time maintainer
this is the most that could rot. It does make stages idempotent/resumable, which
helps, but this is the lowest of the four and it is the decisive weakness given
the recon's clearest single measurement of what always-on infra costs *this*
maintainer.

**servingUtility 6.** Correct in shape (one validated Gold, thin swappable views:
bulk, sibling pgvector+MCP, thin FAISS Space) but under-designed. It defers the
agent surface to the sibling's existing single `gis_layer_search` tool, adds no
jurisdiction facets to serving, and never confronts that both current stacks
have no ANN index. Serving is treated as "views of Gold," not as a design
problem in its own right.

**costRealism 8.** Honest §9 with per-line verdicts (crawl compute, Xet dedup,
incremental embedding, tunable Bronze retention, real pgvector monthly cost).
The accounting is realistic; the burden it accounts for is simply high.

**technicalRisk 7 (safe).** Gate-per-hop fail-closed design is inherently
correctness-protective, and determinism/quarantine/drift-quarantine all lower
the odds of shipping bad data. The offsetting risk is delivery: it is a large
surface to build, and the realistic failure mode is that a spare-time maintainer
never finishes it — producing yet another frozen artifact.

**extensibility 9.** Medallion + schema registry + regenerable-separate
embeddings + a manifest lineage chain make new views, consumers, and model swaps
clean; immutable Bronze means any future re-flatten is always possible.

**Verdict:** the reference standard for data correctness; its cost is a
maintenance burden the single maintainer has already demonstrated (elsewhere) an
inability to sustain.

---

## operational-simplicity — the only spine that survives neglect by construction

**dataQuality 7.** Deliberately less rigorous and honest about it: uuid5
deterministic IDs (fixes the cited defect), a properly typed slim tier, gates
with known-shape bands (~81% empty descriptions, ~3.4% HTML), and a per-run
committed seed snapshot for resilience. But the raw tier is "small typed columns
+ one JSON escape-hatch column" rather than a fully typed/medallion treatment,
and full-snapshot-only means no per-server outcome ledger richer than an `error`
column. Sound, pragmatic, not rigorous.

**licensingCompliance 8.** Strong and realistic: stay-in-the-free-lane, an
email-Elfelt Stage-0 gate, no-PDF, attribution travels in the manifest, capture
each source's own license as a typed column, and — uniquely — a committed
`excluded_servers.txt` denylist as a trivial no-infra takedown/opt-out lever
(an idea the rigor draft lacks). It captures rather than *gates* on license
status, and is candid that adjudicating 50 states' copyright in code is the
opposite of simple. A defensible posture, one notch below the rigor draft's
operative gating.

**maintainability 9.** The whole thesis, and it lands. One pipeline repo + one
dataset repo, stateless full snapshots (no incremental state to corrupt), no
server / DB / always-on surface, managed GHA + pay-per-use HF Jobs, a
drift-issue-bot, and gates-block-publish so the worst failure under six months of
neglect is "no new snapshot + one GitHub issue," never "broken data." Best in
class, and the property that matters most against the recon's evidence.

**servingUtility 5.** The deliberate sacrifice. Hub-is-the-API for bulk, the
existing Space (untouched) for search, an optional Gradio-native MCP surface, and
Postgres kept only as an optional user-run deployment. No jurisdiction facets,
no agent tool design, no ANN index, no spatial query. Fine for bulk + basic
semantic search; weakest for the agentic/spatial consumers three of four drafts
argue are the highest-value.

**costRealism 9.** The most concrete and lowest: ~$0 managed CPU, single-digit-
to-low-tens-of-dollars pay-per-use GPU embeddings run only on model/text change,
Xet sublinear storage, no always-on, ~1–2 hrs/quarter human. Honestly flags its
own open risks (re-embed granularity under schema evolution; unmeasured shard
sizing).

**technicalRisk 8 (safe).** Fewest moving parts, statelessness, managed
everything, and a fail-mode of "no data" not "bad data" give it the safest floor.
Minor unknowns (matrix-shard sizing, changed-row re-embed correctness) are
flagged rather than hidden.

**extensibility 6.** The accepted tradeoff: the escape-hatch JSON column keeps
fields recoverable, but full-snapshot-only, no medallion, no schema registry, and
a minimal serving surface mean spatial/agentic capability is bolt-on-later work,
not something the architecture is shaped for.

**Verdict:** the most sustainable and cheapest by a wide margin; thin on serving
and on the spatial/agentic capability that is arguably this data's highest use.

---

## agentic-serving — the best serving design in the set, by far

**dataQuality 8.** Strong on the enrichment axis (re-join the seed's
jurisdiction/FIPS/agency facets — independently identified as the single
highest-leverage change), typed relational parquet, uuid5 content-addressed IDs,
quarantine-not-drop, added centroid + geohash, per-record license fields, and
incremental content-hash crawl. Slightly less rigorous than the rigor draft (no
medallion immutability emphasis, no explicit count-reconciliation gate framing),
but the enrichment + quarantine + determinism package is excellent.

**licensingCompliance 8.** Per-record `copyrightText`/`licenseInfo`/
`license_status`, no-PDF test, keep-free, a Stage-0 commercial-permission gate,
index-not-redistribute, and `get_live_endpoint` handing back the source's own
declared license. Sharpest of the four on naming the SWCA-is-commercial tension
explicitly. Captures rather than gates-on-status, like drafts 2 and 4.

**maintainability 5.** Five repos (`govgis-core`/`-pipeline`/`-service`/`govgis`/
`-raw`) and a whole unified MCP+REST+UI+engine service to keep running. It
materially helps itself by recommending *embedded* DuckDB over always-on Postgres
(no DB to babysit) and decoupling publish from promote — but incremental crawl
reintroduces the stateful complexity operational-simplicity deliberately avoids,
and a bespoke service is standing surface. Middle of the pack.

**servingUtility 10.** The standout, and it is not close. A composable, typed,
provenance-carrying MCP tool set (`search` / `get_layer` / `list_facets` /
`get_live_endpoint` / `whats_changed`) returning structured JSON instead of
markdown blobs; `list_facets` specifically to stop agents hallucinating filter
values (a correctness feature, not a convenience); `get_live_endpoint` bridging
the index to live features under the source's terms; output-safety written into
the contract because retrieved gov metadata is untrusted LLM input; and the
genuinely caught defect that **both** current stacks do a full sequential scan
over 865k vectors with no ANN index. One backend, four faces, no consumer keeps
its own copy.

**costRealism 8.** Honest — incremental keeps steady state cheap, Xet, embedded
DuckDB kills idle DB cost, publish/promote decoupled, the DuckDB-vs-Postgres bet
stated plainly. Slightly less quantified than operational-simplicity (no dollar
figures).

**technicalRisk 6 (safe).** The embedded DuckDB VSS/HNSW recommendation is the
least-proven load-bearing bet in any draft at this scale (the draft itself
concedes DuckDB's "weaker write/persistence story"), incremental crawl adds
stateful failure modes, and building a whole new service is delivery risk.
Mitigated by fail-closed gates, quarantine, publish/promote split, and keeping
Postgres as a documented fallback behind a `govgis-core` engine abstraction.

**extensibility 9.** `govgis-core` as a shared write/read contract, four faces
over one engine, a swappable engine seam, the facet schema, the diff artifact,
and the tool set all compose cleanly.

**Verdict:** the serving/agent design the synthesis should adopt wholesale;
heavier to build, more repos to carry, and resting on a somewhat unproven
embedded-ANN bet.

---

## geospatial-native-rethink — the most correct reframing, the highest ambition/risk

**dataQuality 8.** Distinctive and strong: FIPS/jurisdiction recovery (shared
highest-leverage change), per-record license capture, a typed rich tier, blake2b
deterministic IDs, both native + 4326 CRS stored, and the best geometry-honesty
treatment — STAC-style bbox, captured `geometryType`, and `geometry_valid=false`
with reason codes so the 560 rows the original silently dropped are *retained*.
On par with agentic-serving, a hair behind rigor on medallion/gate-per-hop
formalism.

**licensingCompliance 8.** The most legally sophisticated framing: two-tier
licensing that separates the Elfelt-derived *list*'s terms from *our crawled
contribution*, the only draft to mention honoring `robots.txt`, per-record
`copyrightText`/`licenseInfo`/`accessInformation` capture, and snapshot-specific
DOI citation. Slightly muddied by "real geometry column" language and
STAC-assets-as-live-services, which flirt with a redistribution ambiguity the
draft otherwise handles well.

**maintainability 4.** The heaviest headline architecture: *extend* the sibling
Postgres/PostGIS/pgvector (inheriting the exact backlog operational-simplicity
flags as the #1 burden) **plus** a STAC API (another always-on service) **plus**
a hybrid query pipeline **plus** a DuckDB-WASM path **plus** three new repos
**plus** three crawl cadences. The explicit minimal/standard/full **tiers** are a
real escape hatch — the minimal tier (GeoParquet + DuckDB-WASM, no Postgres) is
genuinely low-maintenance — but the *recommended* standard/full vision is the
most to keep patched.

**servingUtility 9.** Second only to agentic-serving, and arguably the more
*fundamentally correct* retrieval model: spatial predicate → structured filter →
full-text → optional vector re-rank, with dense vectors demoted to a re-ranking
facet (right for a corpus that is 81% empty descriptions and has real bbox +
FIPS signal). STAC interop, an upgraded MCP tool, a near-zero-cost DuckDB-WASM
in-browser path, and on-demand live feature fetch. Slightly less crisp than
agentic-serving on the agent *tool contract* itself (no `list_facets`-style
anti-hallucination primitive, less on typed-JSON-vs-markdown).

**costRealism 8.** Good, and the tiered budget framing (minimal/standard/full so
the maintainer picks a spend) is a genuinely useful contribution none of the
others offer. Honest about the Postgres persistent-host cost and DuckDB-WASM's
~zero serving cost.

**technicalRisk 5 (safe).** Highest ambition, highest execution risk. STAC-for-
live-services is acknowledged as unconventional; DuckDB-WASM carrying an
865k-and-growing (see shared gap) catalog in-browser is flagged as needing a
benchmark before betting the cheap path on it; extending the sibling inherits its
security backlog. The draft red-teams its own assumptions better than any other
(its closing "What I would red-team" is the strongest self-audit in the set) —
but the raw surface being bet on is the largest.

**extensibility 9.** Canonical store + projected views + STAC ecosystem interop +
separated embeddings repo + budget tiers extend in every direction.

**Verdict:** the most intellectually correct answer to "what is this data
actually for," and the most ambitious and highest-risk; its full form is the
heaviest to maintain, but its minimal tier and its retrieval model are keepers.

---

## Recommended base lens: operational-simplicity

The binding real-world constraint is not stated in any single dimension — it is
the recon's clearest single measurement: **the most sophisticated stack in this
ecosystem (the Postgres+pgvector+FastMCP sibling) is carrying 9–10 months of
unmerged security PRs under exactly the single-spare-time-maintainer conditions
this rebuild will run under.** A plan whose failure-under-neglect mode is "broken
or compromised data" will, on the evidence, produce another multi-year-stale
artifact no matter how elegant its schema. Only operational-simplicity is
*sustainable by construction*: stateless full snapshots, no always-on surface,
gates-block-publish, worst case = "no new data."

Critically, **sustainability cannot be grafted onto a heavy spine, but rigor and
serving can be grafted onto a light one.** Deterministic IDs, typed schemas,
per-hop gates, a manifest with a lineage chain, jurisdiction facets, and an
agent tool surface are all *additive* to a stateless-snapshot + one-dataset-repo
skeleton without touching its low-maintenance property (facets are just more
typed columns; a serving tool set is a separate consumer that reads pinned
snapshots). The reverse is not true — you cannot make the medallion-plus-two-
services spine cheap by bolting simplicity on.

So: build on operational-simplicity's ops/cost/repo-topology/statelessness spine,
and graft the correctness and serving intelligence the other three supply. This
also honors the maintainer's own posture — own the *outcome* (a dataset that
stays fresh and does not rot), not the task.

## Best specific ideas to graft from the non-base drafts

From **data-engineering-rigor:**
1. The `crawl_outcomes` table + count reconciliation (`seed_count == Σ outcomes`)
   — turn the original's silent 560-row attrition into a gated, drift-detectable
   number. Fits full snapshots at ~zero standing cost.
2. Store raw as immutable content-addressed per-file JSON with checksums (not a
   monolithic tarball) as the replay/audit source, enabling any future
   re-flatten — a light Bronze without the full medallion apparatus.
3. The manifest `prior_snapshot: {snapshot_id, manifest_sha256}` lineage-chain
   link — a verifiable snapshot chain, nearly free to add.
4. Preserve native extent + native CRS beside the reprojected bbox, and validate
   reprojection as a gate, so the transform is auditable.
5. The fail-closed `commercial_use_authorized` flag as a pipeline precondition
   (adopt even if not the full redistribution-gating enum).

From **agentic-serving:**
6. The composable, typed, provenance-carrying MCP tool set (`search`/`get_layer`/
   `list_facets`/`get_live_endpoint`/`whats_changed`) returning structured JSON,
   with output-safety in the contract — the serving design the base lacks, added
   as the optional serving view.
7. `list_facets` specifically as an agent anti-hallucination primitive.
8. `get_live_endpoint` bridging the index to live features under the source's
   terms (reinforces index-don't-redistribute).
9. The caught defect that both current stacks lack any ANN index — mandate HNSW,
   via embedded DuckDB VSS as the low-ops way to get it.
10. Decouple data-publish from serving-promote as two explicit steps.

From **geospatial-native-rethink:**
11. FIPS/jurisdiction/agency facet recovery from the seed's columnar schema — the
    single highest-leverage schema change (converged on by drafts 3 and 4).
12. Honest, lossless geometry: STAC-style bbox + captured `geometryType` +
    `geometry_valid=false`-with-reason-codes-and-retained-native-extent instead
    of silent drops.
13. Hybrid retrieval as the query model wherever a real serving tier runs
    (spatial → structured → full-text → optional vector re-rank; vectors demoted
    to a re-ranker) — the geospatially-correct shape given 81% empty descriptions.
14. The DuckDB-WASM in-browser static search path as a near-zero-infra public
    surface (fits the no-always-on ethos better than a hosted API) — benchmark
    first at real scale.
15. The minimal/standard/full **cost tiers** framing, so the maintainer picks a
    budget explicitly.
16. Two-tier licensing (list terms vs. our crawled contribution) + honoring
    `robots.txt`.

## The single biggest gap all four share

**None of the four grounds its seed-ingestion design in a real sample of the
current `.txt` mirror.** Every draft makes the `.txt` mirror the ingestion path,
and infers its *row* structure from the recon's quotes — but the recon only ever
confirmed the `.txt` returns **prose header content** (the "7,500+…" size line,
the license terms, the "each Wednesday" cadence). The columnar schema every draft
cites (`Type, State, County, Town, FIPS, Server-owner, ArcGIS-url, …`) is
documented from the **dead CSV** (via `restgdf_api/mappingsupport.py`), *not*
from the `.txt`'s actual rows, which no one in the recon or the drafts read.

This is load-bearing twice over. First, for ingestion itself: all four assume the
`.txt` is a parseable, columnar, drop-in substitute for the CSV, and stake the
whole rebuild's first step on it. Second — and this is what makes it the *biggest*
gap — the synthesis's best shared idea, **jurisdiction/FIPS facet recovery**
(drafts 3 and 4, grafts 11), depends entirely on the `.txt` carrying those
per-row columns. If the `.txt` mirror is prose-formatted, or carries only URLs,
or a different column set than the CSV did, then the single highest-leverage
schema change in the entire synthesis is built on an unread format. This is
precisely the maintainer's own posture lesson — *a format-reader is a claim about
a format; read a real sample before concluding a field is present or absent* —
and it should be the very first thing the synthesis de-risks: fetch the current
`.txt`, read its actual rows, and confirm (or refute) the columnar-schema
assumption before committing the facet-recovery design or the ingestion parser to
it. A close-second shared blind spot worth fixing in the same pass: all four
anchor their embedding/GPU/storage/index cost on the old snapshot's **865k
layers**, but a 7,500-server seed (~4.5× the 1,684 crawled in 2023) plausibly
yields **~3–4 million layers** — none re-derives scale, cost, index size, or the
DuckDB-WASM in-browser feasibility against that larger figure.
