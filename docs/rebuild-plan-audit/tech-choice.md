# Adversarial audit — TECH-CHOICE axis

Auditor lens: are the plan's crawl / storage / versioning / serving technologies
the **smallest reliable** choices, genuinely justified against real alternatives
(restapi, esri2gpd, Esri's official ArcGIS API for Python, Xet vs LFS,
Postgres+pgvector vs a simpler search index)? Or did the plan default to whatever
was easiest to describe? Default verdict: **refute** until the evidence shows the
axis was handled well. Target: `docs/rebuild-plan.md`, checked against
`docs/ecosystem-recon/upstream-source-findings.md` (comparable-tools table),
`github-findings.md`, and `huggingface-findings.md`.

## What the plan got right (stated up front, so the refutations are calibrated)

Several tech choices are genuinely, not superficially, justified:

- **Crawl tooling (restgdf v3 + `[resilience]`).** The plan compares restgdf
  against every tool in the recon table and the comparison is accurate:
  Esri's ArcGIS API for Python is "heavier and org-administration-oriented"
  (matches `upstream-source-findings.md` §3, "heavier-weight and more oriented to
  full GIS/org administration"); `restapi` is "synchronous/GPL" (matches "GPL-2.0
  … synchronous, not async"); `esri2gpd` is "synchronous" (matches "not async").
  The `restgdf[resilience]` extra it relies on for retry/rate-limiting is real —
  `github-findings.md:92` confirms "`restgdf[resilience]` extra (stamina +
  aiolimiter for retry/rate-limiting)." The async advantage is load-bearing for a
  7,500-host sweep and the two synchronous competitors genuinely can't match it. A
  thin raw-httpx fallback is retained. This choice is *not* a default-to-easiest;
  it survives the axis.
- **Xet over LFS** and **plain-parquet-first over geoparquet** for the Viewer gap
  are both grounded in recon (`huggingface-findings.md:306-311`; the open
  `datasets#6438`) and argued as cost/compat levers, not conveniences.
- **Rejecting Postgres+pgvector as core** is well-argued from the sibling's
  9–10-month unmerged-security-PR backlog (`github-findings.md:70-73`) — the plan
  correctly does *not* default to the heaviest option, and keeps pgvector as a
  documented reversible escalation. That part of "simpler search index vs
  Postgres" is handled well.

The refutations below are therefore concentrated where they actually bite: the
**serving/index layer**, where the plan's chosen technology cannot do what the
plan's own feasibility analysis says is mandatory, and where a second index stack
is stood up without comparison.

---

## Finding 1 (HIGH) — The recommended serving engine (DuckDB VSS) cannot produce the "mandatory quantized ANN index" the plan's own feasibility analysis requires

This is the central tech-choice defect. The plan makes two claims that cannot
both be true of one engine:

Grounding Correction 2 concludes (lines 188-196):

> "3.5M × 1024 dims × 4 bytes (fp32) ≈ **~14 GB** … A flat/brute-force FAISS index
> at that size **does not fit a free `cpu-basic` HF Space** … **at true scale, an
> ANN index with quantization (HNSW/IVF + SQ8/PQ) is MANDATORY** …"

and Stage 3 restates it (line 866): "the **quantized ANN index** (HNSW/IVF + SQ8/PQ,
sized to fit `cpu-basic`)."

But the recommended read engine is DuckDB VSS (lines 621-630):

> "Recommended read engine: embedded DuckDB (spatial + FTS + VSS/HNSW) … it finally
> adds the ANN index both current stacks lack — DuckDB's VSS builds a one-time HNSW
> over the read-only snapshot."

DuckDB's VSS extension builds an **in-memory, full-precision (float32) HNSW** over
`FLOAT[N]` array columns. It supports **neither scalar quantization (SQ8) nor
product quantization (PQ)**, and **IVF is not a DuckDB concept at all** — "IVF +
SQ8 + PQ" is FAISS vocabulary. The plan has attached FAISS-specific index types to
a DuckDB-VSS recommendation, which is internally inconsistent regardless of DuckDB
version: a 14 GB float32 HNSW in DuckDB VSS is exactly the ~14 GB resident
footprint Grounding Correction 2 says does **not** fit `cpu-basic` (16 GB minus
~1.3 GB model/app). DuckDB VSS also historically requires an experimental flag to
persist an HNSW index to disk and loads the whole index into memory on attach —
the least-mature part of DuckDB, chosen as the *load-bearing* serving component
whose reliability the plan asserts but never checks.

Net: the engine named for the ANN index is the one engine that cannot deliver the
quantized ANN index the plan declares mandatory. Either the quantized index must
come from **FAISS** (which does support IVF/PQ/SQ8 and is already in-house), or the
"fits `cpu-basic`" claim is unsupported. The Stage-3 gate "index memory fits the
target serving envelope (measured)" would catch this only at implementation — but
the *technology decision* is already wrong on paper.

## Finding 2 (MEDIUM) — Two independent ANN-index technologies (FAISS + DuckDB VSS) over one embedding set — not the smallest reliable choice, and never compared

The plan simultaneously specifies:

- this repo's Gradio Space consuming `index.faiss` — "a thin consumer of a
  checksummed native index (the modernization plan's Stage-2 target layout:
  `index.faiss` + typed `documents.parquet` + manifest)" (lines 669-673), and
- the MCP/search tier building a **DuckDB VSS HNSW** over the same vectors (lines
  621-630).

That is two separate ANN implementations, built and maintained over the identical
embedding artifact, for one dataset — the opposite of "the smallest reliable
choice." The plan never reconciles them or justifies the duplication. A single
**FAISS** index (IVF/PQ — the quantization the plan needs, the format the Space
already uses, the one that actually fits `cpu-basic`) could serve *both* surfaces:
the Space directly, and the MCP tier by letting DuckDB do only the spatial / FTS /
structured filtering it is genuinely good at, with vector re-rank delegated to
FAISS over the filtered survivors. That single-index option is strictly smaller and
more reliable than standing up DuckDB VSS as a second, less-mature ANN stack — and
it is exactly the kind of real-alternative comparison the axis demands. The plan
defaulted to "one embedded engine does everything" (line 619, "deliverable in one
embedded engine") without weighing the one-index-two-consumers alternative.

## Finding 3 (MEDIUM) — "ANN index MANDATORY" contradicts the plan's own hybrid design, which demotes vector to a post-filter re-ranker that needs neither a 14 GB resident index nor quantization

Grounding Correction 2's "mandatory quantized ANN" argument assumes **vector-first
search over all ~3.5M vectors** (a flat 14 GB scan that won't fit). But Dimension
5's own query model rejects vector-first (lines 611-619):

> "the query model is **spatial predicate (bbox intersect) → structured / FIPS /
> jurisdiction / license filter → full-text … → optional dense-vector re-rank of
> the survivors.** Vector similarity refines an already-relevant, already-in-region
> candidate set instead of being the first and only filter."

If vector similarity only ever re-ranks a **candidate set already reduced by
spatial + structured + FTS filters** (hundreds to low-thousands of rows), then a
brute-force dot-product over the survivors' vectors is sufficient — no 14 GB
resident ANN structure, no quantization, no HNSW build required. The two sections
justify contradictory index requirements: Grounding Correction 2 needs a
quantized ANN because it (implicitly) queries all vectors; Dimension 5 never
queries all vectors. The "mandatory quantized ANN index" conclusion — the single
most consequential tech requirement in the plan — is over-stated relative to the
retrieval design the same plan adopts. At minimum the plan should decide whether
vectors are ever queried before the structured/spatial pre-filter (which would
justify the ANN) or only after (which would not), and size the index technology to
that answer. As written, the flagship "ANN is mandatory, not optional" claim rests
on a search shape the plan elsewhere discards.

## Finding 4 (MEDIUM) — The embedding-model/dimension choice — root cause of the entire footprint problem — is kept at the largest option by default and its re-evaluation deferred to Stage 6, so Stage 3's whole index architecture is built on a number that may be overturned

The 1024-dim `BAAI/bge-large-en-v1.5` is retained by default (Open Decision 7,
lines 1098-1100: "Keep `BAAI/bge-large-en-v1.5` … or evaluate a smaller/current
model in Stage 6"), and evaluating a smaller model is pushed to Stage 6, "only
after a full snapshot ships" (lines 907-915). But the **1024 dimension is the
direct driver** of the ~14 GB footprint, the "does not fit `cpu-basic`" finding,
and the entire "quantization mandatory" cascade (Findings 1-3). A 768-dim model
(bge-base) roughly halves it; a 384-dim model (bge-small-en-v1.5, a credible
retrieval model) cuts raw vectors to **~5 GB fp32** — which fits `cpu-basic` with
headroom and **no quantization at all**, dissolving the Finding-1 contradiction
outright. The smallest-reliable-choice test on the axis applies most sharply to
the component that *causes* the cost: the plan instead keeps the most expensive
embedding model, then engineers quantization + ANN complexity around it, and
defers the one decision that would remove the complexity to after the architecture
is already built. "Both current consumers use it" (the stated reason) is a
migration-continuity argument, not a smallest-reliable-choice argument — and a
from-scratch rebuild with the maintainer's explicit "we don't need to preserve
prior architectural decisions" is exactly the moment to make the dimension choice
*first*, not last. Deferring it means Stage 3 commits index-format, storage-sizing,
and cpu-basic-fit decisions to a 1024-dim assumption Stage 6 may reverse.

## Finding 5 (MEDIUM) — A load-bearing tech-comparison claim ("both current stacks lack any ANN index") contradicts the cited recon and is not sourced

The plan repeatedly justifies adding the new ANN index / DuckDB VSS on this claim
(lines 193-196):

> "Both current stacks lack *any* ANN index (the sibling does a full sequential scan
> `ORDER BY embeddings <=> $1` over 865k vectors; the FAISS blob is flat) —
> untenable for interactive latency and memory at 3.5M."

and again in Dimension 5 (line 628, "it finally adds the ANN index both current
stacks lack"). But the recon characterizes the sibling as loading into "a
**spatially- and vector-indexed** `layers` table" (`github-findings.md:59`) — the
opposite of "lacks any ANN index." pgvector supports both exact (no index,
sequential scan) *and* approximate (ivfflat / HNSW) search, so the plan's claim is
possible — but it directly contradicts the cited recon and is **not** sourced to
any file/line in the sibling (`backend/load_data.py` / schema), while the recon's
"vector-indexed" characterization *is* the cited artifact. `github-findings.md`
contains no `CREATE INDEX … USING ivfflat/hnsw` evidence either way (grep for
`ivfflat|hnsw|ORDER BY|<=>` returns nothing). If the sibling already builds a
pgvector ivfflat/HNSW index, a motivating premise for the new engine weakens. This
is exactly the maintainer's own posture trap — a claim about a codebase's
capability stated as fact without reading the real source — applied to a
tech-comparison the plan leans on. The plan should quote the sibling's index DDL
(or its absence) rather than assert it against the recon.

## Finding 6 (LOW) — Xet cross-snapshot dedup is treated as a settled cost lever for the dominant (embeddings) artifact, but the saving depends on stable file layout the plan never specifies

The plan makes Xet "load-bearing, not conveniences" (lines 210-212) on the premise
that "quarter-over-quarter snapshots (mostly-unchanged layers) store deltas, not
full copies" (line 573). That holds well for the mostly-unchanged **metadata**
parquet, but per-snapshot storage is **dominated by the embeddings file**
(4-14 GB vs 1-2 GB metadata; `huggingface-findings.md:164` shows the current
embeddings geoparquet at 5.27 GB). Content-defined chunking dedups a re-written
embeddings parquet **only if** row order and row-group boundaries are preserved so
that the byte ranges of unchanged rows stay chunk-aligned across snapshots — float
vectors don't compress or dedup like text, and a re-embed that reorders rows or
shifts row-group boundaries defeats chunk-level dedup on precisely the file that
dominates the bill. The plan neither specifies stable row ordering for the
embeddings file nor quantifies the expected dedup ratio on it; it asserts the
saving. This doesn't sink the Xet choice (Xet is still correct vs plain LFS), but
the headline cost claim ("marginal cost ≈ the changed delta," line 985) is
unquantified for the artifact that actually sets the cost, and rests on a layout
invariant the plan doesn't state as a requirement.

---

## Verdict

The **crawl** and **storage/versioning** tech choices survive the axis: restgdf,
Xet, plain-parquet-first, and the Postgres-as-non-core decision are each compared
against real alternatives and grounded in recon. The **serving/index** layer does
not: the recommended engine (DuckDB VSS) cannot produce the quantized ANN index
the plan's own feasibility math declares mandatory (Finding 1); two ANN stacks are
stood up over one embedding set without comparison to a single-FAISS-index option
(Finding 2); the "ANN mandatory" premise contradicts the hybrid design that would
make it unnecessary (Finding 3); the cost-driving embedding dimension is kept
largest-by-default with its re-evaluation deferred past the architecture that
depends on it (Finding 4); a load-bearing "no ANN index exists today"
justification contradicts the cited recon and is unsourced (Finding 5); and the
dominant storage artifact's dedup saving is asserted, not shown (Finding 6). The
axis was handled well on crawl/storage and **not** well on the serving-index
technology, where the plan reached for the most-describable single-engine story
(DuckDB does spatial+FTS+vector) without checking that its chosen engine can
actually carry the vector load its own analysis defines.

---

## Verification

Independent adversarial re-derivation by a second verifier (TECH-CHOICE axis).
Default posture: refute each finding unless personally confirmed against the plan
text (`docs/rebuild-plan.md`, re-read in full) and the cited recon
(`docs/ecosystem-recon/*`, re-read for every load-bearing citation). Line numbers
below re-checked against the current file.

### Finding 1 — DuckDB VSS cannot produce the mandatory quantized ANN index → **CONFIRMED (HIGH)**

Re-checked the three anchor claims. GC2 (lines 190-196) does conclude "an ANN
index with quantization (HNSW/IVF + SQ8/PQ) is MANDATORY, not the optional
Stage-7 refinement." Stage 3 (line 866) restates "the **quantized ANN index**
(HNSW/IVF + SQ8/PQ, sized to fit `cpu-basic`)." The recommended read engine
(lines 621-630) is "embedded DuckDB (spatial + FTS + VSS/HNSW)… DuckDB's VSS
builds a one-time HNSW over the read-only snapshot." All three verbatim as the
auditor quoted.

The technical core holds. DuckDB's VSS extension builds an in-memory HNSW (via
usearch) over fixed-size `FLOAT[N]` `ARRAY` columns at full float32 precision; it
exposes no scalar-quantization (SQ8) or product-quantization (PQ) knob in
`CREATE INDEX`, and IVF is not a DuckDB concept — "IVF/PQ/SQ8" is FAISS
vocabulary. The persistence caveat is also accurate: HNSW persistence to a DuckDB
file historically required `SET hnsw_enable_experimental_persistence = true` and
the index is memory-resident on load. So a 3.5M×1024 float32 HNSW in DuckDB VSS is
~14 GB resident — exactly the footprint GC2 says does *not* fit `cpu-basic`. The
engine the plan names to "finally add the ANN index" is the one engine that cannot
produce the *quantized* index the plan declares mandatory; that index has to come
from FAISS (in-house, supports IVF/PQ/SQ8), which the plan also invokes for the
Space (lines 669-673) without reconciling the two. This is a genuine internal
contradiction in the flagship serving decision (presented as resolved in Conflict
7), and it undercuts the "cheap, sleep-when-idle" ops thesis for the DuckDB tier
(a 14 GB-resident HNSW needs a large-memory host). HIGH is correct — it is the
central serving-layer tech choice and it is self-inconsistent on paper.

### Finding 2 — Two ANN stacks (FAISS + DuckDB VSS) over one embedding set, uncompared → **CONFIRMED (MEDIUM)**

Confirmed both surfaces are specified: the pipeline builds a FAISS `index.faiss`
snapshot artifact the Gradio Space consumes (lines 669-673), and the DuckDB VSS
tier builds its *own* HNSW over the same vectors (lines 621-630). DuckDB VSS's
HNSW is a distinct structure from a FAISS index, so this is genuinely two ANN
implementations over one embedding artifact for one dataset. The plan never
reconciles the duplication nor weighs the single-FAISS-index-serves-both
alternative (FAISS for vector on both surfaces; DuckDB restricted to
spatial/FTS/structured filtering with vector re-rank delegated). The plan's stated
driver — "deliverable in one embedded engine" (line 619) — is precisely the
un-weighed default the axis targets. MEDIUM is right: it is a real
smallest-reliable-choice miss but a second index is additive complexity, not a
correctness break.

### Finding 3 — "ANN MANDATORY" contradicts the hybrid re-rank design → **ADJUSTED (MEDIUM → LOW)**

The tension is real: Dimension 5 (lines 611-619) makes the query model spatial →
structured/FIPS → full-text → "optional dense-vector re-rank of the survivors,"
explicitly refusing vector-first. If vectors only ever re-rank an already-filtered
candidate set (hundreds-to-low-thousands of rows), a brute-force dot-product over
the survivors' vectors suffices — no resident ANN, no quantization, no HNSW build.
The plan does not reconcile that its own primary live-search shape removes the need
for the resident ANN it calls mandatory.

But the finding overstates the contradiction. GC2's mandatory-quantization
conclusion is genuinely well-founded for a *different* real surface the plan keeps:
this repo's Gradio Space (lines 669-673) is a vector-first semantic-search UI on
the free `cpu-basic` tier, and there a ~14 GB flat fp32 index truly does not fit,
so quantization there is a real requirement — not a claim resting on a discarded
search shape. Two serving surfaces with two query models coexist; the plan's defect
is failing to *scope* the ANN requirement per-surface, not asserting a baseless
one. That is a clarity/scoping gap over a valid requirement, and its distinct
substance beyond Finding 1 (that the DuckDB hybrid tier may need no ANN at all) is
narrow. LOW is the accurate severity.

### Finding 4 — Cost-driving embedding dimension kept largest-by-default, re-eval deferred to Stage 6 → **ADJUSTED (MEDIUM → LOW)**

Confirmed: bge-large-en-v1.5 (1024-dim) is the default (Open Decision 7, lines
1098-1100) and smaller-model evaluation is deferred to Stage 6 "only after a full
snapshot ships" (lines 907-915). The arithmetic checks: 3.5M × 384 × 4 B ≈ 5.4 GB,
which fits `cpu-basic` (16 GB − ~1.3 GB model) with headroom and *no* quantization,
dissolving Finding 1's contradiction. The sequencing critique is valid — the
dimension is the root cause of the whole footprint/quantization cascade, and a
from-scratch rebuild is the moment to choose it first, not last; "both current
consumers use it" is a migration-continuity argument, not a smallest-reliable one.

However, the plan materially bounds the blast radius the finding claims. Embeddings
are deliberately kept in their own file so a re-embed does not churn metadata
(lines 529-534), Stage 3 views are explicitly "regenerable from the core" (line
875), and a full re-embed is ~$ low-tens (lines 979-981) — so reversing a
1024-dim decision after Stage 6 is a cheap, isolated regenerate, not a rebuild.
The finding also underweights that bge-large is a genuine *retrieval-quality*
choice, not only a footprint cost — "smallest reliable" cuts both ways for an
embedding model. The plan surfaces the decision explicitly rather than burying it.
Real and worth acting on, but LOW, not MEDIUM.

### Finding 5 — "Both current stacks lack any ANN index" contradicts the cited recon and is unsourced → **CONFIRMED (MEDIUM)**

Verified independently. The plan asserts, twice as fact, that the sibling "does a
full sequential scan `ORDER BY embeddings <=> $1` over 865k vectors" and that the
new engine "finally adds the ANN index both current stacks lack" (lines 193-196,
628). The cited recon says the opposite: `github-findings.md:59-60` describes the
sibling loading data into "a spatially- and **vector-indexed** `layers` table."
Grep of the entire `docs/ecosystem-recon/` tree for `ivfflat|hnsw|<=>|ORDER BY`
(case-insensitive) returns **zero matches** — so the plan's specific mechanism
claim (`ORDER BY … <=>`, no index) is sourced to nothing in the recon, and the one
characterization the recon *does* give ("vector-indexed") directly contradicts it.
This is the maintainer's own posture trap (a claim about a codebase's capability
stated as fact without quoting the source DDL). Consequence is bounded — the
Postgres-as-non-core decision stands on the PR-backlog evidence regardless — but
the claim is a repeated, headline tech-comparison premise ("finally adds the ANN
index both current stacks lack" anchors Conflict 7 and Dimension 5), and it is
unambiguously in conflict with the cited artifact. MEDIUM stands.

### Finding 6 — Xet dedup asserted, not quantified, for the dominant embeddings artifact → **CONFIRMED (LOW)**

Verified: Xet is "load-bearing, not conveniences" (lines 210-212) on the premise
snapshots "store deltas, not full copies" (line 573) and "marginal cost roughly
the changed delta" (line 986); `huggingface-findings.md:164` confirms the current
embeddings geoparquet at 5.27 GB, dominating the ~1-2 GB metadata. The core point
holds: cross-snapshot dedup of a re-written embeddings parquet requires stable row
order and stable row-group composition so unchanged rows stay byte-identical (a
changed row re-compresses its whole row-group), and the plan specifies neither
stable ordering nor a dedup ratio for the file that sets the bill. One calibration:
the finding slightly overstates CDC fragility — Xet's content-defined chunking
re-syncs at chunk boundaries and tolerates byte *shifts* better than fixed-block
dedup, so a pure append/insert is not fatal; but for compressed parquet a reorder
still churns compressed bytes, so the underlying gap (an unstated layout invariant
under a quantified headline claim) is real. LOW is correct; the Xet-over-LFS choice
itself is unaffected.

### Summary

Confirmed: 1 (HIGH), 2 (MEDIUM), 5 (MEDIUM), 6 (LOW). Adjusted down: 3 and 4
(MEDIUM → LOW) — both are real but each rests partly on framing the auditor
overstated (Finding 3 ignores the Space's retained vector-first surface where the
ANN requirement is genuine; Finding 4 underweights the plan's swappable-embeddings
design and the retrieval-quality dimension). None refuted: every finding traces to
real plan text and, where it cites recon, to the actual recon lines. The auditor's
central verdict — crawl/storage tech survives the axis, the serving-index layer
does not — is upheld, with Finding 1 as the load-bearing defect.
