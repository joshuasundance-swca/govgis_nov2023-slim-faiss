# Adversarial audit — ARCHITECTURE / COST axis

Auditor stance: default-refute. Question — is the proposed architecture
right-sized (neither a distributed system for a one-person hobby project, nor
under-built for 7,500+ servers / ~3–4M layers)? Are cost/ops tradeoffs named with
real numbers, not feature lists? Does it account for who actually operates this
(recon: one person, spare time)?

Source under audit: `docs/rebuild-plan.md` (synthesis draft, 2026-07-20), read in
full. Cross-checked against `docs/ecosystem-recon/*` and live verification of the
three load-bearing infrastructure facts the cost claims rest on (GitHub runner
specs, Actions artifact storage, HF GPU pricing).

---

## Verdict up front

On the *central* right-sizing question the plan is genuinely strong, and default-
refute did not break the spine. The operational-simplicity base lens is chosen
with real evidence (the sibling's "nine-to-ten months of unmerged
dependency/security PRs" is used as the empirical measure of what always-on infra
costs a spare-time maintainer — plan §"Base lens", and it is a fair reading of
`local-findings.md`/`huggingface-findings.md`). The stateless-batch spine, the
"turns completely off, breaking nothing" design goal, the DuckDB-embedded-over-
Postgres serving decision, the 3-repo topology (vs. the 4–5 the other drafts
wanted), the measure-first Stage-1 gate that replaces the scale estimate with a
measurement, and the Minimal/Standard/Full budget tiers are all correct
right-sizing moves. The GPU **dollar** figures are grounded and if anything
conservative: I confirmed HF GPU pricing at L4 ≈ $0.80/h, A10G ≈ $1.00/h, A100
≈ $2.50–3.60/h — the plan's "L4 ≈ $0.80/h, A10G ≈ $1.3/h, A100 ≈ $4/h" (§Grounding
Correction 2, §Dimension 9) all round *up*, which is the safe direction. The
re-baselining to ~3–4M layers, and the conclusion that a quantized ANN index is
mandatory at that scale, are the right calls and were missing from the drafts.

That said, default-refute found six real gaps, all on the COST/OPS side and one at
the heart of the serving-engine decision. The through-line: **the plan meticulously
costs the recurring *machine* burden and almost entirely ignores the *human build*
burden and two infrastructure ceilings its "$0" claims depend on.** The most
dangerous of these is that the single largest cost of this entire endeavour — the
months of solo implementation labour — appears nowhere in Dimension 9.

---

## Finding 1 (HIGH) — the dominant cost, upfront build labour, is entirely absent from the cost dimension

Dimension 9 is titled "Cost and ops burden" and it accounts for exactly three
things: recurring dollars (GPU embeddings, storage), and recurring human time
("**Human burden: ~1–2 hours per quarter**" — §Dimension 9). There is **no
estimate anywhere in the document of the effort to build Stages 0–5**, which is
the single largest cost of the project and its single largest risk.

Look at what Stage 0–4 alone requires one spare-time person to design, implement,
test to a fail-closed gate, and wire into CI before the *first* snapshot ships
(§Dimension 7):

- a versioned schema/contract registry (Pydantic v2 + Arrow/pandera, §Dimension 3);
- a **tolerant section-and-entry parser** for Elfelt's ragged prose `.txt`
  (§Grounding Correction 1 — this is genuinely hard: jurisdiction from section
  structure, state from sub-headings, agency from free-text heading lines);
- the `restgdf[resilience]` crawl with per-host rate limiting, a `crawl_outcomes`
  ledger, content-addressed raw landing, and count reconciliation (§Dimension 2);
- the typed transform with content-addressed BLAKE2b IDs, promote-vs-contract
  schema splitting, honest/lossless geometry with native-CRS retention, and
  **FIPS/county derivation via a Census TIGER spatial join plus host/agency
  heuristics** (§Dimension 3, §Grounding Correction 1);
- per-record `license_status` classification with redistribution gating
  (§Dimension 1);
- the embedding pipeline on HF Jobs with changed-rows-only diffing;
- a **quantized ANN index** build (§Dimension 3, Stage 3);
- the checksummed **manifest with lineage chain** and per-snapshot `diff/`
  (§Dimension 4);
- the fail-closed **drift panel of ~10 scripted gates** (§Dimension 6);
- and the matrix-sharded crawl + weekly seed-delta bot automation (§Dimension 6).

That is a multi-person-month engineering programme. The plan's own headline
evidence for the simplicity lens actively *predicts this will be the failure
point*: a maintainer whose scarce spare time left "nine-to-ten months of unmerged
dependency/security PRs" on an existing repo (§"Base lens") is precisely the
maintainer least likely to complete a Stage 0–6 build. The budget tiers
(Minimal/Standard/Full, §Dimension 9) look like they mitigate this but do **not** —
they tier *runtime cost and features*, not build effort, and even "Minimal" is
"the stateless pipeline + one Xet dataset repo + … quarterly manual re-crawl,"
which still requires Stages 0–4, i.e. essentially all of the hard engineering
above. There is no genuine "smallest useful thing I could ship in a weekend"
increment, and no honest statement that the build itself is the cost that matters
most for a one-person project.

A cost dimension for a spare-time solo project that omits the build-labour cost
has skipped the number that actually decides whether this ever exists. This is the
axis's biggest gap.

Evidence: §Dimension 9 ("Human burden: ~1–2 hours per quarter"; "One-time costs:
the Elfelt email (minutes) … a real legal review"); §Dimension 7 (Stages 0–6);
§"Base lens" (the 9–10-month PR backlog as the operator-capacity measure).

---

## Finding 2 (MEDIUM) — the serving-engine decision is internally inconsistent: DuckDB VSS cannot deliver the quantized footprints that make the index "fit free tier"

This is the sharpest technical gap, and it sits inside the plan's own "Resolved
conflict #7" (serving engine at scale). The plan makes two claims that are in
tension at ~3–4M layers:

1. The quantization that makes the index fit a free 16 GB Space (§Grounding
   Correction 2): "fp16 ≈ 7 GB (still tight); **int8 scalar-quantized ≈ 3.5 GB
   (fits); PQ ≈ 0.3–0.9 GB (comfortable)** … an ANN index with quantization
   (HNSW/IVF + SQ8/PQ) is MANDATORY." Those SQ8/PQ footprints are **FAISS-shaped**
   numbers (IVF-PQ / scalar-quantizer compression).
2. The recommended read engine (§Dimension 5, and Conflict #7): "**embedded DuckDB
   (spatial + FTS + VSS/HNSW)** reading the pinned plain-parquet directly …
   DuckDB's VSS builds a one-time HNSW."

The problem: DuckDB's VSS extension builds HNSW over **full `FLOAT[N]` array
columns** — it does not implement product- or scalar-quantized compressed storage.
An HNSW in DuckDB VSS over 3.5M × 1024 therefore stores the vectors at ~14 GB
(fp32) or, at best, fp16-ish — **not** the int8-3.5 GB or PQ-0.3–0.9 GB the
feasibility argument relies on. To actually hit those compressed footprints you use
FAISS (which is exactly what this repo's Space path uses — §Dimension 5 "thin
consumer of a checksummed native index" = `index.faiss`). So the "**one embedded
engine** does hybrid" ops-minimalism story (§Dimension 5: "deliverable in one
embedded engine") quietly becomes two engines at true scale — DuckDB for
spatial/FTS/structured filtering, FAISS for the quantized vector re-rank — which is
more moving parts than the plan's cost/ops case assumes. Additionally, DuckDB VSS's
*persistent* HNSW index has been an experimental, opt-in feature, i.e. an
unvalidated dependency for the "build the index as a snapshot artifact and serve it
read-only" model the plan leans on.

The plan has a partial escape hatch (Stage 3 gate: "index memory fits the target
serving envelope (measured)"; Postgres kept as escalation behind `govgis-core`),
but the *recommendation* and the "mandatory quantized ANN" conclusion are stated as
settled (Conflict #7 "Resolved by re-baselining") on top of a quantization/engine
pairing that does not compose as written. This should be reframed as: DuckDB for
the structured/spatial/FTS tier, a **FAISS** (or equivalent) quantized index for
the vector tier, and an honest acknowledgement that the serving surface is
two components, not one.

Evidence: §Grounding Correction 2 (SQ8/PQ footprints); §Dimension 5 ("embedded
DuckDB (spatial + FTS + VSS/HNSW)", "deliverable in one embedded engine", "thin
consumer of a checksummed native index"); §Conflicts resolved #7.

---

## Finding 3 (MEDIUM) — index *build*-time memory (~14 GB of vectors) is never sized against the build environment, and its compute isn't in the "$0 orchestration" line

The plan gates the index's **serve-time** memory ("index memory fits the target
serving envelope (measured)", Stage 3) but never addresses the **build-time**
memory. Training/building a quantized ANN index requires the full vector set
resident: ~3.5M × 1024 × 4 B ≈ **~14 GB of fp32 training vectors** (the plan's own
number, §Grounding Correction 2), plus the graph/codebook overhead on top. I
confirmed the standard GitHub-hosted Linux runner is **4 vCPU / 16 GiB RAM**
(upgraded Dec 2023) — so a 14 GB working set plus Python/DuckDB/FAISS overhead is
tight-to-infeasible on the free runner the plan assumes for "gates, manifest"
work. Dimension 9 books "Orchestration & CPU (crawl shards, transform, slimming,
manifest, gates): ~$0" and §Dimension 6 keeps "everything inside free managed CI" —
but if the index build must move to a paid high-memory or GPU HF Job (it can
plausibly ride the same HF Jobs GPU the embeddings already use), that is real
compute the "$0" line does not carry, and it needs its own measure-first gate the
way the crawl got one (Stage 1). As written, the plan measures crawl wall-clock and
serve-time index fit but leaves the one other resource-bounded step — index
construction — unmeasured and implicitly free.

Evidence: §Grounding Correction 2 ("~14 GB of raw vectors"); §Dimension 7 Stage 3
gate ("index memory fits the target serving envelope"); §Dimension 9
("Orchestration & CPU … ~$0"); §Dimension 6 ("no self-hosted runner"). Runner spec
verified live (GitHub docs / changelog: standard Linux runner 4 vCPU, 16 GiB).

---

## Finding 4 (MEDIUM) — multi-GB crawl artifacts blow the free Actions artifact-storage quota, contradicting the "$0 orchestration" claim

The crawl design is "matrix-sharded across parallel free runners … each writing
its shard's raw output as a **job artifact**, followed by one lightweight combine
job" (§Dimension 6), and Dimension 9 books orchestration at "~$0" because "GitHub
Actions is free for public repos." The free-minutes claim is correct (I confirmed
public-repo Actions minutes are unlimited/free), **but artifact storage is not**:
the Free plan includes only **500 MB** of Actions storage, then **$0.25/GB/month**
(verified live). The raw crawl output is multi-GB even today — `jsonfiles.tar.gz`
is **898.61 MB** compressed for 1,684 servers (`huggingface-findings.md` §2) — so
at the plan's own ~4× re-baseline the sharded raw passed shard→combine through
Actions artifacts is on the order of **~3–4 GB compressed / 10 GB+ uncompressed**,
6–20× over the free quota. This is a genuine (if modest-dollar) hole in the "$0"
claim, and more importantly an ops footgun: the well-documented
"Artifact storage quota has been hit" failure would silently break refreshes. The
fix is easy and should be in the plan — stream each shard's raw straight to the
private HF raw archive (which the plan already provisions, §Dimension 8 item 2)
rather than through Actions artifacts — but as written the orchestration-is-free
claim is wrong.

Evidence: §Dimension 6 ("writing its shard's raw output as a job artifact");
§Dimension 9 ("Orchestration & CPU … ~$0"; "GitHub Actions is free for public
repos"); `huggingface-findings.md` §2 (`jsonfiles.tar.gz` 898.61 MB @ 1,684
servers). Storage quota verified live (500 MB free, $0.25/GB/mo).

---

## Finding 5 (MEDIUM-LOW) — steady-state ops burden is understated: gate-failure handling and the pipeline's own dependency treadmill are uncounted

"Human burden: ~1–2 hours per quarter" (§Dimension 9) itemises only the happy
path: "Read the drift Issue, approve a snapshot, glance at the gate results and
the manifest diff, approve the promote." Two recurring costs are omitted:

1. **Gate-failure triage — the entire point of the drift panel.** When a gate
   fails, the plan opens an Issue and holds the snapshot (§Dimension 6) — good — but
   *someone then has to diagnose it*, and that is not a glance. Deciding whether a
   "sudden 20-point reachability drop" (§Dimension 6 gate) is a network blip, a
   `restgdf` regression, a shared-runner IP getting rate-limited/blocked by
   government firewalls, or a real upstream change is exactly the kind of
   spare-time debugging that stalls (cf. the 9–10-month PR backlog the plan cites
   as this operator's revealed capacity). The realistic burden isn't "~1–2h/quarter"
   uniformly; it's "~1–2h on a clean quarter, a debugging session on a dirty one,"
   and the plan should say so.
2. **The pipeline repo carries the same dependency/security-PR treadmill it uses
   to disqualify the sibling.** `govgis-pipeline` pins restgdf, pydantic, duckdb,
   faiss, geopandas/pyproj, and the CI actions — all of which will generate the
   same Dependabot/security PRs whose 9–10-month backlog on the sibling is the
   plan's central anti-pattern (§"Base lens", §Conflicts #1, §Dimension 8 item 4).
   An always-*off* batch job can defer those far more safely than an always-*on*
   server — a real and valid distinction — but the plan frames the burden as
   *avoided* ("burden dropped", "the exact burden the … PR backlog measures") when
   it is *reduced*, and never states the residual. A candid cost dimension would
   note that the pipeline still accrues a patch treadmill; it is just lower-stakes.

Evidence: §Dimension 9 ("~1–2 hours per quarter"); §Dimension 6 (fail → open
Issue; the reachability-delta gate); §"Base lens" and §Conflicts #1 ("burden
dropped").

---

## Finding 6 (LOW) — the GPU-hour estimate is internally inconsistent with the plan's own throughput

Grounding Correction 2 states "~3.5M texts through `bge-large-en-v1.5` … at a
realistic GPU throughput of ~500–2,000 texts/s ≈ **~2–7 GPU-hours** for a full
re-embed." The arithmetic from its own throughput gives 3.5M ÷ 2,000/s ≈ 0.49 h to
3.5M ÷ 500/s ≈ 1.94 h — i.e. **~0.5–2 GPU-hours, not 2–7.** The stated range is
~3–4× high. This *errs conservative* (it over-books the one recurring dollar cost,
which is the safe direction, and the dollar figures downstream stay "single-digit
to low-tens" either way), so the practical impact is nil — but it is a
self-inconsistency in precisely the arithmetic class Grounding Correction 2 exists
to police ("the headline number's scope/inclusion basis is a claim the arithmetic
gate can't check"). Worth a one-line correction so the plan's own numbers
reconcile.

Evidence: §Grounding Correction 2 ("~500–2,000 texts/s ≈ ~2–7 GPU-hours");
repeated §Dimension 6 ("~2–7 GPU-hours") and §Dimension 9 ("~2–7 GPU-hours").

---

## What I checked and did NOT find fault with (honest negatives)

- **GPU dollar pricing** — grounded and conservative (verified L4 $0.80, A10G
  ~$1.00, A100 $2.50–3.60 vs. the plan's 0.80/1.3/4.0). No finding.
- **Postgres-avoidance / DuckDB-embedded direction** — correct for the operator
  profile; the always-on DB is rightly demoted to a documented, user-run
  escalation. No over-build here.
- **Repo topology (3 standing surfaces)** — argued down from 4–5 with a real
  change-cadence rationale; right-sized. No finding.
- **Vector footprint arithmetic** (14 GB fp32, fp16 7 GB) — correct.
- **Measure-first Stage-1 crawl gate** — exactly the right de-risking move for the
  scale uncertainty; I only fault that the *index-build* step (Finding 3) didn't
  get the same treatment.
- **Sharding to beat the 6h job ceiling** — the wall-clock reasoning (2h16m for
  2,038 roots → ~8–9h for 7,500, exceeds ~6h) is sound; only the artifact-storage
  consequence (Finding 4) is missed.

The architecture is not over-built for a hobby project and, spine-wise, not
under-built for 7,500 servers. The cost story, however, is materially incomplete
on the human-build axis (Finding 1) and rests on two "$0"/one-engine claims that
don't survive contact with the real infrastructure limits (Findings 2–4).

---

## Verification

Independent adversarial re-derivation by a second verifier (ARCHITECTURE-COST
axis), default-refute. I re-read `docs/rebuild-plan.md` in full (all 1,119 lines,
both pages), re-read the cited `docs/ecosystem-recon/*` evidence directly rather
than trusting the auditor's paraphrase, and independently re-verified the three
load-bearing live-infrastructure facts (GitHub runner spec, Actions artifact
storage for *public* repos, DuckDB VSS internals). Verdict counts: **3 CONFIRMED,
2 ADJUSTED (down), 1 REFUTED.** The refuted one is the auditor's biggest live-fact
claim, and it inverts the finding.

### Finding 1 (arch-cost-01, build labour omitted) — CONFIRMED, HIGH

Confirmed against the text. §Dimension 9 accounts for exactly: recurring dollars
(GPU, storage), recurring human time ("Human burden: ~1–2 hours per quarter"), and
"One-time costs: the Elfelt email (minutes) … a real legal review." I searched the
whole document — there is **no effort/time estimate attached to any stage** in
§Dimension 7 and none in §Dimension 9. The Stage 0–4 build genuinely requires a
solo dev to design/implement/test-to-a-fail-closed-gate: the schema registry, the
tolerant prose-`.txt` section-and-entry parser (§Grounding Correction 1, called
"genuinely hard" and it is — jurisdiction-from-section, state-from-subheading,
agency-from-free-text), the `crawl_outcomes` ledger + content-addressed raw +
count reconciliation, the typed transform with BLAKE2b IDs + TIGER-spatial-join
FIPS derivation + lossless geometry, per-record `license_status` classification,
the changed-rows-only embedding pipeline, a quantized ANN index build, the
checksummed manifest + lineage chain + `diff/`, and the ~10-gate drift panel —
before the first snapshot. I confirmed the budget tiers do **not** provide a
weekend increment: §Dimension 9 "Minimal" is literally "the stateless pipeline +
one Xet dataset repo + the Hub-as-API + the plain-parquet Viewer + quarterly
manual re-crawl," which still transitively requires all of Stages 0–4. The tiers
scale *runtime cost and features*, not build effort — verified. And the plan's own
central simplicity-lens evidence (the sibling's "nine-to-ten months of unmerged
dependency/security PRs," §Base lens) is precisely the revealed-capacity signal
that predicts the multi-person-month build is the likeliest stall point. For a
dimension explicitly titled "Cost and ops burden" on a solo spare-time project,
omitting the single largest cost is the axis's biggest gap. HIGH stands.

### Finding 2 (arch-cost-02, DuckDB VSS / quantization inconsistency) — CONFIRMED, MEDIUM

Confirmed, and I verified the DuckDB internals independently rather than trusting
the claim. The core `duckdb-vss` extension: (a) supports **only FLOAT (32-bit)**
vectors in its ARRAY type; (b) builds HNSW over the **full uncompressed `FLOAT[N]`
column** (usearch-backed) — it implements no product- or scalar-quantized
compressed storage; PQ/SQ/RaBitQ quantizers exist only in a **third-party**
extension (`Icemap/duckdb-vector-index`), not core VSS; (c) its persistent HNSW is
gated behind `SET hnsw_enable_experimental_persistence = true` because WAL crash
recovery is not implemented (documented data-loss/corruption risk), and the index
"must fit into RAM." So the SQ8/PQ footprints the feasibility argument leans on
(§Grounding Correction 2: "int8 ≈ 3.5 GB (fits); PQ ≈ 0.3–0.9 GB") are
**FAISS-shaped** and are *not* achievable in the recommended engine — DuckDB VSS
over 3.5M × 1024 stores ~14 GB fp32, which does **not** fit `cpu-basic` (16 GB
minus the ~1.3 GB bge-large model + overhead). The "deliverable in one embedded
engine" ops-minimalism claim (§Dimension 5, §Conflict #7 "Resolved by
re-baselining") therefore silently becomes two engines at scale (DuckDB for
spatial/FTS/structured; FAISS for the quantized vector tier — which is exactly what
this repo's Space path already uses, `index.faiss`). The quantization-feasibility
argument and the engine recommendation do not compose as written. The plan retains
partial escape hatches (Stage-3 "index memory fits the serving envelope
(measured)" gate; Postgres escalation behind `govgis-core`) and the top-line
conclusion "a quantized ANN index is MANDATORY" remains *true* — it just holds via
FAISS, not DuckDB VSS. Real internal inconsistency at the heart of the serving
decision, with mitigations present → MEDIUM stands.

### Finding 3 (arch-cost-03, index BUILD-time memory unmeasured) — CONFIRMED, MEDIUM

Confirmed. §Dimension 7 Stage 3 gate reads "index memory fits the target serving
envelope (measured)" — **serve-time only**; build-time memory is never sized. I
confirmed the standard GitHub-hosted Linux runner is **4 vCPU / 16 GiB RAM**
(the early-2024 upgrade from 2-core/7 GB), and building/adding a quantized index
over the full set holds ~3.5M × 1024 × 4 B ≈ **~14 GB** of fp32 vectors resident
(the plan's own figure) plus codebook/graph overhead — tight-to-infeasible on a
16 GiB runner. The plan books "Orchestration & CPU (crawl shards, transform,
slimming, manifest, gates): ~$0" (§Dimension 9) and "everything inside free managed
CI … no self-hosted runner" (§Dimension 6), and critically **never states where
the index build runs**. There is a real, unwarned OOM footgun for a maintainer who
builds the index in the free combine/gates CI job as the cost framing implies, and
— unlike the crawl (which got the Stage-1 measure-first gate) — index construction
gets no measurement. I weighed downgrading this to LOW on the grounds that the
index build could ride the already-costed HF Jobs GPU where the vectors are already
resident (zero marginal cost) — but the plan does **not** say that anywhere, so a
maintainer executing the literal text hits the gap. The failure is plausible and
unwarned; a measure-first build gate parallel to Stage 1's is the correct fix.
MEDIUM stands.

### Finding 4 (arch-cost-04, Actions artifact-storage quota) — REFUTED (was MEDIUM → n/a)

**This finding is refuted.** The auditor "verified live" the 500 MB free quota and
the $0.25/GB/mo overage — those numbers are real, but they are the **private-repo /
shared-allowance** figures, and the auditor applied them to a **public** repo. I
verified directly with a GitHub staff answer (community discussion #26438, the
accepted answer, exactly this question): *"GitHub Actions usage is free for public
repositories. The build artifacts in public repo are not counted by storage
limit."* The GitHub billing docs corroborate that public-repo standard-runner usage
is free, and multiple secondary sources confirm public-repo artifacts do **not**
count against any storage quota. The plan's pipeline is explicitly a **public**
repo — the "$0" claim's own predicate is "GitHub Actions is free for public repos"
(§Dimension 9), and `govgis-pipeline` is described as public (§Dimension 8). So the
shard→combine artifacts passing through public-repo Actions incur **no storage
charge and hit no 500 MB quota** — the "$0 orchestration" claim as written is
**correct**, and the finding's central assertion ("6–20× over quota," "$0 claim as
written is wrong," "risking the documented 'Artifact storage quota has been hit'
failure") does not hold. Residual (non-finding) notes, in fairness: individual
uploaded artifacts still have practical size handling, and if the maintainer wanted
the *raw archive itself* kept private they would stream to the private HF raw repo
(which §Dimension 8 already provisions) — but that is an archival-privacy choice,
not the transient shard→combine plumbing this finding is about, and it does not
resurrect a cost gap. The only world where this finding bites is one where
`govgis-pipeline` is made private, which contradicts the plan's stated topology.
REFUTED; severity n/a.

### Finding 5 (arch-cost-05, ops burden understated) — ADJUSTED (MEDIUM → LOW)

Real but over-rated. Both sub-claims are factually confirmed: (1) §Dimension 9's
"~1–2 hours per quarter" itemises only happy-path approvals and does not count
gate-failure triage — and a failed gate like the "20-point reachability drop"
(§Dimension 6) genuinely needs diagnosis (network blip vs restgdf regression vs
shared-runner IP block by a government firewall vs real upstream change), not a
"glance"; (2) `govgis-pipeline` pins restgdf/pydantic/duckdb/faiss/geopandas + CI
actions and will accrue its own Dependabot/security-PR trickle — the same *class*
of burden the 9–10-month sibling backlog is used to disqualify (§Base lens,
§Conflict #1), and the plan frames that burden as "dropped"/"avoided" ("burden
dropped"; "the number this avoids") rather than *reduced*, never stating the
residual. That framing gap is real. But I downgrade to LOW because the plan's
architecture legitimately caps most of the residual: the failure mode is
"no new data," "skipping a quarter is safe by construction," so a dirty-quarter
debugging session is **deferrable, not urgent**, and an always-*off* batch repo
carries no idle security exposure — the always-off-vs-always-on distinction the
plan draws captures the bulk of the genuine reduction. This is a candour/
completeness gap (say "reduced," state the residual, budget "1–2h on a clean
quarter, a debugging session on a dirty one") more than a cost error. The auditor's
own prose labelled it "MEDIUM-LOW"; LOW is the honest landing.

### Finding 6 (arch-cost-06, GPU-hours arithmetic) — CONFIRMED, LOW

Confirmed; the arithmetic is exactly right. 3,500,000 ÷ 2,000 texts/s = 1,750 s =
**0.486 h**; 3,500,000 ÷ 500 texts/s = 7,000 s = **1.94 h** → the plan's own stated
throughput yields **~0.5–2 GPU-hours**, not the stated **2–7** (a ~3–4× overbook),
repeated identically in §Grounding Correction 2, §Dimension 6, and §Dimension 9. It
errs *conservative* (over-books the one recurring dollar cost — safe direction — and
the downstream "single-digit-to-low-tens of dollars" figure is unaffected either
way), so practical impact is nil, but it is a genuine self-inconsistency in exactly
the arithmetic class Grounding Correction 2 exists to police. (A charitable read:
real-world wall-clock including model load, tokenisation, batching inefficiency and
I/O could push 0.5–2 h up toward the stated range — but the plan does not attribute
the gap to overhead; as written, throughput and hours simply don't reconcile.)
Worth a one-line fix. LOW stands.

### Net

The auditor's central thesis — the plan meticulously costs the recurring *machine*
burden and under-counts the *human* burden — survives: Finding 1 (HIGH) is solid
and is the real headline. Finding 2 is a genuine, independently-confirmed technical
inconsistency at the serving-engine decision. Findings 3 and 6 are confirmed
(3 held at MEDIUM after weighing the unstated-GPU-mitigation; 6 at LOW). Finding 5
is real but a low-severity candour gap. **Finding 4 is refuted** — it is a
private-repo fact mis-applied to a public repo, and the "$0 orchestration" claim it
attacks is actually correct as written. That refutation is the point of an
independent pass: the auditor's most confident "verified live" infrastructure claim
was the one that broke.
