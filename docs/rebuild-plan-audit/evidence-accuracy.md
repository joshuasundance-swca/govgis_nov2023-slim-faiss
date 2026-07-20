# Adversarial audit — EVIDENCE-ACCURACY axis

Auditor posture: default-refute. Assume every factual claim in
`docs/rebuild-plan.md` is fabricated, overstated, or contradicts the
`docs/ecosystem-recon/` findings until proven otherwise. Focus: does each claim
trace to something real — the recon, or a primary source I can re-check right
now?

Verdict up front: **this plan is unusually well-grounded on this axis.** The
load-bearing factual claims all trace cleanly to the recon or to primary sources
I independently re-verified, and the plan's two "Grounding Corrections" are
genuine self-catches (one of which I confirmed against the live source, one
against the sibling's real code) rather than embellishment. I found **one real
low-severity numeric inconsistency** (a GPU-hour estimate that does not follow
from its own stated throughput) and **one trivial attribution slip** (stage-0
percentages attributed to "the recon" that actually live in this repo's
modernization stage-0 artifacts). Nothing rises to a correctness-threatening
fabrication or a contradiction of what the recon found.

---

## Independent primary-source re-checks I performed (≥3 required)

**1. Grounding Correction 1 — the seed `.txt` mirror is prose, not a columnar
CSV.** I fetched
`https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.txt`
live during this audit. Result confirms the plan verbatim:

- It is **human-readable report text**, not a CSV. No `FIPS`, `County`, `Town`,
  or `Server-owner` column headers exist.
- Entries use the exact `Website:` / `GIS:` indented-URL format the plan quotes.
  My fetch returned the **Alabama Department of Conservation and Natural
  Resources** block *verbatim identical* to the plan's second example, dated
  annotation included ("`7-30-2023 No tiled data`") — strong evidence the plan
  really fetched the source rather than paraphrasing the recon.
- License terms match verbatim: "Scrapping data from the PDF file is
  prohibited." / commercial-use-needs-written-permission / free-derivative-works
  permitted.
- Size/date/cadence match: "7,500+ ArcGIS server addresses," "June 18, 2026,"
  "usually posted each Wednesday."

This is the single most-load-bearing correction in the plan (it refutes the
columnar-facet-recovery thesis in the agentic/geospatial drafts), and it holds
against the primary source. The recon (`upstream-source-findings.md` §1) had only
ever confirmed the `.txt` returns *prose header* content, never its row format —
so the plan genuinely went past the recon to a primary source, exactly as its
own methodology demands.

**2. "Both current stacks lack any ANN index; the sibling runs a full sequential
scan `ORDER BY embeddings <=> $1`."** This was my prime refute target, because
`local-findings.md` explicitly notes "vector index work" in the sibling's recent
git history — which sounded like it might contradict "no ANN index." I read the
actual code:

- `govgis_nov2023-slim-spatial-server/backend/load_data.py:89-92` creates exactly
  one index: `CREATE INDEX ... ON layers USING gist (geom)` — a **GiST spatial
  index on the geometry column only.** There is **no** `USING ivfflat` / `USING
  hnsw` index on the `embeddings vector(1024)` column anywhere in the load path.
- `backend/models.py:189` issues `ORDER BY "embeddings" <=> $1::vector LIMIT ...`.
  With no ANN index on `embeddings`, pgvector executes this as an exact
  (sequential) scan.

So the plan's claim is **accurate** — the "vector index work" in the git history
never produced a live ANN index on embeddings; only the spatial GiST index
shipped. My refute attempt failed; the claim survives. (Note the plan states the
`ORDER BY embeddings <=> $1` SQL as fact; the recon had flagged `models.py` as
"not fully read" — the plan's assertion nonetheless matches the real code at
`models.py:189`.)

**3. bge-large-en-v1.5 is 1024-dim.** The plan bases its entire vector-footprint
math on 1024 dims. Confirmed against primary source: the sibling's table DDL
declares `embeddings vector(1024)` (`load_data.py:78`). Independent of the model
card, the ecosystem's own schema pins 1024. ✓

**4. Stage-0 corpus stats (81% empty `description`, 3.4% raw HTML).** Confirmed
with real counts in `docs/modernization-plan.md:565-567`: "702,078 of 865,304
records (81%) have an empty `description`, and 29,283 (3.4%) have raw HTML markup"
(702,078/865,304 = 81.1%; 29,283/865,304 = 3.38%). Backed by
`docs/stage0/evidence/html_in_description_example.md`. Numbers are real. (See the
attribution nit in Finding 2 below — these are modernization/stage-0 artifacts,
not the ecosystem-recon.)

---

## Claim-by-claim trace against the recon (spot audit of the high-risk claims)

All of the following trace correctly:

- **Nov-2023 counts** 1,684 servers / 195,479 services / 865,864 layers, and the
  **560-row delta** (865,864 → 865,304): `huggingface-findings.md` §2/§4. ✓
- **Derived ratios**: 865,864/1,684 = 514.2 layers/server; 195,479/1,684 = 116.1
  services/server; 865,864/195,479 = 4.43 layers/service; 1,684/2,038 = 82.6%
  yield. All arithmetic checks out. ✓
- **2h16m crawl for 2,038 roots at `Semaphore(10)`**: `huggingface-findings.md`
  §6. Scaling 136 min × (7,500/2,038) = 500 min ≈ 8.3h → "~8-9h" is internally
  consistent. ✓
- **GitHub Actions ~6h per-job ceiling**: accurate against the well-documented
  GitHub-hosted-runner 6-hour job limit; "free for public repos" is accurate. ✓
- **restgdf v3.0.0, MIT, PyPI, ReadTheDocs, green CI, `[resilience]` = stamina +
  aiolimiter, `Directory.crawl`/`safe_crawl`, Pydantic v2 typed models**:
  `github-findings.md` (restgdf section). ✓
- **Competitor framing** (restapi synchronous/GPL-2.0; esri2gpd not-async/MIT;
  Esri ArcGIS API for Python heavier/org-admin-oriented):
  `upstream-source-findings.md` §3. ✓
- **Dead CSV 404 since 2025-12-15 (issue #74); malformed-row issue #31 "Expected
  15 fields, saw 25"**: `github-findings.md` (restgdf_api). ✓
- **Non-deterministic `hash()` in the parquet path vs. the deterministic
  `random.seed(url); uuid.UUID(...)` Mongo path**: `huggingface-findings.md` §6
  step 4. The plan's framing ("the right instinct existed in-house") is faithful. ✓
- **Geometry = bbox extent reprojected to EPSG:4326, not features; 560 rows
  dropped on reprojection failure**: `huggingface-findings.md` §3/§6. ✓
- **`.geoparquet` unsupported by the Dataset Viewer; `huggingface/datasets#6438`,
  filed by this maintainer in 2023, still open, last activity 2024-02-07**:
  `huggingface-findings.md` §3. ✓
- **Plain LFS, no Xet on both repos; DOIs `10.57967/hf/1368` and `1369`**:
  `huggingface-findings.md` §8/§2/§3. ✓
- **`Lexicom7/EO_Datasets` cites `govgis_nov2023` by name** (used to justify
  never deleting the old repos): `github-findings.md` (identity map). ✓
- **This Space pins the dataset at `ab1220e...` = current `main` HEAD**:
  `huggingface-findings.md` §3. ✓
- **Licensing nuance handled correctly** (the axis-specific trap the task warned
  about): the plan represents federal as public domain (17 U.S.C. §105) but
  state/local as **NOT uniformly public domain**, with NY/SC copyright-permissive
  and FL/CA public-domain — matching `upstream-source-findings.md` §5 exactly, and
  defaulting `unverified_state_local` rows to link/index-only. It does **not**
  overclaim public-domain status anywhere. ✓
- **PDF-scraping tension handled honestly**: the plan explicitly calls the `.txt`
  "the text of the PDF whose scraping Elfelt's own terms explicitly prohibit" and
  labels ingestion permissibility "a genuine open question," elevating the Elfelt
  contact to a Stage-0 gate rather than assuming permission. No overstatement. ✓
- **9-10-month PR backlog on the sibling as "disqualifying" for core infra**:
  `github-findings.md` reports 5 open PRs, of which #76/#73/#71 are Snyk
  (security) base-image upgrades and #75 a dependabot group — so "security-PR
  backlog" is supported (majority of named PRs are Snyk), and "~9-10 months" ≈ the
  gap from the last real commit 2025-10-09 to 2026-07-20 (≈9.3 mo). Fair. ✓

---

## Findings

### Finding 1 (LOW) — GPU-hour estimate does not follow from its own stated throughput

In Grounding Correction 2 (and echoed in Dimensions 6 and 9), the plan writes:
"~3.5M texts through `bge-large-en-v1.5` (1024-dim) at a realistic GPU throughput
of ~500–2,000 texts/s ≈ **~2–7 GPU-hours** for a full re-embed."

The arithmetic does not close. 3,500,000 / 2,000 = 1,750 s ≈ **0.49 h**;
3,500,000 / 500 = 7,000 s ≈ **1.94 h**. So the stated throughput band implies
**~0.5–2 GPU-hours, not 2–7.** Inverting: "2–7 GPU-hours" for 3.5M texts requires
~139–486 texts/s — a band entirely *below* the stated 500–2,000 texts/s. The two
quantities are mutually inconsistent; the GPU-hour figure is ~3.5–4× too high
given the throughput the plan itself asserts.

This is the "headline number's arithmetic is a claim the gate can't check"
failure mode occurring inside the very section built to re-baseline cost
rigorously. Impact is bounded and in the *conservative* direction: the true
compute is cheaper, so the downstream conclusion ("single-digit to low-tens of
dollars," "cents-to-dollars steady state") still holds and is if anything
understated as a cost. Fix by reconciling the two numbers (either lower the hours
to ~0.5–2, or lower the throughput assumption to ~150–500 texts/s and say why).
Severity LOW because it is self-labeled an estimate "to be measured" at the
Stage-1 gate and does not flip any decision.

### Finding 2 (LOW) — stage-0 corpus stats attributed to "the recon" actually live in the modernization/stage-0 artifacts

The plan repeatedly credits "**the recon's** Stage-0 findings (81% of
`description` empty; 3.4% carry raw HTML)" (Dimension 3; also Dimensions 5/6/7).
The **numbers are correct** (verified: `modernization-plan.md:565-567`, real
counts 702,078/865,304 = 81% and 29,283/865,304 = 3.4%, with evidence at
`docs/stage0/evidence/html_in_description_example.md`). But they are **not in the
`docs/ecosystem-recon/` files** — they belong to this repo's separate
modernization Stage-0 work. A reader tracing "the recon" for these figures will
not find them there. The plan elsewhere *does* correctly cite
`docs/stage0/query_set.json`, so this is a loose-attribution nit, not a
fabrication; the underlying data is real and independently confirmed. Severity
LOW (traceability hygiene only).

---

## What I could not independently verify (flagged, not counted as defects)

- The `.txt` **table-of-contents page numbers** the plan quotes ("Federal … p.12
  … State … p.33 … D.C. p.463 … Tribes p.464 … Territories p.465"). My WebFetch
  returned an AI-summarized view that did not surface the TOC page numbers, so I
  could neither confirm nor refute those specific integers. They are incidental
  (not load-bearing), and the structural claim they support — a sectioned,
  jurisdiction-organized prose report — is confirmed. No finding.
- **HF Jobs GPU prices** ("L4 ≈ $0.80/h, A10G ≈ $1.3/h, A100 ≈ $4/h") are
  presented as illustrative "e.g." values and are within the ranges I know HF to
  publish (L4 ≈ $0.80/h and A100 ≈ $4/h are accurate; A10G ≈ $1.3/h is in-range).
  Not independently re-fetched; the cost conclusion is robust to their exact
  values. No finding.
- **"The FAISS blob is flat."** The recon calls it a "legacy LangChain-serialized
  FAISS blob" without naming the index type; LangChain's `FAISS.from_documents`
  defaults to `IndexFlatL2`, so "flat" is a well-grounded inference rather than a
  verified fact. It is consistent with the "no ANN index in either stack"
  argument, which I verified independently for the Postgres sibling. No finding.

---

## Bottom line

On the evidence-accuracy axis the plan is **strong**. Its two grounding
corrections are real, primary-source-backed self-catches (I re-confirmed both —
the prose `.txt` against the live source, and the no-ANN-index claim against the
sibling's actual DDL/query). The licensing nuance — the specific trap flagged for
this axis — is represented faithfully and conservatively, with no public-domain
overclaim. The only substantive defect is a single internally-inconsistent
GPU-hour estimate that errs conservative and changes no decision, plus a trivial
attribution slip on already-correct stage-0 percentages. No fabricated or
recon-contradicting claim survived the audit.

---

## Verification

Independent adversarial re-derivation (default-refute) by a second agent for the
EVIDENCE-ACCURACY axis. I re-read `docs/rebuild-plan.md`, the cited
`docs/modernization-plan.md` lines, and `docs/ecosystem-recon/` myself; I did not
rely on the auditor's paraphrase. Both findings **survive as CONFIRMED at LOW**.

### EA-1 — GPU-hour estimate does not follow from its stated throughput → CONFIRMED (LOW)

Plan text re-read at `rebuild-plan.md:181-184`: "~3.5M texts through
`bge-large-en-v1.5` (1024-dim) at a realistic GPU throughput of ~500–2,000
texts/s ≈ **~2–7 GPU-hours** for a full re-embed."

Re-derived the arithmetic from scratch:
- 3,500,000 / 2,000 texts/s = 1,750 s = **0.486 h**
- 3,500,000 / 500 texts/s = 7,000 s = **1.944 h**
- So the stated throughput band implies **~0.5–2 GPU-hours**, not 2–7.
- Inverting: 2 h needs 3.5e6/(2·3600) ≈ **486 texts/s**; 7 h needs
  3.5e6/(7·3600) ≈ **139 texts/s**. The 2–7 h band therefore requires ~139–486
  texts/s — **entirely below** the stated 500–2,000 texts/s floor. The two
  quantities are mutually inconsistent; GPU-hours overstated ~3.5–4×.

Echo verified by grep — the same "~2–7 GPU-hours" appears three times:
`rebuild-plan.md:182` (Grounding Correction 2), `:733` (Dimension 6), and `:980`
(Dimension 9, the cost section). The finding's "echoed in Dimensions 6 and 9" is
exact.

Severity holds at LOW: the error is bounded and in the **conservative** direction
(true compute is *cheaper*, so "single-digit to low-tens of dollars" and
"cents-to-dollars steady state" still hold — if anything the cost is overstated),
and the figure is self-labeled an estimate to be replaced by a measured number at
the Stage-1 gate (`rebuild-plan.md:161-162, 834-835`). It flips no decision.
Refute attempt failed; the inconsistency is real.

### EA-2 — Stage-0 stats attributed to "the recon" but live in modernization/stage-0 → CONFIRMED (LOW)

Plan text re-read at `rebuild-plan.md:448-450`: "The recon's Stage-0 findings
(81% of `description` empty; 3.4% carry raw HTML) are encoded as schema
expectations and quality-gate bands."

Where the numbers actually live — verified directly at
`docs/modernization-plan.md:564-567`: "702,078 of 865,304 records (81%) have an
empty `description`, and 29,283 (3.4%) have raw HTML markup in `description`
(real example: `docs/stage0/evidence/html_in_description_example.md`)." Arithmetic
re-checked: 702,078/865,304 = 81.1%; 29,283/865,304 = 3.38%. Numbers are real and
correct.

Where they do **not** live — grepped `docs/ecosystem-recon/` for `81%`, `3.4%`,
`702,078`, `29,283`, and description/HTML phrasings: the only hit is
`upstream-source-findings.md:11`, an unrelated note about WebFetch returning an
"AI-summarized paraphrase … rather than a raw HTML dump." None of the recon files
carry these corpus statistics. So a reader tracing "the recon" for these figures
will not find them there; they belong to this repo's separate modernization
Stage-0 work.

This is a loose-attribution / traceability nit, not a fabrication — the data is
real, independently confirmed, and the plan elsewhere correctly cites
`docs/stage0/query_set.json` (`rebuild-plan.md:804, 808`). Severity holds at LOW.
Refute attempt failed; the misattribution is real.

### Verdict

Both findings CONFIRMED at LOW, severities unchanged. Neither is a
correctness-threatening fabrication nor a contradiction of the recon; both are
minor hygiene defects that err conservative (EA-1) or concern provenance only
(EA-2). The auditor's own up-front verdict — the plan is unusually well-grounded
on this axis — is consistent with my independent re-derivation.
