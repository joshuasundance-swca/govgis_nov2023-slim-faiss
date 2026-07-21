# Adversarial audit — COMPLETENESS-STAGING axis

Auditor stance: default-refute. Target: `docs/rebuild-plan.md` (synthesis draft,
2026-07-20). Question: does the plan have a genuinely staged implementation path
with real gates (not a wishlist), a defined test oracle, and rollback thinking —
informed by, not copy-pasted from, this Space's own Stage 0-7 pattern — and are
all 9 mandatory dimensions actually addressed with real decisions?

## Overall verdict

On this axis the plan is **strong, and largely survives refutation.** All 9
mandatory dimensions are present with real, argued decisions (not placeholders):
D1 seed/licensing, D2 crawl/ETL, D3 schema, D4 storage/versioning, D5 serving,
D6 refresh/automation, D7 staged plan, D8 repo structure, D9 cost — each carries
a resolved decision, and the "Conflicts resolved" and "Open decisions" sections
show the choices were made, not hedged. The staged plan (Stage 0-6, Dimension 7)
gives every stage explicit Actions / **Gate** / **Rollback**, defines the test
oracle **once** and reuses it, makes gates fail-closed and script-computed, and
— unusually — includes genuine *negative* tests (Stage 5's "a deliberately-
injected bad refresh is caught and quarantined … the automation is proven to
*fail correctly*, not just to run"; the manifest/consumer oracle that "fails
closed on a deliberately corrupted artifact"). Rollback is real and cheap
("re-pin to the previous tag," "re-run from raw, don't repair," "raw is
append-only"). The parity anchor is not a fiction — I confirmed `jsonfiles.tar.gz`
(raw per-server JSON, huggingface-findings.md:107) and the 865,864-layer count
(README.md:37) exist, so Stage 2's "convert existing raw JSON without re-crawling"
de-risking move is executable. The Stage-0 query set it reuses as the oracle
(`docs/stage0/query_set.json`) exists and holds exactly 23 queries as claimed.

So I did **not** manufacture findings to look thorough. The findings below are
the genuine gaps that survive: they are about **when** validation happens, and
about **which derivations lack an oracle**, not about missing dimensions.

---

## Finding 1 (medium) — the decisive feasibility constraint is validated LATE and its remediation is orphaned in a deferred stage

Grounding Correction 2 makes the plan's single sharpest technical claim: at the
re-baselined ~3-4M layers, a flat fp32 index (~14 GB) "**does not fit** a free
`cpu-basic` HF Space," so "**a quantized ANN index … is MANDATORY, not the
optional Stage-7 refinement**" (lines 186-196). This is presented as a
first-class requirement. But trace it through the stages:

- Stages 2-5 are all built and gated on the **Nov-2023 parity data** (865k
  layers). 865,864 × 1024 × 4 B ≈ **3.5 GB fp32** — which fits `cpu-basic`
  trivially. So Stage 3's gate "**index memory fits the target serving envelope
  (measured, per Grounding Correction 2)**" (line 870) *passes on data four times
  smaller than the scale the requirement was derived for.* The gate's own
  language invokes Grounding Correction 2 while measuring against a corpus that
  never triggers it.
- The real 3-4M-layer index is not produced until **Stage 6's fresh
  7,500-server crawl** (lines 908-909) — and Stage 6 is titled "… **then
  footprint evaluation (deferred)**," folding "deeper quantization, IVF/HNSW
  variants" into an evaluation that runs *only after* a full snapshot ships
  (lines 910-913).
- Stage 5 stands up the embedded-DuckDB serving tier + MCP set "reading the
  pinned snapshot" (line 892) — i.e. the 865k snapshot — and Stage 5's latency
  gate is met against that small index.

Net: the serving architecture is committed, built, and promoted around an index
whose true-scale feasibility is not proven until Stage 6, and the quantization
work that the plan itself calls MANDATORY sits in the same deferred bucket as the
crawl that first needs it. A plausible failure: Stage 6's fresh crawl produces a
~14 GB index that blows the envelope the Stage-5 serving tier was built for, and
the fix is "deferred." The plan would be materially stronger if the mandatory
quantized-ANN path were **built and gated in Stage 3** (against a synthetically
scaled-up or projected vector set), rather than validated implicitly at 865k and
truly tested only at Stage 6.

## Finding 2 (medium) — the Stage-0 Elfelt "launch gate" cannot close the open question it exists to close

The plan elevates contacting Elfelt "**from a recommended nicety to a Stage-0
launch gate**" (lines 146-149, 270-278) precisely *because* it declares the
ingestion path's permissibility "**a genuine open question, not an assumption**"
(line 144) — the sanctioned CSV is dead and the surviving `.txt` is the text of
the PDF whose scraping Elfelt's terms prohibit. But the Stage-0 gate's actual
pass condition is only that contact and a decision were **recorded**: "the
**licensing decision and Elfelt contact are recorded** (the launch gate)"
(lines 824-825). It does not require a *response*, let alone permission.

This matters because the recon establishes Elfelt is effectively non-responsive:
issue #74 (his own project's report that the CSV 404s) has been "**unanswered**"
since 2025-12-15 (README.md:54; plan line 277 even cites it as "his own
project's unanswered request"). So the gate is satisfiable — send an email, log
"we proceed on the reading that parsing the published `.txt` is defensible" —
while the open legal question it was created to resolve stays exactly as open as
before. The plan never specifies the branch when no reply arrives: proceed
unilaterally, or block indefinitely? A launch gate whose success criterion is
"we asked and wrote down our own decision" is a decision-recording gate, not a
question-closing gate; the plan presents it as the latter. Either the gate should
name the explicit fallback decision (and own that it is unilateral), or it should
not be framed as resolving the permissibility question.

## Finding 3 (medium) — no test oracle validates the `license_status` classifier, though it GATES redistribution scope

`license_status` is a legally load-bearing derivation: the plan makes it "**a
typed enum that GATES redistribution scope**" — `federal_public_domain` /
`declared_open` rows "may be redistributed in full," while `unverified_state_local`
/ `declared_restricted` "default to **link/index-only**" (lines 305-315). A
misclassification therefore leaks restricted data into full redistribution.

Yet the test oracle (defined once, lines 786-809) has fixtures for schema
conformance, geometry/reprojection, determinism, manifest/consumer, and
retrieval — and **nothing that asserts the license enum is assigned correctly
against known-answer inputs.** The golden fixtures include "one declaring
`licenseInfo`" (line 792), but its accountable gate is *schema conformance*
(Stage 2 gate, line 855: "100% record schema-conformance"), not "was this row
classified `declared_restricted`?" The only license check anywhere downstream is
the drift-panel band "**License distribution within band**" (lines 759-760) —
which catches a *sudden swing* but is silent on a **systematically-wrong-but-
stable** classifier. A classifier that mislabels a whole class of state/local
rows as `declared_open` from day one produces a stable distribution and sails
through every gate. For a derivation whose failure mode is "redistribute data we
had no right to," the absence of a correctness oracle is a real gap.

## Finding 4 (medium) — the spatial/facet-recovery oracle has no independent ground truth (fixture-is-fiction risk on the highest-leverage new feature)

Grounding Correction 1 demotes Elfelt's CSV (the only authoritative FIPS/county
source) to an optional enrichment and makes **derived** FIPS/county the primary
path: "spatial-join each layer's bbox extent (centroid) against Census TIGER …
plus host and agency-name heuristics" (lines 128-132). The plan repeatedly calls
facet recovery "the highest-leverage change" (line 117).

The oracle meant to validate it (lines 806-809) says the query set carries
"**bbox+FIPS spatial-precision expectations**, grounded against the real Nov-2023
data" and computes "spatial precision/recall for the administrative queries."
But the Nov-2023 dataset carries **no ground-truth FIPS labels** — the CSV that
had them is dead (that is the whole premise), and the plan itself notes bbox
extents are "**coarse** — a statewide layer's extent intersects every county
query" (lines 503-504). So the "expected FIPS" in the oracle can only be
**hand-derived**, and hand-derived by the same class of centroid/heuristic logic
the oracle is supposed to test. This is exactly the maintainer's own documented
"green tests can lie when the fixture is fiction" trap: the oracle would confirm
the derivation against expectations manufactured by the derivation's own method.
The plan needs a *named independent* ground-truth source for the administrative-
query oracle (a curated hand-verified set of layers with confirmed county from
their live service metadata or agency identity, not from centroid-in-polygon),
or it should state that FIPS precision is unmeasured until the CSV is recovered
— rather than implying a real precision/recall gate exists.

## Finding 5 (low) — the Stage-2 parity gate's tolerance band rests on an unverified causal assumption

Stage 2's marquee de-risking gate is "**Nov-2023 parity counts reproduced within
the explained, retained 560-row delta**" (lines 800, 857-858), and the plan
states the cause as settled: "the now-**explained**, retained 560-row
reprojection delta" (line 800), "**reprojection failures**" (line 916 context).
But the recon does not confirm the cause — huggingface-findings.md:140-141 says
the 560 rows are "**likely** layers whose extent failed CRS reprojection/
validation" (emphasis added). The cause is a hypothesis, not a verified fact.

Consequence for the oracle: the gate treats "±560 rows" as the *only* legitimate
discrepancy between the new parser's layer enumeration and the original's. If
the new Pydantic parser enumerates layers even slightly differently from the
original notebook (e.g. group/sub-layer handling), or if some of those 560 rows
dropped for a reason other than reprojection, the count diverges and the gate
**cannot distinguish a correct new behavior from a regression** — it only knows
"not 865,864 ± the number I assumed." The parity oracle should first *inspect the
actual 560 dropped IDs* against the raw JSON to confirm the cause and pin the
expected delta to evidence, rather than importing a "likely" from recon as the
gate's tolerance.

## Finding 6 (low) — Stage 5 reaches across the seam the plan says it never crosses

The plan opens by asserting it "**meets [the modernization effort] at exactly one
seam (Dimension 5) and this document is careful not to reach across it**" (lines
15-16). But Stage 5's actions include "**wire this repo's Space to consume the
checksummed native index**" (line 895-896), and Dimension 5 says the Space
"**becomes a thin consumer of a checksummed native index … built as a snapshot
artifact**" (lines 669-673). Meanwhile the separate modernization plan
independently produces its **own** native index at its Stage 2 (converting the
legacy FAISS artifact) and migrates the Space on its own gated track. Two live
efforts therefore both determine what index the Space loads, and the rebuild plan
never states the ordering/ownership contract: does rebuild-Stage-5 presuppose the
modernization effort has shipped its Gradio app first? Which index is
authoritative if both are ready? For a seam the plan explicitly claims not to
cross, Stage 5 crosses it without a sequencing agreement — a real coordination
gap, low-severity because both are the same maintainer's, but unaddressed.

## Finding 7 (low) — rollback model omits the irreversible outward-facing actions the plan itself introduces

Rollback thinking is strong for data artifacts (re-pin, re-run from raw,
append-only raw) but the plan introduces non-re-pinnable actions and does not
give them a rollback/incident story. Stage 0 sends an email to Elfelt and records
licensing *decisions*; Stage 4 mints new DOIs and posts "supersede / superseded-
by" banners on the old public cards (lines 590-592, 882). None of these is undone
by "re-pin to the previous tag." A wrong `license_status` default that ships
restricted data has only mitigations (the `excluded_servers.txt` denylist and the
link-only default), not a rollback; a premature supersede banner on a
DOI-bearing public repo is a manual reversal the plan never names. This is low
because the conservative defaults shrink the blast radius, but the rollback
sections read as if every stage's worst case is "re-pin," which is not true for
the licensing and DOI actions.

---

## What the plan got right on this axis (credited, not padded)

- Every stage has fail-closed, **script-computed** gates and explicit rollback,
  and the coordinator "**re-runs the actual gate commands**" rather than trusting
  a green build report (lines 780-781) — directly answering the "green gate can
  lie" concern.
- The oracle is defined **once** and reused, and captured "**from the real
  producer, never hand-authored**" (line 787) — the fixture-is-fiction guard is
  explicit (though Finding 4 shows one derivation escaped it).
- Genuine negative/adversarial gates exist: the deliberately-injected bad refresh
  (line 902), the corrupted-artifact fail-closed check (lines 803, 873), the
  no-PDF-URL unit test (line 263). These are what separate a real gate suite from
  a wishlist.
- The Stage-2 "convert existing raw JSON as the first snapshot" parity anchor is
  a strong, executable de-risking move — and its underlying artifact is real
  (verified: `jsonfiles.tar.gz`, huggingface-findings.md:107).
- Publish and serving-promote are decoupled (lines 664-667, 769-771), so a
  passed-but-imperfect snapshot never auto-reaches production — mature rollout
  thinking.
- The validation-gate summary table (lines 1108-1114) gives one fail-closed gate
  per hop with a stated "a bad refresh is caught because…" — the wishlist-vs-gate
  distinction handled well.

All 9 mandatory dimensions are addressed with real decisions; none is a
placeholder. The findings above are refinements to *timing* and *oracle
coverage*, not missing dimensions.

---

## Verification

Independent adversarial re-verification of the seven completeness-staging
findings (CS-01…CS-07). Stance: default-refute. I re-read `docs/rebuild-plan.md`
in full (all 1,119 lines), `docs/ecosystem-recon/README.md`, and
`docs/ecosystem-recon/huggingface-findings.md`, and checked every cited line
against the primary text rather than trusting the auditor's paraphrase.
Verdict summary: **1 ADJUSTED (down), 6 CONFIRMED.** No finding was manufactured;
one was over-scored.

### CS-01 — feasibility constraint validated late — ADJUSTED (medium → low)

Every factual citation checks out. Grounding Correction 2 (lines 186–196) does
call quantized ANN "MANDATORY, not the optional Stage-7 refinement." Stages 2–5
are built on the Nov-2023 parity corpus (Stage 2 line 843 "first snapshot from
existing data"; 865,864 × 1024 × 4 B ≈ 3.5 GB fp32, ~4× under the 3–4M scale the
constraint was derived for). Stage 3's gate (line 870) invokes "measured, per
Grounding Correction 2" while measuring at 865k. The fresh 3–4M crawl is not run
until Stage 6 (lines 908–909), whose title carries "footprint evaluation
(deferred)" and whose deferred bucket holds "deeper quantization, IVF/HNSW
variants" (lines 911–913). Stage 5 stands up DuckDB+MCP around the small index
(line 892). All confirmed.

But the finding's severity rests on "its remediation is orphaned in the deferred
stage" and the plausible failure "Stage 6 produces a ~14 GB index that blows the
envelope … with the fix deferred," and that framing overstates the gap against
two facts the finding did not weigh:

1. **The mandatory remediation is NOT deferred — it is built in Stage 3.** Line
   866 has Stage 3 building "the **quantized ANN index** (HNSW/IVF + SQ8/PQ,
   sized to fit `cpu-basic`)." So the mandatory quantization *technique* is
   implemented at Stage 3; only optimization ("deeper quantization, smaller
   models") is in Stage 6's deferred bucket. A 14 GB fp32 index is not what the
   Stage-3-established build path would even emit at scale — int8 ≈ 3.5 GB, PQ
   ≈ 0.3–0.9 GB (lines 190) both fit.
2. **The envelope constraint is re-gated fail-closed at true scale in Stage 6.**
   Stage 6's gate (lines 916–918) is "the 2026 snapshot passes **every**
   data-quality gate … any footprint change meets the quality floor," and the
   validation-summary table (line 1113) makes "index fits the serving envelope"
   a core→views gate. So the exact failure the finding describes (oversized
   index at 3–4M) is caught by a fail-closed gate that blocks publish, and
   publish/promote are decoupled (lines 664–667) so it never auto-reaches
   production.

What survives is real but smaller: there is no *intermediate* projected-scale
validation between Stage 3 (865k real) and Stage 6 (3–4M real), so the true-scale
behavior of the DuckDB VSS/HNSW build and its latency is first exercised at the
final stage. That is a genuine "validated late" observation for completeness, but
the blast radius is bounded by Stage-3 mandated quantization + Stage-6 fail-closed
re-gate, so **low**, not medium. Not refuted (the late-validation kernel is true);
severity corrected down.

### CS-02 — Stage-0 launch gate can't close the open question — CONFIRMED (medium)

Every citation verified. The plan does elevate contacting Elfelt "from a
recommended nicety to a **Stage-0 launch gate**" (lines 145–147) *because* it
declares the ingestion path's permissibility "a genuine open question, not an
assumption" (line 144). The gate's actual pass condition is only "the licensing
decision and Elfelt contact **are recorded** (the launch gate)" (lines 824–825).
The recon confirms Elfelt is effectively non-responsive: issue #74 "unanswered"
since 2025-12-15 (README.md:54), which the plan itself cites as "his own
project's unanswered request" (line 277). I searched the plan for a no-reply
branch: Open Decision 1 (lines 1076–1080) says contact is "Required before the
first public crawl" but never states what happens if no reply arrives — proceed
unilaterally on the recorded "defensible reading," or block. The gate is
therefore a decision-recording gate that can pass while the legal question it was
created to close stays exactly as open, yet is framed (lines 145–147) as the
mechanism that resolves that question. The missing no-reply branch is a real
completeness omission. Conservative defaults elsewhere (link/index-only,
`commercial_use_authorized=false`) do limit legal exposure, which is why this is
medium and not high — but the framing overstatement + absent fallback branch is a
legitimate medium completeness-staging finding. Confirmed at stated severity.

### CS-03 — no oracle for the `license_status` classifier — CONFIRMED (medium)

Verified end to end. `license_status` "GATES redistribution scope" —
`federal_public_domain`/`declared_open` redistributed in full,
`unverified_state_local`/`declared_restricted` default to link/index-only (lines
303–315). The test oracle (lines 786–809) enumerates fixtures for schema,
geometry, determinism, manifest/consumer, and retrieval, and includes a golden
fixture "one declaring `licenseInfo`" (line 792) — but its accountable Stage-2
gate (lines 853–857) is "100% record schema-conformance," which asserts the enum
value is *valid*, never that it is *correct* against a known answer. The only
downstream license check is the drift-panel "License distribution within band"
(lines 759–760), which catches a sudden swing but is silent on a
systematically-wrong-but-stable classifier. The failure mode is asymmetric and
real: `federal_public_domain` is assigned by heuristic ("section + URL/name
heuristics," line 306) and is redistributed in full automatically, so a heuristic
that mislabels a state/local row as federal leaks restricted data through every
gate with a stable distribution. For a legally load-bearing derivation, the
absence of a known-answer correctness oracle is a genuine gap. Confirmed at
medium.

### CS-04 — facet-recovery oracle has no independent ground truth — CONFIRMED (medium)

Verified. Grounding Correction 1 demotes the CSV to "an enrichment that upgrades
precision if obtained" (lines 130–131) and makes derived FIPS the primary path
("spatial-join each layer's bbox extent (centroid) against Census TIGER … plus
host and agency-name heuristics," lines 128–132), while calling facet recovery
"still the highest-leverage change" (lines 116–117). The oracle (lines 806–809)
claims "bbox+FIPS spatial-precision expectations, **grounded against the real
Nov-2023 data**" with "spatial precision/recall for the administrative queries"
— but the Nov-2023 corpus carries no ground-truth FIPS labels (the CSV that had
them is dead — the plan's own premise), and the plan concedes bbox extents are
"coarse — a statewide layer's extent intersects every county query" (lines
503–504). I confirmed the plan names **no** independent ground-truth source for
the administrative-query oracle; the closest acknowledgment is Open Decision 2
(lines 1081–1084), which offers to "hold the administrative-query tier until the
CSV is recovered" — a decision, not an oracle. So the "expected FIPS" the gate
scores against can only be hand-derived, and at least partly by the same class of
centroid/agency-name heuristic the pipeline itself uses — the maintainer's own
documented "green tests can lie when the fixture is fiction" trap, landing on the
plan's self-described highest-leverage new feature. Confirmed at medium.

### CS-05 — Stage-2 parity tolerance rests on an unverified cause — CONFIRMED (low)

Verified. Stage 2's gate states the cause as settled: "the now-**explained**,
retained 560-row reprojection delta" (line 800), and the gate reads "Nov-2023
parity counts reproduced within the **explained**, retained 560-row delta" (lines
855–857). The recon does not confirm the cause: huggingface-findings.md:141 says
the 560 rows are "**likely** layers whose extent failed CRS reprojection/
validation," and line 279 says "**almost certainly** explains the 560-row
shortfall." Both are hedged; the plan hardened them into "explained" and imports
the precise ±560 as a gate tolerance. If the new Pydantic parser enumerates
layers differently (group/sub-layer handling) or some rows dropped for another
reason, the count diverges and the gate cannot distinguish correct new behavior
from a regression — it only knows "not 865,864 ± the number I assumed." The
finding's fix (inspect the actual 560 dropped IDs against raw JSON to pin the
delta to evidence) is sound. Low is correct: the parity anchor is a de-risking
bonus and the remedy is cheap. Confirmed at low.

### CS-06 — Stage 5 crosses the seam the plan says it never crosses — CONFIRMED (low)

Verified. The plan asserts it "meets [the modernization effort] at exactly one
seam (Dimension 5) and this document is careful **not to reach across it**"
(lines 15–16), and Dimension 8 states the Space is "not otherwise touched here"
(line 943). Yet Stage 5's actions include "**wire this repo's Space to consume
the checksummed native index**" (lines 895–896) — an action on the Space —
and Dimension 5 makes the Space "a thin consumer of a checksummed native index …
built as a snapshot artifact" (lines 669–673), explicitly citing "the
modernization plan's Stage-2 target layout: `index.faiss` + typed
`documents.parquet` + manifest" (line 670). So the rebuild plan itself
acknowledges the modernization effort's Stage-2 produces an index for the same
Space, making two efforts that both determine what index the Space loads. I
confirmed the plan states no ordering/authority contract: it never says whether
Stage 5 presupposes the modernization app has shipped, or which index wins if
both are ready. The natural reading (modernization bootstraps from the legacy
2023 FAISS; rebuild later repoints to a fresh snapshot index) is plausible but
never written down. A real coordination gap; low because both are the same
maintainer's. Confirmed at low.

### CS-07 — rollback omits irreversible outward-facing actions — CONFIRMED (low)

Verified. Rollback is genuinely strong for data artifacts (re-pin, re-run from
raw, append-only). But Stage 0 sends an email to Elfelt (lines 817–820) and its
rollback is "pure code, nothing published" (line 826) — silent on the sent email
and the recorded licensing decision. Stage 4 mints new DOIs and posts
"supersede/superseded-by notes and DOIs" on the old public cards (lines 882,
590–592), yet its rollback frames the worst case purely as additive re-pin:
"publishing is additive (a new tag) … 'Roll back' is 'don't move `latest` to the
new tag'" (lines 887–888) — which does not undo a supersede banner on a
DOI-bearing public card, and DOIs must resolve permanently. A wrong
`license_status` default that ships restricted data has only mitigations (the
`excluded_servers.txt` denylist, the link-only default), not a rollback. The
rollback sections do read as if every stage's worst case is "re-pin," which is
untrue for the licensing/DOI/banner actions. Low is correct: these are
low-frequency and conservative defaults shrink the blast radius, but the
omission is real. Confirmed at low.

### Note on my one downward adjustment (self-check)

CS-01 is the only score I moved, and I moved it *down*, against the advocacy
grain of a verifier looking to confirm. I re-read Stage 3 (line 866) and Stage 6
(lines 916–918) specifically to test whether the "remediation orphaned in the
deferred stage" claim held, because that clause is what carries the medium
severity. It does not hold: the mandatory quantization is built at Stage 3 and
the envelope is re-gated fail-closed at Stage 6. The residual late-validation
kernel is real, so the finding is adjusted, not refuted.
