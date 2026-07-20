# Adversarial audit — AUTOMATION-RELIABILITY axis

Auditor lens: refresh/maintenance automation. Default verdict: refute until the
plan proves it handled the axis. Target: `docs/rebuild-plan.md` (synthesis
draft, 2026-07-20). Cross-checked against `docs/ecosystem-recon/` and the plan's
own Grounding Correction 2 (the ~3–4M-layer / ~15–30 GB-per-snapshot
re-baseline).

## Verdict in one line

The *content* of the drift/quality machinery is genuinely strong and, on its own
terms, the best thing in the plan — a real fail-closed gate panel with named
checks, a negative test that proves the gate quarantines a bad refresh, and a
credible cadence/compute split. But the *automation that is supposed to run that
machinery unattended* is under-specified in ways that collide head-on with the
plan's stated design goal ("safe to neglect for six months") and with its own
re-baselined scale. Several of these are not "weak," they are "as written, the
automation does not survive its own design premise." I return seven findings,
ranked most-severe first.

---

## Finding 1 (HIGH) — GitHub disables scheduled workflows after 60 days of repo inactivity; this defeats the plan's central "safe to neglect for six months" premise

The plan's spine is stated repeatedly and explicitly:

> "a stateless batch pipeline that produces a good dataset when told to and then
> turns completely off, breaking nothing while it sits untouched for six months."
> (line 40)

> "The worst case for a six-month-unattended pipeline is 'no new snapshot + one
> Issue'…" (line 768)

The entire monitoring system is a cron:

> "**Weekly seed-delta check (near-free).** A light GitHub Actions cron fetches
> the `.txt` mirror…" (line 703)

> "This is the entire monitoring system: zero standing infrastructure…" (line 709)

GitHub Actions has a documented, non-optional behavior: **in a public repository,
scheduled (`on: schedule`) workflows are automatically disabled when there has
been no repository activity (commit/push) for 60 days.** A pipeline that is
"untouched for six months" by design will therefore have its weekly seed-delta
cron — and any quarterly cron — silently disabled at the ~60-day mark, roughly
one-third of the way into the very neglect window the plan is engineered to
survive. When it is most needed (a long gap with no human attention), the "entire
monitoring system" is off, and GitHub's own "your scheduled workflow was
disabled" notice is exactly the kind of low-salience signal a
check-quarterly maintainer misses.

`grep -i '60 day|keepalive|heartbeat'` over the plan returns nothing. The plan
never mentions the auto-disable behavior and provides no keepalive
(e.g. a committing job, `workflow_dispatch` re-arm, or an external scheduler).
This is the single sharpest automation-reliability contradiction in the document:
the design goal and the chosen mechanism are mutually exclusive as written.

Fixable (a weekly commit-to-repo keepalive, or an off-GitHub scheduler such as a
tiny cron on the existing sibling infra, or accepting a re-arm ritual), but the
plan is silent, so as written the neglect premise fails.

## Finding 2 (HIGH) — Free GitHub-hosted runners (~14 GB disk, ~16 GB RAM) are never checked against the re-baselined ~15–30 GB / ~3–4M-layer scale; the combine/transform/index-build steps have no described home that fits free CI

The plan re-baselines the corpus 4× (Grounding Correction 2, lines 151–216):
~3–4M layers, and:

> "~15–30 GB per full snapshot, embeddings dominating." (line 209)

> "3.5M × 1024 dims × 4 bytes (fp32) ≈ **~14 GB** of raw vectors." (line 187)

It then asserts the automation is free with no resource caveat:

> "**Orchestration & CPU (crawl shards, transform, slimming, manifest, gates):
> ~$0.** GitHub Actions is free for public repos…" (lines 975–978)

The *only* free-runner constraint the plan engages is the **6-hour wall-clock
job ceiling** (lines 176–180, 716–728). It never engages the free runner's
**resource** limits: standard GitHub-hosted runners provide roughly 14 GB of
usable SSD and ~16 GB RAM. The sharded crawl produces per-shard raw that must be
recombined (line 720: "one lightweight combine job"), then transformed to typed
core (Stage 2), then embedded and indexed (Stage 3). The combine + transform step
must materialize the full raw set (a large fraction of 15–30 GB) on one runner's
~14 GB disk; the ANN index build must hold ~14 GB of fp32 vectors (the plan's own
number) in ~16 GB RAM before quantization. Neither fits. `grep -i 'disk|RAM'` for
runner sizing returns nothing — the plan checks time but not space.

This is the plan's own diagnosed failure mode turned on itself: it warns that "a
headline number's scope/inclusion basis is a claim the arithmetic gate can't
check" and re-baselines *cost and index memory* on 3–4M — but does not
re-baseline the *free-CI runner feasibility* the whole $0 automation budget rests
on. The embedding is explicitly moved to pay-per-use HF Jobs GPU (line 730)
precisely because it won't fit free CI; the same reasoning was never applied to
the transform-combine and index-build steps, which are left implicitly on free
Actions. As written, the automation likely cannot physically execute at the scale
the plan itself argues is real.

## Finding 3 (HIGH) — The volume-drift gate is calibrated against the "prior snapshot band," but the first real crawl is expected to be ~4× larger; the gate would false-positive and block the very growth refresh it exists to capture

The drift panel's volume check is defined relative to the prior snapshot:

> "**volume drift** (per-table row counts within a band of prior; a 40% collapse
> is a truncated crawl)." (lines 748–749)

> "the fail-closed **drift panel**, compared against the prior published
> snapshot." (line 742)

But the plan's headline premise is 4–5× growth:

> "7,500 seed servers × ~83% crawl yield ≈ **~6,200 crawled servers** … ≈ **~3.2
> million layers** … roughly **4× the 865k baseline.**" (lines 168–171)

Stage 6's first fresh full crawl (lines 909–911) publishes exactly this ~4×
snapshot against the Nov-2023 parity anchor as "prior." A fail-closed volume-drift
band tuned to catch a "40% collapse" will, symmetrically, fire on a +300%
expansion unless the band is explicitly asymmetric and wide on the upside — and
the plan never specifies the band's shape or bounds. The result is either (a) the
first real refresh false-positive-fails the gate and does not publish (the
automation blocks the outcome it was built for), or (b) the band is set so wide
upward that it no longer detects a genuine partial/truncated crawl in the growth
direction. The plan does not reconcile "expect 4× growth" with "fail-closed on
row-count deviation from prior," and gives no rule for distinguishing "legitimate
growth" from "runaway/duplicated crawl." This is a concrete correctness gap in
the gate that runs on every refresh, not a thresholds-are-TBD nicety.

Compounding it: essentially every gate threshold in Dimension 6 is an unbound
placeholder — ">10% new roots" (line 707), "a sudden 20-point reachability drop"
(line 745), "40% collapse" (line 749), "near the ~81% baseline … ~3.4%"
(lines 752–753), all prefixed "e.g." or "a … band." For a *fail-closed*,
*unattended* gate, the concrete threshold values and their upside/downside
asymmetry are the load-bearing part, and they are deferred with only "start
conservative and tune from measured data" (line 362) as method.

## Finding 4 (MEDIUM) — The weekly seed-delta cron's own failure modes are unspecified, so the exact founding failure (a silent upstream feed death) may not reach the notification channel

The project exists because an upstream feed died silently:

> "The exact upstream feed this pipeline depended on is dead." (recon README
> line 49) — the CSV endpoint returns HTTP 404, unnoticed until issue #74.

The weekly cron is the designated early-warning system for this class of event.
Its happy path is specified (fetch, checksum, diff, open an Issue on >10% new
roots — line 703–710). Its *unhappy* paths are not: what does the cron do when
the `.txt` itself 404s, moves, is truncated, or changes format (the precise thing
that happened to the CSV)? The plan states elsewhere that "if the mirror 404s on
a future run, the build fails a gate loudly" (line 268) — but that is the *crawl
build*, which only runs quarterly (and, per Finding 1, may be auto-disabled). The
*weekly* check's behavior on upstream death is not specified. If a fetch failure
merely fails the Actions job (a red ✗ on a repo checked every six months) rather
than opening an Issue, the one channel the plan trusts to reach the maintainer's
inbox ("appears in their inbox when it is actually true," line 710) is bypassed
for the single most important signal. Drift detection that only notifies on
*growth* but not on *disappearance/format-break* has a hole exactly where this
project's history says the risk lives. A parse that yields zero or implausibly
few roots, or a checksum-unchanged-for-N-weeks staleness signal, is not among the
weekly triggers.

## Finding 5 (MEDIUM) — The Actions↔HF Jobs GPU embedding seam — the most expensive and most failure-prone step — has no described orchestration, auth, wait/async, or failure handling

Embedding is deliberately offloaded across a system boundary:

> "Embeddings run on **pay-per-use HF Jobs GPU** … **only for changed/new
> layers**." (lines 730–732)

The plan specifies the *economics* of this step in detail (2–7 GPU-hours,
low-tens of dollars, changed-rows-only — lines 183–184, 730–738) but not its
*orchestration*. The quarterly Actions workflow must: authenticate to HF Jobs
(a token in Actions secrets — `grep -i 'token|secret'` returns nothing),
launch the GPU job, then either block-and-wait for a 2–7 hour job (which burns
the launching Actions job's own 6 h budget — the very ceiling the plan sharded
the crawl to avoid) or fire-and-forget and re-collect results in a later
triggered job (an async handoff the plan does not design). Failure handling for
the GPU step — a crashed/OOM/timed-out HF Job, a partial embed, retry policy,
how a failed embed feeds the drift panel — is absent. The maintainer's own global
practice ("After launching an `hf jobs run`: record the job id … monitor it …
never end a session with a running job unmentioned") signals that this handoff is
non-trivial and failure-prone; the plan treats it as a black box. This is the
single most expensive automated step crossing an unspecified seam, which is where
unattended pipelines most often wedge.

## Finding 6 (MEDIUM) — Sharded-crawl partial failure and resume are not designed; an 8–9 h matrix crawl will routinely lose shards, and the combine job's behavior on a missing shard is unspecified

The crawl is matrix-sharded across parallel free runners (lines 716–728), with
each shard "writing its shard's raw output as a job artifact, followed by one
lightweight combine job" (lines 720–721). At the estimated ~8–9 h total across N
shards touching thousands of independent, flaky government hosts, individual
shard failure (a runner eviction, a pathological host, a transient infra blip) is
not an edge case — it is expected on most runs. The plan does not specify:
`fail-fast: false` behavior on the matrix; whether the combine job proceeds with a
missing shard (silently under-counting layers — the exact silent-attrition class
the `crawl_outcomes` ledger was built to eliminate) or aborts the whole snapshot;
or any shard-level resume/retry so a single failed leg re-runs without repeating
the 8–9 h whole. `grep -i 'resume|fail-fast'` finds "resume" only at line 858 in
an unrelated sense ("re-run, don't repair" for the *transform* rollback). The
`seed_count == Σ crawl_outcomes` reconciliation gate (line 372) is a real backstop
that would *catch* a dropped shard at publish time — but catching it means the
whole quarterly run fails and must be fully re-executed, which for an 8–9 h job is
an expensive, non-idempotent failure the automation design should have engineered
against, not merely detected.

## Finding 7 (MEDIUM) — Notification is GitHub-Issues-only with no positive heartbeat; a workflow that never runs, or dies before reaching the gate, produces no signal at all

The plan's notification model is entirely failure/Issue-driven and assumes the
workflow *runs and reaches the gate*:

> "A failed gate means the snapshot is **not published**, an Issue is opened with
> the failing gate…" (lines 766–767)

> "it converts 'is it time to refresh?' … into a thing that appears in their
> inbox…" (line 710)

Every notification in the design is emitted *by* the workflow *from inside* a run
that got far enough to evaluate a gate or a diff. There is no dead-man's-switch or
heartbeat that fires on the *absence* of a successful run — so the failure modes
that matter most for a neglect-safe pipeline (the cron auto-disabled per Finding
1; the runner OOM'd before the gate per Finding 2; the HF Job wedged per Finding
5; the whole workflow never triggered) are exactly the ones that emit *nothing*.
For a design whose explicit worst-case guarantee is "no new snapshot + one Issue"
(line 768), the missing half is: what tells the maintainer that "no new snapshot"
happened for a *bad* reason rather than a benign one? GitHub's default
scheduled-failure email goes only to the last person who touched the repo and
stops entirely once the cron is auto-disabled — so the fallback channel degrades
in lockstep with Finding 1. A positive-heartbeat / expected-run-did-not-happen
alert (an external uptime ping, a "last successful snapshot age" badge the
maintainer can eyeball) is the standard neglect-safe pattern and is absent.

---

## What the plan genuinely got right on this axis (credited, not refuted)

To keep the audit honest and default-refute rather than default-condemn:

- **The drift panel content is real and specific** (lines 739–764): count
  reconciliation, reachability delta, volume drift, schema conformance, null-rate
  bands, determinism/ID-overlap, referential integrity, geometry validity,
  license distribution, embedding integrity, and in-run Recall@k retrieval
  parity. This is a genuine every-refresh data-quality gate set, and recon
  confirms nothing like it exists in the ecosystem today.
- **The gate is proven to fail correctly, not just to run.** Stage 5's gate
  requires "a **deliberately-injected bad refresh is caught and quarantined** by
  the drift panel (the automation is proven to *fail correctly*, not just to
  run)" (lines 901–903). Testing the negative is the right instinct and is rare.
- **Fail-closed with publish/promote decoupled** (lines 664–667, 769–771): a bad
  snapshot cannot auto-reach production, and publish is impossible while
  `all_passed: false`. The safety posture is correct even where the mechanics
  underneath it (Findings 1–2, 5–7) are under-built.
- **The 6-hour job ceiling was correctly identified and addressed** via matrix
  sharding with a *measured* (not guessed) shard count (Stage-1 gate, lines
  828–839). The plan engaged one real free-CI constraint rigorously; the finding
  in this audit is that it engaged only that one.

The net: the axis is well-*conceived* (drift detection and quality gates on every
refresh genuinely exist and are fail-closed) but under-*engineered* as running,
unattended automation. The gates would catch a bad refresh; the machinery that is
supposed to invoke them quarterly, unattended, at 4× scale, on free
infrastructure, is where the refutations land.

---

## Verification

Independent adversarial re-derivation by a second verifier (not the auditor
above). Method: re-read `docs/rebuild-plan.md` in full (lines 1–1119) and the
cited recon, re-ran every grep the findings assert, and defaulted to *refute*
each finding unless the plan text and cited evidence personally confirmed it.
Greps run against the plan: `60 day|keepalive|heartbeat|workflow_dispatch|
cron.*disabl|schedul.*disabl` → **no matches**; `disk|RAM` (runner sizing) →
**no matches** (the five `memory` hits are all serving-envelope / index-memory /
sibling-load, none a CI-runner check); `token|secret|poll|wait|fail-fast|async`
→ **no matches**; `resume` → **no matches anywhere in the plan**. All line
citations spot-checked and accurate except one minor auditor slip noted below.

**Finding 1 (cron-auto-disable) — CONFIRMED, HIGH.** The platform behavior is
real and documented: in a public repo, `on: schedule` workflows are
auto-disabled after 60 days with no repository (commit/push) activity. The plan's
spine is "turns completely off … untouched for six months" (line 40) with the
weekly cron as "the entire monitoring system" (line 709); the weekly check only
*opens an Issue* on drift (lines 703–710) and never commits, so in a quiet window
(no drift, no commits) the repo has zero push activity and the cron is disabled at
~day 60 — one-third into the design's own neglect window. Grep confirms no
keepalive, re-arm, or off-GitHub scheduler anywhere. Even granting the ambiguity
of "repository activity," a genuinely-neglected repo with no drift produces no
activity of any kind, so the auto-disable fires regardless of interpretation. The
design goal and the chosen mechanism are mutually exclusive as written, with no
mitigation in the document. HIGH is correct — this categorically kills the
central premise, unlike the softer findings below.

**Finding 2 (free-runner resource limits) — ADJUSTED, MEDIUM** (down from HIGH).
The *omission* is real and confirmed: the plan engages only the 6h time ceiling
(lines 176–180, 716–728), never the free runner's ~14 GB SSD / ~16 GB RAM, while
asserting transform+gates+manifest CPU at "~$0 … GitHub Actions is free" (lines
975–978); grep for runner disk/RAM sizing returns nothing. That gap is genuine
and worth fixing. But the finding's HIGH rests on absolutist "cannot physically
execute" claims that are refutable: (a) the plan mandates a *quantized* ANN index
(lines 190–196), not the "flat 14 GB fp32 index" the evidence argues can't fit
16 GB — IVF/PQ trains on a sample and adds vectors in memmapped batches, so peak
build RAM is far below 14 GB; (b) the vectors are produced on the HF Jobs GPU
already (line 730), so index build naturally co-locates there — undesigned, but
the natural home exists within the plan's own offload pattern; (c) the "combine
job" can stream shard artifacts to the content-addressed archive rather than hold
15–30 GB on one disk; (d) transform output (~4–8 GB typed parquet) is
state-partitionable/streamable and plausibly fits 14 GB with the raw streamed from
the archive. So this is an *underspecified where-does-it-run + unchecked-$0* gap,
not a hard impossibility. Real, but MEDIUM.

**Finding 3 (volume-drift false-positive on 4× growth) — ADJUSTED, MEDIUM**
(down from HIGH). Confirmed that the volume-drift check is defined only by its
downside example ("a 40% collapse is a truncated crawl," line 748) "compared
against the prior published snapshot" (line 742), that the premise is ~4× growth
(line 171), and that Stage 6's first fresh crawl publishes that ~4× snapshot
against the 865k Nov-2023 anchor (lines 909–911) — the band's *upside* asymmetry
and any legit-growth-vs-runaway discriminator are genuinely unspecified. That is a
real correctness gap in an every-refresh gate. But severity is tempered because
the gate is fail-*closed* → fail-*safe*: a false-positive blocks *publish*, and
publish is human go/no-go anyway (lines 711–714, 877), with a documented
"quarantined with a recorded human override" path (line 884). The plan also
already handles the no-prior baseline case for the retrieval gate ("≥ documented
floor for the first snapshot," line 872), showing baseline-awareness. Worst real
outcome is "false alarm the in-loop human overrides," not silent block and not bad
data — MEDIUM, not HIGH.

**Finding 4 (weekly-cron upstream-death not wired to notification) — CONFIRMED,
MEDIUM.** Verified: the weekly check specifies only the growth happy path (Issue
on >10% new roots, lines 703–710); the "if the mirror 404s … the build fails a
gate loudly" promise (line 266) sits in Dimension 1 point 2 ("before crawling"),
which is the *quarterly crawl build*, not the weekly cron. The weekly check's
behavior on a 404 / move / truncation / format-break / checksum-unchanged
staleness — the exact silent-upstream-death that founded this project (recon
README line 49; plan line 248) — is unspecified, and drift triggers only fire on
*disappearance's opposite* (growth). Notification-channel hole exactly where the
project's history says the risk lives. MEDIUM is right: the quarterly build would
eventually catch a dead mirror and a red Actions run emits *some* default email,
so the signal is delayed/degraded rather than wholly absent.

**Finding 5 (Actions↔HF-Jobs GPU seam unorchestrated) — CONFIRMED, MEDIUM.**
Grep confirms zero `token|secret|poll|wait|async` anywhere: the plan gives the
embedding step's economics (2–7 GPU-h, lines 183–184, 730–738) but no auth, no
block-vs-async completion design, and no crashed/partial/OOM GPU-job failure
handling across the system boundary. A blocking wait on a full re-embed would
reconsume the 6h Actions ceiling the crawl was sharded to avoid; the async
alternative is undesigned. The maintainer's own global rule to record and monitor
`hf jobs run` ids flags this handoff as non-trivial. MEDIUM: real gap, but plainly
implementable (token in Actions secrets + poll, or a follow-up triggered
workflow), and the expensive full re-embed is rare (model-change/first-snapshot
only).

**Finding 6 (sharded-crawl partial failure / resume undesigned) — CONFIRMED,
MEDIUM.** Confirmed no `fail-fast` or `resume` design anywhere (grep: zero
matches for either; the auditor's "resume at line 858" is a minor mis-cite — 858
reads "re-run, don't repair" about *transform* rollback, and "resume" appears
nowhere, which strengthens the finding). At ~8–9h across N shards on flaky
government hosts, shard loss is expected, and GitHub's matrix default
`fail-fast: true` would cancel siblings on one failure — unaddressed. One
correction to the finding's framing: a lost shard is *not* silent — the
`seed_count == Σ crawl_outcomes` gate (line 372) catches it, since dead-shard
servers produce no outcome rows (seed_count > Σ). So the true cost is a caught-but-
expensive full non-idempotent re-run, not silent attrition. The finding itself
acknowledges the gate backstop; the real defect (no fail-fast:false, no
combine-with-missing-shard policy, no shard-level resume despite immutable
content-addressed raw that would make resume cheap) is confirmed. MEDIUM.

**Finding 7 (no positive heartbeat / dead-man's switch) — CONFIRMED, MEDIUM.**
Grep confirms no `heartbeat|dead-man|uptime` construct. All notification is
emitted from *inside* a run that reaches a gate (lines 710, 766–768); nothing
fires on the *absence* of a successful run. The failure modes that matter most for
a neglect-safe design — cron auto-disabled (F1), runner exhausted before the gate
(F2), HF Job wedged (F5), workflow never triggered — are precisely the ones that
emit nothing, and GitHub's default scheduled-failure email goes only to the last
committer and stops once the cron auto-disables, so the fallback degrades in
lockstep with F1. A "last-successful-snapshot age" signal or external uptime ping
is the standard neglect-safe pattern and is absent. MEDIUM: a real missing pattern,
though partly a corollary of F1 rather than fully independent.

**Net.** All seven findings are real defects, none refuted. Two auditor severities
overstated (F2, F3 both HIGH → MEDIUM: F2 leans on refutable "cannot execute"
absolutism when the plan mandates a batchable quantized index and already offloads
to GPU; F3 fails *safe* into a human-gated publish rather than into bad data). One
finding (F1) is a genuine HIGH — a documented platform behavior that categorically
defeats the plan's stated central premise with zero mitigation in the text. The
four MEDIUMs (F4–F7) stand as cited, with two small framing corrections logged
above (F6's "silent" attrition is actually gate-caught; F6's "resume at line 858"
is a mis-cite — "resume" appears nowhere). Confirmed-vs-adjusted tally: 5
CONFIRMED, 2 ADJUSTED, 0 REFUTED.
