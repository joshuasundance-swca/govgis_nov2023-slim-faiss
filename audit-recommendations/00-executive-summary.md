> Audit of docs/modernization-plan.md — read-only — 9fda5a9 — dated 2026-07-20 — back to [README](README.md)

# Executive summary

## Verdict

**The plan is sound and ready to drive Stage 0, with revisions.** Zero critical findings. One high
finding, on secret-handling verification rather than a design flaw. Every Reconciled-baseline claim
and every artifact-inventory number was independently re-derived from live systems and checked out
exact — this is the strongest part of the plan, and it's the part a maintainer would least want to
re-verify by hand. What the audit actually found is a document that gets its *facts* right almost
everywhere and its *enforcement* incomplete in a few recurring, fixable ways: gates that reference
values or mechanisms that don't exist yet, and a merge/rollback story with no branch protection or
executable commands underneath it.

None of the 31 confirmed findings ask the plan to change direction. All of them ask it to close a
specific gap in wording, sequencing, or citation before Stage 0 starts.

## The one high finding

**TRUST-BOUNDARY-01**: the plan's BYOK policy ("keys must never enter logs, caches, telemetry,
URLs, persisted state, or exception messages") is stated correctly but has no verification
mechanism anywhere in the 8 stages — no log-scrubbing test, no exception-sanitization check. On a
public Space accepting user-supplied Anthropic/OpenAI keys, a policy with no gate is a policy that
can regress silently. Recommend adding an explicit test to Stage 4's Gate that asserts no
substring of a live-but-fake test key appears in captured logs/exceptions across a forced-failure
path.

## Systemic theme #1 (the biggest win): the merge/branch-protection gap, found four independent ways

Four different axes — **CLAIM-01**, **RELIABILITY-CICD-01**, **SELF-REF-DOCS-01**, and **CROSS-02**
— independently converged on the same root cause: `main` has **zero branch protection today**
(`gh api .../branches/main/protection` → 404 "Branch not protected", confirmed three separate
times across this audit). Stage 6's gate — "merge only after required checks pass" — has nothing to
attach to. **RELIABILITY-CICD-02** adds a sharper edge: `bumpver.yml` pushes directly to `main` on
manual dispatch, a path any future branch-protection scheme would need to explicitly account for
(exempt it, or route version bumps through the same PR gate).

This is one fix, cited from four places: add an explicit action (sequenced *after* Stage 1's CI gate
exists, since you can't require a check that doesn't exist yet) to configure branch protection with
the new CI tiers as required status checks, admin-enforcement on, and an explicit decision on
`bumpver.yml`'s direct-push path. Highest risk-reduction-per-effort item in this audit — S effort,
closes four findings at once.

## Systemic theme #2: measurement-driven decisions with no corresponding gate

**GATES-01** (Stage 2/7 retrieval-quality threshold is referenced by two gates but never defined —
still listed as an open question), **GATES-02** (Stage 4's provider/model default decision has zero
Gate bullet — the same failure class that let `claude-instant-v1`/`claude-2.1` sit hard-coded past
retirement in the current app), and **GATES-04** (Stage 0's metrics-capture gate doesn't require the
capture to actually succeed, despite the plan already knowing the obvious mechanism — public logs —
returns 401) are the same pattern three times: an Action calls for a measurement or a judgment call,
but the Gate doesn't require the measurement to have actually produced a usable, recorded value.
**ROLLBACK-OBS-03** is the same gap viewed from Stage 6's side (its regression gate has nothing to
compare against). Fix pattern: Stage 0 should be the single place that *records* thresholds/baseline
values as artifacts, and every downstream gate that depends on one should reference "the value
recorded in Stage 0" rather than restate a bare adjective ("agreed," "predeclared").

## Systemic theme #3: rollback and observable checks are described in prose, not captured as commands

**ROLLBACK-OBS-01** (no stage's Rollback subsection contains an executable command — the actual
mechanism, a direct git push to the HF Space remote, is never named), **ROLLBACK-OBS-02** ("verify
its SHA" is prose despite the exact working `curl`/`gh` commands already being used to reconcile
this very document), and **GATES-03** (Stage 7 is missing a Rollback subsection entirely; Stage 0
and Stage 3's are template-symmetric non-issues) point the same direction: capture the actual
commands used during this audit's reconciliation (they work; they're proven) as the plan's rollback
procedures, rather than re-describing them in prose at implementation time.

## Security-specific findings worth separating from the above

**TRUST-BOUNDARY-02** (the http/https link-scheme allowlist is never re-verified against
LLM-*generated* answer text, only against retrieved records), **TRUST-BOUNDARY-03** (timeout/retry/
concurrency mitigations have no numbers or gate), **TRUST-BOUNDARY-04** (rollback to legacy silently
reopens the FAISS-deserialization and XSS risks the rest of the plan closes — worth a one-line
rollback caveat), **TRUST-BOUNDARY-05** (no gate proves the FAISS conversion actually ran isolated/
no-network), and **TRUST-BOUNDARY-06** (the "never use gr.HTML for untrusted content" rule has a
one-time browser test but no durable regression gate against a future re-introduction) are all real,
all low-to-medium, and all cheap: each is a missing assertion in an already-planned test, not new
scope.

## Smaller, independent findings

- **TECH-CHOICE-01**: the uv/pyproject.toml decision never says how it reconciles with the fact that
  the actual Space build contract is driven by README YAML front matter (`sdk_version`,
  `python_version`), not a pyproject version pin — worth one clarifying sentence.
- **TECH-CHOICE-02** / **ARCHITECTURE-01**: two small module-ownership gaps in the "Target
  architecture" flow diagram vs. the "Recommended modules" list (a pipeline-orchestration step and
  the "safe presentation model" type both lack a clearly-owning module) — the second one matters
  more, since it's specifically the seam where the current XSS bug lives.
- **RELIABILITY-CICD-03/04/05**: Stage 1 should say "create," not "update," CI test tiers (none run
  today); `hf-space.yml`'s success-doesn't-mean-deployed defect needs a named owning fix; and
  `dependabot.yml` will need a `pip`→`uv` ecosystem update that nothing currently schedules.
- **CLAIM-02/03/04**: three small evidence-precision corrections (a stale API-shape description, a
  citation pointing at the wrong evidence page for an otherwise-correct claim, and two rollback SHAs
  presented as distinct when they're tree-identical) — none change a conclusion, all worth fixing so
  a future re-verification doesn't waste time chasing a phantom discrepancy.
- **CROSS-01**: preserve MIT license link, restgdf/dataset attribution, and drop stale "written by
  GPT-4"/Claude-Instant copy when the README is rewritten for the Gradio SDK.
- **SELF-REF-DOCS-02**: the 2026-07-20 baseline snapshot has no stated re-reconciliation step at
  Stage 0 kickoff, even though `main` isn't frozen in the meantime.

## Priorities, ranked by risk-reduction per effort

1. **Branch protection / required-checks** (closes CLAIM-01, RELIABILITY-CICD-01, SELF-REF-DOCS-01,
   CROSS-02 at once) — sequence after Stage 1's CI gate lands.
2. **BYOK secret-handling verification gate** (TRUST-BOUNDARY-01) — the one high finding.
3. **Stage 0 as the single source of recorded thresholds/baselines** (GATES-01, GATES-02, GATES-04,
   ROLLBACK-OBS-03) — one structural fix, four gates strengthened.
4. **Capture real rollback/verification commands** (ROLLBACK-OBS-01/02, GATES-03) — the commands
   already exist and were proven working during this very audit.
5. Everything else — independent, cheap (S effort), no sequencing dependency.

## Governance note

This is a single-maintainer, pre-implementation repo with no formal approval process today — the
plan's own "Review mandate for Claude Code" section *is* the closest thing to a governance hook that
exists, and this audit is that mandate being exercised. No finding here was treated as requiring
external sign-off; all are proposed directly as a diff to the plan, per its own instructions.

## Quick wins (S effort, no dependencies, do first)

CLAIM-02, CLAIM-03, CLAIM-04, GATES-03, GATES-05, GATES-06, RELIABILITY-CICD-03, RELIABILITY-CICD-05,
CROSS-01 — nine small, independent wording/citation/gate-bullet fixes with zero sequencing
constraints and zero risk of conflicting with any Confirmed decision.
