# Audit of `docs/modernization-plan.md`

**Read-only.** Nothing in the repository outside this directory (`AGENTS.md` and `CLAUDE.md`
excepted — coordinator-authored deliverables, not audit output) was created, modified, or deleted
by the audit fan-out. Proof: `git status --porcelain` was empty immediately before the fan-out
launched; immediately after, it showed only this new `audit-recommendations/` directory plus the
two coordinator-authored doc files.

**Commit pin**: `9fda5a9` on branch `docs/modernization-plan` (production baseline `5b3caca`
unchanged). Dated 2026-07-20.

## Why this audit exists

`docs/modernization-plan.md` — the tracked planning authority for migrating this public Hugging
Face Space off a deprecated Streamlit/LangChain stack — carries its own closing section, "Review
mandate for Claude Code," asking for exactly this: reconcile every baseline claim against live
systems, identify assumptions with no executable validation gate, challenge whether the chosen
stack is still the smallest reliable choice, and record proposed changes as a diff with evidence.
This directory is that review. The proposed diff itself is applied directly to
`docs/modernization-plan.md` on this branch (not committed) — see that file's revision history /
the accompanying evidence log for exactly what changed and why.

## How to read this

- **[00-executive-summary.md](00-executive-summary.md)** — verdict, systemic themes, priorities.
- One file per axis (`claim.md`, `gates.md`, `tech-choice.md`, `trust-boundary.md`,
  `architecture.md`, `reliability-cicd.md`, `rollback-obs.md`, `self-ref-docs.md`) plus
  **[cross-cutting.md](cross-cutting.md)** for completeness-critic findings that didn't fit a
  single axis.
- **[findings.json](findings.json)** — machine-readable index of all 31 confirmed findings
  (`{id, axis, title, severity, effort, location, files}`).

Every finding cites `docs/modernization-plan.md` by **section heading**, not line number — line
numbers will shift as the document is revised in response to this audit; headings won't.

## Severity and effort legend

Findings target the *planning document*, not yet-unwritten application code — "effort" is almost
always **S** (a document revision), not implementation effort.

| Severity | Meaning here |
|---|---|
| Critical | Following the plan as written would produce a security incident, data loss, or a fundamentally wrong outcome |
| High | A material gap that would surface late (staging/production), an ungated risky action, or a claim that is actually false |
| Medium | A real gap worth closing before Stage 0 starts; not otherwise harmful |
| Low | Precision, citation, or evidence-quality issues |

| Effort | Meaning here |
|---|---|
| S | Edit the document (a paragraph, a gate bullet, a citation fix) |
| M | A multi-section revision or a small concrete pre-implementation action (e.g. one branch-protection config) |
| L | Not used in this audit — nothing found required a large rework |

A documented decision in the plan was **not** treated as a finding unless it conflicted with
verifiable reality, lacked any executable validation gate, or its stated "UNVERIFIED"/open-question
status had quietly become load-bearing elsewhere without being re-flagged.

## Methodology

Pipeline: coordinator inline live-reconciliation (git/GitHub API/Hugging Face Hub API/current
official docs, done **before** the fan-out — see `scratch/coordinator-verified-literals.md`) →
8 axis auditors (max 10 findings each, ranked by impact) → one adversarial verifier per raised
finding, on Opus, instructed to default to `isReal=false` when uncertain → a completeness critic
(also Opus) hunting for what an experienced reviewer would be embarrassed the audit missed →
adversarial verification of the critic's candidates → per-axis writers → this index, authored by
the coordinator from the verified digest (not by re-reading every file).

**56 agents total**: 8 auditors, 34 per-finding verifiers (axis stage), 1 completeness critic,
5 critic-candidate verifiers, 8 per-axis writers. 0 agent errors, 0 empty results.

**39 findings raised → 31 confirmed, 8 refuted (20.5% refutation rate), 0 unverified.**
Refutation rate by axis: CLAIM 0/4, GATES 0/6, TECH-CHOICE 1/3, TRUST-BOUNDARY 0/6,
ARCHITECTURE 2/3, RELIABILITY-CICD 0/5 (audit-stage), ROLLBACK-OBS 3/6, SELF-REF-DOCS 1/3
(audit-stage); completeness critic 1/5. The two ARCHITECTURE refutations and three ROLLBACK-OBS
refutations were the axes' auditors reaching for over-engineering/gap claims that didn't survive a
second read against the document's actual wording — a healthy result, not a weak one.

**Severity rollup of the 31 confirmed findings**: 0 critical, **1 high**, 11 medium, 19 low.

## Coordinator QA performed on this output

1. Every written axis file's `### <ID>. <Title>` headings were diffed against the verified digest's
   confirmed titles — all 29 axis-level findings (plus 2 cross-cutting) matched with no drops.
2. `findings.json` was generated from the same digest, then corrected: the workflow's merge script
   had a bug where the two cross-cutting findings kept their auditor-assigned severity (`medium`)
   instead of the verifier's `correctedSeverity` (`low`) — caught during coordinator QA and fixed in
   both `findings.json` and `cross-cutting.md` before publishing. (Self-caught, not user-caught —
   noted here per this session's own standing verification discipline.)
3. Cross-axis overlap was checked: CLAIM-01, RELIABILITY-CICD-01, SELF-REF-DOCS-01, and CROSS-02 all
   independently surfaced the same root cause (no branch protection / no required-status-check
   enforcement mechanism on `main`) from four different angles. This is flagged as the single
   highest-priority systemic theme in the executive summary rather than four separate fixes.
4. The read-only proof (git status before/after) was re-verified by the coordinator directly, not
   taken on the workflow's word.

## Not covered

- No axis independently re-derived the artifact byte-count arithmetic from raw bytes beyond
  spot-checks — it was computed once, precisely, by the coordinator inline before the fan-out
  (`scratch/coordinator-verified-literals.md`), and axes were told to trust and spot-check it rather
  than repeat the API calls per-axis.
- This is a document audit, not a code audit: `app.py` and the current test suite were read for
  grounding ("what does 'current state' actually mean") but were not audited as code in their own
  right — no findings here claim to be a security review of the *legacy* app, only of the *plan* to
  replace it.
- Live provider-model pricing/latency was not benchmarked (Stage 4 of the plan explicitly defers
  this to implementation time, and GATES-02 recommends gating it then).
