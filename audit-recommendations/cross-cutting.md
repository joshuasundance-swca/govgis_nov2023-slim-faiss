> Audit of docs/modernization-plan.md — read-only — 9fda5a9 (branch docs/modernization-plan; production baseline 5b3caca unchanged) — dated 2026-07-20 — back to [README](README.md)

# Cross-cutting: findings raised by the completeness critic, outside any single axis

## Assessment

The plan is thorough on security, staging, and rollback mechanics; the two gaps below are the kind
an experienced reviewer catches by asking "what would embarrass this audit if it shipped anyway" —
provenance/licensing hygiene, and who actually has the authority to say a stage passed. Both were
downgraded from the critic's initial "medium" to "low" on verification: neither threatens the MIT
license itself (the `LICENSE` file isn't touched) nor introduces a new risk beyond one already
covered by CLAIM-01/RELIABILITY-CICD-01/SELF-REF-DOCS-01 (the branch-protection gap).

## Findings at a glance

| # | Finding | Severity | Effort |
|---|---|---|---|
| CROSS-01 | Target README/UI attribution and MIT-license preservation are never addressed | Low | S |
| CROSS-02 | Plan names no review/approval authority for stage transitions or merges | Low | S |

### CROSS-01. Target README/UI attribution and MIT-license preservation are never addressed; a rewrite risks silently dropping required notices

**Severity**: Low · **Effort**: S · **Location**: docs/modernization-plan.md — absent throughout (grep for `licen|elfelt|restgdf|attribut|acknowledg` returns nothing on-topic); README.md acknowledgments (line 66) + license link (line 62); LICENSE file

**Evidence**: The current README credits "Joseph Elfelt and the creators of the restgdf library" and links `[MIT License](LICENSE.md)`; the LICENSE file carries "Copyright (c) 2023 Joshua Sundance Bailey". The Space card (README YAML front matter) must be rewritten for the Gradio SDK migration, yet the plan never says to preserve the MIT notice, the restgdf/Elfelt acknowledgment, or the dataset credit. Separately, the current README's `LICENSE.md` link target doesn't exist (the file is `LICENSE`) — a pre-existing broken link the rewrite is a natural moment to fix.

**Why it matters**: MIT compliance itself rests on the untouched `LICENSE` file, which the migration doesn't remove — so this is documentation hygiene and provenance preservation, not a legal-compliance emergency. But a from-scratch README/UI "About" surface that silently drops the restgdf/Elfelt/dataset credit is a real provenance regression, and the broken license link would persist through a rewrite that touches everything else.

**Recommendation**: Add a one-line decision/gate: the target Gradio README and UI "About" surface must preserve the MIT license link (fixing `LICENSE.md` → `LICENSE`), keep `license: mit` in the rewritten front matter, retain the Joseph Elfelt / restgdf acknowledgment and `govgis_nov2023` dataset attribution, and drop the stale "written by GPT-4" / "Claude-Instant / Claude-2.1" copy.

### CROSS-02. Plan names no review/approval authority for stage transitions or merges

**Severity**: Low · **Effort**: S · **Location**: docs/modernization-plan.md — staged section (lines 265–429), esp. Stage 6 "merge only after required checks pass" (line 396); Stage 5→6 transition

**Evidence**: Grep for `approv|reviewer|maintainer|sign-off` finds only "pre-approved" staging-Space wording — no human or role owns any stage-gate sign-off anywhere in the document. The repo is effectively single-maintainer with zero branch protection today (`branches/main/protection` → 404, independently confirmed three times across this audit — see CLAIM-01 / RELIABILITY-CICD-01 / SELF-REF-DOCS-01).

**Why it matters**: For a genuinely single-maintainer repo, "review authority" is largely ceremony — approval is inherently self-approval, and the plan already documents rollback at every stage, bounding blast radius. The concrete, non-ceremonial substance is the same missing enforcement mechanism already flagged elsewhere in this audit: without required status checks + admin enforcement on `main`, "gate passed" is a self-assertion with nothing checking it.

**Recommendation**: Add a short "Review, enforcement, and sign-off" note (place it in Stage 1, which already touches GitHub Actions/CI, with a back-reference from Stage 6): (a) enable branch protection on `main` with the CI gate tiers as REQUIRED status checks, admin-enforcement on; (b) record each stage's own Gate bullets as the sign-off checklist, making "approved" a checkable record instead of an assertion. **Note**: this substantively overlaps CLAIM-01 / RELIABILITY-CICD-01 / SELF-REF-DOCS-01 — when revising the plan, fix the branch-protection/required-checks gap once and cross-reference it from all four locations rather than four separate edits.

## Minor notes (not adversarially verified)

None — both critic candidates that survived verification are captured above; other critic candidates were refuted (see the workflow's per-agent verify results, not reproduced here).
