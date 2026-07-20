# Rebuild-plan audit — executive summary

Recorded 2026-07-20. This summarizes the adversarial audit of
[`docs/rebuild-plan.md`](../rebuild-plan.md) (the from-scratch rebuild of the
`govgis` **data pipeline**, distinct from this repo's separate Gradio-Space
modernization) and the revision applied to it. The audit ran seven independent
axes (auditor → verifier per finding) plus a completeness critic, with **refute
as the default verdict**; every surviving finding was applied as a diff to the
plan, not appended as a caveat. Per-axis evidence lives in the sibling files in
this directory; the refuted finding is recorded but not applied.

## The core recommendation (unchanged by the audit — the direction was sound)

The plan builds a **stateless, matrix-sharded batch pipeline** on free GitHub
Actions that: fetches Elfelt's surviving `.txt` seed mirror and commits it;
crawls the ~6,200–7,500 seed roots with a pinned `restgdf` v3; transforms to a
**typed core with deterministic, content-addressed BLAKE2b IDs**, honest lossless
bbox geometry, recovered jurisdiction/FIPS facets, and a per-record
`license_status`; embeds only changed rows on a pay-per-use **HF Jobs GPU** and
builds a **quantized FAISS ANN index** on that same step; and publishes
**immutable snapshot tags** to a single **Xet-backed** Hugging Face dataset repo,
each carrying a **checksummed manifest with a lineage chain** and each gated by a
**fail-closed data-quality drift panel** before publish. **No always-on server or
database sits in the critical path.** Serving is a set of optional,
consumer-specific views (the Hub itself as a bulk API; an embedded DuckDB
spatial/FTS/structured hybrid-search + typed MCP tool set; this repo's Gradio
Space as a thin index consumer), with the sibling Postgres+pgvector stack kept as
a documented, user-run escalation rather than operated as core.

**Rationale — the base lens is operational simplicity, with data-engineering
rigor as a non-negotiable overlay.** Every repo in this ecosystem has exactly one
committer, and the most sophisticated piece of infrastructure in it carries
nine-to-ten months of unmerged security PRs — the clearest available measurement
of what always-on infrastructure costs a spare-time maintainer. The design goal is
therefore a pipeline that produces a good dataset when told to and then **turns
completely off, its worst failure being *no new data*, never *broken data*.** The
audit did not overturn this direction; it hardened the mechanisms that make an
unattended pipeline actually safe to neglect.

## Findings by axis and disposition

The seven axes produced **38 findings: 30 confirmed, 7 adjusted, 1 refuted.** The
completeness critic added **6 material gaps** (1 high) beyond what the axes caught.
All 37 surviving axis findings and all 6 gaps were applied. **Zero critical; seven
high.**

| Axis | Confirmed | Adjusted | Refuted | Highs (applied) |
|---|---:|---:|---:|---:|
| Tech-choice | 4 | 2 | 0 | 1 |
| Licensing-lineage | 4 | 0 | 0 | 1 |
| Architecture-cost | 4 | 1 | 1 | 1 |
| Automation-reliability | 5 | 2 | 0 | 1 |
| Security-trust | 5 | 1 | 0 | 2 |
| Completeness-staging | 6 | 1 | 0 | 0 |
| Evidence-accuracy | 2 | 0 | 0 | 0 |
| **Axis totals** | **30** | **7** | **1** | **6** |
| Completeness critic (separate) | 6 gaps | — | — | 1 |

The **seven high-severity items** applied: the serving-engine internal
contradiction (tech-choice); the flagship license stamp granting downstream
commercial use over Elfelt-derived content (licensing); the omitted multi-month
upfront build labour on a solo project (architecture-cost); GitHub auto-disabling
the scheduled cron after 60 days (automation); the unguarded crawl SSRF/egress
trust boundary and the entirely-absent CI secrets model (security, two);
and the un-evaluated alternative/complementary seed source (completeness critic).

**The one refuted finding**, correctly not applied: the claim that the
shard→combine artifact hand-off exceeds a 500 MB free-Actions storage quota. Those
figures are the private-repo allowance; **public-repo Actions artifacts are free
and uncounted**, so the plan's "$0 orchestration" is correct as written. The
header now records this so the refutation is visible.

## The single most load-bearing change

**The serving-engine reconciliation: the quantized ANN index is now owned by
FAISS, and DuckDB is scoped to spatial/full-text/structured filtering — one index,
not two, and per-surface.** This was the audit's highest-value catch because it
corrected a genuine **internal contradiction in the flagship serving decision**,
surfaced independently by three axes (tech-choice, architecture-cost,
completeness-staging): the plan's own feasibility analysis declared a
quantized ANN index (IVF/PQ/SQ8) *mandatory* to fit the free `cpu-basic` tier at
3–4M vectors, while recommending **DuckDB VSS** as the read engine — the one engine
that builds only a full-precision float32 HNSW, cannot quantize, and would
reproduce the exact ~14 GB footprint the plan says will not fit. Left unfixed, the
serving tier's headline "fits free tier" claim was unsupported and two independent
ANN structures were being stood up over one embedding set.

The fix cascades through Grounding Correction 2, Dimension 5, both relevant
Conflicts, and the staging plan, and it pulled in adjacent corrections: the ANN
requirement is now **scoped per surface** (mandatory for the vector-first Space;
unnecessary for the hybrid tier that only re-ranks small filtered candidate sets),
the embedding **dimension is elevated to a first-class Stage-3 choice** (a 384-dim
model fits `cpu-basic` with no quantization at all), **index build-time memory** is
sized and homed on the GPU step, and a **projected-scale probe** now exercises the
index at 3–4M before the final stage rather than validating only on the 4×-smaller
865k parity corpus.

The most consequential *reliability* correction, close behind, is the **60-day
scheduled-workflow auto-disable**: the plan's entire "safe to neglect for six
months" premise rested on a weekly cron that GitHub would silently disable
one-third of the way into that window. The revision adds a heartbeat-commit
keepalive and an external dead-man's-switch, so the monitoring system can no longer
die without reporting its own death.
