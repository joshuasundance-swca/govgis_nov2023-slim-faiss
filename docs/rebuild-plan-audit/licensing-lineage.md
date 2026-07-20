# Adversarial audit — LICENSING-LINEAGE axis

Auditor posture: default-refute. Assume the plan is wrong or incomplete on
Elfelt's list terms, the dead-CSV recovery path, and state/local GIS copyright
variability until real evidence shows otherwise. Sources checked in full:
`docs/rebuild-plan.md` (all 1,119 lines), `docs/ecosystem-recon/upstream-source-findings.md`
(esp. §1, §5), `docs/ecosystem-recon/github-findings.md` (restgdf_api / issue #74),
`docs/ecosystem-recon/huggingface-findings.md` (§2, §3 — current repo licenses).

## Verdict up front

This is, on balance, one of the **better-handled** axes in the plan. Three of the
recon's hardest licensing facts are handled concretely, not waved at:

1. **The dead CSV has a real, verified recovery path.** The plan does not assume
   the `.txt` mirror's format — it **fetched it live twice while writing**
   (Grounding Correction 1, lines 83-136) and *corrected* every draft's wrong
   assumption that the `.txt` is a columnar CSV. It is "Elfelt's human-readable
   report rendered as plain text — the text of the PDF, not a columnar CSV" (line
   84-85). The seed loader is redesigned as a section-and-entry parser (Dimension
   1, lines 253-263), the exact seed is committed each run with a SHA-256 and the
   last good seed is a fallback if the mirror 404s (lines 264-269). Issue #74's
   404 is cited accurately — confirmed against github-findings ("confirmed via
   live fetch (2026-07-20) that this exact CSV URL now returns HTTP 404", opened
   2025-12-15, 0 comments). This is concrete recovery, not hand-waving.

2. **The state/local copyright nuance is handled correctly — the plan does NOT
   treat government data as uniformly public domain.** Lines 291-295 cite the
   exact court split from upstream §5: "NY/SC permit state copyright assertions
   over GIS data; FL/CA reject them; many states have no clear guidance." A typed
   `license_status` enum (lines 305-315) makes `unverified_state_local` the
   legally-ambiguous **default** and gates it (plus `declared_restricted`) to
   **link/index-only** redistribution, with `federal_public_domain` (17 U.S.C.
   §105) and `declared_open` the only tiers redistributed in full. "The pipeline's
   default behavior is the legally-safe one" (line 314). This is exactly the
   posture upstream §5 demands ("it is not safe to blanket-assume all
   county/city/state ArcGIS layers are public domain") and it is operationalized,
   not merely noted.

3. **Elfelt's three terms are quoted verbatim and each gets an enforcement
   mechanism.** PDF-scraping prohibition → a hard no-PDF guard "enforced by a unit
   test" (line 263); commercial-use prohibition → a fail-closed
   `commercial_use_authorized` flag requiring a checked-in permission reference
   (lines 316-321); free-derivative permission → recorded acknowledgment + a free
   public dataset. The verbatim quotes (lines 287-290) match
   upstream-source-findings §1 exactly, typo ("Scrapping") included.

That said, default-refute turned up four real issues, one of them going to the
plan's **flagship** licensing construct. None is fatal; all are concrete.

---

## Finding 1 (HIGH) — The two-tier MIT/CC0 split may under-scope Elfelt's terms and open a downstream commercial-reuse hole

The plan's headline licensing innovation is a **two-tier** separation (Dimension
1, lines 296-302):

> "The license of *the server-URL list* (Elfelt-derived — inherits his terms …)
> is separated from the license of *our own crawled-metadata contribution*
> (MIT/CC0)."

The intent is good — the current repos' blanket MIT is genuinely wrong. But the
**carve** may itself be unsound, in the direction of *under*-honoring Elfelt.
Elfelt's terms (upstream §1, quoted verbatim in the plan at lines 287-290) are
written broadly around the phrase **"based on this list"**:

> "Commercial products **based on this list** are prohibited unless specific
> written permission is obtained …"
> "Permission is given for anyone to make a derivative work **based on this list**
> as long as the derivative work is available to everyone for free."

The entire crawled catalog is a "product/derivative **based on** this list" — its
selection and existence depend wholly on Elfelt's server roster (github-findings
confirms this list has been Josh's single source of truth "continuously since at
least 2020"). Taken at face value — and the plan explicitly commits to "treat
these terms as binding" (its own Dimension 1 posture, and upstream §5: "should
treat these terms as binding") — Elfelt's **no-commercial-without-permission** and
**free-availability** conditions attach to the *whole derivative*, not merely to
the embedded URL column the plan carves off.

Yet the plan stamps the "crawled-metadata contribution" **MIT/CC0** (line 300).
MIT and CC0 are grants of **unrestricted commercial use to any downstream party**.
So the plan's own label invites a third party to take the crawled-metadata tier
and use it commercially — which is precisely "a commercial product based on
[Elfelt's] list" that his terms prohibit without his written permission. The
`commercial_use_authorized` fail-closed flag (lines 316-321) guards only *the
maintainer's own* commercial use; it does nothing about the downstream commercial
reuse the MIT/CC0 stamp openly authorizes. The result is an **internal
contradiction**: the plan says Elfelt's terms are binding *and* publishes a
license label that grants rights those terms withhold.

Steelman (why this is HIGH but not CRITICAL): facts are not copyrightable
(*Feist*), a bare list of URLs may lack the originality for copyright, and the
crawled metadata is independently obtained from third-party government servers,
not copied from Elfelt — so as a matter of *copyright* the MIT/CC0 stamp on the
maintainer's own contribution is arguably defensible. But Elfelt's terms read as
**stated contractual/moral conditions on use of the list**, not a copyright claim,
and the plan chose to honor them as binding. Under that choice the broad "based on
this list" language reaches the whole product, and a permissive downstream grant
on any tier is in tension with it.

Concrete failure scenario: a commercial GIS vendor downloads the `govgis`
snapshot, reads the crawled-metadata tier's MIT/CC0 label, builds a paid product
on it, and is in breach of Elfelt's terms — with the maintainer having
affirmatively mislicensed the material. Fix: scope Elfelt's conditions
(no-commercial-without-permission + free-availability + share-alike-style "free to
everyone") to the **whole published derivative**, and dual-license the
maintainer's *own* schema/code/enrichment contribution separately and clearly
*subordinate* to Elfelt's conditions on the catalog as a whole — never as an
unconditional MIT/CC0 grant over the catalog content.

Evidence: `docs/rebuild-plan.md` lines 296-321; `upstream-source-findings.md` §1
lines 105-108, §5 lines 265-269.

---

## Finding 2 (MEDIUM) — Frozen old repos keep the blanket MIT license the plan itself calls wrong; only supersede banners are added, not a terms correction

The plan (Dimension 4, lines 586-592; Dimension 8, lines 937-941) freezes the two
2023 dataset repos ("frozen, never deleted"), adds "superseded by → `govgis`"
banners, and moves on. But both existing repos are stamped **`license: mit`**
(confirmed in huggingface-findings §2 line 94 and §3 line 150 — "public, license
MIT"), and the plan's own Dimension 1 explicitly indicts exactly this: "The
dataset does not stamp one blanket permissive license across both, **as the
current repos implicitly do**" (lines 301-302).

So the plan diagnoses the mislabel, fixes it going forward, and then **leaves the
mislabeled artifacts publicly live and downloadable** with nothing but a "see the
new one" banner. The MIT stamp on the frozen repos purports to grant downstream
users unrestricted commercial rights over Elfelt-derived, Elfelt-seeded content —
the same defect Finding 1 describes, except here it is a *known, shipped* state
the plan chooses not to remediate. A "superseded by" banner does not retract a
license grant; MIT is irrevocable as to copies already labeled.

This is MEDIUM not HIGH because the old data is stale (Nov 2023) and low-download
(63 and 36 downloads per huggingface-findings §1), and DOI/citation stability is a
real reason to keep the repos resolvable. But keeping them *resolvable* does not
require keeping the *wrong license label and card*. Fix: when freezing, correct
the old repos' cards/`license` metadata (or add a prominent terms NOTICE
reflecting Elfelt's conditions) rather than only cross-linking — the plan already
touches these cards to add the banner, so the incremental cost is near zero.

Evidence: `docs/rebuild-plan.md` lines 301-302, 586-592, 937-941;
`huggingface-findings.md` §2 line 94, §3 line 150, §1 lines 27-28.

---

## Finding 3 (MEDIUM) — The ingestion path parses content the plan itself calls "the text of the scraping-prohibited PDF," and designs weekly *automated* fetching as the default — ahead of the gate that must authorize it, with a soft "less objectionable" justification

The plan is admirably honest that the surviving `.txt` mirror is "the text of the
scraping-prohibited PDF" (line 141-143) and that "the permissibility of the
ingestion path is now a genuine open question, not an assumption" (line 143-144).
It correctly elevates "contact Elfelt" to a Stage-0 launch gate. Good. Three
residual problems keep this at MEDIUM rather than resolved:

1. **The "less objectionable" justification is an unsupported assertion.** Lines
   142-143: "parsing it is plausibly acceptable and is clearly less objectionable
   than the PDF." The `.txt` *is* the PDF's text content (the plan says so itself);
   Elfelt's prohibition targets "data from the PDF file," i.e. the content, and a
   different file extension serving the same content is not obviously outside it.
   The stronger, evidence-backed argument the plan *could* make — that Elfelt
   himself publishes the `.txt` as one of three parallel representations
   (PDF/CSV/`.txt`, upstream §1 lines 35-37) and the mappingsupport index says
   "Everyone is welcome to share this list" (upstream §1 line 110) — is available
   but not made. As written, "less objectionable" risks giving false comfort if
   the Stage-0 gate is ever skipped or delayed.

2. **The recon's own guidance leaned to manual, not automated, access.** Upstream
   §1 lines 116-119: the list "should probably **not be bulk-scraped
   programmatically** — it should instead be **manually downloaded** per the weekly
   CSV, or Elfelt should be contacted directly." The plan adopts one of the two
   sanctioned options (contact Elfelt — good), but then designs a **weekly
   automated GitHub Actions cron that fetches the `.txt`** as the standing default
   (Dimension 6, lines 703-711) plus the quarterly crawl seed-fetch. Repeated
   automated fetching is the "programmatic" access the recon cautioned against; the
   plan does not put the *automation cadence itself* (vs. one-time manual download)
   into the scope of what the Stage-0 gate must authorize — it frames the gate only
   as confirming that *parsing* the `.txt` is acceptable (line 271-278).

3. **Default-design-ahead-of-gate.** The whole Dimensions 1/2/6 machinery is built
   assuming automated `.txt` ingestion is fine, with the authorizing gate bolted on
   at Stage 0. That ordering is defensible (nothing publishes before Stage 4), but
   the plan should state that the *ingestion mechanism and cadence* — not just the
   licensing paperwork — are contingent on Elfelt's answer, and name the fallback
   if he declines automated access (e.g. a human downloads the weekly file and
   commits it; the committed-seed design at lines 264-269 already supports this,
   but the plan never connects that fallback to a "he said no to automation"
   branch).

Evidence: `docs/rebuild-plan.md` lines 138-149, 253-278, 703-711;
`upstream-source-findings.md` §1 lines 110, 116-119, 35-37.

---

## Finding 4 (LOW) — The "available to everyone for free" condition is not reconciled with the private raw archive

Elfelt's derivative-works permission is conditional: "as long as the derivative
work **is available to everyone for free**" (line 289; upstream §1 lines 107-108).
The plan proposes a **private** companion raw archive (Dimension 8, lines 940-941:
"Raw archival JSON lands in a **private** companion … the audit/replay floor") and
a private-raw standing repo (line 964-965). That archive is crawled from servers
selected entirely by Elfelt's list — arguably a "derivative work based on this
list" — and it is, by design, **not available to everyone for free**.

The plan records "the free-derivative acknowledgment" (line 278, 819) but never
reconciles the *private* archive against the *free-availability* condition it just
acknowledged. LOW because a defensible reading holds the raw crawl is independent
third-party government content (not Elfelt's list) and an internal audit artifact,
not the *published derivative* Elfelt's permission speaks to — but that reasoning
is exactly what the plan should state and does not. Fix: one sentence noting the
private archive is an internal replay/audit floor (not a "derivative work" being
distributed), and that the *published* derivative — the thing Elfelt's condition
governs — is always the free public `govgis` repo.

Evidence: `docs/rebuild-plan.md` lines 289, 940-941, 964-965;
`upstream-source-findings.md` §1 lines 107-108.

---

## What I checked and did NOT flag (credit where due)

- **Uniform-public-domain error: ABSENT.** The specific failure this audit was
  told to hunt for — "a plan that treats 'public government data' as uniformly
  public domain is WRONG" — is not present. The plan does the opposite (Finding
  verdict item 2 above; lines 291-315). This is the single most important thing to
  get right on this axis, and the plan gets it right.
- **Dead-CSV recovery: real, not waved at.** Verified working path (`.txt`
  mirror), live-fetched twice, format corrected, committed-seed fallback,
  no-PDF-URL unit-test guard. The plan out-grounds its own source drafts here.
- **Court-case citations: accurate.** NY(Suffolk)/SC copyright-permissive vs.
  FL(*Microdecisions v. Skinner*)/CA public-domain — matches upstream §5 lines
  296-301 exactly.
- **Verbatim terms: accurate.** Plan lines 287-290 == upstream §1 lines 105-108,
  character for character.
- **Issue #74 / 404 provenance: accurate.** Plan line 247-249 matches
  github-findings' live-fetch confirmation.

---

## Verification

Independent adversarial re-derivation of the four licensing-lineage findings.
Posture: default-refute; each finding confirmed only against the plan text and
the cited recon read personally, not the auditor's paraphrase. Sources re-read in
full for this pass: `docs/rebuild-plan.md` (all 1,119 lines, both pages),
`docs/ecosystem-recon/upstream-source-findings.md` (§1, §5),
`docs/ecosystem-recon/huggingface-findings.md` (§1–3),
`docs/ecosystem-recon/github-findings.md` (single-source-since-2020 claim).

**Bottom line: all four findings CONFIRMED as real. Three at the stated severity
(HIGH, MEDIUM, LOW); Finding 2's MEDIUM is retained after a genuine borderline
LOW/MEDIUM weighing, explained below. No finding refuted; no severity changed.**
Two paraphrase-level inaccuracies inside the findings' own prose are corrected
below (they do not change any verdict).

### Finding 1 — `two-tier-license-underscopes-elfelt` → CONFIRMED, HIGH

Re-derived, not deferred. The plan (lines 297–302) stamps the *crawled-metadata
contribution* **MIT/CC0** while the *server-URL list* tier inherits Elfelt's terms
via a NOTICE. Elfelt's verbatim terms (plan 287–290 == upstream §1 105–108) are
written around **"based on this list"**: "Commercial products based on this list
are prohibited…". The crawled catalog's selection and existence derive wholly from
Elfelt's roster — github-findings 215–216 independently confirms that list is
Josh's **"long-standing, single source of truth… continuously since at least
2020"** — so the whole product, metadata tier included, is plausibly a "product
based on this list." MIT and CC0 are unconditional grants of downstream commercial
use, so the metadata-tier label authorizes exactly what Elfelt's terms withhold.
The `commercial_use_authorized` flag (316–321) guards only the *maintainer's own*
commercial use ("the default posture is free/non-commercial," 320), not downstream
reuse — the two guards address different actors, as the finding states.

This is a genuine **internal contradiction**: the plan commits to treat Elfelt's
terms as binding (280, and upstream §5 267–269) and asserts "the pipeline's
default behavior is the legally-safe one" (314), yet its flagship two-tier
construct (named a grafted best-idea at line 58) publishes a downstream label that
is neither. I checked the strongest defenses and they do not dissolve it: (a)
*Feist* / facts-uncopyrightable makes the MIT stamp on independently-obtained
government facts defensible **as a copyright matter** — but the plan explicitly
chose to honor Elfelt's stated (non-copyright, contractual/moral) conditions as
binding, and that is the standard the contradiction is measured against; (b) the
surviving list-tier NOTICE means the *published-whole* dataset still surfaces the
commercial restriction, so a careful reader isn't wholly unwarned — but the
metadata tier's independent MIT/CC0 label still "invites" a downstream vendor to
lift that tier and commercialize it, which is the concrete failure. Severity HIGH
is correct, not hot: it is the flagship construct, the commercial-use exposure is
explicitly live (SWCA is a commercial consultancy, 320), and it undercuts the
plan's own "legally-safe default" claim. The *Feist* + surviving-NOTICE
mitigations make it a **firm HIGH, not a CRITICAL** — the auditor's own framing.

### Finding 2 — `frozen-old-repos-keep-wrong-mit-license` → CONFIRMED, MEDIUM

Confirmed against source. Plan 301–302 explicitly indicts the current repos'
blanket license ("does not stamp one blanket permissive license… as the current
repos implicitly do"); Dimension 4 (586–592) and Dimension 8 (937–941) freeze both
2023 repos with only "superseded by" banners and cross-links. huggingface-findings
§2 line 94 and §3 line 150 independently confirm both repos are **license MIT**.
So the plan diagnoses the mislabel, fixes it going forward, and leaves the
mislabeled artifacts publicly live with the wrong grant — a real
diagnosis-without-remedy inconsistency. A supersede banner does not retract an
irrevocable MIT grant; correcting the card/`license` metadata is near-zero cost
because the plan already edits those cards to add the banner.

Severity: I weighed LOW vs MEDIUM honestly rather than deferring. The LOW case is
real — the data is stale (Nov 2023), low-download (63 / 36 per hf-findings §1), the
repos are being explicitly superseded, and MIT-irrevocability caps the practical
effect of any fix. The MEDIUM case wins on three grounds: it is the *same
substantive defect as Finding 1 (HIGH)* applied to shipped, still-downloadable
artifacts; the plan **explicitly acknowledges the defect and explicitly touches
the very cards** yet declines the trivial correction; and "low download count" is
a weak mitigation for a licensing exposure whose failure mode is a *single*
commercial reuse — the posture's own caution against a mitigation that doesn't
address the exposure. Staleness/irrevocability hold it below HIGH; the
known-defect-left-unfixed-at-zero-cost quality holds it above LOW. **MEDIUM
retained** (borderline, but defensible on its own reasoning, one notch below
Finding 1 exactly as the same defect on lower-stakes artifacts should sit).

### Finding 3 — `txt-ingestion-is-pdf-text-and-automated-ahead-of-gate` → CONFIRMED, MEDIUM

Confirmed, with one correction to the finding's own prose. Verified:
- The `.txt` is the PDF's text: plan 84–85, 139–141, and 272 all call it "the text
  of the [scraping-prohibited] PDF." Elfelt's prohibition ("Scrapping data from
  the PDF file is prohibited") targets the content, so a same-content different
  extension is not obviously outside it — the "plausibly acceptable and clearly
  less objectionable than the PDF" (142–143) is indeed an **assertion**, softened
  only by "plausibly."
- Standing weekly automated fetch is designed as default: Dimension 6, 703–711 ("A
  light GitHub Actions cron fetches the `.txt` mirror… parses it, and diffs…").
  upstream §1 116–119 independently cautions the list "should probably **not be
  bulk-scraped programmatically**… manually downloaded… or Elfelt contacted."
- The Stage-0 gate (270–278) scopes the Elfelt contact to whether **parsing** is
  acceptable, CSV restoration, and the free-derivative acknowledgment — it does
  **not** scope whether *automated repeated fetching vs. one-time manual download*
  is acceptable. That gate-scope gap is the finding's sharpest and correct point.
- The committed-seed fallback (264–269) is framed for the "mirror 404s" case and
  is **never connected to an 'Elfelt declines automation' branch** — confirmed
  absent.

**Correction to the finding's prose (does not change the verdict):** the finding
says "The stronger available argument (Elfelt himself hosts the .txt…) is not
made." That is only half-right. Plan line 141 **does** make the "Elfelt himself
hosts" half ("a distinct, published representation Elfelt himself hosts"). What is
genuinely absent — I grepped the full plan — is the "three parallel
representations (PDF/CSV/.txt)" framing and the mappingsupport index's "**Everyone
is welcome to share this list**" (upstream §1 110–111). So the strongest form of
the argument is under-made, but not wholly unmade.

Severity MEDIUM: the design builds standing automated ingestion of PDF-equivalent
content and gates only the parsing question while the recon explicitly cautioned
against programmatic access — a real, multi-part scoping gap on a licensing axis.
Held at MEDIUM (not raised) because the plan is unusually honest that
permissibility is "a genuine open question, not an assumption" (143–144), makes
contacting Elfelt a hard Stage-0 launch gate, and nothing ingests operationally
before that gate — the missing pieces are (i) widening the gate's scope to include
automation cadence and (ii) drawing the explicit declines-automation fallback the
committed-seed machinery already supports. Refinements to an already-gated design,
not an ungated one → MEDIUM, not HIGH.

### Finding 4 — `private-raw-archive-vs-free-availability-condition` → CONFIRMED, LOW

Confirmed. Elfelt's condition "the derivative work is available to everyone for
free" is quoted at plan 289 (== upstream §1 107–108) and the acknowledgment is
recorded (278, 819). The private raw archive is proposed at 940–941 ("Raw archival
JSON lands in a **private companion**… the audit/replay floor"), reaffirmed as a
standing surface at 964–965, and its retention discussed at 987. That archive is
crawled from Elfelt-selected servers — arguably a "derivative work based on this
list" — and is by design not free-to-everyone. The plan distinguishes it
*functionally* (audit/replay floor vs. the "published product" public repo, 937)
but never states the **licensing reconciliation** against the free-availability
condition it just acknowledged. That gap is real and the fix is one sentence.

Severity LOW is correct: the reconciliation is readily available and strong (an
undistributed internal archive is not a "derivative work made available" at all,
so Elfelt's condition — which governs the terms under which you *publish* a
derivative — arguably never bites), and the *published* derivative is always the
free public `govgis` repo. A one-sentence clarity gap, not a design flaw. LOW.
