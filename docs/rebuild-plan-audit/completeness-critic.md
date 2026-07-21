# Completeness critic — gaps the seven audit axes missed

Grounding: read `docs/rebuild-plan.md` in full (1119 lines) and all five
`docs/ecosystem-recon/` files in full. Cross-checked every finding below against
the seven axis summaries (TECH-CHOICE, LICENSING-LINEAGE, ARCHITECTURE-COST,
AUTOMATION-RELIABILITY, SECURITY-TRUST, COMPLETENESS-STAGING, EVIDENCE-ACCURACY)
to confirm none already owns it. Absence-claims verified by grep.

---

## C1 [HIGH] — Dimension 1 required an alternative/complementary seed source; the plan evaluates none

The task brief's Dimension 1 explicitly asks how the server list is obtained
"given the dead CSV, an alternative/complementary source, **or both**." The plan
answers only "recover Elfelt's list" — the `.txt` mirror as primary, Elfelt's CSV
as an enrichment. It **never evaluates a single alternative or complementary
source**, despite:

- The entire plan being bottlenecked on **one non-responsive volunteer** (issue
  #74 unanswered since 2025-12-15) whose sole surviving feed is the text of a
  **scraping-prohibited PDF**, whose sanctioned machine-readable form is **dead**,
  and whose terms gate the whole downstream license story.
- The recon handing over concrete candidate sources the plan ignores:
  `upstream-source-findings.md` §2 flags **Shodan/Censys ArcGIS-REST
  fingerprinting** ("a plausible technique for a rebuild ... flagging as an open
  idea"), the **FGDC Clearinghouse Registry** ("worth a follow-up look if a
  rebuild wants an 'official' federal source to complement Elfelt's list"), and
  **data.gov/GeoPlatform.gov** as complementary dataset-level sources with ArcGIS
  distribution URLs.

Grep confirms zero occurrences of Shodan, Censys, Clearinghouse, GeoPlatform,
data.gov, "complementary," or "alternative source" anywhere in the plan. The
plan's own founding lesson is "an upstream feed can vanish silently" — yet its
resilience answer is only "commit the last seed," never "diversify the seed." A
complementary crawl-and-fingerprint source (even as a Stage-6 option) is exactly
the move that would de-risk the plan's single largest structural dependency, and
it is a task-mandated dimension given thin-to-absent treatment. No axis caught
this because each took the single-Elfelt-source framing as given.

---

## C2 [MEDIUM] — TIGER is an unversioned external data dependency the flagship feature depends on; absent from the manifest, cost, and determinism model

FIPS/county recovery — repeatedly called the "highest-leverage" change — is a
**Census TIGER county/place polygon spatial join** over ~3.5M layer-bbox
centroids (lines 128, 850, 1053, 1082). This silently introduces a **new external
geospatial data dependency** that the plan never treats as one:

- The manifest schema (lines 541–555) records `restgdf_version`,
  `embedding_model + revision`, `pipeline_git_sha` — but **no TIGER vintage**.
  TIGER is re-released annually and FIPS/place boundaries change; two snapshots
  built against different TIGER vintages can assign different `fips`/`county` to
  an unchanged layer, silently breaking the plan's own **cross-snapshot
  determinism / stable-ID-diff** premise (lines 468–476, 754–756) at the facet
  layer without any ID change to signal it.
- It is uncosted. ARCHITECTURE-COST flagged missing build-*labour* and index
  build-memory, but not the TIGER download, storage, and 3.5M point-in-polygon
  join compute — a real recurring step with a real memory/time footprint that
  the "~$0 orchestration / everything in free CI" claim silently absorbs.
- The determinism oracle (line 801) runs on "frozen fixtures" and never pins or
  versions the TIGER input, so a "build-twice-identical" pass does not actually
  cover the facet derivation that depends on it.

---

## C3 [MEDIUM] — `metadata_text`, the field that drives all semantic search, is carried but never defined or tested

The slim view carries `metadata_text` (line 445) and it is the text that gets
embedded and vector-searched. The plan **never specifies how it is constructed**
(which fields, concatenated how), never gives it a schema contract, and never
gives it a test oracle — despite:

- `huggingface-findings.md`:166 explicitly flagging that the *original*
  `metadata_text` generation logic **could not be found in any repo** ("that
  logic lives elsewhere ... not in the dataset repo"). This is precisely the kind
  of undocumented derived artifact a from-scratch rebuild must re-specify, not
  inherit by name.
- The plan's own grounding that **81% of `description` is empty** (line 449),
  which makes the composition of `metadata_text` (name + type + field names +
  parent-service context?) *the* determinant of retrieval quality — the retrieval
  oracle (lines 804–809) tests Recall@k but nothing pins the input text it ranks
  over, so a silent change to `metadata_text` construction moves every result
  with no gate to catch it.

The one field the entire search product stands on is left as an unspecified
inherited name. No axis caught this (COMPLETENESS-STAGING covered the license and
FIPS oracles, not this one).

---

## C4 [LOW–MEDIUM] — Per-snapshot DOI/citation mechanism is asserted, not designed (a task-mandated Dimension-4 item)

Dimension 4 of the brief explicitly requires "DOI/citation handling." The plan
handles *superseding* (banners, keeping old DOIs resolvable) well, but claims
"**each snapshot tag is independently citable via its manifest**" (line 591)
without a mechanism. HF DOIs are **repo-level, auto-assigned** (hf-findings §2/§3,
lines 96/152/333), not per-git-tag. The plan never states whether a per-snapshot
citation is a DOI-per-tag (does HF even mint those?), a single repo DOI plus a
manifest-SHA convention, or Zenodo/DataCite alongside. The successive-snapshot
citation story — genuinely novel in this ecosystem, and the reason the immutable
tags matter for research use — is the one part of the versioning design left as
an assertion.

---

## C5 [LOW] — No bus-factor / succession model for a durable, externally-cited public product whose crawl library is the same solo maintainer

The plan's entire simplicity thesis is "single spare-time maintainer," and it
adds three new standing surfaces (`govgis-pipeline`, `govgis`, private archive)
plus a sole-source **human relationship with Elfelt** — all concentrated on one
person, whose revealed capacity is the same 9–10-month PR backlog the plan uses
to disqualify the sibling. Two un-addressed concentration risks:

- **restgdf**, the load-bearing crawl dependency pinned as "the ecosystem's
  healthiest repo," is maintained by the **same single person** and itself
  carries 6 open dependency PRs (github-findings). The only fallback named is a
  "thin raw-httpx" shim (line 349) — no plan for restgdf breaking against a
  future ArcGIS API change inside a neglect window.
- The product is designed to be **cited, DOI'd, and depended on** by others
  (`Lexicom7/EO_Datasets` already cites the old one). A durable public data
  product with zero succession/handoff note is an ownership gap Dimension 8/9 does
  not close — the plan optimizes for "safe to neglect for six months" but not
  "safe if the one maintainer stops."

---

## C6 [LOW] — Recon's Major-TOM structural precedent was surfaced and never used

`huggingface-findings.md` §1 explicitly flags the **Major-TOM** org as "worth
studying as a pattern reference for how a mature geospatial-dataset org structures
multiple companion datasets + a viewer Space." The plan's Dimension 4/8 repo
topology reasons entirely from first principles and never engages this cited
in-ecosystem precedent for exactly the multi-companion-dataset-plus-viewer
topology it is designing. Minor (advisory recon lead dropped), noted for
completeness.

---

## Not-a-gap checks (deliberately cleared)

- Esri ArcGIS-API-for-Python re-evaluation (upstream §3's "worth re-examining"):
  **addressed**, briefly, at lines 345–347.
- The in-house deterministic-UUID Mongo scheme (hf-findings §6): **addressed** at
  line 458.
- The 560-row parity delta: COMPLETENESS-STAGING owns the causal-assumption nit;
  I add only that the gate's `±560` *tolerance band* (line 855) is wide enough to
  pass a pipeline that still silently drops those exact rows, so the parity gate
  does not actually enforce the losslessness requirement (lines 493–497) it sits
  beside — a wording/enforcement gap, not a hard contradiction.
