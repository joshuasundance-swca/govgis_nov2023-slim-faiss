# Adversarial audit — SECURITY-TRUST axis

Auditor posture: **default-refute.** Assume the plan is wrong or incomplete on
trust boundaries until the text proves otherwise. Target:
`docs/rebuild-plan.md` (synthesis draft, 2026-07-20). Grounding checked against
`docs/ecosystem-recon/upstream-source-findings.md`,
`docs/ecosystem-recon/README.md`, and the security sections of
`docs/modernization-plan.md` (lines 165-256, the three risks that plan had to
fix: pickle-like deserialization, raw-HTML XSS, secret leakage).

Method note: I grepped the full `rebuild-plan.md` for the security vocabulary a
plan on this axis would use. The words **`secret`, `token`, `credential`,
`HF_TOKEN`, `leak`, `SSRF`, `redirect`, `egress`, `169.254`, `localhost`,
`authentication`, `weights_only`, `allow_pickle`** appear **nowhere** in the
document. `untrusted` appears **exactly once** (line 657) and only in the
MCP-serving context. `robots`, `User-Agent`, and `rate-limit` appear only as
crawl *politeness*, never framed as security. That vocabulary gap is itself the
headline finding: the plan inherited the modernization plan's *consumer-side*
threat model competently, but the two trust boundaries that are **unique to
being a data pipeline** — the crawler as a client pulling untrusted content from
thousands of hosts, and the CI/publish secret surface — are essentially
unanalyzed.

---

## What the plan genuinely handles well (stated first, so the findings are read as gaps not as a hit-piece)

These are real, and I am not manufacturing findings against them:

- **Deserialization is explicitly avoided.** The plan replaces "deserialize a
  4.28 GB pickle-like blob and hope" (line 561) with content-addressed JSON,
  typed parquet, a checksummed manifest, native `index.faiss` "(…manifest, no
  pickle)" (line 671), and a consumer that "loads the manifest first, verifies
  checksums and counts before trusting a byte, and fails closed on mismatch"
  (lines 559-561). This directly retires the modernization plan's *critical*
  trust boundary rather than re-introducing it. Good.
- **The MCP/serving output contract mirrors the modernization threat model.**
  Lines 656-662: retrieved government metadata is "untrusted LLM input,"
  delimited and labeled "data, not instructions"; URLs pass an `http`/`https`
  allowlist "in records **and** in generated text"; HTML in descriptions is
  converted to plain text at serving; results are typed JSON "not
  markdown-flattened." Stage 5's gate (lines 894-903) tests prompt-injection
  records, filter-value hallucination, and malformed/oversized input, and
  requires injection fixtures to "fail closed." This is a faithful port of
  `modernization-plan.md` lines 200-233.
- **Crawl citizenship** (lines 328-334): honor `robots.txt`, a descriptive
  `User-Agent` with contact, per-host rate limits, a committed
  `excluded_servers.txt` denylist for takedown/opt-out. Good third-party-trust
  hygiene.
- **Licensing as an enforced trust boundary, not a note** (Dimension 1):
  two-tier license separation, `license_status` enum gating redistribution
  scope to link/index-only by default, `commercial_use_authorized` as a
  fail-closed precondition flag. This is the "re-publishing scraped third-party
  content" boundary handled *legally* with real teeth.

The findings below are the boundaries the plan did **not** carry the same rigor
to.

---

## Finding 1 (HIGH) — The crawler pulling from ~6,200–7,500 untrusted hosts is treated purely as a reliability problem, never as a security boundary (SSRF, redirect, response-bomb)

This is the single largest **new** trust boundary the rebuild introduces, and it
is the one the current Space never had — the Space only *consumes* pre-built
artifacts; the rebuild actively makes outbound requests to thousands of arbitrary
third-party servers. The recon underlines how untrusted these are: Elfelt's own
caveat is that "many of these server addresses were found with simple Google
searches" and the data "will be draft and/or temporary"
(`upstream-source-findings.md` lines 51-52). Any of those hosts can be
compromised, be a hijacked/expired domain, or be a deliberately planted entry.

Dimension 2 (lines 338-397) covers this boundary **entirely** as reliability and
politeness: bounded concurrency, per-host rate limits, retry-transient-only,
per-request timeout, overall wall-clock budget, "a hanging endpoint cannot stall
the run." Every one of those is a *liveness* control. **None** of the following
appears anywhere in the plan:

- **SSRF / redirect validation.** restgdf/httpx redirect behavior is never
  pinned or discussed. A seed (or hijacked) server returning `302 →
  http://169.254.169.254/latest/meta-data/…` or `→ http://10.x/…` or `→
  http://localhost:…`, if followed, turns the crawler into an SSRF pivot. The
  `http`/`https` scheme allowlist the plan *does* specify (line 659) is applied
  only at **serving** time and, critically, **a scheme allowlist does not stop
  SSRF** — `http://169.254.169.254/` is a perfectly valid `http` URL.
- **No egress/internal-IP restriction** on the crawler.
- **No response-size cap / decompression-bomb defense.** A malicious server can
  return a multi-GB or gzip-bombed body; the plan caps *time* but never *bytes*.
  (Line 894's "malformed/oversized" refers to MCP *tool input* fixtures, not
  crawl responses.)

The severity is amplified by the plan's own fallback runner choice. Lines 728
and 1089-1091 name a "self-hosted runner / HF Job / ephemeral VM" as the
documented fallback if the free GitHub Actions matrix can't shard the ~8–9h crawl
under the 6h ceiling. Running an **untrusted-content crawler on a self-hosted
runner with no SSRF/egress controls is the textbook cloud-metadata
credential-theft scenario** — and the plan green-lights that fallback with zero
security caveat. Even on GitHub-hosted runners, an SSRF/redirect pivot into the
runner's own internal services is not nothing.

**Failure scenario:** a single compromised or hijacked host in the 7,500-server
seed returns a redirect to the cloud metadata endpoint during the quarterly
crawl; the crawl is running on the documented self-hosted-runner fallback; restgdf
follows the redirect (default httpx follows redirects when enabled and the plan
never pins it off); IAM credentials for the runner are read and exfiltrated in the
crawled "response." Nothing in the plan's gates would detect this — the drift
panel checks counts and reachability, not egress.

---

## Finding 2 (HIGH) — Secrets do not exist in this document, yet the pipeline provably requires privileged ones; no scoping, no leak test, no crawl/publish isolation

The task names "secrets handling" as an axis focus and the modernization plan
made **secret-leak a Stage-4 gate** (`modernization-plan.md` lines 237-242:
"a test that forces a provider-call failure path … and asserts no substring of
the test key appears in captured logs, exceptions, or telemetry"). The rebuild
plan mentions secrets **zero times** — grep for `secret`, `token`, `credential`,
`HF_TOKEN`, `leak` returns nothing.

Yet the pipeline **provably** needs high-privilege secrets:

- A **Hugging Face write token** to publish snapshot tags, manifests, and Xet
  uploads to the public `govgis` dataset repo (Dimension 4, Stage 4).
- Write credentials to the **private raw archive** ("a private HF dataset repo
  or object store," line 941).
- HF Jobs GPU credentials for the embedding step (Dimension 6).
- GitHub Actions secret storage for all of the above, in a workflow that also
  runs the untrusted-content crawl.

The security-relevant design questions are all unaddressed:

- **Least privilege / secret isolation between crawl and publish.** The crawl
  jobs handle untrusted content from thousands of servers (Finding 1). The
  publish job holds a Hub *write* token. If those run in the same workflow /
  share a secret scope, a crawl-stage compromise (via a malicious response, a
  parsing exploit, or a poisoned transitive dependency of `restgdf[resilience]`
  / stamina / aiolimiter / DuckDB extensions) can read the write token and
  tamper with the *public* published dataset. The secure design — crawl jobs
  get **no** secrets, only the isolated combine/publish job holds the write
  token — is never stated.
- **No secret-leak gate.** The crawl logs URLs, errors, and `crawl_outcomes`;
  the manifest is public. Nothing asserts a token can't land in a log/artifact,
  which is exactly the regression the modernization plan built a gate to catch.
- **Token scope.** A repo-scoped fine-grained write token vs. an account-wide
  token is the difference between "a leak dents one dataset" and "a leak owns
  the account." Not discussed.

A public data product whose *write* path is automated in CI, with no secret
model at all, is a strictly larger secret-exposure surface than the Space's
session-scoped BYOK model the modernization plan already hardened — and it is
handled with strictly less rigor (none).

---

## Finding 3 (MEDIUM) — Raw HTML is redistributed in the published dataset; XSS sanitization is deferred to each consumer, re-creating the exact class the modernization plan had to fix

Recon confirms 3.4% (29,283) of `description` values carry raw HTML markup
(`modernization-plan.md` lines 566-568; echoed at rebuild-plan line 450, 659).
The modernization plan's hard-won lesson was that this is an XSS path that must
be neutralized **and CI-enforced durably** because "a later change could
silently reintroduce it with no regression signal"
(`modernization-plan.md` lines 217-223).

The rebuild plan converts HTML to plain text **only "at serving time"**
(line 659) — i.e. inside the optional MCP/DuckDB tier. The **published parquet
retains raw HTML** in `description` (and it flows into `metadata_text`, the slim
view's embedded/returned field, line 445). Consequences:

- Every downstream consumer — this Space, the sibling Postgres server, arbitrary
  `datasets.load_dataset` / `polars` / DuckDB users, and any future consumer —
  must **independently remember to sanitize**. That is precisely the "silently
  reintroduced with no regression signal" failure mode, now multiplied across an
  open, unbounded consumer set instead of one app.
- No `description_plaintext` (pre-sanitized) column is proposed alongside the raw
  field, which would let the data layer neutralize the payload once while
  preserving raw fidelity for auditing.
- No publish-stage gate asserts anything about the safety of redistributed text
  (the drift panel checks the *HTML rate* stays near 3.4% for parse-regression
  detection — line 752 — but that is a data-quality band, not a sanitization
  control; a stable 3.4% of *live* XSS payloads passes that gate cleanly).

Mitigating context (why MEDIUM not HIGH): the HF Dataset Viewer renders parquet
cell values as text, not HTML, so the Viewer preview itself is not an XSS sink;
and a data product arguably *should* preserve raw source fidelity. But the
security-correct move — sanitize-once at the data layer into a companion column
**and** keep the CI-enforced no-raw-HTML rule the modernization plan mandated —
is absent, and "each consumer sanitizes at serving time" is the assumption the
modernization plan explicitly rejected as unsafe.

---

## Finding 4 (MEDIUM) — Server-side live-fetch (`get_live_endpoint` / on-demand feature fetch) is SSRF-exposed; the only named validation is a scheme allowlist, which does not prevent it

The MCP tool `get_live_endpoint(id)` returns "the validated `https`-only live
ArcGIS URL" (lines 650-652) and the "Full" serving tier adds "on-demand live
feature fetch" (line 1019). These URLs originate from **untrusted crawled data**
(Finding 1's hosts). The plan's sole stated validation is the `http`/`https`
scheme allowlist (line 659).

A scheme allowlist is necessary but **not sufficient** against SSRF:
`https://169.254.169.254/…`, `https://[::1]/…`, and
`https://internal-host.corp/…` all pass an `https`-only check. Whether the
**server** performs the live fetch (SSRF against the serving host / its cloud
metadata) or the **agent** does (SSRF against the agent's environment), a
poisoned catalog URL — from a hijacked domain that was valid at crawl time, or a
maliciously planted seed entry — becomes a request the infrastructure makes to an
attacker-chosen internal target. No IP/host allowlisting, DNS-rebinding defense,
or metadata-endpoint block is specified.

Severity is MEDIUM because this lives in the optional/"Full" tier and is
partly agent-mediated, but the plan advertises `get_live_endpoint` as the payoff
of the whole "we index, we don't redistribute" posture (line 651) — it is
core to the design's value proposition, not a fringe feature, and it is the one
place the design deliberately hands a stored-untrusted-URL to a fetcher.

---

## Finding 5 (MEDIUM) — The public-facing serving tier (DuckDB search + MCP) carries none of the auth / rate-limit / concurrency / resource-exhaustion controls the modernization plan required

`modernization-plan.md` lines 235-256 require, for any public surface, "bounded
timeouts, retries only for transient failures, concurrency limits, and
user-visible rate/cost guidance," with a Stage-4 gate. The rebuild's serving
tier (Dimension 5, Stage 5) inherits the *output-safety* half of that threat
model (Finding-1-well-handled list) but **not the abuse/availability half**:

- The word `authentication` and any access-control concept appear nowhere for
  the DuckDB search API or the MCP endpoint. ("`auth…`" hits in the plan are all
  about commercial-use *authorization* under Elfelt's license, lines 287/316/550,
  not endpoint authentication.)
- **No rate limit / concurrency cap on the serving tier.** At ~3–4M layers an
  ANN + spatial + FTS hybrid query is compute-heavy; an unauthenticated,
  unthrottled endpoint is a straightforward DoS / cost-amplification target. The
  Stage-5 gate checks only that "latency [is] acceptable with the ANN index"
  (line 900) — single-query latency, not behavior under abusive concurrent load.
- "Sleep-when-idle" (lines 991-993) is a *cost* property, not an abuse control;
  it does not bound in-flight expensive queries and can itself be a cold-start
  amplification lever.

The plan's simplicity spine ("no always-on server") makes this *smaller* than a
standing Postgres deployment, which is fair — but "optional and sleep-when-idle"
is repeatedly used as if it discharged the abuse question, and it does not. If
the Standard/Full tiers run, this surface needs the same controls the
modernization plan made non-negotiable.

---

## Finding 6 (LOW) — Two residual deserialization/supply-chain trust gaps the "no pickle" headline does not cover

The plan rightly retires the LangChain-FAISS pickle path, but two adjacent
trust paths are unspecified:

- **Embeddings file format.** Dimension 4 stores embeddings "in their own file"
  (lines 530-534) but never names the format. If it is `.npy`/`.npz` loaded with
  `numpy.load(allow_pickle=True)`, that is a pickle-equivalent
  arbitrary-code-execution path re-introduced through the back door — the exact
  class the plan congratulates itself on removing. The safe choices (parquet, or
  `.npy` with `allow_pickle=False`, or safetensors) should be *named and gated*,
  not left implicit.
- **DuckDB community extensions.** The recommended read engine loads DuckDB's
  `vss` and `spatial` extensions (lines 621-629). Community/unsigned DuckDB
  extensions are a code-loading trust decision (`INSTALL`/`LOAD` pulls a binary);
  pinning versions/signatures and sourcing them deliberately is unaddressed.
  Combined with `restgdf[resilience]`'s transitive deps (stamina, aiolimiter)
  running in the crawl job that Finding 2 shows may share the publish token,
  supply-chain pinning deserves an explicit line and does not get one.

LOW because both are latent and easily closed once named — but they are exactly
the kind of "we removed the obvious deserialization risk and stopped looking"
gap the default-refute posture exists to catch.

---

## Summary

The rebuild plan is strong on the trust boundaries it *inherited* from the
modernization plan (deserialization avoidance, MCP output safety, licensing) and
weak-to-silent on the trust boundaries that are *new* because this is a crawling,
publishing data pipeline rather than a read-only Space:

1. the crawler as an SSRF/response-bomb-exposed client of thousands of untrusted
   hosts (HIGH),
2. a completely absent secret model for a CI-automated public *write* path
   (HIGH),
3. raw-HTML XSS deferred to every consumer instead of neutralized at the data
   layer (MEDIUM),
4. server-side live-fetch SSRF behind a scheme-only allowlist (MEDIUM),
5. an abuse/rate-limit/auth gap on the public serving tier (MEDIUM),
6. residual embeddings-format and extension supply-chain deserialization gaps
   (LOW).

None of these are fatal to the design, but items 1 and 2 are load-bearing
omissions for a plan that a maintainer would execute against as-is, and both sit
squarely inside dimensions the plan claims to have handled (crawl architecture;
ops/automation). They should be explicit, gated requirements before Stage 1
(crawl) and Stage 4/5 (publish/serve) ship.

---

## Verification

Verifier posture: **independent, default-refute.** I did not trust the auditor's
paraphrase. I re-read `docs/rebuild-plan.md` in full (both pages, lines 1–1119),
re-ran the vocabulary greps against the plan myself, re-read
`docs/ecosystem-recon/upstream-source-findings.md` §1, and re-read
`docs/modernization-plan.md` lines 195–256 (the three controls the auditor uses
as the contrast standard). Every absence claim and every cited line below was
personally confirmed against the source, not accepted on the auditor's word.

**Independent grep results (my own runs, not the auditor's):**
- `SSRF|redirect|egress|169\.254|localhost|decompression|response-size|byte cap` →
  **zero hits** in `rebuild-plan.md`. Confirms the SEC-1/SEC-4 core absence.
- `secret|token|credential|HF_TOKEN|leak|allow_pickle|weights_only|safetensors|\.npy|\.npz|authentication`
  → **zero hits.** The only `rate.?limit`/`concurrency` hits (lines 329, 345,
  352–354, 683, 831, 994, 1019) are all crawl politeness or the Postgres
  *escalation*, none a serving-tier abuse control. Confirms SEC-2/SEC-5/SEC-6.
- `untrusted` → **exactly one hit (line 657)**, MCP-serving context only. Confirms
  the SEC-1 framing that the crawl boundary is never called untrusted.
- `auth` → hits at 287/316/550 (commercial-use *authorization* under Elfelt's
  license), plus 651 ("authoritative"), 788 ("hand-authored"), 1089
  ("pre-authorize"). **No endpoint-authentication concept anywhere.** Confirms
  SEC-5.
- Elfelt "many of these server addresses were found with simple Google searches"
  → `upstream-source-findings.md` lines 50–52, verbatim. Confirms SEC-1's
  untrustedness premise (auditor cited 51–52; it spans 50–52 — immaterial).
- Modernization-plan contrast lines confirmed verbatim: secret-leak forced-failure
  gate (237–242), CI-enforced no-raw-HTML rule + "silently reintroduced with no
  regression signal" (200–223), mandatory bounded-timeout/concurrency/rate-cost
  abuse controls (249–256, anchored to *owner-funded provider keys* at 245–247).

### SEC-1 — crawler as an SSRF / redirect / response-bomb boundary — **CONFIRMED (HIGH)**

Confirmed on every element. Dimension 2 (lines 338–397) is entirely liveness:
concurrency, per-host rate limit, retry-transient-only, per-request timeout,
wall-clock budget. It caps *time*, never *bytes* — no response-size or
decompression-bomb defense exists. The plan never pins httpx/restgdf redirect
behavior off, and the scheme allowlist it *does* specify (line 659) is
serving-time only and — correctly stated — does not stop SSRF, since
`http://169.254.169.254/` is a valid `http` URL that passes a scheme check. The
self-hosted-runner / HF Job / ephemeral-VM fallback is documented (line 728) and
pre-authorizable (lines 1089–1091) with **zero security caveat**, which is the
textbook cloud-metadata credential-theft posture for an untrusted-content
crawler. This is genuinely the largest *new* trust boundary the rebuild
introduces — the current Space only consumes artifacts; the rebuild actively
fetches ~6,200–7,500 hosts the source itself calls Google-found and
draft/temporary. The drift panel checks counts/reachability, not egress, so
nothing in the plan's gates would detect an SSRF pivot. HIGH is the correct
severity: largest new boundary, entirely unanalyzed, and actively worsened by the
uncaveated self-hosted fallback. I attempted to refute by arguing "these are just
GIS servers" — refuted by the recon's own domain-hijack/draft caveat — and by
arguing ephemeral GitHub runners blunt it — true for the *default* but not the
*documented fallback*, and the response-bomb DoS applies to both. Stands at HIGH.

### SEC-2 — no secret model for a CI-automated public *write* path — **CONFIRMED (HIGH)**

Confirmed. The document mentions secrets zero times, yet the pipeline provably
requires an HF **write** token (you cannot publish snapshot tags / Xet uploads to
the public `govgis` repo without one — Dimension 4 / Stage 4), write credentials
to the private raw archive ("a private HF dataset repo or object store," line
941), and HF Jobs GPU credentials (Dimension 6), all orchestrated in CI. None of
the three security-design questions is addressed: (a) crawl/publish secret
isolation — the crawl job runs untrusted content and third-party deps
(`restgdf[resilience]`/stamina/aiolimiter, line 344; DuckDB extensions) and, if it
shares a secret scope with the publish job, a crawl-stage compromise reads the
write token and tampers with the **public** dataset that external work already
cites (DOIs; `Lexicom7/EO_Datasets`); (b) least-privilege token scoping; (c) a
secret-leak gate — despite the modernization plan making exactly that a Stage-4
gate (237–242). I tested the refutation "use `secrets.HF_TOKEN` is implied
boilerplate": rejected, because the load-bearing parts (isolation, scoping, leak
gate) are *design*, not defaults, and the plan explicitly ports the modernization
threat model's output-safety half while silently dropping its secret-leak half —
a regression against the plan's own stated standard. Impact is high: a leaked
account-wide write token is a supply-chain compromise of every downstream pin, and
the isolation gap chains directly off SEC-1's untrusted-content boundary. HIGH
stands.

### SEC-3 — raw HTML redistributed; sanitization deferred to every consumer — **CONFIRMED (MEDIUM)**

Confirmed as a real gap at the stated severity. Line 659 converts HTML to plain
text "at serving time" only (the optional MCP/DuckDB tier); the published parquet
retains raw HTML in `description`, which flows into `metadata_text` — the slim
view's embedded/returned field (line 445) that feeds embeddings and search
results. No `description_plaintext` companion column is proposed, and the drift
panel's line-752 check is a *rate* band (HTML near 3.4% for parse-regression), not
a sanitization control — a stable 3.4% of live payloads passes it cleanly. This
does push the neutralization burden onto an unbounded consumer set, which is the
"silently reintroduced with no regression signal" pattern the modernization plan
explicitly rejected (200–223). I pressed hard on downgrading to LOW: a data
product legitimately preserves raw source fidelity; parquet cells are not an HTML
sink (the HF Viewer renders them as text); and output sanitization is properly a
sink-side responsibility, so realization requires a *downstream* rendering bug.
But the finding does not overclaim — it explicitly rates MEDIUM-not-HIGH for the
Viewer-not-a-sink reason, proposes the correct fix (companion column + keep the
CI no-raw-HTML rule, not stripping raw), and the propagation into the embedded
`metadata_text` returned by search/MCP plus the dropped-control-vs-own-cited-
standard argument keep it above LOW. MEDIUM stands.

### SEC-4 — server-side/agent live-fetch SSRF behind a scheme-only allowlist — **CONFIRMED (MEDIUM)**

Confirmed. `get_live_endpoint(id)` returns the "validated `https`-only live
ArcGIS URL" (lines 650–652) and the Full tier adds "on-demand live feature fetch"
(line 1019); both operate on URLs stored from the untrusted crawl, and the sole
named validation is the line-659 scheme allowlist, which does not stop
`https://169.254.169.254/`, `https://[::1]/`, or `https://internal.corp/`. No
IP/host allowlist, DNS-rebinding, or metadata-endpoint block is specified. This
is a distinct boundary from SEC-1 (crawl-time outbound vs. serve-time outbound
from stored URLs), so it is not a double-count. MEDIUM is right: it lives in the
optional/Full tier and `get_live_endpoint` itself only *returns* the URL (agent-
side SSRF is partly the agent's boundary), but the Full tier's server-side
on-demand fetch is a direct SSRF against the serving host, and this is the
advertised payoff of the "index, don't redistribute" posture — the one place the
design deliberately hands a stored-untrusted URL to a fetcher. Stands at MEDIUM.

### SEC-5 — no auth / rate-limit / concurrency cap on the public serving tier — **ADJUSTED (HIGH→… ; corrected severity: LOW)**

The finding is **real, not refuted**: the DuckDB search + MCP serving tier has no
authentication, rate limit, or concurrency cap on compute-heavy ANN+spatial+FTS
queries over ~3–4M layers, and the Stage-5 gate checks only single-query latency
(line 900), not behavior under abusive concurrent load; "sleep-when-idle" (lines
991–993) is indeed a cost property leaned on as if it were an abuse control. All
of that is confirmed. **But the stated MEDIUM over-severities it**, and the
over-severity traces to importing the modernization plan's weighting without its
driver. That plan's abuse controls (249–256) were anchored to **owner-funded LLM
provider keys** (245–247) — real dollars per call. The rebuild's DuckDB search
path has **no per-query dollar cost** and **no LLM/paid-API in the query path**,
so the "cost-amplification" half of the finding is largely refuted; what remains
is pure availability risk. That residual risk lands at LOW on every axis: the
serving tier is **optional and off in the default (Minimal) posture** (the Hub
operates the only default surface); the data is **read-only** (no confidentiality
or integrity exposure); and the blast radius is a single **sleep-when-idle free
Space** whose worst case is degraded availability of a non-critical service that
**self-heals on idle**. Availability-only, optional/off-by-default, no billing
lever, no data at risk, self-healing = textbook LOW, not MEDIUM. It remains worth
one hardening line (a concurrency cap and a per-client rate limit if Standard/Full
is deployed), but it is not a MEDIUM. **Corrected severity: LOW.**

### SEC-6 — residual embeddings-format and DuckDB-extension supply-chain gaps — **CONFIRMED (LOW)**

Confirmed at LOW. The embeddings file format is genuinely never named (lines
530–534 say only "in their own file"), so *if* the naive `.npy/.npz` +
`allow_pickle=True` choice is made it re-introduces the exact ACE class the plan
headlines removing (lines 561, 671) — grep confirms neither `allow_pickle` nor
`weights_only` nor `safetensors` appears. The DuckDB `vss`/`spatial` extensions
(lines 621–629) are loaded via `INSTALL`/`LOAD`, which pulls a binary, with no
version/signature pinning discussed. Both are conditional and latent — numpy has
defaulted `allow_pickle=False` since 1.16.3, so the embeddings risk requires an
explicit unsafe choice — which is exactly why LOW (not higher) is correct, and
the finding frames them honestly as "latent and easily closed once named." A plan
that headlines "no pickle" should name and gate the serialization format and pin
its extensions; both are one-line closes. Stands at LOW.

### Verifier summary

Six findings checked independently against source. **Five CONFIRMED at the stated
severity** (SEC-1 HIGH, SEC-2 HIGH, SEC-3 MEDIUM, SEC-4 MEDIUM, SEC-6 LOW); **one
ADJUSTED** (SEC-5 MEDIUM→LOW — the finding is real but its severity imported the
modernization plan's owner-funded-provider-key cost weighting, which does not
transfer to a no-marginal-cost, read-only, optional, self-healing serving tier).
Zero findings refuted. The two HIGH items (crawler SSRF/response-bomb boundary;
absent CI secret model for a public write path) are the load-bearing ones and
both survive default-refute cleanly: they are genuine, unanalyzed trust boundaries
that are *new* to the pipeline (the consuming Space never had them) and that sit
inside dimensions the plan claims to have handled. My one correction reduces the
aggregate severity by one notch on one item; it does not rescue any finding from
refutation, because none needed rescuing.
