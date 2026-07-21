# Upstream Source & Ecosystem Findings: `govgis_nov2023` Provenance

Research date: 2026-07-20. Scope: trace the full provenance of the ~1,684-server list behind
`govgis_nov2023`, survey comparable tooling, and survey adjacent/authoritative government
catalogs, to inform a future rebuild of this data pipeline. Method: WebSearch + WebFetch only
(no code changes, no repo files touched other than this one). Every claim below is sourced; where
I could not independently verify something with a primary-source quote, I say so explicitly rather
than presenting it as fact.

**Important caveat on sourcing mechanics:** several of the quotes below come back from the
WebFetch tool as an AI-summarized paraphrase of the fetched page rather than a raw HTML dump (the
tool runs a small model over the page content). Where I have a direct quoted string in the tool
output, I present it in quotes and attribute it to the source URL. Where the tool only gave me a
paraphrase (no quotable string), I say so. I did not fabricate any quotation.

---

## 1. Joseph Elfelt's curated ArcGIS server list — almost certainly the root source

**Identity and background.** Joseph Elfelt is a (now retired) software developer based near
Redmond, Washington, who runs the site **mappingsupport.com**. Per his GISsurfer "About" page:
"Joseph is a retired software developer and real estate professional who now focuses on volunteer
map-related projects" (paraphrased by WebFetch from
https://mappingsupport.com/p2/gissurfer-about-contact.html — the page footer text was quoted
verbatim as "© Copyright by Joseph Elfelt. All Rights Reserved." and "*None of my projects pester
you with ads, track you or share/sell your location data.*"). He also runs GISsurfer (a Leaflet-based
web map for surfing government GIS data), GeoJPG, FindMeSAR, and wildland-fire mapping projects.
LinkedIn: https://www.linkedin.com/in/joseph-elfelt-a7096b4a/ ("Founder - MappingSupport.com,
PropertyLineMaps.com, FindMeSAR.com and GeoJPG.com"). Mastodon:
https://m.ai6yr.org/@mappingsupport.

**The list itself.** Elfelt maintains a weekly-refreshed catalog titled *"Federal, State, County
and City GIS Servers"*, published as a PDF (and a companion CSV) at:

- PDF: https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.pdf
- CSV: https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.csv
- Plain text mirror: https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.txt

Directly fetching the `.txt` mirror (https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.txt)
returned this verbatim text:

- Current size: **"7,500+ ArcGIS server addresses for the USA"**
- Last updated (as of my fetch): **"June 18, 2026"** — i.e., this list is still actively
  maintained today, about a month before this research (2026-07-20).
- Update cadence: **"An updated list is usually posted each Wednesday"**
- Validation method: **"Once a week my code automatically tests each link in this report. Each
  link is tested at least 3 times spread over a several hours."**
- Scope marker: every entry "include[s] 'rest/services' somewhere in the address" — i.e. this is
  specifically a list of ArcGIS REST **service root** endpoints, not a metadata/dataset catalog.
- Explicit caveat about provenance/quality: **"many of these server addresses were found with
  simple Google searches"**, and the GIS data on these servers "will be draft and/or temporary"
  with risk of "layer renumbering and inconsistent metadata."

**Growth over time (multiple sources, each independently fetched):**
- 2020 (spatialreserves.wordpress.com, fetched): "over 2,200 addresses"
- Jan 2022 (same source, updated): "around 3,500"
- Dec 2021 (GitHub issue, see below): "3,000+"
- Jul 2025 (GeoHipster interview): "5,000+"
- Jun 2026 (my direct fetch): "7,500+"

This means the **~1,684 servers behind `govgis_nov2023`** (dated Nov 2023) represented only a
fraction of even the *then-current* list, and the list has roughly **quadrupled to quintupled**
in size since the `nov2023` snapshot was taken — a strong, directly-evidenced argument for a
rebuild pulling from the current list rather than reusing the stale one.

**Methodology, verbatim from GeoHipster interview** (https://www.geohipster.com/2025/07/29/i-curate-a-list-of-5000-arcgis-server-addresses-at-all-levels-of-usa-government/,
fetched — quotes below are as returned by WebFetch, presented as direct quotations from the
article):
- "Most of the work in maintaining the list is done by code that I wrote which scans each address
  once a week. That code is run three times with a few hours between each run."
- "[the list] has been built bit by bit over the last 7 years" (as of the Jul 2025 article, i.e.
  dating the project's origin to roughly 2018).
- Time investment: "Approximately one hour weekly."
- For non-ArcGIS/global WMS servers, Elfelt explicitly points elsewhere and disclaims ownership:
  "That incredible work is someone else's project, not mine" — referring to
  https://www.geoseer.net/.

**Community history/usage:**
- Esri Community thread, Dec 2021 (I confirmed this thread's URL and opening-post content via
  WebFetch of https://community.esri.com/thread/229339-curated-list-of-1400-usa-based-government-arcgis-servers,
  title: "Curated list of 1,400+ USA-based government ArcGIS servers"), plus later Esri Community
  threads tracking the list's growth ("Curated list of ~3,500 government ArcGIS server
  ad[dresses]" — https://community.esri.com/t5/developers-questions/curated-list-of-3-500-government-arcgis-server/td-p/1136110).
  One Esri Community blog post (https://community.esri.com/t5/arcgis-hub-blog/update-on-the-curated-list-of-arcgis-servers-now-2/ba-p/883178)
  returned HTTP 403 to WebFetch and could not be read.
- GitHub issue, Dec 19 2021, opened by user "Jelfff" (near-certainly Elfelt himself) on the
  **openaddresses/openaddresses** repo (https://github.com/openaddresses/openaddresses/issues/5978),
  introducing the list ("3,000+ government ArcGIS server addresses") and noting many servers
  contain address/parcel data, inviting the OpenAddresses community to use/extend it. I could not
  retrieve the comment thread (WebFetch only surfaced the opening post), so I cannot report
  whether/how OpenAddresses acted on this — **unverified**.
- Also referenced/mirrored elsewhere: a Gist by jalbertbowden
  (https://gist.github.com/jalbertbowden/e47f75677ce71053634ceef5ecb318d6) and a GISCafe article
  (https://giscafe.com/nbc/articles/1/1625226/List-600+-ArcGIS-servers.-Federal-State-Region-County-City),
  neither of which I fetched in full — noted only as evidence the list circulates in the wider GIS
  community.

**Licensing / terms of use — important, and directly relevant to a rebuild.** Directly quoted
from the `.txt` mirror of the list itself
(https://mappingsupport.com/p/surf_gis/list-federal-state-county-city-GIS-servers.txt):

> "Scrapping data from the PDF file is prohibited."
>
> "Commercial products based on this list are prohibited unless specific written permission is
> obtained from Joseph Elfelt authorizing that commercial use."
>
> "Permission is given for anyone to make a derivative work based on this list as long as the
> derivative work is available to everyone for free."

The mappingsupport.com index page (https://mappingsupport.com/index.html, fetched) separately
states: **"Everyone is welcome to share this list."** Note the apparent tension: sharing/derivative
works are explicitly welcomed *if free*, but "scraping" the PDF and any commercial use are
explicitly prohibited without written permission. **For a modernization effort that plans to
re-crawl/re-publish this data (as the current `govgis_nov2023-slim-faiss` Space already does),
this means: (a) the existing free/non-commercial republication is very likely consistent with
Elfelt's stated terms, but (b) explicit written permission from Elfelt should be sought/confirmed
before any commercial use, and (c) the *list itself* (the PDF/CSV of server addresses) should
probably not be bulk-scraped programmatically — it should instead be manually downloaded per the
weekly CSV, or Elfelt should be contacted directly about API-style access.** I found no evidence
of an API or bulk-download authorization beyond the stated CSV. GISsurfer itself has a similar
restriction noted in search results: "for non-commercial use only, except for news media" (from a
WebSearch synthesis, not a page I directly fetched — flagging as **lower-confidence, not directly
quoted from primary source**).

---

## 2. Alternative/complementary catalogs of government GIS servers — current state of the art

I found **no other actively-maintained, comparably-comprehensive, root-server-level catalog** of
US government ArcGIS REST endpoints. What exists instead are catalogs at a different level of
granularity — individual datasets/layers, not server roots:

- **Data.gov** (catalog.data.gov) — federal open-data catalog. Many individual dataset entries
  link to ArcGIS REST/MapServer/FeatureServer endpoints as "distribution" URLs, but data.gov
  indexes *datasets*, not a flat list of *server addresses*. This distinction was captured in a
  WebSearch synthesis, not a directly quoted primary source: "Data.gov and GeoPlatform are two
  websites that can also help you find federal GIS servers, though they index individual layers
  rather than the complete server directory" — **treat this framing as a reasonable inference I
  could not independently confirm with a direct quote**, but it's consistent with how data.gov's
  catalog model works generally.
- **GeoPlatform.gov** (https://www.geoplatform.gov/about, fetched) — described on its own About
  page as focused on "Making Federal GeoData Findable, Accessible, Interoperable, and Reusable."
  It is the FGDC's official access point to **National Geospatial Data Assets (NGDA)**. My direct
  fetch could not confirm or deny whether it exposes bulk API/download access to a flat list of
  ArcGIS REST service roots (the fetched excerpt didn't contain that level of technical detail) —
  **unverified, would need a follow-up technical fetch of GeoPlatform's actual API docs**
  (e.g., data.geoplatform.gov) to confirm.
- **USGS National Map** (https://www.usgs.gov/faqs/where-can-i-find-a-list-urls-national-map-services)
  — a curated but narrow list of USGS's *own* ArcGIS REST endpoints (e.g.
  `https://basemap.nationalmap.gov/arcgis/rest/services`,
  `https://services.nationalmap.gov/arcgis/rest/services`,
  `https://carto.nationalmap.gov/arcgis/rest/services/govunits/MapServer`). This is useful as one
  *entry* in a broader server list (and is presumably already one of the ~1,684 in
  `govgis_nov2023`), but it is USGS-specific, not a cross-government catalog.
- **Federal Geographic Data Committee (FGDC)** (https://www.fgdc.gov/ngda, fetched directly) —
  governs the NGDA *theme* structure mandated by the Geospatial Data Act of 2018 ("the FGDC to
  designate National Geospatial Data Asset data themes and one or more theme lead agencies"). My
  direct fetch found **"the webpage does not mention ArcGIS REST services, catalogs, or registries
  of government GIS servers at all"** — i.e., NGDA is an organizational/governance framework for
  *datasets by theme and lead agency*, not a technical index of live REST endpoints. The FGDC also
  operates a "Clearinghouse Registry" (mentioned in search results as "a database of all
  clearinghouse nodes participating in the clearinghouse activity") — I did not verify this
  further; it appears to be a legacy CSW/ISO-19115 metadata clearinghouse network rather than a
  flat server-address list, but **this is worth a follow-up look** if a rebuild wants an
  "official" federal source to complement Elfelt's list.
- **Security-scanning-based enumeration (Shodan/Censys)**: I confirmed these are general-purpose
  internet-scanning search engines capable of finding exposed services, but I found **no specific,
  citable project or dataset that uses Shodan/Censys to enumerate public ArcGIS REST servers** —
  this remains a plausible *technique* for a rebuild (fingerprinting the ArcGIS REST API's
  characteristic JSON responses at scale) but I found no prior art to point to. **Flagging as an
  open idea, not a verified existing resource.**

**Bottom line for the rebuild:** Elfelt's mappingsupport.com list appears to remain the single
best-known, actively-maintained, comprehensive, server-root-level catalog of US government ArcGIS
servers as of 2026 — larger now (7,500+) than when `govgis_nov2023` was built (~1,684, presumably
a subset of the list as it stood in Nov 2023). No superior alternative source turned up in this
research. GeoPlatform.gov/data.gov are complementary (dataset-level, not server-root-level) rather
than substitutable.

---

## 3. `restgdf` and comparable/competing ArcGIS REST client tools

Per the task's "known facts," `restgdf` (https://github.com/joshuasundance-swca/restgdf) is
described as an "Async, typed Python client for Esri ArcGIS REST services — turn
FeatureServer/MapServer layers into GeoDataFrames," MIT-licensed. I did not re-research restgdf
itself in depth (a sibling recon task covers it) but did spend a little effort trying to verify one
specific cross-cutting claim relevant to "comparable tools," documented below.

### Comparable/competing tools found

| Tool | Repo | License | Status (as found) | Notes |
|---|---|---|---|---|
| **ArcGIS API for Python** (Esri's official SDK) | https://github.com/Esri/arcgis-python-api | Apache-2.0 | Actively maintained; per my fetch: "2.2k stars, 1.1k forks, 39 releases, latest v2.4.3 (March 2026), 5,457 commits" | Official, broad (analysis, deep learning, geocoding, org/user admin), Pandas integration, but heavier-weight and more oriented to full GIS/org administration than a lightweight "grab this layer as a GeoDataFrame" client. |
| **ArcREST** (Esri's older community project) | https://github.com/Esri/ArcREST | — | **Archived March 31, 2020, read-only** (confirmed via search of GitHub) | Explicitly superseded by ArcGIS API for Python per repo history/discussion (GitHub issue "comparison to python API"). Not a live option for a rebuild. |
| **arcrest** (jasonbot's independent package, different from Esri's ArcREST) | https://github.com/jasonbot/arcrest | Apache-2.0 (confirmed via fetch: "Licensed under the Apache License, Version 2.0") | **Archived Jan 17, 2018, read-only** (confirmed via fetch) | Described in its README (quoted): "A Pythonic API for consuming REST services from ArcGIS server," targeting "ArcGIS 10.1 Server." Long dead. |
| **restapi** (Bolton-and-Menk-GIS) | https://github.com/Bolton-and-Menk-GIS/restapi | GPL-2.0 (confirmed via fetch) | **Actively maintained** — confirmed via fetch: "102 stars... Latest Release: Version 2.4.15 (January 9, 2026)... 46 releases" | README (quoted): "Python API designed to work externally with ArcGIS REST Services to query and extract data, and view service properties." Uses `arcpy` when available, falls back to open-source alternatives otherwise; synchronous, not async. Also on PyPI as `bmi-arcgis-restapi`. Still Python-2.7-compatible per fetch summary. Closest live competitor to restgdf in scope, but synchronous and GPL-licensed (vs. restgdf's async/MIT). |
| **esri2gpd** (PhilaController, i.e. City of Philadelphia's Controller's Office) | https://github.com/PhilaController/esri2gpd | MIT (confirmed via fetch) | Confirmed via fetch: "31 stars, 7 forks... 57 commits" | README (quoted): "A lightweight Python tool to scrape features from the ArcGIS Server REST API and return a geopandas GeoDataFrame." Explicitly modeled on an R package, `esri2sf`, per the fetched summary. Closest in *purpose* to restgdf (REST → GeoDataFrame) but smaller/lighter and (per what I could see) not async. |
| **esridumpgdf** (wchatx) | https://github.com/wchatx/esridumpgdf | not verified | not verified | Surfaced only in search-result titles; I did not fetch this repo. **Unverified**, noted only as another name worth a closer look. |
| **DataPillager** (Grant Herbert) | https://github.com/gdherbert/DataPillager | MIT (confirmed via fetch) | Confirmed via fetch: "75 stars, 10 forks... 10 releases (latest v2.3 dated June 30, 2026)" — **actively maintained** | README (quoted): "Download data from Esri REST service." Started in 2014 per one search summary ("Grant Herbert started building the tool... to download data from a REST service... to be available offline" — WebSearch synthesis, not a direct quote I could verify against the README itself, which per my direct fetch **does not mention Elfelt or any government server catalog**). Targets ArcGIS Desktop/Pro workflows (shapefile/file-geodatabase output), not a Python library API per se — more of a standalone downloader script/toolbox. |

### The DataPillager → restgdf lineage claim — flagged as unverified

Multiple independent WebSearch queries returned a consistent, specific claim: that Grant Herbert's
DataPillager (2019 era) inspired the creation of restgdf, and that a person named "Lucas Coleman"
showed restgdf's author DataPillager and ported it to Python 3. This is a detailed, specific claim
that appeared repeatedly and consistently across several different search queries, which would
normally suggest it's grounded in a real source. However, **I made a deliberate, thorough attempt
to verify it against primary sources and could not confirm it**: I directly fetched
- the restgdf GitHub README (raw, via `raw.githubusercontent.com` and via `cdn.jsdelivr.net`),
- the restgdf CHANGELOG.md,
- the restgdf PyPI JSON metadata (`description` and `summary` fields),
- the restgdf ReadTheDocs `llms.txt` and `llms-full.txt` (the latter explicitly designed to be a
  complete LLM-readable dump of the docs),

and **none of these contained any mention of "DataPillager," "Grant Herbert," "Lucas Coleman," or
"Elfelt."** I could not reach web.archive.org from this environment to check for an older README
revision that might have contained this text. **Conclusion: DataPillager and Grant Herbert are
independently well-documented as real (the DataPillager repo exists, is MIT-licensed, and is
actively maintained), but the specific claim that DataPillager directly inspired restgdf (with the
2019/Lucas-Coleman detail) is UNVERIFIED by me — it may be true (it could live in a source I
couldn't access, e.g. a blog post, a private conversation, or an older README revision), but I am
not confident enough in it to present it as established fact. Recommend the maintainer confirm this
directly if the lineage matters for the rebuild's documentation.**

### What's novel vs. restgdf, for the rebuild's benefit

- restgdf's differentiators, as given in the task's "known facts" and cross-checked against the
  comparable tools above: **async** (restapi and esri2gpd both appear synchronous) and **typed**
  (Pydantic-based, per the PyPI summary "lightweight async Esri REST client with optional
  GeoPandas extras" — confirmed via direct fetch of the PyPI JSON API).
- The official **ArcGIS API for Python** is the most credible "why not just use the official SDK"
  alternative for a rebuild — it's actively maintained by Esri, has by far the largest community
  (2.2k GitHub stars vs. restgdf's much smaller footprint, restapi's 102, esri2gpd's 31), and has
  first-class Pandas support. It is heavier and more oriented toward full ArcGIS Online/Enterprise
  administration than a narrow "crawl thousands of arbitrary public servers" use case, which is
  likely why a purpose-built async client was written instead — but this tradeoff is worth
  re-examining explicitly during a rebuild rather than assuming the original choice still holds.

---

## 4. Broader open-government-GIS community / standards bodies

- **Federal Geographic Data Committee (FGDC)** — https://www.fgdc.gov/. Interagency committee
  responsible for the National Spatial Data Infrastructure (NSDI) and, since the Geospatial Data
  Act of 2018, for designating **National Geospatial Data Asset (NGDA)** themes and lead agencies.
  Governance/organizational body, not a live technical server registry (see §2 above for detail).
- **GeoPlatform.gov** — the FGDC's public-facing catalog/discovery tool for NGDA data (see §2).
  Billed as making "Federal GeoData Findable, Accessible, Interoperable, and Reusable" (direct
  quote from https://www.geoplatform.gov/about).
- **Data.gov** — the general US federal open-data catalog; includes a large "geospatial" tag
  namespace under `catalog.data.gov`, much of it FGDC/NGDA-sourced metadata (confirmed via listing
  at https://catalog.data.gov/organization/fgdc-gov?tags=ngda, not deeply fetched).
- I found **no evidence of a standards body or community project that specifically catalogs "which
  government agencies run public ArcGIS REST servers"** at the granularity Elfelt's list operates
  at. The FGDC/GeoPlatform/data.gov ecosystem operates one level up — at the dataset/metadata-record
  level (ISO 19115-style), not the raw-server-endpoint level. This reinforces the §2 conclusion:
  Elfelt's list is filling a real gap that no official US government body currently fills in the
  same form.

---

## 5. Licensing / terms-of-use considerations for a rebuild

**Elfelt's source list itself:** see the direct quotes in §1 above — scraping the PDF is
prohibited, commercial use requires written permission, free derivative works are explicitly
permitted, and general "sharing" of the list is welcomed. **Any rebuild that re-crawls using this
list as a seed should treat these terms as binding and, if commercial use is contemplated, should
seek Elfelt's written permission first** — I found no blanket commercial license on the pages I
fetched.

**The downstream government ArcGIS servers themselves — legally more nuanced than "obviously
public domain":**

- **Federal-level servers**: clearly public domain under **17 U.S.C. § 105**, which I fetched and
  quoted directly from Cornell Law's Legal Information Institute
  (https://www.law.cornell.edu/uscode/text/17/105): *"Copyright protection under this title is
  not available for any work of the United States Government, but the United States Government is
  not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or
  otherwise."* (Note: this section was amended around 2020/2021 to add narrow subsections (b)-(d)
  granting copyright to *individual military-service-academy faculty* for certain scholarly
  publications — not relevant to GIS data, but worth knowing the section isn't purely "everything
  federal is PD" anymore in the literal statutory text, only in the general-work-product sense that
  applies to ordinary agency GIS data.)
- **The OPEN Government Data Act** requires federal agencies to publish data "in open,
  machine-readable formats and to apply open licenses," and per
  https://resources.data.gov/open-licenses/ (fetched, quoted): agencies are *"encouraged to use
  Creative Commons Zero (CC0) for new datasets."* Federal GIS layers should generally be safe to
  reuse.
- **State/local government servers are NOT uniformly public domain** — this is the important
  nuance for a project spanning "federal to city" servers. Quoted from
  https://crln.acrl.org/index.php/crlnews/article/view/17438/19245 (fetched): *"many states assert
  a copyright interest in their materials, and, most concerning, many more lack any clear legal
  guidance on the issue."* The same source describes **conflicting court rulings specifically
  about GIS data**: New York (Suffolk County) held state public-records law does *not* bar
  copyright assertions over GIS data; Florida (*Microdecisions v. Skinner*) held the opposite — "a
  state entity cannot copyright public records unless they fall under an exemption"; California
  aligned with Florida's public-domain position; South Carolina aligned with New York's
  copyright-permissive position. A cited resource, the "State Copyright Resource Center," tracks
  this state-by-state (not independently verified/fetched by me — noted secondhand from the crln
  article summary).
- **Practical implication for the rebuild:** it is **not safe to blanket-assume** all
  county/city/state ArcGIS layers are public domain the way federal ones are. A rebuild that
  intends to redistribute (not just index/link to) scraped metadata or geometry at scale should
  budget time to check individual state/agency terms-of-use pages (many ArcGIS REST services or
  their parent open-data portals publish their own explicit license, e.g. CC-BY or "public domain,"
  in service metadata or on the hosting portal) rather than assuming blanket public-domain status.
- **Esri's own platform terms**: I fetched https://www.esri.com/en-us/legal/terms/data-attributions
  and found it addresses Esri's *own* proprietary datasets (Esri Demographics, etc.) and does not
  address third-party government-hosted ArcGIS REST content at all — **not directly applicable**
  to this project's use case, and I found no Esri terms-of-use page that specifically restricts
  (or blesses) crawling publicly-exposed, no-login-required ArcGIS REST services hosted by
  government customers on their own infrastructure. This remains something to get real legal
  review on if the rebuild scales up redistribution, rather than something I can resolve via
  search.

---

## Summary of confidence levels

- **High confidence (directly quoted from primary source):** Elfelt's list existence, size
  history, update cadence, methodology, and license terms (§1); 17 U.S.C. §105 text and OPEN
  Government Data Act CC0 guidance (§5); the archived/deprecated status of Esri's ArcREST and
  jasonbot/arcrest (§3); restapi, esri2gpd, and DataPillager's own README descriptions and
  maturity stats (§3).
- **Medium confidence (consistent secondary/AI-synthesized sourcing, not a raw primary quote):**
  data.gov/GeoPlatform indexing datasets-not-servers (§2); state-copyright variability details
  beyond the four cited court cases (§5).
- **Explicitly unverified / flagged, do not treat as fact:** the DataPillager→restgdf/"Lucas
  Coleman"/2019 lineage claim (§3); whether GeoPlatform.gov offers bulk/API access to a flat
  server-root list (§2); esridumpgdf's maturity/status (§3); whether OpenAddresses ever acted on
  Elfelt's 2021 GitHub issue (§1); the FGDC Clearinghouse Registry's current technical form (§2).
