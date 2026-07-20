# Real record: raw HTML in `description`

Retained as concrete grounding for the plan's "Unsafe rendering" risk
section — 29,283 of 865,304 records (3.4%) in the pinned dataset revision
(`joshuasundance/govgis_nov2023-slim-spatial@ab1220e6823732093a1c8a0122af98f7da1f4217`)
have HTML markup in their `description` field, of which this is one real
example (matched by `<[a-zA-Z]` in `description`, verified against the live
geoparquet 2026-07-20):

- `id`: `d0d5cc25-2d77-42d4-8941-09f3652a4f41`
- `name`: `ROW Dimensions`
- `type`: `Annotation Layer`
- `url`: `https://pubgis.ci.lubbock.tx.us/server/rest/services/Layers/ROW_BaseMap/MapServer/0`
- `description` (verbatim, unmodified):

  ```html
  <DIV STYLE="text-align:Left;"><DIV><DIV><P><SPAN>This annotation data set shows the right-of-way widths for public streets and alleys. This data is part of the official City of Lubbock base map and is maintained at a scale of 1" = 100'.</SPAN></P></DIV></DIV></DIV>
  ```

This is legitimate ArcGIS REST metadata (not an injected attack payload) —
the current legacy app renders it via `st.components.v1.html(...)`
(sandboxed iframe). The risk the plan's "Unsafe rendering" section names is
architectural: Gradio's `gr.HTML` performs no sanitization at all, so the
same class of real, already-present content would render unsandboxed if
ever bound to `gr.HTML` post-migration. No literal `<script>` or
`javascript:` content was found anywhere in `description` or `name` across
the full 865,304-row corpus (checked 2026-07-20) — the risk is about what
the target architecture must guard against, not an exploit already present
in the data.
