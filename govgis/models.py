"""Typed data shapes shared across the retrieval, artifact, and presentation layers.

See ``docs/modernization-plan.md``'s "Target architecture" section: this
module owns every model in the pipeline except the artifact manifest
(``govgis/artifacts.py``). ``GisRecord`` and ``SearchResult`` carry
untrusted, dataset-sourced content and must never be rendered directly.
``SafePresentationRecord`` is the only shape ``app.py`` may render; per the
plan, only ``govgis/presentation.py`` is permitted to construct one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class GisRecord(BaseModel):
    """A single ArcGIS layer/service record as parsed from the dataset.

    Untrusted: `description`, `parent_service_description`, and `metadata_text`
    originate from third-party ArcGIS server operators, not this project (see
    "Unsafe rendering" and "Prompt injection" in the modernization plan). Field
    shape matches the pinned dataset revision's geoparquet columns (`id`,
    `name`, `type`, `description`, `url`, `metadata_text`) plus `fields` and
    `parent_service_description`, both parsed from `metadata_text`'s YAML by
    the legacy app's `doc_md` template. `url` is intentionally `str`, not a
    validated URL type: scheme/format validation is `presentation.py`'s job
    (see "Target architecture"), not this raw parsing layer's, so a malformed
    or non-http(s) URL in the source data does not block loading the record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    type: str
    url: str
    description: str | None = None
    parent_service_description: str | None = None
    fields: list[str] = Field(default_factory=list)
    metadata_text: str


class SearchResult(BaseModel):
    """A `GisRecord` ranked by a retrieval-time similarity/relevance score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: GisRecord
    score: float


class SafePresentationRecord(BaseModel):
    """The plain-text/escaped shape `app.py` is permitted to render.

    Per "Target architecture" and "Unsafe rendering" in the modernization
    plan: this type is constructed only by `govgis/presentation.py`, which is
    responsible for HTML-to-plain-text conversion, Markdown escaping, and
    restricting `url` to `http`/`https` links validated from both the source
    record and any LLM-generated answer text. `url` uses pydantic's `HttpUrl`
    (which itself rejects non-http(s) schemes) so an invalid or unsafe link
    is represented as `None` rather than smuggled through as a plain string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    type: str
    description: str
    parent_service_description: str | None = None
    fields: list[str] = Field(default_factory=list)
    url: HttpUrl | None = None
    score: float | None = None
