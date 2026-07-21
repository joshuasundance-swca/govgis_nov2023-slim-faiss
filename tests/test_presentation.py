"""Acceptance tests for `govgis.presentation` (Stage 3).

Targets the real `govgis.presentation` module: `html_to_plain_text`,
`escape_markdown`, `sanitize_text_field`, `sanitize_answer_text`,
`validate_safe_url`, and `gis_record_to_safe_presentation`. Two fixtures are
required by this lane's brief and both are exercised here against the real
data they were captured from, not a hand-built stand-in:

- `docs/stage0/query_set.json`'s `adversarial_html_injection` entry (an
  `<img onerror>`/`<script>` payload);
- the real HTML-bearing record documented in
  `docs/stage0/evidence/html_in_description_example.md` (legitimate ArcGIS
  metadata, not an attack -- the risk is architectural, per that file).

Both must come out as inert plain text / escaped output, never live
HTML/JS, per this lane's brief.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import HttpUrl

from govgis.models import GisRecord
from govgis.presentation import (
    escape_markdown,
    gis_record_to_safe_presentation,
    html_to_plain_text,
    sanitize_answer_text,
    sanitize_text_field,
    validate_safe_url,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUERY_SET_PATH = REPOSITORY_ROOT / "docs" / "stage0" / "query_set.json"

# Verbatim from docs/stage0/evidence/html_in_description_example.md -- a
# real record (id d0d5cc25-2d77-42d4-8941-09f3652a4f41, "ROW Dimensions")
# from the pinned dataset revision, not a synthetic stand-in.
_REAL_HTML_DESCRIPTION = (
    '<DIV STYLE="text-align:Left;"><DIV><DIV><P><SPAN>This annotation data '
    "set shows the right-of-way widths for public streets and alleys. This "
    "data is part of the official City of Lubbock base map and is "
    "maintained at a scale of 1\" = 100'.</SPAN></P></DIV></DIV></DIV>"
)
_REAL_HTML_EXPECTED_SUBSTRING = "right-of-way widths for public streets and alleys"


def _query_set_entry(entry_id: str) -> dict[str, object]:
    query_set: list[dict[str, object]] = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    return next(item for item in query_set if item["id"] == entry_id)


def _unescape_markdown(escaped: str) -> str:
    """Inverse of `escape_markdown`, for round-trip assertions in tests only."""
    return re.sub(r"\\(.)", r"\1", escaped)


def _make_record(**overrides: object) -> GisRecord:
    defaults: dict[str, object] = {
        "id": "rec-1",
        "name": "Test Layer",
        "type": "FeatureServer",
        "url": "https://gis.example.gov/arcgis/rest/services/Test/FeatureServer/0",
        "description": "A plain description.",
        "parent_service_description": "A parent service.",
        "fields": ["FIELD_A", "FIELD_B"],
        "metadata_text": "name: Test Layer",
    }
    defaults.update(overrides)
    return GisRecord.model_validate(defaults)


# -- html_to_plain_text -------------------------------------------------


def test_html_to_plain_text_extracts_real_dataset_record() -> None:
    text = html_to_plain_text(_REAL_HTML_DESCRIPTION)

    assert _REAL_HTML_EXPECTED_SUBSTRING in text
    assert "<" not in text
    assert ">" not in text
    assert "DIV" not in text
    assert "SPAN" not in text


def test_html_to_plain_text_drops_script_and_style_content_entirely() -> None:
    raw = (
        "<p>Visible</p><script>document.cookie</script>"
        "<style>.x{color:red}</style><p>Also visible</p>"
    )

    text = html_to_plain_text(raw)

    assert "document.cookie" not in text
    assert "color:red" not in text
    assert "Visible" in text
    assert "Also visible" in text


def test_html_to_plain_text_inserts_space_at_block_boundaries() -> None:
    text = html_to_plain_text("<p>Foo</p><p>Bar</p>")

    assert "FooBar" not in text
    assert text == "Foo Bar"


def test_html_to_plain_text_passes_through_plain_text_unchanged_besides_whitespace() -> None:
    assert html_to_plain_text("No markup here.") == "No markup here."


def test_html_to_plain_text_neutralizes_query_set_html_injection_entry() -> None:
    entry = _query_set_entry("adversarial_html_injection")

    text = html_to_plain_text(str(entry["query"]))

    assert "<script" not in text.lower()
    assert "onerror" not in text.lower()
    assert "document.location" not in text
    assert "document.cookie" not in text


# -- escape_markdown ------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "**bold** _italic_ [link](javascript:alert(1))",
        "# Heading\n- list item",
        "right-of-way widths, 1\" = 100'.",
        "plain text with no special characters",
    ],
)
def test_escape_markdown_round_trips_to_original_text(raw: str) -> None:
    escaped = escape_markdown(raw)

    assert _unescape_markdown(escaped) == raw


def test_escape_markdown_neutralizes_markdown_link_syntax() -> None:
    escaped = escape_markdown("[Click me](javascript:alert(document.cookie))")

    assert "](" not in escaped
    assert "[" not in escaped or "\\[" in escaped


def test_escape_markdown_neutralizes_raw_angle_brackets() -> None:
    # As if an HTML entity like &lt;script&gt; had already been decoded to
    # literal text by html_to_plain_text -- escape_markdown is the layer
    # that must still stop it from being interpreted as a live tag.
    escaped = escape_markdown("<script>alert(1)</script>")

    assert re.search(r"(?<!\\)<", escaped) is None
    assert "\\<script" in escaped


# -- sanitize_text_field ---------------------------------------------------


def test_sanitize_text_field_none_becomes_empty_string() -> None:
    assert sanitize_text_field(None) == ""


def test_sanitize_text_field_neutralizes_real_html_record_end_to_end() -> None:
    sanitized = sanitize_text_field(_REAL_HTML_DESCRIPTION)

    assert "<" not in sanitized or re.search(r"(?<!\\)<", sanitized) is None
    assert _unescape_markdown(sanitized) == html_to_plain_text(_REAL_HTML_DESCRIPTION)


def test_sanitize_text_field_neutralizes_query_set_html_injection_entry() -> None:
    entry = _query_set_entry("adversarial_html_injection")

    sanitized = sanitize_text_field(str(entry["query"]))

    assert "<script" not in sanitized.lower()
    assert "onerror" not in sanitized.lower()
    assert re.search(r"(?<!\\)<", sanitized) is None
    plain = _unescape_markdown(sanitized)
    assert "<script" not in plain.lower()
    assert "onerror" not in plain.lower()


# -- sanitize_answer_text ---------------------------------------------------


def test_sanitize_answer_text_neutralizes_markdown_and_html() -> None:
    answer = (
        "Here you go: [click here](javascript:alert(document.cookie)) <script>alert(1)</script>"
    )

    sanitized = sanitize_answer_text(answer)

    assert "](" not in sanitized
    assert "<script" not in sanitized.lower()


def test_sanitize_answer_text_handles_prompt_injection_entry_without_raising() -> None:
    entry = _query_set_entry("adversarial_prompt_injection")

    sanitized = sanitize_answer_text(str(entry["query"]))

    assert isinstance(sanitized, str)


# -- validate_safe_url -------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.gov/service/0",
        "http://example.gov/service/0",
    ],
)
def test_validate_safe_url_accepts_http_and_https(raw: str) -> None:
    result = validate_safe_url(raw)

    assert isinstance(result, HttpUrl)
    assert result.scheme in ("http", "https")


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "ftp://example.gov/file",
        "not a url at all",
    ],
)
def test_validate_safe_url_rejects_everything_else(raw: str | None) -> None:
    assert validate_safe_url(raw) is None


# -- gis_record_to_safe_presentation --------------------------------------


def test_gis_record_to_safe_presentation_carries_through_safe_fields() -> None:
    record = _make_record()

    result = gis_record_to_safe_presentation(record, score=0.87)

    assert result.id == record.id
    assert result.score == 0.87
    assert str(result.url).startswith("https://gis.example.gov")


def test_gis_record_to_safe_presentation_converts_real_html_description() -> None:
    record = _make_record(description=_REAL_HTML_DESCRIPTION)

    result = gis_record_to_safe_presentation(record)

    assert re.search(r"(?<!\\)<", result.description) is None
    assert _unescape_markdown(result.description) == html_to_plain_text(
        _REAL_HTML_DESCRIPTION,
    )
    assert "right\\-of\\-way" in result.description or "right-of-way" in _unescape_markdown(
        result.description,
    )


def test_gis_record_to_safe_presentation_neutralizes_html_injection_in_description() -> None:
    entry = _query_set_entry("adversarial_html_injection")
    record = _make_record(description=str(entry["query"]))

    result = gis_record_to_safe_presentation(record)

    assert "<script" not in result.description.lower()
    assert "onerror" not in result.description.lower()
    assert re.search(r"(?<!\\)<", result.description) is None


def test_gis_record_to_safe_presentation_neutralizes_html_injection_in_name() -> None:
    entry = _query_set_entry("adversarial_html_injection")
    record = _make_record(name=str(entry["query"]))

    result = gis_record_to_safe_presentation(record)

    assert "<script" not in result.name.lower()
    assert "onerror" not in result.name.lower()


def test_gis_record_to_safe_presentation_none_description_becomes_empty_string() -> None:
    record = _make_record(description=None)

    result = gis_record_to_safe_presentation(record)

    assert result.description == ""


def test_gis_record_to_safe_presentation_none_parent_service_description_stays_none() -> None:
    record = _make_record(parent_service_description=None)

    result = gis_record_to_safe_presentation(record)

    assert result.parent_service_description is None


def test_gis_record_to_safe_presentation_sanitizes_each_field_entry() -> None:
    record = _make_record(fields=["<b>bold field</b>", "plain-field"])

    result = gis_record_to_safe_presentation(record)

    assert len(result.fields) == 2
    assert "<b>" not in result.fields[0]
    assert "bold field" in _unescape_markdown(result.fields[0])


def test_gis_record_to_safe_presentation_drops_unsafe_url_scheme() -> None:
    record = _make_record(url="javascript:alert(document.cookie)")

    result = gis_record_to_safe_presentation(record)

    assert result.url is None


def test_gis_record_to_safe_presentation_default_score_is_none() -> None:
    record = _make_record()

    result = gis_record_to_safe_presentation(record)

    assert result.score is None
