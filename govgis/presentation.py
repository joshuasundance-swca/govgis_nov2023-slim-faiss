"""Safe conversion from untrusted `GisRecord`/answer text into `SafePresentationRecord`.

Per `docs/modernization-plan.md`'s "Target architecture" and "Unsafe
rendering" sections, this is the ONLY module permitted to construct
`SafePresentationRecord` (`govgis/models.py`). `app.py` may render the
result but must never construct or bypass it, and must never bind raw
dataset/user/model content to `gr.HTML` (which performs no sanitization at
all -- see `tests/test_no_unsafe_html_binding.py`).

Responsibilities, all defense-in-depth on top of Gradio's own
`gr.Markdown(sanitize_html=True)` default (kept, not relied on alone):

- convert source HTML descriptions to plain text (`html_to_plain_text`) --
  real dataset rows carry legitimate HTML markup (~3.4% of records; see
  `docs/stage0/evidence/html_in_description_example.md`), not just attacks;
- escape CommonMark's full ASCII-punctuation-escapable set
  (`escape_markdown`) so neither Markdown syntax nor a literal `<`/`>` left
  over from decoded HTML entities can be reinterpreted as live markup by a
  downstream Markdown renderer;
- restrict `url` to `http`/`https` schemes only (`validate_safe_url`),
  representing anything else as `None` rather than smuggling it through as
  a plain string.

`sanitize_answer_text` applies the same HTML-to-plain-text + Markdown-escape
pipeline to LLM-*generated* text, per the plan's requirement to "re-apply
that [http/https] allowlist to any URL appearing in LLM-generated answer
text, not only to URLs in retrieved records" -- escaping every URL-bearing
character neutralizes this without needing a separate link-extraction pass.
`SafePresentationRecord` has no field for an answer today (Stage 2's
`govgis/models.py` predates Stage 4's provider work); this function is a
standalone building block for whichever module wires up answer rendering
once that field/decision exists, not threaded through
`gis_record_to_safe_presentation` as an unused parameter -- see this lane's
final report for the open question this leaves for Stage 4/the coordinator.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Final

from pydantic import HttpUrl, ValidationError

from govgis.models import GisRecord, SafePresentationRecord

_ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# Tags whose content must never reach the plain-text output, regardless of
# nesting -- <script>/<style> bodies are not "text" in any sense a reader
# needs, and dropping them outright (rather than emitting them as escaped
# text) is the simplest correct policy.
_DROPPED_CONTENT_TAGS: Final[frozenset[str]] = frozenset({"script", "style"})

# Tags that separate adjacent text runs with whitespace so stripping them
# does not glue unrelated words together (e.g. "<p>Foo</p><p>Bar</p>" must
# not become "FooBar"). Not exhaustive of all HTML block elements -- covers
# what real ArcGIS metadata descriptions actually use (see
# docs/stage0/evidence/html_in_description_example.md: nested DIV/P/SPAN).
_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "div", "p", "br", "li", "ul", "ol", "tr", "td", "th", "table",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr",
    },
)

# CommonMark's full ASCII-punctuation-escapable set. Escaping all of it
# (not just the "obviously syntactic" subset like `*`/`_`/`[`) is
# deliberately conservative: it also covers `<`/`>`, so a literal `<script>`
# left over from a decoded HTML entity (e.g. source text containing
# `&lt;script&gt;`) cannot be reinterpreted as raw HTML by a downstream
# Markdown-to-HTML renderer that passes untranslated HTML through. Backslash
# escapes render invisibly (the backslash is consumed, not displayed), so
# this does not visibly alter ordinary prose.
_MARKDOWN_SPECIAL_CHARS: Final[str] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_MARKDOWN_ESCAPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    "[" + re.escape(_MARKDOWN_SPECIAL_CHARS) + "]",
)
_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")


class _HtmlToPlainTextParser(HTMLParser):
    """Collects text nodes, dropping `<script>`/`<style>` content and
    inserting a separating space at block-tag boundaries.

    `convert_charrefs=True` (the default, kept explicit) means `handle_data`
    already receives HTML entities decoded to their literal characters --
    e.g. `&lt;script&gt;` arrives as the six-character string `<script>`,
    never as a re-parsed tag. That literal text still needs
    `escape_markdown` before it is safe to embed in a Markdown document (see
    module docstring); this parser's only job is plain-text extraction.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _DROPPED_CONTENT_TAGS:
            self._drop_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROPPED_CONTENT_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def html_to_plain_text(raw: str) -> str:
    """Strip HTML markup (including any `<script>`/`<style>` content) from
    `raw` and return collapsed, stripped plain text.

    Safe to call on text that contains no HTML at all -- a plain string with
    no tags passes through with only whitespace collapsed, so callers can
    apply this uniformly to every untrusted text field rather than needing
    to first detect whether a given field happens to contain markup.
    """
    parser = _HtmlToPlainTextParser()
    parser.feed(raw)
    parser.close()
    return _WHITESPACE_PATTERN.sub(" ", parser.get_text()).strip()


def escape_markdown(text: str) -> str:
    """Backslash-escape every CommonMark ASCII-punctuation-escapable
    character in `text` so it renders as inert literal text, never as
    Markdown syntax or a passed-through raw HTML tag.
    """
    return _MARKDOWN_ESCAPE_PATTERN.sub(lambda match: "\\" + match.group(0), text)


def sanitize_text_field(raw: str | None) -> str:
    """`html_to_plain_text` then `escape_markdown`; `None` becomes `""`.

    The single pipeline every untrusted, dataset-sourced text field
    (`name`, `type`, `description`, `parent_service_description`, each
    `fields` entry) is run through before it is safe for `app.py` to render
    via `gr.Markdown`.
    """
    if raw is None:
        return ""
    return escape_markdown(html_to_plain_text(raw))


def sanitize_answer_text(text: str) -> str:
    """Sanitize LLM-*generated* answer text for eventual safe rendering.

    Applies the identical `sanitize_text_field` pipeline used for
    dataset-sourced fields. This is deliberately the strictest possible
    reading of the plan's "re-apply [the http/https] allowlist to any URL
    appearing in LLM-generated answer text" requirement: escaping every
    Markdown-special character (including `:`, `/`, `[`, `]`, `(`, `)`)
    means no URL in the answer -- validated or not -- can render as a live,
    clickable Markdown link; it renders as plain, inert text instead. A
    future Stage 4 decision may want validated http/https links to stay
    clickable, which would need a link-extraction pass ahead of this one --
    left to that stage's own gate, not assumed here.
    """
    return sanitize_text_field(text)


def validate_safe_url(raw: str | None) -> HttpUrl | None:
    """Parse `raw` as an `http`/`https` URL, or return `None`.

    Pydantic's `HttpUrl` already rejects non-http(s) schemes and malformed
    input by construction; the explicit scheme check below is deliberate
    defense-in-depth documentation of that requirement, not dead code, in
    case `HttpUrl`'s accepted-scheme set ever changes upstream.
    """
    if not raw:
        return None
    try:
        url = HttpUrl(raw)
    except ValidationError:
        return None
    if url.scheme not in _ALLOWED_URL_SCHEMES:
        return None
    return url


def gis_record_to_safe_presentation(
    record: GisRecord,
    *,
    score: float | None = None,
) -> SafePresentationRecord:
    """Convert one untrusted `GisRecord` into a `SafePresentationRecord`.

    The only function anywhere in this codebase permitted to construct
    `SafePresentationRecord` (see module docstring and
    `docs/modernization-plan.md`'s "Target architecture"). `score` is
    accepted separately rather than read off `record` because `GisRecord`
    itself carries no ranking score -- callers typically have a
    `SearchResult` (`record` + `score`) and should pass
    `result.record`/`result.score` through explicitly.
    """
    return SafePresentationRecord(
        # Not run through sanitize_text_field: id is an opaque lookup key
        # (a UUID in every real record observed so far -- see
        # docs/stage0/evidence/html_in_description_example.md), not prose
        # meant for display. If a caller ever renders it as visible text,
        # sanitize it at that render call the same way description/name
        # are sanitized here.
        id=record.id,
        name=sanitize_text_field(record.name),
        type=sanitize_text_field(record.type),
        description=sanitize_text_field(record.description),
        parent_service_description=(
            sanitize_text_field(record.parent_service_description)
            if record.parent_service_description is not None
            else None
        ),
        fields=[sanitize_text_field(field) for field in record.fields],
        url=validate_safe_url(record.url),
        score=score,
    )
