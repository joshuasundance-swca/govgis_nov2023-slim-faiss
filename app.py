"""Gradio Blocks composition and event wiring for the govgis semantic-search Space.

Per ``docs/modernization-plan.md``'s "Target architecture", this module owns
*only* UI composition and event wiring: it loads a ``govgis.retrieval``
index, calls ``govgis.retrieval.RetrievalIndex.search``, converts each
``SearchResult`` to a ``SafePresentationRecord`` via
``govgis.presentation.gis_record_to_safe_presentation`` (the only module
permitted to construct that type), and renders the result. It must never
construct a ``SafePresentationRecord`` itself and must never bind ``gr.HTML``
to dataset-, user-, or model-sourced content -- see "Unsafe rendering" in the
plan. This stage is retrieval-only: the optional provider-neutral
answer-synthesis call (Stage 4) is not implemented here, only left a place to
compose in (see ``_render_results_markdown`` below).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import gradio as gr

from govgis.artifacts import MANIFEST_FILENAME, ArtifactError
from govgis.models import SafePresentationRecord, SearchResult
from govgis.presentation import gis_record_to_safe_presentation
from govgis.retrieval import (
    DEFAULT_TOP_K,
    RetrievalArtifactPaths,
    RetrievalError,
    RetrievalIndex,
    load_retrieval_index,
)

ARTIFACT_DIR_ENV_VAR = "GOVGIS_ARTIFACT_DIR"

# Falls back to the real one-time-converted artifact set produced locally by
# Stage 2 (see this file's Stage 3 lane report) so local/browser testing
# exercises real retrieval. The deployed Space always sets GOVGIS_ARTIFACT_DIR
# explicitly (Stage 5/6 scope, per the modernization plan's mid-run
# amendment) -- this default only matters for local development and must
# never be relied on in production.
_LOCAL_DEV_ARTIFACT_DIR = Path(__file__).resolve().parent / "scratch" / "artifacts"

_INDEX_FILENAME = "index.faiss"
_DOCUMENTS_FILENAME = "documents.jsonl"

_MIN_TOP_K = 1
_MAX_TOP_K = 10

_APP_TITLE = "govgis semantic search"
_APP_DESCRIPTION = (
    "Semantic search over `govgis_nov2023` GIS metadata (government ArcGIS "
    "servers and layers). Search works without signing in or an API key. "
    "AI-generated answers are not available yet."
)
_IDLE_MESSAGE = "Enter a query above and press **Search**."
_LOADING_MESSAGE = "_Searching…_"
_NO_RESULTS_MESSAGE = "No results found for that search. Try different or broader terms."
# Deliberately generic: govgis.artifacts/govgis.retrieval error messages can
# include server-side file paths (see e.g. ArtifactError subclasses), which
# is fine for logs but not something to echo verbatim to a public UI.
_ERROR_MESSAGE = (
    "**Search is temporarily unavailable.** Please try again in a moment; "
    "if the problem persists, retrieval results may be visible again shortly."
)

_retrieval_index: RetrievalIndex | None = None


def _artifact_dir() -> Path:
    raw = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    return Path(raw).expanduser() if raw else _LOCAL_DEV_ARTIFACT_DIR


def _artifact_paths() -> RetrievalArtifactPaths:
    artifact_dir = _artifact_dir()
    return RetrievalArtifactPaths(
        index_path=artifact_dir / _INDEX_FILENAME,
        records_path=artifact_dir / _DOCUMENTS_FILENAME,
        manifest_path=artifact_dir / MANIFEST_FILENAME,
    )


def _get_retrieval_index() -> RetrievalIndex:
    # Lazy, cached on first successful load: importing this module must never
    # load a ~1.2 GB embedding model or a 865k-vector FAISS index, since that
    # would make plain `import app` (e.g. from a test collector) prohibitively
    # slow and would fail outright wherever GOVGIS_ARTIFACT_DIR's artifacts
    # don't exist. A failed load is not cached, so the *next* search retries
    # rather than staying broken for the life of the process.
    global _retrieval_index
    if _retrieval_index is None:
        _retrieval_index = load_retrieval_index(_artifact_paths())
    return _retrieval_index


def _render_record_markdown(record: SafePresentationRecord) -> str:
    lines = [f"### {record.name}", f"**Type:** {record.type}"]
    if record.score is not None:
        lines.append(f"**Relevance score:** {record.score:.4f}")
    lines.append(record.description)
    if record.parent_service_description:
        lines.append(f"**Parent service:** {record.parent_service_description}")
    if record.fields:
        lines.append("**Fields:** " + ", ".join(record.fields))
    if record.url is not None:
        lines.append(f"[Open in ArcGIS REST]({record.url})")
    return "\n\n".join(lines)


def _render_results_markdown(results: list[SearchResult]) -> str:
    # Stage 4 composition point: an optional govgis.providers.* answer
    # synthesis call belongs here, between retrieval and presentation, per
    # the plan's Target architecture ("app.py composes retrieval.py output
    # with an optional providers/*.py call, in that order, and passes the
    # result to presentation.py"). No provider exists yet -- retrieval-only
    # for this stage. `SafePresentationRecord` has no answer field yet either
    # (see govgis/presentation.py's module docstring); rendering a
    # synthesized answer is an open question left to Stage 4/the coordinator,
    # not solved here.
    safe_records = [
        gis_record_to_safe_presentation(result.record, score=result.score) for result in results
    ]
    return "\n\n---\n\n".join(_render_record_markdown(record) for record in safe_records)


def _handle_search(query: str, top_k: float) -> Iterator[str]:
    stripped_query = query.strip()
    if not stripped_query:
        yield _IDLE_MESSAGE
        return

    yield _LOADING_MESSAGE
    try:
        retrieval_index = _get_retrieval_index()
        results = retrieval_index.search(stripped_query, top_k=int(top_k))
    except RetrievalError, ArtifactError:
        yield _ERROR_MESSAGE
        return

    if not results:
        yield _NO_RESULTS_MESSAGE
        return

    yield _render_results_markdown(results)


def build_app() -> gr.Blocks:
    # Assigned from the constructor (not `with gr.Blocks(...) as demo:`) so
    # `demo`'s static type stays `gr.Blocks`: Gradio's `Blocks.__enter__` has
    # no return annotation, which mypy otherwise widens to `Any` and leaks
    # through this function's declared return type.
    demo = gr.Blocks(title=_APP_TITLE)
    with demo:
        gr.Markdown(f"# {_APP_TITLE}", sanitize_html=True)
        gr.Markdown(_APP_DESCRIPTION, sanitize_html=True)

        with gr.Row():
            query_box = gr.Textbox(
                label="Search query",
                placeholder="e.g. Where are the FEMA flood hazard zones?",
                scale=4,
            )
            top_k_slider = gr.Slider(
                minimum=_MIN_TOP_K,
                maximum=_MAX_TOP_K,
                value=DEFAULT_TOP_K,
                step=1,
                label="Number of results",
                scale=1,
            )

        with gr.Row():
            search_button = gr.Button("Search", variant="primary")
            retry_button = gr.Button("Retry last search")

        # sanitize_html=True is Gradio's own default; set explicitly so a
        # future Gradio default change can't silently reopen the raw-HTML
        # XSS path this app is built to avoid (see "Unsafe rendering" in the
        # modernization plan). Every value ever bound to this component is a
        # static string built in this module or a SafePresentationRecord
        # rendered through _render_record_markdown -- never gr.HTML, and
        # never a raw dataset/user/model-sourced value.
        results_markdown = gr.Markdown(_IDLE_MESSAGE, sanitize_html=True)

        last_query_state = gr.State("")

        search_event = search_button.click(
            fn=_handle_search,
            inputs=[query_box, top_k_slider],
            outputs=[results_markdown],
        )
        search_event.then(
            fn=lambda query: query,
            inputs=[query_box],
            outputs=[last_query_state],
        )

        submit_event = query_box.submit(
            fn=_handle_search,
            inputs=[query_box, top_k_slider],
            outputs=[results_markdown],
        )
        submit_event.then(
            fn=lambda query: query,
            inputs=[query_box],
            outputs=[last_query_state],
        )

        retry_event = retry_button.click(
            fn=_handle_search,
            inputs=[last_query_state, top_k_slider],
            outputs=[results_markdown],
        )
        retry_event.then(
            fn=lambda query: query,
            inputs=[last_query_state],
            outputs=[query_box],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
