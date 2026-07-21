"""Acceptance tests for `govgis.retrieval` (Stage 2).

Written as this lane's "failing tests first" deliverable
(`docs/modernization-plan.md` Stage 2's "retrieval expectations" action)
against the real `govgis.retrieval` module: `RetrievalArtifactPaths`,
`RetrievalError`, `RetrievalIndex`, `load_retrieval_index`, and `search`.
Uses `tests/conftest.py`'s synthetic FAISS/records/manifest fixture and a
`FakeEmbeddingModel` duck-typing `SentenceTransformer.encode` (deliberately
NOT the real ~1.2 GB BAAI/bge-large-en-v1.5 model -- see this lane's final
report for that scope decision) so retrieval logic is exercised fast,
deterministically, and offline.

`RetrievalIndex.search` returns `[]` for an empty/whitespace-only query
rather than raising, per `docs/modernization-plan.md`'s requirement
(carried from Stage 0's query set) that the `empty_input` case "return
zero-or-more results without raising" --
`test_search_handles_stage0_malformed_and_empty_queries[empty_input]` below
covers it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from govgis.artifacts import ChecksumMismatchError, VectorCountMismatchError
from govgis.models import SearchResult
from govgis.retrieval import (
    RetrievalArtifactPaths,
    RetrievalError,
    load_retrieval_index,
    search,
)
from tests.conftest import (
    VECTOR_DIM,
    ArtifactFixture,
    fake_embed_known,
    write_synthetic_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUERY_SET_PATH = REPOSITORY_ROOT / "docs" / "stage0" / "query_set.json"

# tests/conftest.py's _RECORD_SPECS order: flood=0, zoning=1, wildfire=2,
# parcel=3, hydrant=4, wetland=5.
_QUERY_VECTOR_MAP = {
    "synthetic query about flooding": 0,
    "synthetic query about zoning": 1,
}
_INJECTED_RECORD = {
    "id": "rec-injected",
    "name": "n",
    "type": "t",
    "url": "https://example.gov",
    "description": "d",
    "fields": [],
    "parent_service_description": "",
    "metadata_text": "x",
}


class FakeEmbeddingModel:
    """Duck-types the slice of `SentenceTransformer` `embed_query` calls.

    Returns a deterministic one-hot vector for known queries (see
    `_QUERY_VECTOR_MAP`) and the zero vector for anything else -- including
    every malformed/empty Stage 0 query -- so it never raises regardless of
    input content.
    """

    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    def encode(
        self,
        texts: list[str],
        *,
        prompt: str,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        del prompt, normalize_embeddings, convert_to_numpy
        (query,) = texts
        vector = fake_embed_known(query, self._mapping)
        return np.expand_dims(vector, axis=0)


def _fake_model() -> SentenceTransformer:
    # FakeEmbeddingModel duck-types only the `.encode(...)` slice of
    # SentenceTransformer that govgis.retrieval.embed_query actually calls;
    # the cast documents that intentional test-double substitution for
    # mypy rather than widening `load_retrieval_index`'s real parameter
    # type just to accommodate a test.
    return cast("SentenceTransformer", FakeEmbeddingModel(_QUERY_VECTOR_MAP))


def _paths(fixture: ArtifactFixture) -> RetrievalArtifactPaths:
    return RetrievalArtifactPaths(
        index_path=fixture.index_path,
        records_path=fixture.records_path,
        manifest_path=fixture.manifest_path,
    )


def test_load_retrieval_index_reads_native_faiss_index(
    valid_artifacts: ArtifactFixture,
) -> None:
    retrieval_index = load_retrieval_index(_paths(valid_artifacts), embedding_model=_fake_model())

    assert retrieval_index.index.ntotal == len(valid_artifacts.records)
    assert retrieval_index.index.d == VECTOR_DIM
    assert len(retrieval_index.records) == len(valid_artifacts.records)


def test_search_returns_best_match_first(valid_artifacts: ArtifactFixture) -> None:
    results = search(
        "synthetic query about flooding",
        _paths(valid_artifacts),
        top_k=3,
        embedding_model=_fake_model(),
    )

    assert len(results) == 3
    assert all(isinstance(result, SearchResult) for result in results)
    assert results[0].record.id == "rec-flood"
    assert results[0].score == pytest.approx(1.0)
    # Orthonormal fixture vectors: every non-matching record scores 0.
    assert all(result.score <= results[0].score for result in results)


def test_search_returns_full_record_fields(valid_artifacts: ArtifactFixture) -> None:
    results = search(
        "synthetic query about zoning",
        _paths(valid_artifacts),
        top_k=1,
        embedding_model=_fake_model(),
    )
    top = results[0]
    expected = next(r for r in valid_artifacts.records if r["id"] == "rec-zoning")

    assert top.record.name == expected["name"]
    assert top.record.type == expected["type"]
    assert top.record.description == expected["description"]
    assert top.record.url == expected["url"]
    assert top.record.fields == list(expected["fields"])
    assert top.record.parent_service_description == expected["parent_service_description"]


def test_search_respects_top_k_parameter(valid_artifacts: ArtifactFixture) -> None:
    results = search(
        "synthetic query about flooding",
        _paths(valid_artifacts),
        top_k=2,
        embedding_model=_fake_model(),
    )

    assert len(results) == 2


def test_search_top_k_larger_than_corpus_does_not_crash(
    valid_artifacts: ArtifactFixture,
) -> None:
    results = search(
        "synthetic query about flooding",
        _paths(valid_artifacts),
        top_k=50,
        embedding_model=_fake_model(),
    )

    assert 0 < len(results) <= len(valid_artifacts.records)


def test_search_propagates_checksum_mismatch_fail_closed(
    valid_artifacts: ArtifactFixture,
) -> None:
    with valid_artifacts.records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_INJECTED_RECORD) + "\n")

    with pytest.raises(ChecksumMismatchError):
        search(
            "synthetic query about flooding",
            _paths(valid_artifacts),
            top_k=3,
            embedding_model=_fake_model(),
        )


def test_load_retrieval_index_propagates_vector_count_mismatch(
    tmp_path: Path,
) -> None:
    fixture = write_synthetic_artifacts(
        tmp_path / "artifacts",
        record_count=6,
        vector_count=5,
    )

    with pytest.raises(VectorCountMismatchError):
        load_retrieval_index(_paths(fixture), embedding_model=_fake_model())


def test_search_raises_retrieval_error_on_missing_index_file(
    valid_artifacts: ArtifactFixture,
) -> None:
    valid_artifacts.index_path.unlink()

    with pytest.raises(RetrievalError):
        search(
            "synthetic query about flooding",
            _paths(valid_artifacts),
            top_k=3,
            embedding_model=_fake_model(),
        )


@pytest.mark.parametrize(
    "query_id",
    ["empty_input", "malformed_long_gibberish", "malformed_unicode_mixed"],
)
def test_search_handles_stage0_malformed_and_empty_queries(
    valid_artifacts: ArtifactFixture,
    query_id: str,
) -> None:
    """docs/stage0/query_set.json's malformed/empty entries must not crash
    retrieval -- they should return zero-or-more results without raising.

    `empty_input`'s query is `""`; `RetrievalIndex.search` returns `[]` for
    an empty/whitespace-only query rather than raising.
    """
    query_set: list[dict[str, Any]] = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in query_set if item["id"] == query_id)

    results = search(
        entry["query"],
        _paths(valid_artifacts),
        top_k=3,
        embedding_model=_fake_model(),
    )

    assert isinstance(results, list)
    assert len(results) <= 3
