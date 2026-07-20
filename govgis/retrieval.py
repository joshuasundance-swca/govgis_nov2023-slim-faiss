"""Query embedding and native FAISS search for the govgis retrieval core.

Loads a native ``index.faiss`` (``faiss.read_index`` -- never
``FAISS.deserialize_from_bytes``/LangChain) plus its records via
``govgis.artifacts``, embeds a query with BAAI/bge-large-en-v1.5, and returns
ranked ``SearchResult``s. No LangChain import belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from govgis.artifacts import ArtifactManifest, load_artifact, validate_vector_count
from govgis.models import GisRecord, SearchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# BAAI/bge-large-en-v1.5's own model card documents this instruction for
# short-query-to-long-passage retrieval; it is applied to queries only, never
# to the indexed passages themselves.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_TOP_K = 3


class RetrievalError(RuntimeError):
    """Raised when the retrieval core cannot load or query its artifacts."""


@dataclass(frozen=True, slots=True)
class RetrievalArtifactPaths:
    """Filesystem locations of one retrieval artifact set (see artifacts.py)."""

    index_path: Path
    records_path: Path
    manifest_path: Path


@dataclass(slots=True)
class RetrievalIndex:
    """A loaded native FAISS index plus its validated records, ready to query."""

    index: faiss.Index
    records: tuple[GisRecord, ...]
    manifest: ArtifactManifest
    embedding_model: SentenceTransformer

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        # An empty/whitespace-only query is a valid Stage 0 query-set case
        # (docs/stage0/query_set.json's `empty_input` entry), not an error --
        # the product contract is "return zero-or-more results without
        # raising", so this returns [] rather than embedding a blank string.
        if not query or not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        limit = min(top_k, self.index.ntotal)
        if limit == 0:
            return []

        query_vector = embed_query(self.embedding_model, query)
        distances, indices = self.index.search(query_vector, limit)

        results: list[SearchResult] = []
        for distance, record_index in zip(distances[0], indices[0], strict=True):
            if record_index < 0:
                continue
            record = self.records[record_index]
            results.append(SearchResult(record=record, score=float(distance)))
        return results


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    *,
    device: str = "cpu",
    cache_folder: str | Path | None = None,
) -> SentenceTransformer:
    """Load the sentence-transformers embedding model used for query encoding."""
    # sentence-transformers' SentenceTransformer resolves to Any under mypy
    # (its torch.nn.Module base loses its static type through the library's
    # own dynamic construction path); the explicit annotation here re-pins
    # the static type instead of leaking Any into every call site.
    model: SentenceTransformer = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=str(cache_folder) if cache_folder is not None else None,
    )
    return model


def embed_query(model: SentenceTransformer, query: str) -> NDArray[np.float32]:
    """Embed a single query string, applying bge-large-en-v1.5's documented
    query instruction and L2-normalizing to match a cosine/inner-product
    FAISS index.
    """
    vector = model.encode(
        [query],
        prompt=QUERY_INSTRUCTION,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vector, dtype=np.float32)


def load_retrieval_index(
    paths: RetrievalArtifactPaths,
    *,
    embedding_model: SentenceTransformer | None = None,
) -> RetrievalIndex:
    """Load and cross-validate one retrieval artifact set: manifest, records,
    and the native FAISS index, failing closed on any mismatch.

    Manifest schema validation, checksum verification, and record-count
    validation are delegated to ``govgis.artifacts.load_artifact`` (which
    raises its own ``ArtifactError`` subclasses); this function additionally
    opens the FAISS index itself and cross-checks its vector count and
    dimensionality, since ``artifacts.py`` is deliberately FAISS-free.
    """
    if not paths.index_path.is_file():
        raise RetrievalError(f"FAISS index not found: {paths.index_path}")

    manifest, records = load_artifact(
        manifest_path=paths.manifest_path,
        index_path=paths.index_path,
        records_path=paths.records_path,
    )

    try:
        index = faiss.read_index(str(paths.index_path))
    except RuntimeError as exc:
        raise RetrievalError(
            f"failed to read FAISS index {paths.index_path}: {exc}",
        ) from exc

    validate_vector_count(manifest, index.ntotal)
    if index.d != manifest.vector_dimensions:
        raise RetrievalError(
            f"FAISS index dimension ({index.d}) does not match manifest "
            f"vector_dimensions ({manifest.vector_dimensions}) for {paths.index_path}",
        )

    model = embedding_model or load_embedding_model(manifest.embedding_model_name)

    return RetrievalIndex(
        index=index,
        records=tuple(records),
        manifest=manifest,
        embedding_model=model,
    )


def search(
    query: str,
    paths: RetrievalArtifactPaths,
    *,
    top_k: int = DEFAULT_TOP_K,
    embedding_model: SentenceTransformer | None = None,
) -> list[SearchResult]:
    """Load the artifact set at ``paths`` and return its top-k results for
    ``query``. Callers that will issue many queries against the same
    artifacts should call ``load_retrieval_index`` once and reuse the
    returned ``RetrievalIndex.search`` instead of re-loading per call.
    """
    retrieval_index = load_retrieval_index(paths, embedding_model=embedding_model)
    return retrieval_index.search(query, top_k=top_k)
