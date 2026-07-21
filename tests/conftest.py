"""Shared fixtures for the Stage 2 test suite.

`docs/modernization-plan.md` Stage 2 asks for failing tests covering
manifest validation, checksum mismatch, count mismatch, malformed metadata,
and retrieval expectations. This module builds small, fully synthetic
on-disk artifacts (NOT the real 4.28 GB legacy FAISS artifact) so those
tests run fast, deterministically, and offline, targeting the real
`govgis.models` / `govgis.artifacts` / `govgis.retrieval` API (written by
the sibling Stage 2 lanes; see each test module's docstring for the exact
exports/signatures asserted).

On-disk layout these fixtures write, matching `govgis/artifacts.py`'s
`ArtifactManifest` defaults (`index_filename="index.faiss"`,
`documents_filename="documents.jsonl"`):

    artifacts_dir/
        index.faiss     -- native FAISS index (faiss.write_index), an
                           IndexFlatIP over `VECTOR_DIM`-dim vectors.
        documents.jsonl  -- one JSON object per line, matching
                           `GisRecord`'s field set (id, name, type, url,
                           description, parent_service_description,
                           fields, metadata_text). Line N (0-indexed) is
                           the record for FAISS vector ID N.
        manifest.json    -- an `ArtifactManifest`-shaped JSON object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytest

VECTOR_DIM = 8
RECORD_COUNT = 6
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# Pinned revisions from docs/modernization-plan.md's "Artifact inventory".
EMBEDDING_MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
SOURCE_DATASET_REVISION = "ab1220e6823732093a1c8a0122af98f7da1f4217"
SCHEMA_VERSION = "1"
DISTANCE_METRIC = "cosine"
INDEX_FILENAME = "index.faiss"
DOCUMENTS_FILENAME = "documents.jsonl"
MANIFEST_FILENAME = "manifest.json"

# Index N here is FAISS vector ID N and the (N+1)th line of documents.jsonl.
_RECORD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "rec-flood",
        "name": "Statewide Flood Hazard Zones",
        "type": "FeatureServer",
        "description": "FEMA flood hazard zone polygons for the pilot county.",
        "url": "https://gis.example.gov/arcgis/rest/services/Flood/FeatureServer/0",
        "fields": ["ZONE", "BFE", "SFHA_TF"],
        "parent_service_description": "County hazard-mitigation layers.",
        "metadata_text": "name: Statewide Flood Hazard Zones\ntype: FeatureServer",
    },
    {
        "id": "rec-zoning",
        "name": "Unincorporated Zoning Designations",
        "type": "MapServer",
        "description": "Zoning designations for unincorporated county land.",
        "url": "https://gis.example.gov/arcgis/rest/services/Zoning/MapServer/9",
        "fields": ["ZONE_CODE", "ZONE_DESC"],
        "parent_service_description": "Planning and development layers.",
        "metadata_text": "name: Unincorporated Zoning Designations\ntype: MapServer",
    },
    {
        "id": "rec-wildfire",
        "name": "Wildfire Hazard Potential",
        "type": "MapServer",
        "description": "Modeled wildfire hazard potential classes.",
        "url": "https://gis.example.gov/arcgis/rest/services/Wildfire/MapServer/0",
        "fields": ["WHP_CLASS"],
        "parent_service_description": "Natural hazard layers.",
        "metadata_text": "name: Wildfire Hazard Potential\ntype: MapServer",
    },
    {
        "id": "rec-parcel",
        "name": "Impervious Area per Parcel",
        "type": "MapServer",
        "description": "Parcel-level impervious surface area.",
        "url": "https://gis.example.gov/arcgis/rest/services/Parcel/MapServer/145",
        "fields": ["PARCEL_ID", "IMPERVIOUS_SQFT"],
        "parent_service_description": "Stormwater utility layers.",
        "metadata_text": "name: Impervious Area per Parcel\ntype: MapServer",
    },
    {
        "id": "rec-hydrant",
        "name": "Hydrants - with Fire Flow",
        "type": "MapServer",
        "description": "Fire hydrant locations with rated fire flow.",
        "url": "https://gis.example.gov/arcgis/rest/services/Hydrants/MapServer/31",
        "fields": ["FLOW_GPM"],
        "parent_service_description": "Public safety layers.",
        "metadata_text": "name: Hydrants - with Fire Flow\ntype: MapServer",
    },
    {
        "id": "rec-wetland",
        "name": "NWI Freshwater Wetlands",
        "type": "FeatureServer",
        "description": "National Wetlands Inventory freshwater wetlands.",
        "url": "https://gis.example.gov/arcgis/rest/services/Wetlands/FeatureServer/11",
        "fields": ["WETLAND_TYPE"],
        "parent_service_description": "Environmental resource layers.",
        "metadata_text": "name: NWI Freshwater Wetlands\ntype: FeatureServer",
    },
)


def unit_vector(index: int, dim: int = VECTOR_DIM) -> np.ndarray:
    """The `index`-th standard basis vector in `dim` dimensions.

    Orthonormal by construction, so an `IndexFlatIP` search with vector `i`
    as the query yields cosine similarity 1.0 against record `i` and
    exactly 0.0 against every other record -- deterministic, no floating
    point tie-breaking ambiguity in test assertions.
    """
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


@dataclass(frozen=True)
class ArtifactFixture:
    artifacts_dir: Path
    index_path: Path
    records_path: Path
    manifest_path: Path
    records: tuple[dict[str, Any], ...]
    vectors: np.ndarray
    manifest: dict[str, Any]


def write_synthetic_artifacts(
    artifacts_dir: Path,
    *,
    record_count: int = RECORD_COUNT,
    vector_count: int | None = None,
) -> ArtifactFixture:
    """Write a small, fully synthetic FAISS/records/manifest artifact set.

    `vector_count` defaults to `record_count` (the normal, consistent
    case); pass a different value to deliberately desynchronize the FAISS
    index's vector count from the manifest/records count for a
    `VectorCountMismatchError` test, while keeping checksums internally
    consistent (checksums always match what's actually written to disk --
    only the *count* is deliberately wrong).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    records = _RECORD_SPECS[:record_count]
    written_vector_count = vector_count if vector_count is not None else record_count
    vectors = np.ascontiguousarray(
        np.stack([unit_vector(i) for i in range(written_vector_count)]),
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors)
    index_path = artifacts_dir / INDEX_FILENAME
    faiss.write_index(index, str(index_path))

    records_path = artifacts_dir / DOCUMENTS_FILENAME
    records_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_count": record_count,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "vector_dimensions": VECTOR_DIM,
        "distance_metric": DISTANCE_METRIC,
        "source_dataset_revision": SOURCE_DATASET_REVISION,
        "creation_command": "pytest synthetic fixture -- not a real conversion",
        "index_filename": INDEX_FILENAME,
        "documents_filename": DOCUMENTS_FILENAME,
        "checksums": {
            INDEX_FILENAME: hashlib.sha256(index_path.read_bytes()).hexdigest(),
            DOCUMENTS_FILENAME: hashlib.sha256(records_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ArtifactFixture(
        artifacts_dir=artifacts_dir,
        index_path=index_path,
        records_path=records_path,
        manifest_path=manifest_path,
        records=records,
        vectors=vectors,
        manifest=manifest,
    )


@pytest.fixture
def valid_artifacts(tmp_path: Path) -> ArtifactFixture:
    return write_synthetic_artifacts(tmp_path / "artifacts")


def fake_embed_known(query: str, mapping: dict[str, int]) -> np.ndarray:
    """A deterministic stand-in for BAAI/bge-large-en-v1.5's real output.

    `govgis.retrieval` embeds via an injected `SentenceTransformer`-shaped
    object rather than a plain function; see `tests/test_retrieval.py`'s
    `FakeEmbeddingModel` for the adapter that wraps this. Known queries map
    to an exact one-hot vector matching one fixture record. Unknown queries
    -- including every malformed Stage 0 query -- fall back to the zero
    vector deliberately: this stand-in must never raise regardless of input
    content, which is exactly the property under test for the
    malformed/empty-input cases.
    """
    index = mapping.get(query)
    if index is None:
        return np.zeros(VECTOR_DIM, dtype=np.float32)
    return unit_vector(index)
