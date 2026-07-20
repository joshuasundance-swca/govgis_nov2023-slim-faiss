"""Acceptance tests for `govgis.artifacts` (Stage 2).

Written as this lane's "failing tests first" deliverable
(`docs/modernization-plan.md` Stage 2's "manifest validation, checksum
mismatch, count mismatch, malformed metadata" action) against the real
`govgis.artifacts` module: `ArtifactManifest`, `ManifestValidationError`,
`ChecksumMismatchError`, `RecordCountMismatchError`,
`VectorCountMismatchError`, `DistanceMetric`, `load_manifest`,
`verify_checksums`, `load_records`, `validate_vector_count`, and
`load_artifact`. See `tests/conftest.py` for the synthetic on-disk artifact
layout these tests assume, and this lane's final report for the
reconciliation between this contract and this lane's original,
independently-authored one (the artifacts.py lane's implementation, not
this test file, changed to close the gap).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from govgis.artifacts import (
    ArtifactManifest,
    ChecksumMismatchError,
    DistanceMetric,
    ManifestValidationError,
    RecordCountMismatchError,
    VectorCountMismatchError,
    load_artifact,
    load_manifest,
    load_records,
    validate_vector_count,
    verify_checksums,
)
from govgis.models import GisRecord
from tests.conftest import (
    DOCUMENTS_FILENAME,
    INDEX_FILENAME,
    ArtifactFixture,
    write_synthetic_artifacts,
)

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


def _dummy_manifest(*, record_count: int) -> ArtifactManifest:
    """A schema-valid `ArtifactManifest` with placeholder checksums.

    For tests that only exercise `load_records`'s per-line/record-count
    behavior and never call `verify_checksums`, so the checksum values
    themselves don't need to be real.
    """
    return ArtifactManifest(
        schema_version="1",
        record_count=record_count,
        embedding_model_name="BAAI/bge-large-en-v1.5",
        embedding_model_revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        vector_dimensions=8,
        distance_metric=DistanceMetric.COSINE,
        source_dataset_revision="ab1220e6823732093a1c8a0122af98f7da1f4217",
        creation_command="test fixture",
        checksums={INDEX_FILENAME: "0" * 64, DOCUMENTS_FILENAME: "0" * 64},
    )


def test_load_manifest_reads_valid_manifest(valid_artifacts: ArtifactFixture) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)

    assert isinstance(manifest, ArtifactManifest)
    assert manifest.record_count == len(valid_artifacts.records)
    assert manifest.vector_dimensions == valid_artifacts.vectors.shape[1]
    assert manifest.embedding_model_name == "BAAI/bge-large-en-v1.5"
    expected_checksums = valid_artifacts.manifest["checksums"]
    assert manifest.checksums[INDEX_FILENAME] == expected_checksums[INDEX_FILENAME]
    assert manifest.checksums[DOCUMENTS_FILENAME] == expected_checksums[DOCUMENTS_FILENAME]


def test_load_manifest_raises_on_malformed_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_manifest(manifest_path)


def test_load_manifest_raises_on_missing_manifest_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError):
        load_manifest(tmp_path / "does-not-exist.json")


def test_load_manifest_raises_on_missing_required_field(
    valid_artifacts: ArtifactFixture,
) -> None:
    payload = json.loads(valid_artifacts.manifest_path.read_text(encoding="utf-8"))
    del payload["record_count"]
    valid_artifacts.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_manifest(valid_artifacts.manifest_path)


def test_load_manifest_raises_on_wrong_field_type(valid_artifacts: ArtifactFixture) -> None:
    payload = json.loads(valid_artifacts.manifest_path.read_text(encoding="utf-8"))
    payload["record_count"] = "six"
    valid_artifacts.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_manifest(valid_artifacts.manifest_path)


def test_load_manifest_raises_when_checksums_missing_a_required_entry(
    valid_artifacts: ArtifactFixture,
) -> None:
    payload = json.loads(valid_artifacts.manifest_path.read_text(encoding="utf-8"))
    del payload["checksums"][DOCUMENTS_FILENAME]
    valid_artifacts.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_manifest(valid_artifacts.manifest_path)


def test_load_records_reads_valid_records(valid_artifacts: ArtifactFixture) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)
    records = load_records(valid_artifacts.records_path, manifest=manifest)

    assert len(records) == len(valid_artifacts.records)
    assert all(isinstance(record, GisRecord) for record in records)
    first, expected_first = records[0], valid_artifacts.records[0]
    assert first.id == expected_first["id"]
    assert first.name == expected_first["name"]
    assert first.url == expected_first["url"]
    assert first.fields == list(expected_first["fields"])


def test_load_records_raises_on_malformed_json_line(valid_artifacts: ArtifactFixture) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)
    with valid_artifacts.records_path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")

    with pytest.raises(ManifestValidationError):
        load_records(valid_artifacts.records_path, manifest=manifest)


def test_load_records_raises_on_missing_required_field(tmp_path: Path) -> None:
    records_path = tmp_path / "documents.jsonl"
    records_path.write_text(json.dumps({"id": "rec-incomplete"}) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        load_records(records_path, manifest=_dummy_manifest(record_count=1))


def test_load_records_raises_on_unsupported_file_extension(tmp_path: Path) -> None:
    records_path = tmp_path / "documents.parquet"
    records_path.write_bytes(b"not really parquet either")

    with pytest.raises(ManifestValidationError):
        load_records(records_path, manifest=_dummy_manifest(record_count=1))


def test_load_records_raises_on_record_count_mismatch(
    valid_artifacts: ArtifactFixture,
) -> None:
    payload = json.loads(valid_artifacts.manifest_path.read_text(encoding="utf-8"))
    # Checksums stay correct for the (untouched) records file -- isolates
    # this test to the count check, not a checksum mismatch.
    payload["record_count"] = len(valid_artifacts.records) + 1
    valid_artifacts.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_manifest(valid_artifacts.manifest_path)

    with pytest.raises(RecordCountMismatchError):
        load_records(valid_artifacts.records_path, manifest=manifest)


def test_verify_checksums_passes_for_valid_fixture(valid_artifacts: ArtifactFixture) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)

    verify_checksums(
        manifest,
        index_path=valid_artifacts.index_path,
        records_path=valid_artifacts.records_path,
    )


def test_verify_checksums_raises_on_index_checksum_mismatch(
    valid_artifacts: ArtifactFixture,
) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)
    with valid_artifacts.index_path.open("ab") as handle:
        handle.write(b"\x00tamper")

    with pytest.raises(ChecksumMismatchError):
        verify_checksums(
            manifest,
            index_path=valid_artifacts.index_path,
            records_path=valid_artifacts.records_path,
        )


def test_verify_checksums_raises_on_records_checksum_mismatch(
    valid_artifacts: ArtifactFixture,
) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)
    with valid_artifacts.records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_INJECTED_RECORD) + "\n")

    with pytest.raises(ChecksumMismatchError):
        verify_checksums(
            manifest,
            index_path=valid_artifacts.index_path,
            records_path=valid_artifacts.records_path,
        )


def test_verify_checksums_raises_when_referenced_file_is_missing(
    valid_artifacts: ArtifactFixture,
) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)
    valid_artifacts.index_path.unlink()

    with pytest.raises(ManifestValidationError):
        verify_checksums(
            manifest,
            index_path=valid_artifacts.index_path,
            records_path=valid_artifacts.records_path,
        )


def test_validate_vector_count_passes_when_counts_match(
    valid_artifacts: ArtifactFixture,
) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)

    validate_vector_count(manifest, len(valid_artifacts.records))


def test_validate_vector_count_raises_on_mismatch(valid_artifacts: ArtifactFixture) -> None:
    manifest = load_manifest(valid_artifacts.manifest_path)

    with pytest.raises(VectorCountMismatchError):
        validate_vector_count(manifest, len(valid_artifacts.records) - 1)


def test_load_artifact_fails_closed_on_corrupt_records(
    valid_artifacts: ArtifactFixture,
) -> None:
    with valid_artifacts.records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_INJECTED_RECORD) + "\n")

    with pytest.raises(ChecksumMismatchError):
        load_artifact(
            manifest_path=valid_artifacts.manifest_path,
            index_path=valid_artifacts.index_path,
            records_path=valid_artifacts.records_path,
        )


def test_load_artifact_succeeds_for_valid_fixture(tmp_path: Path) -> None:
    fixture = write_synthetic_artifacts(tmp_path / "artifacts")

    manifest, records = load_artifact(
        manifest_path=fixture.manifest_path,
        index_path=fixture.index_path,
        records_path=fixture.records_path,
    )

    assert manifest.record_count == len(fixture.records)
    assert len(records) == len(fixture.records)
