"""Fail-closed manifest and record loading for the native FAISS artifact layout.

Defines the artifact manifest schema (schema version, record count,
embedding model name + revision, vector dimensions, distance metric, source
dataset revision, creation command, and SHA-256 checksums) plus load/validate
functions that raise a typed ``ArtifactError`` subclass -- never silently
proceed -- on checksum mismatch, count mismatch, or malformed metadata.

No pickle, no ``eval``, and no LangChain deserialization anywhere in this
file: the manifest is plain JSON, documents are JSON Lines, and file
integrity is plain SHA-256 hashing. This is the exact risk the "Critical
trust boundary: serialized FAISS data" section of the modernization plan
exists to close -- keep it that way.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from govgis.models import GisRecord

MANIFEST_FILENAME: Final[str] = "manifest.json"

_SHA256_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_CHUNK_BYTES: Final[int] = 1024 * 1024


class ArtifactError(Exception):
    """Base class for fail-closed artifact validation and loading errors."""


class ManifestValidationError(ArtifactError):
    """The manifest is missing, unreadable, not valid JSON, or fails schema validation.

    Also raised for a checksummed file that does not exist and for a
    documents file that is unreadable, malformed, or fails per-record
    schema validation -- all "malformed metadata" in the Stage 2 gate sense.
    """


class ChecksumMismatchError(ArtifactError):
    """A SHA-256 checksum recorded in the manifest does not match the file on disk."""


class RecordCountMismatchError(ArtifactError):
    """The manifest-declared record count does not match the number of records loaded."""


class VectorCountMismatchError(ArtifactError):
    """The FAISS index's vector count does not match the manifest-declared record count."""


class DistanceMetric(StrEnum):
    """Distance metrics a FAISS index may be built with -- must be explicit, never assumed."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class ArtifactManifest(BaseModel):
    """Manifest schema from the plan's "Critical trust boundary" section.

    ``documents_filename`` currently must be a ``.jsonl`` file --
    ``load_records`` below only implements JSON Lines (see its docstring for
    why Parquet is out of scope for this pass).
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: str = Field(min_length=1)
    record_count: int = Field(gt=0)
    embedding_model_name: str = Field(min_length=1)
    embedding_model_revision: str = Field(min_length=1)
    vector_dimensions: int = Field(gt=0)
    # `strict=False` override: model_config's strict=True otherwise rejects the
    # plain JSON string every real manifest.json contains -- pydantic v2 strict
    # mode requires an actual DistanceMetric instance, not str-to-StrEnum
    # coercion, even though DistanceMetric IS a str subtype. Confirmed live: a
    # round-trip smoke test loading a real JSON-serialized manifest raised
    # `Input should be an instance of DistanceMetric [type=is_instance_of]`
    # without this override.
    distance_metric: DistanceMetric = Field(strict=False)
    source_dataset_revision: str = Field(min_length=1)
    creation_command: str = Field(min_length=1)
    index_filename: str = Field(default="index.faiss", min_length=1)
    documents_filename: str = Field(default="documents.jsonl", min_length=1)
    checksums: dict[str, str]

    @field_validator("checksums")
    @classmethod
    def _checksums_are_hex_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        for filename, digest in value.items():
            if not _SHA256_HEX_PATTERN.fullmatch(digest):
                msg = (
                    f"checksum for {filename!r} is not a 64-character lowercase hex SHA-256 digest"
                )
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _checksums_cover_artifact_files(self) -> Self:
        required = {self.index_filename, self.documents_filename}
        missing = required - self.checksums.keys()
        if missing:
            msg = f"manifest checksums are missing entries for: {sorted(missing)}"
            raise ValueError(msg)
        return self


def load_manifest(manifest_path: Path) -> ArtifactManifest:
    """Load and schema-validate the manifest at ``manifest_path``.

    Fails closed: raises ``ManifestValidationError`` -- never returns a
    partially-valid manifest -- if the file is missing, unreadable, not
    valid JSON, or fails Pydantic schema validation.
    """
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"could not read manifest at {manifest_path}: {exc}"
        raise ManifestValidationError(msg) from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"manifest at {manifest_path} is not valid JSON: {exc}"
        raise ManifestValidationError(msg) from exc

    try:
        return ArtifactManifest.model_validate(raw)
    except ValidationError as exc:
        msg = f"manifest at {manifest_path} failed schema validation: {exc}"
        raise ManifestValidationError(msg) from exc


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(manifest: ArtifactManifest, *, index_path: Path, records_path: Path) -> None:
    """Recompute SHA-256 for the index and documents files and compare.

    ``index_path``/``records_path`` are the actual on-disk locations (they
    need not share a directory or match ``manifest.index_filename``/
    ``manifest.documents_filename`` verbatim); the manifest's
    ``index_filename``/``documents_filename`` are only used as the lookup
    keys into ``manifest.checksums``. Fails closed: raises
    ``ManifestValidationError`` if a file does not exist, or
    ``ChecksumMismatchError`` if any recomputed digest disagrees with the
    manifest.
    """
    for filename, actual_path in (
        (manifest.index_filename, index_path),
        (manifest.documents_filename, records_path),
    ):
        expected_digest = manifest.checksums[filename]
        if not actual_path.is_file():
            msg = (
                f"manifest checksums reference {filename!r}, "
                f"but no such file exists at {actual_path}"
            )
            raise ManifestValidationError(msg)
        actual_digest = _sha256_of_file(actual_path)
        if actual_digest != expected_digest:
            msg = (
                f"checksum mismatch for {filename!r}: manifest declares {expected_digest}, "
                f"computed {actual_digest} from {actual_path}"
            )
            raise ChecksumMismatchError(msg)


def load_records(records_path: Path, *, manifest: ArtifactManifest) -> list[GisRecord]:
    """Load documents as JSON Lines, validating each line as a ``GisRecord``.

    Line order (0-indexed) is the integer vector ID the paired FAISS index
    uses -- this module does not persist vector IDs explicitly, so callers
    must preserve ordering end-to-end from conversion through retrieval.

    Only JSON Lines is implemented; the plan's target layout also allows
    ``documents.parquet`` but no Parquet reader (pyarrow/pandas) is a
    declared project dependency, so a non-``.jsonl`` ``records_path`` fails
    closed here rather than silently mis-parsing.

    Fails closed: raises ``ManifestValidationError`` on any unreadable
    file, malformed JSON line, or line that fails ``GisRecord`` schema
    validation, and ``RecordCountMismatchError`` if the number of records
    loaded does not match ``manifest.record_count``.
    """
    if records_path.suffix != ".jsonl":
        msg = (
            f"unsupported documents format {records_path.name!r}: "
            "only JSON Lines (.jsonl) is implemented"
        )
        raise ManifestValidationError(msg)

    documents_path = records_path
    records: list[GisRecord] = []
    try:
        with documents_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"malformed JSON at {documents_path}:{line_number}: {exc}"
                    raise ManifestValidationError(msg) from exc
                try:
                    records.append(GisRecord.model_validate(raw_record))
                except ValidationError as exc:
                    msg = (
                        f"record at {documents_path}:{line_number} failed schema validation: {exc}"
                    )
                    raise ManifestValidationError(msg) from exc
    except OSError as exc:
        msg = f"could not read documents file at {documents_path}: {exc}"
        raise ManifestValidationError(msg) from exc

    if len(records) != manifest.record_count:
        msg = (
            f"manifest declares record_count={manifest.record_count} but "
            f"{len(records)} records were loaded from {documents_path}"
        )
        raise RecordCountMismatchError(msg)

    return records


def validate_vector_count(manifest: ArtifactManifest, vector_count: int) -> None:
    """Assert a FAISS index's vector count matches the manifest's declared record count.

    Callers (e.g. ``govgis/retrieval.py``) own opening the FAISS index
    itself; this function only compares the resulting count so this module
    stays free of a FAISS dependency. Fails closed: raises
    ``VectorCountMismatchError`` on any mismatch.
    """
    if vector_count != manifest.record_count:
        msg = (
            f"FAISS index contains {vector_count} vectors but manifest declares "
            f"record_count={manifest.record_count}"
        )
        raise VectorCountMismatchError(msg)


def load_artifact(
    *,
    manifest_path: Path,
    index_path: Path,
    records_path: Path,
) -> tuple[ArtifactManifest, list[GisRecord]]:
    """Load, checksum-verify, and record-count-verify one artifact set.

    Composes ``load_manifest`` -> ``verify_checksums`` -> ``load_records`` in
    that order and fails closed at the first violation. Does not open the
    FAISS index itself (that stays ``govgis/retrieval.py``'s job, to keep
    this module free of a FAISS dependency) -- see ``validate_vector_count``
    for that half of the Stage 2 gate's "vector count equals metadata count"
    check.
    """
    manifest = load_manifest(manifest_path)
    verify_checksums(manifest, index_path=index_path, records_path=records_path)
    records = load_records(records_path, manifest=manifest)
    return manifest, records
