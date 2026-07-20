"""Assert the Hugging Face Space build contract against real repo files.

The Space build reads two things, and only these two things, per
`docs/modernization-plan.md` ("Target architecture" /
"spaces-config-reference"): `README.md`'s YAML front matter (Python/SDK
version) and `requirements.txt` (the pip dependency set actually installed
into the Space container). Neither `pyproject.toml` nor `uv.lock` is read by
the Space build itself -- they govern the dev/CI environment only, and
Stage 1 generates `requirements.txt` from `uv.lock` as a committed artifact
(see the modernization plan's Stage 1 actions and gate).

This module asserts the post-Stage-1 contract: Python 3.14, and a
requirements.txt derived from the new pyproject.toml dependency set (no
LangChain/LangSmith, the new runtime dependency list present). `sdk`/
`sdk_version` assert the Stage 3 Gradio contract (see app.py and this file's
`test_readme_front_matter_declares_gradio_sdk`).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The "govgis" project's declared runtime dependencies (see the modernization
# plan's Stage 1 shared contract). Names are PyPI-distribution names,
# normalized for comparison in _normalize_name below.
EXPECTED_RUNTIME_DEPENDENCIES = frozenset(
    {
        "gradio",
        "pydantic",
        "huggingface_hub",
        "sentence-transformers",
        "faiss-cpu",
        "anthropic",
        "openai",
        "PyYAML",
        "httpx",
        "tenacity",
    },
)

# Removed by the "Target architecture" section: LangChain/LangSmith are no
# longer part of the runtime dependency set, and the legacy pinned SDK
# (Streamlit is supplied by the Space `sdk`/`sdk_version`, never pip) must
# never reappear as a pip requirement.
FORBIDDEN_REQUIREMENTS = frozenset({"langchain", "langsmith", "streamlit"})


def _normalize_name(name: str) -> str:
    """Normalize a PyPI distribution name per PEP 503 (case/`_`/`.` -> `-`)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _space_metadata() -> dict[str, str]:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    front_matter = re.match(r"\A---\r?\n(.*?)\r?\n---", readme, re.DOTALL)
    if front_matter is None:
        raise AssertionError("README.md must start with Space YAML front matter")

    metadata: dict[str, str] = {}
    for line in front_matter.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _requirement_names() -> set[str]:
    """Extract top-level PyPI distribution names from requirements.txt.

    Must tolerate the shape `uv export --format requirements.txt` actually
    produces: a header comment block, optional per-line `--hash=...`
    continuation lines (indented, no `==`), and PEP 508 environment markers
    -- a naive `line.split("==")` over every non-comment line breaks on the
    indented hash-continuation lines.
    """
    text = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    names: set[str] = set()
    for raw_line in text.splitlines():
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        requirement = raw_line.split(";", maxsplit=1)[0].split("\\", maxsplit=1)[0].strip()
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        if match:
            names.add(_normalize_name(match.group(0)))
    return names


class SpaceBuildContractTests(unittest.TestCase):
    def test_readme_front_matter_declares_python_3_14(self) -> None:
        metadata = _space_metadata()
        self.assertEqual("3.14", metadata.get("python_version"))

    def test_readme_front_matter_declares_gradio_sdk(self) -> None:
        # Stage 3: sdk/sdk_version now track the actual installed Gradio
        # version (see app.py's Stage 3 lane report) rather than the legacy
        # Streamlit contract this test asserted pre-Stage-3.
        metadata = _space_metadata()
        self.assertEqual("gradio", metadata.get("sdk"))
        self.assertEqual("6.20.0", metadata.get("sdk_version"))

    def test_requirements_txt_reflects_new_dependency_set(self) -> None:
        requirement_names = _requirement_names()
        expected_names = {_normalize_name(name) for name in EXPECTED_RUNTIME_DEPENDENCIES}
        missing = expected_names - requirement_names
        self.assertFalse(
            missing,
            f"requirements.txt is missing expected runtime dependencies: {sorted(missing)}",
        )

    def test_requirements_txt_excludes_removed_or_sdk_supplied_packages(self) -> None:
        requirement_names = _requirement_names()
        forbidden_names = {_normalize_name(name) for name in FORBIDDEN_REQUIREMENTS}
        present = forbidden_names & requirement_names
        self.assertFalse(
            present,
            f"requirements.txt must not contain: {sorted(present)}",
        )


if __name__ == "__main__":
    unittest.main()
