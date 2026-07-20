from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


class SpaceBuildContractTests(unittest.TestCase):
    def test_space_uses_python_311_and_one_streamlit_version_authority(self) -> None:
        metadata = _space_metadata()
        requirement_names = [
            line.split("==", maxsplit=1)[0].strip().lower()
            for line in (REPOSITORY_ROOT / "requirements.txt")
            .read_text(
                encoding="utf-8",
            )
            .splitlines()
            if line and not line.startswith("#")
        ]

        self.assertEqual("3.11", metadata.get("python_version"))
        self.assertEqual("1.29.0", metadata.get("sdk_version"))
        self.assertNotIn("streamlit", requirement_names)


if __name__ == "__main__":
    unittest.main()
