"""Test package marker.

An explicit package (rather than pytest's rootless single-file collection)
so `from tests.conftest import ...` resolves the same way in every test
file and under every invocation (`pytest`, `python -m pytest`, an IDE
runner) instead of depending on pytest's import-mode/sys.path heuristics.
"""

from __future__ import annotations
