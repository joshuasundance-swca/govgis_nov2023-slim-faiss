"""Package marker for repo-maintenance scripts.

An explicit package (matching `tests/__init__.py`'s rationale) so
`from scripts.check_gr_html_safety import ...` resolves the same way under
every invocation instead of depending on pytest's import-mode/sys.path
heuristics.
"""

from __future__ import annotations
