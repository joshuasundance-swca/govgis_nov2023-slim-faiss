"""Fail if `gr.HTML` is ever bound to non-literal content.

`docs/modernization-plan.md`'s "Unsafe rendering" section and Stage 3's Gate
require this as a *durable* check, not just a one-time browser test: Gradio's
own docs confirm `gr.HTML` performs no sanitization at all, so any
dataset-, user-, or model-sourced value reaching it is an XSS path
(the legacy app's real bug, `app.py:225,227` at the pinned baseline SHA
`5b3caca`, was exactly this). A future change could silently reintroduce it
with no other regression signal, so this check runs on every CI run
(`.github/workflows/ci.yml`), not only when someone remembers to test it by
hand in a browser.

Policy enforced: a call that resolves to Gradio's `HTML` component may only
receive a hardcoded, developer-authored literal string (or no content
argument at all) as its `value`. Anything else -- a variable, an attribute
access, an f-string with a substitution, a function/method call, a
subscript, etc. -- fails the check. This is intentionally stricter than
"sanitized"; the modernization plan's rule is that dataset/user/model
content must never reach `gr.HTML` at all, so the check does not try to
distinguish "safe-looking" dynamic content from unsafe -- it rejects all
dynamic content, full stop. Use `gr.Markdown(sanitize_html=True)` or plain
text for anything that isn't a string the developer wrote directly at the
call site.

Known scope limits (acceptable for this check's purpose -- it targets the
realistic Gradio Blocks-composition pattern this codebase uses, not
arbitrary indirection):

- only `govgis/` and `app.py` are scanned (the modules the Stage 3 shared
  contract names as the composition/rendering surface);
- only direct calls of the form `gr.HTML(...)` / `gradio.HTML(...)` /
  `HTML(...)` (imported by name) are recognized -- a component reference
  smuggled through an intermediate variable before being called
  (`ctor = gr.HTML; ctor(x)`) is not tracked;
- `.update(value=...)`-style mutation calls are not tracked (Gradio's
  current Blocks API does not use that pattern for this codebase).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_TARGETS: tuple[Path, ...] = (REPO_ROOT / "govgis", REPO_ROOT / "app.py")
_GRADIO_MODULES = frozenset({"gradio", "gradio.components"})


@dataclass(frozen=True, slots=True)
class Violation:
    """One `gr.HTML(...)` call whose value argument is not a literal string."""

    path: Path
    line: int
    col: int
    snippet: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.col}: gr.HTML() value must be a "
            f"hardcoded literal string, not {self.snippet}"
        )


def _is_literal_string(node: ast.expr) -> bool:
    """True if `node` is a string built entirely from source-code literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        # An f-string with no `{...}` substitutions is still fully literal;
        # any ast.FormattedValue part means a runtime value is interpolated.
        return all(isinstance(part, ast.Constant) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_literal_string(node.left) and _is_literal_string(node.right)
    return False


class _GradioHtmlNameCollector(ast.NodeVisitor):
    """Collects, per module, which local names resolve to Gradio's `HTML`."""

    def __init__(self) -> None:
        self.module_aliases: set[str] = set()
        self.html_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in _GRADIO_MODULES:
                self.module_aliases.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in _GRADIO_MODULES:
            for alias in node.names:
                if alias.name == "HTML":
                    self.html_names.add(alias.asname or alias.name)
        self.generic_visit(node)


def _resolves_to_gradio_html(
    func: ast.expr,
    module_aliases: set[str],
    html_names: set[str],
) -> bool:
    if isinstance(func, ast.Attribute) and func.attr == "HTML":
        return isinstance(func.value, ast.Name) and func.value.id in module_aliases
    if isinstance(func, ast.Name):
        return func.id in html_names
    return False


def _value_argument(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "value":
            return keyword.value
    return None


def check_source(source: str, path: Path) -> list[Violation]:
    """Check one module's already-read source text for unsafe `gr.HTML` calls."""
    tree = ast.parse(source, filename=str(path))

    collector = _GradioHtmlNameCollector()
    collector.visit(tree)
    if not collector.module_aliases and not collector.html_names:
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _resolves_to_gradio_html(
            node.func,
            collector.module_aliases,
            collector.html_names,
        ):
            continue
        value = _value_argument(node)
        if value is None or _is_literal_string(value):
            continue
        snippet = ast.dump(value)
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        violations.append(
            Violation(path=path, line=value.lineno, col=value.col_offset, snippet=snippet),
        )
    return violations


def check_file(path: Path) -> list[Violation]:
    return check_source(path.read_text(encoding="utf-8"), path)


def iter_scanned_files() -> list[Path]:
    files: list[Path] = []
    for target in SCAN_TARGETS:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            files.append(target)
    return files


def check_repository() -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_scanned_files():
        violations.extend(check_file(path))
    return violations


def main() -> int:
    scanned = iter_scanned_files()
    violations = check_repository()
    if violations:
        print("gr.HTML safety check FAILED:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            "\ngr.HTML performs no sanitization (confirmed against Gradio's own "
            "docs). Only a hardcoded, developer-authored literal string may be "
            "passed to it. Render dataset/user/model-sourced content through "
            "gr.Markdown(sanitize_html=True) or plain text instead -- see "
            "docs/modernization-plan.md's 'Unsafe rendering' section.",
            file=sys.stderr,
        )
        return 1
    print(f"gr.HTML safety check passed ({len(scanned)} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
