"""Tests for `scripts.check_gr_html_safety` (Stage 3's durable gr.HTML gate).

Two layers, per `docs/modernization-plan.md` Stage 3's Gate bullet: "a
durable, CI-enforced check ... confirms gr.HTML is never bound to dataset-,
user-, or model-sourced content anywhere in the codebase". A checker that
never actually flags anything would satisfy a "check exists" box while
providing zero protection, so most of this file proves the checker fires on
realistic unsafe patterns and stays silent on safe ones -- not just that it
runs clean against today's snapshot of the repo.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_gr_html_safety import Violation, check_repository, check_source


def test_real_repository_has_no_gr_html_violations() -> None:
    violations = check_repository()
    assert violations == [], "\n".join(str(v) for v in violations)


def _violations(source: str, filename: str = "module.py") -> list[Violation]:
    return check_source(source, Path(filename))


def test_literal_string_value_is_safe() -> None:
    source = 'import gradio as gr\ngr.HTML("<b>Hello</b>")\n'
    assert _violations(source) == []


def test_no_value_argument_is_safe() -> None:
    source = 'import gradio as gr\ngr.HTML(label="preview")\n'
    assert _violations(source) == []


def test_literal_concatenation_is_safe() -> None:
    source = 'import gradio as gr\ngr.HTML("<b>" + "hi</b>")\n'
    assert _violations(source) == []


def test_fstring_with_no_substitution_is_safe() -> None:
    source = 'import gradio as gr\ngr.HTML(f"<b>literal only</b>")\n'
    assert _violations(source) == []


def test_variable_value_is_flagged() -> None:
    source = "import gradio as gr\ndescription = get_description()\ngr.HTML(description)\n"
    violations = _violations(source)
    assert len(violations) == 1
    assert violations[0].line == 3


def test_fstring_with_substitution_is_flagged() -> None:
    source = 'import gradio as gr\nname = get_name()\ngr.HTML(f"<b>{name}</b>")\n'
    violations = _violations(source)
    assert len(violations) == 1


def test_function_call_return_value_is_flagged() -> None:
    source = "import gradio as gr\ngr.HTML(render_untrusted_description())\n"
    violations = _violations(source)
    assert len(violations) == 1


def test_attribute_access_value_is_flagged() -> None:
    source = "import gradio as gr\ngr.HTML(record.description)\n"
    violations = _violations(source)
    assert len(violations) == 1


def test_keyword_value_argument_is_checked() -> None:
    source = "import gradio as gr\ngr.HTML(value=record.description)\n"
    violations = _violations(source)
    assert len(violations) == 1


def test_aliased_gradio_import_is_still_checked() -> None:
    source = "import gradio as gradio_ui\ngradio_ui.HTML(untrusted_html)\n"
    violations = _violations(source)
    assert len(violations) == 1


def test_named_html_import_literal_is_safe() -> None:
    source = 'from gradio import HTML\nHTML("<b>literal</b>")\n'
    assert _violations(source) == []


def test_named_html_import_variable_is_flagged() -> None:
    source = "from gradio import HTML\nHTML(untrusted_html)\n"
    violations = _violations(source)
    assert len(violations) == 1


def test_unrelated_html_symbol_is_not_flagged() -> None:
    # A same-named `HTML` that has nothing to do with gradio must not trip
    # the check -- the resolution is import-scoped, not name-scoped.
    source = "from some_other_module import HTML\nHTML(untrusted_html)\n"
    assert _violations(source) == []


def test_gradio_module_without_html_call_is_not_flagged() -> None:
    source = 'import gradio as gr\ngr.Markdown("hi", sanitize_html=True)\n'
    assert _violations(source) == []


def test_violation_str_includes_location() -> None:
    source = "import gradio as gr\ngr.HTML(untrusted_html)\n"
    violations = _violations(source, filename="app.py")
    assert len(violations) == 1
    rendered = str(violations[0])
    assert "app.py:2:" in rendered
    assert "gr.HTML" in rendered
