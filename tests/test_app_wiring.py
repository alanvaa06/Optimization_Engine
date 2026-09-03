"""Static guards against controls the app draws but never reads.

A Streamlit widget that is assigned to a variable and then never used is the
quietest possible bug: the control renders, the user sets it, and nothing
happens. It cannot fail a unit test of the library, it does not raise, and
the page looks correct in a screenshot. The walk-forward section shipped with
exactly that — a "re-estimate expected returns per window" checkbox that was
never passed to the call it was supposed to govern — and the only reason it
was harmless is that the default happened to be the safe direction.

These read the page as a syntax tree rather than running it, so they need no
Streamlit install and no browser, and they check the two things that break
silently when the page script is edited: that every control is read, and that
the calls whose behaviour a control governs — the walk-forward, the stress
test, the universe-aware runs — carry the arguments those controls exist to
set.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"

#: Streamlit calls that return a value the page is supposed to act on.
INPUT_WIDGETS = frozenset(
    {
        "checkbox",
        "color_picker",
        "date_input",
        "file_uploader",
        "multiselect",
        "number_input",
        "radio",
        "select_slider",
        "selectbox",
        "slider",
        "text_area",
        "text_input",
        "toggle",
    }
)


@pytest.fixture(scope="module")
def page() -> ast.Module:
    return ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))


def _widget_assignments(tree: ast.Module) -> dict[str, ast.Call]:
    """``name -> call`` for every ``name = st.<widget>(...)`` on the page."""
    found: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if isinstance(value.func, ast.Attribute) and value.func.attr in INPUT_WIDGETS:
            found[target.id] = value
    return found


def _calls_to(tree: ast.Module, method: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ]


def _calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    """Every ``name(...)`` on the page — a bare function, not a method."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_every_control_the_page_draws_is_read_somewhere(page):
    widgets = _widget_assignments(page)
    assert widgets, "found no widgets at all — the parser has drifted from the page"

    loaded = {
        node.id
        for node in ast.walk(page)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    dead = sorted(
        name
        for name, call in widgets.items()
        if name not in loaded
        # A widget with a ``key`` is legitimately read back out of session
        # state rather than through its return value.
        and not any(kw.arg == "key" for kw in call.keywords)
    )
    assert not dead, (
        "these controls are drawn, assigned, and never read — setting them "
        f"does nothing: {dead}"
    )


def test_the_walk_forward_call_carries_what_its_controls_set(page):
    """The two arguments the section's controls exist to supply.

    ``reestimate_expected_returns`` decides whether the run is genuinely out
    of sample; ``rebalance_frequency`` decides how often the book trades
    between re-solves. Both have page controls, and both must reach the call.
    """
    calls = [
        call
        for call in _calls_to(page, "walk_forward")
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "run"
    ]
    assert len(calls) == 1, "expected exactly one run.walk_forward(...) on the page"

    passed = {kw.arg for kw in calls[0].keywords}
    for required in ("reestimate_expected_returns", "rebalance_frequency"):
        assert required in passed, (
            f"run.walk_forward() does not pass {required!r}, so the control "
            "that sets it is decorative"
        )


def test_the_stress_call_carries_the_policy_its_control_sets(page):
    """The Stress tab's one real decision must reach the library.

    ``unknown_assets`` is what a shock naming an asset the book cannot hold
    does: refuse, or apply anyway and record the drop. The page offers that as
    a choice, and a choice that does not reach ``stress_test`` is a control
    that renders and does nothing — the same defect the walk-forward checkbox
    above had.
    """
    calls = _calls_named(page, "stress_test")
    assert len(calls) == 1, "expected exactly one stress_test(...) on the page"
    passed = {kw.arg for kw in calls[0].keywords}
    for required in ("cov_matrix", "unknown_assets"):
        assert required in passed, (
            f"stress_test() does not pass {required!r}, so the control that "
            "sets it is decorative"
        )


def test_the_universe_replay_carries_the_universe_and_the_policy(page):
    """A universe without its policy is a run that cannot say what it decided.

    ``run_backtest`` refuses a ``universe`` with no ``universe_policy``, so a
    page that forgot one would fail loudly — but a page that forgot the
    *universe* would run happily on the panel's columns and report a
    point-in-time universe it never applied.
    """
    calls = _calls_named(page, "run_backtest")
    assert len(calls) == 1, "expected exactly one run_backtest(...) on the page"
    passed = {kw.arg for kw in calls[0].keywords}
    for required in ("universe", "universe_policy"):
        assert required in passed, f"run_backtest() does not pass {required!r}"


def test_the_universe_walk_forward_carries_the_delisting_grace(page):
    """Delisting is a separate opt-in, with its own control on the page."""
    calls = [
        call
        for call in _calls_to(page, "walk_forward_run")
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "run"
    ]
    assert len(calls) == 1, "expected exactly one run.walk_forward_run(...)"
    passed = {kw.arg for kw in calls[0].keywords}
    for required in ("universe", "universe_policy", "delisting_grace"):
        assert required in passed, (
            f"run.walk_forward_run() does not pass {required!r}, so the "
            "control that sets it is decorative"
        )
