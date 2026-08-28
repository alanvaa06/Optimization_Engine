"""End-to-end check that the walk-forward controls actually reach the run.

``test_app_wiring`` proves the arguments are passed; this proves the values
are the ones the user set. Both matter, and they fail differently: a page can
pass ``rebalance_frequency=wf_frequency`` while reading a stale variable, and
a page can read the right variable while never passing it.

Skipped when Streamlit is not installed (it is an optional ``[ui]`` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"

CADENCE_LABEL = "Rebalance between re-solves"
REESTIMATE_LABEL = "Re-estimate expected returns per window"


def _no_exception(at: AppTest) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


def _widget(at: AppTest, kind: str, label: str):
    matches = [w for w in getattr(at, kind) if w.label == label]
    assert len(matches) == 1, f"expected one {kind} labelled {label!r}, got {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def optimized() -> AppTest:
    """The app with a solved run, so the Backtest tab has something to replay."""
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    _no_exception(at)
    next(b for b in at.button if b.label == "Optimize portfolio").click().run()
    _no_exception(at)
    assert at.session_state["last_run"] is not None
    # Few enough re-solves to keep the page responsive; the cadence, not the
    # window length, is what these tests are about.
    _widget(at, "number_input", "Re-solve every (periods)").set_value(252).run()
    _no_exception(at)
    return at


def test_the_rebalancing_cadence_control_reaches_the_run(optimized):
    at = optimized
    _widget(at, "selectbox", CADENCE_LABEL).set_value("none").run()
    next(b for b in at.button if b.label == "Run walk-forward").click().run()
    _no_exception(at)
    drifting = at.session_state["walk_forward"]
    assert drifting is not None, "the walk-forward produced nothing"
    assert drifting.backtest.metadata["rebalance_frequency"] == "none"
    assert drifting.n_trade_dates == drifting.n_resolves

    _widget(at, "selectbox", CADENCE_LABEL).set_value("monthly").run()
    next(b for b in at.button if b.label == "Run walk-forward").click().run()
    _no_exception(at)
    disciplined = at.session_state["walk_forward"]
    assert disciplined.backtest.metadata["rebalance_frequency"] == "monthly"

    # Same optimizer, same windows: only the trading calendar moved.
    assert disciplined.n_resolves == drifting.n_resolves
    assert disciplined.n_trade_dates > drifting.n_trade_dates


def test_the_expected_returns_checkbox_is_not_decorative(optimized):
    """Unticking it used to change nothing at all."""
    at = optimized
    _widget(at, "checkbox", REESTIMATE_LABEL).set_value(True).run()
    next(b for b in at.button if b.label == "Run walk-forward").click().run()
    _no_exception(at)
    assert at.session_state["walk_forward"].backtest.metadata[
        "reestimated_expected_returns"
    ] is True

    _widget(at, "checkbox", REESTIMATE_LABEL).set_value(False).run()
    next(b for b in at.button if b.label == "Run walk-forward").click().run()
    _no_exception(at)
    assert at.session_state["walk_forward"].backtest.metadata[
        "reestimated_expected_returns"
    ] is False
    # And the page says so, which it could never do while the box was dead.
    assert any(
        "held fixed across every window" in w.value for w in at.warning
    ), "the leak warning did not appear"
