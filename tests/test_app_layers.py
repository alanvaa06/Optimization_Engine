"""End-to-end check that the layered-constraint builder is actually wired up.

The pure logic is covered in ``test_ui_state``; what this file protects is the
wiring — that clicking "Add layer" in the real app produces a layer the solve
honours, and that a saved scenario brings its policy back. Those are the two
things that break silently when the page script is refactored, and neither is
visible to a unit test.

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

from optimization_engine.ui_state import layer_states_to_layers  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from layer_editor import _current_bucket_weights  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"

ASSET_CLASS = {
    "US_Equity": "Equity",
    "Intl_Equity": "Equity",
    "EM_Equity": "Equity",
    "Real_Estate": "Alternatives",
    "Infra": "Alternatives",
    "Commodities": "Commodities",
    "Gold": "Commodities",
    "US_Treasuries": "Fixed Income",
    "TIPS": "Fixed Income",
    "IG_Credit": "Fixed Income",
    "HY_Credit": "Fixed Income",
    "EM_Debt": "Fixed Income",
    "Cash": "Cash",
}
SUB_CLASS = {
    "US_Equity": "DM Equity",
    "Intl_Equity": "DM Equity",
    "EM_Equity": "EM Equity",
    "US_Treasuries": "DM Fixed Income",
    "TIPS": "DM Fixed Income",
    "IG_Credit": "DM Fixed Income",
    "HY_Credit": "DM Fixed Income",
    "EM_Debt": "EM Fixed Income",
}
CLASS_CAPS = {
    "Equity": 0.60,
    "Fixed Income": 0.30,
    "Commodities": 0.10,
    "Alternatives": 0.15,
    "Cash": 0.10,
}


def _no_exception(at: AppTest) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


@pytest.fixture(scope="module")
def app() -> AppTest:
    """The app with asset classes named and their budgets set, before layers."""
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    _no_exception(at)

    table = at.session_state["config_table"].copy()
    table["Group"] = [ASSET_CLASS[a] for a in table.index]
    at.session_state["config_table"] = table
    at.run()

    bounds = at.session_state["group_bounds"].copy()
    for group, cap in CLASS_CAPS.items():
        bounds.loc[group, "Max Weight"] = cap
    at.session_state["group_bounds"] = bounds
    at.run()
    _no_exception(at)
    return at


def test_adding_a_layer_from_the_menu_produces_one_the_solve_honours(app):
    app.selectbox(key="layer_preset_choice").set_value(
        "Sub-asset class (DM / EM)"
    ).run()
    app.button(key="layer_add_btn").click().run()
    _no_exception(app)

    states = app.session_state["constraint_layer_states"]
    assert len(states) == 1
    states[0]["assignments"] = {
        a: SUB_CLASS.get(a, "—") for a in app.session_state["config_table"].index
    }
    app.run()
    _no_exception(app)

    next(b for b in app.button if b.label == "Optimize portfolio").click().run()
    _no_exception(app)

    run = app.session_state["last_run"]
    assert run is not None, "the solve produced nothing"
    assert run.result.is_compliant, run.result.violations
    assert [lyr.name for lyr in run.constraint_layers] == [
        "Asset class",
        "Sub-asset class",
    ]

    weights = run.result.weights
    equity = sum(w for a, w in weights.items() if ASSET_CLASS[a] == "Equity")
    em = sum(w for a, w in weights.items() if SUB_CLASS.get(a) == "EM Equity")
    assert equity <= 0.60 + 1e-6
    assert em <= 0.20 + 1e-6

    exposures = run.layer_exposures()
    assert set(exposures["layer"]) == {"Asset class", "Sub-asset class"}
    assert exposures["binding"].any()


def test_a_saved_scenario_brings_its_whole_policy_back(app):
    """Including the basis and parent — a mandate must reopen as it was saved."""
    states = app.session_state["constraint_layer_states"]
    assert states, "depends on the layer added by the previous test"
    uid = states[0]["uid"]
    app.radio(key=f"layer_basis_{uid}").set_value("parent").run()
    _no_exception(app)
    states = app.session_state["constraint_layer_states"]
    states[0]["limits"] = {
        "DM Equity": (0.0, 1.0),
        "EM Equity": (0.0, 0.30),
        "DM Fixed Income": (0.0, 1.0),
        "EM Fixed Income": (0.0, 0.25),
    }
    app.run()
    saved = [lyr.to_dict() for lyr in layer_states_to_layers(states)]
    assert saved[0]["basis"] == "parent"
    assert saved[0]["parent"] == "Asset class"

    app.text_input(key="scn_new_name").set_value("Layered mandate").run()
    app.button(key="scn_save").click().run()
    _no_exception(app)
    assert "Layered mandate" in app.session_state["scenarios"]

    app.session_state["constraint_layer_states"] = []
    app.run()
    app.selectbox(key="scn_select").set_value("Layered mandate").run()
    app.button(key="scn_load").click().run()
    app.run()
    _no_exception(app)

    restored = app.session_state["constraint_layer_states"]
    assert [
        lyr.to_dict() for lyr in layer_states_to_layers(restored)
    ] == saved


def test_removing_a_layer_removes_it_from_the_solve(app):
    app.selectbox(key="layer_preset_choice").set_value("Region").run()
    app.button(key="layer_add_btn").click().run()
    _no_exception(app)
    states = app.session_state["constraint_layer_states"]
    assert [s["name"] for s in states][-1] == "Region"

    app.button(key=f"layer_del_{states[-1]['uid']}").click().run()
    _no_exception(app)
    assert "Region" not in [
        s["name"] for s in app.session_state["constraint_layer_states"]
    ]


def test_the_live_column_states_the_book_in_the_units_of_the_limit_beside_it(app):
    """A number in the wrong units next to a limit invites the very mistake
    the column exists to prevent, so both bases are checked explicitly."""
    run = app.session_state["last_run"]
    assert run is not None, "depends on the solve in the first test"
    weights = run.result.weights
    state = dict(app.session_state["constraint_layer_states"][0])
    state["assignments"] = {
        a: SUB_CLASS.get(a, "—") for a in app.session_state["config_table"].index
    }

    equity = sum(w for a, w in weights.items() if ASSET_CLASS[a] == "Equity")
    em = sum(w for a, w in weights.items() if SUB_CLASS.get(a) == "EM Equity")
    assert equity > 0.01 and em > 0.001

    state["basis"] = "portfolio"
    absolute = _current_bucket_weights(state, weights)
    assert absolute["EM Equity"] == pytest.approx(100 * em)

    state["basis"] = "parent"
    state["parent"] = "Asset class"
    relative = _current_bucket_weights(state, weights, parent_assignments=ASSET_CLASS)
    assert relative["EM Equity"] == pytest.approx(100 * em / equity)


def test_the_live_column_is_absent_when_there_is_nothing_to_compare_against(app):
    state = dict(app.session_state["constraint_layer_states"][0])
    assert _current_bucket_weights(state, None) is None
    state["basis"] = "parent"
    # Relative units cannot be formed without the parent's assignments.
    assert _current_bucket_weights(
        state, app.session_state["last_run"].result.weights
    ) is None
