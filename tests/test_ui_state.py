from __future__ import annotations

import pandas as pd

from optimization_engine.ui_state import yahoo_cache_key, yahoo_prices_for_rerun


def test_yahoo_prices_survive_rerun_after_fetch_button_resets():
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    state: dict[str, object] = {}
    key = yahoo_cache_key(("SPY",), "5y", None, None, "1d")

    fetched = yahoo_prices_for_rerun(
        fetch_clicked=True,
        cache_key=key,
        state=state,
        fetch_prices=lambda: prices,
    )

    rerun = yahoo_prices_for_rerun(
        fetch_clicked=False,
        cache_key=key,
        state=state,
        fetch_prices=lambda: (_ for _ in ()).throw(AssertionError("should not refetch")),
    )

    pd.testing.assert_frame_equal(fetched, prices)
    pd.testing.assert_frame_equal(rerun, prices)


# ---------------------------------------------------------------------------
# The layered-constraint builder's logic, tested without a browser
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from optimization_engine.constraints import BASIS_PARENT  # noqa: E402
from optimization_engine.ui_state import (  # noqa: E402
    LAYER_PRESETS,
    UNASSIGNED,
    assignment_from_source,
    layer_headroom,
    layer_state_from_layer,
    layer_state_to_layer,
    layer_states_to_layers,
    new_layer_state,
    policy_table,
    preset_by_label,
    sync_layer_state,
    unique_layer_name,
)

ASSETS = ["SPY", "EFA", "AGG", "GLD"]
CURRENCIES = {"SPY": "USD", "EFA": "EUR", "AGG": "USD", "GLD": "EUR"}
GROUPS = {"SPY": "Equity", "EFA": "Equity", "AGG": "Fixed Income", "GLD": "Commodities"}


def test_every_preset_produces_a_usable_starting_layer():
    for preset in LAYER_PRESETS:
        state = new_layer_state(
            preset, ASSETS, currencies=CURRENCIES, base_currency="USD", groups=GROUPS
        )
        assert state["name"]
        assert state["buckets"], preset.label
        assert set(state["assignments"]) == set(ASSETS)


def test_the_fx_preset_assigns_itself_from_the_currency_column():
    state = new_layer_state(
        preset_by_label("Currency — local vs foreign"),
        ASSETS, currencies=CURRENCIES, base_currency="USD",
    )
    layer = layer_state_to_layer(state)
    assert layer.assignments == {
        "SPY": "Local FX", "EFA": "Foreign FX",
        "AGG": "Local FX", "GLD": "Foreign FX",
    }
    assert layer.limits["Foreign FX"] == (0.0, 0.30)


def test_the_local_bucket_follows_the_base_currency():
    assigned = assignment_from_source(
        "currency_local_foreign", ASSETS, CURRENCIES, base_currency="EUR"
    )
    assert assigned["EFA"] == "Local FX"
    assert assigned["SPY"] == "Foreign FX"


def test_copying_the_group_column_reproduces_it():
    assigned = assignment_from_source("group", ASSETS, groups=GROUPS)
    assert assigned == GROUPS


def test_an_asset_the_source_says_nothing_about_lands_unassigned():
    assigned = assignment_from_source("group", ASSETS + ["NEW"], groups=GROUPS)
    assert assigned["NEW"] == UNASSIGNED


def test_a_layer_that_constrains_nothing_is_dropped():
    """An all-0–100% layer is noise in the compliance panel, not a constraint."""
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    assert layer_state_to_layer(state) is None
    state["assignments"] = {a: "Bucket A" for a in ASSETS}
    assert layer_state_to_layer(state) is None
    state["limits"]["Bucket A"] = (0.0, 0.5)
    assert layer_state_to_layer(state) is not None


def test_only_buckets_something_is_assigned_to_are_carried_into_the_layer():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"] = {a: "Bucket A" for a in ASSETS}
    state["limits"] = {"Bucket A": (0.0, 0.5), "Bucket B": (0.0, 0.1)}
    layer = layer_state_to_layer(state)
    assert set(layer.limits) == {"Bucket A"}


def test_the_universe_changing_drops_departed_assets_and_adds_new_ones():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"]["SPY"] = "Bucket A"
    synced = sync_layer_state(state, ["SPY", "TLT"])
    assert set(synced["assignments"]) == {"SPY", "TLT"}
    assert synced["assignments"]["SPY"] == "Bucket A"
    assert synced["assignments"]["TLT"] == UNASSIGNED


def test_an_assignment_to_a_deleted_bucket_falls_back_to_unassigned():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"]["SPY"] = "Bucket B"
    state["buckets"] = ["Bucket A"]
    synced = sync_layer_state(state, ASSETS)
    assert synced["assignments"]["SPY"] == UNASSIGNED
    assert set(synced["limits"]) == {"Bucket A"}


def test_layer_names_stay_unique_so_a_parent_reference_is_unambiguous():
    assert unique_layer_name("FX", ["FX"]) == "FX 2"
    assert unique_layer_name("FX", ["FX", "FX 2"]) == "FX 3"
    assert unique_layer_name("FX", []) == "FX"


def test_headroom_catches_caps_that_cannot_fund_a_full_book():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"] = {"SPY": "Bucket A", "EFA": "Bucket A",
                            "AGG": "Bucket B", "GLD": "Bucket B"}
    state["limits"] = {"Bucket A": (0.0, 0.3), "Bucket B": (0.0, 0.3)}
    health = layer_headroom(state, ASSETS)
    assert health["covers_all"]
    assert health["cap_total"] == pytest.approx(0.6)
    assert health["floor_total"] == pytest.approx(0.0)


def test_headroom_ignores_buckets_nothing_is_assigned_to():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"] = {a: UNASSIGNED for a in ASSETS}
    state["assignments"]["SPY"] = "Bucket A"
    state["limits"] = {"Bucket A": (0.0, 0.3), "Bucket B": (0.0, 0.9)}
    health = layer_headroom(state, ASSETS)
    assert not health["covers_all"]
    assert health["cap_total"] == pytest.approx(0.3)


def test_a_saved_layer_round_trips_back_into_the_editor():
    state = new_layer_state(
        preset_by_label("Sub-asset class (DM / EM)"), ASSETS, groups=GROUPS
    )
    state["assignments"] = {"SPY": "DM Equity", "EFA": "EM Equity",
                            "AGG": UNASSIGNED, "GLD": UNASSIGNED}
    state["basis"] = BASIS_PARENT
    state["parent"] = "Asset class"
    layer = layer_state_to_layer(state)
    back = layer_state_to_layer(sync_layer_state(layer_state_from_layer(layer), ASSETS))
    assert back.assignments == layer.assignments
    assert back.limits == layer.limits
    assert back.basis == layer.basis
    assert back.parent == layer.parent


def test_the_policy_table_names_what_each_limit_is_a_share_of():
    state = new_layer_state(preset_by_label("Blank layer"), ASSETS)
    state["assignments"] = {a: "Bucket A" for a in ASSETS}
    state["limits"] = {"Bucket A": (0.0, 0.5), "Bucket B": (0.0, 1.0)}
    state["basis"] = BASIS_PARENT
    state["parent"] = "Asset class"
    table = policy_table(layer_states_to_layers([state]), ASSETS)
    row = table.set_index("Bucket").loc["Bucket A"]
    assert row["Of"] == "Asset class"
    assert row["Assets"] == len(ASSETS)
    assert row["Max"] == pytest.approx(0.5)
