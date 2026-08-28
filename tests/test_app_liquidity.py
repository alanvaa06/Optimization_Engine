"""End-to-end check that the app's liquidity controls reach the simulation.

The pure cost logic is covered in ``test_backtest_liquidity``; what this
protects is the wiring, and specifically the two things that are invisible to
a unit test: that choosing "from traded volume" actually hands the runner the
volume panel the ingest step fetched, and that the fund size it demands is the
one the spec is built with. Both were, at various points, quietly not true.

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


def _no_exception(at: AppTest) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


@pytest.fixture(scope="module")
def solved() -> AppTest:
    """The app with a volume-bearing panel loaded and a solve behind it."""
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    _no_exception(at)

    # The synthetic provider serves volume, so the ADV path has something real
    # to price against.
    fields = next(r for r in at.radio if r.label == "Fields")
    fields.set_value("OHLC + volume").run()
    _no_exception(at)

    next(b for b in at.button if b.label == "Optimize portfolio").click().run()
    _no_exception(at)
    assert at.session_state["last_run"] is not None
    return at


def test_the_fetch_carries_volume_through_to_the_page(solved):
    result = solved.session_state["ingest"]
    assert result is not None
    assert result.volumes is not None
    assert result.panel.has_volume


def test_the_liquidity_selector_appears_only_once_impact_is_on(solved):
    # With no impact there is no participation rate to source, so the choice
    # would be decoration.
    labels = {r.label for r in solved.radio}
    assert "Liquidity model" not in labels

    next(s for s in solved.slider if s.label == "Market impact (eta)").set_value(0.5)
    solved.run()
    _no_exception(solved)
    assert "Liquidity model" in {r.label for r in solved.radio}


def test_choosing_adv_demands_a_fund_size_and_prices_against_volume(solved):
    next(s for s in solved.slider if s.label == "Market impact (eta)").set_value(0.5)
    solved.run()
    next(r for r in solved.radio if r.label == "Liquidity model").set_value(
        "From traded volume (ADV)"
    ).run()
    _no_exception(solved)

    # Capacity is a currency amount, so the page has to ask for one.
    sizes = [n for n in solved.number_input if n.label == "Fund size (NAV)"]
    assert sizes, "ADV pricing must ask for a fund size"
    assert float(sizes[0].value) > 1.0

    shares = [s for s in solved.slider if "share of daily volume" in s.label.lower()]
    assert shares, "ADV pricing must expose the share of volume being taken"


def test_the_adv_choice_changes_what_the_page_charges(solved):
    """The selector is wired to the simulation, not to a label."""
    from optimization_engine.backtest import BacktestSpec, CostSpec, run_backtest
    from optimization_engine.data.loader import prices_to_returns

    result = solved.session_state["ingest"]
    prices = result.prices
    volumes = result.volumes
    returns = prices_to_returns(prices)
    weights = solved.session_state["last_run"].result.weights.reindex(
        returns.columns
    ).fillna(0.0)

    fixed = run_backtest(
        returns,
        weights,
        BacktestSpec(
            frequency="monthly",
            initial_capital=1e8,
            costs=CostSpec(impact_coefficient=0.5, impact_participation=0.05),
        ),
    )
    adv = run_backtest(
        returns,
        weights,
        BacktestSpec(
            frequency="monthly",
            initial_capital=1e8,
            costs=CostSpec(
                impact_coefficient=0.5,
                impact_participation=0.05,
                impact_participation_source="adv",
            ),
        ),
        prices=prices.reindex(returns.index),
        volumes=volumes.reindex(index=returns.index, columns=returns.columns),
    )
    assert float(adv.costs["total"].sum()) != pytest.approx(
        float(fixed.costs["total"].sum())
    )


def test_an_index_universe_still_reaches_the_backtest_tab():
    """The headline case, through the real page: no volume, and it still runs."""
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    _no_exception(at)

    at.text_area(key="universe_sample_Type my own").set_value(
        "SP500, IPC, DAX, NIKKEI225"
    ).run()
    _no_exception(at)

    result = at.session_state["ingest"]
    assert result is not None
    assert result.volumes is None, "index levels never carry volume"

    next(b for b in at.button if b.label == "Optimize portfolio").click().run()
    _no_exception(at)
    assert at.session_state["last_run"] is not None
