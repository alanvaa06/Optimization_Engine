"""Tests for the volume-optional liquidity model.

The property under test throughout: a backtest must run, and run identically
to the fixed-participation case, when there is no volume to price capacity
from — because an index has none by construction. When volume *is* supplied,
capacity must actually be measured from it, and the two must be
distinguishable in the run log rather than silently equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.backtest.costs import (  # noqa: E402
    MarketContext,
    SquareRootImpactCost,
    build_cost_model,
    trailing_dollar_volume,
)
from optimization_engine.backtest.runner import run_backtest  # noqa: E402
from optimization_engine.backtest.spec import (  # noqa: E402
    BacktestSpec,
    CostSpec,
    SpecValidationError,
)


def _returns(n: int = 400, assets=("AAA", "BBB")) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, size=(n, len(assets))),
        index=index,
        columns=list(assets),
    )


def _prices(returns: pd.DataFrame) -> pd.DataFrame:
    return 100.0 * (1.0 + returns).cumprod()


def _volumes(returns: pd.DataFrame, level: float = 2e6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.lognormal(np.log(level), 0.2, size=returns.shape),
        index=returns.index,
        columns=returns.columns,
    )


#: A plausible institutional book. ADV pricing is meaningless without one:
#: capacity is a currency amount, so the fund's size is what decides whether a
#: given daily volume is deep or thin.
CAPITAL = 1e8


def _spec(source: str = "fixed", capital: float = CAPITAL, **costs) -> BacktestSpec:
    return BacktestSpec(
        frequency="monthly",
        initial_capital=capital,
        costs=CostSpec(
            commission_bps=5.0,
            impact_coefficient=0.5,
            impact_participation=0.05,
            impact_participation_source=source,
            **costs,
        ),
    )


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_the_default_cost_model_reads_no_volume():
    assert CostSpec().uses_volume is False
    assert CostSpec(impact_coefficient=0.5).uses_volume is False


def test_uses_volume_is_true_only_for_adv_with_impact_enabled():
    assert CostSpec(
        impact_coefficient=0.5, impact_participation_source="adv"
    ).uses_volume
    # Impact off means participation is never consulted, whatever it says.
    assert not CostSpec(impact_participation_source="adv").uses_volume


def test_an_unknown_participation_source_is_rejected():
    with pytest.raises(SpecValidationError, match="impact_participation_source"):
        CostSpec(impact_participation_source="magic")


def test_adv_share_must_be_a_share():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(SpecValidationError, match="impact_adv_share"):
            CostSpec(impact_adv_share=bad)


def test_participation_source_round_trips_through_dict():
    spec = CostSpec(
        impact_coefficient=0.4, impact_participation_source="adv",
        impact_adv_share=0.25, impact_adv_lookback=42,
    )
    rebuilt = CostSpec.from_dict(spec.to_dict())
    assert rebuilt == spec


# ---------------------------------------------------------------------------
# The cost model in isolation
# ---------------------------------------------------------------------------


def test_fixed_source_ignores_any_observed_participation():
    model = SquareRootImpactCost(eta=1.0, participation=0.05)
    with_adv = model.charge(
        asset="AAA", traded_weight=0.1,
        context=MarketContext(volatility=0.01, participation=0.5),
    )
    without = model.charge(
        asset="AAA", traded_weight=0.1, context=MarketContext(volatility=0.01)
    )
    assert with_adv.slippage == pytest.approx(without.slippage)
    assert with_adv.degraded_reason is None


def test_adv_source_uses_the_observed_participation():
    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv"
    )
    deep = model.charge(
        asset="AAA", traded_weight=0.1,
        context=MarketContext(volatility=0.01, participation=0.50),
    )
    thin = model.charge(
        asset="AAA", traded_weight=0.1,
        context=MarketContext(volatility=0.01, participation=0.01),
    )
    # A thinner market must cost more for the same trade.
    assert thin.slippage > deep.slippage
    assert deep.degraded_reason is None


def test_adv_source_falls_back_to_the_fixed_rate_and_records_it():
    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv"
    )
    fallback = model.charge(
        asset="SP500", traded_weight=0.1, context=MarketContext(volatility=0.01)
    )
    fixed = SquareRootImpactCost(eta=1.0, participation=0.05).charge(
        asset="SP500", traded_weight=0.1, context=MarketContext(volatility=0.01)
    )
    # The trade is still priced — identically to the fixed model — and the
    # fallback is on the record rather than invisible.
    assert fallback.slippage == pytest.approx(fixed.slippage)
    assert "fell back to the fixed participation rate" in fallback.degraded_reason
    assert "SP500" in fallback.degraded_reason


def test_a_missing_volatility_still_degrades_impact_to_zero():
    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv"
    )
    quote = model.charge(
        asset="AAA", traded_weight=0.1,
        context=MarketContext(volatility=None, participation=0.5),
    )
    assert quote.slippage == pytest.approx(0.0)
    assert "insufficient history" in quote.degraded_reason


def test_a_non_positive_observed_participation_is_treated_as_absent():
    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv"
    )
    quote = model.charge(
        asset="AAA", traded_weight=0.1,
        context=MarketContext(volatility=0.01, participation=0.0),
    )
    assert "fell back" in quote.degraded_reason
    assert quote.slippage > 0.0


def test_only_the_adv_model_asks_for_a_volume_lookback():
    assert build_cost_model(CostSpec(impact_coefficient=0.5)).participation_lookback() == 0
    adv = build_cost_model(
        CostSpec(
            impact_coefficient=0.5, impact_participation_source="adv",
            impact_adv_lookback=30,
        )
    )
    assert adv.participation_lookback() == 30


# ---------------------------------------------------------------------------
# Trailing dollar volume
# ---------------------------------------------------------------------------


def test_trailing_dollar_volume_uses_only_the_past():
    returns = _returns(60)
    prices = _prices(returns)
    volumes = _volumes(returns)
    adv = trailing_dollar_volume(prices, volumes, lookback=5, min_observations=5)

    # The window ending at t excludes t: a trade decided on t cannot be sized
    # off that day's own turnover.
    expected = float((prices * volumes).iloc[5:10, 0].mean())
    assert float(adv.iloc[10, 0]) == pytest.approx(expected)
    assert adv.iloc[:5, 0].isna().all()


def test_trailing_dollar_volume_treats_a_missing_column_as_absent():
    returns = _returns(40)
    prices = _prices(returns)
    volumes = _volumes(returns).drop(columns=["BBB"])
    adv = trailing_dollar_volume(prices, volumes, lookback=5, min_observations=5)
    assert adv["BBB"].dropna().empty
    assert adv["AAA"].dropna().size > 0


def test_a_zero_volume_column_reads_as_no_adv_not_as_zero_capacity():
    returns = _returns(40)
    prices = _prices(returns)
    volumes = _volumes(returns)
    volumes["BBB"] = 0.0
    adv = trailing_dollar_volume(prices, volumes, lookback=5, min_observations=5)
    # Zero would mean "cannot trade at any price"; absent means "price it some
    # other way", which is the only defensible reading of a padded column.
    assert adv["BBB"].dropna().empty


# ---------------------------------------------------------------------------
# End to end through the runner
# ---------------------------------------------------------------------------


def test_a_backtest_with_no_volume_runs_and_matches_the_fixed_model():
    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)

    fixed = run_backtest(returns, weights, _spec("fixed"))
    adv_without_volume = run_backtest(returns, weights, _spec("adv"))

    assert float(adv_without_volume.costs["total"].sum()) == pytest.approx(
        float(fixed.costs["total"].sum())
    )
    pd.testing.assert_series_equal(
        fixed.returns, adv_without_volume.returns, check_names=False
    )


def test_the_fallback_is_visible_in_the_run_log():
    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)
    run = run_backtest(returns, weights, _spec("adv"))
    assert any(
        "fell back to the fixed participation rate" in reason
        for reason in run.meta.degradations
    )


def test_supplying_volume_changes_the_cost_the_adv_model_charges():
    returns = _returns()
    prices = _prices(returns)
    volumes = _volumes(returns)
    weights = pd.Series(0.5, index=returns.columns)

    without = run_backtest(returns, weights, _spec("adv"))
    with_volume = run_backtest(
        returns, weights, _spec("adv"), prices=prices, volumes=volumes
    )
    assert float(with_volume.costs["total"].sum()) != pytest.approx(
        float(without.costs["total"].sum())
    )


def test_volume_is_ignored_by_the_fixed_model():
    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)
    plain = run_backtest(returns, weights, _spec("fixed"))
    with_volume = run_backtest(
        returns, weights, _spec("fixed"),
        prices=_prices(returns), volumes=_volumes(returns),
    )
    assert float(with_volume.costs["total"].sum()) == pytest.approx(
        float(plain.costs["total"].sum())
    )


def test_a_thinner_book_costs_more_under_adv():
    returns = _returns()
    prices = _prices(returns)
    weights = pd.Series(0.5, index=returns.columns)

    deep = run_backtest(
        returns, weights, _spec("adv"),
        prices=prices, volumes=_volumes(returns, level=5e7),
    )
    thin = run_backtest(
        returns, weights, _spec("adv"),
        prices=prices, volumes=_volumes(returns, level=5e4),
    )
    assert float(thin.costs["total"].sum()) > float(deep.costs["total"].sum())


def test_a_larger_book_pays_more_for_the_same_names():
    # Capacity is a currency amount, so the same trade is a larger share of a
    # thinner slice of the market as the fund grows.
    returns = _returns()
    prices = _prices(returns)
    volumes = _volumes(returns)
    weights = pd.Series(0.5, index=returns.columns)

    def total_cost(capital: float) -> float:
        spec = BacktestSpec(
            frequency="monthly",
            initial_capital=capital,
            costs=CostSpec(
                commission_bps=0.0, impact_coefficient=0.5,
                impact_participation_source="adv",
            ),
        )
        run = run_backtest(returns, weights, spec, prices=prices, volumes=volumes)
        return float(run.costs["total"].sum())

    assert total_cost(1e9) > total_cost(1e6)


def test_a_mixed_universe_prices_each_asset_from_what_it_has():
    # AAA has volume; the index has none. Both must be priced, by different
    # routes, in the same run.
    returns = _returns(assets=("AAA", "SP500"))
    prices = _prices(returns)
    volumes = _volumes(returns)
    volumes["SP500"] = np.nan
    weights = pd.Series(0.5, index=returns.columns)

    run = run_backtest(
        returns, weights, _spec("adv"), prices=prices, volumes=volumes
    )
    reasons = " ".join(run.meta.degradations)
    assert "SP500" in reasons
    assert "fell back to the fixed participation rate for AAA" not in reasons
    assert float(run.costs["total"].sum()) > 0.0


def test_a_cost_model_without_participation_lookback_still_runs():
    # Custom cost models written before ADV pricing existed must keep working.
    class LegacyModel:
        def volatility_lookback(self) -> int:
            return 0

        def charge(self, *, asset, traded_weight, context):
            from optimization_engine.backtest.costs import CostQuote

            return CostQuote(commission=abs(traded_weight) * 0.001)

    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)
    run = run_backtest(
        returns, weights, _spec("adv"), cost_model=LegacyModel(),
        prices=_prices(returns), volumes=_volumes(returns),
    )
    assert float(run.costs["total"].sum()) > 0.0


# ---------------------------------------------------------------------------
# What the tearsheet says about the cost model it was given
# ---------------------------------------------------------------------------


def _tearsheet_caveats(spec: BacktestSpec) -> str:
    from optimization_engine.backtest.tearsheet import build_tearsheet

    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)
    run = run_backtest(returns, weights, spec)
    return " ".join(build_tearsheet(run, returns).caveats)


def test_a_costless_run_is_still_called_free():
    caveats = _tearsheet_caveats(BacktestSpec(frequency="monthly", costs=CostSpec()))
    assert "modelled as free" in caveats


def test_an_impact_only_run_is_not_called_free():
    # It charged 1.7% of NAV; calling that free contradicts the line above it.
    caveats = _tearsheet_caveats(
        BacktestSpec(
            frequency="monthly",
            costs=CostSpec(impact_coefficient=0.4),
        )
    )
    assert "modelled as free" not in caveats
    assert "Only market impact was charged" in caveats


def test_an_adv_run_states_what_its_impact_was_priced_against():
    caveats = _tearsheet_caveats(
        BacktestSpec(
            frequency="monthly",
            initial_capital=CAPITAL,
            costs=CostSpec(
                commission_bps=5.0, impact_coefficient=0.4,
                impact_participation_source="adv",
            ),
        )
    )
    assert "priced against traded volume" in caveats
    assert "fixed participation rate" in caveats


# ---------------------------------------------------------------------------
# ADV pricing is meaningless without a fund size, and must say so
# ---------------------------------------------------------------------------


def test_adv_pricing_refuses_the_default_one_unit_book():
    # The default initial_capital is 1.0 — a one-currency-unit fund, for which
    # every market on earth is infinitely deep and the impact charge rounds to
    # zero. Completing that run would look like evidence of unlimited
    # capacity.
    with pytest.raises(SpecValidationError, match="real fund size"):
        BacktestSpec(
            costs=CostSpec(
                impact_coefficient=0.5, impact_participation_source="adv"
            )
        )


def test_the_fixed_model_is_unaffected_by_the_fund_size():
    spec = BacktestSpec(costs=CostSpec(impact_coefficient=0.5))
    assert spec.initial_capital == 1.0


def test_adv_pricing_accepts_a_real_book():
    spec = BacktestSpec(
        initial_capital=1e7,
        costs=CostSpec(impact_coefficient=0.5, impact_participation_source="adv"),
    )
    assert spec.costs.uses_volume


def test_the_threshold_is_inclusive_so_a_control_can_offer_it():
    from optimization_engine.backtest.spec import MIN_ADV_CAPITAL

    # The app's fund-size input uses MIN_ADV_CAPITAL as its own minimum, so a
    # user dialling it all the way down must not produce a spec the validator
    # rejects.
    BacktestSpec(
        initial_capital=MIN_ADV_CAPITAL,
        costs=CostSpec(impact_coefficient=0.5, impact_participation_source="adv"),
    )
    with pytest.raises(SpecValidationError):
        BacktestSpec(
            initial_capital=MIN_ADV_CAPITAL * 0.99,
            costs=CostSpec(
                impact_coefficient=0.5, impact_participation_source="adv"
            ),
        )


def test_adv_impact_is_material_at_a_realistic_book_size():
    # The regression this whole guard exists for: at the old default the ADV
    # charge was four orders of magnitude below the fixed one, which read as
    # "this strategy has no capacity problem".
    returns = _returns()
    prices = _prices(returns)
    volumes = _volumes(returns, level=2e5)
    weights = pd.Series(0.5, index=returns.columns)

    fixed = run_backtest(returns, weights, _spec("fixed"))
    adv = run_backtest(
        returns, weights, _spec("adv"), prices=prices, volumes=volumes
    )
    fixed_cost = float(fixed.costs["total"].sum())
    adv_cost = float(adv.costs["total"].sum())
    assert adv_cost > fixed_cost / 100.0


def test_a_cost_model_passed_directly_cannot_dodge_the_fund_size_guard():
    # The spec guard reads spec.costs; a model handed in through cost_model=
    # never touches it. The runner is the one place every route converges on.
    returns = _returns()
    weights = pd.Series(0.5, index=returns.columns)
    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv"
    )
    with pytest.raises(ValueError, match="real fund size"):
        run_backtest(
            returns,
            weights,
            BacktestSpec(frequency="monthly"),  # the default NAV of 1.0
            cost_model=model,
            prices=_prices(returns),
            volumes=_volumes(returns),
        )


def test_a_directly_passed_model_uses_its_own_share_of_volume():
    # An injected model owns its parameters; reading the share off a spec it
    # never came from would silently apply someone else's number.
    returns = _returns()
    prices, volumes = _prices(returns), _volumes(returns)
    weights = pd.Series(0.5, index=returns.columns)
    # A spec whose own share is nothing like either model's, to prove neither
    # run is quietly reading it.
    spec = BacktestSpec(
        frequency="monthly",
        initial_capital=CAPITAL,
        costs=CostSpec(impact_adv_share=0.02),
    )

    def cost(share: float) -> float:
        model = SquareRootImpactCost(
            eta=1.0, participation=0.05, participation_source="adv",
            adv_share=share,
        )
        run = run_backtest(
            returns, weights, spec, cost_model=model, prices=prices, volumes=volumes
        )
        return float(run.costs["total"].sum())

    # Taking a larger share of the day's volume means more capacity, so less
    # impact for the same trade.
    assert cost(0.5) < cost(0.01)


# ---------------------------------------------------------------------------
# Every parameter of the charge belongs to the model, not to the run
# ---------------------------------------------------------------------------


def test_the_spec_hands_every_cost_parameter_to_the_model_it_builds():
    from optimization_engine.backtest import build_cost_model, context_request

    wanted = context_request(
        build_cost_model(
            CostSpec(
                impact_coefficient=0.5,
                impact_participation_source="adv",
                impact_volatility_lookback=40,
                min_impact_observations=9,
                impact_adv_lookback=17,
                min_adv_observations=4,
                impact_adv_share=0.33,
            )
        )
    )
    assert wanted.volatility_lookback == 40
    assert wanted.volatility_min_observations == 9
    assert wanted.participation_lookback == 17
    assert wanted.participation_min_observations == 4
    assert wanted.adv_share == pytest.approx(0.33)


def test_an_injected_models_observation_minimums_are_not_overridden_by_the_spec():
    """The asymmetry this whole arrangement exists to remove.

    The runner used to take the lookbacks from the model and the observation
    minimums from the spec, so a model asking for a two-observation floor got
    the spec's twenty-one — and quietly priced nothing for the first three
    weeks of every run.
    """
    from optimization_engine.backtest import context_request

    model = SquareRootImpactCost(
        eta=1.0, participation=0.05, participation_source="adv",
        lookback=5, min_observations=2, adv_lookback=3, min_adv_observations=2,
    )
    wanted = context_request(model)
    assert wanted.volatility_min_observations == 2
    assert wanted.participation_min_observations == 2


def test_an_injected_models_observation_floor_changes_what_gets_charged():
    # The observable consequence. Two models identical but for the floor: the
    # permissive one can price trades the strict one degrades to zero impact.
    # Before the fix both ran on the spec's floor and were indistinguishable.
    returns = _returns(120)
    prices, volumes = _prices(returns), _volumes(returns)
    weights = pd.Series(0.5, index=returns.columns)
    spec = BacktestSpec(frequency="weekly", initial_capital=CAPITAL, costs=CostSpec())

    def cost(min_observations: int) -> float:
        model = SquareRootImpactCost(
            eta=1.0, participation=0.05, participation_source="adv",
            lookback=20, min_observations=min_observations,
            adv_lookback=5, min_adv_observations=2,
        )
        run = run_backtest(
            returns, weights, spec, cost_model=model, prices=prices, volumes=volumes
        )
        return float(run.costs["total"].sum())

    assert cost(2) > cost(19)


def test_a_cost_model_predating_these_fields_gets_documented_defaults():
    # ``cost_model=`` is public, so a model written against the older protocol
    # — two lookback methods and charge() — must keep working.
    from optimization_engine.backtest import ContextRequest, context_request

    class Legacy:
        def volatility_lookback(self) -> int:
            return 10

        def charge(self, *, asset, traded_weight, context):  # pragma: no cover
            raise NotImplementedError

    wanted = context_request(Legacy())
    assert wanted.volatility_lookback == 10
    assert wanted.participation_lookback == 0
    assert wanted.volatility_min_observations == ContextRequest.volatility_min_observations
    assert wanted.adv_share == ContextRequest.adv_share


def test_a_floor_a_window_can_never_reach_is_refused_by_the_spec():
    with pytest.raises(SpecValidationError, match="min_impact_observations"):
        CostSpec(impact_volatility_lookback=20, min_impact_observations=60)
    with pytest.raises(SpecValidationError, match="min_adv_observations"):
        CostSpec(impact_adv_lookback=5, min_adv_observations=10)


def test_a_floor_a_window_can_never_reach_is_refused_for_an_injected_model():
    # pandas reports this from three frames down as "min_periods 60 must be <=
    # window 20", naming neither the model nor which of its two windows.
    from optimization_engine.backtest import context_request

    model = SquareRootImpactCost(lookback=20, min_observations=60)
    with pytest.raises(ValueError, match="volatility window is 20"):
        context_request(model)

    model = SquareRootImpactCost(
        participation_source="adv", adv_lookback=5, min_adv_observations=10
    )
    with pytest.raises(ValueError, match="traded-volume window is 5"):
        context_request(model)
