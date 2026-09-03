"""One Sharpe definition, the trials that feed it, and gap-free returns.

Task 1.2 of the numerical-rigor plan. Three defects, one file's worth of
regression tests:

* **Three Sharpe ratios.** ``performance.sharpe_ratio`` annualized the excess
  return geometrically, ``rolling_metrics`` did it arithmetically, and
  ``selection._period_sharpe`` did it arithmetically per period. The deflated
  Sharpe therefore deflated an arithmetic number against a distribution of
  geometric ones. There is now one definition with a ``method`` switch, and
  everything else calls it.
* **The trial count.** A sweep cell that failed to build or failed to solve is
  still a configuration you tried, and a Sharpe estimated on a cell's own
  window is not comparable with one estimated on a longer window. The deflated
  Sharpe now counts every cell and measures the dispersion over the sample the
  overfitting report uses.
* **``pct_change``.** pandas 2.0-2.2 forward-fill by default and 3.0 does not,
  so the same price panel produced two different return series depending on
  which pandas was installed. Every call site now says ``fill_method=None``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.analytics.performance import (
    _rf_per_period,
    annualize_returns,
    annualize_volatility,
    rolling_metrics,
    sharpe_ratio,
    summary_stats,
)
from optimization_engine.analytics.selection import _period_sharpe
from optimization_engine.backtest.sweep import SweepSpec, run_sweep
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.data.quality import analyze_prices


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return sample_dataset()


@pytest.fixture(scope="module")
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def equal_weight(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1).rename("equal_weight")


# ---------------------------------------------------------------------------
# A. One Sharpe definition
# ---------------------------------------------------------------------------


def test_sharpe_single_source(equal_weight: pd.Series) -> None:
    """Every other Sharpe in the tree is this one, differently parameterized."""
    # The selection module's per-period Sharpe is the arithmetic ratio on a
    # one-period-per-year clock.
    assert float(sharpe_ratio(equal_weight, periods_per_year=1)) == pytest.approx(
        _period_sharpe(equal_weight), abs=1e-12, rel=0.0
    )

    # And the rolling frame's last row is the Sharpe of the last window.
    window = 63
    rolling = rolling_metrics(
        equal_weight, window, riskfree_rate=0.02, periods_per_year=252
    )
    tail = equal_weight.iloc[-window:]
    assert float(rolling["rolling_sharpe"].iloc[-1]) == pytest.approx(
        float(sharpe_ratio(tail, 0.02, 252)), abs=1e-12, rel=0.0
    )
    # Not just the last row: the whole column.
    manual = pd.Series(
        {
            equal_weight.index[i]: float(
                sharpe_ratio(equal_weight.iloc[i - window + 1 : i + 1], 0.02, 252)
            )
            for i in range(window - 1, len(equal_weight), 137)
        }
    )
    pd.testing.assert_series_equal(
        rolling["rolling_sharpe"].reindex(manual.index),
        manual,
        check_names=False,
        atol=1e-12,
        rtol=0.0,
    )


def test_sharpe_geometric_matches_old_formula(
    equal_weight: pd.Series, returns: pd.DataFrame
) -> None:
    """``method="geometric"`` reproduces the pre-Task-1.2 number exactly."""
    for riskfree_rate in (0.0, 0.03):
        rf = _rf_per_period(riskfree_rate, 252)
        expected = annualize_returns(equal_weight - rf, 252) / annualize_volatility(
            equal_weight, 252
        )
        assert float(
            sharpe_ratio(equal_weight, riskfree_rate, 252, method="geometric")
        ) == pytest.approx(float(expected), abs=0.0, rel=1e-15)

    # Frames too — ``summary_stats`` aggregates column by column.
    rf = _rf_per_period(0.03, 252)
    frame_expected = annualize_returns(returns - rf, 252) / annualize_volatility(
        returns, 252
    )
    pd.testing.assert_series_equal(
        sharpe_ratio(returns, 0.03, 252, method="geometric"),
        frame_expected,
        atol=0.0,
        rtol=1e-15,
    )


def test_sharpe_methods_differ_and_the_summary_shows_both(
    equal_weight: pd.Series,
) -> None:
    """The documented example: the two conventions do not agree.

    On the eight-year sample panel, equal-weighted, the geometric Sharpe is
    0.5950 and the arithmetic one 0.6238 — 4.8% higher, because the geometric
    numerator is the arithmetic one less roughly half the variance. Neither is
    wrong; reporting one under the other's name is. The summary carries both
    for one release so a reader can see which number moved.
    """
    geometric = float(sharpe_ratio(equal_weight, 0.0, 252, method="geometric"))
    arithmetic = float(sharpe_ratio(equal_weight, 0.0, 252, method="arithmetic"))
    assert geometric == pytest.approx(0.5950, abs=5e-5)
    assert arithmetic == pytest.approx(0.6238, abs=5e-5)
    assert arithmetic / geometric - 1.0 == pytest.approx(0.048, abs=5e-4)

    # The default is the arithmetic one.
    assert float(sharpe_ratio(equal_weight, 0.0, 252)) == arithmetic

    stats = summary_stats(equal_weight.to_frame("ew"), riskfree_rate=0.0)
    assert float(stats.loc["ew", "Sharpe Ratio"]) == pytest.approx(arithmetic)
    assert float(stats.loc["ew", "Sharpe Ratio (geometric)"]) == pytest.approx(
        geometric
    )
    # The geometric column sits next to the one it explains.
    columns = list(stats.columns)
    assert columns.index("Sharpe Ratio (geometric)") == columns.index("Sharpe Ratio") + 1


def test_an_unknown_sharpe_method_is_refused(equal_weight: pd.Series) -> None:
    with pytest.raises(ValueError, match="arithmetic"):
        sharpe_ratio(equal_weight, method="log")


def test_probabilistic_sharpe_now_agrees_with_sharpe_ratio(
    equal_weight: pd.Series,
) -> None:
    """The PSR's internal Sharpe was always arithmetic per period.

    Before Task 1.2 it disagreed with ``sharpe_ratio``; now it does not.
    """
    from optimization_engine.analytics.performance import probabilistic_sharpe_ratio

    rf = _rf_per_period(0.0, 252)
    excess = equal_weight.dropna() - rf
    internal = float(excess.mean() / excess.std(ddof=1))
    assert internal == pytest.approx(
        float(sharpe_ratio(equal_weight, 0.0, periods_per_year=1)), abs=1e-12, rel=0.0
    )
    # And the reported probability is a plain function of that same number.
    assert 0.0 <= probabilistic_sharpe_ratio(equal_weight) <= 1.0


def test_the_geometric_ratios_are_documented_as_such() -> None:
    """Sortino, Calmar and Martin keep a geometric numerator — on purpose.

    Task 1.2 changed Sharpe alone. A summary that mixes an arithmetic Sharpe
    with a geometric Sortino is the same class of bug the task exists to kill,
    so the mixing has to be stated where a reader will meet it.
    """
    from optimization_engine.analytics.performance import (
        calmar_ratio,
        martin_ratio,
        sortino_ratio,
    )

    for func in (sortino_ratio, calmar_ratio, martin_ratio, summary_stats):
        doc = func.__doc__ or ""
        assert "geometric" in doc, f"{func.__name__} does not say which convention it uses"


# ---------------------------------------------------------------------------
# B. Deflated Sharpe counts every cell, over one sample
# ---------------------------------------------------------------------------


def _base_config(returns: pd.DataFrame) -> EngineConfig:
    return EngineConfig(
        expected_returns=dict.fromkeys(returns.columns, 0.0),
        optimizer=OptimizerSpec(name="min_variance"),
        periods_per_year=252,
    )


def _noise(rng: np.random.Generator, index: pd.Index) -> pd.Series:
    return pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index)


def test_dsr_uses_arithmetic_trials(returns: pd.DataFrame) -> None:
    """The dispersion fed to the deflation is a distribution of *arithmetic* Sharpes."""
    rng = np.random.default_rng(11)
    index = returns.index[:800]
    streams = {value: _noise(rng, index) for value in (1.0, 2.0, 3.0, 4.0)}
    sweep = SweepSpec(params={"optimizer.risk_aversion": sorted(streams)})
    results = run_sweep(
        _base_config(returns),
        sweep,
        lambda cfg: streams[cfg.optimizer.risk_aversion],
    )

    trials = results.trial_sharpes()
    matrix = results.return_matrix()
    expected = sharpe_ratio(matrix, 0.0, 252, method="arithmetic")
    pd.testing.assert_series_equal(
        trials, expected.astype(float), check_names=False, atol=1e-12, rtol=0.0
    )

    # And they are genuinely not the geometric ones.
    geometric = sharpe_ratio(matrix, 0.0, 252, method="geometric")
    assert not np.allclose(trials.to_numpy(), geometric.to_numpy(), atol=1e-6)

    # The per-cell metric column reports the same convention.
    ok = results.frame[results.frame["status"] == "ok"]
    np.testing.assert_allclose(
        ok["sharpe"].to_numpy(dtype=float), trials.to_numpy(), atol=1e-12
    )


def test_dsr_counts_failed_cells(returns: pd.DataFrame) -> None:
    """A cell that blew up is still a configuration you tried."""
    rng = np.random.default_rng(3)
    index = returns.index[:800]
    streams = {value: _noise(rng, index) for value in (1.0, 2.0, 3.0)}

    def evaluate(cfg: EngineConfig) -> pd.Series:
        if cfg.optimizer.risk_aversion == 4.0:
            raise RuntimeError("forced failure")
        return streams[cfg.optimizer.risk_aversion]

    sweep = SweepSpec(params={"optimizer.risk_aversion": [1.0, 2.0, 3.0, 4.0]})
    results = run_sweep(_base_config(returns), sweep, evaluate)

    assert results.n_cells == 4
    assert results.n_ok == 3 and results.n_failed == 1
    assert results.deflated_sharpe(0).n_trials == 4


def test_dsr_refuses_a_grid_with_no_shared_sample(returns: pd.DataFrame) -> None:
    """Two cells that share no dates have no dispersion, and say so.

    The aligned trial Sharpes come off ``return_matrix``, so a grid whose
    cells do not overlap yields nothing to deflate against. That has to fail
    with a sentence about the overlap, not with the generic "needs at least 2
    entries" from deep inside the deflation.
    """
    rng = np.random.default_rng(9)
    early, late = returns.index[:300], returns.index[400:700]
    streams = {1.0: _noise(rng, early), 2.0: _noise(rng, late)}
    sweep = SweepSpec(params={"optimizer.risk_aversion": sorted(streams)})
    results = run_sweep(
        _base_config(returns), sweep, lambda cfg: streams[cfg.optimizer.risk_aversion]
    )

    assert results.n_ok == 2
    assert results.return_matrix().empty
    with pytest.raises(ValueError, match="date"):
        results.deflated_sharpe(0)


def test_dsr_and_pbo_share_a_sample(returns: pd.DataFrame) -> None:
    """Sweep the estimation window and the two diagnostics still see one sample.

    ``EngineConfig`` has no ``lookback`` field — the walk-forward lookback is
    an argument, not configuration — so the grid sweeps ``ema_span``, which is
    the estimation window a cell would burn before its first decision. Streams
    of unequal length are the point: ``return_matrix`` inner-joins them, and
    before Task 1.2 the trial Sharpes were read off the full-length per-cell
    metrics instead.
    """
    rng = np.random.default_rng(5)
    full = returns.index[:800]
    streams = {60: _noise(rng, full), 240: _noise(rng, full[180:])}
    sweep = SweepSpec(params={"ema_span": sorted(streams)})
    results = run_sweep(
        _base_config(returns), sweep, lambda cfg: streams[cfg.ema_span]
    )

    matrix = results.return_matrix()
    assert len(matrix) == len(full) - 180  # the shorter cell decides the window
    trials = results.trial_sharpes()
    assert list(trials.index) == list(matrix.columns)

    # The unaligned form is the old behaviour, and it disagrees — which is the
    # whole reason the aligned one is the default.
    unaligned = results.trial_sharpes(aligned=False)
    assert list(unaligned.index) == [0, 1]
    assert float(unaligned.iloc[0]) != pytest.approx(float(trials.iloc[0]), abs=1e-9)

    # Both diagnostics now read the same T x N block.
    assert results.deflated_sharpe(0).n_trials == results.n_cells
    assert 0.0 <= results.overfitting_report(n_partitions=4).pbo <= 1.0


# ---------------------------------------------------------------------------
# C. pct_change must not depend on the pandas version
# ---------------------------------------------------------------------------


def _gappy_panel() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, np.nan, 103.0, 104.0],
            "B": [50.0, 50.5, 51.0, 51.5, 52.0],
        },
        index=index,
    )


def test_returns_do_not_pad_gaps() -> None:
    """An interior gap is two missing returns, never a 0% and a compounded jump.

    Under pandas 2.0-2.2's ``fill_method="pad"`` default the gap became a 0%
    return followed by the two-period move booked as one. Whichever pandas is
    installed, the answer here is NaN twice.
    """
    panel = _gappy_panel()
    returns = prices_to_returns(panel)

    assert np.isnan(returns.loc["2024-01-03", "A"])  # the gap itself
    assert np.isnan(returns.loc["2024-01-04", "A"])  # and the period after it
    assert float(returns.loc["2024-01-05", "A"]) == pytest.approx(104.0 / 103.0 - 1.0)
    # The padded answer, which must not appear anywhere.
    assert not np.isclose(float(returns.loc["2024-01-04", "A"]), 103.0 / 101.0 - 1.0)
    # An asset without gaps is untouched.
    assert not returns["B"].isna().any()

    # The volatility and return helpers difference prices themselves, so they
    # carry the same hazard and must reach the same answer.
    padded_prices = panel["A"].ffill()
    for func in (annualize_volatility, annualize_returns):
        gappy = float(func(panel["A"], 252, prices=True))
        unpadded = float(func(panel["A"].pct_change(fill_method=None).dropna(), 252))
        padded = float(func(padded_prices.pct_change(fill_method=None).dropna(), 252))
        assert gappy == pytest.approx(unpadded)
        assert gappy != pytest.approx(padded), f"{func.__name__} padded the gap"

    # And the data-quality report counts the gap, not a fabricated zero return.
    per_asset = analyze_prices(panel).per_asset
    assert float(per_asset.loc["A", "zero_return_share"]) == 0.0
    assert int(per_asset.loc["A", "missing_interior"]) == 1


def test_no_bare_pct_change_survives_in_the_tree() -> None:
    """Every ``pct_change`` call names ``fill_method`` explicitly.

    The default changed between two supported pandas versions, so leaving it
    implicit makes the return series a property of the environment.
    """
    offenders: list[str] = []
    for root in (ROOT / "src", ROOT / "app"):
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr != "pct_change":
                    continue
                if not any(kw.arg == "fill_method" for kw in node.keywords):
                    # POSIX form, so the message reads the same on Windows.
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert offenders == [], "bare pct_change(): " + ", ".join(offenders)
