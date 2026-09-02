"""The expected-return convention, and the single place it is defined.

Mean-variance optimization is a single-period model. It trades ``μ'w`` off
against ``λ·w'Σw`` over *one* period, so the ``μ`` it wants is the
expectation of one period's return — the arithmetic mean. The geometric mean
is the realized compound growth rate, lower by roughly ``σ²/2``, and it
answers a multi-period question instead. Pairing a geometric ``μ`` with an
arithmetic ``Σ`` is not conservatism; it measures reward and penalty on two
different conventions and the size of the discrepancy is exactly the quantity
being traded off.

These tests pin the convention, pin the geometric estimator that used to
occupy its name, and pin that neither formula is written out anywhere else.
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

from optimization_engine.config import (
    EngineConfig,
    expected_return_method_for_estimator,
)
from optimization_engine.data.covariance import (
    EXPECTED_RETURN_DESCRIPTIONS,
    covariance_matrix,
    expected_returns_from_history,
    james_stein_shrinkage,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=17))


# ---------------------------------------------------------------------------
# The convention
# ---------------------------------------------------------------------------


def test_mean_is_arithmetic(returns: pd.DataFrame):
    """``mean`` is ``r̄ · periods_per_year``, and nothing else."""
    mu = expected_returns_from_history(returns, method="mean", periods_per_year=252)
    expected = returns.mean() * 252
    pd.testing.assert_series_equal(mu, expected, check_names=False)
    assert list(mu.index) == list(returns.columns)


def test_geometric_mean_matches_old_formula(returns: pd.DataFrame):
    """The estimator that used to answer to ``mean`` still exists, named."""
    geometric = expected_returns_from_history(
        returns, method="geometric_mean", periods_per_year=252
    )
    old_formula = ((1 + returns).prod() ** (252 / len(returns))) - 1
    pd.testing.assert_series_equal(geometric, old_formula, check_names=False)


def test_arithmetic_exceeds_geometric_by_about_half_the_variance(
    returns: pd.DataFrame,
):
    """The gap is not a rounding difference — it is about ``σ²/2`` per asset.

    This is the whole reason the two cannot be used interchangeably: the
    error scales with variance, which is precisely the quantity the optimizer
    is penalizing, so swapping one convention for the other tilts the answer
    systematically toward the volatile names.

    The relation is a second-order approximation and it only reads cleanly
    where the variance term dominates. On a near-constant series (this
    panel's ``Cash``, at 0.5% annualized volatility) the *compounding* term
    ``≈ ppy(ppy−1)/2 · r̄²`` is the same size or larger and the geometric
    number ends up above the arithmetic one, so those assets are excluded
    rather than fudged into the tolerance.
    """
    arithmetic = expected_returns_from_history(returns, method="mean")
    geometric = expected_returns_from_history(returns, method="geometric_mean")
    gap = arithmetic - geometric
    half_variance = returns.var() * 252 / 2.0
    volatile = (returns.std() * np.sqrt(252)) > 0.05
    assert volatile.sum() >= 10
    assert (gap[volatile] > 0).all()
    # Second-order approximation, so allow it to be loose but not wrong.
    assert np.allclose(
        gap[volatile].values, half_variance[volatile].values, rtol=0.35, atol=2e-3
    )


def test_shrunk_mean_shrinks_the_arithmetic_mean(returns: pd.DataFrame):
    """Shrinkage pulls the arithmetic means in, not the geometric ones."""
    cov = covariance_matrix(returns, method="ledoit_wolf")
    shrunk = expected_returns_from_history(
        returns, method="shrunk_mean", cov_matrix=cov
    )
    arithmetic = expected_returns_from_history(returns, method="mean")
    geometric = expected_returns_from_history(returns, method="geometric_mean")
    # The shrunk vector is a convex combination of the arithmetic means and a
    # scalar target, so it must sit inside their range, not the geometric one.
    assert shrunk.min() >= arithmetic.min() - 1e-12
    assert shrunk.max() <= arithmetic.max() + 1e-12
    assert float(shrunk.mean()) > float(geometric.mean())


def test_capm_market_return_is_arithmetic(returns: pd.DataFrame):
    """The market premium is read as a single-period expectation too.

    It is multiplied by a beta and added to ``rf``; a geometric premium in
    that expression is the same inconsistency one level down.
    """
    cov = covariance_matrix(returns, method="ledoit_wolf")
    weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    implied = expected_returns_from_history(
        returns,
        method="capm",
        market_weights=weights,
        cov_matrix=cov,
        risk_free_rate=0.0,
    )
    market = (returns * weights).sum(axis=1)
    arithmetic_premium = float(market.mean() * 252)
    betas = (cov.values @ weights.values) / float(
        weights.values @ cov.values @ weights.values
    )
    pd.testing.assert_series_equal(
        implied,
        pd.Series(betas * arithmetic_premium, index=returns.columns),
        check_names=False,
    )


def test_unknown_method_names_every_available_one(returns: pd.DataFrame):
    with pytest.raises(ValueError, match="Unknown expected-return method"):
        expected_returns_from_history(returns, method="arithmetic")
    assert "geometric_mean" in EXPECTED_RETURN_DESCRIPTIONS
    assert set(EXPECTED_RETURN_DESCRIPTIONS) == {
        "mean", "geometric_mean", "ema", "capm", "shrunk_mean",
    }


# ---------------------------------------------------------------------------
# The config's own vocabulary
# ---------------------------------------------------------------------------


def test_config_vocabulary_translates_to_estimator_names():
    assert expected_return_method_for_estimator("historical_mean") == "mean"
    assert expected_return_method_for_estimator("geometric_mean") == "geometric_mean"
    for name in ("ema", "capm", "shrunk_mean"):
        assert expected_return_method_for_estimator(name) == name


def test_every_config_method_resolves_to_a_real_estimator(returns: pd.DataFrame):
    """A name the config accepts must be a name the estimator accepts."""
    import typing

    hints = typing.get_type_hints(EngineConfig)
    allowed = typing.get_args(hints["expected_returns_method"])
    assert "geometric_mean" in allowed
    cov = covariance_matrix(returns, method="ledoit_wolf")
    for name in allowed:
        mu = expected_returns_from_history(
            returns,
            method=expected_return_method_for_estimator(name),
            cov_matrix=cov,
        )
        assert list(mu.index) == list(returns.columns)


def test_config_round_trips_the_new_method():
    config = EngineConfig(expected_returns_method="geometric_mean")
    restored = EngineConfig.from_dict(config.to_dict())
    assert restored.expected_returns_method == "geometric_mean"


# ---------------------------------------------------------------------------
# Bayes-Stein: an unshrunk vector reports zero shrinkage
# ---------------------------------------------------------------------------


def test_bayes_stein_degenerate_reports_zero_intensity():
    """Nothing was shrunk, so the reported intensity has to be zero.

    ``λ = 1`` is the analytic limit of Jorion's formula as the quadratic
    form goes to zero, but the intensity is not being asked what the formula
    tends to — it is being asked what the estimator did to the vector it
    returned. It returned the sample means untouched. Reporting full
    shrinkage beside them is a statement about the output that is false, and
    the ``LinAlgError`` exit already reports ``0.0`` for the same reason.
    """
    assets = [f"a{i}" for i in range(5)]
    # Every mean identical ⇒ the deviation from the Jorion target is exactly
    # zero ⇒ the quadratic form is zero and no shrinkage is possible.
    mu = pd.Series(0.07, index=assets)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert intensity == 0.0
    pd.testing.assert_series_equal(shrunk, mu)


def test_bayes_stein_reports_a_real_intensity_when_it_shrinks():
    """The zero above must mean "nothing happened", not "always zero"."""
    assets = [f"a{i}" for i in range(5)]
    mu = pd.Series([0.02, 0.05, 0.09, 0.14, 0.20], index=assets)
    rng = np.random.default_rng(1)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert 0.0 < intensity <= 1.0
    assert float(shrunk.std()) < float(mu.std())


# ---------------------------------------------------------------------------
# One definition site
# ---------------------------------------------------------------------------

#: Trees to scan: the library, the app, and the scripts that ship with it.
#: ``tests`` is deliberately outside it — ``test_geometric_mean_matches_old_
#: formula`` writes the formula out on purpose, as the pin.
_SCANNED = ("src", "app", "scripts")

#: Names that mark an exponent as an *annualization*, as opposed to the
#: ``1/n`` exponent of a plain geometric mean over a window.
_ANNUALIZERS = ("periods_per_year", "ppy", "252", "12", "52")


def _annualized_compounding_sites() -> list[tuple[str, int, str]]:
    """Every ``(...).prod() ** <annualizing exponent>`` in the scanned trees.

    Returns:
        ``(path, lineno, enclosing function)`` for each occurrence.
    """
    found: list[tuple[str, int, str]] = []
    for tree_name in _SCANNED:
        for path in sorted((ROOT / tree_name).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
            enclosing: dict[int, str] = {}
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        enclosing.setdefault(getattr(inner, "lineno", -1), node.name)
            for node in ast.walk(module):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
                    continue
                has_prod = any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "prod"
                    for sub in ast.walk(node.left)
                )
                if not has_prod:
                    continue
                exponent = ast.unparse(node.right)
                if not any(token in exponent for token in _ANNUALIZERS):
                    continue
                found.append(
                    (
                        str(path.relative_to(ROOT)),
                        node.lineno,
                        enclosing.get(node.lineno, "<module>"),
                    )
                )
    return found


def test_no_other_mu_implementation():
    """The annualized compound-return formula lives in exactly one place.

    It used to be written out in six: twice inside ``covariance.py`` itself,
    twice in the Streamlit app, once in the doc-image script, and once as
    the ``mean`` branch. Six copies of a convention is six chances for them
    to disagree, and they did — the app's config table seeded a geometric μ
    into a solve the engine ran against a different one.

    ``analytics.performance.annualize_returns`` is deliberately *not* an
    exception here: it computes the compounding in two statements
    (``compounded = (1 + r).prod()`` then ``compounded ** ...``), so it does
    not match this shape, and it is the sanctioned single source for the
    *realized* annualized return, which is a different quantity from μ.
    """
    sites = _annualized_compounding_sites()
    where = [(path, function) for path, _, function in sites]
    assert where == [
        (
            "src/optimization_engine/data/covariance.py",
            "expected_returns_from_history",
        )
    ], f"annualized compounding written out in more than one place: {sites}"


def test_the_realized_annualizer_is_still_the_one_for_realized_returns():
    """The app's performance tiles must not grow a private copy either."""
    from optimization_engine.analytics.performance import annualize_returns

    series = pd.Series(np.full(504, 0.0004))
    assert float(annualize_returns(series, periods_per_year=252)) == pytest.approx(
        (1.0004**504) ** (252 / 504) - 1
    )
    app_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert ".prod()" not in app_source, (
        "the app compounds a return series itself again; route it through "
        "annualize_returns (realized) or expected_returns_from_history (μ)"
    )
    assert "annualize_returns(" in app_source
