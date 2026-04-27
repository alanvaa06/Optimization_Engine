"""High-level engine façade.

Glues together loading, covariance estimation, and optimizer dispatch so
that callers can run the whole pipeline with a single call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from optimization_engine.analytics.performance import summary_stats
from optimization_engine.analytics.relative import summary_relative
from optimization_engine.analytics.risk import risk_contribution
from optimization_engine.config import EngineConfig
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.frontier import FrontierResult, efficient_frontier
from optimization_engine.optimizers.base import OptimizationResult
from optimization_engine.optimizers.factory import optimizer_factory


@dataclass
class EngineRun:
    config: EngineConfig
    returns: pd.DataFrame
    cov_matrix: pd.DataFrame
    expected_returns: pd.Series
    result: OptimizationResult
    frontier: FrontierResult | None = None

    def risk_contributions(self) -> pd.Series:
        return risk_contribution(self.result.weights, self.cov_matrix)

    def backtest_returns(self, benchmark_returns: pd.Series | None = None) -> pd.DataFrame:
        port = (self.returns * self.result.weights.reindex(self.returns.columns).fillna(0.0)).sum(axis=1)
        out = pd.DataFrame({"portfolio": port})
        if benchmark_returns is not None:
            out["benchmark"] = benchmark_returns.reindex(port.index)
        return out

    def absolute_summary(self, riskfree_rate: float = 0.0) -> pd.DataFrame:
        bt = self.backtest_returns()
        return summary_stats(bt, periods_per_year=self.config.periods_per_year, riskfree_rate=riskfree_rate)

    def relative_summary(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        bt = self.backtest_returns(benchmark_returns)
        return summary_relative(
            bt[["portfolio"]],
            bt["benchmark"],
            periods_per_year=self.config.periods_per_year,
        )


def run_engine(
    returns: pd.DataFrame,
    config: EngineConfig,
    expected_returns: pd.Series | None = None,
    build_frontier: bool = False,
    n_frontier_points: int = 25,
    return_range: tuple[float, float] | None = None,
) -> EngineRun:
    """Run the engine end-to-end.

    Args:
        returns: A DataFrame of asset returns (rows = periods, cols = assets).
        config: An :class:`EngineConfig` describing the optimizer + constraints.
        expected_returns: Override for expected returns. Defaults to
            ``config.expected_returns``.
        build_frontier: If True, also computes the efficient frontier.
        n_frontier_points: Resolution of the frontier sweep.
        return_range: Optional (lo, hi) range to sweep; defaults to
            (min μ, max μ).
    """
    cov = covariance_matrix(
        returns,
        method=config.covariance_method,
        annualize=True,
        periods_per_year=config.periods_per_year,
        ewma_lambda=config.ewma_lambda,
    )

    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)
    if expected_returns is None:
        # default: annualized historical mean.
        expected_returns = (1 + returns).prod() ** (
            config.periods_per_year / len(returns)
        ) - 1
    expected_returns = expected_returns.reindex(returns.columns).fillna(0.0)

    optimizer = optimizer_factory(
        config, cov, expected_returns=expected_returns, returns=returns
    )
    result = optimizer.optimize()

    frontier = None
    if build_frontier:
        frontier = efficient_frontier(
            config,
            cov,
            expected_returns=expected_returns,
            returns=returns,
            n_points=n_frontier_points,
            return_range=return_range,
        )

    return EngineRun(
        config=config,
        returns=returns,
        cov_matrix=cov,
        expected_returns=expected_returns,
        result=result,
        frontier=frontier,
    )
