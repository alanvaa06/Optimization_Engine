"""High-level engine façade.

Glues together loading, covariance estimation, feasibility analysis and
optimizer dispatch so that callers can run the whole pipeline with a single
call — and get back not just an allocation but the evidence needed to decide
whether to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from optimization_engine.analytics.backtest import (
    BacktestResult,
    RebalanceFrequency,
    WalkForwardResult,
    backtest_weights,
    compare_in_and_out_of_sample,
    walk_forward_backtest,
)
from optimization_engine.analytics.performance import summary_stats
from optimization_engine.analytics.relative import summary_relative
from optimization_engine.analytics.risk import (
    group_risk_contribution,
    risk_contribution,
)
from optimization_engine.config import EngineConfig
from optimization_engine.data.covariance import (
    CovarianceDiagnostics,
    covariance_diagnostics,
    covariance_matrix,
)
from optimization_engine.frontier import FrontierResult, efficient_frontier
from optimization_engine.optimizers.base import OptimizationResult
from optimization_engine.optimizers.diagnostics import (
    PortfolioDiagnostics,
    risk_decomposition,
)
from optimization_engine.optimizers.factory import (
    constraints_from_config,
    effective_expected_returns,
    optimizer_factory,
)
from optimization_engine.optimizers.feasibility import (
    FeasibilityReport,
    InfeasibleConstraintsError,
    analyze_feasibility,
)


def apply_fx_conversion(
    prices: pd.DataFrame,
    config: EngineConfig,
    fx_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert ``prices`` into ``config.base_currency`` if needed.

    No-op when every asset is already quoted in the base currency, or
    when ``config.currencies`` is empty.
    """
    if not config.currencies:
        return prices
    base = config.base_currency.upper()
    needed = {config.currencies.get(a, base).upper() for a in prices.columns}
    if needed == {base}:
        return prices

    # Local import: keeps this module importable when urllib is restricted.
    from optimization_engine.data.fx import convert_prices_to_base

    return convert_prices_to_base(
        prices,
        asset_currency=config.currencies,
        base=base,
        fx_rates=fx_rates,
    )


@dataclass
class EngineRun:
    """Everything one optimization produced, plus the evidence behind it."""

    config: EngineConfig
    returns: pd.DataFrame
    cov_matrix: pd.DataFrame
    expected_returns: pd.Series
    result: OptimizationResult
    frontier: FrontierResult | None = None
    feasibility: FeasibilityReport | None = None
    covariance_diagnostics: CovarianceDiagnostics | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # -- allocation views ---------------------------------------------------

    @property
    def diagnostics(self) -> PortfolioDiagnostics | None:
        """Concentration, exposure and compliance summary of the allocation."""
        return self.result.extras.get("diagnostics")

    def risk_contributions(self) -> pd.Series:
        """Per-asset share of total portfolio risk."""
        return risk_contribution(self.result.weights, self.cov_matrix)

    def risk_decomposition(self) -> pd.DataFrame:
        """Euler decomposition of portfolio volatility, in volatility units."""
        return risk_decomposition(self.result.weights, self.cov_matrix)

    def group_risk_contributions(self) -> pd.Series:
        """Risk shares aggregated to asset-class level."""
        return group_risk_contribution(
            self.result.weights, self.cov_matrix, self.config.groups
        )

    # -- backtests ----------------------------------------------------------

    def backtest_returns(self, benchmark_returns: pd.Series | None = None) -> pd.DataFrame:
        """In-sample replay assuming costless rebalancing every period.

        Kept for backwards compatibility and for quick comparisons. It is
        **in-sample** — the weights were chosen knowing these returns — and it
        ignores drift and trading costs. Use :meth:`backtest` for an honest
        replay and :meth:`walk_forward` for an out-of-sample track record.
        """
        port = (self.returns * self.result.weights.reindex(self.returns.columns).fillna(0.0)).sum(axis=1)
        out = pd.DataFrame({"portfolio": port})
        if benchmark_returns is not None:
            out["benchmark"] = benchmark_returns.reindex(port.index)
        return out

    def backtest(
        self,
        frequency: RebalanceFrequency = "monthly",
        transaction_cost_bps: float = 0.0,
    ) -> BacktestResult:
        """Replay the solved weights with realistic drift, rebalancing and costs.

        Still in-sample: the optimizer saw this history. What it adds over
        :meth:`backtest_returns` is that the weights actually drift between
        rebalances and trading is charged for.
        """
        return backtest_weights(
            self.returns,
            self.result.weights,
            frequency=frequency,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=self.config.periods_per_year,
            is_out_of_sample=False,
        )

    def walk_forward(
        self,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        transaction_cost_bps: float = 0.0,
        expanding: bool = False,
        solve: Callable[[pd.DataFrame], pd.Series] | None = None,
        reestimate_expected_returns: bool = True,
    ) -> WalkForwardResult:
        """Out-of-sample evaluation: re-estimate and re-solve on a rolling window.

        Defaults to a two-year lookback rebalanced quarterly, scaled by
        ``periods_per_year`` so the same call works for daily, weekly or
        monthly data.

        Args:
            lookback: Estimation window in periods. Defaults to two years.
            rebalance_every: Periods between re-solves. Defaults to one quarter.
            transaction_cost_bps: One-way cost on traded notional.
            expanding: Grow the window from the start instead of rolling it.
            solve: Override the solver. Defaults to re-running this run's own
                config on each window — which is the point: the *process* is
                what gets evaluated, not one lucky weight vector.
            reestimate_expected_returns: Re-derive expected returns inside each
                window instead of reusing the ones on the config.

                This defaults to True because leaving it off is a look-ahead
                leak in the usual case. ``config.expected_returns`` is normally
                populated — the UI always fills that table, and it seeds it from
                the *full* history — so reusing it hands every "out-of-sample"
                window an estimate computed partly from its own future. On the
                sample panel that lifts walk-forward Sharpe from 0.46 to 0.89.

                Set it to False only when the expected returns are genuinely
                forward-looking capital-market assumptions rather than
                estimates from this history; then holding them fixed is right,
                and the engine cannot tell the two cases apart on its own.
        """
        ppy = self.config.periods_per_year
        lookback = lookback or max(2 * ppy, 24)
        rebalance_every = rebalance_every or max(ppy // 4, 1)

        if solve is None:
            import copy

            base_config = copy.deepcopy(self.config)
            if reestimate_expected_returns:
                # Emptying the vector makes run_engine derive it from the
                # window via expected_returns_method.
                base_config.expected_returns = {}

            def solve(window: pd.DataFrame) -> pd.Series:
                return run_engine(
                    window, base_config, check_feasibility=False
                ).result.weights

        result = walk_forward_backtest(
            self.returns,
            solve,
            lookback=lookback,
            rebalance_every=rebalance_every,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=ppy,
            expanding=expanding,
        )
        result.backtest.metadata["reestimated_expected_returns"] = bool(
            reestimate_expected_returns
        )
        return result

    def in_vs_out_of_sample(
        self, walk_forward_result: WalkForwardResult, riskfree_rate: float = 0.0
    ) -> pd.DataFrame:
        """Side-by-side fitted vs walk-forward statistics, with the gap."""
        oos = walk_forward_result.returns
        in_sample = self.backtest_returns()["portfolio"].reindex(oos.index)
        return compare_in_and_out_of_sample(
            in_sample, oos, self.config.periods_per_year, riskfree_rate
        )

    # -- summaries ----------------------------------------------------------

    def absolute_summary(
        self, riskfree_rate: float = 0.0, extended: bool = False
    ) -> pd.DataFrame:
        bt = self.backtest_returns()
        return summary_stats(
            bt,
            periods_per_year=self.config.periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=extended,
        )

    def relative_summary(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        bt = self.backtest_returns(benchmark_returns)
        return summary_relative(
            bt[["portfolio"]],
            bt["benchmark"],
            periods_per_year=self.config.periods_per_year,
        )

    def assumptions(self) -> dict[str, Any]:
        """Every modelling choice this run rests on, in one place.

        A number without its assumptions is not a result. This is what the UI
        and the Excel report print alongside the weights so the reader can see
        what was assumed rather than infer it.
        """
        spec = self.config.optimizer
        start = self.returns.index.min()
        end = self.returns.index.max()
        return {
            "optimizer": spec.name,
            "objective_mode": self.result.extras.get("mode", "—"),
            "covariance_estimator": self.config.covariance_method,
            "ewma_lambda": (
                self.config.ewma_lambda
                if self.config.covariance_method == "ewma"
                else None
            ),
            "expected_returns_method": self.config.expected_returns_method,
            "risk_free_rate": spec.risk_free_rate,
            "risk_aversion": spec.risk_aversion,
            "target_return": spec.target_return,
            "target_volatility": spec.target_volatility,
            "periods_per_year": self.config.periods_per_year,
            "base_currency": self.config.base_currency,
            "sample_start": str(getattr(start, "date", lambda: start)()),
            "sample_end": str(getattr(end, "date", lambda: end)()),
            "n_observations": int(len(self.returns)),
            "n_assets": int(self.returns.shape[1]),
            "long_only": self.config.long_only,
            "fully_invested": self.config.fully_invested,
            "leverage_cap": self.config.leverage,
            "turnover_limit": self.config.turnover_limit,
            "solver": self.result.extras.get("solver"),
            "solver_status": self.result.extras.get("solver_status"),
        }


def run_engine(
    returns: pd.DataFrame,
    config: EngineConfig,
    expected_returns: pd.Series | None = None,
    build_frontier: bool = False,
    n_frontier_points: int = 25,
    return_range: tuple[float, float] | None = None,
    check_feasibility: bool = True,
    raise_on_infeasible: bool = False,
) -> EngineRun:
    """Run the engine end-to-end.

    Args:
        returns: A DataFrame of asset returns (rows = periods, cols = assets).
        config: An :class:`EngineConfig` describing the optimizer + constraints.
        expected_returns: Override for expected returns. Defaults to
            ``config.expected_returns``.
        build_frontier: If True, also computes the efficient frontier.
        n_frontier_points: Resolution of the frontier sweep.
        return_range: Optional (lo, hi) range to sweep. Defaults to the range
            the constraints can actually reach.
        check_feasibility: Analyze the constraint set before solving and
            attach the report to the run. Cheap relative to the solve, and it
            turns ``status=infeasible`` into an actionable message.
        raise_on_infeasible: Raise :class:`InfeasibleConstraintsError` instead
            of letting the solver fail with a less informative error.

    Raises:
        ValueError: If ``returns`` is empty or has no columns.
        InfeasibleConstraintsError: When ``raise_on_infeasible`` is set and the
            constraints cannot be satisfied.
    """
    if returns is None or returns.empty:
        raise ValueError("run_engine received an empty returns frame.")
    if returns.shape[1] == 0:
        raise ValueError("run_engine received returns with no asset columns.")

    cov = covariance_matrix(
        returns,
        method=config.covariance_method,
        annualize=True,
        periods_per_year=config.periods_per_year,
        ewma_lambda=config.ewma_lambda,
    )
    cov_diag = covariance_diagnostics(
        cov,
        n_observations=len(returns),
        method=config.covariance_method,
        ewma_lambda=config.ewma_lambda,
    )

    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)
    if expected_returns is None:
        from optimization_engine.data.covariance import expected_returns_from_history

        market_w = (
            pd.Series(config.market_weights) if config.market_weights else None
        )
        expected_returns = expected_returns_from_history(
            returns,
            method=("mean" if config.expected_returns_method == "historical_mean"
                    else config.expected_returns_method),
            periods_per_year=config.periods_per_year,
            span=config.ema_span,
            market_return=config.market_return,
            risk_free_rate=config.optimizer.risk_free_rate,
            market_weights=market_w,
            cov_matrix=cov,
        )
    expected_returns = expected_returns.reindex(returns.columns).fillna(0.0)

    feasibility: FeasibilityReport | None = None
    if check_feasibility:
        # Black-Litterman optimizes against its equilibrium posterior, not the
        # configured vector, so a return target has to be checked against the
        # returns the solver will really see.
        feasibility = analyze_feasibility(
            list(returns.columns),
            constraints_from_config(config),
            expected_returns=effective_expected_returns(config, cov, expected_returns),
            cov_matrix=cov,
        )
        if raise_on_infeasible and not feasibility.is_feasible:
            raise InfeasibleConstraintsError(feasibility)

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

    run_warnings: list[str] = list(cov_diag.warnings)
    if feasibility is not None:
        run_warnings.extend(i.message for i in feasibility.warnings)
    run_warnings.extend(result.violations)

    return EngineRun(
        config=config,
        returns=returns,
        cov_matrix=cov,
        expected_returns=expected_returns,
        result=result,
        frontier=frontier,
        feasibility=feasibility,
        covariance_diagnostics=cov_diag,
        warnings=tuple(run_warnings),
    )
