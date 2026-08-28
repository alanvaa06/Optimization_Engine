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
from optimization_engine.analytics.report import PerformanceReport, performance_report
from optimization_engine.analytics.risk import (
    group_risk_contribution,
    risk_contribution,
)
from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import BacktestSpec
from optimization_engine.backtest.sweep import SweepResults, SweepSpec, run_sweep
from optimization_engine.backtest.tearsheet import Tearsheet, build_tearsheet
from optimization_engine.backtest.walkforward import WalkForwardRun, walk_forward_run
from optimization_engine.benchmark import (
    ResolvedBenchmark,
    resolve_benchmark,
)
from optimization_engine.config import EngineConfig
from optimization_engine.constraints import effective_layers, layer_exposures
from optimization_engine.data.covariance import (
    CovarianceDiagnostics,
    covariance_diagnostics,
    covariance_from_config,
)
from optimization_engine.frontier import FrontierResult, efficient_frontier
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import OptimizationResult
from optimization_engine.optimizers.diagnostics import (
    PortfolioDiagnostics,
    risk_decomposition,
)
from optimization_engine.optimizers.factory import (
    constraints_from_config,
    effective_expected_returns,
    optimizer_factory,
    validate_benchmark_constraints,
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
    #: The benchmark this run was measured — and possibly optimized — against,
    #: resolved once at solve time so every downstream view uses the same one.
    benchmark: ResolvedBenchmark | None = None

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

    @property
    def constraint_layers(self):
        """Every layer of the allocation policy this run was solved under."""
        return effective_layers(self.config)

    def layer_exposures(self) -> pd.DataFrame:
        """Realized exposure of every bucket, on every layer, against its limit.

        The table an allocator reads to see *which* part of the policy is
        binding: a book that stops at 60% equity because the asset-class cap
        says so is a different portfolio from one that stops there because the
        EM sub-limit ran out, and only the headroom column tells them apart.
        """
        return layer_exposures(self.result.weights, self.constraint_layers)

    def layer_risk_contributions(self, layer: str | None = None) -> pd.Series:
        """Share of portfolio risk carried by each bucket of one layer.

        Weight limits are set on capital, but the thing they are trying to
        control is risk, and the two diverge sharply — a 30% fixed-income
        sleeve rarely carries 30% of the risk. Defaults to the first layer.

        Raises:
            ValueError: When the run has no layers, or none by that name.
        """
        layers = self.constraint_layers
        if not layers:
            raise ValueError(
                "This run has no allocation layers, so there are no buckets "
                "to attribute risk to."
            )
        chosen = next(
            (lyr for lyr in layers if layer is None or lyr.name == layer), None
        )
        if chosen is None:
            names = ", ".join(lyr.name for lyr in layers)
            raise ValueError(f"No layer named {layer!r}. Available: {names}.")
        return group_risk_contribution(
            self.result.weights, self.cov_matrix, chosen.assignments
        )

    def diversification(self, model: str = "minimum_torsion"):
        """Effective number of bets on uncorrelated factors (Meucci).

        Complements :attr:`diagnostics`, whose ``effective_n`` and
        ``effective_n_risk`` are computed asset by asset and so cannot see
        that several positions are the same bet. Not computed on every solve
        because the minimum-torsion rotation is an iterative fixed point and
        the engine should not pay for it unless asked.
        """
        from optimization_engine.analytics.diversification import (
            diversification_distribution,
        )

        return diversification_distribution(
            self.result.weights, self.cov_matrix, model=model
        )

    def diversification_comparison(self) -> pd.DataFrame:
        """Effective bets under both rotations; the gap is the diagnostic."""
        from optimization_engine.analytics.diversification import (
            compare_diversification,
        )

        return compare_diversification(self.result.weights, self.cov_matrix)

    # -- benchmark-relative -------------------------------------------------

    @property
    def benchmark_returns(self) -> pd.Series | None:
        """The benchmark's return stream, or None when none was chosen."""
        return None if self.benchmark is None else self.benchmark.returns

    @property
    def benchmark_label(self) -> str | None:
        return None if self.benchmark is None else self.benchmark.label

    def _benchmark_weights(self) -> pd.Series:
        """The benchmark's weights over this run's universe.

        Raises:
            ValueError: When no benchmark is set, or when the one that is set
                has no positions in the investable universe. Active analytics
                are meaningless without a weight vector, and defaulting to
                equal weights would invent a benchmark nobody chose.
        """
        assets = list(self.result.weights.index)
        weights = self.config.benchmark_weight_map(assets)
        if weights is None and self.benchmark is not None:
            resolved = self.benchmark.weights
            weights = None if resolved is None else resolved.to_dict()
        if not weights:
            raise ValueError(
                "This run has no position-based benchmark, so there are no "
                "active positions to analyze. Choose a benchmark defined by "
                "weights (1/N, a single asset, or a custom vector) — an "
                "external index has no holdings in this universe."
            )
        return pd.Series(weights).reindex(assets).fillna(0.0)

    def active_risk_decomposition(self) -> pd.DataFrame:
        """Euler decomposition of *tracking error*, per asset.

        Where :meth:`risk_decomposition` says where the risk is, this says
        where the risk differs from the benchmark's — the two disagree
        precisely on the large index positions that carry absolute risk and
        no active risk at all.
        """
        from optimization_engine.analytics.active import active_risk_decomposition

        return active_risk_decomposition(
            self.result.weights, self._benchmark_weights(), self.cov_matrix
        )

    def transfer_coefficient(self, method: str = "optimal") -> float:
        """How much of this run's expected returns survived the mandate.

        The alphas are taken as the expected returns *in excess of the
        benchmark's*, and the active weights as the solved book minus the
        benchmark. A low number says the constraints, not the forecasts, are
        determining the portfolio — which is a fixable problem, and a
        different one from having poor forecasts.
        """
        from optimization_engine.analytics.active import transfer_coefficient

        benchmark = self._benchmark_weights()
        mu = self.expected_returns.reindex(self.result.weights.index).fillna(0.0)
        alphas = mu - float(mu @ benchmark)
        return transfer_coefficient(
            alphas, self.result.weights - benchmark, self.cov_matrix, method=method
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
        if benchmark_returns is None and self.benchmark is not None:
            benchmark_returns = self.benchmark.returns
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
            solve = self._window_solver(reestimate_expected_returns)

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

    # -- the full simulation stack -----------------------------------------

    def simulate(
        self,
        spec: BacktestSpec | None = None,
        *,
        weights: pd.Series | None = None,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> RunResult:
        """Replay the solved weights and return the full result bundle.

        Where :meth:`backtest` gives the compact result, this gives everything
        the simulation core produced: per-trade costs, NAV, the target
        schedule, and the spec and result hashes that let one run be compared
        with another. Take a :class:`~optimization_engine.backtest.spec.BacktestSpec`
        when you need an execution lag or a cost model with market impact.

        Still in-sample unless the spec says otherwise — the optimizer saw
        this history.

        Args:
            spec: The run description.
            weights: Targets to replay. Defaults to this run's solution.
            prices: Close prices, needed only to convert share volume into
                traded notional.
            volumes: Traded volume per asset. Optional: without it the impact
                model prices from a fixed participation rate, which is the
                only thing available for an index universe.
        """
        spec = spec or BacktestSpec(periods_per_year=self.config.periods_per_year)
        target = self.result.weights if weights is None else weights
        return run_backtest(
            self.returns, target, spec, prices=prices, volumes=volumes
        )

    def walk_forward_run(
        self,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        spec: BacktestSpec | None = None,
        expanding: bool = False,
        solve: Callable[[pd.DataFrame], pd.Series] | None = None,
        reestimate_expected_returns: bool = True,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> WalkForwardRun:
        """:meth:`walk_forward`, returning the full bundle instead of the digest.

        Same evaluation, same defaults; what differs is that the result
        carries the trade and cost frames and the provenance hashes, which is
        what :meth:`tearsheet` and the sweep need.

        Args:
            prices: Close prices, needed only to convert share volume into
                traded notional.
            volumes: Traded volume per asset. Optional everywhere: without it
                the impact model prices from a fixed participation rate, which
                is the only thing available for an index universe.
        """
        ppy = self.config.periods_per_year
        spec = spec or BacktestSpec(periods_per_year=ppy)
        return walk_forward_run(
            self.returns,
            solve or self._window_solver(reestimate_expected_returns),
            lookback=lookback or max(2 * ppy, 24),
            rebalance_every=rebalance_every or max(ppy // 4, 1),
            spec=spec,
            expanding=expanding,
            prices=prices,
            volumes=volumes,
        )

    def tearsheet(
        self,
        run: RunResult | None = None,
        *,
        riskfree_rate: float | None = None,
        n_trials: int | None = None,
        trial_sharpes: pd.Series | None = None,
        overfitting: Any = None,
    ) -> Tearsheet:
        """The assembled reading of a run, caveats attached.

        Defaults to describing an in-sample replay of this run's own weights,
        which is the cheapest thing to produce and the least informative — the
        tearsheet says so in its caveats rather than leaving it to the reader.
        Pass a walk-forward run for a number worth quoting.
        """
        rf = self.config.optimizer.risk_free_rate if riskfree_rate is None else riskfree_rate
        return build_tearsheet(
            run if run is not None else self.simulate(),
            self.returns,
            riskfree_rate=rf,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            overfitting=overfitting,
        )

    def sweep(
        self,
        sweep: SweepSpec,
        *,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        spec: BacktestSpec | None = None,
        expanding: bool = False,
        progress: Callable[[int, int], None] | None = None,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> SweepResults:
        """Walk-forward every cell of a grid, and count the trials.

        Each cell is evaluated out of sample, because a grid scored in sample
        measures how well each configuration memorized the history rather than
        how well it would have done. The results carry the trial count that
        the deflated Sharpe and the overfitting probability both need.

        Args:
            prices: Close prices, needed only to turn share volume into traded
                notional.
            volumes: Traded volume per asset. Every cell is priced the same
                way, so a grid run with a capacity-aware cost model must be
                handed the same panel the single run was — otherwise the grid
                is cheaper than the run it is supposed to contextualize.
        """
        ppy = self.config.periods_per_year
        run_spec = spec or BacktestSpec(periods_per_year=ppy)
        window = lookback or max(2 * ppy, 24)
        step = rebalance_every or max(ppy // 4, 1)

        def evaluate(cell_config: EngineConfig) -> pd.Series:
            import copy

            cell = copy.deepcopy(cell_config)
            cell.expected_returns = {}

            def solve(window_returns: pd.DataFrame) -> pd.Series:
                return run_engine(
                    window_returns, cell, check_feasibility=False
                ).result.weights

            return walk_forward_run(
                self.returns,
                solve,
                lookback=window,
                rebalance_every=step,
                spec=run_spec,
                expanding=expanding,
                prices=prices,
                volumes=volumes,
            ).returns

        return run_sweep(
            self.config,
            sweep,
            evaluate,
            periods_per_year=ppy,
            progress=progress,
        )

    def _window_solver(
        self, reestimate_expected_returns: bool
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """This run's own config, re-solved on whatever window it is handed."""
        import copy

        base_config = copy.deepcopy(self.config)
        if reestimate_expected_returns:
            # Emptying the vector makes run_engine derive it from the window
            # via expected_returns_method.
            base_config.expected_returns = {}

        def solve(window: pd.DataFrame) -> pd.Series:
            return run_engine(window, base_config, check_feasibility=False).result.weights

        return solve

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

    def performance(
        self,
        riskfree_rate: float | None = None,
        frequency: RebalanceFrequency | None = "monthly",
        transaction_cost_bps: float = 0.0,
        benchmark_returns: pd.Series | None = None,
        rolling_window: int | None = None,
        period_freq: str = "yearly",
        returns_override: pd.Series | None = None,
    ) -> PerformanceReport:
        """Absolute and relative performance of this run, in one object.

        Args:
            riskfree_rate: Annual rate for the ratios. Defaults to the
                optimizer's own, so the report and the solve agree on cash.
            frequency: Rebalancing rule for the replay. ``None`` uses the
                costless constant-weight replay instead, which is the older
                and more optimistic convention.
            transaction_cost_bps: One-way cost charged at each rebalance.
            benchmark_returns: Override the run's own benchmark stream — used
                to report against something other than the one optimized
                against, and by the walk-forward view.
            rolling_window: Window for the rolling frames. Defaults to a year.
            period_freq: ``yearly``, ``quarterly`` or ``monthly`` table.
            returns_override: Use this return stream as the portfolio's
                instead of replaying the weights. This is how a walk-forward
                track record gets the same report as the fitted one.

        Note:
            Unless ``returns_override`` carries an out-of-sample stream, every
            number here is in-sample: the optimizer estimated its inputs from
            these same returns.
        """
        rf = (
            self.config.optimizer.risk_free_rate
            if riskfree_rate is None
            else float(riskfree_rate)
        )
        metadata: dict[str, Any] = {
            "optimizer": self.config.optimizer.name,
            "rebalancing": str(frequency or "none (constant weights)"),
            "transaction_cost_bps": float(transaction_cost_bps),
            "out_of_sample": returns_override is not None,
        }
        if returns_override is not None:
            portfolio = pd.Series(returns_override).dropna()
        elif frequency is None:
            portfolio = self.backtest_returns()["portfolio"]
        else:
            bt = self.backtest(
                frequency=frequency, transaction_cost_bps=transaction_cost_bps
            )
            portfolio = bt.returns
            metadata["annualized_turnover"] = float(bt.annualized_turnover)
            metadata["total_cost"] = float(bt.total_cost)

        bench = benchmark_returns
        if bench is None and self.benchmark is not None:
            bench = self.benchmark.returns
        label = self.benchmark_label if benchmark_returns is None else "Benchmark"

        benchmark_weights = None
        if self.benchmark is not None and self.benchmark.weights is not None:
            benchmark_weights = self.benchmark.weights
        elif self.config.benchmark_weight_map(list(self.result.weights.index)):
            benchmark_weights = pd.Series(
                self.config.benchmark_weight_map(list(self.result.weights.index))
            )

        return performance_report(
            portfolio,
            bench,
            periods_per_year=self.config.periods_per_year,
            riskfree_rate=rf,
            portfolio_weights=self.result.weights,
            benchmark_weights=benchmark_weights,
            benchmark_label=label,
            rolling_window=rolling_window,
            period_freq=period_freq,
            metadata=metadata,
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
            "benchmark": (self.benchmark_label or "—"),
            "benchmark_kind": (
                self.benchmark.spec.kind if self.benchmark is not None else "none"
            ),
            "max_tracking_error": self.config.max_tracking_error,
            "max_active_share": self.config.max_active_share,
            "constraint_layers": (
                "; ".join(
                    f"{lyr.name} ({len(lyr.limits)} buckets"
                    + (f", % of {lyr.parent}" if lyr.is_relative else "")
                    + ")"
                    for lyr in self.constraint_layers
                    if lyr.is_active
                )
                or "—"
            ),
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
    external_returns: pd.DataFrame | pd.Series | None = None,
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
        external_returns: Return series from outside the investable universe,
            needed only when ``config.benchmark`` names an external index.

    Raises:
        ValueError: If ``returns`` is empty or has no columns.
        InfeasibleConstraintsError: When ``raise_on_infeasible`` is set and the
            constraints cannot be satisfied.
    """
    if returns is None or returns.empty:
        raise ValueError("run_engine received an empty returns frame.")
    if returns.shape[1] == 0:
        raise ValueError("run_engine received returns with no asset columns.")

    cov = covariance_from_config(returns, config)
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

    benchmark = resolve_benchmark(config.benchmark, returns, external_returns)
    constraints = constraints_from_config(config, list(returns.columns))
    # Before the feasibility LP, not after: a budget with no benchmark to
    # measure it against is a configuration error, and the LP would otherwise
    # be the first thing to trip over it and report it as a solver problem.
    validate_benchmark_constraints(config.optimizer, constraints)

    feasibility: FeasibilityReport | None = None
    if check_feasibility:
        # Black-Litterman optimizes against its equilibrium posterior, not the
        # configured vector, so a return target has to be checked against the
        # returns the solver will really see.
        feasibility = analyze_feasibility(
            list(returns.columns),
            constraints,
            expected_returns=effective_expected_returns(config, cov, expected_returns),
            cov_matrix=cov,
        )
        if raise_on_infeasible and not feasibility.is_feasible:
            raise InfeasibleConstraintsError(feasibility)

    optimizer = optimizer_factory(
        config, cov, expected_returns=expected_returns, returns=returns
    )
    try:
        result = optimizer.optimize()
    except SolverFailure as exc:
        # A solver that reports "infeasible" has found the same thing the
        # pre-solve analysis did, but says it in solver terms. When the
        # analysis named the culprit, attach its findings rather than leaving
        # the caller with "no allocation satisfies every constraint at once" —
        # the useful sentence is "the 7% return target is above the 6.8% these
        # constraints reach". The exception type is unchanged, so callers
        # catching SolverFailure keep working.
        if feasibility is not None and not feasibility.is_feasible:
            raise SolverFailure(
                exc.status, exc.attempts, detail=feasibility.describe()
            ) from exc
        raise

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
        benchmark=benchmark,
    )
