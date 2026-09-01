"""Multi-asset portfolio optimization engine.

The data-ingestion API is reachable two ways. The names most callers touch —
:class:`IngestRequest`, :class:`IngestResult`, :class:`PricePanel` and the
:class:`IngestError` base — are re-exported here. Everything else stays in
:mod:`optimization_engine.ingest`, which is bound as an attribute of this
package::

    from optimization_engine import ingest
    result = ingest.ingest(IngestRequest(identifiers=["SPY"], provider="yahoo"))

The subpackage is not flattened into this namespace on purpose. Its field
constants are named for what they mean *inside a price panel* — ``CLOSE``,
``OPEN``, ``HIGH``, ``LOW``, ``VOLUME`` — and those are far too generic to
own a slot in the top-level namespace of a library that also exports
optimizers, estimators and analytics. The ``ingest`` function keeps its
qualified spelling for the same reason: unqualified, it would shadow the
subpackage that shares its name.
"""

from optimization_engine import ingest
from optimization_engine.analytics.active import (
    FundamentalLawReport,
    InformationCoefficient,
    active_risk_decomposition,
    fundamental_law,
    grinold_alpha,
    implied_breadth,
    information_coefficient,
    optimal_active_risk,
    risk_aversion_from_information_ratio,
    transfer_coefficient,
    value_added,
)
from optimization_engine.analytics.backtest import (
    BacktestResult,
    WalkForwardResult,
    backtest_weights,
    compare_in_and_out_of_sample,
    walk_forward_backtest,
)
from optimization_engine.analytics.diversification import (
    DiversificationReport,
    compare_diversification,
    diversification_distribution,
    effective_number_of_bets,
    minimum_torsion,
)
from optimization_engine.analytics.report import (
    PerformanceReport,
    compare_performance,
    performance_report,
    period_returns,
    rolling_relative,
)
from optimization_engine.analytics.selection import (
    DeflatedSharpe,
    OverfittingReport,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
)
from optimization_engine.backtest import (
    BacktestSpec,
    CostSpec,
    HoldoutOutcome,
    PositionStats,
    RunResult,
    SweepResults,
    SweepSpec,
    TcaPanel,
    Tearsheet,
    WalkForwardRun,
    build_tearsheet,
    compute_position_stats,
    compute_tca,
    final_holdout_run,
    gate_returns,
    run_backtest,
    run_sweep,
    sweep_from_optimizers,
    walk_forward_run,
)
from optimization_engine.benchmark import (
    BenchmarkSpec,
    ResolvedBenchmark,
    resolve_benchmark,
)
from optimization_engine.config import (
    EngineConfig,
    OptimizerSpec,
    load_config,
    save_config,
)
from optimization_engine.constraints import (
    ConstraintLayer,
    currency_layer,
    effective_layers,
    layer_exposures,
    layer_from_mapping,
)
from optimization_engine.data.covariance import (
    CovarianceDiagnostics,
    covariance_diagnostics,
    covariance_from_config,
    covariance_matrix,
    expected_returns_from_history,
    james_stein_shrinkage,
    nearest_psd,
)
from optimization_engine.data.denoise import (
    DenoiseReport,
    denoise_covariance,
    detone_correlation,
    fit_marchenko_pastur,
    marchenko_pastur_pdf,
)
from optimization_engine.data.fred import (
    FREDError,
    load_fred_series,
    load_risk_free_rate,
)
from optimization_engine.data.fx import (
    FXError,
    convert_prices_to_base,
    fetch_fx_to_base,
    supported_currencies,
)
from optimization_engine.data.loader import (
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.data.quality import (
    DataQualityReport,
    align_panel,
    analyze_prices,
)
from optimization_engine.data.yahoo import (
    YahooFinanceError,
    load_prices_yahoo,
)
from optimization_engine.engine import EngineRun, run_engine
from optimization_engine.frontier import FrontierResult, efficient_frontier
from optimization_engine.ingest import (
    IngestError,
    IngestRequest,
    IngestResult,
    PricePanel,
)
from optimization_engine.optimizers import (
    BaseOptimizer,
    BlackLittermanOptimizer,
    CDaROptimizer,
    CVaROptimizer,
    EqualWeightOptimizer,
    HERCOptimizer,
    HRPOptimizer,
    InverseVolatilityOptimizer,
    MaxDiversificationOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    NCOOptimizer,
    RiskParityOptimizer,
    optimizer_factory,
)
from optimization_engine.optimizers.black_litterman import View
from optimization_engine.optimizers.diagnostics import (
    PortfolioDiagnostics,
    diversification_ratio,
    effective_n,
    herfindahl_index,
    portfolio_diagnostics,
    risk_decomposition,
)
from optimization_engine.optimizers.feasibility import (
    FeasibilityReport,
    InfeasibleConstraintsError,
    analyze_feasibility,
)
from optimization_engine.resampling import (
    FrontierUncertainty,
    MCOSResult,
    bootstrap_frontier,
    monte_carlo_optimization_selection,
    resample_returns,
    resampled_efficient_frontier,
)
from optimization_engine.scenarios import (
    Scenario,
    config_signature,
    delete_scenario,
    dump_scenarios_yaml,
    load_scenarios,
    load_scenarios_yaml,
    rename_scenario,
    save_scenarios,
    scenario_from_dict,
    scenario_signature,
    scenario_to_dict,
)

__version__ = "0.5.2"

__all__ = [
    "BenchmarkSpec",
    "PerformanceReport",
    "ResolvedBenchmark",
    "compare_performance",
    "performance_report",
    "period_returns",
    "resolve_benchmark",
    "rolling_relative",
    "ConstraintLayer",
    "EngineConfig",
    "OptimizerSpec",
    "currency_layer",
    "effective_layers",
    "layer_exposures",
    "layer_from_mapping",
    "load_config",
    "save_config",
    "covariance_matrix",
    "covariance_from_config",
    "covariance_diagnostics",
    "CovarianceDiagnostics",
    "DenoiseReport",
    "denoise_covariance",
    "detone_correlation",
    "fit_marchenko_pastur",
    "marchenko_pastur_pdf",
    "expected_returns_from_history",
    "james_stein_shrinkage",
    "nearest_psd",
    "DataQualityReport",
    "align_panel",
    "analyze_prices",
    "BacktestResult",
    "WalkForwardResult",
    "backtest_weights",
    "compare_in_and_out_of_sample",
    "walk_forward_backtest",
    "PortfolioDiagnostics",
    "portfolio_diagnostics",
    "risk_decomposition",
    "diversification_ratio",
    "effective_n",
    "herfindahl_index",
    "FeasibilityReport",
    "InfeasibleConstraintsError",
    "analyze_feasibility",
    "View",
    "load_prices",
    "prices_to_returns",
    "sample_dataset",
    "YahooFinanceError",
    "load_prices_yahoo",
    "FREDError",
    "load_fred_series",
    "load_risk_free_rate",
    "FXError",
    "convert_prices_to_base",
    "fetch_fx_to_base",
    "supported_currencies",
    "EngineRun",
    "run_engine",
    "FrontierResult",
    "efficient_frontier",
    "ingest",
    "IngestError",
    "IngestRequest",
    "IngestResult",
    "PricePanel",
    "FrontierUncertainty",
    "MCOSResult",
    "bootstrap_frontier",
    "monte_carlo_optimization_selection",
    "resample_returns",
    "resampled_efficient_frontier",
    "DeflatedSharpe",
    "OverfittingReport",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "minimum_track_record_length",
    "probability_of_backtest_overfitting",
    "FundamentalLawReport",
    "InformationCoefficient",
    "active_risk_decomposition",
    "fundamental_law",
    "grinold_alpha",
    "implied_breadth",
    "information_coefficient",
    "optimal_active_risk",
    "risk_aversion_from_information_ratio",
    "transfer_coefficient",
    "value_added",
    "DiversificationReport",
    "compare_diversification",
    "diversification_distribution",
    "effective_number_of_bets",
    "minimum_torsion",
    "Scenario",
    "config_signature",
    "delete_scenario",
    "dump_scenarios_yaml",
    "load_scenarios",
    "load_scenarios_yaml",
    "rename_scenario",
    "save_scenarios",
    "scenario_from_dict",
    "scenario_signature",
    "scenario_to_dict",
    "BaseOptimizer",
    "BlackLittermanOptimizer",
    "CDaROptimizer",
    "CVaROptimizer",
    "EqualWeightOptimizer",
    "HERCOptimizer",
    "HRPOptimizer",
    "InverseVolatilityOptimizer",
    "MaxDiversificationOptimizer",
    "MaxSharpeOptimizer",
    "MeanVarianceOptimizer",
    "MinVarianceOptimizer",
    "NCOOptimizer",
    "RiskParityOptimizer",
    "optimizer_factory",
    "BacktestSpec",
    "CostSpec",
    "HoldoutOutcome",
    "PositionStats",
    "RunResult",
    "SweepResults",
    "SweepSpec",
    "TcaPanel",
    "Tearsheet",
    "WalkForwardRun",
    "build_tearsheet",
    "compute_position_stats",
    "compute_tca",
    "final_holdout_run",
    "gate_returns",
    "run_backtest",
    "run_sweep",
    "sweep_from_optimizers",
    "walk_forward_run",
    "__version__",
]
