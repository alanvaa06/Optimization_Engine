"""Multi-asset portfolio optimization engine."""

from optimization_engine.config import (
    EngineConfig,
    OptimizerSpec,
    load_config,
    save_config,
)
from optimization_engine.data.covariance import (
    covariance_matrix,
    expected_returns_from_history,
)
from optimization_engine.data.loader import (
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.data.yahoo import (
    YahooFinanceError,
    load_prices_yahoo,
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
from optimization_engine.engine import EngineRun, run_engine
from optimization_engine.frontier import FrontierResult, efficient_frontier
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
from optimization_engine.optimizers import (
    BaseOptimizer,
    BlackLittermanOptimizer,
    CVaROptimizer,
    EqualWeightOptimizer,
    HRPOptimizer,
    InverseVolatilityOptimizer,
    MaxDiversificationOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    optimizer_factory,
)

__version__ = "0.2.0"

__all__ = [
    "EngineConfig",
    "OptimizerSpec",
    "load_config",
    "save_config",
    "covariance_matrix",
    "expected_returns_from_history",
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
    "CVaROptimizer",
    "EqualWeightOptimizer",
    "HRPOptimizer",
    "InverseVolatilityOptimizer",
    "MaxDiversificationOptimizer",
    "MaxSharpeOptimizer",
    "MeanVarianceOptimizer",
    "MinVarianceOptimizer",
    "RiskParityOptimizer",
    "optimizer_factory",
    "__version__",
]
