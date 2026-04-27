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
from optimization_engine.engine import EngineRun, run_engine
from optimization_engine.frontier import FrontierResult, efficient_frontier
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
    "EngineRun",
    "run_engine",
    "FrontierResult",
    "efficient_frontier",
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
