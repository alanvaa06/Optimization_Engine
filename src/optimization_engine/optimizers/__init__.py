"""Portfolio optimizers."""


class ConfigurationError(ValueError):
    """The supplied EngineConfig is missing inputs the chosen optimizer requires."""


from optimization_engine.optimizers.base import (  # noqa: E402
    BaseOptimizer,
    OptimizationResult,
    PortfolioConstraints,
)
from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer  # noqa: E402
from optimization_engine.optimizers.cvar import CVaROptimizer  # noqa: E402
from optimization_engine.optimizers.factory import optimizer_factory  # noqa: E402
from optimization_engine.optimizers.hrp import HRPOptimizer  # noqa: E402
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer  # noqa: E402
from optimization_engine.optimizers.mean_variance import (  # noqa: E402
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
)
from optimization_engine.optimizers.naive import (  # noqa: E402
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from optimization_engine.optimizers.requirements import (  # noqa: E402
    MethodRequirements,
    requirements_for,
)
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer  # noqa: E402

__all__ = [
    "BaseOptimizer",
    "BlackLittermanOptimizer",
    "ConfigurationError",
    "CVaROptimizer",
    "EqualWeightOptimizer",
    "HRPOptimizer",
    "InverseVolatilityOptimizer",
    "MaxDiversificationOptimizer",
    "MaxSharpeOptimizer",
    "MeanVarianceOptimizer",
    "MethodRequirements",
    "MinVarianceOptimizer",
    "OptimizationResult",
    "PortfolioConstraints",
    "RiskParityOptimizer",
    "optimizer_factory",
    "requirements_for",
]
