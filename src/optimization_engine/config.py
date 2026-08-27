"""Configuration objects for the optimization engine.

The engine is intentionally driven by data. All asset metadata —
expected returns, group mappings, weight bounds, optimizer choice —
lives in a `EngineConfig` object that can be loaded from YAML/JSON,
mutated programmatically, or built from a UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from optimization_engine.benchmark import BenchmarkSpec


@dataclass
class OptimizerSpec:
    """Specification of which optimizer to run and its hyperparameters.

    Attributes:
        name: Registered optimizer name (see ``available_optimizers()``).
        target_return: Hard annualized expected-return target.
        target_volatility: Hard annualized volatility cap.
        risk_free_rate: Annual risk-free rate used for Sharpe and for the
            tangency and CAPM/Black-Litterman calculations.
        risk_aversion: Utility coefficient λ in ``μ'w − λ·w'Σw``.
        cvar_alpha: Tail probability for mean-CVaR (``0.05`` ⇒ 95% CVaR).
        risk_budget: ``asset -> target share of total risk`` for risk parity.
        bl_views: Black-Litterman views. Either ``{asset: return}`` for
            absolute views, or a list of ``{"weights": {...},
            "expected_return": x, "confidence": y, "label": z}`` mappings for
            relative (spread) views.
        bl_view_confidences: Ω diagonal for the mapping form of ``bl_views``.
        bl_tau: Prior-uncertainty scale in the Black-Litterman posterior.
        bl_market_caps: Equilibrium market portfolio for reverse optimization.
        bl_market_return: Observed market return, used to imply δ.
        bl_calibrate_risk_aversion: Imply δ from the market Sharpe ratio
            instead of using ``risk_aversion`` verbatim.
        hrp_linkage: Hierarchical clustering linkage rule for HRP.
        cdar_alpha: Tail probability for mean-CDaR (``0.05`` ⇒ the worst 5%
            of the drawdown path).
        cluster_linkage: Linkage rule for the clustering methods that
            *partition* the tree (HERC, NCO) rather than merely order it.
        n_clusters: Force a cluster count for HERC/NCO. ``None`` selects it
            by maximizing the silhouette t-statistic.
        max_clusters: Upper bound for that search.
        herc_risk_measure: Cluster-level risk measure for HERC.
        nco_objective: Objective solved at both NCO layers.
        nco_detone_for_clustering: Strip the market eigenvector before
            clustering in NCO.
    """

    name: str = "mean_variance"
    target_return: float | None = None
    target_volatility: float | None = None
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    cvar_alpha: float = 0.05
    risk_budget: dict[str, float] | None = None
    bl_views: dict[str, float] | None = None
    bl_view_confidences: dict[str, float] | None = None
    bl_tau: float = 0.05
    bl_market_caps: dict[str, float] | None = None
    hrp_linkage: Literal["single", "average", "complete", "ward"] = "single"
    cdar_alpha: float = 0.05
    cluster_linkage: Literal["single", "average", "complete", "ward"] = "ward"
    n_clusters: int | None = None
    max_clusters: int | None = None
    herc_risk_measure: Literal[
        "variance", "std", "cvar", "cdar", "equal_weight"
    ] = "variance"
    nco_objective: Literal["min_variance", "max_sharpe"] = "min_variance"
    nco_detone_for_clustering: bool = True
    bl_market_return: float | None = None
    bl_calibrate_risk_aversion: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EngineConfig:
    """Complete engine configuration.

    Attributes:
        expected_returns: Expected (annual) return per asset.
        bounds: Min/max weight per asset, as ``[lo, hi]`` pairs.
        groups: Optional ``asset -> group`` mapping (e.g. asset class).
        group_bounds: Optional ``group -> [lo, hi]`` pairs.
        periods_per_year: Number of return observations per year.
        covariance_method: ``sample``, ``ledoit_wolf``, ``oas``,
            ``shrink`` (Ledoit-Wolf via riskfolio when available),
            ``ewma``, ``semi``, or ``denoised`` (sample covariance filtered
            through the Marchenko-Pastur eigenvalue cutoff).
        ewma_lambda: Decay used when ``covariance_method == "ewma"``.
        denoise: Apply the Marchenko-Pastur eigenvalue filter to the
            covariance estimate (López de Prado, 2020). Composable with any
            estimator; implied by ``covariance_method="denoised"``.
        denoise_method: ``constant_residual`` or ``targeted_shrinkage``.
        denoise_alpha: Noise-block shrinkage retained under
            ``targeted_shrinkage``.
        detone: Number of leading eigenvectors (the market component) to
            remove after denoising. Non-zero makes the covariance singular —
            appropriate for the clustering methods, not for solves that
            invert it.
        expected_returns_method: How to seed expected returns when
            ``expected_returns`` is empty: ``historical_mean``, ``ema``,
            or ``capm``.
        ema_span: Span for the ``ema`` method.
        market_return: Optional CAPM market return (defaults to estimated).
        market_weights: Optional CAPM market portfolio (defaults to equal).
        optimizer: ``OptimizerSpec`` describing the run.
        benchmark: How the benchmark is defined — 1/N, a single asset, an
            explicit policy vector, or an external index. It drives both the
            relative performance report and the benchmark-relative
            constraints below, so the report and the solve cannot disagree
            about what the benchmark is.
        benchmark_weights: Explicit benchmark weight vector. Overrides
            whatever ``benchmark`` would resolve to; kept because a caller
            may have the vector without wanting to describe how it was built.
        max_tracking_error: Cap on annualized active risk versus the
            benchmark, ``√((w−b)'Σ(w−b))``. Imposed inside the solve by the
            mean-variance family, mean-CVaR/CDaR and active mean-variance;
            reported but not enforced by the projection-based methods.
        max_active_share: Cap on ``½·Σ|w_i − b_i|``. Binds on positions
            rather than on realized risk, so it holds in a calm market where
            a tracking-error budget silently permits an unrecognizable
            portfolio.
        long_only: Forbid short positions. When False, per-asset minimum
            weights may go negative.
        fully_invested: Require ``sum(w) == 1``.
        leverage: Cap on gross exposure ``Σ|w_i|``. Only meaningful once
            ``long_only`` is off — a long-only, fully-invested book always
            has gross exposure of exactly 1.
        previous_weights: The portfolio being traded from. Needed for the
            turnover budget and for turnover reporting.
        turnover_limit: Cap on ``Σ|w_i − w_prev,i|``. Honoured by the
            mean-variance family and mean-CVaR; the homogeneous solves
            (max-Sharpe, max-diversification, risk parity) warn instead.
    """

    expected_returns: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, list[float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    group_bounds: dict[str, list[float]] = field(default_factory=dict)
    currencies: dict[str, str] = field(default_factory=dict)
    base_currency: str = "USD"
    periods_per_year: int = 252
    covariance_method: str = "ledoit_wolf"
    ewma_lambda: float = 0.94
    denoise: bool = False
    denoise_method: str = "constant_residual"
    denoise_alpha: float = 0.0
    detone: int = 0
    expected_returns_method: Literal[
        "historical_mean", "ema", "capm", "shrunk_mean"
    ] = "historical_mean"
    ema_span: int = 180
    market_return: float | None = None
    market_weights: dict[str, float] | None = None
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    benchmark: BenchmarkSpec = field(default_factory=BenchmarkSpec)
    benchmark_weights: dict[str, float] | None = None
    max_tracking_error: float | None = None
    max_active_share: float | None = None
    long_only: bool = True
    fully_invested: bool = True
    leverage: float | None = None
    previous_weights: dict[str, float] | None = None
    turnover_limit: float | None = None

    @property
    def assets(self) -> list[str]:
        return list(self.expected_returns.keys())

    def benchmark_weight_map(
        self, assets: list[str] | None = None
    ) -> dict[str, float] | None:
        """The benchmark's weights, or ``None`` when it has none.

        An explicit ``benchmark_weights`` vector wins over the spec: a caller
        that supplied the numbers directly meant those numbers. Otherwise the
        spec is expanded over ``assets`` — which matters for the rule-based
        kinds, since 1/N over ten assets is a different portfolio from 1/N
        over twelve. An external-index benchmark has no weights in the
        investable universe and correctly returns ``None``.
        """
        if self.benchmark_weights:
            return {str(k): float(v) for k, v in self.benchmark_weights.items()}
        universe = list(assets) if assets else self.assets
        if not universe or not self.benchmark.has_weights:
            return None
        weights = self.benchmark.weight_vector(universe)
        return None if weights is None else {str(k): float(v) for k, v in weights.items()}

    def get_bounds(self, asset: str, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        if asset in self.bounds:
            lo, hi = self.bounds[asset]
            return float(lo), float(hi)
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_returns": dict(self.expected_returns),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "groups": dict(self.groups),
            "group_bounds": {k: list(v) for k, v in self.group_bounds.items()},
            "currencies": dict(self.currencies),
            "base_currency": self.base_currency,
            "periods_per_year": self.periods_per_year,
            "covariance_method": self.covariance_method,
            "ewma_lambda": self.ewma_lambda,
            "denoise": self.denoise,
            "denoise_method": self.denoise_method,
            "denoise_alpha": self.denoise_alpha,
            "detone": self.detone,
            "expected_returns_method": self.expected_returns_method,
            "ema_span": self.ema_span,
            "market_return": self.market_return,
            "market_weights": (dict(self.market_weights) if self.market_weights else None),
            "optimizer": self.optimizer.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "benchmark_weights": self.benchmark_weights,
            "max_tracking_error": self.max_tracking_error,
            "max_active_share": self.max_active_share,
            "long_only": self.long_only,
            "fully_invested": self.fully_invested,
            "leverage": self.leverage,
            "previous_weights": (
                dict(self.previous_weights) if self.previous_weights else None
            ),
            "turnover_limit": self.turnover_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        opt_raw = data.get("optimizer") or {}
        if isinstance(opt_raw, str):
            opt_raw = {"name": opt_raw}
        return cls(
            expected_returns=dict(data.get("expected_returns") or {}),
            bounds={k: list(v) for k, v in (data.get("bounds") or {}).items()},
            groups=dict(data.get("groups") or {}),
            group_bounds={k: list(v) for k, v in (data.get("group_bounds") or {}).items()},
            currencies=dict(data.get("currencies") or {}),
            base_currency=str(data.get("base_currency", "USD")).upper(),
            periods_per_year=int(data.get("periods_per_year", 252)),
            covariance_method=str(data.get("covariance_method", "ledoit_wolf")),
            ewma_lambda=float(data.get("ewma_lambda", 0.94)),
            denoise=bool(data.get("denoise", False)),
            denoise_method=str(data.get("denoise_method", "constant_residual")),
            denoise_alpha=float(data.get("denoise_alpha", 0.0)),
            detone=int(data.get("detone", 0) or 0),
            expected_returns_method=str(
                data.get("expected_returns_method", "historical_mean")
            ),
            ema_span=int(data.get("ema_span", 180)),
            market_return=(
                float(data["market_return"])
                if data.get("market_return") is not None else None
            ),
            market_weights=(
                dict(data["market_weights"])
                if data.get("market_weights") else None
            ),
            optimizer=OptimizerSpec(**opt_raw),
            benchmark=BenchmarkSpec.from_dict(data.get("benchmark")),
            benchmark_weights=data.get("benchmark_weights"),
            max_tracking_error=(
                float(data["max_tracking_error"])
                if data.get("max_tracking_error") is not None
                else None
            ),
            max_active_share=(
                float(data["max_active_share"])
                if data.get("max_active_share") is not None
                else None
            ),
            long_only=bool(data.get("long_only", True)),
            fully_invested=bool(data.get("fully_invested", True)),
            leverage=(
                float(data["leverage"]) if data.get("leverage") is not None else None
            ),
            previous_weights=(
                dict(data["previous_weights"]) if data.get("previous_weights") else None
            ),
            turnover_limit=(
                float(data["turnover_limit"])
                if data.get("turnover_limit") is not None
                else None
            ),
        )


def load_config(path: str | Path) -> EngineConfig:
    """Load an `EngineConfig` from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    return EngineConfig.from_dict(data)


def save_config(config: EngineConfig, path: str | Path) -> None:
    """Persist an `EngineConfig` to YAML or JSON depending on extension."""
    p = Path(path)
    data = config.to_dict()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
