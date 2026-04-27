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
from typing import Any

import yaml


@dataclass
class OptimizerSpec:
    """Specification of which optimizer to run and its hyperparameters."""

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
            or ``ewma``.
        ewma_lambda: Decay used when ``covariance_method == "ewma"``.
        optimizer: ``OptimizerSpec`` describing the run.
        benchmark_weights: Optional benchmark weight vector for
            comparison.
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
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    benchmark_weights: dict[str, float] | None = None

    @property
    def assets(self) -> list[str]:
        return list(self.expected_returns.keys())

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
            "optimizer": self.optimizer.to_dict(),
            "benchmark_weights": self.benchmark_weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
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
            optimizer=OptimizerSpec(**opt_raw),
            benchmark_weights=data.get("benchmark_weights"),
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
