"""Efficient frontier construction.

Sweeps either target returns (mean-variance) or risk-aversion levels
(utility-maximization) to trace out a frontier of optimal portfolios.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from optimization_engine.config import EngineConfig
from optimization_engine.optimizers.factory import optimizer_factory


@dataclass
class FrontierResult:
    summary: pd.DataFrame
    weights: pd.DataFrame
    group_weights: pd.DataFrame | None = None

    @property
    def max_sharpe_index(self) -> int:
        return int(np.argmax(self.summary["sharpe_ratio"].values))


def _group_weights(weights: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    g = pd.Series(groups)
    expanded = weights.copy()
    expanded["__group__"] = expanded.index.map(g)
    grouped = expanded.groupby("__group__").sum(numeric_only=True)
    return grouped


def efficient_frontier(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    target_returns: Iterable[float] | None = None,
    n_points: int = 25,
    sweep: Literal["return", "risk_aversion"] = "return",
    return_range: tuple[float, float] | None = None,
) -> FrontierResult:
    """Trace the efficient frontier.

    The default ``sweep="return"`` solves a target-return MV problem at
    each candidate target. ``sweep="risk_aversion"`` instead sweeps the
    utility coefficient λ — useful with optimizers that don't support a
    hard return target.
    """
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)

    if sweep == "return":
        if target_returns is None:
            if expected_returns is None:
                raise ValueError("Cannot sweep returns without expected_returns")
            mu = expected_returns.reindex(cov_matrix.columns).fillna(0.0).values
            lo, hi = (return_range
                      if return_range is not None
                      else (float(mu.min()), float(mu.max())))
            target_returns = np.linspace(lo, hi, n_points)
    elif sweep == "risk_aversion":
        target_returns = np.geomspace(0.5, 50.0, n_points)
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    weights_rows: list[pd.Series] = []
    summary_rows: list[dict[str, float]] = []

    base_config = copy.deepcopy(config)
    if base_config.optimizer.name not in {"mean_variance", "cvar"}:
        base_config.optimizer.name = "mean_variance"

    for target in target_returns:
        cfg = copy.deepcopy(base_config)
        if sweep == "return":
            cfg.optimizer.target_return = float(target)
            cfg.optimizer.risk_aversion = 1.0
        else:
            cfg.optimizer.target_return = None
            cfg.optimizer.risk_aversion = float(target)

        try:
            optimizer = optimizer_factory(
                cfg, cov_matrix, expected_returns=expected_returns, returns=returns
            )
            result = optimizer.optimize()
        except Exception as exc:  # infeasible target — skip the point
            summary_rows.append({
                "target": float(target),
                "expected_return": np.nan,
                "expected_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "status": f"failed: {exc}",
            })
            weights_rows.append(
                pd.Series(np.nan, index=cov_matrix.columns, name=float(target))
            )
            continue

        summary_rows.append({
            "target": float(target),
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "status": "ok",
        })
        weights_rows.append(result.weights.rename(float(target)))

    summary = pd.DataFrame(summary_rows)
    weights_df = pd.concat(weights_rows, axis=1)
    weights_df.columns = summary["target"].values
    return FrontierResult(
        summary=summary,
        weights=weights_df,
        group_weights=_group_weights(weights_df, base_config.groups),
    )
