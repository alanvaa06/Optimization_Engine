"""Efficient frontier construction.

Sweeps either target returns (mean-variance) or risk-aversion levels
(utility-maximization) to trace out a frontier of optimal portfolios.
Solves are dispatched on a thread pool because CVXPY/CLARABEL/ECOS spend
most of their time inside C extensions that release the GIL.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
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


def _solve_one(
    target: float,
    base_config: EngineConfig,
    sweep: str,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None,
    returns: pd.DataFrame | None,
):
    cfg = copy.deepcopy(base_config)
    if sweep == "return":
        cfg.optimizer.target_return = float(target)
        cfg.optimizer.risk_aversion = 1.0
    else:
        cfg.optimizer.target_return = None
        cfg.optimizer.risk_aversion = float(target)
    try:
        result = optimizer_factory(
            cfg, cov_matrix,
            expected_returns=expected_returns,
            returns=returns,
        ).optimize()
        return float(target), result, None
    except Exception as exc:
        return float(target), None, str(exc)


def efficient_frontier(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    target_returns: Iterable[float] | None = None,
    n_points: int = 25,
    sweep: Literal["return", "risk_aversion"] = "return",
    return_range: tuple[float, float] | None = None,
    n_workers: int | None = None,
) -> FrontierResult:
    """Trace the efficient frontier.

    The default ``sweep="return"`` solves a target-return MV problem at
    each candidate target. ``sweep="risk_aversion"`` instead sweeps the
    utility coefficient λ — useful with optimizers that don't support a
    hard return target.

    ``n_workers`` controls the size of the thread pool used to solve
    individual frontier points in parallel. ``None`` defaults to
    ``min(8, n_points)``; ``1`` (or less) runs sequentially.
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
            target_returns = list(np.linspace(lo, hi, n_points))
        else:
            target_returns = list(target_returns)
    elif sweep == "risk_aversion":
        target_returns = list(np.geomspace(0.5, 50.0, n_points))
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    base_config = copy.deepcopy(config)
    if base_config.optimizer.name not in {"mean_variance", "cvar"}:
        base_config.optimizer.name = "mean_variance"

    workers = n_workers if n_workers is not None else min(8, len(target_returns))
    if workers <= 1:
        rows = [
            _solve_one(t, base_config, sweep, cov_matrix, expected_returns, returns)
            for t in target_returns
        ]
    else:
        n = len(target_returns)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(
                _solve_one,
                target_returns,
                [base_config] * n,
                [sweep] * n,
                [cov_matrix] * n,
                [expected_returns] * n,
                [returns] * n,
            ))

    summary_rows: list[dict[str, float]] = []
    weights_rows: list[pd.Series] = []
    for target, result, err in rows:
        if result is None:
            summary_rows.append({
                "target": target,
                "expected_return": np.nan,
                "expected_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "status": f"failed: {err}",
            })
            weights_rows.append(
                pd.Series(np.nan, index=cov_matrix.columns, name=target)
            )
        else:
            summary_rows.append({
                "target": target,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "status": "ok",
            })
            weights_rows.append(result.weights.rename(target))

    summary = pd.DataFrame(summary_rows)
    weights_df = pd.concat(weights_rows, axis=1)
    weights_df.columns = summary["target"].values
    return FrontierResult(
        summary=summary,
        weights=weights_df,
        group_weights=_group_weights(weights_df, base_config.groups),
    )
