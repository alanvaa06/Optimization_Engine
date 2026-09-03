"""Efficient frontier construction.

Sweeps either target returns (mean-variance) or risk-aversion levels
(utility-maximization) to trace out a frontier of optimal portfolios.
Solves are dispatched on a thread pool because CVXPY/CLARABEL/ECOS spend
most of their time inside C extensions that release the GIL.

Two properties this module is careful about, because getting them wrong
produces a chart that looks right and is not:

**The sweep range must be reachable.** Sweeping from ``min(μ)`` to ``max(μ)``
is only correct without constraints. Add a 15% per-asset cap and roughly half
the targets become infeasible — the chart silently loses its ends. The range
is therefore derived from two LPs over the actual constraint set.

**Only the upper branch is efficient.** Targets below the global
minimum-variance return are *dominated*: the same volatility buys more return
higher up. The mean-variance sweep imposes its return target as ``μ'w ≥ R*``
(see :class:`~optimization_engine.optimizers.mean_variance.MeanVarianceOptimizer`),
so a target below the minimum-variance return simply returns the
minimum-variance portfolio rather than a dominated one — the lower branch is
not reachable through this sweep at all. ``is_efficient`` is still computed and
reported, because a mean-CVaR sweep minimizes a different risk measure and can
land below the minimum-*variance* return.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from optimization_engine.config import EngineConfig
from optimization_engine.optimizers.factory import constraints_from_config, optimizer_factory

#: Optimizers whose own objective can be swept along a return target.
_SWEEPABLE = {"mean_variance", "cvar"}


@dataclass
class FrontierResult:
    """A traced frontier: summary rows, per-point weights, and key portfolios.

    Attributes:
        summary: One row per swept target. Carries ``expected_return``,
            ``expected_volatility``, ``sharpe_ratio``, ``status``, and
            ``is_efficient`` — False for points below the minimum-variance
            return. The mean-variance sweep cannot produce one (its target is
            an inequality); a mean-CVaR sweep can.
        weights: Assets × targets weight matrix.
        group_weights: Same, aggregated by group. Empty when no groups.
        min_variance: Summary row of the global minimum-variance portfolio.
        tangency: Summary row of the maximum-Sharpe portfolio, when one exists.
        risk_measure: What the x-axis actually is — ``"volatility"`` for the
            mean-variance sweep, ``"CVaR"`` for a mean-CVaR sweep.
        reachable_range: ``(min, max)`` expected return the constraints allow.
        failures: Human-readable reasons for any point that did not solve.
        anchor_failures: ``{"min_variance"|"tangency": reason}`` for any anchor
            portfolio that could not be solved. An anchor that fails leaves its
            field ``None``, which on its own is indistinguishable from "not
            requested"; this says why, and the chart prints it.
    """

    summary: pd.DataFrame
    weights: pd.DataFrame
    group_weights: pd.DataFrame | None = None
    min_variance: pd.Series | None = None
    tangency: pd.Series | None = None
    risk_measure: str = "volatility"
    reachable_range: tuple[float, float] | None = None
    failures: tuple[str, ...] = ()
    anchor_failures: dict[str, str] = field(default_factory=dict)

    @property
    def max_sharpe_index(self) -> int:
        """Positional index of the highest-Sharpe point among solved rows.

        NaN-safe: a frontier where every point failed returns ``-1`` rather
        than silently pointing at row 0, which would put the "max Sharpe"
        marker on an arbitrary portfolio.
        """
        sharpe = self.summary["sharpe_ratio"].values
        if np.all(np.isnan(sharpe)):
            return -1
        return int(np.nanargmax(sharpe))

    @property
    def efficient(self) -> pd.DataFrame:
        """Solved points on the efficient (upper) branch only."""
        df = self.summary
        mask = (df["status"] == "ok") & df.get("is_efficient", True)
        return df[mask]

    @property
    def n_failed(self) -> int:
        """How many frontier points failed to solve.

        A target return can be unreachable under the mandate's constraints, so a
        non-zero count is ordinary — those points are kept as rows with a
        non-``ok`` status rather than dropped.
        """
        return int((self.summary["status"] != "ok").sum())

    def plot_frame(self, efficient_only: bool = True) -> pd.DataFrame:
        """Rows ready to plot, with a positional index the caller can trust.

        Charting code that drops NaN rows and then indexes with a position
        taken from the *undropped* frame highlights the wrong point; this
        returns one frame so both come from the same place.

        Args:
            efficient_only: Keep only the solved points on the efficient branch.

        Returns:
            A frame with a fresh ``RangeIndex``, so a position taken from it
            indexes it correctly.
        """
        df = self.efficient if efficient_only else self.summary[self.summary["status"] == "ok"]
        return df.dropna(subset=["expected_volatility", "expected_return"]).reset_index(drop=True)


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


def _anchor_portfolios(
    base_config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None,
) -> tuple[pd.Series | None, pd.Series | None, dict[str, str]]:
    """Solve the two portfolios every frontier chart should mark.

    The global minimum-variance point is where the efficient branch starts;
    the tangency point is where a risk-free asset would take you. Both are
    solved directly rather than picked off the sweep, so they land on the
    true frontier instead of the nearest grid point.

    Neither is allowed to fail silently. A mandate can make the tangency
    portfolio infeasible while leaving the frontier itself perfectly
    solvable; swallowing that exception drops the marker off the chart and
    the analyst has no way to tell "no tangency portfolio exists here" from
    "the chart forgot to draw it".

    Returns:
        ``(min_variance_row, tangency_row, anchor_failures)``. The failure
        map is keyed by the :class:`FrontierResult` field the anchor lands
        in — ``"min_variance"`` and ``"tangency"`` — and carries the
        exception text for every anchor that did not solve.
    """
    from optimization_engine.optimizers.mean_variance import (
        MaxSharpeOptimizer,
        MinVarianceOptimizer,
    )

    constraints = constraints_from_config(base_config, list(cov_matrix.columns))
    constraints.target_return = None
    constraints.target_volatility = None

    def _row(result, label: str) -> pd.Series:
        return pd.Series(
            {
                "label": label,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
            }
        )

    gmv = tangency = None
    anchor_failures: dict[str, str] = {}
    try:
        gmv_result = MinVarianceOptimizer(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            risk_free_rate=base_config.optimizer.risk_free_rate,
        ).optimize()
        gmv = _row(gmv_result, "Minimum variance")
        gmv["weights"] = gmv_result.weights
    except Exception as exc:
        anchor_failures["min_variance"] = str(exc)

    if expected_returns is not None:
        try:
            tan_result = MaxSharpeOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                constraints=constraints,
                risk_free_rate=base_config.optimizer.risk_free_rate,
            ).optimize()
            tangency = _row(tan_result, "Maximum Sharpe")
            tangency["weights"] = tan_result.weights
        except Exception as exc:
            anchor_failures["tangency"] = str(exc)

    return gmv, tangency, anchor_failures


def _resolve_return_range(
    base_config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series,
    return_range: tuple[float, float] | None,
    efficient_only: bool,
    gmv: pd.Series | None,
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """Pick the sweep endpoints, and report what the constraints can reach.

    Returns ``(sweep_range, reachable_range)``. An explicit ``return_range``
    is honoured verbatim — a caller who asks for an unreachable range wants
    to see it fail, and the tests rely on that.
    """
    from optimization_engine.optimizers.feasibility import reachable_return_range

    constraints = constraints_from_config(base_config, list(cov_matrix.columns))
    constraints.target_return = None
    constraints.target_volatility = None
    reachable = reachable_return_range(
        expected_returns, constraints, list(cov_matrix.columns), cov_matrix=cov_matrix
    )

    if return_range is not None:
        return (float(return_range[0]), float(return_range[1])), reachable

    if reachable is None:
        mu = expected_returns.reindex(cov_matrix.columns).fillna(0.0).values
        return (float(mu.min()), float(mu.max())), None

    lo, hi = reachable
    # Nudge off the exact endpoints: an equality target sitting on the
    # boundary of the feasible set is where solvers report
    # optimal_inaccurate or fail outright.
    span = hi - lo
    pad = max(span * 1e-4, 1e-9)
    lo, hi = lo + pad, hi - pad
    if efficient_only and gmv is not None:
        gmv_return = float(gmv["expected_return"])
        if lo < gmv_return < hi:
            lo = gmv_return
    return (lo, max(hi, lo)), reachable


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
    efficient_only: bool = True,
) -> FrontierResult:
    """Trace the efficient frontier.

    The default ``sweep="return"`` solves a target-return problem at each
    candidate target. ``sweep="risk_aversion"`` instead sweeps the utility
    coefficient λ — useful with optimizers that don't support a hard return
    target, and it never produces an infeasible point.

    Args:
        config: The run's configuration; its constraints bound the sweep.
        cov_matrix: Annualized covariance matrix.
        expected_returns: Annualized expected returns. Falls back to
            ``config.expected_returns``.
        returns: Scenario returns, required for a mean-CVaR frontier.
        target_returns: Explicit targets, bypassing range selection.
        n_points: Resolution of the sweep.
        sweep: ``"return"`` or ``"risk_aversion"``.
        return_range: Explicit ``(lo, hi)``. When omitted the range is derived
            from what the constraints can actually reach.
        n_workers: Thread-pool size. ``None`` uses ``min(8, n_points)``;
            ``1`` or less runs sequentially.
        efficient_only: Start the sweep at the minimum-variance return. The
            mean-variance target is an inequality, so targets below that
            return all resolve to the same minimum-variance portfolio and
            add nothing; a mean-CVaR sweep can genuinely land below it, and
            those points are flagged ``is_efficient=False``.

    Raises:
        ValueError: On an unknown ``sweep``, or a return sweep with no
            expected returns to sweep over.
    """
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)

    base_config = copy.deepcopy(config)
    if base_config.optimizer.name not in _SWEEPABLE:
        base_config.optimizer.name = "mean_variance"
    risk_measure = "CVaR" if base_config.optimizer.name == "cvar" else "volatility"

    gmv, tangency, anchor_failures = _anchor_portfolios(
        base_config, cov_matrix, expected_returns
    )
    reachable: tuple[float, float] | None = None

    if sweep == "return":
        if target_returns is None:
            if expected_returns is None:
                raise ValueError("Cannot sweep returns without expected_returns")
            (lo, hi), reachable = _resolve_return_range(
                base_config, cov_matrix, expected_returns,
                return_range, efficient_only, gmv,
            )
            target_returns = list(np.linspace(lo, hi, n_points))
        else:
            target_returns = list(target_returns)
    elif sweep == "risk_aversion":
        target_returns = list(np.geomspace(0.5, 50.0, n_points))
    else:
        raise ValueError(f"Unknown sweep: {sweep!r}. Use 'return' or 'risk_aversion'.")

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

    gmv_return = float(gmv["expected_return"]) if gmv is not None else None
    # Solver tolerance, not economics, decides whether a point sitting exactly
    # on the minimum-variance return lands a hair below it.
    efficiency_tol = (
        max(1e-6, abs(gmv_return) * 1e-3) if gmv_return is not None else 0.0
    )
    summary_rows: list[dict[str, object]] = []
    weights_rows: list[pd.Series] = []
    failures: list[str] = []
    for target, result, err in rows:
        if result is None:
            failures.append(f"target {target:.4%}: {err}")
            summary_rows.append({
                "target": target,
                "expected_return": np.nan,
                "expected_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "is_efficient": False,
                "status": f"failed: {err}",
            })
            weights_rows.append(
                pd.Series(np.nan, index=cov_matrix.columns, name=target)
            )
        else:
            is_efficient = (
                True if gmv_return is None
                else result.expected_return >= gmv_return - efficiency_tol
            )
            row: dict[str, object] = {
                "target": target,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "is_efficient": is_efficient,
                "status": "ok",
            }
            if risk_measure == "CVaR":
                row["cvar"] = result.extras.get("cvar_sqrt_t_scaled", np.nan)
            summary_rows.append(row)
            weights_rows.append(result.weights.rename(target))

    summary = pd.DataFrame(summary_rows)
    weights_df = pd.concat(weights_rows, axis=1)
    weights_df.columns = summary["target"].values
    return FrontierResult(
        summary=summary,
        weights=weights_df,
        group_weights=_group_weights(weights_df, base_config.groups),
        min_variance=gmv,
        tangency=tangency,
        risk_measure=risk_measure,
        reachable_range=reachable,
        failures=tuple(failures),
        anchor_failures=anchor_failures,
    )
