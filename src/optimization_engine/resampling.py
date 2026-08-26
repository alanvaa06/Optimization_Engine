"""Estimation-error analysis by resampling.

The engine spends a lot of effort telling an analyst that expected returns
and covariances are noisy — condition numbers, T/N ratios, walk-forward
degradation — and then draws the efficient frontier as a single crisp line.
That line is a point estimate of a curve, and on realistic sample sizes it
moves a great deal.

Two standard remedies, both here:

* :func:`bootstrap_frontier` resamples the return history, re-estimates, and
  re-traces the frontier on each draw. The spread of the resulting curves is
  the frontier's own confidence band.
* :func:`resampled_efficient_frontier` implements Michaud-style resampling:
  average the *weights* across draws at each rank of the frontier. The result
  is markedly more diversified and more stable out of sample than the
  point-estimate frontier, because it stops the optimizer from acting on
  differences between assets that the sample cannot resolve.

Both are expensive relative to a single solve — cost scales linearly in the
number of draws — so the draw count is an explicit argument rather than a
hidden default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from optimization_engine.config import EngineConfig
from optimization_engine.data.covariance import covariance_matrix, expected_returns_from_history
from optimization_engine.frontier import FrontierResult, efficient_frontier

BootstrapMethod = Literal["iid", "block", "parametric"]


def resample_returns(
    returns: pd.DataFrame,
    method: BootstrapMethod = "block",
    block_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw one resampled return history.

    Args:
        returns: The observed history.
        method:
            ``"iid"`` — draw rows with replacement. Simple, but it destroys
            any autocorrelation and volatility clustering in the data.
            ``"block"`` — draw contiguous blocks with replacement, preserving
            short-horizon dependence. The right default for financial returns.
            ``"parametric"`` — draw from a multivariate normal fitted to the
            sample. Smooth, but it imposes normality the data rarely has.
        block_size: Block length for ``"block"``. Defaults to
            ``ceil(T ** (1/3))``, the usual rule of thumb.
        rng: Seeded generator, for reproducibility.
    """
    rng = rng or np.random.default_rng()
    n_obs = len(returns)
    if n_obs < 2:
        raise ValueError("Need at least 2 observations to resample.")

    if method == "iid":
        idx = rng.integers(0, n_obs, size=n_obs)
        return returns.iloc[idx].reset_index(drop=True)

    if method == "block":
        size = block_size or max(2, int(np.ceil(n_obs ** (1 / 3))))
        size = min(size, n_obs)
        n_blocks = int(np.ceil(n_obs / size))
        starts = rng.integers(0, n_obs - size + 1, size=n_blocks)
        rows = np.concatenate([np.arange(s, s + size) for s in starts])[:n_obs]
        return returns.iloc[rows].reset_index(drop=True)

    if method == "parametric":
        mu = returns.mean().values
        cov = np.cov(returns.values, rowvar=False)
        draws = rng.multivariate_normal(mu, cov, size=n_obs)
        return pd.DataFrame(draws, columns=returns.columns)

    raise ValueError(
        f"Unknown resampling method {method!r}. Use 'iid', 'block' or 'parametric'."
    )


@dataclass
class FrontierUncertainty:
    """Distribution of efficient frontiers across resampled histories.

    Attributes:
        volatility: Volatility grid the curves are evaluated on.
        quantiles: Expected return at each grid point, per quantile.
        point_estimate: The frontier traced on the observed sample.
        n_draws: Successful resamples.
        n_failed: Draws whose frontier could not be traced.
        weight_dispersion: Per-asset standard deviation of the weights across
            draws, at the middle of the frontier. Large values name the
            positions the sample cannot pin down.
    """

    volatility: np.ndarray
    quantiles: pd.DataFrame
    point_estimate: FrontierResult
    n_draws: int
    n_failed: int = 0
    weight_dispersion: pd.Series = field(default_factory=pd.Series)

    def band_width(self, quantile_low: float = 0.05, quantile_high: float = 0.95) -> pd.Series:
        """Height of the confidence band at each volatility level.

        This is the number to quote when someone asks how much to trust the
        frontier: "at 10% volatility, expected return is somewhere in a
        4-percentage-point range."
        """
        lo = self.quantiles[f"q{int(quantile_low * 100):02d}"]
        hi = self.quantiles[f"q{int(quantile_high * 100):02d}"]
        return (hi - lo).rename("band_width")

    def summary(self) -> str:
        width = self.band_width()
        if width.empty or not np.isfinite(width).any():
            return "Frontier uncertainty could not be estimated."
        mid = float(np.nanmedian(width.values))
        return (
            f"Across {self.n_draws} resampled histories, the frontier's "
            f"expected return spans a {mid:.2%} band at a typical risk level. "
            "Differences smaller than that are not distinguishable from "
            "estimation noise."
        )


def bootstrap_frontier(
    returns: pd.DataFrame,
    config: EngineConfig,
    n_draws: int = 50,
    n_points: int = 15,
    method: BootstrapMethod = "block",
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    seed: int | None = 0,
    reestimate_expected_returns: bool = True,
) -> FrontierUncertainty:
    """Trace the efficient frontier on many resampled histories.

    Each draw re-estimates the covariance (and, by default, the expected
    returns) from the resampled data, so the band reflects estimation error
    in *both* inputs rather than only in the one the caller happened to fix.

    Args:
        returns: Observed return history.
        config: The run's configuration; its constraints apply to every draw.
        n_draws: Number of resampled histories. Cost is linear in this.
        n_points: Frontier resolution per draw.
        method: See :func:`resample_returns`.
        quantiles: Quantiles of the return distribution to report.
        seed: Base seed, for reproducibility.
        reestimate_expected_returns: Re-estimate μ on each draw. Set False to
            hold the configured expected returns fixed and isolate covariance
            uncertainty.

    Raises:
        ValueError: If every draw fails, or ``n_draws`` is below 2.
    """
    if n_draws < 2:
        raise ValueError(f"Need at least 2 draws to describe a spread; got {n_draws}.")

    import copy

    point_estimate = efficient_frontier(
        config,
        covariance_matrix(
            returns,
            method=config.covariance_method,
            periods_per_year=config.periods_per_year,
            ewma_lambda=config.ewma_lambda,
        ),
        expected_returns=(
            pd.Series(config.expected_returns) if config.expected_returns else None
        ),
        returns=returns,
        n_points=n_points,
    )

    rng = np.random.default_rng(seed)
    curves: list[pd.Series] = []
    mid_weights: list[pd.Series] = []
    n_failed = 0

    for _ in range(n_draws):
        sample = resample_returns(returns, method=method, rng=rng)
        try:
            draw_config = copy.deepcopy(config)
            cov = covariance_matrix(
                sample,
                method=config.covariance_method,
                periods_per_year=config.periods_per_year,
                ewma_lambda=config.ewma_lambda,
            )
            if reestimate_expected_returns:
                mu = expected_returns_from_history(
                    sample,
                    method=(
                        "mean"
                        if config.expected_returns_method == "historical_mean"
                        else config.expected_returns_method
                    ),
                    periods_per_year=config.periods_per_year,
                    span=config.ema_span,
                    risk_free_rate=config.optimizer.risk_free_rate,
                    cov_matrix=cov,
                )
                draw_config.expected_returns = mu.to_dict()
            else:
                mu = pd.Series(config.expected_returns)

            frontier = efficient_frontier(
                draw_config, cov, expected_returns=mu, returns=sample,
                n_points=n_points, n_workers=1,
            )
            solved = frontier.efficient.dropna(
                subset=["expected_volatility", "expected_return"]
            )
            if len(solved) < 2:
                n_failed += 1
                continue
            curves.append(
                pd.Series(
                    solved["expected_return"].values,
                    index=solved["expected_volatility"].values,
                )
            )
            mid = frontier.weights.iloc[:, len(frontier.weights.columns) // 2]
            mid_weights.append(mid)
        except Exception:
            n_failed += 1
            continue

    if not curves:
        raise ValueError(
            f"Every one of the {n_draws} resampled frontiers failed to solve. "
            "The constraints may be feasible only on the observed sample."
        )

    # Evaluate every curve on a shared volatility grid so the quantiles
    # compare like with like. Each curve is monotone in volatility along the
    # efficient branch, so linear interpolation is well behaved.
    lo = max(float(c.index.min()) for c in curves)
    hi = min(float(c.index.max()) for c in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        grid = np.array([float(np.median([c.index.min() for c in curves]))])
    else:
        grid = np.linspace(lo, hi, n_points)

    stacked = np.vstack(
        [np.interp(grid, c.index.values, c.values) for c in curves]
    )
    quantile_frame = pd.DataFrame(
        {
            f"q{int(q * 100):02d}": np.quantile(stacked, q, axis=0)
            for q in quantiles
        },
        index=grid,
    )
    quantile_frame.index.name = "expected_volatility"

    dispersion = (
        pd.concat(mid_weights, axis=1).std(axis=1).sort_values(ascending=False)
        if mid_weights
        else pd.Series(dtype=float)
    )

    return FrontierUncertainty(
        volatility=grid,
        quantiles=quantile_frame,
        point_estimate=point_estimate,
        n_draws=len(curves),
        n_failed=n_failed,
        weight_dispersion=dispersion,
    )


def resampled_efficient_frontier(
    returns: pd.DataFrame,
    config: EngineConfig,
    n_draws: int = 50,
    n_points: int = 15,
    method: BootstrapMethod = "block",
    seed: int | None = 0,
) -> pd.DataFrame:
    """Michaud-style resampled frontier: average weights at each frontier rank.

    Rather than optimizing once on the point estimate, each draw is optimized
    and the weights are averaged across draws at the same *rank* along the
    frontier. Averaging weights (not inputs) is what makes the result robust:
    an asset the optimizer loads on only in some draws ends up with a
    moderate weight instead of a corner.

    Returns:
        Assets × rank frame of averaged weights. Each column sums to 1.

    Raises:
        ValueError: If no draw produces a usable frontier.
    """
    import copy

    rng = np.random.default_rng(seed)
    stacks: list[pd.DataFrame] = []

    for _ in range(n_draws):
        sample = resample_returns(returns, method=method, rng=rng)
        try:
            draw_config = copy.deepcopy(config)
            cov = covariance_matrix(
                sample,
                method=config.covariance_method,
                periods_per_year=config.periods_per_year,
                ewma_lambda=config.ewma_lambda,
            )
            mu = expected_returns_from_history(
                sample,
                method=(
                    "mean"
                    if config.expected_returns_method == "historical_mean"
                    else config.expected_returns_method
                ),
                periods_per_year=config.periods_per_year,
                span=config.ema_span,
                risk_free_rate=config.optimizer.risk_free_rate,
                cov_matrix=cov,
            )
            draw_config.expected_returns = mu.to_dict()
            frontier = efficient_frontier(
                draw_config, cov, expected_returns=mu, returns=sample,
                n_points=n_points, n_workers=1,
            )
            weights = frontier.weights
            ok = frontier.summary["status"].values == "ok"
            if ok.sum() < 2:
                continue
            usable = weights.loc[:, ok]
            usable.columns = range(usable.shape[1])
            stacks.append(usable)
        except Exception:
            continue

    if not stacks:
        raise ValueError(
            "No resampled frontier could be traced; nothing to average."
        )

    # Ranks present in every draw, so each averaged column is built from the
    # same number of observations.
    common = min(s.shape[1] for s in stacks)
    averaged = sum(s.iloc[:, :common] for s in stacks) / len(stacks)
    averaged.columns = [f"rank_{i}" for i in range(common)]
    return averaged
