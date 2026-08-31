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
from optimization_engine.data.covariance import (
    covariance_from_config,
    expected_returns_from_history,
)
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

    Returns:
        One drawn history of the same shape as ``returns``.

    Raises:
        ValueError: On an unknown ``method``, or fewer than 2 observations.
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

        Args:
            quantile_low: Lower quantile of the band, as a fraction.
            quantile_high: Upper quantile, as a fraction.

        Returns:
            Band height in annualized-return units, indexed by volatility level.

        Raises:
            KeyError: If either quantile was not computed when the band was
                built.
        """
        lo = self.quantiles[f"q{int(quantile_low * 100):02d}"]
        hi = self.quantiles[f"q{int(quantile_high * 100):02d}"]
        return (hi - lo).rename("band_width")

    def summary(self) -> str:
        """How wide the frontier's confidence band is, in one sentence.

        Returns:
            A statement of the typical band width across the resampled histories,
            and the reminder it implies: differences smaller than that are not
            distinguishable from estimation noise.
        """
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
        covariance_from_config(returns, config),
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
            cov = covariance_from_config(sample, config)
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

    Args:
        returns: The observed history to resample from.
        config: The mandate each draw is optimized under.
        n_draws: How many histories to draw and optimize.
        n_points: How many frontier ranks to trace per draw.
        method: The resampling scheme. See :func:`resample_returns`.
        seed: Seed for the generator, for reproducibility. ``None`` leaves it
            unseeded.

    Returns:
        An assets × rank frame of averaged weights. Each column sums to 1.

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
            cov = covariance_from_config(sample, config)
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


# ---------------------------------------------------------------------------
# Monte Carlo Optimization Selection
# ---------------------------------------------------------------------------


@dataclass
class MCOSResult:
    """Which method/estimator combination recovers the truth most reliably.

    Attributes:
        weight_rmse: Per method, the root-mean-square distance between the
            weights it produced on a noisy sample and the weights the same
            method produces on the ground truth. This is the headline: it is
            estimation error expressed in the only unit that matters, the
            allocation.
        variance_error: Mean absolute error in the portfolio's *true*
            volatility — how wrong the risk turns out to be, not how far the
            weights moved.
        return_error: Mean absolute error in the portfolio's true expected
            return.
        max_weight_drift: Largest single-position error, averaged across
            draws. A method can look fine on RMSE and still move one position
            by 20%.
        n_simulations: Successful draws.
        n_failed: Draws in which a method failed to solve.
        n_observations: Sample length drawn per simulation.
        denoised: Whether the per-draw covariance was denoised.
    """

    weight_rmse: pd.Series
    variance_error: pd.Series
    return_error: pd.Series
    max_weight_drift: pd.Series
    n_simulations: int
    n_failed: int
    n_observations: int
    denoised: bool

    def ranking(self) -> pd.DataFrame:
        """Methods ordered by weight RMSE, best first."""
        frame = pd.DataFrame(
            {
                "weight_rmse": self.weight_rmse,
                "max_weight_drift": self.max_weight_drift,
                "volatility_error": self.variance_error,
                "return_error": self.return_error,
            }
        )
        return frame.sort_values("weight_rmse")

    def describe(self) -> str:
        """Which method recovered the true allocation most reliably, and which least.

        Returns:
            A sentence naming the best and worst methods by weight RMSE over the
            simulated histories, or a statement that no method produced a usable
            allocation on any draw.
        """
        ranked = self.ranking()
        if ranked.empty:
            return "No method produced a usable allocation on any draw."
        best = ranked.index[0]
        worst = ranked.index[-1]
        line = (
            f"Over {self.n_simulations} simulated histories of "
            f"{self.n_observations} observations drawn from the fitted "
            f"distribution, '{best}' recovered the true allocation most "
            f"reliably (weight RMSE {ranked.loc[best, 'weight_rmse']:.2%}) and "
            f"'{worst}' least ({ranked.loc[worst, 'weight_rmse']:.2%})."
        )
        if self.denoised:
            line += " Each draw's covariance was denoised before optimizing."
        if self.n_failed:
            line += f" {self.n_failed} solve(s) failed and were dropped."
        return line


def monte_carlo_optimization_selection(
    returns: pd.DataFrame,
    config: EngineConfig,
    methods: tuple[str, ...] = ("mean_variance", "min_variance", "hrp", "nco"),
    n_simulations: int = 20,
    n_observations: int | None = None,
    denoise: bool | None = None,
    seed: int | None = 0,
) -> MCOSResult:
    """Monte Carlo Optimization Selection (López de Prado, 2019).

    Every backtest in this library asks "would this have worked?". MCOS asks
    the prior question: *given a universe like this one, and a sample this
    long, which method can be trusted to find the right answer at all?*

    The experiment sidesteps the usual problem — that nobody knows the true
    ``μ`` and ``Σ`` — by declaring the sample estimates to be the truth, and
    then testing whether each method can recover the allocation that truth
    implies from a noisy sample of it:

    1. Fit ``μ`` and ``Σ`` on the observed history; call them the truth.
    2. Solve each method against the truth. Those are the target weights.
    3. Draw ``n_simulations`` synthetic histories from that distribution,
       re-estimate ``μ`` and ``Σ`` on each, and re-solve.
    4. Report how far each method's weights land from its own target.

    The comparison is deliberately self-referential — each method is scored
    against *its own* answer on the truth, not against a common one — because
    the question is estimation stability, not which objective is right. A
    method that is wrong but consistent scores well here, and should: it is
    telling you the method is not the source of your uncertainty.

    In López de Prado's original experiment this is what shows NCO cutting
    the weight RMSE of a direct mean-variance solve roughly in half, and it is
    the cleanest way to justify a method choice to someone who does not accept
    "the literature says so".

    Args:
        returns: Observed history, used to fit the ground-truth distribution.
        config: Run configuration. Its constraints apply to every solve.
        methods: Optimizer names to compare.
        n_simulations: Number of synthetic histories. Cost is linear in this
            times the number of methods.
        n_observations: Length of each synthetic history. Defaults to the
            length of ``returns`` — change it to ask "what would another five
            years of data buy me?".
        denoise: Denoise each draw's covariance before optimizing. ``None``
            follows ``config``. Setting it explicitly is how you isolate the
            benefit of denoising from the benefit of the method.
        seed: Base seed, for reproducibility.

    Returns:
        An :class:`MCOSResult`.

    Raises:
        ValueError: If ``n_simulations < 2``, no method is supplied, or every
            simulation fails.
    """
    import copy

    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    if n_simulations < 2:
        raise ValueError(
            f"Need at least 2 simulations to describe a spread; got "
            f"{n_simulations}."
        )
    if not methods:
        raise ValueError("Supply at least one optimizer name to compare.")

    assets = list(returns.columns)
    n_obs = int(n_observations or len(returns))
    use_denoise = config.denoise if denoise is None else bool(denoise)

    def estimate(sample: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Estimate the inputs one simulated history implies.

        Uses the same estimators the config names, so a simulation is scored
        against the pipeline it will actually be run through.

        Args:
            sample: One simulated (or the observed) return history.

        Returns:
            An ``(expected_returns, covariance)`` pair.
        """
        cov = covariance_matrix(
            sample,
            method=config.covariance_method,
            periods_per_year=config.periods_per_year,
            ewma_lambda=config.ewma_lambda,
            denoise=use_denoise,
            denoise_method=config.denoise_method,
            denoise_alpha=config.denoise_alpha,
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
        return mu, cov

    def solve(name: str, mu: pd.Series, cov: pd.DataFrame, sample: pd.DataFrame) -> pd.Series:
        """Solve one method on one simulated history.

        Args:
            name: The optimizer to run.
            mu: Expected returns estimated from ``sample``.
            cov: Covariance estimated from ``sample``.
            sample: The simulated history, for the methods that need the path
                rather than its moments.

        Returns:
            Weights reindexed onto the full universe, with anything the solve did
            not hold filled as zero.
        """
        run_config = copy.deepcopy(config)
        run_config.optimizer.name = name
        run_config.expected_returns = mu.to_dict()
        optimizer = optimizer_factory(
            run_config, cov, expected_returns=mu, returns=sample
        )
        return optimizer.optimize().weights.reindex(assets).fillna(0.0)

    true_mu, true_cov = estimate(returns)
    targets: dict[str, pd.Series] = {}
    for name in methods:
        try:
            targets[name] = solve(name, true_mu, true_cov, returns)
        except Exception as exc:  # noqa: BLE001 - a method may not fit this universe
            raise ValueError(
                f"Optimizer {name!r} could not be solved on the fitted "
                f"distribution, so there is nothing to compare its draws "
                f"against: {exc}"
            ) from exc

    # The ground-truth solves above already logged, once per method, anything
    # the config asks for that a method cannot honour (a return target on HRP,
    # say). Repeating that on every draw would bury the result under hundreds
    # of identical lines, so the simulation loop runs quiet.
    import logging

    factory_log = logging.getLogger("optimization_engine.optimizers.factory")
    previous_level = factory_log.level
    factory_log.setLevel(logging.ERROR)

    rng = np.random.default_rng(seed)
    mean_vector = returns.mean().values
    covariance = np.cov(returns.values, rowvar=False)
    errors: dict[str, list[float]] = {name: [] for name in methods}
    drifts: dict[str, list[float]] = {name: [] for name in methods}
    vol_errors: dict[str, list[float]] = {name: [] for name in methods}
    ret_errors: dict[str, list[float]] = {name: [] for name in methods}
    n_failed = 0
    n_done = 0

    true_sigma = true_cov.reindex(assets, axis=0).reindex(assets, axis=1).values
    true_mu_vector = true_mu.reindex(assets).fillna(0.0).values

    try:
        for _ in range(n_simulations):
            draws = rng.multivariate_normal(mean_vector, covariance, size=n_obs)
            sample = pd.DataFrame(draws, columns=assets)
            try:
                mu, cov = estimate(sample)
            except Exception:
                n_failed += 1
                continue
            any_solved = False
            for name in methods:
                try:
                    weights = solve(name, mu, cov, sample)
                except Exception:
                    n_failed += 1
                    continue
                gap = (weights - targets[name]).values
                errors[name].append(float(np.sqrt(np.mean(gap**2))))
                drifts[name].append(float(np.max(np.abs(gap))))
                # Evaluated against the *truth*, which is the point: a wrong
                # allocation is only expensive to the extent the real world
                # punishes it.
                w = weights.values
                t = targets[name].values
                vol_errors[name].append(
                    abs(
                        float(np.sqrt(max(w @ true_sigma @ w, 0.0)))
                        - float(np.sqrt(max(t @ true_sigma @ t, 0.0)))
                    )
                )
                ret_errors[name].append(abs(float((w - t) @ true_mu_vector)))
                any_solved = True
            if any_solved:
                n_done += 1
    finally:
        factory_log.setLevel(previous_level)

    if n_done == 0:
        raise ValueError(
            f"Every one of the {n_simulations} simulations failed to solve. "
            "The constraints may only be feasible on the observed sample."
        )

    def collect(store: dict[str, list[float]]) -> pd.Series:
        """Average one error store across simulations, per method.

        Args:
            store: ``method -> list of per-simulation errors``.

        Returns:
            One mean per method, with ``nan`` for a method that never produced a
            usable allocation.
        """
        return pd.Series(
            {
                name: (float(np.mean(values)) if values else float("nan"))
                for name, values in store.items()
            }
        )

    return MCOSResult(
        weight_rmse=collect(errors),
        variance_error=collect(vol_errors),
        return_error=collect(ret_errors),
        max_weight_drift=collect(drifts),
        n_simulations=n_done,
        n_failed=n_failed,
        n_observations=n_obs,
        denoised=use_denoise,
    )
