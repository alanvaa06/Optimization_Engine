"""Covariance and expected-return estimation.

Wraps the common estimators (sample, Ledoit-Wolf, OAS, EWMA, semicovariance)
behind a single, ergonomic API.

Every method name resolves to exactly one estimator, on every machine. That
is a deliberate constraint rather than an obvious one: ``"shrink"`` used to
route through ``riskfolio-lib`` when it happened to be installed and fall
back to Ledoit-Wolf when it was not, which meant the same config on the same
data produced different numbers depending on the environment, with nothing
in the output saying which had run. An engine that reports what an
allocation rests on cannot have an estimator that varies with the virtualenv.

Every estimator returns a matrix that is symmetric and positive semi-definite:
estimators that can produce indefinite matrices (EWMA on short samples,
semicovariance) are passed through :func:`nearest_psd`. Downstream convex
solvers rely on that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CovarianceMethod = Literal[
    "sample", "ledoit_wolf", "oas", "shrink", "ewma", "semi", "denoised"
]

ExpectedReturnMethod = Literal["mean", "ema", "capm", "shrunk_mean"]

#: Human-facing description of each estimator, surfaced by the UI so the
#: analyst can see the assumption they are buying into.
COVARIANCE_DESCRIPTIONS: dict[str, str] = {
    "sample": (
        "Unbiased sample covariance. Unbiased but noisy: needs T ≫ N "
        "observations or the matrix becomes ill-conditioned (and singular "
        "once T ≤ N)."
    ),
    "ledoit_wolf": (
        "Ledoit-Wolf shrinkage toward a scaled identity. Well-conditioned "
        "and the safe default when the number of assets is large relative "
        "to the sample."
    ),
    "oas": (
        "Oracle Approximating Shrinkage. Like Ledoit-Wolf but with a "
        "shrinkage intensity tuned for Gaussian data — usually shrinks "
        "harder on short samples."
    ),
    "shrink": (
        "Alias for ledoit_wolf, kept so that configs and saved scenarios "
        "written against it keep loading. Prefer ledoit_wolf in new work."
    ),
    "ewma": (
        "RiskMetrics exponentially-weighted covariance. Reacts fast to "
        "regime changes; effective sample is only ~1/(1−λ) observations, "
        "so it is noisier than it looks."
    ),
    "semi": (
        "Semicovariance built from downside deviations below the MAR. "
        "Targets downside co-movement rather than total variance; "
        "portfolio 'volatility' from this matrix is a semi-deviation, not "
        "a standard deviation."
    ),
    "denoised": (
        "Sample covariance with the Marchenko-Pastur noise eigenvalues "
        "replaced by their average (López de Prado, 2020). Keeps the factor "
        "structure intact instead of shrinking it along with the noise, so "
        "it conditions the matrix without flattening the signal."
    ),
}


# ---------------------------------------------------------------------------
# PSD repair + diagnostics
# ---------------------------------------------------------------------------


def nearest_psd(matrix: pd.DataFrame | np.ndarray, eig_floor: float = 0.0):
    """Return the nearest positive semi-definite version of ``matrix``.

    Symmetrizes, then clips eigenvalues at ``eig_floor``. When the input is
    already PSD this is (numerically) the identity transform.

    Args:
        matrix: A square covariance-like matrix.
        eig_floor: Minimum eigenvalue to keep. ``0.0`` gives PSD; pass a
            small positive number (e.g. ``1e-12``) to force strict positive
            definiteness for solvers that need it.
    """
    is_frame = isinstance(matrix, pd.DataFrame)
    values = np.asarray(matrix.values if is_frame else matrix, dtype=float)
    sym = (values + values.T) / 2.0
    eigval, eigvec = np.linalg.eigh(sym)
    if (eigval >= eig_floor).all():
        repaired = sym
    else:
        repaired = (eigvec * np.clip(eigval, eig_floor, None)) @ eigvec.T
        repaired = (repaired + repaired.T) / 2.0
    if is_frame:
        return pd.DataFrame(repaired, index=matrix.index, columns=matrix.columns)
    return repaired


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Conditioning and sample-adequacy diagnostics for a covariance estimate.

    Attributes:
        n_assets: Number of assets ``N``.
        n_observations: Number of return observations ``T``.
        observations_per_asset: ``T / N``. Below ~10 the sample covariance is
            badly conditioned; at ``T <= N`` it is singular.
        condition_number: Ratio of largest to smallest eigenvalue.
        min_eigenvalue: Smallest eigenvalue of the estimate.
        is_psd: Whether every eigenvalue is non-negative (within tolerance).
        effective_observations: For EWMA, ``1/(1−λ)``; otherwise ``T``.
        warnings: Human-facing warnings, ordered most severe first.
    """

    n_assets: int
    n_observations: int
    observations_per_asset: float
    condition_number: float
    min_eigenvalue: float
    is_psd: bool
    effective_observations: float
    warnings: tuple[str, ...] = ()

    @property
    def is_reliable(self) -> bool:
        """True when nothing worth warning the analyst about was found."""
        return not self.warnings


def covariance_diagnostics(
    cov: pd.DataFrame,
    n_observations: int,
    method: str = "sample",
    ewma_lambda: float = 0.94,
) -> CovarianceDiagnostics:
    """Assess whether a covariance estimate is fit to optimize against.

    The checks are the ones that actually bite in portfolio construction:
    too few observations per asset, near-singularity (which makes
    mean-variance weights explode), and non-PSD matrices.
    """
    values = np.asarray(cov.values, dtype=float)
    n_assets = values.shape[0]
    eigval = np.linalg.eigvalsh((values + values.T) / 2.0)
    min_eig = float(eigval.min())
    max_eig = float(eigval.max())
    cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    tol = max(abs(max_eig), 1.0) * 1e-10
    obs_per_asset = (
        float(n_observations) / n_assets if n_assets else float("nan")
    )
    eff_obs = (
        1.0 / (1.0 - ewma_lambda)
        if method == "ewma" and ewma_lambda < 1.0
        else float(n_observations)
    )

    msgs: list[str] = []
    if n_observations <= n_assets:
        msgs.append(
            f"Only {n_observations} observations for {n_assets} assets (T ≤ N): "
            "the sample covariance is singular. Use a shrinkage estimator "
            "(ledoit_wolf / oas) or shorten the universe."
        )
    elif obs_per_asset < 10:
        msgs.append(
            f"Only {obs_per_asset:.1f} observations per asset (T/N). Below ~10, "
            "covariance estimates are dominated by noise — prefer a shrinkage "
            "estimator or HRP, which does not invert the matrix."
        )
    if eff_obs < 2 * n_assets:
        msgs.append(
            f"Effective sample size is ~{eff_obs:.0f} for {n_assets} assets. "
            "Estimation error will dominate optimizer output."
        )
    if not np.isfinite(cond) or cond > 1e8:
        msgs.append(
            f"Condition number is {cond:.3g}. The matrix is near-singular, so "
            "mean-variance weights are unstable — small changes in expected "
            "returns will produce very different portfolios."
        )
    elif cond > 1e4:
        msgs.append(
            f"Condition number is {cond:.3g}: the covariance is poorly "
            "conditioned. Consider shrinkage or fewer, less-collinear assets."
        )
    if min_eig < -tol:
        msgs.append(
            f"Smallest eigenvalue is {min_eig:.3g} (< 0): the matrix is not "
            "positive semi-definite and has been repaired by eigenvalue "
            "clipping."
        )

    return CovarianceDiagnostics(
        n_assets=n_assets,
        n_observations=int(n_observations),
        observations_per_asset=obs_per_asset,
        condition_number=cond,
        min_eigenvalue=min_eig,
        is_psd=min_eig >= -tol,
        effective_observations=float(eff_obs),
        warnings=tuple(msgs),
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _sample(returns: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    return returns.cov(ddof=ddof)


def _ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _oas(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import OAS

    cov = OAS().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _ewma(returns: pd.DataFrame, lam: float = 0.94, demean: bool = True) -> pd.DataFrame:
    """Exponentially weighted (RiskMetrics-style) covariance.

    ``demean=False`` reproduces the original RiskMetrics recursion, which
    treats the conditional mean as zero — the usual choice for daily data,
    where the mean is small relative to the noise in estimating it.
    """
    r = returns - returns.mean() if demean else returns
    weights = (1 - lam) * lam ** np.arange(len(r))[::-1]
    weights /= weights.sum()
    cov = (r.T * weights) @ r
    return cov


def _semi(returns: pd.DataFrame, mar: float | pd.Series | None = None) -> pd.DataFrame:
    """Semicovariance from downside deviations below a minimum acceptable return.

    ``Σ_semi = (1/(T−1)) · Dᵀ D`` with ``D = min(r − MAR, 0)``.

    The deviations are *not* re-centred: ``D`` already measures shortfall
    against the MAR, so subtracting its own (negative) mean would understate
    downside co-movement. ``mar`` defaults to the per-asset sample mean,
    which is the Markowitz (1959) semivariance convention; pass ``0.0`` for
    the zero-threshold convention, or a Series for per-asset thresholds.
    """
    threshold = returns.mean() if mar is None else mar
    deviation = returns.sub(threshold, axis=1) if not np.isscalar(threshold) else returns - threshold
    deviation = deviation.where(deviation < 0, 0.0)
    n = len(deviation)
    denom = max(n - 1, 1)
    cov = deviation.T.values @ deviation.values / denom
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def covariance_matrix(
    returns: pd.DataFrame,
    method: CovarianceMethod = "ledoit_wolf",
    annualize: bool = True,
    periods_per_year: int = 252,
    ewma_lambda: float = 0.94,
    semi_mar: float | pd.Series | None = None,
    ensure_psd: bool = True,
    denoise: bool = False,
    denoise_method: str = "constant_residual",
    denoise_alpha: float = 0.0,
    detone: int = 0,
) -> pd.DataFrame:
    """Estimate a covariance matrix on returns.

    Set ``annualize=True`` to scale by ``periods_per_year`` (e.g. daily ⇒ annual).

    Args:
        returns: Wide frame of periodic returns, one column per asset.
        method: Estimator name; see :data:`COVARIANCE_DESCRIPTIONS`.
        annualize: Scale the result by ``periods_per_year``.
        periods_per_year: Observations per year in ``returns``.
        ewma_lambda: Decay factor when ``method == "ewma"``.
        semi_mar: Minimum acceptable return for ``method == "semi"``.
            ``None`` uses the per-asset sample mean.
        ensure_psd: Repair the estimate with :func:`nearest_psd`. Leave on
            unless you specifically want the raw estimator output.
        denoise: Apply the Marchenko-Pastur eigenvalue filter after
            estimating (López de Prado, 2020). Implied by
            ``method="denoised"``, and composable with any other estimator —
            denoising a Ledoit-Wolf matrix is legitimate, if belt-and-braces.
        denoise_method: ``"constant_residual"`` or ``"targeted_shrinkage"``;
            see :func:`~optimization_engine.data.denoise.denoise_correlation`.
        denoise_alpha: Noise-block shrinkage retained under
            ``"targeted_shrinkage"``.
        detone: Remove this many leading eigenvectors (the market component)
            after denoising. Non-zero values make the result **singular** —
            correct for clustering methods, fatal for anything that inverts
            the matrix.

    Returns:
        The estimate. When denoising ran, the :class:`DenoiseReport` is
        attached as ``result.attrs["denoise_report"]`` so callers that want
        the spectrum diagnostics can reach them without a second API.

    Raises:
        ValueError: If ``returns`` is empty or ``method`` is unknown.
    """
    if returns is None or returns.empty:
        raise ValueError("Cannot estimate a covariance matrix from empty returns.")
    if returns.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 return observations to estimate covariance; got "
            f"{returns.shape[0]}."
        )
    if returns.isna().any().any():
        raise ValueError(
            "Returns contain missing values. Align or fill the price panel "
            "before estimating covariance — silently dropping rows would "
            "change the sample each asset is estimated on."
        )

    if method in ("sample", "denoised"):
        cov = _sample(returns)
    elif method == "ledoit_wolf":
        cov = _ledoit_wolf(returns)
    elif method == "oas":
        cov = _oas(returns)
    elif method == "shrink":
        # An alias, not a third shrinkage estimator — see the module docstring.
        cov = _ledoit_wolf(returns)
    elif method == "ewma":
        cov = _ewma(returns, lam=ewma_lambda)
    elif method == "semi":
        cov = _semi(returns, mar=semi_mar)
    else:
        raise ValueError(
            f"Unknown covariance method: {method!r}. "
            f"Available: {sorted(COVARIANCE_DESCRIPTIONS)}"
        )

    if annualize:
        cov = cov * periods_per_year

    report = None
    if denoise or method == "denoised" or detone:
        from optimization_engine.data.denoise import denoise_covariance

        cov, report = denoise_covariance(
            cov,
            n_observations=len(returns),
            method=denoise_method,
            alpha=denoise_alpha,
            detone=detone,
        )

    if ensure_psd:
        cov = nearest_psd(cov)
    if report is not None:
        cov.attrs["denoise_report"] = report
    return cov


def covariance_from_config(returns: pd.DataFrame, config) -> pd.DataFrame:
    """Estimate the covariance the way one :class:`EngineConfig` asks for.

    Every part of the engine that re-estimates a covariance — the frontier
    sweep, the bootstrap, the walk-forward, the CLI, the UI — has to make the
    same choices about estimator, annualization, decay and denoising. Doing
    that inline five times is how a run ends up bootstrapping a different
    matrix than it optimized. This is the single place those choices live.
    """
    return covariance_matrix(
        returns,
        method=config.covariance_method,
        periods_per_year=config.periods_per_year,
        ewma_lambda=config.ewma_lambda,
        denoise=getattr(config, "denoise", False),
        denoise_method=getattr(config, "denoise_method", "constant_residual"),
        denoise_alpha=getattr(config, "denoise_alpha", 0.0),
        detone=getattr(config, "detone", 0),
    )


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------

EXPECTED_RETURN_DESCRIPTIONS: dict[str, str] = {
    "mean": (
        "Annualized geometric mean of realized returns. Simple, but the "
        "standard error of a mean estimate is huge — decades of data are "
        "needed to distinguish two assets' means."
    ),
    "ema": (
        "Exponentially-weighted mean: recent returns count more. Responsive "
        "but even noisier than the full-sample mean."
    ),
    "capm": (
        "CAPM-implied returns: rf + β·(market premium). Cross-sectionally "
        "disciplined, so far more stable than raw historical means."
    ),
    "shrunk_mean": (
        "James-Stein shrinkage of historical means toward the grand mean. "
        "Keeps the ranking information while pulling in the extremes that "
        "drive mean-variance to corner solutions."
    ),
}


def james_stein_shrinkage(
    sample_mean: pd.Series,
    cov_matrix: pd.DataFrame,
    n_observations: int,
    target: float | None = None,
) -> tuple[pd.Series, float]:
    """Shrink a vector of sample means toward a common target.

    Implements the Jorion (1986) Bayes-Stein estimator: the shrinkage
    intensity grows with the number of assets and the dispersion of the
    covariance, and shrinks as the sample lengthens.

    Args:
        sample_mean: Annualized sample means, one per asset.
        cov_matrix: Annualized covariance matrix for the same assets.
        n_observations: Number of return observations behind the means.
        target: Grand mean to shrink toward. ``None`` uses the
            minimum-variance portfolio's expected return, which is the
            Jorion prior.

    Returns:
        ``(shrunk_means, intensity)`` where ``intensity`` is in ``[0, 1]``.
    """
    mu = sample_mean.astype(float)
    assets = list(mu.index)
    sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
    n = len(assets)
    if n < 3 or n_observations <= n + 2:
        # Stein shrinkage has no advantage below 3 assets and the intensity
        # formula is undefined once the sample is too short.
        return mu, 0.0

    try:
        inv = np.linalg.pinv(sigma)
        ones = np.ones(n)
        denom = float(ones @ inv @ ones)
        mu_target = (
            float(target)
            if target is not None
            else float(ones @ inv @ mu.values) / denom
        )
        diff = mu.values - mu_target
        quad = float(diff @ inv @ diff)
        if quad <= 0:
            return mu, 1.0
        lam = (n + 2) / (n + 2 + n_observations * quad)
    except np.linalg.LinAlgError:
        return mu, 0.0

    intensity = float(np.clip(lam, 0.0, 1.0))
    shrunk = (1 - intensity) * mu.values + intensity * mu_target
    return pd.Series(shrunk, index=assets, name=mu.name), intensity


def expected_returns_from_history(
    returns: pd.DataFrame,
    method: ExpectedReturnMethod = "mean",
    periods_per_year: int = 252,
    span: int = 180,
    market_return: float | None = None,
    risk_free_rate: float = 0.0,
    market_weights: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
) -> pd.Series:
    """Build an expected-return vector from realized history.

    * ``mean``        — annualized geometric historical mean.
    * ``ema``         — exponentially-weighted mean with the given ``span``.
    * ``capm``        — implied returns from a single-factor CAPM where the
      market portfolio is approximated by ``market_weights`` (defaulting to
      equal weights).
    * ``shrunk_mean`` — ``mean`` pulled toward the minimum-variance
      portfolio's return via :func:`james_stein_shrinkage`.

    Raises:
        ValueError: On an unknown ``method`` or empty ``returns``.
    """
    if returns is None or returns.empty:
        raise ValueError("Cannot estimate expected returns from empty data.")

    if method == "mean":
        return ((1 + returns).prod() ** (periods_per_year / len(returns))) - 1
    if method == "ema":
        ema = returns.ewm(span=span, adjust=False).mean().iloc[-1]
        return (1 + ema) ** periods_per_year - 1
    if method == "shrunk_mean":
        raw = ((1 + returns).prod() ** (periods_per_year / len(returns))) - 1
        if cov_matrix is None:
            cov_matrix = covariance_matrix(
                returns, method="ledoit_wolf", periods_per_year=periods_per_year
            )
        shrunk, _ = james_stein_shrinkage(raw, cov_matrix, len(returns))
        return shrunk
    if method == "capm":
        if cov_matrix is None:
            cov_matrix = covariance_matrix(
                returns, method="ledoit_wolf", periods_per_year=periods_per_year
            )
        if market_weights is None:
            market_weights = pd.Series(
                np.ones(returns.shape[1]) / returns.shape[1], index=returns.columns
            )
        market_weights = market_weights.reindex(returns.columns).fillna(0.0)
        if float(market_weights.sum()) <= 0:
            raise ValueError(
                "CAPM market weights sum to zero — cannot define a market "
                "portfolio."
            )
        market_weights = market_weights / float(market_weights.sum())
        market_return_est = market_return
        if market_return_est is None:
            mkt = (returns * market_weights).sum(axis=1)
            market_return_est = (
                (1 + mkt).prod() ** (periods_per_year / len(mkt)) - 1
            )
        market_var = float(market_weights.values @ cov_matrix.values @ market_weights.values)
        if market_var <= 0:
            raise ValueError(
                "CAPM market portfolio has zero variance; betas are undefined."
            )
        betas = (cov_matrix.values @ market_weights.values) / market_var
        return pd.Series(
            risk_free_rate + betas * (market_return_est - risk_free_rate),
            index=returns.columns,
        )
    raise ValueError(
        f"Unknown expected-return method: {method!r}. "
        f"Available: {sorted(EXPECTED_RETURN_DESCRIPTIONS)}"
    )
