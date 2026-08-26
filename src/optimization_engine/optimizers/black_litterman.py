"""Black-Litterman expected-return blender.

Combines the implied equilibrium returns from a market-cap weighted
portfolio with subjective views to produce a posterior expected-return
vector. The optimizer plugs that posterior into a standard mean-variance
solve so all bound/group/turnover constraints continue to apply.

Two things this implementation does that a minimal one does not:

* **Relative views.** A view is a row of the pick matrix ``P``, so
  "European equity beats US equity by 2%" is expressible, not just
  "European equity returns 8%". Absolute views are the special case of a
  single ``+1`` in the row. Relative views are how practitioners actually
  hold opinions, and they leave the overall market level untouched.
* **He-Litterman Ω.** The default view uncertainty is
  ``Ω = diag(P · τΣ · Pᵀ)``, which scales each view's error with the prior
  variance of *that view's* portfolio. Using ``τ·σ_i²`` instead — as a
  per-asset default does — is only correct for single-asset views.

References:
    Black, F., & Litterman, R. (1992). Global portfolio optimization.
    He, G., & Litterman, R. (1999). The intuition behind Black-Litterman
    model portfolios. Goldman Sachs Quantitative Research.
    Idzorek, T. (2007). A step-by-step guide to the Black-Litterman model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimization_engine.optimizers.base import BaseOptimizer
from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer


@dataclass(frozen=True)
class View:
    """One Black-Litterman view.

    Attributes:
        weights: ``asset -> coefficient`` in the view portfolio. An absolute
            view on a single asset is ``{"US_Equity": 1.0}``; a relative view
            is ``{"US_Equity": 1.0, "Intl_Equity": -1.0}``.
        expected_return: The return the view portfolio is expected to earn.
            For a relative view this is the *spread*, not a level.
        confidence: Optional variance of the view's error term (Ω diagonal).
            Smaller means more confident. ``None`` uses the He-Litterman
            default ``p' τΣ p``.
        label: Optional human-facing name, echoed back in diagnostics.
    """

    weights: dict[str, float]
    expected_return: float
    confidence: float | None = None
    label: str = ""

    @property
    def is_relative(self) -> bool:
        """True when the view's coefficients net to ~0 (a spread, not a level)."""
        return abs(sum(self.weights.values())) < 1e-9

    def describe(self) -> str:
        longs = [a for a, v in self.weights.items() if v > 0]
        shorts = [a for a, v in self.weights.items() if v < 0]
        if shorts:
            return (
                f"{' + '.join(longs)} outperforms {' + '.join(shorts)} "
                f"by {self.expected_return:.2%}"
            )
        return f"{' + '.join(longs)} returns {self.expected_return:.2%}"


def normalize_views(
    views: dict[str, float] | list[View] | None,
    view_confidences: dict[str, float] | None = None,
) -> list[View]:
    """Accept either the simple ``asset -> return`` mapping or full ``View``s.

    The mapping form stays supported because it is what the config schema and
    the simple UI table produce; it is interpreted as one absolute view per
    entry.
    """
    if not views:
        return []
    if isinstance(views, dict):
        confidences = view_confidences or {}
        return [
            View(
                weights={asset: 1.0},
                expected_return=float(q),
                confidence=(
                    float(confidences[asset]) if asset in confidences else None
                ),
                label=f"{asset} absolute",
            )
            for asset, q in views.items()
        ]
    return list(views)


def build_pick_matrix(
    views: list[View], assets: list[str]
) -> tuple[np.ndarray, np.ndarray, list[View]]:
    """Assemble ``(P, Q)`` from a list of views, dropping unusable ones.

    A view is dropped when none of its assets are in the universe, or when
    every coefficient is zero — either way it carries no information about
    the assets being optimized.
    """
    idx = {a: i for i, a in enumerate(assets)}
    rows: list[np.ndarray] = []
    q: list[float] = []
    kept: list[View] = []
    for view in views:
        row = np.zeros(len(assets))
        touched = False
        for asset, coeff in view.weights.items():
            if asset in idx and coeff != 0:
                row[idx[asset]] = float(coeff)
                touched = True
        if not touched:
            continue
        rows.append(row)
        q.append(float(view.expected_return))
        kept.append(view)
    if not rows:
        return np.zeros((0, len(assets))), np.zeros(0), []
    return np.vstack(rows), np.array(q), kept


def implied_risk_aversion(
    market_return: float, market_variance: float, risk_free_rate: float = 0.0
) -> float:
    """Back out δ from the market portfolio: ``δ = (E[r_m] − rf) / σ_m²``.

    Picking δ by hand is the weakest link in most Black-Litterman
    implementations — it sets the entire level of the equilibrium returns.
    Calibrating it to an observed market Sharpe ratio makes the prior
    reproducible.
    """
    if market_variance <= 0:
        raise ValueError(
            "Market variance must be positive to imply a risk-aversion "
            "coefficient."
        )
    return float((market_return - risk_free_rate) / market_variance)


def implied_equilibrium_returns(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Reverse-optimize equilibrium returns from market weights.

    ``π = δ · Σ · w_mkt + rf``

    The ``rf`` term makes ``π`` a *total* expected return rather than an
    excess return, matching the convention used everywhere else in the
    engine (expected returns are totals, and Sharpe subtracts ``rf``).
    """
    w = market_weights.reindex(cov_matrix.columns).fillna(0.0).values
    pi = risk_aversion * cov_matrix.values @ w
    return pd.Series(pi + risk_free_rate, index=cov_matrix.columns)


def black_litterman_posterior(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | list[View] | None = None,
    view_confidences: dict[str, float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the Black-Litterman posterior mean and covariance.

    ``views`` accepts either ``{asset: annualized return}`` (absolute views)
    or a list of :class:`View` objects (absolute *or* relative).
    ``view_confidences`` only applies to the mapping form; ``View.confidence``
    carries it otherwise.

    Returns:
        ``(posterior_mean, posterior_covariance)`` where the covariance is
        ``Σ + M`` — the prior plus the posterior parameter uncertainty, which
        is what should be fed to the downstream mean-variance solve.
    """
    if not 0 < tau <= 1:
        raise ValueError(
            f"tau must be in (0, 1]; got {tau}. It scales the prior "
            "covariance, so values outside that range make the prior either "
            "degenerate or wider than the return distribution itself."
        )
    pi = implied_equilibrium_returns(
        market_weights, cov_matrix, risk_aversion, risk_free_rate
    )
    view_list = normalize_views(views, view_confidences)
    if not view_list:
        return pi, cov_matrix

    assets = list(cov_matrix.columns)
    P, Q, kept = build_pick_matrix(view_list, assets)
    if P.shape[0] == 0:
        return pi, cov_matrix

    sigma = cov_matrix.values
    tau_sigma = tau * sigma

    # He-Litterman default: each view's uncertainty is the prior variance of
    # that view's own portfolio. Explicit confidences override per view.
    default_omega = np.diag(P @ tau_sigma @ P.T).copy()
    default_omega = np.maximum(default_omega, 1e-12)
    omega_diag = np.array(
        [
            view.confidence if view.confidence is not None else default_omega[j]
            for j, view in enumerate(kept)
        ],
        dtype=float,
    )
    if (omega_diag <= 0).any():
        raise ValueError(
            "View confidences are variances and must be strictly positive; a "
            "zero would assert a view held with perfect certainty."
        )
    omega = np.diag(omega_diag)

    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.inv(omega)
    M = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
    bl_mean = M @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
    bl_cov = sigma + M

    return (
        pd.Series(bl_mean, index=assets),
        pd.DataFrame(bl_cov, index=assets, columns=assets),
    )


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman + mean-variance optimizer.

    Pass ``market_weights`` (or omit for equal weights), ``views``, and
    optionally ``view_confidences``. The posterior is built and a downstream
    mean-variance solve respecting all the standard constraints is run.

    Set ``calibrate_risk_aversion=True`` with a ``market_return`` to imply δ
    from the market's own Sharpe ratio instead of hard-coding 2.5.
    """

    name = "black_litterman"
    bounds_mode = "hard"

    def __init__(
        self,
        *args,
        market_weights: pd.Series | dict[str, float] | None = None,
        views: dict[str, float] | list[View] | None = None,
        view_confidences: dict[str, float] | None = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        market_return: float | None = None,
        calibrate_risk_aversion: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.market_weights = (
            pd.Series(market_weights) if isinstance(market_weights, dict) else market_weights
        )
        self.views = views or {}
        self.view_confidences = view_confidences or {}
        self.tau = float(tau)
        self.risk_aversion = float(risk_aversion)
        self.market_return = market_return
        self.calibrate_risk_aversion = bool(calibrate_risk_aversion)

    def _market_portfolio(self) -> pd.Series:
        if self.market_weights is None:
            return pd.Series(
                np.ones(len(self.assets)) / len(self.assets), index=self.assets
            )
        mkt = self.market_weights.reindex(self.assets).fillna(0.0)
        total = float(mkt.sum())
        if total <= 0:
            raise ValueError(
                "Market-cap weights sum to zero, so there is no equilibrium "
                "portfolio to reverse-optimize from. Provide positive weights "
                "or leave them empty for equal weights."
            )
        return mkt / total

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for Black-Litterman")
        mkt = self._market_portfolio()

        delta = self.risk_aversion
        if self.calibrate_risk_aversion and self.market_return is not None:
            market_var = float(mkt.values @ self.cov_matrix.values @ mkt.values)
            delta = implied_risk_aversion(
                float(self.market_return), market_var, self.risk_free_rate
            )
            self._diagnostics["implied_risk_aversion"] = delta

        post_mean, post_cov = black_litterman_posterior(
            self.cov_matrix,
            mkt,
            self.views,
            self.view_confidences,
            tau=self.tau,
            risk_aversion=delta,
            risk_free_rate=self.risk_free_rate,
        )

        prior = implied_equilibrium_returns(
            mkt, self.cov_matrix, delta, self.risk_free_rate
        )
        view_list = normalize_views(self.views, self.view_confidences)
        self._diagnostics.update(
            {
                "bl_prior_returns": prior,
                "bl_posterior_returns": post_mean,
                "bl_view_impact": post_mean - prior,
                "bl_views": [v.describe() for v in view_list],
                "bl_tau": self.tau,
                "bl_risk_aversion": delta,
                "bl_market_weights": mkt,
            }
        )

        sub_optimizer = MeanVarianceOptimizer(
            expected_returns=post_mean,
            cov_matrix=post_cov,
            constraints=self.constraints,
            risk_free_rate=self.risk_free_rate,
            risk_aversion=delta,
        )
        result = sub_optimizer.optimize()
        self._diagnostics.update(
            {
                k: v
                for k, v in result.extras.items()
                if k in ("solver", "solver_status", "solve_seconds", "mode")
            }
        )
        # The posterior covariance is what the sub-solve optimized against;
        # reporting portfolio risk against the prior would understate it.
        self.cov_matrix = post_cov
        self.expected_returns = post_mean
        return result.weights.reindex(self.assets).fillna(0.0).values
