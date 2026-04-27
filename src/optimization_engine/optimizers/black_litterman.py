"""Black-Litterman expected-return blender.

Combines the implied equilibrium returns from a market-cap weighted
portfolio with subjective views to produce a posterior expected-return
vector. The optimizer plugs that posterior into a standard mean-variance
solve so all bound/group/turnover constraints continue to apply.

References:
    Black, F., & Litterman, R. (1992). Global portfolio optimization.
    He, G., & Litterman, R. (1999). The intuition behind Black-Litterman
    model portfolios. Goldman Sachs Quantitative Research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine.optimizers.base import BaseOptimizer
from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer


def implied_equilibrium_returns(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Reverse-optimize equilibrium returns from market weights.

    π = δ · Σ · w_mkt  +  rf
    """
    w = market_weights.reindex(cov_matrix.columns).fillna(0.0).values
    pi = risk_aversion * cov_matrix.values @ w
    return pd.Series(pi + risk_free_rate, index=cov_matrix.columns)


def black_litterman_posterior(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the Black-Litterman posterior mean and covariance.

    `views` is a mapping of asset → annualized expected return.
    `view_confidences` is the variance of each view's error term (Ω diagonal).
    Defaults to ``tau · σ_i^2``.
    """
    pi = implied_equilibrium_returns(
        market_weights, cov_matrix, risk_aversion, risk_free_rate
    )
    if not views:
        return pi, cov_matrix

    assets = list(cov_matrix.columns)
    asset_idx = {a: i for i, a in enumerate(assets)}
    valid = [a for a in views if a in asset_idx]
    if not valid:
        return pi, cov_matrix
    k = len(valid)
    n = len(assets)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    for j, a in enumerate(valid):
        P[j, asset_idx[a]] = 1.0
        Q[j] = float(views[a])

    var_diag = np.diag(cov_matrix.values)
    if view_confidences:
        omega = np.diag([float(view_confidences.get(a, tau * var_diag[asset_idx[a]])) for a in valid])
    else:
        omega = np.diag([tau * var_diag[asset_idx[a]] for a in valid])

    sigma = cov_matrix.values
    tau_sigma = tau * sigma

    M_inv = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
    M = np.linalg.inv(M_inv)
    bl_mean = M @ (np.linalg.inv(tau_sigma) @ pi.values + P.T @ np.linalg.inv(omega) @ Q)
    bl_cov = sigma + M

    return (
        pd.Series(bl_mean, index=assets),
        pd.DataFrame(bl_cov, index=assets, columns=assets),
    )


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman + mean-variance optimizer.

    Pass `market_weights` (or omit for equal weights), optional `views`,
    and `view_confidences`. The posterior is built and a downstream
    mean-variance solve respecting all the standard constraints is run.
    """

    name = "black_litterman"

    def __init__(
        self,
        *args,
        market_weights: pd.Series | dict[str, float] | None = None,
        views: dict[str, float] | None = None,
        view_confidences: dict[str, float] | None = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
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

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for Black-Litterman")
        if self.market_weights is None:
            mkt = pd.Series(
                np.ones(len(self.assets)) / len(self.assets), index=self.assets
            )
        else:
            mkt = self.market_weights.reindex(self.assets).fillna(0.0)
            if mkt.sum() > 0:
                mkt = mkt / mkt.sum()

        post_mean, post_cov = black_litterman_posterior(
            self.cov_matrix,
            mkt,
            self.views,
            self.view_confidences,
            tau=self.tau,
            risk_aversion=self.risk_aversion,
            risk_free_rate=self.risk_free_rate,
        )

        sub_optimizer = MeanVarianceOptimizer(
            expected_returns=post_mean,
            cov_matrix=post_cov,
            constraints=self.constraints,
            risk_free_rate=self.risk_free_rate,
            risk_aversion=self.risk_aversion,
        )
        result = sub_optimizer.optimize()
        return result.weights.reindex(self.assets).fillna(0.0).values
