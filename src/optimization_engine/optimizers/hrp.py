"""Hierarchical Risk Parity (López de Prado, 2016).

HRP avoids matrix inversion entirely. Steps:
1. Build a correlation distance ``d = √(½(1 − ρ))``.
2. Cluster with single linkage (or any linkage method).
3. Quasi-diagonalize: reorder assets so similar items sit together.
4. Recursive bisection: walk the cluster tree, splitting risk between
   sub-clusters using inverse-variance weights at each split.

The result is robust to ill-conditioned covariance matrices and to noisy
estimates — particularly useful with many assets or limited history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
from optimization_engine.optimizers.base import BaseOptimizer


def _correl_distance(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1 - corr) / 2.0) ** 0.5


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([int(link[-1, 0]), int(link[-1, 1])])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index.to_numpy()
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.astype(int).tolist()


def _ivp_weights(cov: pd.DataFrame, items: list[str]) -> np.ndarray:
    sub = np.diag(cov.loc[items, items].values)
    inv = 1.0 / sub
    inv = inv / inv.sum()
    return inv


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub_cov = cov.loc[items, items].values
    w = _ivp_weights(cov, items)
    return float(w @ sub_cov @ w)


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list[str]) -> pd.Series:
    weights = pd.Series(np.ones(len(sort_ix)), index=sort_ix, dtype=float).copy()
    clusters: list[list[str]] = [list(sort_ix)]
    while clusters:
        next_clusters: list[list[str]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            split = len(c) // 2
            left, right = c[:split], c[split:]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            denom = v_left + v_right
            alpha = 1 - v_left / denom if denom > 0 else 0.5
            weights.loc[left] = weights.loc[left].values * alpha
            weights.loc[right] = weights.loc[right].values * (1 - alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights


class HRPOptimizer(BaseOptimizer):
    """Hierarchical Risk Parity (HRP).

    Linkage is configurable: ``single`` (default, López de Prado),
    ``average``, ``complete``, ``ward``.
    """

    name = "hrp"

    def __init__(self, *args, linkage_method: str = "single", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linkage_method = linkage_method

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for HRP")
        cov = self.cov_matrix
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)

        dist = np.array(_correl_distance(corr_df).values, copy=True)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        sort_ix = _quasi_diag(link)
        ordered = [cov.columns[i] for i in sort_ix]

        w = _recursive_bisection(cov, ordered)
        w = w.reindex(self.assets).fillna(0.0)
        weights = w.values.astype(float)
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds(weights, lb, ub)
