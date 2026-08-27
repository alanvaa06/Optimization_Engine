"""Hierarchical Equal Risk Contribution (Raffinot, 2017/2018).

HERC and HRP share a first step — build a dendrogram from the correlation
distance — and then part company on how they walk it.

HRP quasi-diagonalizes the matrix into a *list* and then bisects that list
down the middle at each step. The ordering comes from the tree, but the
splits do not: the midpoint of a sorted list is not in general where the
dendrogram actually branches, so HRP can cut straight through a tight
cluster and split a genuine group's risk budget between two arms of the
recursion.

HERC splits at the tree's own merge points, and stops once it reaches the
number of clusters the data supports rather than recursing down to single
assets. Between two sibling branches it allocates so that each contributes
equal risk — the "equal risk contribution" in the name — and within a final
cluster it falls back to naive risk parity.

The practical differences that follow:

* the allocation respects the cluster structure the tree actually found,
  rather than an artefact of the ordering;
* the number of clusters is a reported, inspectable choice rather than an
  implicit one; and
* the cluster-level risk measure is pluggable, so downside risk (CVaR,
  conditional drawdown) can drive the split when variance is the wrong
  measure — the reason Raffinot's results improve most on non-normal assets.

Like HRP, constraints are applied to the finished allocation by projection,
so ``bounds_mode`` is ``"soft_iterated"`` and the distance moved is reported.

References:
    Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation".
    *The Journal of Portfolio Management* 44(2).

    Raffinot, T. (2018). "The Hierarchical Equal Risk Contribution
    Portfolio". SSRN 3237540.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers._clustering import (
    build_linkage,
    correlation_distance,
    correlation_from_covariance,
    inverse_variance_weights,
    optimal_clusters,
)
from optimization_engine.optimizers.base import BaseOptimizer

#: Risk measures available for the cluster-level split.
HERC_RISK_MEASURES = ("variance", "std", "cvar", "cdar", "equal_weight")


class HERCOptimizer(BaseOptimizer):
    """Hierarchical Equal Risk Contribution.

    Args:
        linkage_method: Linkage rule. ``ward`` by default — HERC partitions
            the tree, and single linkage's chaining produces partitions where
            one cluster swallows most of the universe.
        n_clusters: Force a cluster count. ``None`` selects it by maximizing
            the silhouette t-statistic.
        max_clusters: Upper bound for that search.
        risk_measure: How a cluster's risk is measured when splitting the
            budget between two branches. ``"variance"`` and ``"std"`` use the
            covariance matrix; ``"cvar"`` and ``"cdar"`` need ``returns`` and
            let downside risk drive the allocation; ``"equal_weight"``
            reproduces Raffinot's HCAA, which splits capital rather than risk
            at each branch.
        alpha: Tail probability for the ``"cvar"`` and ``"cdar"`` measures.
        returns: Periodic return history. Required for the downside measures.
    """

    name = "herc"
    bounds_mode = "soft_iterated"

    def __init__(
        self,
        *args,
        linkage_method: str = "ward",
        n_clusters: int | None = None,
        max_clusters: int | None = None,
        risk_measure: str = "variance",
        alpha: float = 0.05,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if risk_measure not in HERC_RISK_MEASURES:
            raise ValueError(
                f"Unknown HERC risk measure {risk_measure!r}. "
                f"Available: {list(HERC_RISK_MEASURES)}"
            )
        if risk_measure in ("cvar", "cdar") and returns is None:
            raise ValueError(
                f"risk_measure={risk_measure!r} is estimated from the return "
                "history, so HERC needs a `returns` frame. Pass one, or use "
                "'variance'."
            )
        if not 0 < alpha < 0.5:
            raise ValueError(
                f"alpha is a tail probability and must lie in (0, 0.5); got "
                f"{alpha}."
            )
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.risk_measure = risk_measure
        self.alpha = float(alpha)
        self.returns = returns

    # -- risk of a cluster held at its own naive risk-parity weights --------

    def _cluster_risk(self, items: list[str]) -> float:
        cov = self.cov_matrix
        weights = inverse_variance_weights(cov, items)
        if self.risk_measure == "equal_weight":
            return 1.0
        variance = float(weights.values @ cov.loc[items, items].values @ weights.values)
        if self.risk_measure == "variance":
            return max(variance, 1e-16)
        if self.risk_measure == "std":
            return max(np.sqrt(max(variance, 0.0)), 1e-16)

        series = (self.returns[items] * weights).sum(axis=1)
        if self.risk_measure == "cvar":
            from optimization_engine.analytics.risk import cvar_historic

            return max(float(cvar_historic(series, level=self.alpha * 100)), 1e-16)

        from optimization_engine.analytics.risk import drawdown_series

        drawdown = drawdown_series(series)
        threshold = float(np.quantile(drawdown, self.alpha))
        tail = drawdown[drawdown <= threshold]
        return max(float(-tail.mean()) if len(tail) else 1e-16, 1e-16)

    # -- the recursion ------------------------------------------------------

    def _split_clusters(self, link: np.ndarray, labels: dict[str, int]) -> pd.Series:
        """Allocate the budget between clusters, walking the tree top-down.

        At each merge point the two sides receive shares inversely
        proportional to their risk, so that each side contributes the same
        amount — the equal-risk-contribution rule applied one level at a
        time, which is what makes the result depend on the tree's own shape.
        """
        from scipy.cluster.hierarchy import to_tree

        assets = self.assets
        index_to_asset = {i: a for i, a in enumerate(assets)}
        root = to_tree(link)

        def leaves(node) -> list[str]:
            return [index_to_asset[i] for i in node.pre_order(lambda x: x.id)]

        def clusters_below(node) -> set[int]:
            return {labels[a] for a in leaves(node)}

        weights: dict[int, float] = {}

        def recurse(node, budget: float) -> None:
            below = clusters_below(node)
            if len(below) == 1:
                label = next(iter(below))
                weights[label] = weights.get(label, 0.0) + budget
                return
            left, right = node.get_left(), node.get_right()
            if left is None or right is None:  # pragma: no cover - defensive
                for label in below:
                    weights[label] = weights.get(label, 0.0) + budget / len(below)
                return
            risk_left = self._cluster_risk(leaves(left))
            risk_right = self._cluster_risk(leaves(right))
            total = risk_left + risk_right
            share_left = 1.0 - risk_left / total if total > 0 else 0.5
            recurse(left, budget * share_left)
            recurse(right, budget * (1.0 - share_left))

        recurse(root, 1.0)
        return pd.Series(weights, dtype=float)

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for HERC")
        assets = self.assets
        if len(assets) < 2:
            raise ValueError("HERC needs at least 2 assets to build a tree.")
        if (np.array([self.constraints.get_bounds(a)[0] for a in assets]) < 0).any():
            raise ValueError(
                "HERC produces long-only weights by construction; a negative "
                "minimum weight cannot be honoured."
            )
        if self.returns is not None:
            missing = [a for a in assets if a not in self.returns.columns]
            if missing:
                raise ValueError(
                    f"The return history is missing {len(missing)} of the "
                    f"universe's assets ({', '.join(map(str, missing[:5]))}); "
                    "the downside risk measures cannot be evaluated."
                )

        corr = correlation_from_covariance(self.cov_matrix)
        link = build_linkage(correlation_distance(corr), method=self.linkage_method)
        assignment = optimal_clusters(
            corr, link, max_clusters=self.max_clusters, n_clusters=self.n_clusters
        )
        if self.constraints.has_layer_limits:
            warnings.warn(
                "HERC allocates down its own correlation-derived clusters, "
                "which generally disagree with a hand-specified grouping. The "
                "layered bucket budgets will be met by projecting the result, "
                "which moves it away from HERC's own answer.",
                stacklevel=3,
            )

        cluster_weights = self._split_clusters(link, assignment.labels)
        allocation = pd.Series(0.0, index=assets)
        for label, members in assignment.members.items():
            intra = inverse_variance_weights(self.cov_matrix, members)
            allocation.loc[members] = intra.values * float(cluster_weights.get(label, 0.0))

        total = float(allocation.sum())
        if total <= 0:
            raise RuntimeError(
                "HERC produced an all-zero allocation; the cluster risk "
                "measure returned no usable values."
            )
        allocation = allocation / total

        self._record_diagnostics(assignment, cluster_weights)
        projected, distance = project_to_constraints(
            allocation.values, assets, self.constraints
        )
        self._diagnostics["projection_distance"] = distance
        if distance > 1e-6:
            self._diagnostics["bounds_note"] = (
                f"Constraints moved {distance:.2%} of the book away from the "
                "raw HERC allocation."
            )
        return projected

    def _record_diagnostics(self, assignment, cluster_weights: pd.Series) -> None:
        self._diagnostics.update(
            {
                "herc_linkage": self.linkage_method,
                "herc_risk_measure": self.risk_measure,
                "herc_clusters": {
                    str(label): members for label, members in assignment.members.items()
                },
                "herc_n_clusters": assignment.n_clusters,
                "herc_silhouette": assignment.silhouette,
                "herc_cluster_weights": {
                    str(label): float(cluster_weights.get(label, 0.0))
                    for label in assignment.members
                },
                "herc_note": (
                    f"Cut into {assignment.n_clusters} clusters (mean silhouette "
                    f"{assignment.silhouette:.2f}); capital split between them by "
                    f"equal {self.risk_measure} contribution, then naive risk "
                    "parity inside each."
                ),
            }
        )
