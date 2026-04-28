"""Small state helpers for Streamlit UI reruns."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Hashable

import pandas as pd

YAHOO_PRICES_CACHE_KEY = "yahoo_prices_cache"


def yahoo_cache_key(
    tickers: tuple[str, ...],
    period: str,
    start: str | None,
    end: str | None,
    interval: str,
) -> tuple[Hashable, ...]:
    """Build a stable key for the Yahoo inputs that define a price download."""
    return (tuple(tickers), period, start, end, interval)


def yahoo_prices_for_rerun(
    *,
    fetch_clicked: bool,
    cache_key: tuple[Hashable, ...],
    state: MutableMapping[str, object],
    fetch_prices: Callable[[], pd.DataFrame],
) -> pd.DataFrame | None:
    """Return fetched Yahoo prices, reusing them after Streamlit button reruns."""
    cached = state.get(YAHOO_PRICES_CACHE_KEY)
    if (
        not fetch_clicked
        and isinstance(cached, dict)
        and cached.get("key") == cache_key
        and isinstance(cached.get("prices"), pd.DataFrame)
    ):
        return cached["prices"]

    if not fetch_clicked:
        return None

    prices = fetch_prices()
    state[YAHOO_PRICES_CACHE_KEY] = {"key": cache_key, "prices": prices}
    return prices


from optimization_engine.optimizers.requirements import (  # noqa: E402
    MethodRequirements,
    requirements_for,
)


_NOT_USED = "Not used by this optimizer."


def _state(enabled: bool, tooltip: str | None = None) -> dict[str, object]:
    return {"enabled": enabled, "tooltip": tooltip if not enabled else None}


def derive_widget_state(method_name: str) -> dict[str, dict[str, object]]:
    """Map widget keys to enabled/tooltip state for the given optimizer.

    Pure function — used by the Streamlit app and easy to unit-test.
    """
    req: MethodRequirements = requirements_for(method_name)
    extra_keys = {e.key for e in req.extras}

    state: dict[str, dict[str, object]] = {
        "risk_free_rate": _state(
            req.supports_risk_free_rate,
            f"{_NOT_USED} (risk-free rate)",
        ),
        "cov_method": _state(
            req.requires_cov,
            f"{_NOT_USED} (no covariance estimate needed)",
        ),
        "ewma_lambda": _state(
            req.requires_cov,
            f"{_NOT_USED} (no covariance estimate needed)",
        ),
        "expected_returns_column": _state(
            req.requires_mu,
            f"{method_name} doesn't use expected returns.",
        ),
        "expected_returns_method": _state(
            req.requires_mu,
            f"{method_name} doesn't use expected returns.",
        ),
        "group_bounds": _state(
            req.supports_group_bounds,
            f"{method_name} does not enforce group bounds.",
        ),
        "frontier": _state(
            req.supports_frontier,
            "Frontier sweep is only meaningful for mean-variance / Black-Litterman.",
        ),
        "target_return": _state(
            req.supports_target_return,
            f"{method_name} does not accept a return target.",
        ),
        "target_volatility": _state(
            req.supports_target_volatility,
            f"{method_name} does not accept a volatility target.",
        ),
        "risk_aversion": _state(
            req.supports_risk_aversion,
            f"{method_name} does not use a risk-aversion utility.",
        ),
        "soft_bounds_caption": _state(
            req.bounds_mode != "hard",
            "Hard bounds — no soft-bounds caption shown.",
        ),
    }

    # Optimizer-specific extras: enabled iff present in this method's extras.
    for extra_key in (
        "risk_budget", "bl_views", "bl_view_confidences",
        "bl_tau", "bl_market_caps", "cvar_alpha", "hrp_linkage",
    ):
        state[extra_key] = _state(
            extra_key in extra_keys,
            f"Used only by methods that expose '{extra_key}'.",
        )
    return state
