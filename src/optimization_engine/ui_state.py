"""Small state helpers for Streamlit UI reruns."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass

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
        "benchmark_limits": _state(
            req.supports_benchmark_limits,
            f"{method_name} cannot impose a tracking-error or active-share "
            "budget inside its solve — the limit would be reported in the "
            "compliance panel but never bind. Use mean-variance, minimum "
            "variance or active mean-variance.",
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
        "cdar_alpha", "cluster_linkage", "n_clusters", "max_clusters",
        "herc_risk_measure", "nco_objective", "nco_detone_for_clustering",
    ):
        state[extra_key] = _state(
            extra_key in extra_keys,
            f"Used only by methods that expose '{extra_key}'.",
        )
    return state


# ---------------------------------------------------------------------------
# Layered constraint builder
# ---------------------------------------------------------------------------

from optimization_engine.constraints import (  # noqa: E402
    BASIS_PARENT,
    BASIS_PORTFOLIO,
    LEGACY_LAYER_NAME,
    ConstraintLayer,
)

#: What an unassigned asset shows as in the editor. The engine drops it.
UNASSIGNED = "—"

#: Ways a layer's asset→bucket map can be filled in one click. Typing a bucket
#: for forty tickers by hand is where a policy editor stops being used.
ASSIGNMENT_SOURCES = (
    "manual",
    "group",
    "currency",
    "currency_local_foreign",
)


@dataclass
class LayerPreset:
    """A starting point for a new layer.

    Presets exist because the first minute decides whether an analyst builds
    the policy or gives up and goes back to per-asset bounds. Each one lands
    on a layer that is already named, already bucketed and — where the data
    allows it — already assigned.

    Attributes:
        label: What the "add layer" menu shows.
        name: Default layer name.
        buckets: Bucket names to start with.
        limits: Default ``bucket -> (min, max)``; buckets not named here start
            uncapped at ``(0, 1)``.
        basis: ``portfolio`` or ``parent``.
        parent: Default parent layer name for a relative layer.
        source: How to seed the assignments — see :data:`ASSIGNMENT_SOURCES`.
        help: One line explaining what the layer is for.
    """

    label: str
    name: str
    buckets: tuple[str, ...] = ()
    limits: dict = None  # type: ignore[assignment]
    basis: str = BASIS_PORTFOLIO
    parent: str | None = None
    source: str = "manual"
    help: str = ""

    def __post_init__(self) -> None:
        if self.limits is None:
            self.limits = {}


LAYER_PRESETS: tuple[LayerPreset, ...] = (
    LayerPreset(
        label="Sub-asset class (DM / EM)",
        name="Sub-asset class",
        buckets=("DM Equity", "EM Equity", "DM Fixed Income", "EM Fixed Income"),
        limits={
            "DM Equity": (0.0, 0.40),
            "EM Equity": (0.0, 0.20),
            "DM Fixed Income": (0.0, 0.20),
            "EM Fixed Income": (0.0, 0.10),
        },
        help=(
            "The second level of the policy: developed and emerging inside "
            "each asset class."
        ),
    ),
    LayerPreset(
        label="Currency — local vs foreign",
        name="FX exposure",
        buckets=("Local FX", "Foreign FX"),
        limits={"Local FX": (0.0, 0.70), "Foreign FX": (0.0, 0.30)},
        source="currency_local_foreign",
        help=(
            "Splits the book by the currency each series is quoted in, "
            "relative to the base currency. Assigned automatically."
        ),
    ),
    LayerPreset(
        label="Currency — one bucket per currency",
        name="Currency",
        source="currency",
        help="A bucket per currency in the panel, assigned automatically.",
    ),
    LayerPreset(
        label="Region",
        name="Region",
        buckets=("North America", "Europe", "Asia-Pacific", "Latin America", "Other"),
        help="A geographic overlay that cuts across asset classes.",
    ),
    LayerPreset(
        label="Blank layer",
        name="New layer",
        buckets=("Bucket A", "Bucket B"),
        help="Start from nothing and name the buckets yourself.",
    ),
)


def preset_by_label(label: str) -> LayerPreset:
    for preset in LAYER_PRESETS:
        if preset.label == label:
            return preset
    raise KeyError(f"No layer preset labelled {label!r}.")


def assignment_from_source(
    source: str,
    assets: Sequence[str],
    currencies: Mapping[str, str] | None = None,
    base_currency: str = "USD",
    groups: Mapping[str, str] | None = None,
) -> dict:
    """Seed a layer's ``asset -> bucket`` map from data the app already has.

    Returns an empty-ish map (everything unassigned) for ``"manual"``.
    """
    currencies = dict(currencies or {})
    groups = dict(groups or {})
    base = str(base_currency).upper()
    out: dict = {}
    for asset in assets:
        key = str(asset)
        if source == "group":
            out[key] = str(groups.get(key, UNASSIGNED))
        elif source == "currency":
            out[key] = str(currencies.get(key, base)).upper()
        elif source == "currency_local_foreign":
            ccy = str(currencies.get(key, base)).upper()
            out[key] = "Local FX" if ccy == base else "Foreign FX"
        else:
            out[key] = UNASSIGNED
    return out


def new_layer_state(
    preset: LayerPreset,
    assets: Sequence[str],
    currencies: Mapping[str, str] | None = None,
    base_currency: str = "USD",
    groups: Mapping[str, str] | None = None,
    existing_names: Sequence[str] = (),
    uid: int = 0,
) -> dict:
    """Build the editable session-state dict for a new layer from a preset."""
    assignments = assignment_from_source(
        preset.source, assets, currencies, base_currency, groups
    )
    buckets = list(preset.buckets)
    if preset.source in ("currency", "currency_local_foreign"):
        # The data decides the buckets here, and an empty one would only
        # confuse the limits table.
        seen = [b for b in dict.fromkeys(assignments.values()) if b != UNASSIGNED]
        buckets = seen or buckets
    limits = {
        b: tuple(preset.limits.get(b, (0.0, 1.0)))  # type: ignore[union-attr]
        for b in buckets
    }
    return {
        "uid": int(uid),
        "name": unique_layer_name(preset.name, existing_names),
        "basis": preset.basis,
        "parent": preset.parent or LEGACY_LAYER_NAME,
        "buckets": buckets,
        "assignments": assignments,
        "limits": limits,
    }


def unique_layer_name(name: str, existing: Sequence[str]) -> str:
    """``name``, suffixed if a layer already goes by it.

    Layer names are how a relative layer points at its parent, so two layers
    sharing one is not a cosmetic problem.
    """
    taken = {str(n) for n in existing}
    if name not in taken:
        return name
    for i in range(2, 100):
        candidate = f"{name} {i}"
        if candidate not in taken:
            return candidate
    return f"{name} {len(taken) + 1}"


def sync_layer_state(state: Mapping, assets: Sequence[str]) -> dict:
    """Reconcile a layer's state with the current universe and bucket list.

    Assets that left the panel are dropped, new ones arrive unassigned, and an
    assignment pointing at a bucket the user has since renamed away falls back
    to unassigned rather than silently constraining nothing.
    """
    buckets = [str(b) for b in state.get("buckets", []) if str(b).strip()]
    buckets = list(dict.fromkeys(buckets))
    old = dict(state.get("assignments") or {})
    assignments = {}
    for asset in assets:
        current = str(old.get(str(asset), UNASSIGNED))
        assignments[str(asset)] = current if current in buckets else UNASSIGNED
    old_limits = dict(state.get("limits") or {})
    limits = {b: tuple(old_limits.get(b, (0.0, 1.0))) for b in buckets}
    return {
        "uid": int(state.get("uid", 0)),
        "name": str(state.get("name") or "Layer"),
        "basis": str(state.get("basis") or BASIS_PORTFOLIO),
        "parent": state.get("parent") or LEGACY_LAYER_NAME,
        "buckets": buckets,
        "assignments": assignments,
        "limits": limits,
    }


def layer_state_to_layer(state: Mapping) -> ConstraintLayer | None:
    """Convert one editor state into a :class:`ConstraintLayer`.

    Returns ``None`` when the layer constrains nothing — no buckets, nothing
    assigned, or every limit left at the full 0–100% range. A layer that binds
    nothing is noise in the feasibility report and in the compliance panel, so
    it is dropped rather than carried.
    """
    assignments = {
        a: b
        for a, b in dict(state.get("assignments") or {}).items()
        if b and b != UNASSIGNED
    }
    if not assignments:
        return None
    used = set(assignments.values())
    limits = {}
    for bucket, value in dict(state.get("limits") or {}).items():
        if bucket not in used:
            continue
        lo, hi = float(value[0]), float(value[1])
        if lo <= 0.0 and hi >= 1.0:
            continue
        limits[bucket] = (lo, hi)
    if not limits:
        return None
    basis = str(state.get("basis") or BASIS_PORTFOLIO)
    return ConstraintLayer(
        name=str(state.get("name") or "Layer"),
        assignments=assignments,
        limits=limits,
        basis=basis,
        parent=(state.get("parent") if basis == BASIS_PARENT else None),
    )


def layer_states_to_layers(states: Iterable[Mapping]) -> list[ConstraintLayer]:
    """Every editor state that actually constrains something, in order."""
    out = []
    for state in states or []:
        layer = layer_state_to_layer(state)
        if layer is not None:
            out.append(layer)
    return out


def layer_state_from_layer(layer: ConstraintLayer, uid: int = 0) -> dict:
    """Round-trip a saved layer back into editor state."""
    buckets = layer.buckets()
    return {
        "uid": int(uid),
        "name": layer.name,
        "basis": layer.basis,
        "parent": layer.parent or LEGACY_LAYER_NAME,
        "buckets": buckets,
        "assignments": dict(layer.assignments),
        "limits": {b: tuple(layer.limits.get(b, (0.0, 1.0))) for b in buckets},
    }


def layer_headroom(state: Mapping, assets: Sequence[str]) -> dict:
    """Whether a portfolio-basis layer's caps can add up to a full book.

    The arithmetic an analyst gets wrong first: caps of 60/30/10 across every
    asset leave exactly no slack, and caps of 50/20/10 cannot reach 100% at
    all. Returns ``{covers_all, cap_total, floor_total, unassigned}``; for a
    relative layer the totals are shares of the parent instead.
    """
    synced = sync_layer_state(state, assets)
    assignments = synced["assignments"]
    unassigned = [a for a, b in assignments.items() if b == UNASSIGNED]
    used = {b for b in assignments.values() if b != UNASSIGNED}
    cap_total = sum(float(synced["limits"][b][1]) for b in used if b in synced["limits"])
    floor_total = sum(
        float(synced["limits"][b][0]) for b in used if b in synced["limits"]
    )
    return {
        "covers_all": not unassigned,
        "cap_total": cap_total,
        "floor_total": floor_total,
        "unassigned": unassigned,
    }


def policy_table(layers: Sequence[ConstraintLayer], assets: Sequence[str]):
    """One flat table of the whole policy: every layer, bucket, limit and count.

    This is the summary an allocator signs off on, so it states the limits in
    the units they were written in and says which layer each one belongs to.
    """
    rows = []
    for layer in layers:
        members = layer.members(assets)
        for bucket in layer.buckets():
            lo, hi = layer.limits.get(bucket, (0.0, 1.0))
            rows.append(
                {
                    "Layer": layer.name,
                    "Bucket": bucket,
                    "Min": float(lo),
                    "Max": float(hi),
                    "Of": (
                        f"{layer.parent}" if layer.is_relative else "portfolio"
                    ),
                    "Assets": len(members.get(bucket, [])),
                }
            )
    return pd.DataFrame(
        rows, columns=["Layer", "Bucket", "Min", "Max", "Of", "Assets"]
    )
