"""The layered-constraint builder shown before an optimization is run.

A mandate has levels — 60% equity, and inside equity 40% developed; 30% in
foreign currency across all of it — and until the analyst can type those
levels in the app they end up approximated with per-asset bounds that do not
bind on the aggregate at all.

This module renders that policy as a stack of layers the user can add,
rename, re-bucket and remove, and hands back a list of
:class:`~optimization_engine.constraints.ConstraintLayer` objects for the
solve. Everything that can be decided without Streamlit lives in
``optimization_engine.ui_state`` so it can be tested without a browser; what
is here is the drawing and the session-state bookkeeping.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from optimization_engine.constraints import (
    BASIS_PARENT,
    BASIS_PORTFOLIO,
    LEGACY_LAYER_NAME,
    ConstraintLayer,
    layer_exposures,
)
from optimization_engine.ui_state import (
    LAYER_PRESETS,
    UNASSIGNED,
    assignment_from_source,
    layer_headroom,
    layer_state_from_layer,
    layer_state_to_layer,
    layer_states_to_layers,
    new_layer_state,
    policy_table,
    preset_by_label,
    sync_layer_state,
    unique_layer_name,
)

#: Where the editor's per-layer state lives.
STATE_KEY = "constraint_layer_states"
_UID_KEY = "constraint_layer_next_uid"

_BASIS_LABELS = {
    BASIS_PORTFOLIO: "% of total portfolio",
    BASIS_PARENT: "% of the parent layer's bucket",
}

_SOURCE_LABELS = {
    "group": "Copy the Group column",
    "currency_local_foreign": "Local vs foreign FX",
    "currency": "One bucket per currency",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _states() -> list[dict]:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = []
    return st.session_state[STATE_KEY]


def _next_uid() -> int:
    uid = int(st.session_state.get(_UID_KEY, 1))
    st.session_state[_UID_KEY] = uid + 1
    return uid


def _forget_widgets(uid: int) -> None:
    """Drop a layer's keyed-editor state.

    Streamlit's ``data_editor`` remembers edits against the widget key, not
    against the frame it was handed. Without this, filling the assignments
    from the Currency column would redraw underneath edits the user made
    before pressing the button, and the old values would win.
    """
    for prefix in ("layer_assign_", "layer_limits_"):
        st.session_state.pop(f"{prefix}{uid}", None)


def seed_layers_from_config(config) -> None:
    """Load a saved scenario's layers into the editor.

    Called when a scenario or config file is loaded, so that reopening a saved
    mandate shows the policy that was saved rather than an empty builder.
    """
    for state in st.session_state.get(STATE_KEY, []):
        _forget_widgets(state.get("uid", 0))
    layers = list(getattr(config, "constraint_layers", []) or [])
    # Fresh uids, so a keyed editor from the previous policy cannot be
    # matched against a layer it knows nothing about.
    base = int(st.session_state.get(_UID_KEY, 1))
    st.session_state[STATE_KEY] = [
        layer_state_from_layer(lyr, uid=base + i) for i, lyr in enumerate(layers)
    ]
    st.session_state[_UID_KEY] = base + len(layers)


def current_layers(assets) -> list[ConstraintLayer]:
    """The layers the editor currently describes, ready for the solve."""
    return layer_states_to_layers(
        sync_layer_state(state, assets) for state in _states()
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_layer_builder(
    assets,
    groups: dict,
    currencies: dict,
    base_currency: str,
    base_layer_name: str = LEGACY_LAYER_NAME,
    base_layer_limits: dict | None = None,
) -> list[ConstraintLayer]:
    """Draw the whole builder and return the layers it describes.

    Args:
        assets: The investable universe, in panel order.
        groups: The ``asset -> group`` map from the assets table. Used by the
            "copy the Group column" shortcut and offered as a parent layer.
        currencies: ``asset -> ISO code``, for the FX shortcuts.
        base_currency: What counts as local currency.
        base_layer_name: What the first (Group-column) layer is called.
        base_layer_limits: That layer's ``bucket -> (lo, hi)``, so a relative
            layer can show what it nests inside.
    """
    assets = [str(a) for a in assets]
    states = _states()

    st.markdown("**Layered allocation limits**")
    st.caption(
        "A mandate has levels. Layer 1 is the Group column above — the asset "
        "classes and their budgets. Add a layer for anything that slices the "
        "book a *different* way: developed vs emerging inside each class, "
        "local vs foreign currency across all of them, a regional overlay. "
        "Every layer binds at the same time, and each one stays a linear "
        "constraint, so nothing here costs the optimizer its guarantees."
    )

    _render_base_layer(assets, groups, base_layer_name, base_layer_limits)

    kept: list[dict] = []
    for position, state in enumerate(states):
        synced = sync_layer_state(state, assets)
        parent_options = _parent_options(base_layer_name, kept, synced["name"])
        updated, removed = _render_layer(
            synced,
            position=position,
            assets=assets,
            groups=groups,
            currencies=currencies,
            base_currency=base_currency,
            parent_options=parent_options,
            taken_names=[base_layer_name] + [s["name"] for s in kept],
        )
        if removed:
            # Redraw straight away rather than finishing a page that still
            # shows the layer half-rendered above the one being removed.
            st.session_state[STATE_KEY] = kept + states[position + 1 :]
            _forget_widgets(synced["uid"])
            st.rerun()
        kept.append(updated)
    if kept != states:
        st.session_state[STATE_KEY] = kept

    _render_add_controls(assets, groups, currencies, base_currency, kept)

    layers = layer_states_to_layers(kept)
    _render_policy_summary(layers, assets, base_layer_name, groups, base_layer_limits)
    return layers


def _parent_options(
    base_layer_name: str, earlier: list[dict], own_name: str
) -> list[str]:
    """Layers a relative layer may nest inside: the base plus everything above it.

    Only earlier layers are offered, which keeps the hierarchy acyclic without
    having to detect cycles: a layer can never point at itself or at one
    defined below it.
    """
    return [base_layer_name] + [s["name"] for s in earlier if s["name"] != own_name]


def _render_base_layer(
    assets, groups: dict, name: str, limits: dict | None
) -> None:
    """A read-only line naming the layer the Group column already provides."""
    limits = limits or {}
    buckets = sorted({str(g) for g in groups.values() if str(g).strip()})
    if not buckets:
        return
    capped = sum(
        1 for b in buckets if b in limits and tuple(limits[b]) != (0.0, 1.0)
    )
    st.caption(
        f"**Layer 1 · {name}** — {len(buckets)} bucket(s) from the Group "
        f"column, {capped} of them capped in the table above."
    )


def _render_layer(
    state: dict,
    position: int,
    assets,
    groups: dict,
    currencies: dict,
    base_currency: str,
    parent_options: list[str],
    taken_names: list[str],
) -> tuple[dict, bool]:
    """Draw one layer's editor. Returns ``(new_state, removed)``."""
    uid = state["uid"]
    label = state["name"] or f"Layer {position + 2}"
    n_capped = sum(
        1 for b, (lo, hi) in state["limits"].items() if lo > 0.0 or hi < 1.0
    )
    basis_note = (
        f" · % of {state['parent']}" if state["basis"] == BASIS_PARENT else ""
    )
    with st.expander(
        f"Layer {position + 2} · {label} — {n_capped} limit(s){basis_note}",
        expanded=True,
    ):
        head_l, head_m, head_r = st.columns([3, 3, 1])
        with head_l:
            new_name = st.text_input(
                "Layer name",
                value=state["name"],
                key=f"layer_name_{uid}",
                help="How this level of the policy is labelled everywhere else.",
            ).strip()
            if new_name and new_name != state["name"]:
                state["name"] = unique_layer_name(
                    new_name, [n for n in taken_names if n != state["name"]]
                )
            elif not new_name:
                state["name"] = state["name"] or f"Layer {position + 2}"
        with head_m:
            basis_choice = st.radio(
                "Limits are read as",
                options=list(_BASIS_LABELS),
                format_func=lambda b: _BASIS_LABELS[b],
                index=list(_BASIS_LABELS).index(state["basis"]),
                key=f"layer_basis_{uid}",
                horizontal=False,
                help=(
                    "“40% developed” means 40% of the whole book, or 40% of "
                    "the equity sleeve. They are different constraints, and "
                    "the second one moves with whatever the optimizer "
                    "allocates to equity."
                ),
            )
            state["basis"] = basis_choice
            if basis_choice == BASIS_PARENT:
                if state["parent"] not in parent_options:
                    state["parent"] = parent_options[0]
                state["parent"] = st.selectbox(
                    "…of which layer",
                    options=parent_options,
                    index=parent_options.index(state["parent"]),
                    key=f"layer_parent_{uid}",
                )
        with head_r:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"layer_del_{uid}", help="Remove this layer"):
                return state, True

        buckets_text = st.text_input(
            "Buckets (comma-separated)",
            value=", ".join(state["buckets"]),
            key=f"layer_buckets_{uid}",
            help=(
                "The categories this layer splits the universe into. Rename "
                "one here and the assignments follow."
            ),
        )
        typed = [b.strip() for b in buckets_text.split(",") if b.strip()]
        if typed != state["buckets"]:
            state = _rename_buckets(state, typed)
            _forget_widgets(uid)

        _render_fill_shortcuts(state, assets, groups, currencies, base_currency)

        state = sync_layer_state(state, assets)
        if not state["buckets"]:
            st.info("Name at least one bucket to start assigning assets to it.")
            return state, False

        assign_col, limit_col = st.columns([1, 1])
        with assign_col:
            st.caption("**Assign each asset to a bucket**")
            frame = pd.DataFrame(
                {"Bucket": [state["assignments"][a] for a in assets]}, index=assets
            )
            edited = st.data_editor(
                frame,
                width="stretch",
                num_rows="fixed",
                height=min(38 * (len(assets) + 1) + 3, 320),
                key=f"layer_assign_{uid}",
                column_config={
                    "Bucket": st.column_config.SelectboxColumn(
                        "Bucket",
                        options=[UNASSIGNED] + state["buckets"],
                        required=True,
                        help=(
                            f"“{UNASSIGNED}” leaves the asset outside this "
                            "layer entirely — which is right for anything the "
                            "layer has nothing to say about."
                        ),
                    )
                },
            )
            state["assignments"] = {
                a: str(edited.loc[a, "Bucket"]) for a in assets
            }

        with limit_col:
            unit = (
                f"% of {state['parent']}"
                if state["basis"] == BASIS_PARENT
                else "% of portfolio"
            )
            st.caption(f"**Limits per bucket** ({unit})")
            limits_frame = pd.DataFrame(
                {
                    "Min": [float(state["limits"][b][0]) for b in state["buckets"]],
                    "Max": [float(state["limits"][b][1]) for b in state["buckets"]],
                },
                index=state["buckets"],
            )
            edited_limits = st.data_editor(
                limits_frame,
                width="stretch",
                num_rows="fixed",
                key=f"layer_limits_{uid}",
                column_config={
                    "Min": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.05, format="%.2f"
                    ),
                    "Max": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.05, format="%.2f"
                    ),
                },
            )
            state["limits"] = {
                b: (
                    float(edited_limits.loc[b, "Min"]),
                    float(edited_limits.loc[b, "Max"]),
                )
                for b in state["buckets"]
            }
            st.caption("A bucket left at 0.00–1.00 imposes nothing and is dropped.")

        _render_layer_health(state, assets)
    return state, False


def _rename_buckets(state: dict, typed: list[str]) -> dict:
    """Follow a bucket rename through the assignments and the limits.

    Renaming positionally rather than by identity is what lets "DM" become
    "Developed" without every asset falling out of the bucket.
    """
    old = list(state["buckets"])
    mapping = {o: n for o, n in zip(old, typed)}
    state = dict(state)
    state["buckets"] = list(dict.fromkeys(typed))
    state["assignments"] = {
        a: mapping.get(b, b if b in typed else UNASSIGNED)
        for a, b in state["assignments"].items()
    }
    state["limits"] = {
        mapping.get(b, b): v for b, v in state["limits"].items()
    }
    return state


def _render_fill_shortcuts(
    state: dict, assets, groups: dict, currencies: dict, base_currency: str
) -> None:
    """One-click ways to fill the assignment column."""
    uid = state["uid"]
    cols = st.columns(len(_SOURCE_LABELS) + 1)
    for col, (source, label) in zip(cols, _SOURCE_LABELS.items()):
        with col:
            if st.button(label, key=f"layer_fill_{source}_{uid}", width="stretch"):
                assignments = assignment_from_source(
                    source, assets, currencies, base_currency, groups
                )
                buckets = [
                    b for b in dict.fromkeys(assignments.values()) if b != UNASSIGNED
                ]
                state["buckets"] = buckets
                state["assignments"] = assignments
                state["limits"] = {
                    b: tuple(state["limits"].get(b, (0.0, 1.0))) for b in buckets
                }
                _forget_widgets(uid)
                st.rerun()
    with cols[-1]:
        if st.button("Clear", key=f"layer_fill_clear_{uid}", width="stretch"):
            state["assignments"] = {str(a): UNASSIGNED for a in assets}
            _forget_widgets(uid)
            st.rerun()


def _render_layer_health(state: dict, assets) -> None:
    """The arithmetic an analyst gets wrong first, checked as they type.

    Not a substitute for the feasibility panel at the bottom of the tab — that
    one solves the actual LP — but it catches the two mistakes that produce an
    unhelpful "infeasible" before the solve is ever attempted.
    """
    health = layer_headroom(state, assets)
    whole = "the portfolio" if state["basis"] == BASIS_PORTFOLIO else state["parent"]
    if health["unassigned"]:
        n = len(health["unassigned"])
        st.caption(
            f"↳ {n} asset(s) unassigned — this layer says nothing about them."
        )
    elif health["cap_total"] < 1.0 - 1e-9:
        st.warning(
            f"Caps sum to {health['cap_total']:.0%} while every asset is in "
            f"this layer, so at most {health['cap_total']:.0%} of {whole} can "
            "be filled. Raise a cap or leave some assets out of the layer."
        )
    if health["floor_total"] > 1.0 + 1e-9:
        st.error(
            f"Minimums sum to {health['floor_total']:.0%} — more than all of "
            f"{whole}. Lower them."
        )


def _render_add_controls(
    assets, groups: dict, currencies: dict, base_currency: str, kept: list[dict]
) -> None:
    """The add-a-layer row, with presets that land on a usable layer."""
    left, right = st.columns([3, 1])
    with left:
        labels = [p.label for p in LAYER_PRESETS]
        choice = st.selectbox(
            "Add a layer",
            options=labels,
            key="layer_preset_choice",
            help=preset_by_label(
                st.session_state.get("layer_preset_choice", labels[0])
            ).help,
        )
    with right:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("➕ Add layer", key="layer_add_btn", width="stretch"):
            preset = preset_by_label(choice)
            st.session_state[STATE_KEY] = kept + [
                new_layer_state(
                    preset,
                    assets,
                    currencies=currencies,
                    base_currency=base_currency,
                    groups=groups,
                    existing_names=[LEGACY_LAYER_NAME]
                    + [s["name"] for s in kept],
                    uid=_next_uid(),
                )
            ]
            st.rerun()
    st.caption(preset_by_label(choice).help)


def _render_policy_summary(
    layers: list[ConstraintLayer],
    assets,
    base_layer_name: str,
    groups: dict,
    base_layer_limits: dict | None,
) -> None:
    """The whole mandate on one screen, the way it would be signed off."""
    all_layers = []
    base_limits = {
        b: tuple(v)
        for b, v in (base_layer_limits or {}).items()
        if tuple(v) != (0.0, 1.0)
    }
    if groups and base_limits:
        all_layers.append(
            ConstraintLayer(
                name=base_layer_name, assignments=dict(groups), limits=base_limits
            )
        )
    all_layers.extend(layers)
    if not all_layers:
        return
    table = policy_table(all_layers, assets)
    with st.expander("The policy in one table", expanded=False):
        st.dataframe(
            table.style.format({"Min": "{:.0%}", "Max": "{:.0%}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "“Of” says what each limit is a percentage *of*. A limit read as a "
            "share of a parent bucket moves with that bucket, so its cap on "
            "the whole book is only known once the optimizer has run — the "
            "Optimize tab reports both."
        )


def render_layer_exposures(run) -> None:
    """Post-solve: where the book landed on every layer, and what stopped it.

    The binding rows are the answer to "why this portfolio": a book that
    stopped at 60% equity because the asset-class cap says so is a different
    portfolio from one that stopped there because the EM sub-limit ran out.
    """
    try:
        exposures = run.layer_exposures()
    except Exception:  # a policy the run cannot describe must not kill the tab
        return
    if exposures.empty:
        return

    st.markdown("**Policy exposures**")
    binding = exposures[exposures["binding"]]
    if not binding.empty:
        names = ", ".join(f"{r.layer} · {r.bucket}" for r in binding.itertuples())
        st.caption(f"Binding: {names}.")
    else:
        st.caption(
            "No bucket limit is binding — this allocation was chosen by the "
            "objective, not by the policy."
        )

    display = exposures.copy()
    display["Limit"] = [
        (
            f"{row.min:.0%}–{row.max:.0%}"
            + (f" of {row.parent}" if row.basis == "parent" else "")
        )
        for row in exposures.itertuples()
    ]
    display = display.rename(
        columns={
            "layer": "Layer",
            "bucket": "Bucket",
            "weight": "Weight",
            "effective_max": "Cap (of book)",
            "headroom": "Headroom",
            "binding": "Binding",
        }
    )
    st.dataframe(
        display[
            ["Layer", "Bucket", "Weight", "Limit", "Cap (of book)", "Headroom", "Binding"]
        ].style.format(
            {
                "Weight": "{:.2%}",
                "Cap (of book)": "{:.2%}",
                "Headroom": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


__all__ = [
    "STATE_KEY",
    "current_layers",
    "layer_exposures",
    "layer_state_to_layer",
    "render_layer_builder",
    "render_layer_exposures",
    "seed_layers_from_config",
]
