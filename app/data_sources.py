"""The data step: choosing a provider, fetching a panel, and seeing what came back.

The old data step offered three radio buttons and, when one of them failed,
a red box. This one is built around a different premise: the analyst should be
able to answer *before* running anything — can this source give me what I am
about to ask it for? — and *after* — what actually arrived, from where, and
what is missing?

So the picker shows each provider's capabilities and key status up front, the
field selector disables what the chosen provider cannot serve rather than
letting the run fail later, and the result is rendered as a per-identifier log
next to a coverage table. Missing volume is called out as an expected property
of index data rather than as an error, because that is what it is.

Every function here is a render function: it draws, reads widget state, and
returns what the page needs. The fetching itself lives in
:mod:`optimization_engine.ingest`.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import streamlit as st

from optimization_engine.data.loader import SAMPLE_UNIVERSE
from optimization_engine.ingest import (
    IngestError,
    IngestRequest,
    catalog_entries,
    describe_providers,
    entries_for,
    ingest,
)
from optimization_engine.ingest import fields as F
from optimization_engine.ui_state import ingest_result_for_rerun

#: Field presets, in the order the selector shows them. Each is a
#: ``(label, fields, help)`` triple; the label is what the analyst reads and
#: the fields are what the request carries.
FIELD_PRESETS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "Close only",
        F.PRICE_ONLY,
        "Everything the optimizer and the backtest need. Fastest, smallest, "
        "and the only thing some providers publish.",
    ),
    (
        "OHLC",
        F.OHLC,
        "Adds the intraday range. Useful for range-based volatility "
        "estimators; still needs no volume, so indices work.",
    ),
    (
        "OHLC + volume",
        F.OHLCV,
        "Adds traded volume, which is what lets the backtest price market "
        "impact from real capacity instead of an assumed participation rate.",
    ),
)

#: Universes worth one click. Real tickers, chosen so each row demonstrates a
#: different property of the pipeline rather than just being a list.
PRESET_UNIVERSES: dict[str, tuple[str, str]] = {
    "Synthetic asset classes": (
        ", ".join(SAMPLE_UNIVERSE),
        "The `sample` provider's own thirteen asset classes, with the "
        "correlation structure the engine's examples are written against.",
    ),
    "Global multi-asset ETFs": (
        "SPY, EFA, EEM, AGG, TLT, GLD, DBC, VNQ",
        "Liquid ETFs with real volume — the capacity-aware cost model works here.",
    ),
    "US sector ETFs": (
        "XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC",
        "Eleven sectors, one currency, deep books.",
    ),
    "World equity indices": (
        "SP500, STOXX50, NIKKEI225, FTSE100, IPC, BOVESPA",
        "Index levels in six currencies. No volume exists for any of them — "
        "the backtest prices impact from a fixed participation rate instead.",
    ),
    "Mexico + US": (
        "IPC, SP500",
        "A two-index pair in MXN and USD; set a base currency to compare them.",
    ),
}


#: Providers whose universe is whatever they are handed, so the symbol
#: catalog never applies to them.
_PASSTHROUGH_PROVIDERS = frozenset({"sample", "file"})


def _provider_label(row: dict[str, Any]) -> str:
    mark = "✓" if row["ready"] else "⚠"
    return f"{mark}  {row['provider']}"


def render_source_picker(state: Any) -> tuple[pd.DataFrame | None, Any]:
    """Draw the sidebar data step and return ``(prices, ingest_result)``.

    Returns ``(None, None)`` when nothing has been fetched yet, which the page
    reads as "show the empty state and stop".
    """
    rows = {row["provider"]: row for row in describe_providers()}
    names = list(rows)
    default = names.index("sample") if "sample" in names else 0

    provider = st.selectbox(
        "Provider",
        options=names,
        index=default,
        format_func=lambda name: _provider_label(rows[name]),
        help=(
            "Every provider returns the same column names, so switching one "
            "does not change anything downstream — only where the numbers "
            "come from."
        ),
    )
    row = rows[provider]
    st.caption(row["description"])
    _render_capability_chips(row)

    api_key = _render_key_input(row) if row["requires_key"] else None
    file_path = (
        st.text_input(
            "File path",
            help="CSV, Excel or Parquet, in wide (date × asset) or long "
                 "(date, identifier, OHLCV) layout.",
        )
        if provider == "file"
        else None
    )

    identifiers = _render_universe_input(provider)
    start, end, period = _render_window()
    fields = _render_field_selector(row)
    currency = _render_currency()

    with st.expander("Caching", expanded=False):
        cache_dir = st.text_input(
            "Cache directory",
            value="",
            help=(
                "Where fetched panels are stored so a re-run does not "
                "re-download. Leave empty to disable — nothing is written to "
                "disk unless you name a directory."
            ),
        )

    try:
        request = IngestRequest(
            identifiers=identifiers,
            provider=provider,
            start=start,
            end=end,
            period=period,
            fields=fields,
            currency=currency or None,
            cache_dir=cache_dir or None,
        )
    except IngestError as exc:
        st.error(str(exc))
        return None, None

    fetch_clicked = st.button("Fetch data", type="primary", width="stretch")

    options: dict[str, Any] = {}
    if file_path:
        options["path"] = file_path

    # An offline provider costs nothing to run: no request leaves the machine,
    # no quota is spent, no key is needed. Making the user press a button for
    # that only produces an empty first screen, so it loads on its own and the
    # button stays available for a re-run.
    if row["offline"] and (provider != "file" or file_path):
        fetch_clicked = True

    try:
        result = ingest_result_for_rerun(
            fetch_clicked=fetch_clicked,
            cache_key=f"{request.fingerprint()}:{bool(api_key)}:{file_path or ''}",
            state=state,
            fetch=lambda: ingest(request, api_key=api_key or None, **options),
        )
    except IngestError as exc:
        st.error(str(exc))
        return None, None
    except Exception as exc:  # a provider or network failure, not a bug
        st.error(f"Could not load data: {type(exc).__name__}: {exc}")
        return None, None

    if result is None:
        return None, None

    _render_fetch_summary(result)
    return result.prices, result


def _render_capability_chips(row: dict[str, Any]) -> None:
    """A one-line, scannable statement of what this provider can do.

    Fields arrive in canonical OHLCV order rather than alphabetical, because
    the point is to be read at a glance.
    """
    served = ", ".join(f.removeprefix("m_") for f in row["fields"])
    if not row["serves_volume"]:
        served += " — no volume"
    st.caption(f"**Serves** {served}  ·  **Bars** {', '.join(row['intervals'])}")


def _render_key_input(row: dict[str, Any]) -> str | None:
    """Let a key be pasted in-session rather than demanding a restart.

    The key is held in Streamlit's per-session state and passed straight to the
    provider. It is never written to disk, never logged, and never shown back.
    For a key that should persist, the environment variable named here is the
    right place.
    """
    if row["key_present"]:
        st.success(f"API key found in `{row['key_env_var']}`.", icon="🔑")
        return None

    st.warning(
        f"`{row['provider']}` needs an API key. Set `{row['key_env_var']}` in "
        "your environment or in a `.env` file, or paste one below for this "
        "session only.",
        icon="🔑",
    )
    if row["signup_url"]:
        st.caption(f"Get a free key: {row['signup_url']}")
    key = st.text_input(
        "API key (this session only)",
        type="password",
        help="Kept in memory for this browser session. Never written to disk.",
    )
    return key.strip() or None


#: What "Type my own" starts from, per provider. The synthetic provider's own
#: universe is its asset classes; a live provider's is a set of real tickers.
_DEFAULT_UNIVERSES: dict[str, str] = {
    "sample": ", ".join(SAMPLE_UNIVERSE),
    "fred": "SP500, NASDAQCOM, DJIA",
    "stooq": "SP500, DAX, FTSE100, NIKKEI225",
}
_DEFAULT_UNIVERSE = "SPY, AGG, GLD"


def _render_universe_input(provider: str) -> str:
    """The universe box, with catalog and preset shortcuts above it."""
    preset = st.selectbox(
        "Start from",
        options=["Type my own", *PRESET_UNIVERSES],
        help="A starting universe you can then edit freely.",
    )
    if preset in PRESET_UNIVERSES:
        default = PRESET_UNIVERSES[preset][0]
        st.caption(PRESET_UNIVERSES[preset][1])
    else:
        default = _DEFAULT_UNIVERSES.get(provider, _DEFAULT_UNIVERSE)

    identifiers = st.text_area(
        "Universe",
        value=default,
        height=80,
        key=f"universe_{provider}_{preset}",
        help=(
            "Comma- or space-separated. Catalog names such as SP500 or IPC "
            "are translated into whichever symbol the chosen provider uses, "
            "so the column keeps its name when you switch."
        ),
    )

    servable = entries_for(provider)
    passthrough = provider in _PASSTHROUGH_PROVIDERS
    if passthrough:
        st.caption(
            f"`{provider}` serves whatever names you type. Catalog names such "
            "as SP500 still work, and still come back marked as indices."
        )
    elif servable:
        with st.expander(f"Catalog: {len(servable)} indices this provider serves"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "name": entry.key,
                            "instrument": entry.name,
                            "ccy": entry.currency,
                            f"{provider} symbol": entry.symbol_for(provider),
                        }
                        for entry in servable
                    ]
                ).set_index("name"),
                width="stretch",
                height=min(320, 40 + 35 * len(servable)),
            )
    elif catalog_entries():
        st.caption(
            "No catalog index maps to this provider; type the tickers it uses "
            "directly and they pass through unchanged."
        )
    return identifiers


def _render_window() -> tuple[dt.date | None, dt.date | None, str | None]:
    """Period shorthand, or an explicit range."""
    # Eight years by default: long enough for a covariance estimate to mean
    # something across a full cycle, short enough to stay recent.
    choice = st.selectbox(
        "History",
        options=["1y", "2y", "3y", "5y", "8y", "10y", "20y", "Custom range"],
        index=4,
    )
    if choice != "Custom range":
        return None, None, choice

    today = dt.date.today()
    columns = st.columns(2)
    start = columns[0].date_input("From", value=today - dt.timedelta(days=5 * 365))
    end = columns[1].date_input("To", value=today)
    return start, end, None


def _render_field_selector(row: dict[str, Any]) -> tuple[str, ...]:
    """Offer only the presets this provider can actually serve.

    Disabling the impossible is the whole point: asking FRED for volume is a
    mistake that should be unavailable, not an error message after a fetch.
    """
    available = set(row["fields"])
    options = [
        label for label, fields, _ in FIELD_PRESETS if set(fields) <= available
    ]
    helps = {label: text for label, _, text in FIELD_PRESETS}
    fields_by_label = {label: fields for label, fields, _ in FIELD_PRESETS}

    choice = st.radio("Fields", options=options, index=0, horizontal=True)
    st.caption(helps[choice])
    if "OHLC + volume" not in options:
        st.caption(
            f"`{row['provider']}` publishes no volume, so a capacity-aware "
            "backtest on this data prices impact from a fixed participation "
            "rate. That is expected for index levels."
        )
    return fields_by_label[choice]


def _render_currency() -> str:
    from optimization_engine.data.fx import supported_currencies

    choice = st.selectbox(
        "Base currency",
        options=["Leave as quoted", *supported_currencies()],
        index=0,
        help=(
            "Converts every series into one currency at the price level, so "
            "the optimizer sees a homogeneous universe. Uses FRED daily rates."
        ),
    )
    return "" if choice == "Leave as quoted" else choice


def _render_fetch_summary(result: Any) -> None:
    """The two lines that say whether the fetch is worth trusting."""
    if result.is_complete:
        st.success(result.summary(), icon="✅")
    else:
        st.warning(result.summary(), icon="⚠️")
        for outcome in result.failed:
            st.caption(f"**{outcome.identifier}** — {outcome.status}: {outcome.message}")


def render_ingest_panel(result: Any) -> None:
    """The provenance block on the Data tab: what came from where.

    Rendered after every successful fetch, and deliberately not hidden behind
    an expander. A panel's provenance is not an advanced topic — it is the
    first thing anyone should be able to check.
    """
    if result is None:
        return

    request = result.request
    st.subheader("Where this data came from")
    columns = st.columns(4)
    columns[0].metric("Provider", request.provider)
    columns[1].metric(
        "Loaded", f"{len(result.loaded)}/{len(result.outcomes)}",
        help="Identifiers that returned a usable series.",
    )
    columns[2].metric("Bars", request.interval)
    columns[3].metric(
        "Volume",
        "yes" if result.panel.has_volume else "none",
        help=(
            "Whether any series carries traded volume. Without it the "
            "backtest prices market impact from a fixed participation rate."
        ),
    )

    for note in result.warnings:
        st.info(note, icon="ℹ️")

    coverage = result.panel.coverage()
    display = coverage.rename(
        columns={
            "provider": "Provider",
            "symbol": "Symbol",
            "kind": "Instrument",
            "currency": "Ccy",
            "observations": "Obs",
            "first_date": "From",
            "last_date": "To",
            "fields": "Fields",
            "has_volume": "Volume",
        }
    )
    for column in ("From", "To"):
        display[column] = pd.to_datetime(display[column]).dt.date
    # Rendered as text rather than a boolean: Streamlit draws a bool column as
    # a checkbox, which reads as something you can toggle.
    display["Volume"] = display["Volume"].map({True: "yes", False: "—"})
    st.dataframe(display, width="stretch")

    if not result.is_complete:
        with st.expander("Identifiers that did not load", expanded=True):
            st.dataframe(
                result.report().loc[[o.identifier for o in result.failed]],
                width="stretch",
            )

    st.caption(
        f"Request fingerprint `{request.fingerprint()}` · "
        f"{request.start} → {request.end} · "
        f"fields {', '.join(f.removeprefix('m_') for f in request.fields)}"
        + (" · served from cache" if result.from_cache else "")
    )


def render_liquidity_selector(volumes: pd.DataFrame | None) -> dict[str, Any]:
    """Choose how market impact is priced, and say what that choice rests on.

    Two honest options and no hidden third. The fixed rate is an assumption
    and is labelled as one; ADV is a measurement and needs data that not every
    universe has. When it is chosen without that data, the fallback is stated
    here rather than discovered in the run log afterwards.
    """
    has_volume = volumes is not None and not volumes.dropna(how="all").empty
    options = ["Fixed participation rate", "From traded volume (ADV)"]
    source = st.radio(
        "Liquidity model",
        options=options,
        index=0,
        horizontal=True,
        help=(
            "What the square-root impact law divides by. The fixed rate is an "
            "assumption about how much of the book you can trade in one name "
            "in one period; ADV measures it from the data."
        ),
    )
    use_adv = source == options[1]

    settings: dict[str, Any] = {
        "impact_participation_source": "adv" if use_adv else "fixed",
    }
    if use_adv:
        if has_volume:
            st.success(
                f"Volume available for {int(volumes.notna().any().sum())} of "
                f"{volumes.shape[1]} series.",
                icon="📈",
            )
        else:
            st.info(
                "This universe carries no volume — index levels never do — so "
                "every trade falls back to the fixed rate below and the run "
                "log records that it did.",
                icon="ℹ️",
            )
        settings["impact_adv_share"] = (
            st.slider(
                "Share of daily volume we are willing to be",
                min_value=1, max_value=50, value=10, step=1,
                format="%d%%",
                help="Capacity grows with this; so does the impact you would "
                     "really pay.",
            )
            / 100.0
        )
        settings["impact_adv_lookback"] = st.number_input(
            "ADV lookback (periods)", min_value=2, max_value=252, value=21,
        )
    return settings


def render_empty_state(awaiting_upload: bool = False) -> None:
    """What the page shows before anything has been loaded.

    A blank canvas under a title is the worst possible first screen: it says
    nothing about what to do or what the tool is for. This says both, in the
    space the data is about to occupy.
    """
    st.divider()
    if awaiting_upload:
        st.subheader("Drop a price file in the sidebar")
        st.markdown(
            "Excel, CSV or Parquet, either wide (a date index and one column "
            "per asset) or long (`date`, `identifier`, and any of `open`, "
            "`high`, `low`, `close`, `volume`).\n\n"
            "Prefer to fetch instead? Switch **Source** to **Data provider** "
            "— the `sample` provider needs no key and no network."
        )
        return

    st.subheader("Start by loading a universe")
    columns = st.columns(3)
    columns[0].markdown(
        "**1 · Pick a provider**\n\n"
        "`sample` needs no key and no network. `yahoo`, `stooq` and `fred` "
        "are keyless and live. `fmp` and `tiingo` take a free API key."
    )
    columns[1].markdown(
        "**2 · Name a universe**\n\n"
        "Tickers, or catalog names like `SP500` and `IPC` that resolve to "
        "whichever symbol your provider uses. Switching provider later does "
        "not rename a single column."
    )
    columns[2].markdown(
        "**3 · Press Fetch data**\n\n"
        "You get the panel plus a line per identifier saying what arrived, "
        "from where, and what is missing."
    )
    st.info(
        "Index universes work without volume. An index has none by "
        "construction, so the backtest prices market impact from a fixed "
        "participation rate and says so — rather than refusing to run.",
        icon="📈",
    )


__all__ = [
    "FIELD_PRESETS",
    "PRESET_UNIVERSES",
    "render_empty_state",
    "render_ingest_panel",
    "render_liquidity_selector",
    "render_source_picker",
]
