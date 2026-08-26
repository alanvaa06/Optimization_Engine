"""Reusable Streamlit rendering blocks for the optimization app.

Kept separate from ``streamlit_app.py`` so the page script stays a readable
sequence of steps rather than a wall of chart plumbing. Everything here is a
pure render function: it takes data and draws it, and never mutates session
state.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

PERCENT_COLUMNS = {
    "Annualized Return", "Annualized Vol", "Max Drawdown",
    "Annualized Excess", "Annualized T.E.", "Expected Return", "Expected Vol",
    "Historic VaR(5%)", "Cornish-Fisher VaR(5%)", "Historic CVaR(5%)",
    "In-sample (fitted)", "Out-of-sample (walk-forward)", "Degradation",
}
RATIO_COLUMNS = {
    "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Omega Ratio",
    "Tail Ratio", "Skewness", "Kurtosis", "Ulcer Index", "Beta",
    "Information Ratio", "Up Capture", "Down Capture", "Capture", "Sharpe",
}


def format_table(df: pd.DataFrame) -> Any:
    """Style a frame by column *name*, so new metrics format themselves.

    The previous approach passed a hand-written dict of column formats; every
    metric added to ``summary_stats`` then rendered as a raw float until
    someone remembered to update the dict in three places.
    """
    fmt: dict[str, str] = {}
    for col in df.columns:
        name = str(col)
        if name in PERCENT_COLUMNS or name.startswith(("Historic ", "Cornish")):
            fmt[col] = "{:.2%}"
        elif name in RATIO_COLUMNS:
            fmt[col] = "{:.3f}"
        elif name in ("Hit Rate", "Prob. Sharpe > 0"):
            fmt[col] = "{:.1%}"
        elif pd.api.types.is_float_dtype(df[col]):
            fmt[col] = "{:.4f}"
    try:
        return df.style.format(fmt)
    except Exception:
        return df


def metric_row(items: Iterable[tuple[str, str, str | None]]) -> None:
    """Render a row of ``st.metric`` cards from ``(label, value, help)``."""
    items = list(items)
    if not items:
        return
    for col, (label, value, helptext) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=helptext)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Diagnostics panels
# ---------------------------------------------------------------------------


def render_data_quality(report, expanded_default: bool = False) -> None:
    """Summarize a :class:`DataQualityReport` as a badge plus detail."""
    n_err = len(report.errors)
    n_warn = len(report.warnings)

    if n_err:
        st.error(
            f"**{n_err} data problem(s) that will corrupt the optimization.** "
            "Fix these before trusting any result below."
        )
    elif n_warn:
        st.warning(
            f"**{n_warn} data caveat(s).** The optimizer will run, but these "
            "affect what the answer means."
        )
    else:
        st.success("Data checks passed — no gaps, stale runs, or thin samples found.")

    if report.issues:
        with st.expander(
            f"Data-quality detail ({n_err} error, {n_warn} warning)"
            if n_err or n_warn else "Data-quality detail",
            expanded=expanded_default or bool(n_err),
        ):
            rows = [
                {
                    "Severity": i.severity.title(),
                    "Asset": i.asset or "—",
                    "Finding": i.message,
                    "What to do": i.suggestion,
                }
                for i in report.issues
            ]
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True
            )


def render_feasibility(report, show_ok: bool = True) -> bool:
    """Render a :class:`FeasibilityReport`. Returns True when solvable.

    This is the block that replaces ``Solver failed: status=infeasible`` with
    a sentence naming the constraint and the fix.
    """
    if report is None:
        return True
    if report.is_feasible and not report.issues:
        if show_ok:
            msg = "Constraints are feasible."
            if report.min_return is not None and report.max_return is not None:
                msg += (
                    f" Reachable expected return: "
                    f"{report.min_return:.2%} to {report.max_return:.2%}"
                )
                if report.min_variance_return is not None:
                    msg += (
                        f" (efficient above {report.min_variance_return:.2%})."
                    )
            st.success(msg)
        return True

    if not report.is_feasible:
        st.error("**These constraints have no solution.** Nothing will solve until one of these changes:")
        for issue in report.fatal_issues:
            st.markdown(f"- **{issue.message}**  \n  ↳ {issue.suggestion}")
    for issue in report.warnings:
        st.warning(f"{issue.message}  \n↳ {issue.suggestion}")
    return report.is_feasible


def render_covariance_diagnostics(diag) -> None:
    """Show whether the covariance estimate is fit to optimize against."""
    if diag is None:
        return
    metric_row(
        [
            ("Observations (T)", f"{diag.n_observations:,}", None),
            ("Assets (N)", str(diag.n_assets), None),
            (
                "T / N",
                num(diag.observations_per_asset, 1),
                "Below ~10, covariance estimates are dominated by noise.",
            ),
            (
                "Condition number",
                f"{diag.condition_number:.3g}",
                "Largest over smallest eigenvalue. Above 1e4 the matrix is "
                "poorly conditioned and mean-variance weights get unstable.",
            ),
        ]
    )
    for warning in diag.warnings:
        st.warning(warning)


def render_compliance(result) -> None:
    """Banner stating whether the solved weights respect every constraint."""
    diag = result.extras.get("diagnostics")
    status = result.extras.get("solver_status", "unknown")
    if result.violations:
        st.error(
            "**The solved weights breach constraints you set.** "
            + " · ".join(result.violations)
        )
    elif status == "optimal_inaccurate":
        st.warning(
            "The solver converged only to loose tolerance "
            "(`optimal_inaccurate`). Treat the weights as approximate — "
            "usually a sign of a near-singular covariance or nearly-binding "
            "constraints."
        )
    elif diag is not None:
        st.success(
            f"Solved with {result.extras.get('solver', 'the default solver')} "
            f"in {result.extras.get('solve_seconds', 0):.2f}s — every bound, "
            "group budget and exposure limit is respected."
        )

    ignored = result.extras.get("ignored_constraints")
    if ignored:
        st.info(
            f"This method cannot enforce: {', '.join(ignored)}. "
            "The setting was kept but not imposed."
        )
    note = result.extras.get("bounds_note") or result.extras.get("fallback_reason")
    if note:
        st.info(note)
    rb_note = result.extras.get("risk_budget_note")
    if rb_note:
        st.info(rb_note)
    cvar_note = result.extras.get("cvar_note")
    if cvar_note:
        st.info(cvar_note)


def render_portfolio_diagnostics(diag) -> None:
    """Concentration and diversification cards."""
    if diag is None:
        return
    metric_row(
        [
            ("Positions", str(diag.n_positions), "Weights above 1bp."),
            (
                "Effective N",
                num(diag.effective_n, 1),
                "Inverse Herfindahl: how many equally-weighted positions this "
                "book is really worth. Far below the position count means the "
                "portfolio is concentrated.",
            ),
            (
                "Effective N (risk)",
                num(diag.effective_n_risk, 1),
                "The same measure applied to risk contributions. A 60/40 book "
                "is diversified by capital and concentrated in equity risk.",
            ),
            (
                "Diversification ratio",
                num(diag.diversification_ratio, 2),
                "Weighted-average asset volatility over portfolio volatility. "
                "1.0 means correlations bought you nothing.",
            ),
        ]
    )
    extra: list[tuple[str, str, str | None]] = [
        ("Largest position", pct(diag.max_weight), None),
        ("Gross exposure", pct(diag.gross_exposure), "Sum of absolute weights."),
    ]
    if abs(diag.short_exposure) > 1e-9:
        extra.append(("Short exposure", pct(diag.short_exposure), None))
    if diag.turnover is not None:
        extra.append(
            (
                "Turnover vs. previous",
                pct(diag.turnover),
                "One-way: the fraction of the book that changes hands.",
            )
        )
    metric_row(extra)


def render_method_card(req) -> None:
    """Explain what the chosen optimizer does and what it assumes."""
    st.markdown(f"**{req.display_name}** — {req.summary}")
    with st.expander("What this method assumes", expanded=False):
        st.markdown(f"**Use it when:** {req.when_to_use}")
        if req.assumptions:
            st.markdown("**It assumes:**")
            for a in req.assumptions:
                st.markdown(f"- {a}")
        st.caption(req.bounds_note)
        needs = []
        if req.requires_mu:
            needs.append("expected returns")
        if req.requires_cov:
            needs.append("a covariance matrix")
        if req.requires_returns:
            needs.append("the full return history")
        if needs:
            st.caption(f"Inputs used: {', '.join(needs)}.")


def render_assumptions(assumptions: dict[str, Any]) -> None:
    """Print every modelling choice a result rests on."""
    labels = {
        "optimizer": "Optimizer",
        "objective_mode": "Objective",
        "covariance_estimator": "Covariance estimator",
        "ewma_lambda": "EWMA λ",
        "expected_returns_method": "Expected-returns method",
        "risk_free_rate": "Risk-free rate",
        "risk_aversion": "Risk aversion λ",
        "target_return": "Target return",
        "target_volatility": "Target volatility",
        "periods_per_year": "Periods per year",
        "base_currency": "Base currency",
        "sample_start": "Sample start",
        "sample_end": "Sample end",
        "n_observations": "Observations",
        "n_assets": "Assets",
        "long_only": "Long only",
        "fully_invested": "Fully invested",
        "leverage_cap": "Gross-exposure cap",
        "turnover_limit": "Turnover budget",
        "solver": "Solver",
        "solver_status": "Solver status",
    }
    rows = [
        {"Assumption": labels.get(k, k), "Value": _fmt_assumption(k, v)}
        for k, v in assumptions.items()
        if v is not None
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _fmt_assumption(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in ("risk_free_rate", "target_return", "target_volatility") and isinstance(
        value, (int, float)
    ):
        return f"{value:.2%}"
    return str(value)


def render_frontier_health(frontier) -> None:
    """Report failed sweep points and the reachable return range."""
    if frontier is None:
        return
    if frontier.reachable_range is not None:
        lo, hi = frontier.reachable_range
        gmv = (
            frontier.min_variance["expected_return"]
            if frontier.min_variance is not None
            else None
        )
        caption = (
            f"Under these constraints, expected return can range from "
            f"{lo:.2%} to {hi:.2%}"
        )
        if gmv is not None:
            caption += f"; only the part above {gmv:.2%} is efficient."
        else:
            caption += "."
        st.caption(caption)
    if frontier.n_failed:
        with st.expander(
            f"{frontier.n_failed} frontier point(s) did not solve", expanded=False
        ):
            for f in frontier.failures[:10]:
                st.text(f)
