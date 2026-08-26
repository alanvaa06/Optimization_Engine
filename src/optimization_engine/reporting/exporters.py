"""Excel exporter for optimization results.

The workbook is meant to be read by someone who was not in the room when the
optimization ran, so :func:`run_sheets` assembles the assumptions, the
estimation diagnostics and the data-quality findings alongside the weights.
A weights tab on its own is a number without its provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

#: Excel refuses sheet names longer than this.
_SHEET_NAME_LIMIT = 31


def write_excel_report(
    path: str | Path, sheets: Mapping[str, pd.DataFrame | pd.Series]
) -> Path:
    """Write a multi-sheet Excel workbook from a mapping of name → frame.

    ``None`` entries are skipped, Series are promoted to single-column frames,
    and names are truncated to Excel's 31-character limit. Truncation can
    collide, so a numeric suffix is appended rather than letting one sheet
    silently overwrite another.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            sheet = _unique_sheet_name(name, used)
            used.add(sheet)
            df.to_excel(writer, sheet_name=sheet, index=True)
    return p


def _unique_sheet_name(name: str, used: set[str]) -> str:
    base = str(name)[:_SHEET_NAME_LIMIT]
    if base not in used:
        return base
    for i in range(2, 100):
        suffix = f"_{i}"
        candidate = base[: _SHEET_NAME_LIMIT - len(suffix)] + suffix
        if candidate not in used:
            return candidate
    raise ValueError(f"Cannot find a unique Excel sheet name for {name!r}.")


def run_sheets(
    run,
    riskfree_rate: float = 0.0,
    data_quality=None,
    walk_forward=None,
) -> dict[str, pd.DataFrame]:
    """Assemble the standard workbook contents for an :class:`EngineRun`.

    Shared by the CLI and the Streamlit app so both produce the same report.

    Args:
        run: The :class:`~optimization_engine.engine.EngineRun` to describe.
        riskfree_rate: Annual rate used for the performance summary.
        data_quality: Optional :class:`DataQualityReport` for the input panel.
        walk_forward: Optional :class:`WalkForwardResult` to include the
            out-of-sample track record and its degradation table.
    """
    sheets: dict[str, pd.DataFrame] = {
        "weights": run.result.weights.to_frame("weight"),
        "summary": pd.DataFrame(
            [
                {
                    "expected_return": run.result.expected_return,
                    "expected_volatility": run.result.expected_volatility,
                    "sharpe_ratio": run.result.sharpe_ratio,
                }
            ]
        ),
        "assumptions": pd.DataFrame(
            [{"assumption": k, "value": v} for k, v in run.assumptions().items()]
        ),
        "risk_decomposition": run.risk_decomposition(),
        "expected_returns": run.expected_returns.to_frame("annualized"),
        "cov_matrix": run.cov_matrix,
        "absolute_summary": run.absolute_summary(
            riskfree_rate=riskfree_rate, extended=True
        ),
    }

    if run.diagnostics is not None:
        sheets["portfolio_diagnostics"] = pd.DataFrame(
            [
                {"metric": k, "value": v}
                for k, v in run.diagnostics.to_dict().items()
                if not isinstance(v, list)
            ]
        )
    if run.covariance_diagnostics is not None:
        diag = run.covariance_diagnostics
        sheets["estimation_diagnostics"] = pd.DataFrame(
            [
                {"metric": "n_assets", "value": diag.n_assets},
                {"metric": "n_observations", "value": diag.n_observations},
                {"metric": "observations_per_asset", "value": diag.observations_per_asset},
                {"metric": "condition_number", "value": diag.condition_number},
                {"metric": "min_eigenvalue", "value": diag.min_eigenvalue},
                {"metric": "effective_observations", "value": diag.effective_observations},
                *[{"metric": "warning", "value": w} for w in diag.warnings],
            ]
        )
    if run.feasibility is not None and run.feasibility.issues:
        sheets["feasibility"] = pd.DataFrame(
            [
                {
                    "severity": "fatal" if i.fatal else "warning",
                    "finding": i.message,
                    "suggestion": i.suggestion,
                }
                for i in run.feasibility.issues
            ]
        )
    if data_quality is not None:
        sheets["data_quality"] = (
            pd.DataFrame(
                [
                    {
                        "severity": i.severity,
                        "asset": i.asset or "",
                        "finding": i.message,
                        "suggestion": i.suggestion,
                    }
                    for i in data_quality.issues
                ]
            )
            if data_quality.issues
            else pd.DataFrame([{"severity": "info", "finding": "No issues found."}])
        )
    if run.frontier is not None:
        sheets["frontier_summary"] = run.frontier.summary
        sheets["frontier_weights"] = run.frontier.weights
        if (
            run.frontier.group_weights is not None
            and not run.frontier.group_weights.empty
        ):
            sheets["frontier_groups"] = run.frontier.group_weights
    if walk_forward is not None:
        sheets["walk_forward_weights"] = walk_forward.weights_history
        sheets["walk_forward_windows"] = walk_forward.windows
        sheets["in_vs_out_of_sample"] = run.in_vs_out_of_sample(
            walk_forward, riskfree_rate
        )
    return sheets
