"""JSON payloads for machine consumers.

The Excel report and the CLI's printed tables are written for a person who
will read them. This module is the other audience: a script, a pipeline, or
an agent that has to *act* on a result and cannot parse a formatted table
without guessing.

The distinction that matters is a contract versus a dump. Serialising the
result objects with ``dataclasses.asdict`` would be a dump: every internal
rename becomes a silent breaking change for anyone parsing it, and the shape
depends on which optional pieces happened to be computed. So each function
here builds an explicit dictionary, keys are chosen for the reader rather
than inherited from the attribute they came from, and a key that is absent
this run is present as ``null`` rather than missing — a consumer can test a
value, never a key.

Every payload carries ``schema_version``. It is the only promise this module
makes: within a major version, a key that exists keeps its meaning and keys
are only added. See ``SCHEMA_VERSION``.

Floats are plain Python floats and never NumPy scalars, because
``json.dumps`` cannot serialise the latter and the failure surfaces at the
worst moment — after a long solve, with the result already computed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

#: Bumped major on a breaking change to any payload below, minor when keys
#: are added. Consumers should check the major and refuse a version they do
#: not know rather than parsing optimistically.
SCHEMA_VERSION = "1.1"


def _num(value: Any) -> float | None:
    """A JSON-safe float, or ``None`` where JSON has no way to say it.

    NaN and infinity are the reason this exists. Both are legal Python
    floats and legal in ``json.dumps`` output by default, but ``NaN`` is not
    valid JSON — a strict parser on the other end rejects the whole
    document. A metric that could not be computed becomes ``null``.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _series(values: pd.Series | None) -> dict[str, float | None] | None:
    """A pandas Series as ``{label: value}``, keys stringified."""
    if values is None:
        return None
    return {str(k): _num(v) for k, v in values.items()}


def _strings(values: Any) -> list[str]:
    """Any iterable of message-ish things as a list of strings."""
    if not values:
        return []
    return [str(v) for v in values]


def portfolio_diagnostics_payload(diagnostics: Any) -> dict[str, Any] | None:
    """Concentration and exposure diagnostics.

    ``effective_n`` against ``effective_n_risk`` is the pair worth reading
    together: the first counts positions by capital, the second by risk
    contribution, and the gap between them is what a weights table hides.

    Args:
        diagnostics: A
            :class:`~optimization_engine.optimizers.diagnostics.PortfolioDiagnostics`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if diagnostics is None:
        return None
    return {
        "n_positions": int(diagnostics.n_positions),
        "gross_exposure": _num(diagnostics.gross_exposure),
        "net_exposure": _num(diagnostics.net_exposure),
        "long_exposure": _num(diagnostics.long_exposure),
        "short_exposure": _num(diagnostics.short_exposure),
        "max_weight": _num(diagnostics.max_weight),
        "herfindahl": _num(diagnostics.herfindahl),
        "effective_n": _num(diagnostics.effective_n),
        "effective_n_risk": _num(diagnostics.effective_n_risk),
        "diversification_ratio": _num(diagnostics.diversification_ratio),
        "turnover": _num(diagnostics.turnover),
        "violations": _strings(diagnostics.violations),
    }


def covariance_diagnostics_payload(diagnostics: Any) -> dict[str, Any] | None:
    """Whether the covariance estimate is worth the weights built on it.

    ``observations_per_asset`` below roughly 10 and a large
    ``condition_number`` together mean the optimiser is fitting noise, which
    no amount of solver precision fixes.

    Args:
        diagnostics: A
            :class:`~optimization_engine.data.covariance.CovarianceDiagnostics`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if diagnostics is None:
        return None
    return {
        "n_assets": int(diagnostics.n_assets),
        "n_observations": int(diagnostics.n_observations),
        "observations_per_asset": _num(diagnostics.observations_per_asset),
        "condition_number": _num(diagnostics.condition_number),
        "min_eigenvalue": _num(diagnostics.min_eigenvalue),
        "is_psd": bool(diagnostics.is_psd),
        "effective_observations": _num(diagnostics.effective_observations),
        "warnings": _strings(diagnostics.warnings),
    }


def feasibility_payload(report: Any) -> dict[str, Any] | None:
    """Whether the mandate could be satisfied at all, and over what range.

    ``feasible`` is lifted to the top of the object so a consumer can branch
    on one boolean instead of interpreting an issue list.

    Args:
        report: A
            :class:`~optimization_engine.optimizers.feasibility.FeasibilityReport`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if report is None:
        return None
    issues = _strings(getattr(report, "issues", None))
    # `is_feasible` is the report's own verdict; deriving it from an empty
    # issue list would silently disagree the day the two stop coinciding.
    feasible = getattr(report, "is_feasible", None)
    return {
        "feasible": bool(feasible) if feasible is not None else not issues,
        "issues": issues,
        "min_return": _num(getattr(report, "min_return", None)),
        "max_return": _num(getattr(report, "max_return", None)),
        "min_variance_return": _num(getattr(report, "min_variance_return", None)),
    }


def optimization_payload(
    run: Any,
    *,
    output_path: str | None = None,
    alignment: Any = None,
) -> dict[str, Any]:
    """The full result of one solve: weights, and what they rest on.

    Args:
        run: An :class:`~optimization_engine.engine.EngineRun`.
        output_path: Where the workbook was written, when one was.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel` — one
            sentence per change made to the panel before it was
            differenced. Empty means the panel needed no changes, which
            is a different claim from "nobody looked".

    Returns:
        A JSON-serialisable dict. ``weights`` maps asset name to weight;
        every other top-level key is either a nested object of the same
        shape as its ``*_payload`` builder, or ``null``.
    """
    result = run.result
    extras = dict(getattr(result, "extras", {}) or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "optimize",
        "optimizer": extras.get("optimizer"),
        # Which solver actually answered, and how it terminated. A run that
        # fell through to a later solver in the chain is not wrong, but it
        # is worth knowing about.
        "solver": extras.get("solver"),
        "solver_status": extras.get("solver_status"),
        "weights": _series(result.weights),
        "metrics": {
            "expected_return": _num(result.expected_return),
            "expected_volatility": _num(result.expected_volatility),
            "sharpe_ratio": _num(result.sharpe_ratio),
        },
        "diagnostics": portfolio_diagnostics_payload(getattr(run, "diagnostics", None)),
        "covariance": covariance_diagnostics_payload(
            getattr(run, "covariance_diagnostics", None)
        ),
        "feasibility": feasibility_payload(getattr(run, "feasibility", None)),
        "warnings": _strings(getattr(run, "warnings", None)),
        # What the sample the numbers above were computed on had to lose
        # to become rectangular. A late-listing asset truncates every
        # other series, and that is invisible in the weights.
        "alignment": _strings(alignment),
        "output_path": output_path,
    }


def check_payload(
    quality: Any,
    feasibility: Any,
    covariance: Any = None,
    *,
    alignment: Any = None,
) -> dict[str, Any]:
    """A pre-flight verdict: can this mandate be solved, and on what data.

    Deliberately narrower than :func:`optimization_payload` — ``check`` runs
    *before* a solve, so there are no weights to report and claiming any
    would describe something that was never computed.

    ``ready`` is the single field a caller needs to branch on, and it
    mirrors the command's exit code: both data-quality errors and an
    infeasible constraint set make it false.

    Args:
        quality: A :class:`~optimization_engine.data.quality.DataQualityReport`.
        feasibility: A ``FeasibilityReport`` from ``analyze_feasibility``.
        covariance: Covariance diagnostics, when they were computed.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel`.
    """
    feas = feasibility_payload(feasibility)
    quality_errors = _strings(getattr(quality, "errors", None))
    ready = not quality_errors and bool(feas["feasible"] if feas else True)
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "check",
        "ready": ready,
        "data_quality": {
            "errors": quality_errors,
            "issues": _strings(getattr(quality, "issues", None)),
            "n_common_periods": int(getattr(quality, "n_common_periods", 0) or 0),
            "common_start": str(getattr(quality, "common_start", None) or "") or None,
            "common_end": str(getattr(quality, "common_end", None) or "") or None,
        },
        # `data_quality` describes the panel as it arrived; `alignment`
        # describes what was done to it afterwards. Reading the first
        # without the second says what was wrong, not what was kept.
        "alignment": _strings(alignment),
        "feasibility": feas,
        "covariance": covariance_diagnostics_payload(covariance),
    }


def _frame(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    """A DataFrame as a list of row objects, index carried as ``"index"``.

    Row objects rather than a column-major dict because a consumer that
    wants one row should not have to zip several parallel arrays and trust
    that they stayed aligned.
    """
    if frame is None or frame.empty:
        return None
    rows = []
    for idx, row in frame.iterrows():
        record: dict[str, Any] = {"index": str(idx)}
        for col, value in row.items():
            record[str(col)] = _num(value) if isinstance(value, (int, float, np.number)) else str(value)
        rows.append(record)
    return rows


def backtest_payload(
    result: Any,
    *,
    tearsheet: Any = None,
    output_path: str | None = None,
    alignment: Any = None,
) -> dict[str, Any]:
    """What a simulated run of the process actually produced.

    The two hashes are the reason a machine caller should read this rather
    than the workbook: two runs carrying the same ``spec_hash`` were asked
    the same question, and ``result_hash`` says whether they got the same
    answer. That is the cheapest way to tell a genuine change from a
    re-run.

    ``degradations`` is the honest part of the record — where the
    simulation had to fall back (a missing volatility estimate dropping
    market impact to the linear charge, say). A run with degradations is
    not wrong, but its costs are optimistic in a specific, named way.

    Args:
        result: A :class:`~optimization_engine.backtest.results.RunResult`.
        tearsheet: The built tearsheet, when one was.
        output_path: Where the workbook was written, when one was.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel`. A
            simulated track record that starts three years late because
            one asset listed late is not the same track record.
    """
    meta = getattr(result, "meta", None)
    metrics = None
    if tearsheet is not None:
        performance = getattr(tearsheet, "performance", None)
        if performance is not None and not performance.empty:
            # One row, named for the book it describes; a machine caller
            # wants the metrics, not the row label.
            metrics = {
                str(k): _num(v) for k, v in performance.iloc[0].items()
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "backtest",
        "spec_hash": getattr(meta, "spec_hash", None),
        "result_hash": getattr(meta, "result_hash", None),
        "window": {
            "n_periods": int(getattr(meta, "n_periods", 0) or 0),
            "n_assets": int(getattr(meta, "n_assets", 0) or 0),
            "start": str(getattr(meta, "start", None) or "") or None,
            "end": str(getattr(meta, "end", None) or "") or None,
            "is_out_of_sample": bool(getattr(meta, "is_out_of_sample", False)),
        },
        "degradations": _strings(getattr(meta, "degradations", None)),
        "notes": _strings(getattr(meta, "notes", None)),
        "alignment": _strings(alignment),
        "metrics": metrics,
        "output_path": output_path,
    }


def describe_payload(req: Any) -> dict[str, Any]:
    """One optimizer's contract: what it needs, and what it will honour.

    This is the payload an agent reads *before* building a config, to find
    out whether the method it wants can express the mandate at all — a
    turnover budget handed to a method with ``supports_turnover`` false is
    silently ignored, not rejected.

    Args:
        req: A
            :class:`~optimization_engine.optimizers.requirements.MethodRequirements`.

    Returns:
        A JSON-serializable dict of what the method requires, supports and
        assumes.
    """
    inputs = {
        "expected_returns": bool(req.requires_mu),
        "covariance": bool(req.requires_cov),
        "return_history": bool(req.requires_returns),
        "benchmark": bool(req.requires_benchmark),
    }
    supports = {
        "target_return": bool(req.supports_target_return),
        "target_volatility": bool(req.supports_target_volatility),
        "risk_aversion": bool(req.supports_risk_aversion),
        "risk_free_rate": bool(req.supports_risk_free_rate),
        "group_bounds": bool(req.supports_group_bounds),
        "frontier": bool(req.supports_frontier),
        "turnover": bool(req.supports_turnover),
        "benchmark_limits": bool(req.supports_benchmark_limits),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "describe",
        "name": req.name,
        "label": getattr(req, "label", None),
        "summary": getattr(req, "summary", None),
        "when_to_use": getattr(req, "when_to_use", None),
        "risk_measure": getattr(req, "risk_measure", None),
        "bounds_mode": getattr(req, "bounds_mode", None),
        "assumptions": _strings(getattr(req, "assumptions", None)),
        "requires": inputs,
        "supports": supports,
    }
