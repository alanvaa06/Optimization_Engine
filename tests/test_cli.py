"""CLI behaviour: validation, reporting, and exit codes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.cli import main
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.reporting.exporters import (
    _unique_sheet_name,
    run_sheets,
    write_excel_report,
)

EXAMPLE_CONFIG = ROOT / "config" / "example_multi_asset.yaml"


@pytest.fixture
def feasible_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def infeasible_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    # Cap everything at 5%: the caps cannot sum to a fully-invested book.
    data["bounds"] = {a: [0.0, 0.05] for a in data["expected_returns"]}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_list_optimizers_describes_each_method(capsys):
    assert main(["list-optimizers"]) == 0
    out = capsys.readouterr().out
    assert "mean_variance" in out
    # Not just a bare list of names any more.
    assert "variance" in out.lower() and len(out.splitlines()) >= 10


def test_describe_reports_assumptions(capsys):
    assert main(["describe", "hrp"]) == 0
    out = capsys.readouterr().out
    assert "Hierarchical Risk Parity" in out
    assert "Assumptions:" in out
    assert "Group budgets cannot be enforced" in out


def test_describe_rejects_an_unknown_method(capsys):
    assert main(["describe", "not_a_method"]) == 2
    assert "Available" in capsys.readouterr().err


def test_check_passes_on_a_feasible_config(feasible_config, capsys):
    assert main(["check", "--config", str(feasible_config), "--sample"]) == 0
    out = capsys.readouterr().out
    assert "Ready to optimize." in out
    assert "Reachable expected return" in out
    assert "T/N" in out


def test_check_fails_and_names_the_constraint(infeasible_config, capsys):
    assert main(["check", "--config", str(infeasible_config), "--sample"]) == 1
    captured = capsys.readouterr()
    assert "cannot reach 100% invested" in captured.out
    assert "Raise the caps" in captured.out
    assert "Not ready to optimize." in captured.err


def test_optimize_writes_a_report_with_provenance(feasible_config, tmp_path, capsys):
    out_path = tmp_path / "report.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(feasible_config), "--sample",
                "--output", str(out_path),
            ]
        )
        == 0
    )
    assert out_path.exists()
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert {"weights", "assumptions", "risk_decomposition", "data_quality"} <= set(sheets)
    out = capsys.readouterr().out
    assert "effective N" in out


def test_strict_optimize_refuses_an_infeasible_config(
    infeasible_config, tmp_path, capsys
):
    out_path = tmp_path / "report.xlsx"
    code = main(
        [
            "optimize", "--config", str(infeasible_config), "--sample",
            "--strict", "--output", str(out_path),
        ]
    )
    assert code == 2
    assert "cannot reach 100% invested" in capsys.readouterr().err
    assert not out_path.exists()


def test_optimize_with_walk_forward_reports_degradation(
    feasible_config, tmp_path, capsys
):
    out_path = tmp_path / "wf.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(feasible_config), "--sample",
                "--walk-forward", "--lookback", "504", "--rebalance-every", "126",
                "--cost-bps", "10", "--output", str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Walk-forward" in out and "out of sample" in out
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert "in_vs_out_of_sample" in sheets


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def test_run_sheets_carries_the_evidence():
    returns = prices_to_returns(sample_dataset(252 * 4, seed=5))
    cfg = EngineConfig(
        bounds={a: [0.0, 0.3] for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg, build_frontier=True, n_frontier_points=6)
    sheets = run_sheets(run)
    for expected in (
        "weights", "assumptions", "risk_decomposition",
        "portfolio_diagnostics", "estimation_diagnostics",
        "frontier_summary", "frontier_weights",
    ):
        assert expected in sheets, expected


def test_long_sheet_names_do_not_collide(tmp_path):
    long_a = "a_very_long_sheet_name_that_exceeds_limits_one"
    long_b = "a_very_long_sheet_name_that_exceeds_limits_two"
    frame = pd.DataFrame({"x": [1, 2]})
    out = write_excel_report(tmp_path / "r.xlsx", {long_a: frame, long_b: frame})
    sheets = pd.read_excel(out, sheet_name=None)
    assert len(sheets) == 2, "truncated names silently overwrote each other"


def test_unique_sheet_name_suffixes_collisions():
    used = {"abc"}
    assert _unique_sheet_name("abc", used) != "abc"
    assert _unique_sheet_name("xyz", used) == "xyz"


def test_write_excel_report_skips_none_and_promotes_series(tmp_path):
    out = write_excel_report(
        tmp_path / "r.xlsx",
        {"a": pd.Series([1.0, 2.0], name="v"), "b": None},
    )
    sheets = pd.read_excel(out, sheet_name=None)
    assert set(sheets) == {"a"}


# ---------------------------------------------------------------------------
# Benchmark flags
# ---------------------------------------------------------------------------


def test_optimize_reports_against_a_benchmark(feasible_config, tmp_path, capsys):
    out_path = tmp_path / "bench.xlsx"
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "equal_weight", "--output", str(out_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "vs Equal weight (1/N)" in out
    assert "IR" in out and "T.E." in out
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert "benchmark" in sheets
    assert "performance_relative" in sheets


def test_optimize_accepts_an_asset_name_as_the_benchmark(
    feasible_config, tmp_path, capsys
):
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "US_Equity", "--output", str(tmp_path / "a.xlsx"),
        ]
    )
    assert code == 0
    assert "vs US_Equity" in capsys.readouterr().out


def test_optimize_rejects_an_unknown_benchmark(feasible_config, tmp_path, capsys):
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "NOT_A_THING", "--output", str(tmp_path / "b.xlsx"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "neither a benchmark kind nor an asset" in err


def test_a_tracking_error_budget_binds_from_the_command_line(
    feasible_config, tmp_path
):
    import numpy as np

    data = yaml.safe_load(feasible_config.read_text())
    # The example config targets a return the tight budget cannot reach;
    # this test is about the budget, so the target comes off.
    data["optimizer"].pop("target_return", None)
    path = tmp_path / "no_target.yaml"
    path.write_text(yaml.safe_dump(data))

    out_path = tmp_path / "te.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(path), "--sample",
                "--benchmark", "equal_weight",
                "--max-tracking-error", "0.02",
                "--output", str(out_path),
            ]
        )
        == 0
    )
    sheets = pd.read_excel(out_path, sheet_name=None)
    weights = sheets["benchmark_weights"].set_index(
        sheets["benchmark_weights"].columns[0]
    )
    active = weights["active_weight"].values
    cov = sheets["cov_matrix"].set_index(sheets["cov_matrix"].columns[0])
    realized = float(np.sqrt(max(active @ cov.values @ active, 0.0)))
    assert realized <= 0.02 + 1e-4


def test_an_unreachable_target_under_a_budget_names_the_cause(
    feasible_config, tmp_path, capsys
):
    # The example config targets 7%; a 4% tracking-error budget puts that
    # out of reach, and the message should say so rather than "infeasible".
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "equal_weight", "--max-tracking-error", "0.04",
            "--output", str(tmp_path / "c.xlsx"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Target return" in err
    assert "tracking-error or active-share budget is in force" in err
