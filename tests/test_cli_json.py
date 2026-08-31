"""The `--json` contract, held to what a machine consumer needs from it.

These tests are deliberately about *shape*, not values. A consumer parsing
this output cares that `weights` is an object of floats and that
`feasibility.feasible` is a boolean it can branch on; it does not care what
the optimizer decided today. Asserting on numbers here would make the suite
fail every time an estimator improves, which trains people to update the
expected values without reading them.

The one thing worth guarding hardest is that stdout parses at all. Every
command prints as it works, and a single stray `print` reaching stdout in
JSON mode breaks every caller — silently, because a truncated document
usually still looks like output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optimization_engine.cli import main  # noqa: E402
from optimization_engine.reporting.payloads import SCHEMA_VERSION  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "example_multi_asset.yaml")


def _run(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run the CLI and parse stdout, asserting the streams stayed separate."""
    code = main(argv)
    captured = capsys.readouterr()
    # The real assertion: stdout is a single JSON document and nothing else.
    # json.loads is strict about trailing content, so a leaked print fails
    # here rather than corrupting a downstream parse.
    payload = json.loads(captured.out)
    return code, payload


def test_describe_json_reports_the_optimizer_contract(capsys):
    code, payload = _run(capsys, ["describe", "risk_parity", "--json"])
    assert code == 0
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["command"] == "describe"
    assert payload["name"] == "risk_parity"
    # `requires` and `supports` are what an agent reads before building a
    # config: a turnover budget handed to a method that does not support one
    # is ignored, not rejected, so the flag has to be discoverable up front.
    assert set(payload["requires"]) == {
        "expected_returns",
        "covariance",
        "return_history",
        "benchmark",
    }
    assert all(isinstance(v, bool) for v in payload["requires"].values())
    assert all(isinstance(v, bool) for v in payload["supports"].values())


def test_describe_json_on_an_unknown_name_still_emits_json(capsys):
    """A failure a caller can parse beats a failure it has to guess at."""
    code, payload = _run(capsys, ["describe", "no_such_optimizer", "--json"])
    assert code != 0
    assert payload["command"] == "describe"
    assert payload["error"]
    assert payload["exit_code"] == code


def test_check_json_answers_ready_with_a_boolean(capsys):
    code, payload = _run(capsys, ["check", "--config", CONFIG, "--sample", "--json"])
    assert code == 0
    assert payload["command"] == "check"
    # One boolean to branch on, mirroring the exit code.
    assert payload["ready"] is True
    assert payload["feasibility"]["feasible"] is True
    assert isinstance(payload["data_quality"]["errors"], list)
    assert payload["covariance"]["is_psd"] is True
    assert payload["covariance"]["n_assets"] > 0


def test_optimize_json_carries_weights_and_the_evidence(capsys, tmp_path):
    out = tmp_path / "report.xlsx"
    code, payload = _run(
        capsys,
        ["optimize", "--config", CONFIG, "--sample", "--output", str(out), "--json"],
    )
    assert code == 0
    assert payload["command"] == "optimize"

    weights = payload["weights"]
    assert weights and all(isinstance(v, float) for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    # The claim this library makes is that weights alone are not a result.
    # If these ever come back None on a successful solve, the payload has
    # stopped carrying the thing that distinguishes it.
    assert payload["diagnostics"] is not None
    assert payload["covariance"] is not None
    assert payload["diagnostics"]["effective_n"] is not None
    assert payload["diagnostics"]["effective_n_risk"] is not None
    assert payload["solver"], "the solver that answered should be named"
    assert payload["output_path"] == str(out)


def test_backtest_json_carries_the_hashes_that_identify_the_run(capsys):
    code, payload = _run(
        capsys,
        [
            "backtest",
            "--config", CONFIG,
            "--sample",
            "--lookback", "504",
            "--rebalance-every", "252",
            "--json",
        ],
    )
    assert code == 0
    assert payload["command"] == "backtest"
    # Without these two a caller cannot tell a real change from a re-run,
    # which is the whole reason to read this instead of the workbook.
    assert payload["spec_hash"]
    assert payload["result_hash"]
    assert payload["window"]["n_periods"] > 0
    assert isinstance(payload["degradations"], list)
    # No --output was passed, so there is no workbook to point at. This also
    # covers the path where the writer never runs.
    assert payload["output_path"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["describe", "risk_parity", "--json"],
        ["check", "--config", CONFIG, "--sample", "--json"],
    ],
)
def test_narration_goes_to_stderr_not_stdout(capsys, argv):
    """Human narration is preserved — it just moves off the parsed stream."""
    main(argv)
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is exactly one document
    assert captured.err.strip(), "the human-readable output should still exist"


def test_every_payload_declares_its_schema_version(capsys):
    """A consumer must be able to refuse a version it does not know."""
    for argv in (
        ["describe", "risk_parity", "--json"],
        ["check", "--config", CONFIG, "--sample", "--json"],
    ):
        _, payload = _run(capsys, argv)
        assert payload["schema_version"] == SCHEMA_VERSION
