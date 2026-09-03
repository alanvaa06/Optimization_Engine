"""The stress panel in the JSON payloads.

``--stress`` reached the console and the workbook long before it reached
``--json``: a machine caller reading the structured output could see the
weights, the feasibility verdict and the mandate audit, but not what a named
bad day does to the book — which is the one number worth automating an alert
on. These pin the shape it arrived in.

The standard the rest of ``payloads`` sets is the one applied here: named
fields a consumer can threshold on, never a stringified repr, and ``null`` for
"not computed" rather than a missing key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.reporting.payloads import (
    SCHEMA_VERSION,
    backtest_payload,
    optimization_payload,
    stress_payload,
)
from optimization_engine.stress import Shock, load_shocks, stress_test

SHOCKS_EXAMPLE = ROOT / "config" / "shocks.yaml"


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=400))


@pytest.fixture(scope="module")
def report(returns):
    book = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    shocks = [
        Shock(
            name="Mild",
            returns={"US_Equity": -0.05},
            notes="a quiet day",
        ),
        Shock(
            name="Severe",
            returns={"US_Equity": -0.40, "US_Treasuries": 0.08},
            covariance_scale=4.0,
            notes="a loud one",
        ),
    ]
    return stress_test(book, shocks, cov_matrix=returns.cov() * 252)


def test_the_schema_version_was_bumped_for_the_new_key():
    major, minor = SCHEMA_VERSION.split(".")[:2]
    assert major == "2", "adding a key is a minor bump, never a major one"
    assert int(minor) >= 2


def test_absent_is_null_rather_than_missing():
    assert stress_payload(None) is None


def test_the_worst_case_is_lifted_to_the_top(report):
    payload = stress_payload(report)
    assert payload["worst_scenario"] == "Severe"
    assert payload["worst_pnl"] == pytest.approx(report.worst.pnl)
    assert payload["worst_contributor"] == report.worst.largest_contributor
    assert payload["n_scenarios"] == 2


def test_scenarios_come_back_worst_first(report):
    payload = stress_payload(report)
    names = [scenario["name"] for scenario in payload["scenarios"]]
    assert names == [s.name for s in report.by_severity()]
    pnls = [scenario["pnl"] for scenario in payload["scenarios"]]
    assert pnls == sorted(pnls)


def test_every_field_is_a_named_value_not_a_repr(report):
    payload = stress_payload(report)
    severe = next(s for s in payload["scenarios"] if s["name"] == "Severe")
    assert isinstance(severe["pnl"], float)
    assert isinstance(severe["volatility_ratio"], float)
    assert severe["volatility_ratio"] == pytest.approx(2.0)
    assert severe["largest_contributor"] == "US_Equity"
    assert severe["notes"] == "a loud one"
    assert severe["ignored_assets"] == []
    # Nothing in the document is a dataclass printed out.
    assert "ScenarioStress(" not in json.dumps(payload)


def test_the_contributions_still_sum_to_the_pnl_after_serialisation(report):
    """The identity is what makes the loss attributable, so it must survive."""
    payload = stress_payload(report)
    for scenario in payload["scenarios"]:
        assert sum(scenario["contributions"].values()) == pytest.approx(
            scenario["pnl"], abs=1e-12
        )


def test_the_unknown_asset_policy_is_recorded(returns):
    book = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    wide = Shock(name="Wider", returns={"US_Equity": -0.3, "NOT_HELD": -0.9})
    ignored = stress_payload(
        stress_test(book, [wide], unknown_assets="ignore")
    )
    assert ignored["unknown_asset_policy"] == "ignore"
    assert ignored["scenarios"][0]["ignored_assets"] == ["NOT_HELD"]
    # Volatility fields are null rather than absent when no covariance was given.
    assert ignored["scenarios"][0]["stressed_volatility"] is None
    assert ignored["base_volatility"] is None


def test_the_payload_is_json_serialisable(report):
    text = json.dumps(stress_payload(report))
    assert json.loads(text)["worst_scenario"] == "Severe"


def test_the_optimize_payload_carries_the_run_s_own_stress(returns):
    config = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
        stress=load_shocks(SHOCKS_EXAMPLE),
    )
    run = run_engine(returns, config, build_frontier=False, run_stress=True)
    payload = optimization_payload(run)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["stress"] is not None
    assert payload["stress"]["n_scenarios"] == len(config.stress)
    assert payload["stress"]["worst_scenario"] in {s.name for s in config.stress}
    # And the same run without the scenarios says so with null, not silence.
    plain = optimization_payload(run_engine(returns, config, build_frontier=False))
    assert "stress" in plain
    assert plain["stress"] is None


def test_the_backtest_payload_carries_the_tearsheet_s_stress(returns):
    config = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
        stress=load_shocks(SHOCKS_EXAMPLE),
    )
    run = run_engine(returns, config, build_frontier=False)
    sheet = run.tearsheet()
    assert sheet.stress is not None, "the config's scenarios reach the tearsheet"
    payload = backtest_payload(sheet.run, tearsheet=sheet)
    assert payload["stress"]["n_scenarios"] == len(config.stress)
    # The tearsheet stresses the book the run *ended* on, and the payload says
    # which date that was, so two documents cannot be compared as if they
    # described the same holdings. Spelled ISO, like every other date here:
    # the tearsheet's own ``str(Timestamp)`` form is not.
    assert payload["stress"]["as_of"] == pd.Timestamp(
        sheet.metadata["stress_as_of"]
    ).isoformat()
    assert "T" in payload["stress"]["as_of"]
    assert backtest_payload(sheet.run)["stress"] is None


def test_a_solve_s_own_stress_has_no_as_of(returns):
    """Null there means "the weights this payload already carries"."""
    config = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
        stress=load_shocks(SHOCKS_EXAMPLE),
    )
    run = run_engine(returns, config, build_frontier=False, run_stress=True)
    assert optimization_payload(run)["stress"]["as_of"] is None
