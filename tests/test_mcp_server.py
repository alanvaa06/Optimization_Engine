"""The MCP server's tool surface, exercised through the server.

Calling the decorated functions directly would test the bodies and skip
everything that makes this a server: schema generation from the signatures,
argument coercion, and the error wrapping that decides whether a caller sees
a useful message or "Error executing tool optimize".

The SDK needs Python 3.10, one minor above this package's floor, so the
whole module skips where it is unavailable rather than failing the suite on
3.9.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mcp.server.mcpserver", reason="the `mcp` extra is not installed")

from optimization_engine.mcp_server import ToolError, mcp  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "example_multi_asset.yaml")


def call(name: str, args: dict):
    """Invoke a tool the way a client would, returning its structured result."""
    result = asyncio.run(mcp.call_tool(name, args))
    assert not result.is_error, f"{name} reported an error"
    return result.structured_content


def failure(name: str, args: dict) -> str:
    """Invoke a tool expecting it to fail, returning the message a client sees."""
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 — SDK wraps the type
        asyncio.run(mcp.call_tool(name, args))
    return str(excinfo.value)


def test_every_tool_is_registered_with_a_schema():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {
        "list_optimizers",
        "describe_optimizer",
        "check_mandate",
        "optimize",
        "backtest",
    }
    # A tool whose parameters did not make it into the schema is unusable:
    # the client has nothing to fill in.
    assert "name" in tools["describe_optimizer"].input_schema["properties"]
    assert "sample" in tools["optimize"].input_schema["properties"]
    # The description is what a model reads when choosing a tool, so an
    # empty one is a silent failure rather than a cosmetic one.
    assert all(t.description for t in tools.values())


def test_list_optimizers_enumerates_the_methods():
    payload = call("list_optimizers", {})
    names = [o["name"] for o in payload["optimizers"]]
    assert "risk_parity" in names
    assert "max_sharpe" in names
    assert all(o["summary"] for o in payload["optimizers"])


def test_describe_optimizer_reports_the_contract():
    payload = call("describe_optimizer", {"name": "risk_parity"})
    assert payload["name"] == "risk_parity"
    assert payload["requires"]["covariance"] is True
    assert isinstance(payload["supports"]["turnover"], bool)


def test_optimize_returns_weights_and_the_evidence():
    payload = call("optimize", {"sample": True, "optimizer": "risk_parity"})
    weights = payload["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert payload["solver"]
    # Weights without these are what this engine exists not to return.
    assert payload["diagnostics"]["effective_n"] is not None
    assert payload["diagnostics"]["effective_n_risk"] is not None
    assert payload["covariance"]["is_psd"] is True


def test_check_and_optimize_agree_about_the_same_mandate():
    """The regression this pair exists for.

    `check_mandate` used to derive expected returns as zeros when the config
    carried no `expected_returns` block, while `optimize` derived them from
    the return history. The check therefore validated a mandate the solve
    never saw: a reachable range of exactly zero to zero, against a solve
    that returned a real expected return outside it.
    """
    checked = call("check_mandate", {"sample": True, "optimizer": "max_sharpe"})
    lo = checked["feasibility"]["min_return"]
    hi = checked["feasibility"]["max_return"]
    assert lo is not None and hi is not None
    assert hi > lo, "a degenerate range means the two are out of step again"

    solved = call("optimize", {"sample": True, "optimizer": "max_sharpe"})
    achieved = solved["metrics"]["expected_return"]
    assert lo - 1e-9 <= achieved <= hi + 1e-9, (
        f"the solve returned {achieved}, outside the {lo}..{hi} range check promised"
    )


def test_backtest_returns_the_hashes_that_identify_the_run():
    payload = call(
        "backtest",
        {"sample": True, "optimizer": "risk_parity", "lookback": 504, "rebalance_every": 252},
    )
    assert payload["spec_hash"]
    assert payload["result_hash"]
    assert payload["window"]["n_periods"] > 0


def test_a_config_path_is_honoured():
    payload = call("check_mandate", {"config_path": CONFIG, "sample": True})
    assert payload["ready"] is True


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "No data given"),
        ({"sample": True, "prices_path": "/nope.csv"}, "not both"),
        ({"sample": True, "config_path": "/nope.yaml"}, "No such config file"),
        ({"prices_path": "/nope.csv"}, "No such price file"),
    ],
)
def test_anticipated_failures_keep_their_message(args, expected):
    """A `ToolError` reaches the client; anything else is wrapped and lost.

    The SDK turns an unanticipated exception into `UnexpectedToolError` with
    the text replaced by "Error executing tool optimize" — a failure an
    agent can neither act on nor explain. Every reachable bad-input path
    therefore has to raise `ToolError` specifically.
    """
    assert expected in failure("optimize", args)


def test_an_unknown_optimizer_names_the_alternatives():
    message = failure("describe_optimizer", {"name": "definitely_not_a_method"})
    assert "definitely_not_a_method" in message
    # A dead end that lists the valid names is one call from recovery.
    assert "risk_parity" in message


def test_tool_error_is_the_sdk_class():
    """Guards the import in `mcp_server`, which is what makes the above work."""
    assert issubclass(ToolError, Exception)
    assert ToolError.__name__ == "ToolError"
