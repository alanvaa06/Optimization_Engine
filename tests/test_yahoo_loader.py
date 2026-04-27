"""Tests for the Yahoo Finance price loader.

All network calls are mocked — these tests run offline and deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.data import yahoo  # noqa: E402
from optimization_engine.data.yahoo import (  # noqa: E402
    YahooFinanceError,
    _validate_tickers,
    load_prices_yahoo,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal frames matching yfinance's two output shapes.
# ---------------------------------------------------------------------------


def _multi_ticker_frame(tickers: list[str], n: int = 30) -> pd.DataFrame:
    """Mimic ``yf.download(group_by='column', tickers=[...])`` output."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(0)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], tickers],
        names=[None, "Ticker"],
    )
    data = rng.uniform(50, 200, size=(n, len(columns)))
    return pd.DataFrame(data, index=dates, columns=columns)


def _single_ticker_frame(ticker: str, n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.uniform(50, 200, size=(n, 5)),
        index=dates,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ticker",
    ["AAPL", "spy", "BRK.B", "BTC-USD", "^GSPC", "ES=F", "EURUSD=X"],
)
def test_validate_accepts_real_tickers(ticker: str):
    assert _validate_tickers([ticker]) == [ticker.upper()]


@pytest.mark.parametrize(
    "bad",
    [
        "AAPL; rm -rf /",      # shell injection attempt
        "AAPL\nDROP TABLE",     # newline / sql-ish payload
        "<script>alert(1)</script>",
        "AAPL?q=hack",
        "AAPL&extra=1",
        "AAPL%20",
        "../../etc/passwd",
        "",                     # empty
        "A" * 25,               # over length
    ],
)
def test_validate_rejects_dangerous_tickers(bad: str):
    with pytest.raises(YahooFinanceError):
        _validate_tickers([bad])


def test_validate_rejects_non_strings():
    with pytest.raises(YahooFinanceError):
        _validate_tickers([123])  # type: ignore[list-item]


def test_validate_deduplicates_and_uppercases():
    assert _validate_tickers(["aapl", "AAPL", "msft"]) == ["AAPL", "MSFT"]


def test_validate_requires_at_least_one_ticker():
    with pytest.raises(YahooFinanceError):
        _validate_tickers([])


# ---------------------------------------------------------------------------
# load_prices_yahoo — happy paths with mocked yfinance
# ---------------------------------------------------------------------------


def test_load_multi_ticker(monkeypatch):
    tickers = ["AAPL", "MSFT", "SPY"]
    frame = _multi_ticker_frame(tickers)

    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    out = load_prices_yahoo(tickers, period="1y")
    assert list(out.columns) == tickers
    assert out.shape[0] == frame.shape[0]
    assert pd.api.types.is_datetime64_any_dtype(out.index)


def test_load_single_ticker(monkeypatch):
    frame = _single_ticker_frame("AAPL")

    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    out = load_prices_yahoo("aapl", period="6mo")
    assert list(out.columns) == ["AAPL"]
    assert (out["AAPL"] > 0).all()


def test_load_extracts_requested_field(monkeypatch):
    tickers = ["SPY", "QQQ"]
    frame = _multi_ticker_frame(tickers)
    captured: dict = {}

    def fake_download(**kw):
        captured.update(kw)
        return frame

    fake_yf = type("FakeYF", (), {"download": staticmethod(fake_download)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    out = load_prices_yahoo(tickers, period="1y", field="Close")
    expected = frame["Close"][tickers]
    np.testing.assert_array_equal(out.values, expected.values)
    assert captured["tickers"] == tickers
    assert captured["progress"] is False
    assert captured["threads"] is False


def test_load_with_start_end(monkeypatch):
    tickers = ["SPY"]
    frame = _multi_ticker_frame(tickers)

    captured: dict = {}

    def fake_download(**kw):
        captured.update(kw)
        return frame

    fake_yf = type("FakeYF", (), {"download": staticmethod(fake_download)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    load_prices_yahoo(tickers, start="2023-01-01", end="2023-12-31")
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-12-31"
    assert "period" not in captured


def test_load_string_input_is_split(monkeypatch):
    tickers = ["AAPL", "MSFT", "GOOGL"]
    frame = _multi_ticker_frame(tickers)

    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    out = load_prices_yahoo("aapl, msft GOOGL", period="1y")
    assert list(out.columns) == tickers


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_load_requires_period_or_start(monkeypatch):
    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: None)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    with pytest.raises(YahooFinanceError, match="period.*start"):
        load_prices_yahoo(["SPY"])


def test_load_propagates_yfinance_errors(monkeypatch):
    def boom(**kw):
        raise RuntimeError("network down")

    fake_yf = type("FakeYF", (), {"download": staticmethod(boom)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    with pytest.raises(YahooFinanceError, match="yfinance download failed"):
        load_prices_yahoo(["SPY"], period="1y")


def test_load_handles_empty_response(monkeypatch):
    empty = pd.DataFrame()
    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: empty)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    with pytest.raises(YahooFinanceError, match="no data"):
        load_prices_yahoo(["SPY"], period="1y")


def test_load_handles_all_nan_response(monkeypatch):
    tickers = ["SPY"]
    frame = _multi_ticker_frame(tickers)
    frame.loc[:, ("Close", "SPY")] = np.nan

    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    with pytest.raises(YahooFinanceError, match="missing"):
        load_prices_yahoo(tickers, period="1y")


def test_load_unknown_field_raises(monkeypatch):
    tickers = ["SPY"]
    frame = _multi_ticker_frame(tickers)

    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    with pytest.raises(YahooFinanceError, match="Field 'Bogus'"):
        load_prices_yahoo(tickers, period="1y", field="Bogus")


# ---------------------------------------------------------------------------
# End-to-end: hand the loader's frame to the engine.
# ---------------------------------------------------------------------------


def test_yahoo_output_is_engine_compatible(monkeypatch):
    from optimization_engine.config import EngineConfig, OptimizerSpec
    from optimization_engine.data.loader import prices_to_returns
    from optimization_engine.engine import run_engine

    tickers = ["SPY", "AGG", "GLD", "QQQ"]
    frame = _multi_ticker_frame(tickers, n=400)
    fake_yf = type("FakeYF", (), {"download": staticmethod(lambda **kw: frame)})
    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: fake_yf)

    prices = load_prices_yahoo(tickers, period="2y")
    returns = prices_to_returns(prices)
    cfg = EngineConfig(
        expected_returns={t: 0.05 for t in tickers},
        bounds={t: [0.0, 1.0] for t in tickers},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    assert run.result.weights.sum() == pytest.approx(1.0, abs=1e-3)
