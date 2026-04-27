"""Tests for the FRED loader.

All HTTP calls are mocked via ``_fetch_fred_csv`` so the suite runs
offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.data import fred  # noqa: E402
from optimization_engine.data.fred import (  # noqa: E402
    FREDError,
    _validate_series_ids,
    load_fred_series,
    load_risk_free_rate,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sid",
    ["DGS10", "DEXMXUS", "DFF", "VIXCLS", "CPIAUCSL", "DGS3MO", "EFFR"],
)
def test_validate_accepts_real_series_ids(sid: str):
    assert _validate_series_ids([sid]) == [sid]


@pytest.mark.parametrize(
    "bad",
    [
        "DGS10; rm -rf /",
        "DGS10\nDROP",
        "<script>",
        "../../etc/passwd",
        "DGS-10",       # hyphen rejected (FRED ids are alnum + underscore)
        "DGS 10",
        "id=DGS10",
        "",
        "A" * 35,
    ],
)
def test_validate_rejects_dangerous(bad: str):
    with pytest.raises(FREDError):
        _validate_series_ids([bad])


def test_validate_dedupe_and_uppercase():
    assert _validate_series_ids(["dgs10", "DGS10", "vixcls"]) == ["DGS10", "VIXCLS"]


# ---------------------------------------------------------------------------
# load_fred_series — happy path
# ---------------------------------------------------------------------------


def _frame(series_id: str, n: int = 30, value: float = 4.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({series_id: np.full(n, value)}, index=pd.DatetimeIndex(dates, name="date"))


def test_load_single_series(monkeypatch):
    df = _frame("DGS10")
    monkeypatch.setattr(fred, "_fetch_fred_csv", lambda sid, **kw: df)
    out = load_fred_series("DGS10")
    assert list(out.columns) == ["DGS10"]
    np.testing.assert_array_equal(out["DGS10"].values, df["DGS10"].values)


def test_load_multi_series_concat(monkeypatch):
    fakes = {"DGS10": _frame("DGS10", value=4.0), "VIXCLS": _frame("VIXCLS", value=18.0)}
    monkeypatch.setattr(fred, "_fetch_fred_csv", lambda sid, **kw: fakes[sid])
    out = load_fred_series(["DGS10", "VIXCLS"])
    assert list(out.columns) == ["DGS10", "VIXCLS"]
    assert (out["DGS10"] == 4.0).all()
    assert (out["VIXCLS"] == 18.0).all()


def test_load_string_input_is_split(monkeypatch):
    fakes = {"DGS10": _frame("DGS10"), "VIXCLS": _frame("VIXCLS")}
    monkeypatch.setattr(fred, "_fetch_fred_csv", lambda sid, **kw: fakes[sid])
    out = load_fred_series("dgs10, VIXCLS")
    assert list(out.columns) == ["DGS10", "VIXCLS"]


def test_load_passes_dates_through(monkeypatch):
    captured: dict = {}

    def fake_fetch(sid, **kw):
        captured.update(kw)
        return _frame(sid)

    monkeypatch.setattr(fred, "_fetch_fred_csv", fake_fetch)
    load_fred_series(["DGS10"], start="2023-01-01", end="2023-12-31")
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-12-31"


def test_load_drops_all_nan_rows(monkeypatch):
    df = _frame("DGS10", n=10)
    df.iloc[5:, 0] = np.nan
    monkeypatch.setattr(fred, "_fetch_fred_csv", lambda sid, **kw: df)
    out = load_fred_series("DGS10")
    assert out.shape[0] == 5


def test_load_risk_free_rate_divides_by_100(monkeypatch):
    df = _frame("DGS10", value=4.5)
    monkeypatch.setattr(fred, "_fetch_fred_csv", lambda sid, **kw: df)
    rate = load_risk_free_rate("DGS10")
    assert isinstance(rate, pd.Series)
    assert (rate == 0.045).all()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_load_propagates_fetch_errors(monkeypatch):
    def boom(sid, **kw):
        raise FREDError("network down")

    monkeypatch.setattr(fred, "_fetch_fred_csv", boom)
    with pytest.raises(FREDError, match="network down"):
        load_fred_series(["DGS10"])
