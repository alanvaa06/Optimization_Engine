from __future__ import annotations

import pandas as pd

from optimization_engine.ui_state import yahoo_cache_key, yahoo_prices_for_rerun


def test_yahoo_prices_survive_rerun_after_fetch_button_resets():
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    state: dict[str, object] = {}
    key = yahoo_cache_key(("SPY",), "5y", None, None, "1d")

    fetched = yahoo_prices_for_rerun(
        fetch_clicked=True,
        cache_key=key,
        state=state,
        fetch_prices=lambda: prices,
    )

    rerun = yahoo_prices_for_rerun(
        fetch_clicked=False,
        cache_key=key,
        state=state,
        fetch_prices=lambda: (_ for _ in ()).throw(AssertionError("should not refetch")),
    )

    pd.testing.assert_frame_equal(fetched, prices)
    pd.testing.assert_frame_equal(rerun, prices)
