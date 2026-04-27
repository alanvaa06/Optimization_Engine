"""Data loading and covariance estimation."""

from optimization_engine.data.covariance import (
    covariance_matrix,
    expected_returns_from_history,
)
from optimization_engine.data.fred import (
    FREDError,
    load_fred_series,
    load_risk_free_rate,
)
from optimization_engine.data.fx import (
    FXError,
    convert_prices_to_base,
    fetch_fx_to_base,
    fetch_fx_to_usd,
    supported_currencies,
)
from optimization_engine.data.loader import (
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.data.yahoo import (
    YahooFinanceError,
    load_prices_yahoo,
)

__all__ = [
    "covariance_matrix",
    "expected_returns_from_history",
    "load_prices",
    "prices_to_returns",
    "sample_dataset",
    "YahooFinanceError",
    "load_prices_yahoo",
    "FREDError",
    "load_fred_series",
    "load_risk_free_rate",
    "FXError",
    "convert_prices_to_base",
    "fetch_fx_to_base",
    "fetch_fx_to_usd",
    "supported_currencies",
]
