"""Data loading and covariance estimation."""

from optimization_engine.data.covariance import (
    CovarianceDiagnostics,
    covariance_diagnostics,
    covariance_matrix,
    expected_returns_from_history,
    james_stein_shrinkage,
    nearest_psd,
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
from optimization_engine.data.quality import (
    DataIssue,
    DataQualityReport,
    align_panel,
    analyze_prices,
)
from optimization_engine.data.yahoo import (
    YahooFinanceError,
    load_prices_yahoo,
)

__all__ = [
    "covariance_matrix",
    "covariance_diagnostics",
    "CovarianceDiagnostics",
    "expected_returns_from_history",
    "james_stein_shrinkage",
    "nearest_psd",
    "DataIssue",
    "DataQualityReport",
    "align_panel",
    "analyze_prices",
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
