"""
ThetaData API Client
====================

Unified access to ThetaData's REST API for options market data. The client can
route calls to V2 (legacy) or V3 (recommended) based on THETADATA_USE_V3.

Version Selection
-----------------
The default behavior of this package uses the per-call switcher in
switcher.py so the environment variable is evaluated on each call.

You can also use the import-time switcher in import_switcer.py when you want
the version locked at import time.

Environment variable:
    THETADATA_USE_V3 : str
        Set to "false" for V2. Any other value selects V3.

Data Retrieval
--------------
- Realtime and historical quotes
- Intraday and EOD OHLC data
- Open interest (single and bulk)
- Option chain snapshots and contract listings

Utilities
---------
Resampling, OHLC bootstrapping, interval parsing, proxy helpers, and a
quote-to-EOD patch for edge cases.

See Also
--------
- V2 API: dbase.DataAPI.ThetaData.v2
- V3 API: dbase.DataAPI.ThetaData.v3
- Per-call switcher: dbase.DataAPI.ThetaData.switcher
- Import-time switcher: dbase.DataAPI.ThetaData.import_switcer
"""

from .switcher import (
    retrieve_quote_rt,
    retrieve_bulk_quote_rt,
    retrieve_quote,
    retrieve_ohlc,
    retrieve_eod_ohlc,
    retrieve_bulk_eod,
    retrieve_openInterest,
    retrieve_bulk_open_interest,
    retrieve_chain_bulk,
    list_contracts,
    list_dates,
    ping_proxy,
    quote_to_eod_patch,
)
from .proxy import PingProxyResult, set_use_proxy, set_should_schedule, get_proxy_url
from .utils import (
    resample,
    bootstrap_ohlc,
    convert_time_to_miliseconds,
    extract_numeric_value,
    identify_length,
    is_theta_data_retrieval_successful,
)
from dbase.utils import enforce_bus_hours, add_eod_timestamp
from ..ThetaExceptions import raise_thetadata_exception

DO_NOT_EXPORT = [
    "retrieve_eod_ohlc_async",
    "retrieve_option_ohlc",
    "retrieve_openInterest_async",
]


def __getattr__(name):
    """Handle deprecated function imports with helpful error messages."""
    if name in DO_NOT_EXPORT:

        def dummy_function(*args, **kwargs):
            raise AttributeError(
                f"'{name}' has been removed from the public API. "
                f"Please update your code to use synchronous alternatives."
            )

        return dummy_function
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


## Define module exports
__all__ = [
    "retrieve_quote_rt",
    "retrieve_bulk_quote_rt",
    "retrieve_quote",
    "retrieve_ohlc",
    "retrieve_eod_ohlc",
    "retrieve_bulk_eod",
    "retrieve_openInterest",
    "retrieve_bulk_open_interest",
    "retrieve_chain_bulk",
    "list_contracts",
    "list_dates",
    "ping_proxy",
    "PingProxyResult",
    "set_use_proxy",
    "get_proxy_url",
    "resample",
    "bootstrap_ohlc",
    "convert_time_to_miliseconds",
    "extract_numeric_value",
    "identify_length",
    "raise_thetadata_exception",
    "enforce_bus_hours",
    "add_eod_timestamp",
    "quote_to_eod_patch",
    "set_should_schedule",
    "is_theta_data_retrieval_successful",
]
