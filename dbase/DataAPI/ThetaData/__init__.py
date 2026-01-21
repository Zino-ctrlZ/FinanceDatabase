"""
ThetaData API Client - Version-Agnostic Interface
==================================================

This module provides a unified interface to ThetaData's REST API for accessing
historical and real-time options market data. It automatically selects between
API V2 (legacy) and V3 (recommended) based on configuration.

Overview
--------
ThetaData offers comprehensive options market data including:
- End-of-day (EOD) and intraday OHLC data
- Historical and realtime quote data (bid/ask/midpoint)
- Open interest data (historical and bulk)
- Option chain snapshots and contract listings
- Bulk data retrieval for entire expirations

This module serves as a version-agnostic facade that:
1. Detects which API version to use via THETADATA_USE_V3 environment variable
2. Imports appropriate functions from v2.py or v3/endpoints.py
3. Provides consistent function signatures across versions
4. Exports utility functions for data processing

Version Selection
-----------------
The API version is selected at import time:

.. code-block:: python

    # Use V3 (default and recommended)
    import os
    os.environ["THETADATA_USE_V3"] = "true"  # or omit (defaults to true)
    from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

    # Use V2 (legacy)
    os.environ["THETADATA_USE_V3"] = "false"
    from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

Exported Functions
------------------
The following functions are exported from either v2.py or v3.endpoints:

Data Retrieval:
    - retrieve_quote_rt : Realtime quote snapshot
    - retrieve_bulk_quote_rt : Bulk realtime quotes (V3 only)
    - retrieve_quote : Historical intraday quotes
    - retrieve_ohlc : Historical intraday OHLC
    - retrieve_eod_ohlc : End-of-day OHLC
    - retrieve_bulk_eod : Bulk EOD for entire expiration
    - retrieve_openInterest : Historical open interest
    - retrieve_bulk_open_interest : Bulk open interest
    - retrieve_chain_bulk : Complete option chain snapshot
    - list_contracts : List available contracts for a date
    - list_dates : List available dates for a contract (V3 only)

Utility Functions:
    - resample : Resample data to different timeframes
    - bootstrap_ohlc : Format OHLC data structure
    - convert_time_to_miliseconds : Convert time to milliseconds
    - extract_numeric_value : Parse interval strings (e.g., '5m')
    - identify_length : Calculate interval length in minutes
    - is_theta_data_retrieval_successful : Check if retrieval succeeded

Proxy Functions:
    - ping_proxy : Test proxy connectivity
    - set_use_proxy : Enable/disable proxy usage
    - set_should_schedule : Configure request scheduling
    - get_proxy_url : Get current proxy URL

Patch Functions:
    - quote_to_eod_patch : Convert quote data to EOD format (fallback)

Key Differences: V2 vs V3
--------------------------
V3 improvements over V2:
- Port 25503 instead of 25510
- Native bulk realtime quote endpoint
- Ability to list available dates for contracts
- Multi-threaded range fetching for endpoints without native date ranges
- Improved error messages and handling
- More consistent data structures and column naming
- Better ticker symbol change handling
- Standardized datetime formatting

V2 limitations:
- No bulk realtime quotes (returns None)
- No list dates function (returns empty array)
- Manual looping required for some queries
- Less consistent error handling

Migration from V2 to V3
-----------------------
Most function signatures are identical. Key changes:

1. Enable V3:
   .. code-block:: python

       os.environ["THETADATA_USE_V3"] = "true"

2. New functions available in V3:
   .. code-block:: python

       # Bulk realtime quotes (V3 only)
       bulk_quotes = retrieve_bulk_quote_rt(symbol='AAPL', exp='2024-12-20')

       # List available dates (V3 only)
       dates = list_dates(symbol='AAPL', exp='2024-12-20', right='C', strike=180.0)

3. Port change:
   - V2: http://localhost:25510/v2/...
   - V3: http://localhost:25503/v3/...

   Ensure ThetaData Terminal is configured correctly.

Usage Examples
--------------
.. code-block:: python

    from dbase.DataAPI.ThetaData import (
        retrieve_eod_ohlc,
        retrieve_quote,
        list_contracts,
        resample
    )

    # Get EOD data
    eod_data = retrieve_eod_ohlc(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    # Get intraday quotes with 5-minute intervals
    quotes = retrieve_quote(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0,
        start_date='2024-12-01',
        end_date='2024-12-15',
        interval='5m'
    )

    # Resample to different timeframe
    hourly = resample(quotes, interval='1h')

    # List available contracts
    contracts = list_contracts(symbol='AAPL', date='2024-01-15')

Configuration
-------------
Environment Variables:
    THETADATA_USE_V3 : str
        Set to "false" to use V2, otherwise uses V3 (default: "true")

    PROXY_URL : str
        URL of proxy server for distributed fetching (optional)

    GEN_CACHE_PATH : str
        Path for caching latency logs (default: ".cache")

Requirements
------------
- ThetaData Terminal must be running locally
- V2: Port 25510, V3: Port 25503
- Internet connection for API calls
- Valid ThetaData subscription

See Also
--------
- V2 Module: dbase.DataAPI.ThetaData.v2
- V3 Module: dbase.DataAPI.ThetaData.v3
- Exceptions: dbase.DataAPI.ThetaExceptions
- Utilities: dbase.DataAPI.ThetaData.utils

Notes
-----
- V3 is recommended for all new code
- V2 is maintained for backward compatibility only
- Function signatures are mostly compatible between versions
- Data structures may differ slightly between versions
"""

from .proxy import ping_proxy, set_use_proxy, set_should_schedule, get_proxy_url
from .utils import (
    resample,
    bootstrap_ohlc,
    convert_time_to_miliseconds,
    extract_numeric_value,
    identify_length,
    is_theta_data_retrieval_successful,
)
from dbase.utils import enforce_bus_hours, add_eod_timestamp
import pandas as pd
import numpy as np
from ..ThetaExceptions import raise_thetadata_exception
import os
from trade.helpers.Logging import setup_logger


logger = setup_logger("dbase.DataAPI.ThetaData", stream_log_level="INFO")

## Determine whether to use V2 or V3 of the ThetaData API
USE_V2 = (
    os.environ.get("THETADATA_USE_V3", "false").lower() == "false"
)  # V2 usage by default


def get_use_v2():
    return USE_V2


if get_use_v2():
    logger.info("Using V2 of the ThetaData API")
    from .v2 import (
        retrieve_quote_rt,
        retrieve_quote,
        retrieve_ohlc,
        retrieve_eod_ohlc,
        retrieve_bulk_eod,
        retrieve_openInterest,
        retrieve_bulk_open_interest,
        retrieve_chain_bulk,
        list_contracts,
        list_dates
    )

    ## V2 does not support bulk realtime quotes
    def retrieve_bulk_quote_rt(*args, **kwargs):
        logger.warning("Bulk realtime quotes not supported in V2 ThetaData API")
        return None

else:
    logger.info("Using V3 of the ThetaData API")
    from .v3.endpoints import (
        _retrieve_quote_rt as retrieve_quote_rt,
        _retrieve_bulk_quote_rt as retrieve_bulk_quote_rt,
        _retrieve_quote as retrieve_quote,
        _retrieve_ohlc as retrieve_ohlc,
        _retrieve_eod_ohlc as retrieve_eod_ohlc,
        _retrieve_bulk_eod as retrieve_bulk_eod,
        _retrieve_openInterest as retrieve_openInterest,
        _retrieve_bulk_open_interest as retrieve_bulk_open_interest,
        _retrieve_chain_bulk as retrieve_chain_bulk,
        _list_contracts as list_contracts,
        _list_dates as list_dates,
    )


## Patch function to convert quote data to EOD format if needed
def quote_to_eod_patch(
    symbol: str,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    print_url=False,
    *,
    quote_func=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve end-of-day quote data for a specific option contract.
    Sometimes ThetaData EOD API has issues with parsing. If the specific error is caught,
    this function will try to retrieve the quote data and convert it to EOD format.
    Parameters
    ----------
    symbol : str
        The root symbol of the option (e.g., 'AAPL').
    end_date : str
        The end date for the data retrieval in 'YYYY-MM-DD' format.
    exp : str
        The expiration date of the option in 'YYYY-MM-DD' format.
    right : str
        The option type, either 'c' for call or 'p' for put.
    start_date : str
        The start date for the data retrieval in 'YYYY-MM-DD' format.
    strike : float
        The strike price of the option.
    print_url : bool, optional
        Whether to print the URL used for the data request (default is False).
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the end-of-day quote data for the specified option contract.
    """
    if quote_func is None:
        quote_func = retrieve_quote

    q = quote_func(
        symbol=symbol,
        end_date=end_date,
        exp=exp,
        right=right,
        start_date=start_date,
        strike=strike,
        print_url=print_url,
        interval="1d",
    )
    q.index = add_eod_timestamp(q.index)
    if not q.empty:
        q_to_eod = q[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Bid_size",
                "Closebid",
                "Ask_size",
                "Closeask",
                "Midpoint",
                "Weighted_midpoint",
            ]
        ]
    else:
        q_to_eod = pd.DataFrame(
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Bid_size",
                "Closebid",
                "Ask_size",
                "Closeask",
                "Midpoint",
                "Weighted_midpoint",
            ]
        )
    q_to_eod.rename(
        columns={
            "Closebid": "CloseBid",
            "Closeask": "CloseAsk",
        },
        inplace=True,
    )
    q_to_eod.index = pd.to_datetime(q_to_eod.index)
    q_to_eod.index.name = "Datetime"
    return q_to_eod


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
