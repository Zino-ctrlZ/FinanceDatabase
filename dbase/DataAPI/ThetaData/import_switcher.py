"""
ThetaData API Import-Time Switcher
==================================

This module selects between V2 (legacy) and V3 (recommended) at import time
using the THETADATA_USE_V3 environment variable. Once imported, the selection
is fixed for the process. For per-call switching, use switcher.py instead.
"""

from .proxy import set_use_proxy, set_should_schedule, get_proxy_url
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
from ..ThetaExceptions import raise_thetadata_exception
import os
from trade.helpers.Logging import setup_logger


logger = setup_logger("dbase.DataAPI.ThetaData", stream_log_level="INFO")

## Determine whether to use V2 or V3 of the ThetaData API
USE_V2 = os.environ.get("THETADATA_USE_V3", "false").lower() == "false"  # V2 usage by default


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
        list_dates,
    )
    from .proxy import ping_proxy_v2 as ping_proxy

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
    from .proxy import ping_proxy_v3 as ping_proxy


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
