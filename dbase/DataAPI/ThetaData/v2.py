"""
ThetaData API V2 Client Module (Legacy)
========================================

This module provides a Python interface to ThetaData's V2 REST API for accessing
historical and real-time options market data. This is the legacy API that predates V3.

DEPRECATION NOTICE
----------------------
This V2 API is maintained for backward compatibility. New code should use the V3 API
(dbase.DataAPI._ThetaData.v3) which offers improved performance, better error handling,
and more consistent data structures.

Overview
--------
This module provides comprehensive access to ThetaData's options market data including:
    - End-of-day (EOD) OHLC data for individual contracts and bulk chains
    - Intraday OHLC data with customizable intervals
    - Historical and realtime quote data with bid/ask spreads
    - Open interest data (historical and bulk snapshots)
    - Option chain snapshots at specific times
    - Contract listings for available options
    - Greeks snapshots (delta, gamma, theta, vega, rho)

Key Features
------------
- Automatic retry logic with exponential backoff for API failures
- Ticker symbol change handling for corporate actions (FB → META, etc.)
- Proxy support for distributed data fetching
- Automatic data resampling to requested intervals
- Business hours enforcement for intraday data
- Comprehensive error handling and logging
- Support for both synchronous and async operations (select functions)

Architecture
------------
Functions follow a naming convention:
    - retrieve_* : Core data retrieval functions
    - retrieve_*_async : Async versions of select functions
    - list_* : Functions for listing available contracts
    - *_snapshot : Realtime snapshot functions for current market data

Data Types
----------
EOD (End of Day):
    - retrieve_eod_ohlc() - Single contract EOD OHLC
    - retrieve_eod_ohlc_async() - Async version
    - retrieve_bulk_eod() - All contracts for an expiration

Intraday:
    - retrieve_ohlc() - Intraday OHLC with custom intervals
    - retrieve_quote() - Intraday quote data (bid/ask/midpoint)
    - retrieve_quote_rt() - Realtime quote snapshot

Open Interest:
    - retrieve_openInterest() - Historical OI for single contract
    - retrieve_openInterest_async() - Async version
    - retrieve_bulk_open_interest() - Bulk OI for all contracts

Chain/Contracts:
    - retrieve_chain_bulk() - Complete chain snapshot at specific time
    - list_contracts() - List available contracts for a date

Snapshots:
    - greek_snapshot() - Current greeks for all contracts
    - ohlc_snapshot() - Current OHLC for all contracts
    - quote_snapshot() - Current quotes for all contracts
    - open_interest_snapshot() - Current OI for all contracts

Ticker Change Handling
----------------------
The module automatically handles ticker symbol changes due to corporate actions.
When querying historical data for symbols like META (formerly FB):
    1. Detects ticker change events from TICK_CHANGE_ALIAS mapping
    2. Splits queries across change date (queries FB before 2022-06-09, META after)
    3. Combines results seamlessly into single DataFrame
    4. Removes duplicate records at transition points

This is handled transparently via resolve_ticker_history() wrapper function.

Usage Examples
--------------
```python
from dbase.DataAPI._ThetaData import v2

# Get EOD data for a single option contract
data = v2.retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Get bulk EOD data for all strikes/rights in an expiration
bulk = v2.retrieve_bulk_eod(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Get intraday quotes with 5-minute intervals
quotes = v2.retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'
)

# List all available contracts for a date
contracts = v2.list_contracts(
    symbol='AAPL',
    start_date='2024-01-15'
)

# Get realtime quote snapshot
snapshot = v2.retrieve_quote_rt(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0
)
```

Proxy Support
-------------
All functions support proxying requests through a remote server:

```python
data = v2.retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31',
    proxy='http://remote-server:8080/thetadata'
)
```

Error Handling
--------------
The module uses custom exceptions from ThetaExceptions:
    - ThetaDataNotFound: No data available for request
    - ThetaDataOSLimit: Rate limit exceeded
    - ThetaDataDisconnected: Connection lost to ThetaData Terminal
    - ThetaDataServerRestart: Server restarted during request
    - ThetaDataParseError: API returned malformed data

Functions with @backoff decorators automatically retry on transient errors:
    - Exponential backoff between retries
    - Maximum 5 retry attempts
    - Logs retry attempts for debugging

Data Format
-----------
All functions return pandas DataFrames with:
    - DatetimeIndex named 'Datetime'
    - Standardized column names (capitalized)
    - OHLC columns: Open, High, Low, Close, Volume
    - Quote columns: CloseBid, CloseAsk, Bid_size, Ask_size, Midpoint, Weighted_midpoint
    - Timestamps adjusted to business hours (9:30 AM - 4:00 PM ET)

Performance Considerations
--------------------------
- Bulk functions are more efficient than individual queries when retrieving multiple contracts
- Intraday queries are slower than EOD due to larger data volumes
- Use resampling functions to aggregate high-frequency data
- Enable proxy for distributed fetching in production environments
- Consider async versions for concurrent multi-contract queries

Notes
-----
- Requires ThetaData Terminal running locally (default port 25510)
- Strike prices must be provided as floats (e.g., 180.0, not 180000)
- Dates accept 'YYYY-MM-DD' format strings or pandas Timestamps
- Option rights: 'C' for calls, 'P' for puts (case-insensitive in API)
- Intervals: '1m', '5m', '15m', '30m', '1h', '1d' (intraday functions)

Migration to V3
---------------
When migrating to V3, key differences to note:
1. V3 uses port 25503 instead of 25510
2. V3 has native multi-threaded range fetching
3. V3 standardizes all datetime handling
4. V3 removes legacy formatting quirks
5. V3 has improved error messages

See Also
--------
- V3 API (recommended): dbase.DataAPI._ThetaData.v3
- ThetaData API Docs: https://http-docs.thetadata.us/
- Custom Exceptions: dbase.DataAPI.ThetaExceptions
- Utility Functions: dbase.DataAPI._ThetaData.utils
"""

from trade.helpers.Logging import setup_logger
from trade import reload_pricing_config
import requests
import re
import numpy as np
import time
from io import StringIO
import pandas as pd
import json
from datetime import datetime
from dbase.utils import add_eod_timestamp, enforce_bus_hours
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS
from trade.helpers.helper import compare_dates, generate_option_tick
from copy import deepcopy
from ..ThetaExceptions import (
    ThetaDataNotFound,
    ThetaDataOSLimit,
    ThetaDataDisconnected,
    ThetaDataServerRestart,
    ThetaDataParseError,
    raise_thetadata_exception,
)
from .proxy import get_proxy_url, schedule_kwargs, get_should_schedule
import backoff

reload_pricing_config()
from trade import PRICING_CONFIG # noqa

logger = setup_logger("dbase.DataAPI.ThetaData")
duplicated_logger = setup_logger("dbase.DataAPI.ThetaData.duplicated")
EMPTY_DF_SAMPLES = {
    "retrieve_openInterest": pd.DataFrame(
        columns=["Open_interest", "Date", "time", "Datetime"]
    )
}


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

    ## Quote v2 doesn't have volume data, so we set it to NaN
    q_to_eod["Volume"] = np.nan
    return q_to_eod


def resolve_ticker_history(kwargs, _callable, _type="historical"):
    if _type == "historical":
        tick = kwargs["symbol"]
        change_date = TICK_CHANGE_ALIAS[tick][-1]
        old_tick = TICK_CHANGE_ALIAS[tick][0]
        new_tick = TICK_CHANGE_ALIAS[tick][1]
        old_tick_kwargs = deepcopy(kwargs)
        new_tick_kwargs = deepcopy(kwargs)
        old_tick_kwargs["symbol"] = old_tick
        new_tick_kwargs["symbol"] = new_tick

        ## Retrieve the data for the old tick
        try:
            old_tick_data = (
                _callable(**old_tick_kwargs)
                if compare_dates.is_before(
                    pd.Timestamp(kwargs["start_date"]), pd.Timestamp(change_date)
                )
                else None
            )
            old_tick_data = (
                old_tick_data[old_tick_data.index.duplicated(keep="first")]
                if old_tick_data is not None
                else None
            )
        except ThetaDataNotFound as e:
            logger.info(
                f"No data found for Old_tick {old_tick} on {kwargs['start_date']}"
            )
            logger.info(f"Error: {e}")
            old_tick_data = None

        ## Retrieve the data for the new tick
        try:
            new_tick_data = (
                _callable(**new_tick_kwargs)
                if compare_dates.is_on_or_after(
                    pd.Timestamp(kwargs["exp"]), pd.Timestamp(change_date)
                )
                else None
            )  ## Opting for expiration date instead of end date cause data cannot go beyond expiration date
            new_tick_data = (
                new_tick_data[~new_tick_data.index.duplicated(keep="first")]
                if new_tick_data is not None
                else None
            )
        except ThetaDataNotFound as e:
            logger.info(f"No data found for new_tick {new_tick} on {kwargs['exp']}")
            logger.info(f"Error: {e}")
            new_tick_data = None

        ## If no data is found for the old tick, then we will just return the new tick data. Change to dataframe to avoid errors when concatenating
        if old_tick_data is None:
            logger.info(f"No data found for Old_tick {old_tick}")
            old_tick_data = EMPTY_DF_SAMPLES.get(_callable.__name__, pd.DataFrame())
        if new_tick_data is None:
            logger.info(f"No data found for new_tick {new_tick}")
            new_tick_data = EMPTY_DF_SAMPLES.get(_callable.__name__, pd.DataFrame())
        full_data = pd.concat([old_tick_data, new_tick_data])
        return full_data
    elif _type == "snapshot":
        tick = kwargs["symbol"]
        change_date = TICK_CHANGE_ALIAS[tick][-1]
        old_tick = TICK_CHANGE_ALIAS[tick][0]
        new_tick = TICK_CHANGE_ALIAS[tick][1]
        new_tick_kwargs = deepcopy(kwargs)
        new_tick_kwargs["symbol"] = (
            old_tick
            if compare_dates.is_before(
                pd.Timestamp(kwargs["start_date"]), pd.Timestamp(change_date)
            )
            else new_tick
        )
        return _callable(**new_tick_kwargs)


def request_from_proxy(thetaUrl, queryparam, instanceUrl, print_url=False):
    request_string = f"{thetaUrl}?{'&'.join([f'{key}={value}' for key, value in queryparam.items()])}"
    print(request_string) if print_url else None
    payload = json.dumps(
        {
            "url": request_string,
            "method": "GET",
        }
    )
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", instanceUrl, headers=headers, data=payload)
    return response


## Migrate to new API
def greek_snapshot(symbol, proxy=None):
    if not proxy:
        proxy = get_proxy_url()
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )


## Migrate to new API
def ohlc_snapshot(symbol, proxy=None):
    if not proxy:
        proxy = get_proxy_url()
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/ohlc"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )


## Migrate to new API
def open_interest_snapshot(symbol, proxy=None):
    if not proxy:
        proxy = get_proxy_url()
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )


## Migrate to new API
def quote_snapshot(symbol, proxy=None):
    if not proxy:
        proxy = get_proxy_url()
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def list_contracts(symbol, start_date, print_url=False, proxy=None, **kwargs):
    if not proxy:
        proxy = get_proxy_url()
    pass_kwargs = {"start_date": start_date, "symbol": symbol, "print_url": print_url}
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    url = "http://127.0.0.1:25510/v2/list/contracts/option/quote"
    querystring = {"start_date": start_date, "root": symbol, "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(pass_kwargs, list_contracts, _type="snapshot")

    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        print(response.url) if print_url else None

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if data.shape[0] == 0:
        logger.error(f"No contracts found for {symbol} on {start_date}")
        logger.error(f"response: {response.text}")
        logger.info(f"Kwargs: {locals()}")
        return
    data["strike"] = data.strike / 1000
    return data


## Break away function to helpers
def identify_length(string, integer, rt=False):
    """

    Identify the length of the timeframe in minutes based on the string and integer provided.
    Parameters

    ----------
    string : str
        The string representing the timeframe (e.g., 'm', 'h', 'd', 'w', 'm', 'y', 'q').
    integer : int
    The integer representing the number of units for the timeframe.
    rt : bool, optional
        If True, the function will use real-time values for timeframes. Default is False.
    Returns
    -------
    int
        The length of the timeframe in minutes.

    """
    if rt:
        TIMEFRAMES_VALUES = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}
    else:
        TIMEFRAMES_VALUES = {"d": 1, "w": 5, "m": 30, "y": 252, "q": 91}
    assert (
        string in TIMEFRAMES_VALUES.keys()
    ), f'Available timeframes are {TIMEFRAMES_VALUES.keys()}, recieved "{string}"'
    return integer * TIMEFRAMES_VALUES[string]


## Break away function to helpers
def extract_numeric_value(timeframe_str):
    match = re.findall(r"(\d+)([a-zA-Z]+)", timeframe_str)
    integers = [int(num) for num, _ in match][0]
    strings = [str(letter) for _, letter in match][0]
    return strings, integers


## Migrate to new API
# @backoff.on_exception(
#     backoff.expo,
#     (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
#     max_tries=5,
#     logger=logger,
# )
def retrieve_ohlc(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    start_time: str = PRICING_CONFIG["MARKET_OPEN_TIME"],
    print_url=False,
    proxy: str = None,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    proxy the endpoint to the proxy server http://<ip>:<port>/thetadata
    """

    if not proxy:
        proxy = get_proxy_url()
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    interval = PRICING_CONFIG["INTRADAY_AGG"]
    strike_og, start_og, end_og, exp_og, start_time_og = (
        strike,
        start_date,
        end_date,
        exp,
        start_time,
    )
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    ivl = identify_length(*extract_numeric_value(interval), rt=True) * 60000
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "ivl": ivl,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "start_time": start_time,
        "rth": False,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        print(response.url) if print_url else None

    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_ohlc")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
        logger.error("Nothing returned at all")
        logger.error("Column mismatch. Check log")
        logger.info(f"Kwargs: {locals()}")
        return
    else:
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data["time"] = data.Ms_of_day.apply(lambda c: convert_milliseconds(c))
        # use proxy option
        if proxy:
            quote_data = retrieve_quote(
                symbol,
                end_og,
                exp_og,
                right,
                start_og,
                strike_og,
                start_time=start_time_og,
                proxy=proxy,
            )
        else:
            quote_data = retrieve_quote(
                symbol,
                end_og,
                exp_og,
                right,
                start_og,
                strike_og,
                start_time=start_time_og,
            )
        ## Merging data into quote data, because quote data has complete dates, whereas OHLC only has dates when traded
        quote_data = quote_data[
            [
                "Date",
                "time",
                "Ask_size",
                "Closeask",
                "Closebid",
                "Bid_size",
                "Weighted_midpoint",
                "Midpoint",
            ]
        ]

        data = quote_data.merge(data, on=["Date", "time"], how="left")
        data.rename(
            columns={"Closeask": "CloseAsk", "Closebid": "CloseBid"}, inplace=True
        )
        data["Date"] = data["Date"].astype(str) + " " + data["time"]
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        data["Date3"] = data.Date2
        data["datetime"] = pd.to_datetime(data.Date3)
        data.set_index("datetime", inplace=True)
        data.rename(columns={"Bid": "CloseBid", "Ask": "CloseAsk"}, inplace=True)
        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Midpoint",
            "Weighted_midpoint",
        ]
        data = data[columns]
        data.index.name = "Datetime"
        data = enforce_bus_hours(resample(data, PRICING_CONFIG["INTRADAY_AGG"]))

    return data


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_eod_ohlc(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    print_url=False,
    rt=True,
    proxy=None,
    **kwargs,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    if not proxy:
        proxy = get_proxy_url()
    ## Scheduling to update to database
    start_date_str, end_date_str = deepcopy(start_date), deepcopy(end_date)
    sm_kwargs = dict(
        exp=exp,
        right=right,
        strike=strike,
        start=start_date_str,
        end=end_date_str,
        tick=symbol,
        type_="single",
        save_func="save_to_database",
    )
    schedule_kwargs(sm_kwargs) if get_should_schedule() else None

    ## Start processing
    pass_kwargs = {
        "symbol": symbol,
        "end_date": end_date,
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "print_url": print_url,
    }
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_eod_ohlc, _type="historical"
        )
    end_date_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp_int = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    start_date_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike_scaled = strike * 1000
    strike_int = int(strike_scaled)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {
        "end_date": end_date_int,
        "root": symbol,
        "use_csv": "true",
        "exp": exp_int,
        "right": right,
        "start_date": start_date_int,
        "strike": strike_int,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    try:
        if proxy:
            response = request_from_proxy(url, querystring, proxy)
            response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
            print(response_url) if print_url else None
            raise_thetadata_exception(response, querystring, proxy)
        else:
            response = requests.get(url, headers=headers, params=querystring)
            raise_thetadata_exception(response, querystring, proxy)
            print(response.url) if print_url else None
    except ThetaDataParseError as e:
        logger.error(
            f"ThetaDataParseError encountered for {generate_option_tick(symbol=symbol, exp=exp, right=right, strike=strike)}. Attempting quote to EOD patch. Expect longer response time."
        )
        logger.info(f"Error details: {e}")
        return quote_to_eod_patch(
            symbol,
            end_date_str,
            exp_og := pd.to_datetime(exp).strftime("%Y-%m-%d"),
            right,
            start_date_str,
            strike_og := strike,
            print_url=print_url,
        )

    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_eod_ohlc")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
        logger.error("Nothing returned at all")
        return
    else:
        data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
        data["weighted_midpoint"] = (
            (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["ask"])
        ) + (
            (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["bid"])
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)

        data["time"] = "16:00:00" if rt else ""
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime(f"%Y-%m-%d {PRICING_CONFIG['MARKET_CLOSE_TIME']}")
        )
        data["Date3"] = data.Date2
        data["datetime"] = pd.to_datetime(data.Date3)
        data.set_index("datetime", inplace=True)
        data.rename(columns={"Bid": "CloseBid", "Ask": "CloseAsk"}, inplace=True)
        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Midpoint",
            "Weighted_midpoint",
        ]
        data = data[columns]
        data.index.name = "Datetime"

    return data


## Migrate to new API
async def retrieve_eod_ohlc_async(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    print_url=False,
    rt=True,
    proxy=None,
    **kwargs,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    if not proxy:
        proxy = get_proxy_url()
    pass_kwargs = {
        "symbol": symbol,
        "end_date": end_date,
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
    }
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_eod_ohlc_async, _type="historical"
        )
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()

    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)

    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    print(response.url) if print_url else None
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_eod_ohlc")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
        logger.error("Nothing returned at all")
        return
    else:
        data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
        data["weighted_midpoint"] = (
            (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["ask"])
        ) + (
            (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["bid"])
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)

        data["time"] = "16:00:00" if rt else ""
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        data["Date3"] = data.Date2
        data["datetime"] = pd.to_datetime(data.Date3)
        data["datetime"].hour = 16
        data.set_index("datetime", inplace=True)
        data.rename(columns={"Bid": "CloseBid", "Ask": "CloseAsk"}, inplace=True)
        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Midpoint",
            "Weighted_midpoint",
        ]
        data = data[columns]
        data.index.name = "Datetime"

    return data


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_bulk_eod(
    symbol, exp, start_date, end_date, proxy=None, print_url=False, **kwargs
):
    if not proxy:
        proxy = get_proxy_url()

    pass_kwargs = {
        "symbol": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "print_url": print_url,
    }
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_bulk_eod, _type="historical"
        )

    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    url = "http://127.0.0.1:25510/v2/bulk_hist/option/eod"
    querystring = {
        "root": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
    }

    headers = {"Accept": "application/json"}
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        print(response.url) if print_url else None

    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for Bulk EOD {symbol}, {exp}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_bulk_eod")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
        logger.error("Nothing returned at all")
    else:
        data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
        data["weighted_midpoint"] = (
            (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["ask"])
        ) + (
            (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["bid"])
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)

        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime(f"%Y-%m-%d {PRICING_CONFIG['MARKET_CLOSE_TIME']}")
        )
        data["Date3"] = data.Date2
        data["datetime"] = pd.to_datetime(data.Date3)
        data.set_index("datetime", inplace=True)
        data.rename(columns={"Bid": "CloseBid", "Ask": "CloseAsk"}, inplace=True)
        data["Strike"] = data.Strike / 1000
        data["Expiration"] = pd.to_datetime(data.Expiration, format="%Y%m%d")
        columns = [
            "Root",
            "Strike",
            "Expiration",
            "Right",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Midpoint",
            "Weighted_midpoint",
        ]
        data = data[columns]
        data.index.name = "Datetime"

    return data


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_quote_rt(
    symbol,
    exp: str,
    right: str,
    strike: float,
    start_time: str = PRICING_CONFIG["MARKET_OPEN_TIME"],
    print_url=False,
    end_time=PRICING_CONFIG["MARKET_CLOSE_TIME"],
    ts=False,
    proxy=None,
    start_date: str = None,
    end_date: str = None,
    **kwargs,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    Returns realtime data
    """
    if not proxy:
        proxy = get_proxy_url()
    interval = "1h"
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    pass_kwargs = {
        "symbol": symbol,
        "end_date": end_date,
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
        "interval": interval,
        "print_url": print_url,
        "ts": ts,
    }

    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1

        return resolve_ticker_history(
            pass_kwargs, retrieve_quote_rt, _type="historical"
        )
    end_date = int(datetime.now().strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    ivl = identify_length(*extract_numeric_value(interval), rt=True) * 60000
    start_date = int(datetime.now().strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = (
        "http://127.0.0.1:25510/v2/snapshot/option/quote"
        if not ts
        else "http://127.0.0.1:25510/v2/hist/option/quote"
    )
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "ivl": ivl,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "start_time": start_time,
        "rth": False,
        "end_time": end_time,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
    end_timer = time.time()

    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    print(response.url) if print_url else None
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_quote_rt")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
    else:
        data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
        data["weighted_midpoint"] = (
            (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["ask"])
        ) + (
            (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["bid"])
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data["time"] = data["Ms_of_day"].apply(lambda c: convert_milliseconds(c))
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime("%Y-%m-%d")
        )  ##TODO: Change this to "%Y-%m-%d %H:%M:%S"
        data["Date3"] = data.Date2 + " " + data.time
        data["datetime"] = pd.to_datetime(data.Date3)
        data.set_index("datetime", inplace=True)
        data.drop(columns=["Date2", "Date3", "Ms_of_day", "time", "Date"], inplace=True)

    return data


## Break away function to helpers
def bootstrap_ohlc(data: pd.DataFrame, copy_column: str = "Midpoint"):
    """
    Format the OHLC data to have a consistent structure.
    Parameters
    ----------
    data : pd.DataFrame
        The OHLC data to format.
    copy_column : str, optional
        The column to copy values from, by default 'Midpoint'.

    Returns
    -------
    pd.DataFrame
        The formatted OHLC data.
    """

    new_cols = ["Open", "High", "Low", "Close"]
    copy_column = "Midpoint"
    for col in new_cols:
        if col not in data.columns:
            data[col.capitalize()] = data[copy_column]
    if "volume" in data.columns.str.lower():
        copycat = [col for col in data.columns if col.lower() == "volume"][0]
        data["Volume"] = data[copycat]
    else:
        data["Volume"] = np.nan

    return data


## Migrate to new API
# @backoff.on_exception(
#     backoff.expo,
#     (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
#     max_tries=5,
#     logger=logger,
# )


def list_dates(
    symbol,
    exp: str,
    right: str,
    strike: float,
    print_url=False,
    proxy=None,
    **kwargs,
):
    """
    List available dates for option quotes
    """
    if not proxy:
        proxy = get_proxy_url()
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    pass_kwargs = {
        "symbol": symbol,
        "exp": exp,
        "right": right,
        "strike": strike,
        "print_url": print_url,
    }
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1

        return resolve_ticker_history(pass_kwargs, list_dates, _type="historical")
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/list/dates/option/quote"
    querystring = {
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "right": right,
        "strike": strike,
    }
    headers = {"Accept": "application/json"}
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        print(response.url) if print_url else None
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )

    return pd.to_datetime(data["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d").tolist()

def retrieve_quote(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    start_time: str = None,
    print_url=False,
    end_time=PRICING_CONFIG["MARKET_CLOSE_TIME"],
    interval="30m",
    proxy=None,
    ohlc_format=True,
    **kwargs,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    """

    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"

    ##FIXME: ONE Time fix. We use 9:45 for start_time when bootstrapping ohlc to ensure there is data for open
    if start_time is None:
        if ohlc_format:
            start_time = PRICING_CONFIG["QUOTE_DATA_START_TIME"]
        else:
            start_time = PRICING_CONFIG["MARKET_OPEN_TIME"]

    pass_kwargs = {
        "symbol": symbol,
        "end_date": end_date,
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
        "interval": interval,
        "print_url": print_url,
    }
    if not proxy:
        proxy = get_proxy_url()

    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1

        return resolve_ticker_history(pass_kwargs, retrieve_quote, _type="historical")
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    ivl = (
        identify_length(
            *extract_numeric_value(PRICING_CONFIG.get("MIN_BAR_TIME_INTERVAL", "5m")),
            rt=True,
        )
        * 60_000
    )
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike = round(strike * 1000, 0)
    strike = int(strike)
    #  if not ohlc_format else PRICING_CONFIG['QUOTE_DATA_START_TIME']
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/hist/option/quote"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "ivl": ivl,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "start_time": start_time,
        "rth": True,
        "end_time": end_time,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        print(response.url) if print_url else None
    end_timer = time.time()

    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_quote function")
        logger.error(f"Following error for: {locals()}")
        logger.error(
            f"EOD OHLC mismatching dataframe size. Response: {data.columns[0]}"
        )
        logger.error("No data returned at all")
        logger.info(f"Kwargs: {locals()}")
        return
    data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
    data["weighted_midpoint"] = (
        (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1)) * (data["ask"])
    ) + (
        (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1)) * (data["bid"])
    )
    data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
    data["time"] = data["Ms_of_day"].apply(lambda c: convert_milliseconds(c))
    data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    data["Date3"] = data.Date2 + " " + data.time
    data["datetime"] = pd.to_datetime(data.Date3)
    data.set_index("datetime", inplace=True)
    data = data.iloc[data.index.indexer_between_time("9:30", "16:00")]
    data.drop(columns=["Date2", "Date3", "Ms_of_day"], inplace=True)
    data.rename(
        columns={
            "Bid": "Closebid",
            "Ask": "Closeask",
        },
        inplace=True,
    )

    if ohlc_format:
        data = bootstrap_ohlc(data, copy_column="Midpoint")

    return resample(data, interval=interval)
    # return data


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_openInterest(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    print_url=False,
    proxy=None,
    **kwargs,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    pass_kwargs = {
        "symbol": symbol,
        "end_date": end_date,
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "print_url": print_url,
    }
    if not proxy:
        proxy = get_proxy_url()
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_openInterest, _type="historical"
        )
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "rth": False,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    end_timer = time.time()

    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url=print_url)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)

    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)

    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    if not __isSuccesful(response.status_code):
        logger.error("")
        logger.error("Error in retrieve_openInterest")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"Error in retrieving data: {response.text}")
        logger.error("Nothing returned at all")
        logger.info(f"Kwargs: {locals()}")
        return

    try:
        print(response.url) if print_url else None
        data = (
            pd.read_csv(StringIO(response.text))
            if proxy is None
            else pd.read_csv(StringIO(response.json()["data"]))
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)

        data["time"] = data["Ms_of_day"].apply(convert_milliseconds)
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime(f"%Y-%m-%d {PRICING_CONFIG['MARKET_CLOSE_TIME']}")
        )
        data["Date3"] = data.Date2
        data["Datetime"] = pd.to_datetime(data.Date3)
        data.drop(columns=["Date2", "Date3", "Ms_of_day"], inplace=True)

        if data.Datetime.duplicated().any():
            duplicated_logger.info(
                f"Duplicated index found for {symbol}, {exp}, {right}, {strike}"
            )
            duplicated_logger.info(f"url: {response.url}")
            data = data[~data.Datetime.duplicated(keep="last")]  ## Last timestamp

    except Exception as e:
        logger.error("")
        logger.error(f"Error in retrieve_openInterest. Error: {e}")
        logger.error(f"Error in retrieving data: {response.text}")
        logger.error("Nothing returned at all")
        logger.info(f"Kwargs: {locals()}")
        raise e
    return data


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_bulk_open_interest(
    symbol, exp, start_date, end_date, proxy=None, print_url=False, **kwargs
):
    if not proxy:
        proxy = get_proxy_url()

    pass_kwargs = {
        "symbol": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "print_url": print_url,
    }
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_bulk_open_interest, _type="snapshot"
        )

    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d")) if exp else 0
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    url = "http://127.0.0.1:25510/v2/bulk_hist/option/open_interest"
    querystring = {
        "root": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
    }

    headers = {"Accept": "application/json"}
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for Bulk EOD {symbol}, {exp}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if not __isSuccesful(response.status_code):
        logger.error("")
        logger.error("Error in retrieve_openInterest")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"Error in retrieving data: {response.text}")
        logger.error("Nothing returned at all")
        logger.info(f"Kwargs: {locals()}")
        return
    try:
        print(response.url) if print_url else None
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data["time"] = data["Ms_of_day"].apply(convert_milliseconds)
        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        data["Date3"] = data.Date2
        data["Datetime"] = pd.to_datetime(data.Date3)
        data.drop(columns=["Date2", "Date3", "Ms_of_day"], inplace=True)
        data.Expiration = pd.to_datetime(data.Expiration, format="%Y%m%d")
        data.Strike = data.Strike / 1000
    except Exception as e:
        logger.error("")
        logger.error(f"Error in retrieve_openInterest. Error: {e}")
        logger.error(f"Error in retrieving data: {response.text}")
        logger.error("Nothing returned at all")
        logger.info(f"Kwargs: {locals()}")
        raise e
    return data


## Migrate to new API
async def retrieve_openInterest_async(
    symbol,
    end_date: str,
    exp: str,
    right: str,
    start_date: str,
    strike: float,
    print_url=False,
    proxy=None,
):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float
    ), f"strike should be type float, recieved {type(strike)}"
    if not proxy:
        proxy = get_proxy_url()
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d"))
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "right": right,
        "start_date": start_date,
        "strike": strike,
        "rth": False,
    }
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    end_timer = time.time()

    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)

    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp}, {right}, {strike}")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response.url}")

    if not __isSuccesful(response.status_code):
        logger.error("")
        logger.error("Error in retrieve_openInterest")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"Error in retrieving data: {response.text}")
        logger.error("Nothing returned at all")
        logger.info(f"Kwargs: {locals()}")
        return

    print(response.url) if print_url else None
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
    data["time"] = data["Ms_of_day"].apply(convert_milliseconds)
    data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    data["Date3"] = data.Date2
    data["Datetime"] = pd.to_datetime(data.Date3)
    data.drop(columns=["Date2", "Date3", "Ms_of_day"], inplace=True)
    return data


## Break away function to helpers
def resample(data, interval, custom_agg_columns=None, method="ffill", **kwargs):
    """
    Resamples to a specific interval size
    ps: ffills missing values

    parameters
    ----------
    data : pd.DataFrame or pd.Series
        Data to be resampled

    interval : str
        Interval to resample to. Can be a string like '1m', '5m', '1h', '1d', etc.
        or a number followed by a letter, e.g. '5m' or '1h'

    custom_agg_columns : dict, optional
        Custom aggregation dictionary to use for resampling. The default is None, which uses the default aggregation functions.
        The dictionary should have the format {'column_name': 'agg_func'} where 'agg_func' is a string representing the aggregation function to use.

    method : str, optional
        Method to use for resampling. Default is 'ffill'. Other options include 'mean', 'sum', 'max', 'min', etc.
        If a column is not passed in the `custom_agg_columns`, it will use the method provided if not in default custom columns

    kwargs : dict, optional

    Returns
    -------
    pd.DataFrame or pd.Series
        Resampled data.
    """

    ## Will allow for custom aggregation functions, using this resample function
    if isinstance(data.index, pd.MultiIndex):
        logger.info("Resampling a MultiIndex")
        if len(data.index.names) == 2:
            try:
                datetime_col_name = kwargs["datetime_col_name"]
            except KeyError:
                logger.critical(
                    "`datetime_col_name` not provided for multi index resample, setting to `Datetime`"
                )
                datetime_col_name = "Datetime"

            return _handle_multi_index_resample(
                data, datetime_col_name, interval, resample_col=custom_agg_columns
            )

        else:
            raise NotImplementedError(
                "Currently only supports multi index with 2 levels"
            )

    string, integer = extract_numeric_value(interval)
    TIMEFRAME_MAP = {
        "d": "B",
        "h": "BH",
        "m": "MIN",
        "M": "BME",
        "w": "W-FRI",
        "q": "BQE",
        "y": "BYS",
    }

    if custom_agg_columns is not None:
        columns = custom_agg_columns
    else:
        columns = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "bid_size": "last",
            "closebid": "last",
            "close_bid": "last",
            "close_ask": "last",
            "ask_size": "last",
            "closeask": "last",
            "midpoint": "last",
            "weighted_midpoint": "last",
        }

    assert (
        string in TIMEFRAME_MAP.keys()
    ), f"Available Timeframe Alias are {TIMEFRAME_MAP.keys()}, recieved '{string}'"

    ## Add EOD time if DateTimeIndex is EOD Series (Dunno why I did this, so taking it for now)
    ## If I remember, will write it here.
    if isinstance(data, pd.DataFrame):
        resampled = []

        for col in data.columns:
            if col.lower() in columns.keys():  ## Standard Column Resample
                resampled.append(
                    resample(data[col], interval, method=columns[col.lower()])
                )
            else:
                resampled.append(resample(data[col], interval, method=method))
        data = pd.concat(resampled, axis=1)
        data.columns = [col for col in data.columns]
        return enforce_bus_hours(data.fillna(0))

    elif isinstance(data, pd.Series):
        if string == "h":
            data = data.resample(
                f"{integer * 60}T", origin=PRICING_CONFIG["MARKET_OPEN_TIME"]
            ).__getattr__(method)()
        else:
            data = data.resample(f"{integer}{TIMEFRAME_MAP[string]}").__getattr__(
                method
            )()
        return enforce_bus_hours(data.fillna(0))


## Break away function to helpers
def _handle_multi_index_resample(
    data: pd.DataFrame,
    datetime_col_name: str,
    interval: str,
    resample_col: str,
) -> pd.DataFrame:
    """
    Handle the edge case where the data is a multi index and we need to resample it

    Args:
        data: The data to resample
        datetime_col_name: The name of the datetime column
        interval: The interval to resample to
        resample_col: The column to resample
    """
    assert (
        len(data.index.names) == 2
    ), f"Currently only supports multi index with 2 levels, got {len(data.index.names)}"

    ## Save the order of the MultiIndex
    idx_names = list(data.index.names)

    ## Provide a split by to resample off
    split_by = deepcopy(idx_names)
    split_by.remove(datetime_col_name)

    ## Break down the data into smaller chunks
    pack_data = dict(tuple(data.groupby(level=split_by)))
    resampled_data_list = []

    ## Resample Individual Data
    for k, v in pack_data.items():
        data = v.copy()
        data = data.reset_index().set_index(datetime_col_name)
        data = resample(data, interval, resample_col)
        data[split_by] = data[split_by].replace(0, k)
        data.set_index(split_by, inplace=True, append=True)
        resampled_data_list.append(data)

    resampled_data = pd.concat(resampled_data_list, axis=0)
    return resampled_data


## Break away function to helpers
def convert_milliseconds(ms):
    hours = ms // 3600000
    ms = ms % 3600000
    minutes = ms // 60000
    ms = ms % 60000
    seconds = ms // 1000
    _ = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}"


## Break away function to helpers
def convert_time_to_miliseconds(time):
    time_obj = pd.to_datetime(time)
    hour = time_obj.hour * 3_600_000
    minute = time_obj.minute * 60_000
    secs = time_obj.second * 1_000
    mili = time_obj.microsecond
    return hour + minute + secs + mili


## Migrate to new API
def retrieve_option_ohlc(
    symbol: str,
    exp: str,
    strike: float,
    right: str,
    start_date: str,
    end_date: str,
    proxy=None,
):
    """
    returns eod ohlc for all the days between start_date and end_date
    Interval is default to 3600000
    """
    if not proxy:
        proxy = get_proxy_url()
    strike = strike * 1000
    strike = int(strike) if strike.is_integer() else strike
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {
        "end_date": end_date,
        "root": symbol,
        "use_csv": "true",
        "exp": exp,
        "ivl": 3600000,
        "right": right,
        "start_date": start_date,
        "strike": strike,
    }
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    if __isSuccesful(response.status_code):
        if (len(data.columns)) > 1:
            data["mean_volume"] = data.groupby("date")["volume"].transform("mean")
            data = data.loc[
                data.groupby("date")["volume"].apply(
                    lambda x: (x - x.mean()).abs().idxmin()
                )
            ]
            data = data.drop_duplicates(subset="date", keep="last")
            data = data.drop(columns=["mean_volume"])
            data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
            return data
        else:
            return response
    else:
        return response


## Break away function to helpers
def __isSuccesful(status_code: int):
    return status_code >= 200 and status_code < 300


## Break away function to helpers
def is_theta_data_retrieval_successful(response):
    return not isinstance(response, str)


## Migrate to new API
@backoff.on_exception(
    backoff.expo,
    (ThetaDataOSLimit, ThetaDataDisconnected, ThetaDataServerRestart),
    max_tries=5,
    logger=logger,
)
def retrieve_chain_bulk(
    symbol,
    exp,
    start_date,
    end_date,
    end_time,
    right=None,
    proxy=None,
    print_url=False,
    oi=False,
    **kwargs,
) -> pd.DataFrame:
    pass_kwargs = {
        "symbol": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "end_time": end_time,
        "print_url": print_url,
        "oi": oi,
    }
    if not proxy:
        proxy = get_proxy_url()
    depth = pass_kwargs["depth"] = kwargs.get("depth", 0)
    end_date = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    exp = int(pd.to_datetime(exp).strftime("%Y%m%d")) if exp else 0
    start_date = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = f"http://127.0.0.1:25510/v2/bulk_at_time/option/{'quote' if not oi else 'open_interest'}"
    querystring = {
        "root": symbol,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "ivl": end_time,
        "use_csv": "true",
    }
    if right:
        querystring["right"] = right
    headers = {"Accept": "application/json"}
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs["depth"] += 1
        return resolve_ticker_history(
            pass_kwargs, retrieve_chain_bulk, _type="snapshot"
        )

    start_timer = time.time()

    # use proxy option
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}"
        print(response_url) if print_url else None
        raise_thetadata_exception(response, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
        raise_thetadata_exception(response, querystring, proxy)
        response_url = response.url
        print(response_url) if print_url else None
    data = (
        pd.read_csv(StringIO(response.text))
        if proxy is None
        else pd.read_csv(StringIO(response.json()["data"]))
    )
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info("")
        logger.info(f"Long response time for {symbol}, {exp} in Bulk Chain")
        logger.info(f"Response time: {end_timer - start_timer}")
        logger.info(f"Response URL: {response_url}")
    if len(data.columns) == 1:
        logger.error("")
        logger.error("Error in retrieve_bulk_chain")
        logger.error(f"Following error for: {locals()}")
        logger.error(f"ThetaData Response: {data.columns[0]}")
        logger.error("Nothing returned at all")

    else:
        data.columns = data.columns.str.lower()
        data["midpoint"] = data[["bid", "ask"]].sum(axis=1) / 2
        data["weighted_midpoint"] = (
            (data["ask_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["ask"])
        ) + (
            (data["bid_size"] / data[["bid_size", "ask_size"]].sum(axis=1))
            * (data["bid"])
        )
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)

        data["Date2"] = pd.to_datetime(data.Date.astype(str)).apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        data["Date3"] = data.Date2
        data["Expiration"] = pd.to_datetime(data.Expiration, format="%Y%m%d")
        data["Strike"] = data.Strike / 1000
        data["datetime"] = pd.to_datetime(data.Date3)
        data.set_index("datetime", inplace=True)
        data.rename(columns={"Bid": "CloseBid", "Ask": "CloseAsk"}, inplace=True)
        columns = [
            "Root",
            "Expiration",
            "Strike",
            "Right",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Date",
            "Midpoint",
            "Weighted_midpoint",
        ]
        data = data[columns]
    return data


# ## Break away function to helpers
# def ping_proxy():
#     try:
#         headers = {"Accept": "application/json", "Content-Type": "application/json"}
#         payload = {
#             "method": "GET",
#             "url": "http://127.0.0.1:25510/v2/hist/option/eod?end_date=20250619&root=AAPL&use_csv=true&exp=20241220&right=C&start_date=20240101&strike=220000",
#         }
#         proxy_url = os.environ["PROXY_URL"]
#         response = requests.post(proxy_url, headers=headers, json=payload)
#         return response.status_code == 200
#     except Exception as e:  # noqa
#         return False
