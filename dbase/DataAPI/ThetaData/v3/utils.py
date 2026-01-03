"""
ThetaData V3 API Utility Functions
===================================

This module provides V3-specific utility functions for data formatting, ticker
symbol change handling, parameter building, and multi-threaded data fetching.

Overview
--------
Key functionality:
- DataFrame formatting and normalization (_new_dataframe_formatting)
- Request parameter building (_build_params)
- Multi-threaded date range fetching (_multi_threaded_range_fetch)
- Ticker symbol change detection and handling
- Date range splitting for corporate actions

These utilities are used internally by v3/endpoints.py to provide consistent
data structures and automatic ticker change handling.

Key Functions
-------------
Data Formatting:
    _new_dataframe_formatting(df, interval, is_bulk, ignore_drop_conditional)
        Standardize DataFrame structure, column names, indexes, and resampling

Parameter Building:
    _build_params(symbol, start_date, end_date, date, exp, strike, right, interval, time_of_day)
        Build parameter dictionary for API requests

Range Fetching:
    _multi_threaded_range_fetch(symbol, start_date, end_date, url, **kwargs)
        Fetch data across date range using multi-threading (when endpoint doesn't support ranges)

Ticker Change Handling:
    _get_symbol_for_date(symbol, date)
        Get appropriate symbol for specific date (handles ticker changes)

    _split_date_range_by_ticker_change(symbol, start_date, end_date)
        Split date range at ticker change boundary

    _with_ticker_change_handling(func, symbol, **kwargs)
        Generic wrapper that adds ticker change handling to any function

Data Formatting Details
-----------------------
_new_dataframe_formatting performs:

1. Column Normalization:
   - Lowercase all column names
   - Rename 'timestamp' → 'datetime'
   - Convert datetime to proper pandas Timestamp

2. Column Cleanup:
   - Drop unnecessary columns (bid_exchange, bid_condition, etc.)
   - Drop option identifier columns for single-contract queries
   - Keep identifier columns for bulk queries

3. Data Processing:
   - Calculate midpoint from bid/ask
   - Calculate weighted_midpoint from bid/ask with sizes
   - Format strike as float with 3 decimals
   - Format right as single letter ('C' or 'P')
   - Format expiration as datetime

4. Resampling:
   - Resample intraday data to requested interval
   - Skip resampling for EOD data
   - Skip resampling for bulk data (too complex)

5. Index Setting:
   - Set datetime as index
   - Add Strike/Right/Expiration to index for bulk data

6. Legacy Formatting (if SETTINGS.use_old_formatting=True):
   - Capitalize column names
   - Rename Bid → CloseBid, Ask → CloseAsk
   - Add EOD timestamp adjustment

Usage Examples
--------------
Format API response:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.utils import _new_dataframe_formatting
    import pandas as pd
    from io import StringIO

    # Raw API response
    csv_text = _fetch_data(url, params)
    df = pd.read_csv(StringIO(csv_text))

    # Format to standard structure
    df = _new_dataframe_formatting(df, interval='5m', is_bulk=False)

Build request parameters:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.utils import _build_params

    params = _build_params(
        symbol='AAPL',
        start_date='2024-01-01',
        end_date='2024-12-31',
        exp='2024-12-20',
        strike=180.0,
        right='C',
        interval='5m'
    )
    # Returns: {
    #     'symbol': 'AAPL',
    #     'start_date': '20240101',
    #     'end_date': '20241231',
    #     'expiration': '20241220',
    #     'strike': '180.00',
    #     'right': 'C',
    #     'interval': '5m'
    # }

Multi-threaded range fetch:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.utils import _multi_threaded_range_fetch
    from dbase.DataAPI.ThetaData.v3.vars import HISTORICAL_QUOTE

    # For endpoints that don't support date ranges
    df = _multi_threaded_range_fetch(
        symbol='AAPL',
        start_date='2024-12-01',
        end_date='2024-12-15',
        url=HISTORICAL_QUOTE,
        exp='2024-12-20',
        strike=180.0,
        right='C',
        interval='5m'
    )

Handle ticker changes:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.utils import _split_date_range_by_ticker_change

    # Split date range for META (formerly FB)
    segments = _split_date_range_by_ticker_change(
        symbol='META',
        start_date='2022-05-01',
        end_date='2022-07-31'
    )
    # Returns: [
    #     ('FB', '2022-05-01', '2022-06-08'),
    #     ('META', '2022-06-09', '2022-07-31')
    # ]

Wrap function with ticker handling:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.utils import _with_ticker_change_handling

    def _raw_retrieve_data(symbol, start_date, end_date, **kwargs):
        # Direct API call without ticker handling
        ...

    def retrieve_data(symbol, start_date, end_date, **kwargs):
        # Automatically handles ticker changes
        return _with_ticker_change_handling(
            _raw_retrieve_data,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

Ticker Change Logic
-------------------
The ticker change handling system:

1. Detects if symbol has historical change (e.g., FB → META)
2. Determines query type:
   - Historical: has start_date + end_date
   - At-time: has single date parameter
   - Snapshot: no date parameters

3. For historical queries:
   - Splits date range at change boundary
   - Queries old symbol before change
   - Queries new symbol after change
   - Merges results and removes duplicates

4. For at-time queries:
   - Uses appropriate symbol for that date

5. For snapshot queries:
   - Uses current symbol

Multi-Threading Details
-----------------------
_multi_threaded_range_fetch is used when endpoints don't support native date ranges:

1. Generate business day range (excluding holidays)
2. Build separate parameter sets for each date
3. Use default intraday interval (from PRICING_CONFIG)
4. Execute requests in parallel using runThreads()
5. Concatenate results
6. Return combined DataFrame

This is necessary for endpoints like /option/at_time/quote that only accept
single dates, not ranges.

Parameter Building
------------------
_build_params converts Python parameters to API format:

Symbol/Strike/Right:
    - symbol: passed as-is
    - strike: formatted as "%.2f" or "*" for all
    - right: passed as-is or "both" for all

Dates:
    - Converts 'YYYY-MM-DD' to 'YYYYMMDD' format
    - Validates start_date/end_date both present or both absent

Expiration:
    - Converts to 'YYYYMMDD' or "*" for all expirations

Interval:
    - Validates against VALID_INTERVALS
    - Passed as-is (e.g., '5m', '1h')

Time of Day:
    - Converts to 'HH:MM:SS.mmm' format

Performance Considerations
--------------------------
- Multi-threading significantly speeds up date-range queries
- Formatting adds minimal overhead (<10ms per query)
- Ticker change detection is instant (dictionary lookup)
- Resampling large datasets can be slow (use larger intervals)

Notes
-----
- All functions are designed for internal use by endpoints.py
- Ticker change data comes from TICK_CHANGE_ALIAS mapping
- Holiday exclusion uses HOLIDAY_SET from trade module
- Resampling respects business hours via enforce_bus_hours()

See Also
--------
- endpoints.py : Uses these utilities for all API calls
- vars.py : Configuration and constants
- ../utils.py : Shared utilities across V2 and V3
- trade.assets.helpers.utils : TICK_CHANGE_ALIAS mapping
"""

import pandas as pd
from dbase.utils import add_eod_timestamp
from dbase.DataAPI.ThetaExceptions import MissingColumnError
from dbase.DataAPI.ThetaData.v3.vars import (
    SETTINGS,
    ONE_DAY_MILLISECONDS,
    MINIMUM_MILLISECONDS,
    VALID_INTERVALS,
    LOOP_WARN_MSG,
)

from trade import PRICING_CONFIG, HOLIDAY_SET
from trade.helpers.threads import runThreads
from io import StringIO
from ..utils import _fetch_data
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData.utils import convert_string_interval_to_miliseconds, resample, normalize_date_format
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS
from typing import Callable, Any
from trade.helpers.decorators import timeit

logger = setup_logger("dbase.DataAPI.ThetaData.v3.utils")


##NOTE: Interested in seeing additional overhead
@timeit
def _new_dataframe_formatting(
    df: pd.DataFrame, interval: str, is_bulk: bool = False, ignore_drop_conditional: bool = False
) -> pd.DataFrame:
    """
    Formats the DataFrame to a new standard structure.
    """

    ## Must have timestamp column
    if "timestamp" not in df.columns:
        raise MissingColumnError(
            "Dataframe is missing required 'timestamp' column. Reach out to chidi if you see this error."
        )

    df = df.copy()
    df.columns = df.columns.str.lower()
    df.rename(columns={"timestamp": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    drop_candidates = [
        "last_trade",
        "bid_exchange",
        "bid_condition",
        "ask_exchange",
        "ask_condition",
    ]

    interval_ms = convert_string_interval_to_miliseconds(interval)
    is_intraday = interval_ms < ONE_DAY_MILLISECONDS
    if interval_ms < MINIMUM_MILLISECONDS:
        raise ValueError(f"Interval {interval} is too small. Minimum allowed is {PRICING_CONFIG['INTRADAY_AGG']}")

    conditional_drop_candidates = [
        "right",
        "root",
        "symbol",
        "strike",
        "expiration",
    ]

    if not is_bulk and not ignore_drop_conditional:
        for col in conditional_drop_candidates:
            if col in df.columns:
                drop_candidates.append(col)

    ## Drop unnecessary columns
    for col in drop_candidates:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    ## Right column formatting
    if "right" in df.columns:
        df["right"] = df["right"].astype(str)
        df["right"] = df["right"].apply(lambda x: x.upper()[0])

    ## Strike Formatting. float type with 3 decimal places
    if "strike" in df.columns:
        df["strike"] = df["strike"].astype(float).round(3)

    ## Expiration Formatting
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])

    ## Rename symbol column to root
    if "symbol" in df.columns:
        df.rename(columns={"symbol": "root"}, inplace=True)

    ## If bid & ask columns exist, calculate mid price
    if "bid" in df.columns and "ask" in df.columns:
        df["midpoint"] = (df["bid"] + df["ask"]) / 2

        ## If bid_siz & ask_siz columns exist, calculate weighted mid price
        if "bid_size" in df.columns and "ask_size" in df.columns:
            total_size = df["bid_size"] + df["ask_size"]
            df["weighted_midpoint"] = ((df["bid"] * df["bid_size"]) + (df["ask"] * df["ask_size"])) / total_size

    ## Bulk/Single formatting
    ## First index setting for resampling purposes.
    def set_index_columns():
        index_cols = ["datetime"]
        # if is_bulk:
        #     index_cols.extend(["strike", "right", "expiration"])
        df.set_index(index_cols, inplace=True)

    set_index_columns()

    ## Resample
    ## Only resample on None bulk and intraday
    if not is_bulk and is_intraday:
        df = resample(df, interval=interval)

    ## Set timestamp as index
    ## Reset index to datetime
    df.reset_index(inplace=True)
    _format = SETTINGS.intra_format if is_intraday else SETTINGS.eod_format
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime(_format)
    df["datetime"] = pd.to_datetime(df["datetime"])
    set_index_columns()

    ## OLD FORMATTING SECTION
    if SETTINGS.use_old_formatting:
        ## Col formatting
        df.columns = df.columns.str.capitalize()

        ## Bid -> CloseBid, Ask -> CloseAsk
        if "Bid" in df.columns:
            df.rename(columns={"Bid": "CloseBid"}, inplace=True)
        if "Ask" in df.columns:
            df.rename(columns={"Ask": "CloseAsk"}, inplace=True)

        ## Add EOD Timestamp
        if not is_intraday:
            df.index = add_eod_timestamp(df.index)

    return df


def _build_params(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    date: str = None,
    exp: str = None,
    strike: float = None,
    right: str = None,
    interval: str = None,
    time_of_day: str = None,
    **kwargs,
) -> dict:
    """Helper to build parameters dictionary for requests."""
    params = {"symbol": symbol}
    if start_date or end_date:
        assert end_date is not None, "end_date must be provided if start_date is provided"
        assert start_date is not None, "start_date must be provided if end_date is provided"
        params["start_date"] = normalize_date_format(start_date, _type=2)
        params["end_date"] = normalize_date_format(end_date, _type=2)
    if exp:
        params["expiration"] = normalize_date_format(exp, _type=2)
    else:
        params["expiration"] = "*"
    if strike is not None:
        params["strike"] = f"{strike:.2f}"
    else:
        params["strike"] = "*"
    if right:
        params["right"] = right
    else:
        params["right"] = "both"

    if interval:
        assert interval in VALID_INTERVALS, f"Invalid interval. Recieved {interval}, expected {VALID_INTERVALS}"
        params["interval"] = interval

    if date:
        params["date"] = normalize_date_format(date, _type=2)

    if time_of_day:
        params["time_of_day"] = pd.to_datetime(time_of_day).strftime("%H:%M:%S.%f")[:-3]
    return params


def _multi_threaded_range_fetch(
    symbol: str,
    start_date: str,
    end_date: str,
    url: str,
    print_url: bool = False,
    omit_interval: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch data over a date range using multithreading.
    Some endpoint do not support range dates, so we loop through each date in the range
    Args:
        symbol (str): The option symbol.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        url (str): The API endpoint URL.
        print_url (bool): Whether to print the request URL for the first request.
        **kwargs: Additional parameters for the request.
    Returns:
        pd.DataFrame: The concatenated DataFrame containing historical quote data.
    """
    logger.warning(LOOP_WARN_MSG + f" Endpoint: {url}")

    ## Generate business day date range excluding holidays & weekends
    dt_range = pd.date_range(start=start_date, end=end_date, freq="1b").strftime("%Y-%m-%d").tolist()
    dt_range = [dt for dt in dt_range if dt not in HOLIDAY_SET]

    ## For any endpoint that requires interval, set default. Down the pipeline we resample to requested interval
    ## ThetaData V3 currently doesnt support 1d interval, so we set to default intraday.
    default_interval = PRICING_CONFIG["INTRADAY_AGG"]

    ## Remove interval from kwargs if exists
    if "interval" in kwargs:
        kwargs.pop("interval")

    ## Build params for each date
    params_set = [
        _build_params(
            symbol=symbol,
            date=dt,
            interval=default_interval if not omit_interval else None,
            **kwargs,
        )
        for dt in dt_range
    ]

    ## Prepare inputs for threading
    inputs = [[url] * len(dt_range), params_set, [print_url] + [False] * (len(dt_range) - 1)]

    ## Thread fetch function
    def _thread_fetch(url, params, print_url):
        try:
            txt = _fetch_data(url, params, print_url)
            return pd.read_csv(StringIO(txt))
        except Exception as e:
            logger.error(f"Error fetching data for params {params}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    return pd.concat(runThreads(_thread_fetch, inputs))


def _get_symbol_for_date(symbol: str, date: str) -> str:
    """
    Get the appropriate symbol to use for a specific date.

    Parameters
    ----------
    symbol : str
        Current ticker symbol
    date : str
        Query date (YYYY-MM-DD)

    Returns
    -------
    str
        Symbol to use for that date (could be old or new symbol)
    """
    # Check if symbol has a ticker change
    if symbol not in TICK_CHANGE_ALIAS:
        return symbol

    old_symbol, new_symbol, change_date = TICK_CHANGE_ALIAS[symbol]
    date_dt = pd.to_datetime(date)
    change_dt = pd.to_datetime(change_date)

    # If date is before change, use old symbol
    if date_dt < change_dt:
        return old_symbol

    # Otherwise use current symbol
    return symbol


def _split_date_range_by_ticker_change(symbol: str, start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """
    Split a date range into segments based on ticker symbol changes.

    Parameters
    ----------
    symbol : str
        Current ticker symbol
    start_date : str
        Query start date (YYYY-MM-DD)
    end_date : str
        Query end date (YYYY-MM-DD)

    Returns
    -------
    list[tuple[str, str, str]]
        List of (symbol, start_date, end_date) tuples for each segment

    Example
    -------
    >>> _split_date_range_by_ticker_change("META", "2022-05-01", "2022-07-31")
    [("FB", "2022-05-01", "2022-06-08"), ("META", "2022-06-09", "2022-07-31")]
    """
    # Check if symbol has a ticker change
    if symbol not in TICK_CHANGE_ALIAS:
        # No ticker change, return single segment
        return [(symbol, start_date, end_date)]

    old_symbol, new_symbol, change_date = TICK_CHANGE_ALIAS[symbol]

    # Convert dates to datetime for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    change_dt = pd.to_datetime(change_date)

    # Determine which segments to query
    segments = []

    # If date range ends before ticker change, use old symbol only
    if end_dt < change_dt:
        segments.append((old_symbol, start_date, end_date))

    # If date range starts after ticker change, use new symbol only
    elif start_dt >= change_dt:
        segments.append((symbol, start_date, end_date))

    # Date range spans the ticker change - need both symbols
    else:
        # Old symbol: from start_date to day before change
        day_before_change = (change_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        segments.append((old_symbol, start_date, day_before_change))

        # New symbol: from change date to end_date
        segments.append((symbol, change_date, end_date))

    return segments


def _with_ticker_change_handling(func: Callable, symbol: str, **kwargs: Any) -> pd.DataFrame:
    """
    Generic wrapper that handles ticker symbol changes for ANY data retrieval function.

    Automatically detects whether the function is:
    - Historical (has start_date + end_date): Splits date range and merges results
    - At-time (has single date/at_date): Uses appropriate symbol for that date
    - Snapshot (no date params): Uses current symbol as-is

    This function is used internally by all retrieval functions.

    Parameters
    ----------
    func : Callable
        The data retrieval function to wrap (e.g., _raw_retrieve_eod_ohlc, etc.)
    symbol : str
        Current ticker symbol
    **kwargs : Any
        All other parameters to pass to the function

    Returns
    -------
    pd.DataFrame
        Combined data with ticker changes handled automatically
    """
    # Detect query type based on kwargs
    has_start_end = "start_date" in kwargs and "end_date" in kwargs
    has_date = "date" in kwargs
    has_at_date = "at_date" in kwargs

    # Case 1: Historical query with date range
    if has_start_end:
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]

        # Split date range by ticker changes
        segments = _split_date_range_by_ticker_change(symbol, start_date, end_date)

        # If only one segment, just call function directly
        if len(segments) == 1:
            return func(symbol=segments[0][0], **kwargs)

        # Multiple segments: fetch and merge
        dataframes = []
        for segment_symbol, seg_start, seg_end in segments:
            logger.info(f"Fetching {segment_symbol} data: {seg_start} to {seg_end}")

            # Update kwargs with segment-specific dates
            segment_kwargs = kwargs.copy()
            segment_kwargs["start_date"] = seg_start
            segment_kwargs["end_date"] = seg_end

            try:
                df = func(symbol=segment_symbol, **segment_kwargs)

                # Normalize root column to current symbol
                if "root" in df.columns:
                    df["root"] = symbol

                dataframes.append(df)

            except Exception as e:
                logger.warning(f"Failed to fetch {segment_symbol} data: {e}")
                continue

        # Merge results
        if not dataframes:
            raise ValueError(f"No data retrieved for {symbol}")

        if len(dataframes) == 1:
            return dataframes[0]

        # Concatenate and sort
        combined = pd.concat(dataframes, axis=0)
        combined = combined.sort_index()

        # Remove duplicates
        if combined.index.duplicated().any():
            logger.warning(f"Removing {combined.index.duplicated().sum()} duplicate timestamps")
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined

    # Case 2: At-time query (single date)
    elif has_date:
        date = kwargs["date"]
        correct_symbol = _get_symbol_for_date(symbol, date)
        logger.info(f"Using symbol {correct_symbol} for date {date}")
        return func(symbol=correct_symbol, **kwargs)

    # Case 3: At-time query (alternative date param)
    elif has_at_date:
        at_date = kwargs["at_date"]
        correct_symbol = _get_symbol_for_date(symbol, at_date)
        logger.info(f"Using symbol {correct_symbol} for date {at_date}")
        return func(symbol=correct_symbol, **kwargs)

    # Case 4: Snapshot query (no date params) - use current symbol
    else:
        logger.info(f"Snapshot query - using current symbol {symbol}")
        return func(symbol=symbol, **kwargs)
