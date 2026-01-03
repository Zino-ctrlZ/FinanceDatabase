"""
ThetaData API Utility Functions Module
=======================================

This module provides shared utility functions used across both V2 and V3 APIs
for data processing, formatting, validation, and fetching.

Overview
--------
This module contains functions for:
- Data resampling and aggregation to different timeframes
- OHLC data formatting and bootstrapping
- Time/interval conversions and calculations
- Date format normalization
- Option ticker parameter parsing
- API request handling with proxy support
- Data validation helpers

These utilities are version-agnostic and used by both v2.py and v3/ modules.

Key Functions
-------------
Data Resampling:
    - resample() : Resample data to different intervals
    - _handle_multi_index_resample() : Resample multi-indexed data

OHLC Formatting:
    - bootstrap_ohlc() : Fill missing OHLC columns

Time/Interval Functions:
    - identify_length() : Calculate timeframe length in minutes
    - convert_time_to_miliseconds() : Convert time to milliseconds
    - convert_milliseconds() : Convert milliseconds to HH:MM:SS
    - convert_string_interval_to_miliseconds() : Parse interval strings
    - extract_numeric_value() : Extract number and unit from intervals

Date Formatting:
    - normalize_date_format() : Normalize dates to YYYY-MM-DD or YYYYMMDD

Parameter Parsing:
    - _handle_opttick_param() : Parse option ticker or individual params
    - _all_is_provided() : Check if all kwargs are non-None

API Requests:
    - _fetch_data() : Main fetch function with proxy support
    - request_from_proxy() : Send request through proxy server
    - is_theta_data_retrieval_successful() : Validate response

Usage Examples
--------------
Resampling data:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import resample
    import pandas as pd

    # Resample 1-minute data to 5-minute bars
    df_5m = resample(df_1m, interval='5m')

    # Resample to hourly
    df_1h = resample(df_1m, interval='1h')

    # Resample to daily (business days)
    df_1d = resample(df_1m, interval='1d')

Custom aggregation:

.. code-block:: python

    # Custom aggregation functions per column
    custom_agg = {
        'volume': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'custom_col': 'mean'
    }

    df_resampled = resample(df, interval='15m', custom_agg_columns=custom_agg)

Bootstrap OHLC data:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import bootstrap_ohlc

    # If data only has Midpoint column, copy to OHLC columns
    df = bootstrap_ohlc(df, copy_column='Midpoint')

Parse interval strings:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import extract_numeric_value, convert_string_interval_to_miliseconds

    # Extract components
    unit, num = extract_numeric_value('5m')  # Returns ('m', 5)

    # Convert to milliseconds
    ms = convert_string_interval_to_miliseconds('5m')  # Returns 300000

Handle option tickers:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import _handle_opttick_param

    # Parse option ticker
    strike, right, symbol, exp = _handle_opttick_param(opttick='AAPL20241220C180')
    # Returns: (180.0, 'C', 'AAPL', '2024-12-20')

    # Or use individual parameters
    strike, right, symbol, exp = _handle_opttick_param(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0
    )

Normalize dates:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import normalize_date_format

    # Convert to YYYY-MM-DD
    date1 = normalize_date_format('20241220', _type=1)  # '2024-12-20'

    # Convert to YYYYMMDD
    date2 = normalize_date_format('2024-12-20', _type=2)  # '20241220'

Fetch data with proxy:

.. code-block:: python

    from dbase.DataAPI.ThetaData.utils import _fetch_data

    # Fetch data (automatically uses proxy if configured)
    text = _fetch_data(
        theta_url='http://localhost:25503/v3/option/history/eod',
        params={'symbol': 'AAPL', 'start_date': '20240101'},
        print_url=True
    )

Resampling Details
------------------
The resample() function supports various interval formats:
    - Minutes: '5m', '15m', '30m'
    - Hours: '1h', '2h'
    - Days: '1d' (business days)
    - Weeks: '1w' (ending Friday)
    - Months: '1M' (business month end)
    - Quarters: '1q' (business quarter end)
    - Years: '1y' (business year start)

Default aggregation rules:
    - open: first value
    - high: maximum value
    - low: minimum value
    - close: last value
    - volume: sum
    - bid/ask columns: last value
    - midpoint/weighted_midpoint: last value

Multi-Index Support:
For DataFrames with multi-level indexes (e.g., [Datetime, Strike]),
resample automatically handles grouping and preserves index structure.

Time Conversion Constants
--------------------------
INTERVAL_MAP_IN_SECONDS:
    - 's': 1 second
    - 'm': 60 seconds
    - 'h': 3600 seconds (1 hour)
    - 'd': 86400 seconds (1 day)
    - 'b': 86400 seconds (business day)
    - 'M': 2592000 seconds (30 days)
    - 'q': 7776000 seconds (90 days)
    - 'y': 31536000 seconds (365 days)

Performance Considerations
--------------------------
- Resampling large datasets can be memory-intensive
- Use custom aggregation to avoid processing unnecessary columns
- For very large datasets, consider chunking before resampling
- Multi-index resampling is slower due to grouping operations

Notes
-----
- All time-related functions assume US Eastern Time (market hours)
- Business day calculations respect HOLIDAY_SET from trade module
- Interval parsing is case-insensitive
- Missing OHLC columns are filled with copy_column (default: 'Midpoint')

See Also
--------
- v2.py : V2 API implementation (uses these utilities)
- v3/endpoints.py : V3 API implementation (uses these utilities)
- v3/utils.py : V3-specific utilities
- proxy.py : Proxy configuration
- log.py : Request latency logging
"""

import json
import requests
from dbase.DataAPI.ThetaExceptions import raise_thetadata_exception
from trade.helpers.helper import parse_option_tick
from trade.helpers.Logging import setup_logger
from typing import Tuple
import re
import pandas as pd
from copy import deepcopy
from trade import PRICING_CONFIG
from dbase.DataAPI.ThetaData.proxy import get_proxy_url
from dbase.utils import enforce_bus_hours
from .log import _submit_log

logger = setup_logger("dbase.DataAPI.ThetaData.utils")


def is_theta_data_retrieval_successful(response):
    return not isinstance(response, str)


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


def convert_milliseconds(ms):
    hours = ms // 3600000
    ms = ms % 3600000
    minutes = ms // 60000
    ms = ms % 60000
    seconds = ms // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_time_to_miliseconds(time):
    time_obj = pd.to_datetime(time)
    hour = time_obj.hour * 3_600_000
    minute = time_obj.minute * 60_000
    secs = time_obj.second * 1_000
    mili = time_obj.microsecond
    return hour + minute + secs + mili


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

    new_cols = ["Open", "High", "Low", "Close", "Volume"]
    copy_column = "Midpoint"
    for col in new_cols:
        if col not in data.columns:
            data[col.capitalize()] = data[copy_column]

    return data


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
                logger.critical("`datetime_col_name` not provided for multi index resample, setting to `Datetime`")
                datetime_col_name = "Datetime"

            return _handle_multi_index_resample(data, datetime_col_name, interval, resample_col=custom_agg_columns)

        else:
            raise NotImplementedError("Currently only supports multi index with 2 levels")

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

    assert string in TIMEFRAME_MAP.keys(), f"Available Timeframe Alias are {TIMEFRAME_MAP.keys()}, recieved '{string}'"

    ## Add EOD time if DateTimeIndex is EOD Series (Dunno why I did this, so taking it for now)
    ## If I remember, will write it here.
    if isinstance(data, pd.DataFrame):
        resampled = []

        for col in data.columns:
            if col.lower() in columns.keys():  ## Standard Column Resample
                resampled.append(resample(data[col], interval, method=columns[col.lower()]))
            else:
                resampled.append(resample(data[col], interval, method=method))
        data = pd.concat(resampled, axis=1)
        data.columns = [col for col in data.columns]
        return enforce_bus_hours(data.fillna(0))

    elif isinstance(data, pd.Series):
        if string == "h":
            data = data.resample(f"{integer * 60}T", origin=PRICING_CONFIG["MARKET_OPEN_TIME"]).__getattr__(method)()
        else:
            data = data.resample(f"{integer}{TIMEFRAME_MAP[string]}").__getattr__(method)()
        return enforce_bus_hours(data.fillna(0))


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
    assert len(data.index.names) == 2, f"Currently only supports multi index with 2 levels, got {len(data.index.names)}"

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


def normalize_date_format(date_str: str, _type: int = 1) -> str:
    """Normalize date string to 'YYYY-MM-DD' format."""
    try:
        dt = pd.to_datetime(date_str)
        if _type == 1:
            return dt.strftime("%Y-%m-%d")
        elif _type == 2:
            return dt.strftime("%Y%m%d")
        else:
            raise ValueError(f"Unsupported _type value: {_type}")
    except Exception as e:
        raise ValueError(f"Invalid date format: {date_str}") from e


def extract_numeric_value(timeframe_str):
    match = re.findall(r"(\d+)([a-zA-Z]+)", timeframe_str)
    integers = [int(num) for num, _ in match][0]
    strings = [str(letter) for _, letter in match][0]
    return strings, integers


INTERVAL_MAP_IN_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "b": 86400,  # business day
    "M": 2592000,  # 30 days
    "Q": 7776000,  # 90 days
    "q": 7776000,  # 90 days
    "y": 31536000,  # 365 days
}


def _all_is_provided(**kwargs) -> bool:
    """
    Check if all provided keyword arguments are not None.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to check.`
    Returns
    -------
    bool
        True if all arguments are provided, False otherwise.
    """
    for _, value in kwargs.items():
        if value is None:
            return False
    return True


def convert_string_interval_to_miliseconds(timeframe_str: str) -> int:
    """Convert a string interval like '5m', '1h', '1d' to milliseconds.

    Args:
        timeframe_str (str): The timeframe string to convert.
    Returns:
        int: The equivalent time in milliseconds.
    """
    unit, num = extract_numeric_value(timeframe_str)
    length_in_ms = INTERVAL_MAP_IN_SECONDS.get(unit.lower())
    if length_in_ms is None:
        raise ValueError(f"Unsupported time unit: {unit}")

    return num * length_in_ms * 1000


def _handle_opttick_param(
    strike: float = None, right: str = None, symbol: str = None, exp: str = None, opttick: str = None
) -> Tuple[float, str, str, str]:
    """Helper function to parse and validate option tick parameters.

    Args:
        strike (float, optional): The strike price of the option.
        right (str, optional): The right of the option ('C' for call, 'P' for put).
        symbol (str, optional): The underlying symbol of the option.
        exp (str, optional): The expiration date of the option in 'YYYY-MM-DD' format.
        opttick (str, optional): The option ticker string.

    Returns:
        Tuple[float, str, str, str]: A tuple containing strike, right, symbol, and exp.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    if not any([symbol, opttick]):
        raise ValueError("Either 'symbol' or 'opttick' must be provided.")
    if opttick:
        parsed_symbol, parsed_right, parsed_exp, parsed_strike = parse_option_tick(opttick).values()
        return parsed_strike, parsed_right, parsed_symbol, parsed_exp
    else:
        return strike, right, symbol, exp


def request_from_proxy(thetaUrl, queryparam, instanceUrl, print_url=False):
    request_string = f"{thetaUrl}?{'&'.join([f'{key}={value}' for key, value in queryparam.items()])}"

    payload = json.dumps(
        {
            "url": request_string,
            "method": "GET",
        }
    )
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", instanceUrl, headers=headers, data=payload)
    return response


def _fetch_data(theta_url: str, params: dict, print_url: bool = False) -> str:
    """
    Fetch data from ThetaData API, using proxy if available.
    Args:
        theta_url (str): The ThetaData API endpoint URL.
        params (dict): Query parameters for the API request.
        print_url (bool): Whether to print the request URL.
    Returns:
        str: The response data as a string.
    """

    instance_url = get_proxy_url()
    if instance_url:
        response = request_from_proxy(theta_url, params, instance_url)
        text = response.json()["data"]
        url = response.json().get("url", "N/A")
    else:
        response = requests.get(theta_url, params=params)
        text = response.text
        url = response.url

    ## Format text for consistency
    text = text.replace("created", "timestamp")

    ## Log the request latency
    _submit_log(url, response)

    ## Print URL if required
    if print_url:
        print(f"Request URL: {url}")
    raise_thetadata_exception(response=response, params=params, proxy=instance_url)
    return text
