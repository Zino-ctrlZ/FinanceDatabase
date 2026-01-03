"""
ThetaData V3 API Endpoint Functions
====================================

This module implements all data retrieval functions for the ThetaData V3 REST API.
Each endpoint function retrieves specific types of options market data with automatic
ticker change handling and consistent data formatting.

Overview
--------
This module provides direct access to ThetaData V3 API endpoints for:
- Realtime and historical quote data (bid/ask/midpoint)
- EOD and intraday OHLC data
- Open interest data (historical and bulk)
- Option chain snapshots at specific times
- Contract and date listings

All functions follow a consistent pattern:
1. _raw_* functions: Direct API calls without ticker change handling
2. _inner_* functions: Add data formatting
3. Public _* functions: Add ticker change handling (exported to __init__.py)

Ticker Change Handling
----------------------
All public functions (_retrieve_*, _list_*) automatically handle ticker symbol
changes due to corporate actions (e.g., FB → META on 2022-06-09):

For historical queries (start_date + end_date):
    - Automatically splits date range at ticker change date
    - Queries old symbol before change, new symbol after
    - Combines results seamlessly

For at-time queries (single date):
    - Uses appropriate symbol for that specific date

For snapshot queries (no date):
    - Uses current symbol

This is handled transparently via _with_ticker_change_handling() wrapper.

Function Categories
-------------------
Quote Data:
    _retrieve_quote_rt(symbol, exp, right, strike, **kwargs)
        Realtime quote snapshot for a single contract

    _retrieve_bulk_quote_rt(symbol, exp, **kwargs)
        Bulk realtime quotes for entire expiration

    _retrieve_quote(symbol, exp, right, strike, start_date, end_date, interval, **kwargs)
        Historical intraday quotes

OHLC Data:
    _retrieve_ohlc(symbol, exp, right, strike, start_date, end_date, interval, **kwargs)
        Historical intraday OHLC data

    _retrieve_eod_ohlc(symbol, exp, right, strike, start_date, end_date, **kwargs)
        End-of-day OHLC data

    _retrieve_bulk_eod(symbol, exp, start_date, end_date, **kwargs)
        Bulk EOD for all contracts in expiration

Open Interest:
    _retrieve_openInterest(symbol, exp, right, strike, start_date, end_date, **kwargs)
        Historical open interest for single contract

    _retrieve_bulk_open_interest(symbol, exp, start_date, end_date, **kwargs)
        Bulk open interest for all contracts

Chain Data:
    _retrieve_chain_bulk(symbol, exp, start_date, end_date, end_time, **kwargs)
        Complete option chain snapshot at specific time

Listings:
    _list_contracts(symbol, date, **kwargs)
        List all available contracts for a date

    _list_dates(symbol, exp, right, strike, **kwargs)
        List all available dates for a contract

Parameter Formats
-----------------
symbol : str
    Underlying ticker symbol (e.g., 'AAPL', 'SPX')

exp : str
    Expiration date in 'YYYY-MM-DD' format (e.g., '2024-12-20')

right : str
    Option type: 'call', 'put', 'C', or 'P' (case-insensitive)

strike : float
    Strike price as decimal (e.g., 180.0, not 180000)

start_date, end_date : str
    Date range in 'YYYY-MM-DD' format

date : str
    Single date in 'YYYY-MM-DD' format

interval : str
    Timeframe for intraday data: '1m', '5m', '15m', '30m', '1h', etc.
    See VALID_INTERVALS in vars.py for full list

opttick : str (optional)
    Alternative to providing symbol/exp/right/strike separately.
    Format: 'AAPL20241220C180' → AAPL, 2024-12-20, Call, $180

print_url : bool
    Whether to print the API request URL (useful for debugging)

Return Format
-------------
All functions return pandas DataFrames with:
- DatetimeIndex (or MultiIndex for bulk data)
- Standardized column names
- Business hours filtering applied
- Timestamps formatted per SETTINGS configuration

Quote data columns:
    CloseBid, CloseAsk, Bid_size, Ask_size, Midpoint, Weighted_midpoint

OHLC data columns:
    Open, High, Low, Close, Volume

Open Interest columns:
    Open_interest (or OpenInterest depending on formatting)

Bulk data includes additional columns:
    Root, Strike, Expiration, Right (for multi-contract data)

Usage Examples
--------------
Realtime quote:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.endpoints import _retrieve_quote_rt

    quote = _retrieve_quote_rt(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0
    )

Historical quotes:

.. code-block:: python

    quotes = _retrieve_quote(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='5m'
    )

EOD data:

.. code-block:: python

    eod = _retrieve_eod_ohlc(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

Bulk data:

.. code-block:: python

    bulk_eod = _retrieve_bulk_eod(
        symbol='AAPL',
        exp='2024-12-20',
        start_date='2024-12-01',
        end_date='2024-12-31'
    )

Using opttick:

.. code-block:: python

    # Instead of separate parameters
    quote = _retrieve_quote_rt(opttick='AAPL20241220C180')

List contracts:

.. code-block:: python

    contracts = _list_contracts(symbol='AAPL', date='2024-12-01')
    print(contracts[['strike', 'right', 'expiration']])

List dates:

.. code-block:: python

    dates = _list_dates(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0
    )

Rules and Constraints
---------------------
Opttick Parameter:
    - If opttick is provided, it takes precedence over other parameters
    - If opttick is NOT provided, all required parameters must be provided
    - Opttick is not used for bulk endpoints (they return multiple contracts)

Index Structure:
    - Single contract queries: DatetimeIndex only
    - Bulk queries: May include MultiIndex with [Datetime, Strike, Right, etc.]

Resampling:
    - Intraday data is automatically resampled to requested interval
    - Bulk data is NOT resampled (too complex for multi-contract data)
    - EOD data doesn't need resampling

Ticker Changes:
    - Handled automatically for all public functions
    - Uses TICK_CHANGE_ALIAS mapping from trade.assets.helpers.utils
    - Logs info messages when querying old ticker symbols

Error Handling
--------------
Functions raise ThetaData-specific exceptions on errors:
    - ThetaDataNotFound: No data available
    - ThetaDataOSLimit: Rate limit exceeded
    - ThetaDataDisconnected: Terminal not running
    - MissingColumnError: Unexpected API response format

Performance Considerations
--------------------------
- Bulk endpoints are much more efficient than looping single contracts
- Some endpoints don't support native date ranges (fall back to multi-threading)
- Intraday queries with fine intervals can return large amounts of data
- Consider using EOD data when intraday precision isn't needed

Notes
-----
- Requires ThetaData Terminal running on localhost:25503
- All dates in 'YYYY-MM-DD' format
- Strike prices as floats (not integer cents)
- Functions log at INFO level for debugging
- Use print_url=True to see exact API requests

See Also
--------
- v3/utils.py : Helper functions for formatting and ticker changes
- v3/vars.py : Configuration and endpoint URLs
- v3/__init__.py : Module overview and configuration
- ../utils.py : Shared utilities (resampling, etc.)
"""

## Rules of Opttick Param Passing
# - If opttick is provided, it takes precedence over other option parameters (strike, right, symbol, exp).
# - If opttick is not provided, all other option parameters (strike, right, symbol, exp) will be used.
# - Opttick will not be used on bulks. Simply because bulks contain multiple options.


## Rules of ThetaData Querying
# - There is no bulk Intraday/Quote endpoint. The resampling is too complicated
# - Index is always only datetime

from io import StringIO
import pandas as pd
import numpy as np
from dbase.DataAPI.ThetaData.v3.vars import (
    EOD_OHLC,
    HISTORICAL_QUOTE,
    LIST_CONTRACTS,
    LIST_CONTRACTS_QUOTE,
    OHLC_URL,
    REALTIME_QUOTE_RAW,
    ALL_MUST_BE_PROVIDED_ERR,
    SETTINGS,
    OI_URL,
    LIST_DATES,
)
from dbase.DataAPI.ThetaData.utils import _handle_opttick_param, _all_is_provided, _fetch_data, bootstrap_ohlc
from dbase.utils import default_timestamp
from dbase.DataAPI.ThetaData.v3.utils import (
    _with_ticker_change_handling,
    _build_params,
    _new_dataframe_formatting,
    _multi_threaded_range_fetch,
)


def _raw_list_dates(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve list of available dates for an option contract.
    Use _list_dates() instead for automatic ticker change handling.
    """
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )

    txt = _fetch_data(
        LIST_DATES,
        _build_params(
            symbol=symbol,
            exp=exp,
            right=right,
            strike=strike,
        ),
        print_url=print_url,
    )
    data = pd.read_csv(StringIO(txt))
    return data


def _list_dates(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Retrieve list of available dates for an option contract.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the list of available dates.
    """
    # Use ticker change handler for automatic symbol resolution
    return (
        _with_ticker_change_handling(
            _raw_list_dates,
            symbol=symbol,
            exp=exp,
            right=right,
            strike=strike,
            opttick=opttick,
            print_url=print_url,
            **kwargs,
        )
        .to_numpy()
        .flatten()
    )


def _raw_retrieve_quote_rt(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve realtime Quote snapshot.
    Use _retrieve_quote_rt() instead for automatic ticker change handling.
    """
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )

    params = _build_params(
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
    )
    txt = _fetch_data(REALTIME_QUOTE_RAW, params, print_url=print_url)
    data = pd.read_csv(StringIO(txt))
    return data


def _inner_retrieve_quote_rt(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal wrapper for _retrieve_quote_rt.
    Use _retrieve_quote_rt() instead for automatic ticker change handling.
    """
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )
    assert _all_is_provided(symbol=symbol, exp=exp, right=right, strike=strike), ALL_MUST_BE_PROVIDED_ERR
    data = _raw_retrieve_quote_rt(symbol, exp, right, strike, print_url=print_url)
    data = _new_dataframe_formatting(data, interval="30m")

    ## Additional to match old formatting
    if SETTINGS.use_old_formatting:
        data["Bid"] = data["CloseBid"]
        data["Ask"] = data["CloseAsk"]

    return data


def _retrieve_quote_rt(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve realtime Quote snapshot for a symbol.

    Automatically handles ticker symbol changes (e.g., FB → META).

    This is a snapshot query (current/realtime data), so it uses the current symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the realtime quote snapshot.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _inner_retrieve_quote_rt,
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        opttick=opttick,
        print_url=print_url,
        **kwargs,
    )


def _inner_retrieve_bulk_quote_rt(
    symbol: str, exp: str = None, right: str = None, strike: float = None, *, print_url: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Internal function for bulk realtime quote retrieval.
    Use _retrieve_bulk_quote_rt() instead for automatic ticker change handling.
    """
    data = _raw_retrieve_quote_rt(symbol, exp, right, strike, print_url=print_url)
    data = _new_dataframe_formatting(data, interval="30m", is_bulk=True)

    ## Additional to match old formatting
    if SETTINGS.use_old_formatting:
        data["Bid"] = data["CloseBid"]
        data["Ask"] = data["CloseAsk"]

    return data


def _retrieve_bulk_quote_rt(
    symbol: str, exp: str = None, right: str = None, strike: float = None, *, print_url: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Retrieve bulk realtime Quote snapshot for a symbol.

    Automatically handles ticker symbol changes (e.g., FB → META).

    This is a snapshot query (current/realtime data), so it uses the current symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the realtime quote snapshot.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _inner_retrieve_bulk_quote_rt, symbol=symbol, exp=exp, right=right, strike=strike, print_url=print_url, **kwargs
    )


def _raw_retrieve_openInterest(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_date: str = None,
    end_date: str = None,
    at_date: str = None,
    *,
    print_url: bool = False,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve Open Interest.
    This is called by wrapper functions that handle ticker changes.
    """
    if all([start_date, end_date]) and at_date is not None:
        raise ValueError("Provide either start_date & end_date for range or at_date for specific date, not both.")
    is_timeseries = all([start_date, end_date]) or (at_date is None)
    if is_timeseries:
        return _multi_threaded_range_fetch(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            url=OI_URL,
            print_url=print_url,
            exp=exp,
            right=right,
            strike=strike,
            omit_interval=True,
            **kwargs,
        )

    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike,
        right=right,
        symbol=symbol,
        exp=exp,
        opttick=opttick,
    )

    params = _build_params(
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        date=at_date,
    )

    txt = _fetch_data(OI_URL, params, print_url=print_url)
    return pd.read_csv(StringIO(txt))


def _inner_retrieve_openInterest(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_date: str = None,
    end_date: str = None,
    at_date: str = None,
    *,
    print_url: bool = False,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal wrapper for _retrieve_openInterest.
    Use _retrieve_openInterest() instead for automatic ticker change handling.
    """
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )
    assert _all_is_provided(symbol=symbol, exp=exp, right=right, strike=strike), ALL_MUST_BE_PROVIDED_ERR
    data = _raw_retrieve_openInterest(
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        print_url=print_url,
        start_date=start_date,
        end_date=end_date,
        at_date=at_date,
        **kwargs,
    )
    data = _new_dataframe_formatting(df=data, interval="1d", is_bulk=False)

    if SETTINGS.use_old_formatting:
        data["Datetime"] = data.index
        data["Date"] = data.index.date
    return data


def _retrieve_openInterest(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_date: str = None,
    end_date: str = None,
    at_date: str = None,
    *,
    print_url: bool = False,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve Open Interest for an option over a date range or specific date.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Passing either start_date & end_date for range or at_date for specific date.
    If both are passed, raises ValueError.
    If range is passed, uses multithreading to fetch data for each date in range.
    If specific date is passed, fetches data for that date.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the open interest snapshot.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _inner_retrieve_openInterest,
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        start_date=start_date,
        end_date=end_date,
        at_date=at_date,
        print_url=print_url,
        opttick=opttick,
        **kwargs,
    )


def _inner_retrieve_bulk_open_interest(
    symbol: str,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_date: str = None,
    end_date: str = None,
    at_date: str = None,
    *,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function for bulk open interest retrieval.
    Use _retrieve_bulk_open_interest() instead for automatic ticker change handling.
    """
    data = _raw_retrieve_openInterest(
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        print_url=print_url,
        start_date=start_date,
        end_date=end_date,
        at_date=at_date,
        **kwargs,
    )
    data = _new_dataframe_formatting(df=data, interval="1d", is_bulk=False)

    if SETTINGS.use_old_formatting:
        data["Datetime"] = data.index
        data["Date"] = data.index.date
    return data


def _retrieve_bulk_open_interest(
    symbol: str,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_date: str = None,
    end_date: str = None,
    at_date: str = None,
    *,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve Bulk Open Interest for a symbol over a date range or specific date.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Passing either start_date & end_date for range or at_date for specific date.
    If both are passed, raises ValueError.
    If range is passed, uses multithreading to fetch data for each date in range.
    If specific date is passed, fetches data for that date.

    Omitting exp, right, strike fetches all contracts for the symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the bulk open interest snapshot.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _inner_retrieve_bulk_open_interest,
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        start_date=start_date,
        end_date=end_date,
        at_date=at_date,
        print_url=print_url,
        **kwargs,
    )


def _raw_retrieve_eod_ohlc(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    strike: float = None,
    right: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve historical EOD OHLC data.
    Use _retrieve_eod_ohlc() instead for automatic ticker change handling.
    """
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )

    params = _build_params(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        strike=strike,
        right=right,
        exp=exp,
    )

    assert _all_is_provided(
        symbol=symbol, start_date=start_date, end_date=end_date, exp=exp, strike=strike, right=right
    ), ALL_MUST_BE_PROVIDED_ERR + " Both start_date and end_date must be provided."
    text = _fetch_data(EOD_OHLC, params)
    df = pd.read_csv(StringIO(text))
    df = _new_dataframe_formatting(df, interval="1d")
    return df


def _retrieve_eod_ohlc(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    strike: float = None,
    right: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical EOD OHLC data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str, optional): Expiration date in 'YYYY-MM-DD' format.
            If exp is None, retrieves data for all expirations.
        strike (float, optional): Strike price of the option.
            if strike is None, retrieves data for all strikes.
        right (str, optional): Option type - 'call', 'put', or 'both'.
            If right is None, retrieves data for both calls and puts.
    Returns:
        pd.DataFrame: DataFrame containing the EOD OHLC data.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _raw_retrieve_eod_ohlc,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        strike=strike,
        right=right,
        opttick=opttick,
        **kwargs,
    )


def _raw_retrieve_bulk_eod(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    print_url: str = None,
    *,
    exp: str = None,
    strike: float = None,
    right: str = None,
) -> pd.DataFrame:
    """
    Internal function to retrieve bulk historical EOD OHLC data.
    Use _retrieve_bulk_eod() instead for automatic ticker change handling.
    """
    params = _build_params(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        strike=strike,
        right=right,
    )
    txt = _fetch_data(EOD_OHLC, params, print_url=print_url)
    data = pd.read_csv(StringIO(txt))
    data = _new_dataframe_formatting(data, interval="1d", is_bulk=True)
    return data


def _retrieve_bulk_eod(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    print_url: str = None,
    *,
    exp: str = None,
    strike: float = None,
    right: str = None,
) -> pd.DataFrame:
    """
    Retrieve bulk historical EOD OHLC data for option contracts.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        expiration (str, optional): Specific expiration date to filter results.
        strike (float, optional): Strike price to filter results.
        right (str, optional): Option type - 'call', 'put', or 'both'.
            If right is None, retrieves data for both calls and puts.
    Returns:
        pd.DataFrame: DataFrame containing the bulk EOD OHLC data.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _raw_retrieve_bulk_eod,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        strike=strike,
        right=right,
        print_url=print_url,
    )


def _raw_list_contracts(symbol: str, date: str, print_url: bool = False, **kwargs) -> pd.DataFrame:
    """
    Internal function to retrieve current option contracts.
    Use _list_contracts() instead for automatic ticker change handling.
    """
    response = _fetch_data(LIST_CONTRACTS, {"symbol": symbol, "date": date}, print_url=print_url)
    df = pd.read_csv(StringIO(response))
    df["timestamp"] = date
    df = _new_dataframe_formatting(df, interval="1d", ignore_drop_conditional=True)
    if SETTINGS.use_old_formatting:
        df.columns = df.columns.str.lower()
    return df


def _list_contracts(symbol: str, date: str = None, print_url: bool = False, **kwargs) -> pd.DataFrame:
    """
    Retrieve current option contracts for a symbol.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Args:
        symbol (str): The underlying asset symbol.
        date (str): The date for which to retrieve contracts (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing the option contracts.

    """
    # Use ticker change handler for automatic symbol resolution

    ## Catch start_date for backward compatible
    start_date = kwargs.pop("start_date")
    date = date or start_date
    return _with_ticker_change_handling(_raw_list_contracts, symbol=symbol, date=date, print_url=print_url, **kwargs)


def _raw_retrieve_chain_bulk(
    symbol: str = None,
    exp: str = None,
    date: str = None,
    right: str = None,
    strike: float = None,
    oi: bool = False,
    end_time: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve bulk option chain data.
    Use _retrieve_chain_bulk() instead for automatic ticker change handling.
    """
    assert date is not None, "date parameter must be provided."
    assert symbol is not None, "symbol parameter must be provided."

    end_time = end_time or "16:00:00"
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike,
        right=right,
        symbol=symbol,
        exp=exp,
    )

    params = _build_params(
        symbol=symbol,
        date=date,
        exp=exp,
        right=right,
        strike=strike,
    )
    if oi:
        data = _retrieve_bulk_open_interest(
            symbol=symbol, exp=exp, right=right, strike=strike, print_url=print_url, at_date=date, **kwargs
        )
    else:
        ## FOR FUTURE REFERENCE: This isn't an ideal endpoint. We are using at_time quote endpoint to simulate chain bulk
        ## Would be better to use list contracts quote. But this isn't available/supported in ThetaData V3 yet.

        params = _build_params(
            symbol=symbol,
            start_date=date,
            end_date=date,
            date=date,
            exp=exp,
            right=right,
            strike=strike,
            time_of_day=end_time,
        )
        txt = _fetch_data(LIST_CONTRACTS_QUOTE, params, print_url=print_url)
        data = pd.read_csv(StringIO(txt))

    if "timestamp" not in data.columns:
        data["timestamp"] = date

    data = _new_dataframe_formatting(data, interval="1d", is_bulk=True)
    if SETTINGS.use_old_formatting:
        data["Date"] = data.index.date
        data.index = default_timestamp(data.index)
    return data


def _retrieve_chain_bulk(
    symbol: str = None,
    exp: str = None,
    date: str = None,
    right: str = None,
    strike: float = None,
    oi: bool = False,
    end_time: str = None,
    print_url: bool = False,
    start_date: str = None,
    end_date: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve bulk option chain data for a symbol on a specific date.

    Automatically handles ticker symbol changes (e.g., FB → META).

    This function can also retrieve bulk open interest data if 'oi' is set to True.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    date : str
        Date for the chain bulk query in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    oi : bool
        If True, retrieves open interest data instead of chain data.
    end_time : str
        End time in 'HH:MM:SS' format.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the bulk option chain data.
    """
    ## Decide date from date, start_date & end_date to ensure backward-compatibility
    assert start_date == end_date, "start_date and end_date must be the same for bulk chain retrieval."
    assert not all(
        d is None for d in [start_date, end_date, date]
    ), "Either pass ONLY date or both start_date & end_date"

    date = date or start_date
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _raw_retrieve_chain_bulk,
        symbol=symbol,
        exp=exp,
        date=date,
        right=right,
        strike=strike,
        oi=oi,
        end_time=end_time,
        print_url=print_url,
        **kwargs,
    )


def _raw_retrieve_quote(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    interval: str = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve historical Quote data.
    Use _retrieve_quote() instead for automatic ticker change handling.
    """
    assert all([start_date, end_date]), "Both start_date and end_date must be provided."
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )

    data = _multi_threaded_range_fetch(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        url=HISTORICAL_QUOTE,
        exp=exp,
        right=right,
        strike=strike,
        interval=interval,
        print_url=print_url,
        **kwargs,
    )

    data = _new_dataframe_formatting(data, interval=interval or "30m")
    data = bootstrap_ohlc(data)

    if SETTINGS.use_old_formatting:
        data.rename(columns={"CloseBid": "Closebid"}, inplace=True)
        data.rename(columns={"CloseAsk": "Closeask"}, inplace=True)
        data["Date"] = data.index.date
    return data


def _retrieve_quote(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    interval: str = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical Quote data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str, optional): Expiration date in 'YYYY-MM-DD' format.
            If exp is None, retrieves data for all expirations.
        right (str, optional): Option type - 'call', 'put', or 'both'.
            If right is None, retrieves data for both calls and puts.
        strike (float, optional): Strike price of the option.
            if strike is None, retrieves data for all strikes.
    Returns:
        pd.DataFrame: DataFrame containing the historical Quote data.
    """
    # Use ticker change handler for automatic symbol resolution
    return _with_ticker_change_handling(
        _raw_retrieve_quote,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        right=right,
        strike=strike,
        interval=interval,
        opttick=opttick,
        print_url=print_url,
        **kwargs,
    )


def _raw_retrieve_ohlc(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    interval: str = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Internal function to retrieve historical OHLC data.
    Use _retrieve_ohlc() instead for automatic ticker change handling.
    """
    assert all([start_date, end_date]), "Both start_date and end_date must be provided."
    strike, right, symbol, exp = _handle_opttick_param(
        strike=strike, right=right, symbol=symbol, exp=exp, opttick=opttick
    )

    data = _multi_threaded_range_fetch(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        url=OHLC_URL,
        exp=exp,
        right=right,
        strike=strike,
        interval=interval,
        print_url=print_url,
        **kwargs,
    )
    data = _new_dataframe_formatting(df=data, interval=interval or "30m", is_bulk=False)
    return data


def _retrieve_ohlc(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    interval: str = None,
    *,
    opttick: str = None,
    print_url: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical OHLC data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB → META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        right (str): Option type - 'call', 'put', or 'both'.
        strike (float): Strike price of the option.
        interval (str): Data interval (e.g., '1m', '5m', etc.).
    Returns:
        pd.DataFrame: DataFrame containing the historical OHLC data.
    """
    # Use ticker change handler for automatic symbol resolution
    ##TODO: For future iteration we need to add following columns from retrieve_quote endpoint:
    ##      Bid_size, CloseBid, Ask_size, CloseAsk, Midpoint, Weighted_midpoint
    return _with_ticker_change_handling(
        _raw_retrieve_ohlc,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        right=right,
        strike=strike,
        interval=interval,
        opttick=opttick,
        print_url=print_url,
        **kwargs,
    )
