"""
ThetaData API Switcher (Per-call Version Selection)
===================================================

This module provides wrappers that choose between V2 and V3 at call time,
using the THETADATA_USE_V3 environment variable. This avoids locking the
selection at import time.
"""

import os
import pandas as pd
from trade.helpers.Logging import setup_logger
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
from ..ThetaExceptions import raise_thetadata_exception

logger = setup_logger("dbase.DataAPI.ThetaData.switcher", stream_log_level="INFO")


def _use_v2() -> bool:
    return os.environ.get("THETADATA_USE_V3", "false").lower() == "false"


def get_use_v2() -> bool:
    return _use_v2()


def _log_call(func_name: str, use_v2: bool) -> None:
    version = "v2" if use_v2 else "v3"
    logger.info("Switcher call %s -> %s", func_name, version)


def retrieve_quote_rt(
    symbol: str = None,
    exp: str = None,
    right: str = None,
    strike: float = None,
    start_time: str = None,
    print_url: bool = False,
    end_time: str = None,
    ts: bool = False,
    proxy: str = None,
    start_date: str = None,
    end_date: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve realtime Quote snapshot for a symbol.

    Automatically handles ticker symbol changes (e.g., FB -> META).

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
    start_time : str
        Start time for intraday data (for backward compatibility, unused in V3).
    print_url : bool
        Whether to print the request URL.
    end_time : str
        End time for intraday data (for backward compatibility, unused in V3).
    ts : bool
        Time series flag (for backward compatibility, unused in V3).
    proxy : str
        Proxy URL for backward compatibility (unused in V3).
    start_date : str
        Start date (for backward compatibility, unused in V3).
    end_date : str
        End date (for backward compatibility, unused in V3).
    Returns
    -------
    pd.DataFrame
        DataFrame containing the realtime quote snapshot.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_quote_rt", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_quote_rt(
            symbol=symbol,
            exp=exp,
            right=right,
            strike=strike,
            start_time=start_time,
            print_url=print_url,
            end_time=end_time,
            ts=ts,
            proxy=proxy,
            start_date=start_date,
            end_date=end_date,
            opttick=opttick,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_quote_rt(
        symbol=symbol,
        exp=exp,
        right=right,
        strike=strike,
        start_time=start_time,
        print_url=print_url,
        end_time=end_time,
        ts=ts,
        proxy=proxy,
        start_date=start_date,
        end_date=end_date,
        opttick=opttick,
        **kwargs,
    )


def retrieve_bulk_quote_rt(
    symbol: str, exp: str = None, right: str = None, strike: float = None, *, print_url: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Retrieve bulk realtime Quote snapshot for a symbol.

    Automatically handles ticker symbol changes (e.g., FB -> META).

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
    use_v2 = _use_v2()
    _log_call("retrieve_bulk_quote_rt", use_v2)
    if use_v2:
        logger.warning("Bulk realtime quotes not supported in V2 ThetaData API")
        return None
    from .v3 import endpoints

    return endpoints._retrieve_bulk_quote_rt(
        symbol=symbol, exp=exp, right=right, strike=strike, print_url=print_url, **kwargs
    )


def retrieve_quote(
    symbol: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    start_date: str = None,
    strike: float = None,
    start_time: str = None,
    print_url: bool = False,
    end_time: str = None,
    interval: str = None,
    proxy: str = None,
    ohlc_format: bool = True,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical Quote data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        right (str): Option type - 'call', 'put', 'C', or 'P'.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        strike (float): Strike price of the option.
        start_time (str): Start time for intraday filtering (for backward compatibility, unused in V3).
        print_url (bool): Whether to print the request URL.
        end_time (str): End time for intraday filtering (for backward compatibility, unused in V3).
        interval (str): Data interval (e.g., '1m', '5m', '30m').
        proxy (str): Proxy URL for distributed fetching (unused in V3).
        ohlc_format (bool): Format flag (for backward compatibility, unused in V3).
    Returns:
        pd.DataFrame: DataFrame containing the historical Quote data.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_quote", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_quote(
            symbol=symbol,
            end_date=end_date,
            exp=exp,
            right=right,
            start_date=start_date,
            strike=strike,
            start_time=start_time,
            print_url=print_url,
            end_time=end_time,
            interval=interval,
            proxy=proxy,
            ohlc_format=ohlc_format,
            opttick=opttick,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_quote(
        symbol=symbol,
        end_date=end_date,
        exp=exp,
        right=right,
        start_date=start_date,
        strike=strike,
        start_time=start_time,
        print_url=print_url,
        end_time=end_time,
        interval=interval,
        proxy=proxy,
        ohlc_format=ohlc_format,
        opttick=opttick,
        **kwargs,
    )


def retrieve_ohlc(
    symbol: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    start_date: str = None,
    strike: float = None,
    start_time: str = None,
    print_url: bool = False,
    proxy: str = None,
    interval: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical OHLC data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        right (str): Option type - 'call', 'put', or 'both'.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        strike (float): Strike price of the option.
        start_time (str): Start time for intraday filtering (for backward compatibility, unused in V3).
        print_url (bool): Whether to print the request URL.
        proxy (str): Proxy URL for distributed fetching (unused in V3).
        interval (str): Data interval (e.g., '1m', '5m', etc.).
    Returns:
        pd.DataFrame: DataFrame containing the historical OHLC data.

    WARNING - INCOMPLETE V2 COMPATIBILITY:
        V2's retrieve_ohlc returned 6 additional quote-related columns that are missing in V3:
        - Bid_size
        - CloseBid (or Closebid)
        - Ask_size
        - CloseAsk (or Closeask)
        - Midpoint
        - Weighted_midpoint

        These columns are not available from V3's OHLC endpoint. To achieve full V2 compatibility,
        you would need to:
        1. Retrieve OHLC data (Open, High, Low, Close, Volume)
        2. Retrieve quote data for the same time range (bid/ask data)
        3. Merge the two datasets

        For now, this function returns only the OHLC columns. If you need the quote columns,
        use retrieve_quote() instead which returns the full quote data including bid/ask.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_ohlc", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_ohlc(
            symbol=symbol,
            end_date=end_date,
            exp=exp,
            right=right,
            start_date=start_date,
            strike=strike,
            start_time=start_time,
            print_url=print_url,
            proxy=proxy,
            interval=interval,
            opttick=opttick,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_ohlc(
        symbol=symbol,
        end_date=end_date,
        exp=exp,
        right=right,
        start_date=start_date,
        strike=strike,
        start_time=start_time,
        print_url=print_url,
        proxy=proxy,
        interval=interval,
        opttick=opttick,
        **kwargs,
    )


def retrieve_option_ohlc(*args, **kwargs):
    use_v2 = _use_v2()
    _log_call("retrieve_option_ohlc", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_option_ohlc(*args, **kwargs)
    logger.warning("retrieve_option_ohlc is not available in V3; using retrieve_ohlc instead")
    from .v3 import endpoints

    return endpoints._retrieve_ohlc(*args, **kwargs)


def retrieve_eod_ohlc(
    symbol: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    start_date: str = None,
    strike: float = None,
    print_url: bool = False,
    rt: bool = True,
    proxy: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve historical EOD OHLC data for an option contract.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        end_date (str): End date in 'YYYY-MM-DD' format.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        right (str): Option type - 'call', 'put', 'C', or 'P'.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        strike (float): Strike price of the option.
        print_url (bool): Whether to print the request URL.
        rt (bool): Real-time flag for compatibility (unused in V3).
        proxy (str): Proxy URL for distributed fetching (unused in V3).
    Returns:
        pd.DataFrame: DataFrame containing the EOD OHLC data.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_eod_ohlc", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_eod_ohlc(
            symbol=symbol,
            end_date=end_date,
            exp=exp,
            right=right,
            start_date=start_date,
            strike=strike,
            print_url=print_url,
            rt=rt,
            proxy=proxy,
            opttick=opttick,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_eod_ohlc(
        symbol=symbol,
        end_date=end_date,
        exp=exp,
        right=right,
        start_date=start_date,
        strike=strike,
        print_url=print_url,
        rt=rt,
        proxy=proxy,
        opttick=opttick,
        **kwargs,
    )


def retrieve_bulk_eod(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    print_url: bool = False,
    proxy: str = None,
    *,
    exp: str = None,
    strike: float = None,
    right: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve bulk historical EOD OHLC data for option contracts.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Parameters:
        symbol (str): Underlying asset ticker symbol.
        exp (str): Expiration date in 'YYYY-MM-DD' format.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        print_url (bool): Whether to print the request URL.
        proxy (str): Proxy URL for distributed fetching (unused in V3).
        expiration (str, optional): Specific expiration date to filter results.
        strike (float, optional): Strike price to filter results.
        right (str, optional): Option type - 'call', 'put', or 'both'.
            If right is None, retrieves data for both calls and puts.
    Returns:
        pd.DataFrame: DataFrame containing the bulk EOD OHLC data.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_bulk_eod", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_bulk_eod(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            print_url=print_url,
            proxy=proxy,
            exp=exp,
            strike=strike,
            right=right,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_bulk_eod(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        print_url=print_url,
        proxy=proxy,
        exp=exp,
        strike=strike,
        right=right,
        **kwargs,
    )


def retrieve_openInterest(
    symbol: str = None,
    end_date: str = None,
    exp: str = None,
    right: str = None,
    start_date: str = None,
    strike: float = None,
    print_url: bool = False,
    proxy: str = None,
    at_date: str = None,
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve Open Interest for an option over a date range or specific date.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Passing either start_date & end_date for range or at_date for specific date.
    If both are passed, raises ValueError.
    If range is passed, uses multithreading to fetch data for each date in range.
    If specific date is passed, fetches data for that date.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'AAPL').
    end_date : str
        End date in 'YYYY-MM-DD' format.
    exp : str
        Expiration date in 'YYYY-MM-DD' format.
    right : str
        Option right ('call' or 'put').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    strike : float
        Strike price of the option.
    print_url : bool
        Whether to print the request URL.
    proxy : str
        Proxy URL for backward compatibility (unused in V3).
    Returns
    -------
    pd.DataFrame
        DataFrame containing the open interest data with Datetime as a column.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_openInterest", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_openInterest(
            symbol=symbol,
            end_date=end_date,
            exp=exp,
            right=right,
            start_date=start_date,
            strike=strike,
            print_url=print_url,
            proxy=proxy,
            at_date=at_date,
            opttick=opttick,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_openInterest(
        symbol=symbol,
        end_date=end_date,
        exp=exp,
        right=right,
        start_date=start_date,
        strike=strike,
        print_url=print_url,
        proxy=proxy,
        at_date=at_date,
        opttick=opttick,
        **kwargs,
    )


def retrieve_bulk_open_interest(
    symbol: str,
    exp: str = None,
    start_date: str = None,
    end_date: str = None,
    print_url: bool = False,
    proxy: str = None,
    right: str = None,
    strike: float = None,
    at_date: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve Bulk Open Interest for a symbol over a date range or specific date.

    Automatically handles ticker symbol changes (e.g., FB -> META).

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
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    print_url : bool
        Whether to print the request URL.
    proxy : str
        Proxy URL for backward compatibility (unused in V3).
    right : str
        Option right ('call' or 'put').
    strike : float
        Strike price of the option.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the bulk open interest data.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_bulk_open_interest", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_bulk_open_interest(
            symbol=symbol,
            exp=exp,
            start_date=start_date,
            end_date=end_date,
            print_url=print_url,
            proxy=proxy,
            right=right,
            strike=strike,
            at_date=at_date,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_bulk_open_interest(
        symbol=symbol,
        exp=exp,
        start_date=start_date,
        end_date=end_date,
        print_url=print_url,
        proxy=proxy,
        right=right,
        strike=strike,
        at_date=at_date,
        **kwargs,
    )


def retrieve_chain_bulk(
    symbol: str = None,
    exp: str = None,
    date: str = None,
    right: str = None,
    strike: float = None,
    oi: bool = False,
    end_time: str = None,
    print_url: bool = False,
    proxy: str = None,
    start_date: str = None,
    end_date: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve bulk option chain data for a symbol on a specific date.

    Automatically handles ticker symbol changes (e.g., FB -> META).

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
    proxy : str
        Proxy URL for backward compatibility (unused in V3).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the bulk option chain data.
    """
    use_v2 = _use_v2()
    _log_call("retrieve_chain_bulk", use_v2)
    if use_v2:
        from . import v2

        return v2.retrieve_chain_bulk(
            symbol=symbol,
            exp=exp,
            date=date,
            right=right,
            strike=strike,
            oi=oi,
            end_time=end_time,
            print_url=print_url,
            proxy=proxy,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    from .v3 import endpoints

    return endpoints._retrieve_chain_bulk(
        symbol=symbol,
        exp=exp,
        date=date,
        right=right,
        strike=strike,
        oi=oi,
        end_time=end_time,
        print_url=print_url,
        proxy=proxy,
        start_date=start_date,
        end_date=end_date,
        **kwargs,
    )


def list_contracts(
    symbol: str, date: str = None, print_url: bool = False, proxy: str = None, **kwargs
) -> pd.DataFrame:
    """
    Retrieve current option contracts for a symbol.

    Automatically handles ticker symbol changes (e.g., FB -> META).

    Args:
        symbol (str): The underlying asset symbol.
        date (str): The date for which to retrieve contracts (YYYY-MM-DD).
        print_url (bool): Whether to print the request URL.
        proxy (str): Proxy URL for backward compatibility (unused in V3).

    Returns:
        pd.DataFrame: DataFrame containing the option contracts.

    """
    use_v2 = _use_v2()
    _log_call("list_contracts", use_v2)
    if use_v2:
        from . import v2

        return v2.list_contracts(symbol=symbol, date=date, print_url=print_url, proxy=proxy, **kwargs)
    from .v3 import endpoints

    return endpoints._list_contracts(symbol=symbol, date=date, print_url=print_url, proxy=proxy, **kwargs)


def list_dates(
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
    Retrieve list of available dates for an option contract.

    Automatically handles ticker symbol changes (e.g., FB  META).

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
    use_v2 = _use_v2()
    _log_call("list_dates", use_v2)
    if use_v2:
        from . import v2

        return v2.list_dates(
            symbol=symbol, exp=exp, right=right, strike=strike, opttick=opttick, print_url=print_url, **kwargs
        )
    from .v3 import endpoints

    return endpoints._list_dates(
        symbol=symbol, exp=exp, right=right, strike=strike, opttick=opttick, print_url=print_url, **kwargs
    )


def ping_proxy(*args, **kwargs):
    use_v2 = _use_v2()
    _log_call("ping_proxy", use_v2)
    if use_v2:
        from .proxy import ping_proxy_v2

        return ping_proxy_v2(*args, **kwargs)
    from .proxy import ping_proxy_v3

    return ping_proxy_v3(*args, **kwargs)


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
