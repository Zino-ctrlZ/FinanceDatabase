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

1. Prefetch vendor ``list_dates`` when the contract is fully specified
2. Query listed sessions inside ``[start_date, end_date]`` (weekday grid only if ids are incomplete);
   quote history also pads NY today when it is a trading day, unexpired, and already in the window
   (``list_dates`` can lag ``history/quote``)
3. Build separate parameter sets for each date
4. Use default intraday interval (from PRICING_CONFIG)
5. Execute requests in parallel using runThreads()
6. On 472 (ThetaDataNotFound): omit that day so the rest of the range survives
7. Concatenate non-empty frames
8. After quote-to-EOD, ``enforce_listed_quote_coverage`` raises or omits
   missing listed sessions per ``SETTINGS.listed_session_not_found``

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
- Quote range queries for a full contract use vendor ``list_dates`` plus at most
  one unlisted today pad; incomplete contracts still use a holiday-stripped weekday grid
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
from dbase.DataAPI.ThetaExceptions import (
    MissingColumnError,
    ThetaDataContainsFutureDateError,
    ThetaDataNotFound,
    is_thetadata_exception,
)
from dbase.DataAPI.ThetaData.v3.vars import (
    SETTINGS,
    ONE_DAY_MILLISECONDS,
    MINIMUM_MILLISECONDS,
    VALID_INTERVALS,
    LOOP_WARN_MSG,
    ListedSessionNotFoundPolicy,
    HISTORICAL_QUOTE,
)

from trade import PRICING_CONFIG, HOLIDAY_SET
from trade.helpers.helper import is_weekend, ny_now
from trade.helpers.threads import runThreads
import numpy as np  # noqa
from ..utils import _fetch_data, _parse_csv_to_dataframe
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData.utils import convert_string_interval_to_miliseconds, resample, normalize_date_format
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS
from typing import Callable, Any, List, Optional, Set
from urllib.parse import urlencode
from trade.helpers.decorators import timeit  # noqa

logger = setup_logger("dbase.DataAPI.ThetaData.v3.utils")
## 472 on a session list_dates advertised: omit that day instead of aborting the range.
listed_gap_logger = setup_logger("dbase.DataAPI.ThetaData.v3.listed_quote_gap")


##NOTE: Interested in seeing additional overhead
# @timeit
def _new_dataframe_formatting(
    df: pd.DataFrame,
    interval: str,
    is_bulk: bool = False,
    ignore_drop_conditional: bool = False,
    force_resampling: bool = False,
) -> pd.DataFrame:
    """Normalize a raw ThetaData CSV frame to the v3 column/index contract.

    Empty frames (every session omitted, e.g. listed 472s) return empty without
    requiring ``timestamp``. Column names are lowercased before the timestamp
    check so ``Timestamp`` still parses.

    Args:
        df: Raw concatenated per-date CSV frame.
        interval: Requested interval (used for resampling).
        is_bulk: Keep contract identifier columns when True.
        ignore_drop_conditional: Skip dropping strike/right/expiration columns.
        force_resampling: Resample even when the interval looks like EOD.

    Returns:
        Formatted DataFrame with a datetime index.

    Raises:
        MissingColumnError: Non-empty frame with no timestamp column.
        ValueError: Interval below the configured minimum.
    """
    ## Listed-472 omit can concat to an empty frame with no columns.
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    df = df.copy()
    ## Some parsers leave timestamp as the index name rather than a column.
    if (
        df.index.name is not None
        and str(df.index.name).lower() == "timestamp"
        and "timestamp" not in [str(c).lower() for c in df.columns]
    ):
        df = df.reset_index()
    df.columns = df.columns.str.lower()
    if "timestamp" not in df.columns:
        raise MissingColumnError(
            "Dataframe is missing required 'timestamp' column. Reach out to chidi if you see this error."
        )
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
    if (not is_bulk and is_intraday) or force_resampling:
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


def _iso_session_date(date_param: Optional[str]) -> Optional[str]:
    """Normalize a request date param to ``YYYY-MM-DD``.

    Args:
        date_param: Vendor date string (``YYYYMMDD`` or ``YYYY-MM-DD``).

    Returns:
        ISO date string, or None when ``date_param`` is missing.
    """
    if date_param is None:
        return None
    return normalize_date_format(str(date_param), _type=1)


def _load_listed_quote_dates(
    symbol: str,
    exp: Optional[Any],
    right: Optional[str],
    strike: Optional[float],
) -> Optional[Set[str]]:
    """Load vendor quote ``list_dates`` once for a contract.

    Lazy-imports ``list_dates`` so this module can load before the switcher.

    Args:
        symbol: Underlying ticker.
        exp: Expiration (``YYYY-MM-DD`` or datetime).
        right: Option right.
        strike: Strike price.

    Returns:
        Set of ISO dates, or None when contract ids are incomplete or the
        calendar call fails (the range fetch then raises instead of guessing
        a weekday grid).
    """
    if symbol is None or exp is None or right is None or strike is None:
        return None
    ## Lazy import: switcher imports endpoints which import this module.
    from dbase.DataAPI.ThetaData.list_dates_cache import get_listed_option_dates

    try:
        dates = get_listed_option_dates(ticker=symbol, strike=float(strike), right=right, expiration=exp)
    except Exception as exc:
        logger.warning(
            "Could not prefetch list_dates for %s %s%s exp=%s: %s.",
            symbol,
            strike,
            right,
            exp,
            exc,
        )
        return None
    if dates is None:
        return None
    listed: Set[str] = set()
    for d in dates:
        iso = _iso_session_date(None if d is None else str(d))
        if iso is not None:
            listed.add(iso)
    return listed


def _listed_dates_in_request_window(
    listed_dates: Set[str],
    start_date: str,
    end_date: str,
) -> List[str]:
    """Return listed ISO dates clipped to ``[start_date, end_date]``.

    Args:
        listed_dates: Vendor ``list_dates`` ISO set.
        start_date: Inclusive request start (``YYYY-MM-DD`` or ``YYYYMMDD``).
        end_date: Inclusive request end.

    Returns:
        Sorted ISO dates inside the request window.

    Raises:
        ThetaDataNotFound: Start or end could not be parsed.
    """
    start_iso = _iso_session_date(start_date)
    end_iso = _iso_session_date(end_date)
    if start_iso is None or end_iso is None:
        raise ThetaDataNotFound(f"Could not parse quote range window start={start_date!r} end={end_date!r}")
    return sorted(d for d in listed_dates if start_iso <= d <= end_iso)


def _today_session_iso(start_date: str, end_date: str, exp: Optional[Any]) -> Optional[str]:
    """Return today's ISO date when it is a live session in the request window.

    ``list_dates`` often lags after the close while quote and end-of-day history
    already have prints. Pads at most one day when today is a trading day, the
    option is unexpired, and today falls inside ``[start_date, end_date]``.

    Args:
        start_date: Inclusive request start.
        end_date: Inclusive request end.
        exp: Option expiration. No pad when missing or when today is after expiry.

    Returns:
        Today's ``YYYY-MM-DD``, or None when the pad does not apply.
    """
    if exp is None:
        return None
    start_iso = _iso_session_date(start_date)
    end_iso = _iso_session_date(end_date)
    exp_iso = _iso_session_date(str(exp))
    if start_iso is None or end_iso is None or exp_iso is None:
        return None
    today_iso = ny_now().strftime("%Y-%m-%d")
    if today_iso < start_iso or today_iso > end_iso:
        return None
    if today_iso > exp_iso:
        return None
    if is_weekend(today_iso) or today_iso in HOLIDAY_SET:
        return None
    return today_iso


def _expected_listed_sessions(
    listed_dates: Set[str],
    start_date: str,
    end_date: str,
    exp: Optional[Any],
) -> List[str]:
    """Return vendor listed sessions in the window, plus today when it applies.

    Args:
        listed_dates: Vendor ``list_dates`` ISO set for the contract.
        start_date: Inclusive request start.
        end_date: Inclusive request end.
        exp: Option expiration.

    Returns:
        Sorted ISO session dates expected for quote, end-of-day, and OHLC coverage.
    """
    dates = _listed_dates_in_request_window(listed_dates, start_date, end_date)
    today_iso = _today_session_iso(start_date, end_date, exp)
    if today_iso is not None and today_iso not in dates:
        dates = sorted(dates + [today_iso])
    return dates


def _index_iso_dates(index: pd.Index) -> Set[str]:
    """Map a DatetimeIndex (possibly with EOD timestamps) to calendar ISO dates.

    Args:
        index: Quote or EOD frame index.

    Returns:
        Set of ``YYYY-MM-DD`` session dates present on the index.
    """
    if index is None or len(index) == 0:
        return set()
    ts = pd.to_datetime(index)
    return set(pd.Index(ts).strftime("%Y-%m-%d"))


def _omits_missing_listed_sessions() -> bool:
    """Return True when SETTINGS says to drop missing listed sessions.

    Returns:
        True for ``omit`` (string or enum); False for raise.
    """
    policy = SETTINGS.listed_session_not_found
    if isinstance(policy, ListedSessionNotFoundPolicy):
        return policy is ListedSessionNotFoundPolicy.OMIT
    return str(policy).strip().lower() == ListedSessionNotFoundPolicy.OMIT.value


def _v2_hist_request_url(
    endpoint: str,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    exp: str,
    right: str,
    strike: float,
) -> str:
    """Rebuild a v2 hist URL (root / YYYYMMDD / strike*1000).

    Args:
        endpoint: v2 path such as ``/v2/hist/option/eod``.
        symbol: Underlying root.
        start_date: Inclusive start.
        end_date: Inclusive end.
        exp: Expiration.
        right: Option right as sent to v2.
        strike: Strike in dollars.

    Returns:
        ``endpoint?root=...`` string.
    """
    query = {
        "end_date": int(pd.to_datetime(end_date).strftime("%Y%m%d")),
        "root": symbol,
        "use_csv": "true",
        "exp": int(pd.to_datetime(exp).strftime("%Y%m%d")),
        "right": right,
        "start_date": int(pd.to_datetime(start_date).strftime("%Y%m%d")),
        "strike": int(float(strike) * 1000),
    }
    return f"{endpoint}?{urlencode(query)}"


def _format_coverage_location(
    *,
    url: Optional[str],
    endpoint: Optional[str],
    symbol: str,
    start_date: str,
    end_date: str,
    exp: str,
    right: str,
    strike: float,
    interval: Optional[str] = None,
    missing: Optional[List[str]] = None,
) -> str:
    """Prefer a reconstructed request URL; otherwise the endpoint path.

    Quote history is per session. When ``missing`` is set and the endpoint is
    quote, pin ``date`` to the first missing listed day so the URL matches the
    worker that 472'd.

    Args:
        url: Caller-supplied request URL, used when reconstruction fails.
        endpoint: v2 or v3 hist path.
        symbol: Underlying ticker.
        start_date: Inclusive request start.
        end_date: Inclusive request end.
        exp: Expiration.
        right: Option right.
        strike: Strike price.
        interval: OHLC interval when valid for v3.
        missing: Listed ISO dates absent from the frame.

    Returns:
        ``url=...`` or ``endpoint=...`` fragment for logs and ``ThetaDataNotFound``.
    """
    reconstructed: Optional[str] = None
    if endpoint:
        try:
            if "/v2/" in str(endpoint):
                reconstructed = _v2_hist_request_url(
                    endpoint,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    exp=exp,
                    right=right,
                    strike=strike,
                )
            else:
                ## Quote workers do not send a range; pin the first hole.
                date = None
                range_start, range_end = start_date, end_date
                if missing and "quote" in str(endpoint).lower():
                    date = missing[0]
                    range_start, range_end = None, None
                build_interval = interval if interval in VALID_INTERVALS else None
                params = _build_params(
                    symbol=symbol,
                    start_date=range_start,
                    end_date=range_end,
                    date=date,
                    exp=exp,
                    strike=strike,
                    right=right,
                    interval=build_interval,
                )
                reconstructed = _quote_history_request_url(endpoint, params)
        except Exception:
            reconstructed = None
    if reconstructed:
        return f"url={reconstructed}"
    if url:
        return f"url={url}"
    if endpoint:
        return f"endpoint={endpoint}"
    return "endpoint=unknown"


def enforce_listed_session_coverage(
    df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    symbol: str,
    exp: str,
    right: str,
    strike: float,
    listed_dates: Optional[Set[str]] = None,
    url: Optional[str] = None,
    endpoint: Optional[str] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """Drop unlisted extra days; raise or omit missing listed sessions.

    Expected sessions are ``list_dates ∩ [start_date, end_date]``, plus today when
    it is a trading day, the option is unexpired, and today is in the window.
    Days before first listed / after last listed, and interior holes not on
    ``list_dates``, are not expected. Extra frame days not in that set are dropped.

    Used for single-contract quote-to-EOD, EOD, and OHLC (session dates from the
    index). Bulk queries are not covered here.

    Args:
        df: Quote, EOD, or OHLC frame (may be empty but should keep columns).
        start_date: Inclusive request start.
        end_date: Inclusive request end.
        symbol: Underlying ticker.
        exp: Expiration.
        right: Option right.
        strike: Strike price.
        listed_dates: Prefetched calendar; loaded if omitted.
        url: Request URL when the caller already has it (fallback if reconstruct fails).
        endpoint: v2/v3 hist path used to reconstruct ``url=`` in messages.
        interval: OHLC interval for reconstructed v3 query strings.

    Returns:
        Frame restricted to expected sessions. Missing expected dates are
        omitted by default (``SETTINGS.listed_session_not_found`` is ``omit``).

    Raises:
        ThetaDataNotFound: Calendar could not be loaded, or a listed session
            in the window is missing and policy is ``raise``.
    """
    if listed_dates is None:
        listed_dates = _load_listed_quote_dates(symbol=symbol, exp=exp, right=right, strike=strike)
    if listed_dates is None:
        loc = _format_coverage_location(
            url=url,
            endpoint=endpoint,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            exp=exp,
            right=right,
            strike=strike,
            interval=interval,
        )
        raise ThetaDataNotFound(
            f"Could not prefetch list_dates for {symbol} {strike}{right} exp={exp}; cannot check coverage. {loc}"
        )
    expected = set(_expected_listed_sessions(listed_dates, start_date, end_date, exp))
    if df.empty:
        clipped = df
    else:
        iso = pd.Index(pd.to_datetime(df.index).strftime("%Y-%m-%d"))
        extra = sorted(set(iso) - expected)
        if extra:
            loc = _format_coverage_location(
                url=url,
                endpoint=endpoint,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                exp=exp,
                right=right,
                strike=strike,
                interval=interval,
            )
            listed_gap_logger.warning(
                "Dropping rows not on list_dates in the request window. "
                "symbol=%s expiration=%s strike=%s right=%s extra=%s %s",
                symbol,
                exp,
                strike,
                right,
                extra,
                loc,
            )
        mask = np.asarray(iso.isin(list(expected)), dtype=bool)
        clipped = df[mask]
    present = _index_iso_dates(clipped.index)
    missing = sorted(expected - present)
    if not missing:
        return clipped
    loc = _format_coverage_location(
        url=url,
        endpoint=endpoint,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        right=right,
        strike=strike,
        interval=interval,
        missing=missing,
    )
    msg = (
        "Listed session(s) missing after fetch. "
        f"symbol={symbol} expiration={exp} strike={strike} right={right} "
        f"missing={missing} {loc}"
    )
    if _omits_missing_listed_sessions():
        listed_gap_logger.warning("%s; omitting those dates (SETTINGS.listed_session_not_found=omit).", msg)
        return clipped
    raise ThetaDataNotFound(msg)


enforce_listed_quote_coverage = enforce_listed_session_coverage


def _quote_history_request_url(endpoint: str, params: dict) -> str:
    """Build the v3 history URL that would be POSTed for this date.

    Args:
        endpoint: ThetaData path (e.g. HISTORICAL_QUOTE).
        params: Per-date query params.

    Returns:
        ``endpoint?k=v&...`` string for the 472 CSV.
    """
    query = urlencode({key: value for key, value in params.items() if value is not None})
    if not query:
        return endpoint
    return f"{endpoint}?{query}"


def _record_quote_472_csv(params: dict, url: Optional[str], missing_date: Optional[str]) -> None:
    """Append a QuantTools ``.cache`` CSV row for a quote-history 472.

    Args:
        params: Per-date request params.
        url: History/quote endpoint for this worker.
        missing_date: ISO session date.
    """
    if url is not None and url != HISTORICAL_QUOTE:
        return
    endpoint = url or HISTORICAL_QUOTE
    try:
        from trade.datamanager.utils.quote_472_log import append_quote_472_row

        append_quote_472_row(
            symbol=params.get("symbol"),
            expiration=params.get("expiration"),
            strike=params.get("strike"),
            right=params.get("right"),
            missing_date=missing_date,
            url=_quote_history_request_url(endpoint, params),
        )
    except Exception as exc:
        listed_gap_logger.warning("Could not record quote 472 csv: %s", exc)


def _frame_for_listed_not_found(
    params: dict,
    listed_dates: Optional[Set[str]],
    exc: ThetaDataNotFound,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """Return an empty frame for a per-date 472 so the range pool can finish.

    A raise here aborts ``runThreads`` for the whole window. Coverage of
    listed sessions is enforced after resample via
    ``enforce_listed_quote_coverage``. Quote-history 472s are also appended
    to ``GEN_CACHE_PATH/thetadata/quote_472.csv``.

    Args:
        params: Per-date request params (includes ``date`` as ``YYYYMMDD``).
        listed_dates: Prefetched ``list_dates`` ISO set, or None if unknown.
        exc: The 472 exception from this date's fetch.
        url: Worker endpoint URL (CSV is quote-history only).

    Returns:
        Empty DataFrame so ``concat`` omits this session.
    """
    iso = _iso_session_date(params.get("date"))
    on_calendar = listed_dates is not None and iso is not None and iso in listed_dates
    listed_gap_logger.warning(
        "ThetaData 472; omitting session from range concat. "
        "symbol=%s expiration=%s strike=%s right=%s date=%s on_list_dates=%s err=%s",
        params.get("symbol"),
        params.get("expiration"),
        params.get("strike"),
        params.get("right"),
        iso,
        on_calendar,
        exc,
    )
    _record_quote_472_csv(params, url, iso)
    return pd.DataFrame()


def _frame_for_future_date(params: dict, exc: ThetaDataContainsFutureDateError) -> pd.DataFrame:
    """Omit a session ThetaData rejected as still in the future.

    Args:
        params: Per-date request params (includes ``date`` as ``YYYYMMDD``).
        exc: The 400 future-date exception from this date's fetch.

    Returns:
        Empty DataFrame so ``concat`` skips this session.
    """
    iso = _iso_session_date(params.get("date"))
    listed_gap_logger.warning(
        "ThetaData 400 future-date; omitting from range. symbol=%s expiration=%s strike=%s right=%s date=%s err=%s",
        params.get("symbol"),
        params.get("expiration"),
        params.get("strike"),
        params.get("right"),
        iso,
        exc,
    )
    return pd.DataFrame()


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
        pd.DataFrame: Concatenated per-date frames. Per-date 472s are omitted
            from the concat; listed-session coverage is checked after EOD patch.

    Raises:
        ThetaDataNotFound: Vendor ``list_dates`` could not be prefetched.
        Exception: Other ThetaData domain errors from a worker. Transient
            fetch/parse errors log and omit that date.
    """
    logger.warning(LOOP_WARN_MSG + f" Endpoint: {url}")

    ## For any endpoint that requires interval, set default. Down the pipeline we resample to requested interval
    ## ThetaData V3 currently doesnt support 1d interval, so we set to default intraday.
    default_interval = PRICING_CONFIG["INTRADAY_AGG"]

    ## Remove interval from kwargs if exists
    if "interval" in kwargs:
        kwargs.pop("interval")

    ## Query listed sessions when the contract is fully specified; otherwise weekdays.
    listed_dates = _load_listed_quote_dates(
        symbol=symbol,
        exp=kwargs.get("exp"),
        right=kwargs.get("right"),
        strike=kwargs.get("strike"),
    )
    contract_complete = (
        symbol is not None
        and kwargs.get("exp") is not None
        and kwargs.get("right") is not None
        and kwargs.get("strike") is not None
    )
    if contract_complete:
        if listed_dates is None:
            raise ThetaDataNotFound(
                "Could not prefetch list_dates for "
                f"{symbol} {kwargs.get('strike')}{kwargs.get('right')} "
                f"exp={kwargs.get('exp')}; refusing a weekday quote grid."
            )
        dt_range = _expected_listed_sessions(
            listed_dates,
            start_date,
            end_date,
            kwargs.get("exp"),
        )
    else:
        dt_range = pd.date_range(start=start_date, end=end_date, freq="1b").strftime("%Y-%m-%d").tolist()
        dt_range = [dt for dt in dt_range if dt not in HOLIDAY_SET]
    if not dt_range:
        return pd.DataFrame(columns=["timestamp"])

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

    def _thread_fetch(url, params, print_url):
        """Fetch one date; omit 472s and future-date 400s; re-raise other ThetaData errors."""
        try:
            ## Connection resets are retried inside _fetch_data (expo backoff, 5 tries).
            txt = _fetch_data(url, params, print_url)
            return _parse_csv_to_dataframe(txt)
        except ThetaDataNotFound as e:
            ## 472 must not abort runThreads; coverage is enforced after EOD patch.
            return _frame_for_listed_not_found(params, listed_dates, e, url=url)
        except ThetaDataContainsFutureDateError as e:
            ## +1d pads (and similar) can include tomorrow; omit that session only.
            return _frame_for_future_date(params, e)
        except Exception as e:
            ## Other ThetaData domain errors stay fatal (permissions, disconnect, …).
            ## Transient per-date fetch/parse noise should not fail the whole range.
            if is_thetadata_exception(e) or isinstance(e, MissingColumnError):
                raise
            logger.error(f"Error fetching data for params {params}: {e}")
            return pd.DataFrame()

    frames = runThreads(_thread_fetch, inputs)
    nonempty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not nonempty:
        ## Keep a timestamp column so callers that format immediately do not
        ## raise MissingColumnError when every session was a listed 472 omit.
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(nonempty, ignore_index=True)


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


def _get_all_symbols_for_ticker_change(symbol: str) -> list[str]:
    """
    Get all relevant symbols for a ticker that has changed.

    This is used for snapshot queries where we want to try all possible symbols.

    Parameters
    ----------
    symbol : str
        Current ticker symbol

    Returns
    -------
    list[str]
        List of all symbols to try for this ticker (e.g., [old_symbol, new_symbol])
    """
    if symbol not in TICK_CHANGE_ALIAS:
        return [symbol]

    old_symbol, new_symbol, _ = TICK_CHANGE_ALIAS[symbol]
    return [old_symbol, new_symbol]


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
        missing_roots = []
        fetched_roots = []
        for segment_symbol, seg_start, seg_end in segments:
            logger.info(f"Fetching {segment_symbol} data: {seg_start} to {seg_end}")

            # Update kwargs with segment-specific dates
            segment_kwargs = kwargs.copy()
            segment_kwargs["start_date"] = seg_start
            segment_kwargs["end_date"] = seg_end

            try:
                df = func(symbol=segment_symbol, **segment_kwargs)

                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    missing_roots.append(segment_symbol)
                    continue

                # Normalize root column to current symbol
                if "root" in df.columns:
                    df["root"] = symbol

                dataframes.append(df)
                fetched_roots.append(segment_symbol)

            except Exception as e:
                logger.warning(f"Failed to fetch {segment_symbol} data: {e}")
                missing_roots.append(segment_symbol)
                continue

        # Merge results
        if not dataframes:
            raise ThetaDataNotFound(
                "No data after ticker-change split. "
                f"requested_symbol={symbol} missing_roots={missing_roots} "
                f"fetched_roots={fetched_roots} segments={segments} params={kwargs}"
            )

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

    # Case 4: Function name == list_dates endpoint - special handling to try all symbols
    elif func.__name__ == "_raw_list_dates":
        ## list_dates endpoint is a special case where it doesn't have any date parameters
        ## but still needs to handle ticker changes. In this case, we will try to run for all symbols and return the one that works.
        ## Edge case: What if all work??
        ## Then we run the following logic:
        ## 1 If only one works, return that one
        ## 2 If multiple work, combine results and

        def _run_without_printing_error():
            all_symbols = _get_all_symbols_for_ticker_change(symbol)
            res = {}
            for sym in all_symbols:
                try:
                    res[sym] = func(symbol=sym, **kwargs)
                except Exception as f:
                    logger.warning(f"Failed to fetch data for symbol {sym}: {f}")
            return res

        results = _run_without_printing_error()
        if not results:
            raise ThetaDataNotFound(f"No data found for any symbol related to {symbol}")

        ## Combine results if multiple symbols worked, and remove duplicates
        res = []
        for lst in results.values():
            res.append(lst)
        res = pd.concat(res).drop_duplicates().sort_values("date")
        return res

    # Case 5: Snapshot query (no date params) - use current symbol
    else:
        logger.info(f"Snapshot query - using current symbol {symbol}")
        ## If it's a snapshot query, it returns.
        return func(symbol=symbol, **kwargs)

        # except Exception as e:
        # def _run_without_printing_error():
        #     all_symbols = _get_all_symbols_for_ticker_change(symbol)
        #     res = {}
        #     for sym in all_symbols:
        #         try:
        #             res[sym] = func(symbol=sym, **kwargs)
        #         except Exception as f:
        #             logger.warning(f"Failed to fetch data for symbol {sym}: {f}")
        #     return res
        # results = _run_without_printing_error()
        # if not results:
        #     raise ThetaDataNotFound(f"No data found for any symbol related to {symbol}") from e
        # if len(results) == 1:
        #     return list(results.values())[0]
        # if symbol in results:
        #     return results[symbol]
        # else:
        #     logger.warning(f"Multiple symbols returned data, but none matched the current symbol {symbol}. Returning data for {list(results.keys())}")
        #     return list(results.values())[0]
