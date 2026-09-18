"""Expired-only ThetaData ``list_dates`` cache owned by FinanceDatabase.

QuantTools (and dbase coverage checks) read this cache; ``BaseDataManager.clear_all_caches``
clears the same ``LIST_DATE_CACHE`` instance via the QuantTools re-export.

Comment density: orchestration

Core Functions:
    get_listed_option_dates: Vendor calendar; disk cache only after expiration.
    clear_list_date_cache: Drop all cached calendars (used by QuantTools clear-all).

Processing Flow:
    1. Build opttick key and decide expired vs live from expiration vs now.
    2. Expired + cache hit → return stored ``range`` without calling ThetaData.
    3. Always call ``list_dates`` for live contracts (calendar can still grow).
    4. Persist ``range`` / min / max only when caching is enabled and the contract expired.

Caching Strategy:
    Same on-disk location as the former QuantTools ``dm_gen_cache/list_date_cache`` so
    existing expired entries remain valid.

Usage:
    >>> from dbase.DataAPI.ThetaData.list_dates_cache import get_listed_option_dates
    >>> dates = get_listed_option_dates("AAPL", 150.0, "C", "2020-01-17")
"""

from datetime import datetime
from pathlib import Path
from typing import Any, List, Union
import os

from trade.helpers.helper import CustomCache, generate_option_tick_new, to_datetime
from trade.helpers.Logging import setup_logger

logger = setup_logger("dbase.DataAPI.ThetaData.list_dates_cache")

PATH = Path(os.environ["GEN_CACHE_PATH"]) / "dm_gen_cache"

## Shared with QuantTools ``trade.datamanager.utils.date.LIST_DATE_CACHE`` (re-export).
LIST_DATE_CACHE = CustomCache(
    location=PATH.as_posix(),
    fname="list_date_cache",
    clear_on_exit=False,
    expire_days=365,
)


def _list_dates_caching_enabled() -> bool:
    """Return QuantTools cache flag when available; otherwise cache expired calendars.

    Returns:
        True when disk writes for expired ``list_dates`` are allowed.
    """
    try:
        from trade.datamanager.vars import get_enable_caching

        return bool(get_enable_caching())
    except Exception:
        return True


def _normalize_right(right: str) -> str:
    """Map call/put aliases to the C/P cache-key form.

    Args:
        right: Vendor or QuantTools right string.

    Returns:
        ``C`` or ``P`` (or the uppercased input if already a single letter).
    """
    raw = str(right).strip().upper()
    if raw in ("CALL", "C"):
        return "C"
    if raw in ("PUT", "P"):
        return "P"
    return raw


def _contract_has_expired(expiration: Union[datetime, str]) -> bool:
    """Return True when expiration is strictly before local now.

    Args:
        expiration: Contract expiry.

    Returns:
        True if the contract can no longer list new sessions.
    """
    return to_datetime(expiration).date() < datetime.now().date()


def clear_list_date_cache() -> None:
    """Clear every cached expired ``list_dates`` entry."""
    LIST_DATE_CACHE.clear()


def get_listed_option_dates(
    ticker: str,
    strike: float,
    right: str,
    expiration: Union[datetime, str],
) -> List[Any]:
    """Return vendor ``list_dates`` for one option; cache only after expiration.

    Args:
        ticker: Underlying root (current ticker, e.g. META).
        strike: Strike price.
        right: ``C`` / ``P`` or call/put.
        expiration: Option expiration.

    Returns:
        List of vendor session dates (datetimes or date-like, as ``list_dates`` returns).

    Raises:
        ThetaDataNotFound: Vendor has no calendar for the contract (propagated).
    """
    ## Lazy import: package __init__ must not import this module before switcher exists.
    from dbase.DataAPI.ThetaData.switcher import list_dates

    option_has_expired = _contract_has_expired(expiration)
    opttick = generate_option_tick_new(
        symbol=ticker,
        strike=float(strike),
        right=_normalize_right(right),
        exp=expiration,
    )

    ## Live calendars still grow; never serve them from disk.
    if opttick in LIST_DATE_CACHE and option_has_expired:
        dates = LIST_DATE_CACHE[opttick]["range"]
        logger.info("Using cached list of dates for %s: %s", opttick, dates)
        return dates

    available_dates = list(
        list_dates(
            symbol=ticker,
            strike=strike,
            right=right,
            exp=expiration,
        )
    )
    logger.info("List of dates for %s: %s", opttick, available_dates)

    if _list_dates_caching_enabled() and option_has_expired:
        LIST_DATE_CACHE[opttick] = {
            "range": available_dates,
            "last_updated": datetime.now(),
            "min_date": min(available_dates) if available_dates else None,
            "max_date": max(available_dates) if available_dates else None,
        }

    return available_dates
