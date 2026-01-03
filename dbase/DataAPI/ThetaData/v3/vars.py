"""
ThetaData V3 API Configuration and Constants
=============================================

This module defines configuration settings, API endpoints, and constants for the
ThetaData V3 API client.

Overview
--------
Contains:
- ThetaDataV3Controls dataclass for runtime configuration
- API endpoint URLs for all V3 services
- Valid interval and option right values
- Time conversion constants
- Error messages and warnings

Configuration Class
-------------------
ThetaDataV3Controls (Singleton):
    use_old_formatting : bool
        Match V2 output format (capitalize columns, add EOD timestamp).
        Default: True (for backward compatibility)

    eod_format : str
        Date format string for end-of-day data.
        Default: "%Y-%m-%d" (e.g., "2024-12-20")

    intra_format : str
        Datetime format string for intraday data.
        Default: "%Y-%m-%d %H:%M:%S" (e.g., "2024-12-20 15:30:00")

Usage:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.vars import SETTINGS

    # Disable old formatting (use new V3 format)
    SETTINGS.use_old_formatting = False

    # Customize date formats
    SETTINGS.eod_format = "%Y%m%d"
    SETTINGS.intra_format = "%Y-%m-%d %H:%M"

API Endpoints
-------------
BASE_URL : str
    Base URL for V3 API: http://localhost:25503/v3

Contract Listing:
    LIST_CONTRACTS : str
        List available contracts: /option/list/contracts/trade

    LIST_DATES : str
        List available dates for a contract: /option/list/dates/quote

    LIST_CONTRACTS_QUOTE : str
        List contracts at specific time: /option/at_time/quote

OHLC Data:
    OHLC_URL : str
        Intraday OHLC history: /option/history/ohlc

    EOD_OHLC : str
        End-of-day OHLC history: /option/history/eod

Quote Data:
    REALTIME_QUOTE_RAW : str
        Realtime quote snapshot: /option/snapshot/quote

    SNAPSHOT_QUOTES : str
        Alias for realtime quotes

    HISTORICAL_QUOTE : str
        Historical quote data: /option/history/quote

Open Interest:
    OI_URL : str
        Open interest history: /option/history/open_interest

Valid Values
------------
VALID_INTERVALS : list[str]
    Supported interval values for intraday data:
    ['tick', '10ms', '100ms', '500ms', '1s', '5s', '10s', '15s', '30s',
     '1m', '5m', '10m', '15m', '30m', '1h']

VALID_RIGHTS : list[str]
    Supported option right values:
    ['call', 'put', 'both']

Time Constants
--------------
ONE_DAY_MILLISECONDS : int
    Milliseconds in one day: 86,400,000

MINIMUM_MILLISECONDS : int
    Minimum allowed interval in milliseconds.
    Derived from PRICING_CONFIG["INTRADAY_AGG"].

Messages
--------
LOOP_WARN_MSG : str
    Warning message when endpoint doesn't support date ranges.
    Used when falling back to multi-threaded looping.

ALL_MUST_BE_PROVIDED_ERR : str
    Error message when required option parameters are missing.
    Used in parameter validation.

Usage Examples
--------------
Import configuration:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.vars import (
        SETTINGS,
        EOD_OHLC,
        VALID_INTERVALS,
        VALID_RIGHTS
    )

Customize formatting:

.. code-block:: python

    # Use new V3 formatting
    SETTINGS.use_old_formatting = False

    # Now columns will be lowercase, no EOD timestamp adjustment
    data = retrieve_eod_ohlc(symbol='AAPL', ...)
    # Columns: ['open', 'high', 'low', 'close', 'volume']

Validate intervals:

.. code-block:: python

    interval = '5m'
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval: {interval}")

Build API URLs:

.. code-block:: python

    from dbase.DataAPI.ThetaData.v3.vars import BASE_URL

    custom_endpoint = f"{BASE_URL}/custom/path"

Notes
-----
- All endpoints are relative to localhost:25503
- ThetaData Terminal must be running for API to work
- SETTINGS is a singleton (changes affect all module users)
- Changing use_old_formatting affects all subsequent API calls
- MINIMUM_MILLISECONDS prevents overly granular intervals

See Also
--------
- endpoints.py : Uses these constants for API calls
- utils.py : Uses SETTINGS for data formatting
- __init__.py : Module overview and usage guide
"""

from dataclasses import dataclass
from trade.helpers.helper_types import SingletonMetaClass
from trade import PRICING_CONFIG
from dbase.DataAPI.ThetaData.utils import convert_string_interval_to_miliseconds


@dataclass
class ThetaDataV3Controls(metaclass=SingletonMetaClass):
    use_old_formatting: bool = True
    eod_format: str = "%Y-%m-%d"
    intra_format: str = "%Y-%m-%d %H:%M:%S"


SETTINGS = ThetaDataV3Controls()

LOOP_WARN_MSG = (
    "ThetaData currently doesn't support range dates for this endpoint. Falling back to looping and multithreading"
)
ALL_MUST_BE_PROVIDED_ERR = (
    "If opttick is not provided, all other option parameters (strike, right, symbol, exp) must be provided."
)
BASE_URL = "http://localhost:25503/v3"
LIST_CONTRACTS = BASE_URL + "/option/list/contracts/trade"
LIST_CONTRACTS_QUOTE = BASE_URL + "/option/at_time/quote"
OHLC_URL = BASE_URL + "/option/history/ohlc"
EOD_OHLC = BASE_URL + "/option/history/eod"
REALTIME_QUOTE_RAW = BASE_URL + "/option/snapshot/quote"
HISTORICAL_QUOTE = BASE_URL + "/option/history/quote"
OI_URL = BASE_URL + "/option/history/open_interest"
SNAPSHOT_QUOTES = BASE_URL + "/option/snapshot/quote"
LIST_DATES = BASE_URL + "/option/list/dates/quote"
ONE_DAY_MILLISECONDS = 86400000
MINIMUM_MILLISECONDS = convert_string_interval_to_miliseconds(PRICING_CONFIG["INTRADAY_AGG"])

VALID_INTERVALS = [
    "tick",
    "10ms",
    "100ms",
    "500ms",
    "1s",
    "5s",
    "10s",
    "15s",
    "30s",
    "1m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
]

VALID_RIGHTS = ["call", "put", "both"]
