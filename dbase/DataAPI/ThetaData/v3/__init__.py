"""
ThetaData API V3 Client Module
===============================

This module provides a modernized Python interface to ThetaData's V3 REST API for
accessing historical and real-time options market data.

Overview
--------
The V3 API offers improved performance, more consistent data structures, and enhanced
functionality compared to V2. Key improvements include:
    - More efficient bulk data retrieval endpoints
    - Standardized datetime handling and index structures
    - Built-in ticker symbol change handling for corporate actions (e.g., FB → META)
    - Multi-threaded date range fetching for endpoints without native range support
    - Automatic data formatting and resampling capabilities
    - Better error handling and retry logic with exponential backoff

Architecture
------------
The module is organized into several components:

endpoints.py
    Core data retrieval functions for options data including:
    - Realtime and historical quotes
    - EOD and intraday OHLC data
    - Open interest data
    - Option chain listings
    - Contract listings

utils.py
    Helper functions for data processing, including:
    - Ticker change resolution for corporate actions
    - DataFrame formatting and normalization
    - Multi-threaded date range fetching
    - Interval and timestamp conversions

vars.py
    Configuration constants and URL endpoints:
    - API base URL and endpoint paths
    - Settings for data formatting preferences
    - Valid intervals and time constants

Data Retrieval Functions
-------------------------
All public data retrieval functions support automatic ticker change handling:

- `_retrieve_quote_rt()` - Realtime option quotes (snapshot)
- `_retrieve_bulk_quote_rt()` - Bulk realtime quotes for entire chain
- `_retrieve_quote()` - Historical intraday quotes
- `_retrieve_ohlc()` - Historical intraday OHLC data
- `_retrieve_eod_ohlc()` - End-of-day OHLC data
- `_retrieve_bulk_eod()` - Bulk EOD data for entire chain
- `_retrieve_openInterest()` - Historical open interest data
- `_retrieve_bulk_open_interest()` - Bulk open interest data
- `_retrieve_chain_bulk()` - Complete option chain snapshot at specific time
- `_list_contracts()` - List available option contracts for a date

Ticker Change Handling
----------------------
Functions automatically handle ticker symbol changes (corporate actions) by:
1. Detecting if a symbol has a historical ticker change (e.g., FB → META on 2022-06-09)
2. For historical queries: splitting date ranges and querying both old/new symbols
3. For at-time queries: using the correct symbol for the specific date
4. For snapshot queries: always using the current symbol

This behavior is transparent to the caller - simply provide the current ticker symbol
and the module handles the rest.

Usage Example
-------------
```python
from dbase.DataAPI._ThetaData.v3 import endpoints

# Retrieve EOD data for a specific option contract
data = endpoints._retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Get bulk EOD data for all contracts in an expiration
bulk_data = endpoints._retrieve_bulk_eod(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# List all available contracts for a date
contracts = endpoints._list_contracts(
    symbol='AAPL',
    date='2024-01-15'
)
```

Configuration
-------------
Data formatting can be customized via the SETTINGS singleton:

```python
from dbase.DataAPI._ThetaData.v3.vars import SETTINGS

# Use legacy formatting to match V2 output
SETTINGS.use_old_formatting = True

# Customize date formats
SETTINGS.eod_format = "%Y-%m-%d"
SETTINGS.intra_format = "%Y-%m-%d %H:%M:%S"
```

Notes
-----
- All functions require ThetaData Terminal to be running locally on port 25503
- Date parameters accept strings in 'YYYY-MM-DD' format or pandas Timestamps
- Strike prices should be provided as floats (e.g., 180.0, not 180000)
- Option rights should be 'C' for calls or 'P' for puts
- Functions use exponential backoff retry logic for transient API errors

See Also
--------
- ThetaData API Documentation: https://http-docs.thetadata.us/
- V2 API (legacy): dbase.DataAPI._ThetaData.v2
- Custom exceptions: dbase.DataAPI.ThetaExceptions
"""
