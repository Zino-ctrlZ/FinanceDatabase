# ThetaData API Client

A comprehensive Python client for accessing ThetaData's options market data API with support for both V2 (legacy) and V3 (recommended) endpoints.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [API Versions](#api-versions)
  - [V2 vs V3 Comparison](#v2-vs-v3-comparison)
  - [Migration Guide](#migration-from-v2-to-v3)
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Data Types & Endpoints](#data-types--endpoints)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [API Reference](#api-reference)

## Overview

The ThetaData API Client provides seamless access to comprehensive options market data including:

- **OHLC Data**: End-of-day and intraday open/high/low/close with volume
- **Quote Data**: Historical and realtime bid/ask spreads with sizes
- **Open Interest**: Historical and snapshot open interest data
- **Option Chains**: Complete chain snapshots at specific times
- **Contract Listings**: Available contracts and trading dates
- **Greeks**: Delta, gamma, theta, vega, and rho (V2 only)

### Key Features

✅ **Automatic Ticker Change Handling**: Seamlessly handles corporate actions (e.g., FB → META)  
✅ **Version Agnostic Interface**: Switch between V2 and V3 with a single environment variable  
✅ **Multi-threaded Fetching**: Parallel data retrieval for date ranges  
✅ **Proxy Support**: Distributed fetching across multiple machines  
✅ **Business Hours Filtering**: Automatic market hours enforcement  
✅ **Flexible Resampling**: Convert data to any timeframe  
✅ **Comprehensive Logging**: Request latency tracking and debugging  
✅ **Type Safety**: Full type hints and docstrings

## Installation & Setup

### Prerequisites

1. **ThetaData Terminal** must be installed and running
   - Download from: https://thetadata.net
   - V2: Default port 25510
   - V3: Default port 25503

2. **Python Dependencies**:
```bash
pip install pandas numpy requests
```

3. **Environment Setup**:
```python
import os

# Use V3 (recommended)
os.environ["THETADATA_USE_V3"] = "true"  # Default

# Use V2 (legacy)
os.environ["THETADATA_USE_V3"] = "false"

# Optional: Configure proxy
os.environ["PROXY_URL"] = "http://proxy-server:8080/api"

# Optional: Set cache path for latency logs
os.environ["GEN_CACHE_PATH"] = "/path/to/cache"
```

### Basic Import

```python
from dbase.DataAPI.ThetaData import (
    retrieve_eod_ohlc,
    retrieve_quote,
    retrieve_ohlc,
    list_contracts,
    resample
)
```

## Quick Start

### Get End-of-Day Data

```python
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

# Get EOD OHLC for a specific option contract
data = retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(data.head())
#                        Open   High    Low  Close  Volume
# Datetime                                                 
# 2024-01-02 16:00:00  15.20  15.80  14.90  15.50    5430
# 2024-01-03 16:00:00  15.45  16.20  15.30  16.10    7892
# ...
```

### Get Intraday Quotes

```python
from dbase.DataAPI.ThetaData import retrieve_quote

# Get 5-minute quote data
quotes = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'
)

print(quotes.head())
#                      CloseBid  CloseAsk  Midpoint  Volume
# Datetime                                                  
# 2024-12-01 09:30:00    14.80     14.95    14.875    123
# 2024-12-01 09:35:00    14.85     15.00    14.925    156
# ...
```

### Get Bulk Data

```python
from dbase.DataAPI.ThetaData import retrieve_bulk_eod

# Get all contracts for an expiration
bulk = retrieve_bulk_eod(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-12-01',
    end_date='2024-12-15'
)

print(bulk.head())
#                      Root  Strike Right  Open  High   Low  Close  Volume
# Datetime                                                                 
# 2024-12-01 16:00:00  AAPL   170.0     C  18.5  19.2  18.3   19.0    3421
# 2024-12-01 16:00:00  AAPL   170.0     P   3.2   3.4   3.1    3.3     892
# 2024-12-01 16:00:00  AAPL   175.0     C  15.1  15.8  14.9   15.6    4123
# ...
```

## API Versions

### V2 vs V3 Comparison

| Feature | V2 (Legacy) | V3 (Recommended) |
|---------|-------------|------------------|
| **Port** | 25510 | 25503 |
| **Release** | Original | 2024+ |
| **Status** | Maintenance only | Active development |
| **Bulk Realtime Quotes** | ❌ Not available | ✅ Native support |
| **List Dates** | ❌ Not available | ✅ Native support |
| **Date Range Support** | Manual looping | Multi-threaded |
| **Ticker Changes** | Manual handling | Automatic |
| **Error Messages** | Generic | Detailed |
| **Column Names** | Inconsistent | Standardized |
| **Performance** | Good | Better |
| **Recommended For** | Legacy systems | New projects |

### Key Improvements in V3

1. **Better Endpoint Coverage**:
   - Bulk realtime quotes endpoint
   - Native date listing for contracts
   - More consistent API design

2. **Performance**:
   - Multi-threaded range fetching for unsupported endpoints
   - Optimized data formatting
   - Reduced memory footprint

3. **Developer Experience**:
   - Better error messages with actionable information
   - Standardized column naming conventions
   - More consistent datetime handling
   - Improved logging and debugging

4. **Automatic Features**:
   - Ticker symbol change handling built-in
   - Business hours filtering
   - Data type conversions

## Architecture

```
dbase/DataAPI/ThetaData/
│
├── __init__.py          # Version-agnostic interface
├── utils.py             # Shared utilities (resampling, formatting)
├── proxy.py             # Proxy configuration and management
├── log.py               # Request latency logging
│
├── v2.py                # V2 API implementation (legacy)
│
└── v3/                  # V3 API implementation
    ├── __init__.py      # V3 module overview
    ├── endpoints.py     # All V3 endpoint functions
    ├── utils.py         # V3-specific utilities
    └── vars.py          # Configuration and constants
```

### Module Responsibilities

- **`__init__.py`**: Version detection and function exports
- **`utils.py`**: Data resampling, OHLC formatting, time conversions
- **`proxy.py`**: Proxy server configuration
- **`log.py`**: Automatic latency logging to CSV
- **`v2.py`**: Legacy V2 API implementation
- **`v3/endpoints.py`**: All V3 data retrieval functions
- **`v3/utils.py`**: Ticker changes, data formatting, multi-threading
- **`v3/vars.py`**: API URLs, valid values, configuration

## Core Features

### 1. Automatic Ticker Change Handling

The client automatically handles ticker symbol changes from corporate actions:

```python
# Query META (formerly FB) - automatically splits at change date (2022-06-09)
data = retrieve_eod_ohlc(
    symbol='META',  # Use current ticker
    exp='2022-12-20',
    right='C',
    strike=200.0,
    start_date='2022-05-01',  # Before change
    end_date='2022-07-31'      # After change
)

# Behind the scenes:
# 1. Detects META had ticker change from FB
# 2. Queries FB data for 2022-05-01 to 2022-06-08
# 3. Queries META data for 2022-06-09 to 2022-07-31
# 4. Combines results seamlessly
```

**Supported ticker changes** (from `TICK_CHANGE_ALIAS`):
- FB → META (2022-06-09)
- GOOGL → GOOG (various dates)
- And more...

### 2. Flexible Data Resampling

Convert data to any timeframe:

```python
from dbase.DataAPI.ThetaData import retrieve_quote, resample

# Get 1-minute data
quotes_1m = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='1m'
)

# Resample to 15-minute bars
quotes_15m = resample(quotes_1m, interval='15m')

# Resample to hourly
quotes_1h = resample(quotes_1m, interval='1h')

# Resample to daily (business days)
quotes_1d = resample(quotes_1m, interval='1d')
```

**Supported intervals**: `1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `1d`, `1w`, `1M`, `1q`, `1y`

### 3. Proxy Support

Distribute API calls across multiple machines:

```python
from dbase.DataAPI.ThetaData.proxy import set_use_proxy, ping_proxy

# Enable proxy
set_use_proxy(True)

# Test connectivity
if ping_proxy():
    print("Proxy is reachable")
    
    # All subsequent calls use proxy automatically
    data = retrieve_eod_ohlc(...)
else:
    print("Proxy unavailable - using direct connection")
    set_use_proxy(False)
```

### 4. Latency Logging

Automatic request latency tracking:

```python
# Latency is logged automatically to CSV
data = retrieve_eod_ohlc(...)

# View logs
import pandas as pd
logs = pd.read_csv('.cache/theta_latency/latency_log.csv')

print(f"Mean latency: {logs['latency_seconds'].mean():.4f}s")
print(f"95th percentile: {logs['latency_seconds'].quantile(0.95):.4f}s")

# Find slow endpoints
slow = logs[logs['latency_seconds'] > 1.0]
print(f"Requests over 1s: {len(slow)}")
```

## Data Types & Endpoints

### EOD (End-of-Day) Data

```python
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc, retrieve_bulk_eod

# Single contract EOD
eod = retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Bulk EOD (all strikes/rights for expiration)
bulk_eod = retrieve_bulk_eod(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-12-01',
    end_date='2024-12-15'
)
```

**Returns**: DataFrame with columns `Open`, `High`, `Low`, `Close`, `Volume`

### Intraday OHLC Data

```python
from dbase.DataAPI.ThetaData import retrieve_ohlc

# Intraday OHLC with custom interval
ohlc = retrieve_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'  # 5-minute bars
)
```

**Supported intervals**: `1m`, `5m`, `10m`, `15m`, `30m`, `1h`

### Quote Data (Bid/Ask)

```python
from dbase.DataAPI.ThetaData import retrieve_quote, retrieve_quote_rt

# Historical intraday quotes
quotes = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'
)

# Realtime quote snapshot
rt_quote = retrieve_quote_rt(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0
)
```

**Returns**: DataFrame with `CloseBid`, `CloseAsk`, `Bid_size`, `Ask_size`, `Midpoint`, `Weighted_midpoint`

### Open Interest

```python
from dbase.DataAPI.ThetaData import retrieve_openInterest, retrieve_bulk_open_interest

# Single contract OI
oi = retrieve_openInterest(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Bulk OI (all contracts)
bulk_oi = retrieve_bulk_open_interest(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-12-01',
    end_date='2024-12-15'
)
```

**Returns**: DataFrame with `Open_interest` column

### Option Chain Snapshots

```python
from dbase.DataAPI.ThetaData import retrieve_chain_bulk

# Get entire option chain at specific time
chain = retrieve_chain_bulk(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-12-01',
    end_date='2024-12-15',
    end_time='16:00:00'  # Market close
)
```

**Returns**: Multi-indexed DataFrame with all strikes and rights

### Contract Listings

```python
from dbase.DataAPI.ThetaData import list_contracts, list_dates

# List all available contracts on a date
contracts = list_contracts(
    symbol='AAPL',
    date='2024-12-01'
)
print(contracts[['strike', 'right', 'expiration']])

# List available trading dates for a contract (V3 only)
dates = list_dates(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0
)
print(dates)  # Array of dates
```

## Usage Examples

### Example 1: Analyze Option Price vs Underlying

```python
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc
import pandas as pd
import matplotlib.pyplot as plt

# Get option data
option_data = retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Get underlying data (SPX example - would need equity API)
# underlying_data = get_underlying_prices('AAPL', '2024-01-01', '2024-12-31')

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(option_data.index, option_data['Close'], label='Option Price')
ax1.set_ylabel('Option Price ($)')
ax1.legend()
ax1.grid(True)

# ax2.plot(underlying_data.index, underlying_data['Close'], label='AAPL Price', color='green')
ax2.set_ylabel('Stock Price ($)')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Example 2: Calculate Option Spreads

```python
from dbase.DataAPI.ThetaData import retrieve_quote

# Get quotes for both legs
call_180 = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'
)

call_190 = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=190.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='5m'
)

# Calculate spread midpoint
spread = pd.DataFrame({
    'long_leg': call_180['Midpoint'],
    'short_leg': call_190['Midpoint'],
    'spread_value': call_180['Midpoint'] - call_190['Midpoint']
})

print(spread.describe())
```

### Example 3: Build Volatility Surface

```python
from dbase.DataAPI.ThetaData import retrieve_bulk_eod
import numpy as np

# Get all contracts for multiple expirations
expirations = ['2024-12-20', '2025-01-17', '2025-02-21']
surface_data = []

for exp in expirations:
    bulk = retrieve_bulk_eod(
        symbol='AAPL',
        exp=exp,
        start_date='2024-12-01',
        end_date='2024-12-01'  # Single day
    )
    bulk['expiration'] = exp
    surface_data.append(bulk)

all_contracts = pd.concat(surface_data)

# Filter calls only and calculate metrics
calls = all_contracts[all_contracts['Right'] == 'C']
calls['days_to_exp'] = (pd.to_datetime(calls['expiration']) - pd.to_datetime('2024-12-01')).dt.days

# Now you can calculate implied volatility and build surface
# (requires additional pricing models not shown here)
```

### Example 4: Monitor Real-Time Spreads

```python
from dbase.DataAPI.ThetaData import retrieve_quote_rt
import time

def monitor_spread(symbol, exp, strike_long, strike_short, interval=5):
    """Monitor spread in real-time"""
    print(f"Monitoring {symbol} {exp} {strike_long}/{strike_short} call spread")
    
    while True:
        try:
            # Get realtime quotes
            long_quote = retrieve_quote_rt(
                symbol=symbol, exp=exp, right='C', strike=strike_long
            )
            short_quote = retrieve_quote_rt(
                symbol=symbol, exp=exp, right='C', strike=strike_short
            )
            
            # Calculate spread
            long_mid = long_quote['Midpoint'].iloc[0]
            short_mid = short_quote['Midpoint'].iloc[0]
            spread = long_mid - short_mid
            
            print(f"[{pd.Timestamp.now()}] Spread: ${spread:.2f} "
                  f"(Long: ${long_mid:.2f}, Short: ${short_mid:.2f})")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

# Run monitor
# monitor_spread('AAPL', '2024-12-20', 180.0, 190.0, interval=5)
```

## Advanced Features

### Custom Aggregation in Resampling

```python
from dbase.DataAPI.ThetaData import retrieve_quote, resample

quotes = retrieve_quote(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    interval='1m'
)

# Custom aggregation per column
custom_agg = {
    'closebid': 'mean',  # Average bid
    'closeask': 'mean',  # Average ask
    'volume': 'sum',     # Total volume
    'midpoint': 'last'   # Last midpoint
}

quotes_custom = resample(quotes, interval='15m', custom_agg_columns=custom_agg)
```

### Using Option Ticker Strings

```python
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

# Instead of individual parameters
data1 = retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Use opttick string (more concise)
data2 = retrieve_eod_ohlc(
    opttick='AAPL20241220C180',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Both return identical results
assert data1.equals(data2)
```

### Debugging with print_url

```python
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

# Print the actual API request URL
data = retrieve_eod_ohlc(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0,
    start_date='2024-12-01',
    end_date='2024-12-15',
    print_url=True  # Shows the URL
)

# Output:
# Request URL: http://localhost:25503/v3/option/history/eod?symbol=AAPL&expiration=20241220&strike=180.00&right=C&start_date=20241201&end_date=20241215
```

## Configuration

### ThetaDataV3Controls (V3 Only)

```python
from dbase.DataAPI.ThetaData.v3.vars import SETTINGS

# View current settings
print(f"Old formatting: {SETTINGS.use_old_formatting}")
print(f"EOD format: {SETTINGS.eod_format}")
print(f"Intraday format: {SETTINGS.intra_format}")

# Disable old formatting (use new V3 format)
SETTINGS.use_old_formatting = False
# Now columns will be lowercase: 'open', 'high', 'low', 'close'
# Instead of capitalized: 'Open', 'High', 'Low', 'Close'

# Customize date formats
SETTINGS.eod_format = "%Y%m%d"  # YYYYMMDD instead of YYYY-MM-DD
SETTINGS.intra_format = "%Y-%m-%d %H:%M"  # No seconds
```

### Environment Variables

```bash
# Select API version
export THETADATA_USE_V3="true"   # Use V3 (default)
export THETADATA_USE_V3="false"  # Use V2

# Configure proxy
export PROXY_URL="http://proxy.example.com:8080/api"

# Set cache location
export GEN_CACHE_PATH="/path/to/cache"
```

## Migration from V2 to V3

### 1. Update Environment Variable

```python
import os
os.environ["THETADATA_USE_V3"] = "true"
```

### 2. Update Port (if accessing directly)

- V2: `http://localhost:25510/v2/...`
- V3: `http://localhost:25503/v3/...`

### 3. New Functions Available

```python
# V3-only: Bulk realtime quotes
from dbase.DataAPI.ThetaData import retrieve_bulk_quote_rt

bulk_rt = retrieve_bulk_quote_rt(
    symbol='AAPL',
    exp='2024-12-20'
)

# V3-only: List available dates
from dbase.DataAPI.ThetaData import list_dates

dates = list_dates(
    symbol='AAPL',
    exp='2024-12-20',
    right='C',
    strike=180.0
)
```

### 4. Column Name Changes (if use_old_formatting=False)

```python
# V2 / V3 with old formatting
columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'CloseBid', 'CloseAsk']

# V3 with new formatting (SETTINGS.use_old_formatting = False)
columns = ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask']
```

### 5. Benefits of Migration

- ✅ Faster multi-threaded range fetching
- ✅ More consistent error messages
- ✅ Better ticker change handling
- ✅ New bulk realtime endpoint
- ✅ Date listing capability
- ✅ Improved data normalization

## Troubleshooting

### ThetaData Terminal Not Running

**Error**: `Connection refused` or `ThetaDataDisconnected`

**Solution**:
1. Start ThetaData Terminal application
2. Check it's listening on correct port:
   - V2: 25510
   - V3: 25503
3. Verify in Terminal settings

### No Data Returned

**Error**: `ThetaDataNotFound`

**Possible causes**:
- Contract doesn't exist on that date
- Date is a non-trading day (weekend/holiday)
- Symbol ticker changed (use current ticker)
- Strike price doesn't exist

**Solution**:
```python
# Check available contracts first
contracts = list_contracts(symbol='AAPL', date='2024-12-01')
print(contracts[['strike', 'right', 'expiration']])

# Then query available strikes
data = retrieve_eod_ohlc(
    symbol='AAPL',
    exp=contracts['expiration'].iloc[0],
    right=contracts['right'].iloc[0],
    strike=contracts['strike'].iloc[0],
    start_date='2024-12-01',
    end_date='2024-12-15'
)
```

### Rate Limit Errors

**Error**: `ThetaDataOSLimit`

**Solution**:
- Reduce request frequency
- Use bulk endpoints instead of looping
- Enable proxy for distributed requests
- Contact ThetaData for rate limit increase

### Proxy Connection Issues

**Error**: `Proxy not responding`

**Solution**:
```python
from dbase.DataAPI.ThetaData.proxy import ping_proxy, set_use_proxy

# Test proxy
if not ping_proxy():
    print("Proxy unreachable - disabling")
    set_use_proxy(False)
```

### Memory Issues with Large Queries

**Solution**:
- Use smaller date ranges and combine results
- Use EOD data instead of intraday when possible
- Use larger intervals (1h instead of 1m)
- Process data in chunks

## Performance Tips

### 1. Use Bulk Endpoints

```python
# ❌ Slow - Multiple individual queries
strikes = [170, 175, 180, 185, 190]
data = []
for strike in strikes:
    df = retrieve_eod_ohlc(symbol='AAPL', exp='2024-12-20', 
                            right='C', strike=strike, ...)
    data.append(df)

# ✅ Fast - Single bulk query
bulk_data = retrieve_bulk_eod(
    symbol='AAPL',
    exp='2024-12-20',
    start_date='2024-12-01',
    end_date='2024-12-15'
)
```

### 2. Use EOD Instead of Intraday

```python
# ❌ Slow - Intraday data is much larger
intraday = retrieve_ohlc(..., interval='1m')  # Huge dataset

# ✅ Fast - EOD data is much smaller
eod = retrieve_eod_ohlc(...)  # Small dataset

# If you need intraday, use larger intervals
intraday_hourly = retrieve_ohlc(..., interval='1h')  # Reasonable size
```

### 3. Enable Proxy for Large Jobs

```python
from dbase.DataAPI.ThetaData.proxy import set_use_proxy

# Distribute load across multiple machines
set_use_proxy(True)

# Now queries can be load-balanced
for exp in expirations:
    data = retrieve_bulk_eod(symbol='AAPL', exp=exp, ...)
```

### 4. Reuse Data

```python
# ❌ Re-fetch same data multiple times
data1 = retrieve_eod_ohlc(...)
# ... do analysis ...
data2 = retrieve_eod_ohlc(...)  # Same query again!

# ✅ Fetch once, reuse
data = retrieve_eod_ohlc(...)
# ... do analysis with data ...
# ... do more analysis with same data ...
```

### 5. Cache Results

```python
import pandas as pd
from pathlib import Path

cache_dir = Path('.cache')
cache_dir.mkdir(exist_ok=True)

# Check cache first
cache_file = cache_dir / 'aapl_20241220_eod.parquet'

if cache_file.exists():
    data = pd.read_parquet(cache_file)
else:
    data = retrieve_eod_ohlc(...)
    data.to_parquet(cache_file)
```

## API Reference

### Main Functions

#### `retrieve_eod_ohlc(symbol, exp, right, strike, start_date, end_date, **kwargs)`
Get end-of-day OHLC data for a specific option contract.

#### `retrieve_ohlc(symbol, exp, right, strike, start_date, end_date, interval, **kwargs)`
Get intraday OHLC data with custom interval.

#### `retrieve_quote(symbol, exp, right, strike, start_date, end_date, interval, **kwargs)`
Get historical quote data (bid/ask) with custom interval.

#### `retrieve_quote_rt(symbol, exp, right, strike, **kwargs)`
Get realtime quote snapshot.

#### `retrieve_bulk_eod(symbol, exp, start_date, end_date, **kwargs)`
Get EOD data for all contracts in an expiration.

#### `retrieve_openInterest(symbol, exp, right, strike, start_date, end_date, **kwargs)`
Get historical open interest data.

#### `retrieve_bulk_open_interest(symbol, exp, start_date, end_date, **kwargs)`
Get open interest for all contracts in an expiration.

#### `retrieve_chain_bulk(symbol, exp, start_date, end_date, end_time, **kwargs)`
Get complete option chain snapshot at specific time.

#### `list_contracts(symbol, date, **kwargs)`
List all available option contracts for a symbol on a date.

#### `list_dates(symbol, exp, right, strike, **kwargs)` (V3 only)
List all available trading dates for a specific contract.

### Utility Functions

#### `resample(data, interval, custom_agg_columns=None, method='ffill', **kwargs)`
Resample data to different timeframe.

#### `bootstrap_ohlc(data, copy_column='Midpoint')`
Fill missing OHLC columns by copying from another column.

### Proxy Functions

#### `set_use_proxy(use_proxy: bool)`
Enable or disable proxy usage.

#### `ping_proxy() -> bool`
Test if proxy server is reachable.

#### `get_proxy_url() -> str | None`
Get currently configured proxy URL.

---

## Support & Resources

- **ThetaData Website**: https://thetadata.net
- **API Documentation**: https://http-docs.thetadata.us/
- **Terminal Download**: https://thetadata.net/download

For issues with this client, check the module docstrings or source code comments.

---

**Version**: 3.0  
**Last Updated**: January 2026  
**Maintainer**: Finance Database Team
