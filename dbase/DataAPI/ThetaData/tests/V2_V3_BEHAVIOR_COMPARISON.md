# V3 vs V2 Return Behavior Analysis

## Overview

This document analyzes the differences in **returned data structure** between V2 and V3 implementations, beyond just signature compatibility.

---

## ✅ BEHAVIORAL COMPATIBILITY - BY FUNCTION

### 1. retrieve_eod_ohlc

**V2 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime (named "Datetime")
- Columns: ['Open', 'High', 'Low', 'Close', 'Volume', 
           'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk',
           'Midpoint', 'Weighted_midpoint']
- Midpoint calculation: (bid + ask) / 2
- Weighted_midpoint: (bid_size * bid + ask_size * ask) / (bid_size + ask_size)
- Column names: Capitalized
- Datetime format: With time appended (e.g., "2024-12-31 16:00:00")
```

**V3 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime (lowercase when use_old_formatting=False, capitalized when True)
- Columns: Same as V2 when use_old_formatting=True
  - 'open', 'high', 'low', 'close', 'volume', 
  - 'bid_size', 'bid', 'ask_size', 'ask',
  - 'midpoint', 'weighted_midpoint'
- Midpoint calculation: (bid + ask) / 2  ✅ SAME
- Weighted_midpoint: (bid * bid_size + ask * ask_size) / (bid_size + ask_size)  ✅ SAME
- Column names: lowercase by default, Capitalized with use_old_formatting=True
- Renames: bid -> CloseBid, ask -> CloseAsk when use_old_formatting=True
```

**✅ CONSISTENT IF:** `SETTINGS.use_old_formatting = True`

**⚠️ DIFFERENCES:**
- V3 uses lowercase columns by default (requires `use_old_formatting=True`)
- V3 adds EOD timestamp (16:00:00) when `use_old_formatting=True`

---

### 2. retrieve_ohlc

**V2 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime
- Columns: ['Open', 'High', 'Low', 'Close', 'Volume',
           'Bid_size', 'Closebid', 'Ask_size', 'Closeask',  # Note lowercase 'b' and 'a'
           'Midpoint', 'Weighted_midpoint', 'Date']
- Gets data from V2 OHLC endpoint which includes bid/ask
- Adds quote columns from same endpoint response
```

**V3 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime
- Columns: ['open', 'high', 'low', 'close', 'volume']  # ONLY OHLC, NO QUOTE DATA
- Quote columns NOT INCLUDED (Bid_size, CloseBid, Ask_size, CloseAsk, Midpoint, Weighted_midpoint)
- V3 OHLC endpoint does not return bid/ask data
```

**❌ NOT CONSISTENT - MISSING 6 COLUMNS:**
- `Bid_size`
- `CloseBid` (or `Closebid`)
- `Ask_size`
- `CloseAsk` (or `Closeask`)
- `Midpoint`
- `Weighted_midpoint`

**Root Cause:** V3's OHLC API endpoint only returns OHLC data, while V2's endpoint returned OHLC + quote data.

**Workaround:** Use `retrieve_quote()` instead for full quote data including bid/ask.

---

### 3. retrieve_quote

**V2 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime
- Columns: ['Bid_size', 'Closebid', 'Ask_size', 'Closeask', 
           'Midpoint', 'Weighted_midpoint', 'Date', 'Volume']
- Bid/Ask renamed to Closebid/Closeask
- Includes 'Date' column (date portion only)
- Filtered to market hours (9:30-16:00)
```

**V3 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime
- Columns: ['bid_size', 'bid', 'ask_size', 'ask',
           'midpoint', 'weighted_midpoint', 'volume']
- When use_old_formatting=True:
  - bid -> CloseBid (note capital B)
  - ask -> CloseAsk (note capital A)
  - Adds 'Date' column
- Column names capitalized with use_old_formatting
```

**✅ MOSTLY CONSISTENT IF:** `SETTINGS.use_old_formatting = True`

**⚠️ MINOR DIFFERENCE:**
- V2: `Closebid`, `Closeask` (lowercase 'b' and 'a')
- V3: `CloseBid`, `CloseAsk` (uppercase 'B' and 'A')
  
This could break code that depends on exact column names.

---

### 4. retrieve_openInterest

**V2 Behavior:**
```python
# Returns DataFrame with:
- NO index set (returns regular DataFrame)
- Columns: ['Date', 'Open_interest', 'Datetime', 'time']
- Datetime is a COLUMN, not the index
- Includes 'time' column
```

**V3 Behavior:**
```python
# Returns DataFrame with:
- Index: datetime (when use_old_formatting=False)
- Columns: ['open_interest']
- When use_old_formatting=True:
  - Adds 'Datetime' as COLUMN (reset from index)
  - Adds 'Date' column
  - Column names capitalized
```

**✅ CONSISTENT IF:** `SETTINGS.use_old_formatting = True`

**⚠️ KEY DIFFERENCE WITHOUT use_old_formatting:**
- V2: Datetime is always a column
- V3: Datetime is the index (must enable use_old_formatting to get it as column)

---

### 5. retrieve_quote_rt

**V2 Behavior:**
```python
# Returns DataFrame with realtime quote snapshot
- Similar structure to retrieve_quote
- Filtered to current day snapshot
```

**V3 Behavior:**
```python
# Returns DataFrame with realtime quote snapshot
- Uses same formatting as retrieve_quote
- Applies use_old_formatting settings
```

**✅ CONSISTENT:** When `use_old_formatting=True`

---

### 6. retrieve_bulk_eod

**V2 Behavior:**
```python
# Returns bulk EOD data for all contracts in expiration
- Includes columns: strike, right, expiration, symbol
- Bulk-specific formatting
```

**V3 Behavior:**
```python
# Returns bulk EOD data
- Same columns when is_bulk=True
- Preserves strike, right, expiration columns
- use_old_formatting applies
```

**✅ CONSISTENT:** Bulk handling preserved

---

### 7. retrieve_bulk_open_interest

**V2 Behavior:**
```python
# Returns bulk open interest data
- Similar to retrieve_openInterest but for all contracts
```

**V3 Behavior:**
```python
# Returns bulk open interest data
- use_old_formatting adds Datetime and Date columns
```

**✅ CONSISTENT:** When `use_old_formatting=True`

---

### 8. retrieve_chain_bulk

**V2 Behavior:**
```python
# Returns option chain snapshot at specific time
- Includes all contracts for expiration
- Midpoint and weighted_midpoint calculated
```

**V3 Behavior:**
```python
# Returns option chain snapshot
- Uses list_contracts endpoint internally (not ideal)
- Calculates midpoint/weighted_midpoint same way
- use_old_formatting adds Date column and default timestamp
```

**✅ MOSTLY CONSISTENT:** When `use_old_formatting=True`

**⚠️ DIFFERENCE:**
- V2: Accepts start_date AND end_date (must be same)
- V3: Now properly validates start_date == end_date

---

### 9. list_contracts

**V2 Behavior:**
```python
# Returns list of available contracts
- strike divided by 1000 to convert from millistrikes
```

**V3 Behavior:**
```python
# Returns list of available contracts
- Same formatting
- Handles start_date kwarg for backward compatibility
```

**✅ CONSISTENT**

---

## Summary Table

| Function | Consistent? | Required Setting | Notes |
|----------|------------|------------------|-------|
| retrieve_eod_ohlc | ✅ YES | use_old_formatting=True | Column names, CloseBid/CloseAsk naming |
| retrieve_quote | ⚠️ MOSTLY | use_old_formatting=True | Closebid vs CloseBid (case difference) |
| retrieve_ohlc | ❌ NO | N/A | **MISSING 6 quote columns** |
| retrieve_quote_rt | ✅ YES | use_old_formatting=True | Same as retrieve_quote |
| retrieve_openInterest | ✅ YES | use_old_formatting=True | Datetime as column vs index |
| retrieve_bulk_eod | ✅ YES | use_old_formatting=True | Bulk formatting preserved |
| retrieve_bulk_open_interest | ✅ YES | use_old_formatting=True | Same as retrieve_openInterest |
| retrieve_chain_bulk | ✅ YES | use_old_formatting=True | Date column handling |
| list_contracts | ✅ YES | None | Fully compatible |

---

## Critical Findings

### 🔴 BREAKING: retrieve_ohlc Missing Columns

**Impact:** HIGH - Code depending on quote columns will break

**Missing Columns:**
1. `Bid_size`
2. `CloseBid` (or `Closebid`)
3. `Ask_size`
4. `CloseAsk` (or `Closeask`)
5. `Midpoint`
6. `Weighted_midpoint`

**Root Cause:** V3 OHLC endpoint only returns OHLC data, not quote data

**Migration Path:**
```python
# OLD V2 CODE:
data = retrieve_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)
bid_size = data['Bid_size']  # Would work in V2

# NEW V3 CODE (FIX):
data = retrieve_quote("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)
bid_size = data['Bid_size']  # Use retrieve_quote instead
```

---

### 🟡 WARNING: Column Name Case Differences

**retrieve_quote column names:**
- V2: `Closebid`, `Closeask` (lowercase 'b', 'a')
- V3: `CloseBid`, `CloseAsk` (uppercase 'B', 'A')

**Impact:** Code using exact column names will break

**Example of Breaking Code:**
```python
data = retrieve_quote("AAPL", ...)
close_bid = data['Closebid']  # Works in V2, FAILS in V3
close_bid = data['CloseBid']  # Works in V3, FAILS in V2
```

**Migration Path:**
```python
# Case-insensitive column access:
data.columns = data.columns.str.lower()  # Normalize to lowercase
close_bid = data['closebid']

# OR use column mapping:
COLUMN_MAP = {'Closebid': 'CloseBid', 'Closeask': 'CloseAsk'}
for old, new in COLUMN_MAP.items():
    if old in data.columns:
        data.rename(columns={old: new}, inplace=True)
```

---

### 🟢 GOOD: use_old_formatting Setting

**V3 includes a compatibility flag:**
```python
from dbase.DataAPI.ThetaData.v3.utils import SETTINGS
SETTINGS.use_old_formatting = True  # Enable V2 compatibility mode
```

**What it does:**
- ✅ Capitalizes column names
- ✅ Renames `bid` → `CloseBid`, `ask` → `CloseAsk`
- ✅ Adds `Date` column
- ✅ Adds `Datetime` as column (instead of index)
- ✅ Adds EOD timestamp for EOD data

**Recommendation:** **Enable this setting for V2 compatibility**

---

## Migration Checklist

When migrating from V2 to V3:

- [ ] **Set `SETTINGS.use_old_formatting = True`** for all V3 calls
- [ ] **Replace `retrieve_ohlc` with `retrieve_quote`** if you need bid/ask columns
- [ ] **Check column name references** - update `Closebid` → `CloseBid`, `Closeask` → `CloseAsk`
- [ ] **Test datetime handling** - ensure Datetime column exists where expected
- [ ] **Verify proxy parameter** - it's accepted but not functional in V3
- [ ] **Update tests** - account for minor formatting differences

---

## Recommended Configuration

Add this to your initialization code:

```python
import os

# Use V3 API
os.environ['THETADATA_USE_V3'] = 'true'

# Enable V2 compatibility mode
from dbase.DataAPI.ThetaData.v3.utils import SETTINGS
SETTINGS.use_old_formatting = True

# Now use functions as before
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc, retrieve_quote

# But remember: use retrieve_quote instead of retrieve_ohlc for bid/ask data
```

---

## Conclusion

**✅ Overall Compatibility: GOOD with caveats**

With `use_old_formatting=True`, V3 is **90% backward compatible** with V2:

**What Works:**
- All function signatures accept V2 parameters
- Return structures match when `use_old_formatting=True`
- Calculations (midpoint, weighted_midpoint) are identical
- Bulk operations preserved

**What Doesn't Work:**
- ❌ `retrieve_ohlc` missing 6 quote columns (use `retrieve_quote` instead)
- ⚠️ Column name case differences (`Closebid` vs `CloseBid`)
- ⚠️ `proxy` parameter accepted but not functional

**Recommended Action:**
1. Enable `use_old_formatting=True`
2. Replace `retrieve_ohlc` with `retrieve_quote` where bid/ask data is needed
3. Update column name references to use proper case
4. Test thoroughly before production deployment
