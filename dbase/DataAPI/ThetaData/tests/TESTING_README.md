# Testing V3 Without ThetaData Terminal

## Overview

This testing infrastructure allows you to test V3 backward compatibility **without the ThetaData Terminal running**. It uses dry-run mode and mock responses.

---

## Quick Start

### Option 1: Quick Test Script (Recommended)

```bash
cd /Users/chiemelienwanisobi/cloned_repos/FinanceDatabase/dbase/DataAPI/ThetaData
python test_v3_quick.py
```

This will:
- Run 8 compatibility tests
- Show which functions are compatible with V2
- Identify breaking changes
- No ThetaData Terminal required

### Option 2: Pytest Suite

```bash
cd /Users/chiemelienwanisobi/cloned_repos/FinanceDatabase
pytest dbase/DataAPI/ThetaData/tests/test_v3_compatibility.py -v
```

### Option 3: Interactive Python

```python
import os
os.environ['THETADATA_DRY_RUN'] = 'true'
os.environ['THETADATA_USE_V3'] = 'true'

from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

# This will not make actual API call
result = retrieve_eod_ohlc(
    "AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0
)

print(result.columns)  # Check returned columns
print(result.head())   # Check mock data
```

---

## How It Works

### Dry-Run Mode

When `THETADATA_DRY_RUN=true`:
1. Functions accept parameters normally
2. Parameters are validated and logged
3. Mock data is returned instead of making API calls
4. No ThetaData Terminal needed

### Mock Responses

Located in `tests/mock_responses.py`:
- Realistic CSV responses for all endpoint types
- Based on actual ThetaData API responses
- Covers EOD, quote, OHLC, open interest, etc.

---

## Testing Strategies

### 1. Test Parameter Acceptance

**Goal:** Verify V3 accepts all V2 parameters

```python
from dbase.DataAPI.ThetaData.tests.dry_run import enable_capture_mode, get_captured_calls

enable_capture_mode()

# Call function with V2 signature
retrieve_eod_ohlc(
    "AAPL",           # Works?
    "2024-12-31",     # Correct order?
    "2024-12-20",
    "C",
    "2024-01-01",
    180.0,
    proxy="http://localhost:8080"  # Accepted?
)

# Check what was captured
calls = get_captured_calls()
print(calls[0]['params'])  # Inspect parameters sent
```

### 2. Test Return Structure

**Goal:** Verify V3 returns same columns as V2

```python
result = retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)

# Check columns
expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk',
                   'Midpoint', 'Weighted_midpoint']

for col in expected_columns:
    assert col in result.columns, f"Missing: {col}"

# Check index
assert result.index.name == 'Datetime'
```

### 3. Test Positional Arguments

**Goal:** Verify parameter order matches V2

```python
# V2 positional call
result = retrieve_eod_ohlc(
    "AAPL",           # symbol
    "2024-12-31",     # end_date
    "2024-12-20",     # exp
    "C",              # right
    "2024-01-01",     # start_date
    180.0             # strike
)

# If this doesn't raise TypeError, order is correct
assert result is not None
```

### 4. Capture and Inspect Calls

**Goal:** See exactly what would be sent to API

```python
from dbase.DataAPI.ThetaData.tests.dry_run import (
    enable_capture_mode, 
    get_captured_calls, 
    print_dry_run_summary,
    clear_captured_calls
)

enable_capture_mode()
clear_captured_calls()

# Make some calls
retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)
retrieve_quote("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)

# Print summary
print_dry_run_summary()
```

Output:
```
================================================================================
DRY RUN SUMMARY - 2 calls captured
================================================================================

Call 1:
  Endpoint Type: eod_ohlc
  URL: http://localhost:25503/v3/option/history/eod
  Params: {
      "symbol": "AAPL",
      "start_date": "20240101",
      "end_date": "20241231",
      "expiration": "20241220",
      "strike": "180.00",
      "right": "C"
  }

Call 2:
  Endpoint Type: quote
  URL: http://localhost:25503/v3/option/history/quote
  ...
```

---

## Test Before vs After Fixes

### Before Fix (Current State)

```bash
python test_v3_quick.py
```

**Expected Output:**
```
❌ FAIL: retrieve_ohlc - Missing quote columns
❌ FAIL: retrieve_openInterest - Datetime is index, not column
❌ FAIL: retrieve_quote - proxy parameter causes TypeError
...
Tests Failed: 5/8
```

### After Fix (Target State)

```bash
python test_v3_quick.py
```

**Expected Output:**
```
✅ PASS: retrieve_ohlc - includes quote data columns
✅ PASS: retrieve_openInterest - Datetime as column
✅ PASS: retrieve_quote - accepts proxy parameter
...
Tests Passed: 8/8
✅ ALL TESTS PASSED - V3 is backward compatible!
```

---

## Advanced Testing

### Mock Specific Responses

```python
from dbase.DataAPI.ThetaData.tests.mock_responses import get_mock_response

# Get mock data for specific endpoint
eod_data = get_mock_response('eod_ohlc', symbol='AAPL')
print(eod_data)

# Customize for your test
custom_response = """timestamp,open,high,low,close,volume
20240101,100.0,101.0,99.0,100.5,1000"""
```

### Test with unittest.mock

```python
from unittest import mock
import pandas as pd

MOCK_RESPONSE = """timestamp,open,high,low,close,volume,bid,ask,bid_size,ask_size
20240101,100.0,105.0,99.0,104.0,1000,103.5,104.5,10,15"""

with mock.patch('requests.get') as mock_get:
    mock_get.return_value.text = MOCK_RESPONSE
    mock_get.return_value.status_code = 200
    
    # Now call function
    result = retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)
    
    # Verify it was called correctly
    assert mock_get.called
    call_args = mock_get.call_args
    print(f"Called with: {call_args}")
```

### Parameter Validation Only

```python
from dbase.DataAPI.ThetaData.tests.dry_run import validate_parameters

# Check if parameters are valid before calling
errors = validate_parameters(
    func_name='retrieve_eod_ohlc',
    args=('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0),
    kwargs={'proxy': 'http://localhost'}
)

if errors:
    print(f"Invalid parameters: {errors}")
else:
    print("Parameters are valid")
```

---

## Troubleshooting

### Issue: ImportError when running tests

**Solution:** Make sure you're in the right directory

```bash
cd /Users/chiemelienwanisobi/cloned_repos/FinanceDatabase
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python dbase/DataAPI/ThetaData/test_v3_quick.py
```

### Issue: Tests pass but real API calls fail

**Solution:** The issue is in actual implementation, not parameters

1. Check the TODO file for implementation tasks
2. Run with real ThetaData Terminal to see actual errors
3. Compare real response vs mock response

### Issue: Want to see what would be sent to API

**Solution:** Use capture mode

```python
from dbase.DataAPI.ThetaData.tests.dry_run import enable_capture_mode, print_dry_run_summary

enable_capture_mode()

# Make calls
retrieve_eod_ohlc(...)

# See summary
print_dry_run_summary()
```

---

## Next Steps

1. **Run quick test:** `python test_v3_quick.py`
2. **Identify failures:** Note which tests fail
3. **Check TODO:** See `V3_COMPATIBILITY_TODO.md` for implementation plan
4. **Fix issues:** Implement fixes from TODO
5. **Re-test:** Run quick test again to verify
6. **Full test:** Run pytest suite

---

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `THETADATA_DRY_RUN` | `true`/`false` | Enable dry-run mode (no API calls) |
| `THETADATA_USE_V3` | `true`/`false` | Use V3 instead of V2 |
| `PYTHONPATH` | path | Add workspace to Python path |

---

## Files Reference

| File | Purpose |
|------|---------|
| `test_v3_quick.py` | Quick compatibility test script |
| `tests/test_v3_compatibility.py` | Full pytest suite |
| `tests/dry_run.py` | Dry-run mode implementation |
| `tests/mock_responses.py` | Mock API responses |
| `V3_COMPATIBILITY_TODO.md` | Implementation plan |
| `utils.py` | Modified to support dry-run mode |

---

## Contact

For questions or issues with testing:
- Check `V3_COMPATIBILITY_TODO.md` for implementation details
- Run `python test_v3_quick.py` to identify specific failures
- Use dry-run mode to inspect parameters without API calls
