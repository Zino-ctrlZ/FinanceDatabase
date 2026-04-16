# V3 Backward Compatibility Implementation Plan

## Overview
This document outlines the step-by-step plan to make V3 fully backward compatible with V2, ensuring no breaking changes when switching between versions.

---

## Phase 1: Setup Testing Infrastructure (Priority: CRITICAL)

### 1.1 Create Mock Testing Framework
**File:** `dbase/DataAPI/ThetaData/tests/test_v3_compatibility.py`

**Tasks:**
- [ ] Create `@mock.patch` decorators for `requests.get` and `requests.post`
- [ ] Create fixture responses for all endpoint types (EOD, quote, OHLC, OI, etc.)
- [ ] Create parameter capture decorator to inspect what's sent to API
- [ ] Create signature comparison tests (V2 vs V3 parameter compatibility)

### 1.2 Add Dry-Run Mode
**File:** `dbase/DataAPI/ThetaData/utils.py`

**Tasks:**
- [ ] Add `DRY_RUN_MODE` environment variable support
- [ ] Modify `_fetch_data()` to return mock data when `DRY_RUN_MODE=true`
- [ ] Log all parameters that would be sent to API
- [ ] Return sample CSV data for each endpoint type

**Implementation:**
```python
def _fetch_data(theta_url: str, params: dict, print_url: bool = False, dry_run: bool = None) -> str:
    """Fetch data from ThetaData API, using proxy if available."""
    
    # Check dry run mode
    if dry_run is None:
        dry_run = os.environ.get("THETADATA_DRY_RUN", "false").lower() == "true"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would call: {theta_url}")
        logger.info(f"[DRY RUN] With params: {json.dumps(params, indent=2)}")
        return _get_mock_response(theta_url, params)
    
    # Normal execution...
```

### 1.3 Create Parameter Inspector
**File:** `dbase/DataAPI/ThetaData/tests/inspect_params.py`

**Tasks:**
- [ ] Create context manager to capture function calls
- [ ] Inspect all kwargs passed to V3 functions
- [ ] Compare against V2 expected parameters
- [ ] Generate compatibility report

**Usage:**
```python
from dbase.DataAPI.ThetaData.tests.inspect_params import ParamInspector

with ParamInspector() as inspector:
    retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)
    
print(inspector.get_report())  # Shows all captured params
```

---

## Phase 2: Critical Fixes (Priority: P0 - BLOCKING)

### 2.1 Add Proxy Support to V3
**Files:** 
- `v3/endpoints.py` (all functions)
- `v3/utils.py` (`_fetch_data`, `_multi_threaded_range_fetch`)

**Tasks:**
- [ ] Add `proxy: str = None` parameter to all 9 exported functions
- [ ] Modify `_fetch_data()` to accept and use proxy parameter
- [ ] Update `_multi_threaded_range_fetch()` to pass proxy through
- [ ] Test proxy mode with mock server

**Affected Functions:**
1. `_retrieve_quote_rt`
2. `_retrieve_bulk_quote_rt`
3. `_retrieve_quote`
4. `_retrieve_ohlc`
5. `_retrieve_eod_ohlc`
6. `_retrieve_bulk_eod`
7. `_retrieve_openInterest`
8. `_retrieve_bulk_open_interest`
9. `_retrieve_chain_bulk`
10. `_list_contracts`

**Template:**
```python
def _retrieve_eod_ohlc(
    symbol: str = None,
    end_date: str = None,  # REORDERED
    exp: str = None,
    right: str = None,
    start_date: str = None,  # REORDERED
    strike: float = None,
    print_url: bool = False,
    rt: bool = True,  # ADDED
    proxy: str = None,  # ADDED
    *,
    opttick: str = None,
    **kwargs,
) -> pd.DataFrame:
```

### 2.2 Fix retrieve_ohlc - Add Quote Data
**File:** `v3/endpoints.py` (line ~1180)

**Tasks:**
- [ ] After OHLC fetch, call `_retrieve_quote` with same parameters
- [ ] Merge quote data (Bid_size, CloseBid, Ask_size, CloseAsk, Midpoint, Weighted_midpoint)
- [ ] Match V2 behavior exactly
- [ ] Add tests for column existence

**Implementation:**
```python
def _raw_retrieve_ohlc(...):
    # Fetch OHLC data
    ohlc_data = _multi_threaded_range_fetch(...)
    
    # Fetch quote data to merge
    quote_data = _multi_threaded_range_fetch(
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
    
    # Extract quote columns only
    quote_cols = ['Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
    quote_data = quote_data[quote_cols]
    
    # Merge on datetime index
    merged_data = ohlc_data.merge(quote_data, left_index=True, right_index=True, how='left')
    
    return merged_data
```

### 2.3 Fix retrieve_openInterest - Return Structure
**File:** `v3/endpoints.py` (line ~620)

**Tasks:**
- [ ] Change from returning Datetime as index to returning as column
- [ ] Match V2 structure: columns = [Open_interest, Date, time, Datetime]
- [ ] Update `_new_dataframe_formatting` to handle this case
- [ ] Add flag to preserve column vs index behavior

**Implementation:**
```python
def _inner_retrieve_openInterest(...):
    data = _raw_retrieve_openInterest(...)
    data = _new_dataframe_formatting(df=data, interval="1d", is_bulk=False, preserve_datetime_column=True)
    
    # V2 compatibility: Datetime as column, not index
    if data.index.name == 'Datetime':
        data = data.reset_index()
    
    # Add Date and time columns
    data['Date'] = data['Datetime'].dt.date
    data['time'] = data['Datetime'].dt.strftime('%H:%M:%S')
    
    return data
```

### 2.4 Fix Parameter Order for All Functions
**File:** `v3/endpoints.py`

**Tasks:**
- [ ] Reorder parameters in all functions to match V2 exactly
- [ ] Create compatibility tests to verify positional args work
- [ ] Document the changes

**V2 Parameter Orders to Match:**

| Function | V2 Order |
|----------|----------|
| `retrieve_quote` | `symbol, end_date, exp, right, start_date, strike` |
| `retrieve_ohlc` | `symbol, end_date, exp, right, start_date, strike` |
| `retrieve_eod_ohlc` | `symbol, end_date, exp, right, start_date, strike` |
| `retrieve_openInterest` | `symbol, end_date, exp, right, start_date, strike` |

---

## Phase 3: High Priority Fixes (Priority: P1)

### 3.1 Add Missing Parameters to retrieve_quote_rt
**File:** `v3/endpoints.py` (line ~395)

**Tasks:**
- [ ] Add `start_time: str = PRICING_CONFIG["MARKET_OPEN_TIME"]`
- [ ] Add `end_time: str = PRICING_CONFIG["MARKET_CLOSE_TIME"]`
- [ ] Add `ts: bool = False` (switches endpoint)
- [ ] Add `start_date: str = None`
- [ ] Add `end_date: str = None`
- [ ] Implement filtering logic based on these params

### 3.2 Add Missing Parameters to retrieve_quote
**File:** `v3/endpoints.py` (line ~1085)

**Tasks:**
- [ ] Add `start_time: str = None`
- [ ] Add `end_time: str = PRICING_CONFIG["MARKET_CLOSE_TIME"]`
- [ ] Add `ohlc_format: bool = True`
- [ ] Set `interval: str = "30m"` as default (not None)
- [ ] Conditionally call `bootstrap_ohlc()` based on `ohlc_format`

### 3.3 Add Missing Parameters to retrieve_ohlc
**File:** `v3/endpoints.py` (line ~1180)

**Tasks:**
- [ ] Add `start_time: str = PRICING_CONFIG["MARKET_OPEN_TIME"]`
- [ ] Remove `interval` parameter (not in V2)
- [ ] Implement time filtering

### 3.4 Add Missing Parameters to retrieve_eod_ohlc
**File:** `v3/endpoints.py` (line ~800)

**Tasks:**
- [ ] Add `print_url: bool = False` parameter
- [ ] Add `rt: bool = True` parameter
- [ ] Implement timestamp formatting based on `rt` flag

### 3.5 Fix list_contracts Parameter
**File:** `v3/endpoints.py` (line ~930)

**Tasks:**
- [ ] Change signature: `def _list_contracts(symbol: str, start_date: str, ...)`
- [ ] Remove fragile kwargs.pop() logic
- [ ] Keep `date` as internal variable: `date = start_date`

### 3.6 Fix retrieve_chain_bulk Date Range
**File:** `v3/endpoints.py` (line ~1000)

**Tasks:**
- [ ] Remove assertion `assert start_date == end_date`
- [ ] Support date ranges (even though typically equal)
- [ ] Use `start_date` as the query date if equal

### 3.7 Fix retrieve_bulk_eod Type Annotation
**File:** `v3/endpoints.py` (line ~869)

**Tasks:**
- [ ] Change `print_url: str = None` to `print_url: bool = False`

---

## Phase 4: Consistency Fixes (Priority: P2)

### 4.1 Standardize Column Names
**File:** `v3/utils.py` (`_new_dataframe_formatting`)

**Tasks:**
- [ ] Always use lowercase 'b', 'a': `Closebid`, `Closeask` (match V2)
- [ ] OR set `SETTINGS.use_old_formatting = True` by default
- [ ] Add tests for column name consistency

### 4.2 Add Date/time Columns
**File:** `v3/utils.py` (`_new_dataframe_formatting`)

**Tasks:**
- [ ] Always add `Date` column (not just with use_old_formatting)
- [ ] Add `time` column to quote results
- [ ] Match V2 format exactly

### 4.3 Implement ohlc_format Parameter
**File:** `v3/endpoints.py` (`_retrieve_quote`)

**Tasks:**
- [ ] Add conditional logic:
  ```python
  if ohlc_format:
      data = bootstrap_ohlc(data)
  ```
- [ ] Test both True and False cases

### 4.4 Implement rt Parameter
**File:** `v3/endpoints.py` (`_retrieve_eod_ohlc`)

**Tasks:**
- [ ] Add conditional timestamp formatting:
  ```python
  if rt:
      data.index = add_eod_timestamp(data.index)
  else:
      # Date only, no time component
      data.index = data.index.normalize()
  ```

---

## Phase 5: Testing & Validation (Priority: P1)

### 5.1 Unit Tests for Each Function
**File:** `tests/test_v3_compatibility.py`

**Tasks:**
- [ ] Test parameter acceptance (all V2 params accepted)
- [ ] Test parameter order (positional calls work)
- [ ] Test return structure (columns match V2)
- [ ] Test proxy parameter (passed correctly)
- [ ] Test kwargs handling (depth, ohlc_format, rt ignored gracefully)

### 5.2 Integration Tests
**File:** `tests/test_v3_integration.py`

**Tasks:**
- [ ] Mock full request/response cycle
- [ ] Test ticker change handling
- [ ] Test date range splitting
- [ ] Test multi-threaded fetching
- [ ] Compare V2 vs V3 output for same inputs

### 5.3 Regression Tests
**File:** `tests/test_v2_v3_parity.py`

**Tasks:**
- [ ] For each function, call with V2 signature
- [ ] Verify V3 accepts same call
- [ ] Compare output structure
- [ ] Flag any differences

**Test Template:**
```python
def test_retrieve_eod_ohlc_parity():
    """Test V2 and V3 have same signature and output structure"""
    
    # V2 style positional call
    v2_params = ("AAPL", "2024-12-31", "2024-12-20", "C", "2024-01-01", 180.0)
    
    with mock.patch('requests.get') as mock_get:
        mock_get.return_value.text = MOCK_EOD_RESPONSE
        
        # Should accept same positional params
        result = _retrieve_eod_ohlc(*v2_params)
        
        # Check structure
        assert 'Open' in result.columns
        assert 'High' in result.columns
        assert 'Midpoint' in result.columns
        assert result.index.name == 'Datetime'
```

### 5.4 Dry-Run Testing
**File:** `tests/test_dry_run.py`

**Tasks:**
- [ ] Set `THETADATA_DRY_RUN=true`
- [ ] Call all functions with various params
- [ ] Verify parameters are formatted correctly
- [ ] Check no actual requests are made

---

## Phase 6: Documentation (Priority: P2)

### 6.1 Update Migration Guide
**File:** `dbase/DataAPI/ThetaData/MIGRATION_V2_TO_V3.md`

**Tasks:**
- [ ] Document all changes made
- [ ] Provide before/after examples
- [ ] List any remaining differences
- [ ] Update version compatibility matrix

### 6.2 Update Docstrings
**Files:** All modified functions

**Tasks:**
- [ ] Update parameter descriptions
- [ ] Add "V2 Compatible" notes
- [ ] Document any behavioral differences
- [ ] Add examples

### 6.3 Update __init__.py Documentation
**File:** `dbase/DataAPI/ThetaData/__init__.py`

**Tasks:**
- [ ] Update "Key Differences: V2 vs V3" section
- [ ] Remove incompatibility warnings
- [ ] Add "Fully backward compatible as of vX.X.X"

---

## Testing Strategy Without ThetaData Terminal

### Option 1: Dry-Run Mode (RECOMMENDED)
```bash
export THETADATA_DRY_RUN=true
python -c "
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc
result = retrieve_eod_ohlc('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0)
print('Success! Parameters accepted.')
"
```

### Option 2: Mock Responses
```python
import mock
from dbase.DataAPI.ThetaData import retrieve_eod_ohlc

MOCK_RESPONSE = """timestamp,open,high,low,close,volume,bid,ask,bid_size,ask_size
20240101,100.0,105.0,99.0,104.0,1000,103.5,104.5,10,15"""

with mock.patch('requests.get') as mock_get:
    mock_get.return_value.text = MOCK_RESPONSE
    mock_get.return_value.status_code = 200
    
    result = retrieve_eod_ohlc('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0)
    print(result)
```

### Option 3: Parameter Validator
```python
# Create validator that checks params without making requests
from dbase.DataAPI.ThetaData.tests.validators import validate_params

errors = validate_params(
    function='retrieve_eod_ohlc',
    args=('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0),
    kwargs={'proxy': 'http://localhost:8080'}
)

if errors:
    print(f"Validation errors: {errors}")
else:
    print("Parameters valid!")
```

### Option 4: Request Interceptor
```python
# Monkey-patch requests to inspect but not send
import requests

original_get = requests.get
captured_calls = []

def intercepted_get(url, **kwargs):
    captured_calls.append({'url': url, 'params': kwargs.get('params')})
    raise Exception("INTERCEPTED - Not making real request")

requests.get = intercepted_get

try:
    retrieve_eod_ohlc('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0)
except Exception as e:
    if "INTERCEPTED" in str(e):
        print(f"Captured call: {captured_calls[-1]}")
finally:
    requests.get = original_get
```

---

## Implementation Order

### Week 1: Infrastructure
1. Create dry-run mode
2. Create mock testing framework
3. Create parameter inspector
4. Set up CI/CD for tests

### Week 2: Critical Fixes
1. Add proxy support (all functions)
2. Fix retrieve_ohlc (add quote data)
3. Fix retrieve_openInterest (return structure)
4. Fix parameter order (all functions)

### Week 3: High Priority
1. Add missing parameters (retrieve_quote_rt, retrieve_quote)
2. Fix list_contracts
3. Fix retrieve_chain_bulk
4. Fix type annotations

### Week 4: Polish & Testing
1. Consistency fixes (columns, formatting)
2. Comprehensive testing
3. Documentation updates
4. Code review and refinement

---

## Success Criteria

### Must Have (Blocking Release)
- [ ] All V2 function signatures accepted by V3
- [ ] All V2 positional calls work in V3
- [ ] Proxy support functional
- [ ] retrieve_ohlc returns quote columns
- [ ] retrieve_openInterest returns correct structure
- [ ] 100% of compatibility tests pass

### Should Have
- [ ] Column names match V2 exactly
- [ ] Date/time columns always present
- [ ] ohlc_format parameter functional
- [ ] rt parameter functional
- [ ] Dry-run mode working

### Nice to Have
- [ ] Performance improvement vs V2
- [ ] Better error messages
- [ ] Enhanced logging

---

## Rollout Plan

### Phase 1: Internal Testing
- Run all tests with dry-run mode
- Manual verification of parameter handling
- Code review

### Phase 2: Limited Beta
- Enable V3 for select test cases
- Monitor for errors
- Collect feedback

### Phase 3: Full Deployment
- Set `THETADATA_USE_V3=true` as default
- Monitor production errors
- Keep V2 available as fallback

### Phase 4: Deprecation (Future)
- After 6 months of stable V3
- Announce V2 deprecation
- Remove V2 code

---

## Risk Mitigation

**Risk:** Breaking existing code
- **Mitigation:** Comprehensive testing, gradual rollout, V2 fallback option

**Risk:** Untested edge cases
- **Mitigation:** Dry-run mode, extensive mocking, regression tests

**Risk:** Performance regression
- **Mitigation:** Benchmark tests, profiling, optimization

**Risk:** Proxy mode failures
- **Mitigation:** Separate proxy tests, fallback to direct mode

---

## Contact & Support

**Owner:** [Your Name]  
**Reviewers:** [Team Members]  
**Timeline:** 4 weeks  
**Status:** Planning → Implementation → Testing → Deployment
