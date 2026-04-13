#!/usr/bin/env python3
"""
Quick Test Script for V3 Compatibility
=======================================

This script tests V3 backward compatibility without requiring ThetaData Terminal.

Usage:
    python test_v3_quick.py

Output:
    - Shows which functions accept V2 parameters
    - Shows which functions return correct structure
    - Identifies breaking changes
"""

import os
import sys

# Enable dry-run mode
os.environ['THETADATA_DRY_RUN'] = 'true'
os.environ['THETADATA_USE_V3'] = 'true'

from dbase.DataAPI.ThetaData import (
    retrieve_quote_rt,
    retrieve_quote,
    retrieve_ohlc,
    retrieve_eod_ohlc,
    retrieve_bulk_eod,
    retrieve_openInterest,
    retrieve_bulk_open_interest,
    retrieve_chain_bulk,
    list_contracts,
)


def test_function(name, func, test_call, expected_result):
    """Test a single function call."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    try:
        result = test_call()
        
        # Check result
        if expected_result(result):
            print(f"✅ PASS: {name}")
            return True
        else:
            print(f"❌ FAIL: {name} - Unexpected result structure")
            return False
            
    except TypeError as e:
        print(f"❌ FAIL: {name} - Signature incompatible")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  WARNING: {name} - Unexpected error")
        print(f"   Error: {e}")
        return False


def main():
    """Run all compatibility tests."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                V3 BACKWARD COMPATIBILITY QUICK TEST                          ║
║                                                                              ║
║  Testing without ThetaData Terminal (Dry-Run Mode)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    tests = []
    
    # Test 1: retrieve_eod_ohlc - V2 signature
    tests.append(test_function(
        name="retrieve_eod_ohlc (V2 positional signature)",
        func=retrieve_eod_ohlc,
        test_call=lambda: retrieve_eod_ohlc(
            "AAPL",           # symbol
            "2024-12-31",     # end_date
            "2024-12-20",     # exp
            "C",              # right
            "2024-12-01",     # start_date
            180.0,            # strike
            print_url=False,
            rt=True,
            proxy=None
        ),
        expected_result=lambda r: hasattr(r, 'columns') and 'Open' in r.columns
    ))
    
    # Test 2: retrieve_eod_ohlc - has all columns
    tests.append(test_function(
        name="retrieve_eod_ohlc (returns all expected columns)",
        func=retrieve_eod_ohlc,
        test_call=lambda: retrieve_eod_ohlc(
            "AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0
        ),
        expected_result=lambda r: all(col in r.columns for col in 
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Midpoint'])
    ))
    
    # Test 3: retrieve_quote - V2 positional signature
    tests.append(test_function(
        name="retrieve_quote (V2 positional signature)",
        func=retrieve_quote,
        test_call=lambda: retrieve_quote(
            "AAPL",           # symbol
            "2024-12-31",     # end_date
            "2024-12-20",     # exp
            "C",              # right
            "2024-12-01",     # start_date
            180.0,            # strike
            interval="30m"
        ),
        expected_result=lambda r: hasattr(r, 'columns') and 'Open' in r.columns
    ))
    
    # Test 4: retrieve_quote - with proxy parameter
    tests.append(test_function(
        name="retrieve_quote (accepts proxy parameter)",
        func=retrieve_quote,
        test_call=lambda: retrieve_quote(
            symbol="AAPL",
            exp="2024-12-20",
            right="C",
            strike=180.0,
            start_date="2024-12-01",
            end_date="2024-12-31",
            proxy="http://localhost:8080"
        ),
        expected_result=lambda r: hasattr(r, 'columns')
    ))
    
    # Test 5: retrieve_ohlc - returns quote columns
    tests.append(test_function(
        name="retrieve_ohlc (includes quote data columns)",
        func=retrieve_ohlc,
        test_call=lambda: retrieve_ohlc(
            "AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0
        ),
        expected_result=lambda r: all(col in r.columns for col in 
            ['CloseBid', 'CloseAsk', 'Midpoint'])
    ))
    
    # Test 6: retrieve_openInterest - Datetime as column
    tests.append(test_function(
        name="retrieve_openInterest (Datetime as column)",
        func=retrieve_openInterest,
        test_call=lambda: retrieve_openInterest(
            "AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0
        ),
        expected_result=lambda r: 'Datetime' in r.columns
    ))
    
    # Test 7: list_contracts - accepts start_date
    tests.append(test_function(
        name="list_contracts (accepts start_date parameter)",
        func=list_contracts,
        test_call=lambda: list_contracts(
            symbol="AAPL",
            start_date="2024-12-01"
        ),
        expected_result=lambda r: hasattr(r, 'columns')
    ))
    
    # Test 8: retrieve_quote_rt - accepts all V2 params
    tests.append(test_function(
        name="retrieve_quote_rt (accepts all V2 parameters)",
        func=retrieve_quote_rt,
        test_call=lambda: retrieve_quote_rt(
            symbol="AAPL",
            exp="2024-12-20",
            right="C",
            strike=180.0,
            start_time="09:30:00",
            end_time="16:00:00",
            ts=False,
            proxy=None
        ),
        expected_result=lambda r: hasattr(r, 'columns')
    ))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(tests)
    total = len(tests)
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - V3 is backward compatible!")
        return 0
    else:
        print(f"\n❌ {total - passed} TESTS FAILED - V3 has compatibility issues")
        print("\nSee V3_COMPATIBILITY_TODO.md for implementation plan")
        return 1


if __name__ == '__main__':
    sys.exit(main())
