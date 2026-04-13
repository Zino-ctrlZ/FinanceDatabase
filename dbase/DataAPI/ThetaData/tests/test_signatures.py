"""
Simple signature validation test for V3 backward compatibility.
"""
import inspect
import sys

# Test without actually calling the functions - just check signatures
def test_function_signatures():
    """Test that V3 functions now accept all V2 parameters."""
    
    print("=" * 80)
    print("V3 BACKWARD COMPATIBILITY - SIGNATURE VALIDATION")
    print("=" * 80)
    print()
    
    try:
        # Import v3 endpoints
        sys.path.insert(0, '/Users/chiemelienwanisobi/cloned_repos/FinanceDatabase')
        from dbase.DataAPI.ThetaData.v3 import endpoints
        
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: retrieve_eod_ohlc
        print("Test 1: retrieve_eod_ohlc")
        sig = inspect.signature(endpoints._retrieve_eod_ohlc)
        params = list(sig.parameters.keys())
        expected_params = ['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike', 
                          'print_url', 'rt', 'proxy', 'opttick', 'kwargs']
        
        # Check parameter order (first 9 params should match V2 order)
        v2_order = ['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike', 
                   'print_url', 'rt', 'proxy']
        matches_order = params[:9] == v2_order
        has_all_params = all(p in params for p in expected_params)
        
        if matches_order and has_all_params:
            print("  ✅ PASS: Parameter order matches V2, all params present")
            print(f"     Order: {params[:9]}")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: Expected {v2_order}")
            print(f"     Got:      {params[:9]}")
            tests_failed += 1
        print()
        
        # Test 2: retrieve_quote
        print("Test 2: retrieve_quote")
        sig = inspect.signature(endpoints._retrieve_quote)
        params = list(sig.parameters.keys())
        v2_order = ['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike',
                   'start_time', 'print_url', 'end_time', 'interval', 'proxy', 'ohlc_format']
        
        matches_order = params[:12] == v2_order
        has_proxy = 'proxy' in params
        has_start_time = 'start_time' in params
        
        if matches_order and has_proxy and has_start_time:
            print("  ✅ PASS: Parameter order matches V2, proxy and start_time present")
            print(f"     Order: {params[:12]}")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: Expected {v2_order}")
            print(f"     Got:      {params[:12]}")
            tests_failed += 1
        print()
        
        # Test 3: retrieve_ohlc
        print("Test 3: retrieve_ohlc")
        sig = inspect.signature(endpoints._retrieve_ohlc)
        params = list(sig.parameters.keys())
        v2_order = ['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike',
                   'start_time', 'print_url', 'proxy']
        
        matches_order = params[:9] == v2_order
        has_proxy = 'proxy' in params
        has_start_time = 'start_time' in params
        
        if matches_order and has_proxy and has_start_time:
            print("  ✅ PASS: Parameter order matches V2, proxy and start_time present")
            print(f"     Order: {params[:9]}")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: Expected {v2_order}")
            print(f"     Got:      {params[:9]}")
            tests_failed += 1
        print()
        
        # Test 4: retrieve_quote_rt
        print("Test 4: retrieve_quote_rt")
        sig = inspect.signature(endpoints._retrieve_quote_rt)
        params = list(sig.parameters.keys())
        required_params = ['symbol', 'exp', 'right', 'strike', 'start_time', 
                          'print_url', 'end_time', 'ts', 'proxy', 'start_date', 'end_date']
        
        has_all_params = all(p in params for p in required_params)
        
        if has_all_params:
            print("  ✅ PASS: All V2 parameters present (including start_time, end_time, ts, proxy)")
            print(f"     Params: {params[:11]}")
            tests_passed += 1
        else:
            missing = [p for p in required_params if p not in params]
            print(f"  ❌ FAIL: Missing parameters: {missing}")
            tests_failed += 1
        print()
        
        # Test 5: retrieve_openInterest
        print("Test 5: retrieve_openInterest")
        sig = inspect.signature(endpoints._retrieve_openInterest)
        params = list(sig.parameters.keys())
        v2_order = ['symbol', 'end_date', 'exp', 'right', 'start_date', 'strike',
                   'print_url', 'proxy']
        
        matches_order = params[:8] == v2_order
        has_proxy = 'proxy' in params
        
        if matches_order and has_proxy:
            print("  ✅ PASS: Parameter order matches V2, proxy present")
            print(f"     Order: {params[:8]}")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: Expected {v2_order}")
            print(f"     Got:      {params[:8]}")
            tests_failed += 1
        print()
        
        # Test 6: retrieve_bulk_eod
        print("Test 6: retrieve_bulk_eod")
        sig = inspect.signature(endpoints._retrieve_bulk_eod)
        params = list(sig.parameters.keys())
        has_proxy = 'proxy' in params
        print_url_param = sig.parameters.get('print_url')
        correct_type = print_url_param.annotation == bool if print_url_param else False
        
        if has_proxy and correct_type:
            print("  ✅ PASS: Has proxy parameter, print_url type is bool")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: proxy={has_proxy}, print_url is bool={correct_type}")
            tests_failed += 1
        print()
        
        # Test 7: retrieve_bulk_open_interest
        print("Test 7: retrieve_bulk_open_interest")
        sig = inspect.signature(endpoints._retrieve_bulk_open_interest)
        params = list(sig.parameters.keys())
        has_proxy = 'proxy' in params
        
        if has_proxy:
            print("  ✅ PASS: Has proxy parameter")
            tests_passed += 1
        else:
            print("  ❌ FAIL: Missing proxy parameter")
            tests_failed += 1
        print()
        
        # Test 8: retrieve_chain_bulk
        print("Test 8: retrieve_chain_bulk")
        sig = inspect.signature(endpoints._retrieve_chain_bulk)
        params = list(sig.parameters.keys())
        has_proxy = 'proxy' in params
        has_start_date = 'start_date' in params
        has_end_date = 'end_date' in params
        
        if has_proxy and has_start_date and has_end_date:
            print("  ✅ PASS: Has proxy, start_date, and end_date parameters")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: proxy={has_proxy}, start_date={has_start_date}, end_date={has_end_date}")
            tests_failed += 1
        print()
        
        # Test 9: list_contracts
        print("Test 9: list_contracts")
        sig = inspect.signature(endpoints._list_contracts)
        params = list(sig.parameters.keys())
        has_proxy = 'proxy' in params
        has_kwargs = 'kwargs' in params
        
        if has_proxy and has_kwargs:
            print("  ✅ PASS: Has proxy parameter and kwargs for start_date handling")
            tests_passed += 1
        else:
            print(f"  ❌ FAIL: proxy={has_proxy}, kwargs={has_kwargs}")
            tests_failed += 1
        print()
        
        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Tests Passed: {tests_passed}/9")
        print(f"Tests Failed: {tests_failed}/9")
        print()
        
        if tests_failed == 0:
            print("✅ ALL SIGNATURE TESTS PASSED - V3 is backward compatible with V2 signatures!")
        else:
            print(f"❌ {tests_failed} test(s) failed - some signatures still need fixing")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_function_signatures()
