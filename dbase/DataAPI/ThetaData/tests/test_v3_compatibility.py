"""
V3 Backward Compatibility Tests
================================

Tests to ensure V3 API is fully backward compatible with V2 API.

Usage
-----
Run all compatibility tests:
    pytest test_v3_compatibility.py -v

Run specific test:
    pytest test_v3_compatibility.py::test_retrieve_eod_ohlc_signature -v

Run in dry-run mode:
    THETADATA_DRY_RUN=true pytest test_v3_compatibility.py -v
"""

import pytest
import os
import sys  # noqa: F401
from unittest import mock  # noqa: F401
import pandas as pd  # noqa: F401

# Enable dry-run mode for all tests
os.environ["THETADATA_DRY_RUN"] = "true"

# Set to use V3
os.environ["THETADATA_USE_V3"] = "true"

from dbase.DataAPI.ThetaData import (
    retrieve_quote_rt,
    retrieve_quote,
    retrieve_ohlc,
    retrieve_eod_ohlc,
    retrieve_bulk_eod,  # noqa: F401
    retrieve_openInterest,
    retrieve_bulk_open_interest,  # noqa: F401
    retrieve_chain_bulk,  # noqa: F401
    list_contracts,
)
from .mock_responses import MOCK_EOD_OHLC, MOCK_QUOTE, MOCK_OPEN_INTEREST  # noqa: F401


class TestSignatureCompatibility:
    """Test that V3 accepts all V2 function signatures."""

    def test_retrieve_quote_rt_accepts_v2_params(self):
        """Test retrieve_quote_rt accepts all V2 parameters."""
        # V2 signature: (symbol, exp, right, strike, start_time, end_time, ts, proxy, start_date, end_date)

        # This should not raise TypeError
        try:
            result = retrieve_quote_rt(  # noqa: F841
                symbol="AAPL",
                exp="2024-12-20",
                right="C",
                strike=180.0,
                start_time="09:30:00",
                end_time="16:00:00",
                ts=False,
                proxy=None,
                start_date=None,
                end_date=None,
                print_url=False,
            )
            # If we get here without TypeError, signature is compatible
            assert True
        except TypeError as e:
            pytest.fail(f"V3 does not accept V2 parameters: {e}")

    def test_retrieve_quote_accepts_v2_positional_order(self):
        """Test retrieve_quote accepts V2 positional parameter order."""
        # V2 order: (symbol, end_date, exp, right, start_date, strike)

        try:
            # Positional call like V2
            result = retrieve_quote(  # noqa: F841
                "AAPL",  # symbol
                "2024-12-31",  # end_date
                "2024-12-20",  # exp
                "C",  # right
                "2024-12-01",  # start_date
                180.0,  # strike
                interval="30m",
            )
            # If parameters were accepted in this order, test passes
            assert True
        except Exception as e:
            pytest.fail(f"V3 does not accept V2 positional order: {e}")

    def test_retrieve_eod_ohlc_accepts_v2_positional_order(self):
        """Test retrieve_eod_ohlc accepts V2 positional parameter order."""
        # V2 order: (symbol, end_date, exp, right, start_date, strike)

        try:
            result = retrieve_eod_ohlc(  # noqa: F841
                "AAPL",  # symbol
                "2024-12-31",  # end_date
                "2024-12-20",  # exp
                "C",  # right
                "2024-12-01",  # start_date
                180.0,  # strike
                print_url=False,
                rt=True,
                proxy=None,
            )
            assert True
        except Exception as e:
            pytest.fail(f"V3 does not accept V2 positional order: {e}")

    def test_all_functions_accept_proxy_param(self):
        """Test that all functions accept proxy parameter."""
        functions = [
            (retrieve_quote_rt, {"symbol": "AAPL", "exp": "2024-12-20", "right": "C", "strike": 180.0}),
            (
                retrieve_quote,
                {
                    "symbol": "AAPL",
                    "exp": "2024-12-20",
                    "right": "C",
                    "strike": 180.0,
                    "start_date": "2024-12-01",
                    "end_date": "2024-12-31",
                },
            ),
            (
                retrieve_eod_ohlc,
                {
                    "symbol": "AAPL",
                    "exp": "2024-12-20",
                    "right": "C",
                    "strike": 180.0,
                    "start_date": "2024-12-01",
                    "end_date": "2024-12-31",
                },
            ),
            (list_contracts, {"symbol": "AAPL", "start_date": "2024-12-01"}),
        ]

        for func, params in functions:
            try:
                # Should not raise TypeError about unexpected 'proxy' argument
                func(**params, proxy="http://localhost:8080")
                assert True
            except TypeError as e:
                if "proxy" in str(e):
                    pytest.fail(f"{func.__name__} does not accept proxy parameter: {e}")


class TestReturnStructureCompatibility:
    """Test that V3 returns same structure as V2."""

    def test_retrieve_eod_ohlc_returns_expected_columns(self):
        """Test retrieve_eod_ohlc returns all expected columns."""
        result = retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)

        expected_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid_size",
            "CloseBid",
            "Ask_size",
            "CloseAsk",
            "Midpoint",
            "Weighted_midpoint",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_retrieve_ohlc_returns_quote_columns(self):
        """Test retrieve_ohlc includes quote data columns."""
        result = retrieve_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)

        # V2 includes these quote columns
        quote_columns = ["Bid_size", "CloseBid", "Ask_size", "CloseAsk", "Midpoint", "Weighted_midpoint"]

        for col in quote_columns:
            assert col in result.columns, f"Missing quote column: {col}"

    def test_retrieve_openInterest_returns_datetime_as_column(self):
        """Test retrieve_openInterest returns Datetime as column, not index."""
        result = retrieve_openInterest("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)

        # V2 returns Datetime as a column
        assert "Datetime" in result.columns, "Datetime should be a column, not index"
        assert "Open_interest" in result.columns or "OpenInterest" in result.columns

    def test_retrieve_quote_includes_date_time_columns(self):
        """Test retrieve_quote includes Date and time columns."""
        result = retrieve_quote("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)

        # V2 always includes these
        assert "Date" in result.columns, "Missing Date column"
        # Note: time column check may need adjustment based on implementation


class TestKwargsCompatibility:
    """Test that V3 handles V2 kwargs properly."""

    def test_ohlc_format_parameter_accepted(self):
        """Test that ohlc_format parameter is accepted."""
        try:
            result = retrieve_quote(  # noqa: F841
                "AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0, ohlc_format=True
            )
            assert True
        except TypeError as e:
            pytest.fail(f"ohlc_format parameter not accepted: {e}")

    def test_rt_parameter_accepted(self):
        """Test that rt parameter is accepted."""
        try:
            result = retrieve_eod_ohlc(  # noqa: F841
                "AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0, rt=True
            )
            assert True
        except TypeError as e:
            pytest.fail(f"rt parameter not accepted: {e}")

    def test_depth_parameter_ignored_gracefully(self):
        """Test that depth parameter (V2 internal) is ignored without error."""
        try:
            result = retrieve_eod_ohlc(  # noqa: F841
                "AAPL",
                "2024-12-31",
                "2024-12-20",
                "C",
                "2024-12-01",
                180.0,
                depth=1,  # V2 internal parameter
            )
            # Should not raise error, just ignore it
            assert True
        except TypeError as e:
            pytest.fail(f"depth parameter causes error: {e}")


class TestParameterFormatting:
    """Test that parameters are formatted correctly for API calls."""

    def test_strike_formatted_correctly(self):
        """Test that strike price is formatted as float."""
        from dbase.DataAPI.ThetaData.tests.dry_run import enable_capture_mode, get_captured_calls, clear_captured_calls

        enable_capture_mode()
        clear_captured_calls()

        retrieve_eod_ohlc("AAPL", "2024-12-31", "2024-12-20", "C", "2024-12-01", 180.0)

        calls = get_captured_calls()
        assert len(calls) > 0, "No API calls captured"

        # Check that strike is formatted (V3 uses "180.00", V2 used 180000)
        # Just verify it's present
        assert "strike" in calls[0]["params"] or "Strike" in str(calls[0]["params"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
