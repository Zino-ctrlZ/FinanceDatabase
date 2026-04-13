"""
Dry-Run Mode for Testing Without ThetaData Terminal
===================================================

This module enables testing ThetaData functions without the terminal running.
In dry-run mode, functions validate parameters and return mock data instead
of making actual API calls.

Usage
-----
Set environment variable:
    export THETADATA_DRY_RUN=true

Or programmatically:
    >>> import os
    >>> os.environ['THETADATA_DRY_RUN'] = 'true'
    >>> from dbase.DataAPI.ThetaData import retrieve_eod_ohlc
    >>> result = retrieve_eod_ohlc('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0)
    >>> # Returns mock data, no API call made

Capture mode:
    >>> from dbase.DataAPI.ThetaData.tests.dry_run import enable_capture_mode, get_captured_calls
    >>> enable_capture_mode()
    >>> retrieve_eod_ohlc('AAPL', '2024-12-31', '2024-12-20', 'C', '2024-01-01', 180.0)
    >>> calls = get_captured_calls()
    >>> print(calls[0]['params'])
"""

import os
import json
from typing import List, Dict, Any
from .mock_responses import get_mock_response, detect_endpoint_type

# Global capture storage
_captured_calls: List[Dict[str, Any]] = []
_capture_enabled = False


def is_dry_run_enabled() -> bool:
    """Check if dry-run mode is enabled."""
    return os.environ.get("THETADATA_DRY_RUN", "false").lower() == "true"


def enable_dry_run():
    """Enable dry-run mode."""
    os.environ["THETADATA_DRY_RUN"] = "true"


def disable_dry_run():
    """Disable dry-run mode."""
    os.environ["THETADATA_DRY_RUN"] = "false"


def enable_capture_mode():
    """Enable capture mode to record all API calls."""
    global _capture_enabled
    _capture_enabled = True
    enable_dry_run()


def disable_capture_mode():
    """Disable capture mode."""
    global _capture_enabled
    _capture_enabled = False


def get_captured_calls() -> List[Dict[str, Any]]:
    """Get all captured API calls."""
    return _captured_calls.copy()


def clear_captured_calls():
    """Clear all captured API calls."""
    global _captured_calls
    _captured_calls = []


def capture_call(url: str, params: dict, caller: str = None):
    """
    Capture an API call for inspection.

    Parameters
    ----------
    url : str
        The API endpoint URL
    params : dict
        Query parameters
    caller : str, optional
        Name of the calling function
    """
    global _captured_calls

    if _capture_enabled:
        _captured_calls.append(
            {"url": url, "params": params, "caller": caller, "endpoint_type": detect_endpoint_type(url, params)}
        )


def get_dry_run_response(url: str, params: dict) -> str:
    """
    Get a mock response for dry-run mode.

    Parameters
    ----------
    url : str
        The API endpoint URL
    params : dict
        Query parameters

    Returns
    -------
    str
        Mock CSV response data
    """
    # Capture the call
    capture_call(url, params)

    # Detect endpoint type and return appropriate mock data
    endpoint_type = detect_endpoint_type(url, params)
    return get_mock_response(endpoint_type, **params)


def validate_parameters(func_name: str, args: tuple, kwargs: dict) -> List[str]:
    """
    Validate parameters for a function call.

    Parameters
    ----------
    func_name : str
        Name of the function being called
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments

    Returns
    -------
    List[str]
        List of validation errors (empty if valid)
    """
    errors = []

    # Add validation logic here
    # For now, just check basic types

    return errors


def print_dry_run_summary():
    """Print a summary of captured calls."""
    if not _captured_calls:
        print("No calls captured.")
        return

    print(f"\n{'=' * 80}")
    print(f"DRY RUN SUMMARY - {len(_captured_calls)} calls captured")
    print(f"{'=' * 80}\n")

    for i, call in enumerate(_captured_calls, 1):
        print(f"Call {i}:")
        print(f"  Endpoint Type: {call['endpoint_type']}")
        print(f"  URL: {call['url']}")
        print(f"  Params: {json.dumps(call['params'], indent=4)}")
        if call.get("caller"):
            print(f"  Caller: {call['caller']}")
        print()
