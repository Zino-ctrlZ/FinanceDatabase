"""
Mock API Responses for Testing
===============================

This module provides realistic mock responses for all ThetaData API endpoints,
allowing testing without the ThetaData Terminal running.

Usage
-----
>>> from dbase.DataAPI.ThetaData.tests.mock_responses import get_mock_response
>>> response = get_mock_response('eod_ohlc', symbol='AAPL')
>>> print(response)
"""

MOCK_EOD_OHLC = """timestamp,open,high,low,close,volume,bid,ask,bid_size,ask_size
20240101160000,100.50,105.25,99.75,104.00,15000,103.50,104.50,100,150
20240102160000,104.00,106.50,103.25,105.75,18000,105.25,106.25,120,180
20240103160000,105.75,107.00,104.50,106.25,16500,105.75,106.75,110,140"""

MOCK_QUOTE = """timestamp,bid,ask,bid_size,ask_size
20240101093000,100.25,100.75,50,75
20240101100000,100.50,101.00,60,80
20240101103000,100.75,101.25,55,70
20240101110000,101.00,101.50,65,85"""

MOCK_OHLC = """timestamp,open,high,low,close,volume
20240101093000,100.50,101.00,100.25,100.75,500
20240101100000,100.75,101.25,100.50,101.00,600
20240101103000,101.00,101.50,100.75,101.25,550"""

MOCK_OPEN_INTEREST = """timestamp,open_interest
20240101,5000
20240102,5100
20240103,5200"""

MOCK_BULK_EOD = """timestamp,symbol,strike,expiration,right,open,high,low,close,volume,bid,ask,bid_size,ask_size
20240101160000,AAPL,180.00,20241220,C,10.50,11.25,10.25,11.00,1000,10.75,11.25,50,60
20240101160000,AAPL,185.00,20241220,C,8.50,9.00,8.25,8.75,800,8.50,9.00,40,50
20240101160000,AAPL,180.00,20241220,P,2.50,2.75,2.25,2.60,600,2.55,2.65,30,40"""

MOCK_BULK_OPEN_INTEREST = """timestamp,symbol,strike,expiration,right,open_interest
20240101,AAPL,180.00,20241220,C,5000
20240101,AAPL,185.00,20241220,C,4500
20240101,AAPL,180.00,20241220,P,3000"""

MOCK_CHAIN_BULK = """timestamp,symbol,strike,expiration,right,bid,ask,bid_size,ask_size
20240101160000,AAPL,175.00,20241220,C,15.50,16.00,100,120
20240101160000,AAPL,180.00,20241220,C,11.00,11.50,150,180
20240101160000,AAPL,185.00,20241220,C,7.50,8.00,80,100
20240101160000,AAPL,175.00,20241220,P,1.50,1.75,50,60
20240101160000,AAPL,180.00,20241220,P,2.50,2.75,70,80
20240101160000,AAPL,185.00,20241220,P,4.50,5.00,60,70"""

MOCK_LIST_CONTRACTS = """symbol,strike,expiration,right
AAPL,175.00,20241220,C
AAPL,180.00,20241220,C
AAPL,185.00,20241220,C
AAPL,190.00,20241220,C
AAPL,175.00,20241220,P
AAPL,180.00,20241220,P
AAPL,185.00,20241220,P
AAPL,190.00,20241220,P"""

MOCK_LIST_DATES = """timestamp
20240101
20240102
20240103
20240104
20240105"""

MOCK_QUOTE_RT = """timestamp,bid,ask,bid_size,ask_size
20260405143000,100.50,101.00,100,150"""

MOCK_BULK_QUOTE_RT = """timestamp,symbol,strike,expiration,right,bid,ask,bid_size,ask_size
20260405143000,AAPL,180.00,20241220,C,11.00,11.50,100,120
20260405143000,AAPL,185.00,20241220,C,8.00,8.50,80,100
20260405143000,AAPL,180.00,20241220,P,2.50,2.75,60,70"""


RESPONSE_MAP = {
    "eod_ohlc": MOCK_EOD_OHLC,
    "quote": MOCK_QUOTE,
    "ohlc": MOCK_OHLC,
    "open_interest": MOCK_OPEN_INTEREST,
    "bulk_eod": MOCK_BULK_EOD,
    "bulk_open_interest": MOCK_BULK_OPEN_INTEREST,
    "chain_bulk": MOCK_CHAIN_BULK,
    "list_contracts": MOCK_LIST_CONTRACTS,
    "list_dates": MOCK_LIST_DATES,
    "quote_rt": MOCK_QUOTE_RT,
    "bulk_quote_rt": MOCK_BULK_QUOTE_RT,
}


def get_mock_response(endpoint_type: str, **kwargs) -> str:
    """
    Get a mock response for a specific endpoint type.

    Parameters
    ----------
    endpoint_type : str
        Type of endpoint: 'eod_ohlc', 'quote', 'ohlc', 'open_interest', etc.
    **kwargs : dict
        Additional parameters (currently ignored, but available for customization)

    Returns
    -------
    str
        CSV-formatted mock response data
    """
    response = RESPONSE_MAP.get(endpoint_type)
    if response is None:
        # Default response for unknown endpoints
        return "timestamp\n20240101"
    return response


def detect_endpoint_type(url: str, params: dict) -> str:
    """
    Detect endpoint type from URL and parameters.

    Parameters
    ----------
    url : str
        The API endpoint URL
    params : dict
        Query parameters

    Returns
    -------
    str
        Detected endpoint type
    """
    url_lower = url.lower()

    if "eod" in url_lower:
        if "bulk" in url_lower:
            return "bulk_eod"
        return "eod_ohlc"
    elif "quote" in url_lower:
        if "snapshot" in url_lower or "realtime" in url_lower:
            if params.get("strike") == "*":
                return "bulk_quote_rt"
            return "quote_rt"
        return "quote"
    elif "ohlc" in url_lower:
        return "ohlc"
    elif "open_interest" in url_lower:
        if "bulk" in url_lower:
            return "bulk_open_interest"
        return "open_interest"
    elif "chain" in url_lower or "at_time" in url_lower:
        return "chain_bulk"
    elif "list" in url_lower:
        if "dates" in url_lower:
            return "list_dates"
        return "list_contracts"

    # Default
    return "eod_ohlc"
