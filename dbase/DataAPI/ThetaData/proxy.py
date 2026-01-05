"""
ThetaData API Proxy Configuration Module
=========================================

This module manages proxy server configuration for distributed ThetaData API requests.
It enables routing API calls through remote proxy servers for load distribution,
rate limiting avoidance, and geographic distribution.

Overview
--------
The proxy system provides:
- Dynamic proxy URL configuration via environment variables
- Enable/disable proxy usage at runtime
- Proxy connectivity testing
- Request scheduling support (optional)
- Automatic fallback to direct API access if proxy unavailable

When enabled, API requests are routed through the proxy server instead of directly
to the local ThetaData Terminal. This is useful for:
- Distributed data fetching across multiple machines
- Centralized rate limiting and request management
- Load balancing for high-volume queries
- Remote access to ThetaData Terminal

Proxy Architecture
------------------
Without Proxy:
    Python App → ThetaData Terminal (localhost:25503/25510)

With Proxy:
    Python App → Proxy Server → ThetaData Terminal

The proxy server receives requests, forwards them to ThetaData Terminal,
and returns the response to the client.

Configuration
-------------
Enable proxy via environment variable:

.. code-block:: bash

    export PROXY_URL="http://remote-server:8080/api/proxy"

Enable/disable at runtime:

.. code-block:: python

    from dbase.DataAPI.ThetaData.proxy import set_use_proxy

    # Enable proxy (reads from PROXY_URL environment variable)
    set_use_proxy(True)

    # Disable proxy (direct API calls)
    set_use_proxy(False)

Functions
---------
get_proxy_url() -> str | None
    Get the currently configured proxy URL.
    Returns None if proxy is disabled.

set_use_proxy(use_proxy: bool) -> None
    Enable or disable proxy usage.
    When enabled, refreshes URL from PROXY_URL environment variable.

refresh_proxy_url() -> None
    Reload proxy URL from PROXY_URL environment variable.
    Called automatically when enabling proxy.

ping_proxy() -> bool
    Test connectivity to configured proxy server.
    Returns True if proxy responds successfully, False otherwise.

get_should_schedule() -> bool
    Check if request scheduling is enabled.
    Used by SaveManager for asynchronous request queuing.

set_should_schedule(should_schedule: bool) -> None
    Enable or disable request scheduling.
    When True, requests can be queued via SaveManager.

schedule_kwargs(kwargs: dict) -> None
    Schedule a request for later execution via SaveManager.
    Only works if should_schedule is True.

Usage Examples
--------------
Basic proxy setup:

.. code-block:: python

    from dbase.DataAPI.ThetaData import retrieve_eod_ohlc
    from dbase.DataAPI.ThetaData.proxy import set_use_proxy, ping_proxy

    # Enable proxy
    set_use_proxy(True)

    # Test connectivity
    if ping_proxy():
        print("Proxy is reachable")
    else:
        print("Proxy is not responding")

    # Make API call (automatically uses proxy)
    data = retrieve_eod_ohlc(
        symbol='AAPL',
        exp='2024-12-20',
        right='C',
        strike=180.0,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

Disabling proxy:

.. code-block:: python

    from dbase.DataAPI.ThetaData.proxy import set_use_proxy

    # Disable proxy (use direct local connection)
    set_use_proxy(False)

Checking proxy status:

.. code-block:: python

    from dbase.DataAPI.ThetaData.proxy import get_proxy_url

    url = get_proxy_url()
    if url:
        print(f"Proxy enabled: {url}")
    else:
        print("Proxy disabled - using direct connection")

Proxy Server Requirements
--------------------------
The proxy server must:
1. Accept POST requests with JSON payload
2. Handle request format: {"method": "GET", "url": "<thetadata_url>"}
3. Return JSON response: {"data": "<csv_data>", "url": "<original_url>"}
4. Forward requests to local ThetaData Terminal
5. Return data in same format as ThetaData API

Example proxy request payload:

.. code-block:: json

    {
        "method": "GET",
        "url": "http://127.0.0.1:25510/v2/hist/option/eod?symbol=AAPL&exp=20241220&..."
    }

Example proxy response:

.. code-block:: json

    {
        "data": "timestamp,open,high,low,close,volume\\n...",
        "url": "http://127.0.0.1:25510/v2/hist/option/eod?symbol=AAPL&..."
    }

Environment Variables
---------------------
PROXY_URL : str
    Full URL to proxy server endpoint.
    Example: "http://proxy.example.com:8080/api/thetadata"

    When not set or empty, proxy is disabled and direct connection is used.

Module Variables
----------------
proxy_url : str | None
    Currently active proxy URL (None if disabled)

_SHOULD_SCHEDULE : bool
    Whether request scheduling is enabled (default: True)

Performance Considerations
--------------------------
- Proxy adds network latency (typically 10-100ms depending on location)
- Useful for rate limiting protection across multiple clients
- Can improve throughput for bulk queries via load balancing
- Test proxy latency vs direct connection for your use case

Error Handling
--------------
If proxy is unreachable:
- ping_proxy() returns False
- API calls will fail with connection errors
- Disable proxy and use direct connection as fallback

Logging
-------
All proxy configuration changes are logged at INFO level:
- "Set should_schedule to <value>"
- "Refreshed proxy URL: <url>"
- "Proxy URL has been unset."
- "Using Proxy URL: <url>"
- "No Proxy URL found. ThetaData API will default to direct access"

See Also
--------
- utils._fetch_data : Main function that uses proxy for API calls
- utils.request_from_proxy : Low-level proxy request function
- SaveManager : Request scheduling system (when should_schedule=True)

Notes
-----
- Proxy is optional - module works perfectly fine without it
- Direct connection is faster for single-machine setups
- Proxy is recommended for multi-machine distributed systems
- Always test proxy connectivity before production use
"""

from trade.helpers.Logging import setup_logger
import os
import requests

logger = setup_logger("dbase.DataAPI.ThetaData.proxy", stream_log_level="INFO")
_SHOULD_SCHEDULE = True
proxy_url = None  ## Initial initiation


def set_should_schedule(should_schedule):
    global _SHOULD_SCHEDULE
    _SHOULD_SCHEDULE = should_schedule
    logger.info(f"Set should_schedule to {should_schedule}")


def get_should_schedule():
    return _SHOULD_SCHEDULE


def schedule_kwargs(kwargs):
    from module_test.raw_code.DataManagers.SaveManager import SaveManager

    SaveManager.schedule(kwargs=kwargs)


def get_proxy_url_from_env():
    return os.environ.get("PROXY_URL") if os.environ.get("PROXY_URL") else None


def get_proxy_url():
    return proxy_url


def refresh_proxy_url():
    global proxy_url
    proxy_url = get_proxy_url_from_env()
    logger.info(f"Refreshed proxy URL: {proxy_url}")


def set_use_proxy(use_proxy: bool):
    if use_proxy:
        refresh_proxy_url()
    else:
        global proxy_url
        proxy_url = None
        logger.info("Proxy URL has been unset.")


refresh_proxy_url()  ## Refreshing proxy url


def ping_proxy():
    try:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {
            "method": "GET",
            "url": "http://127.0.0.1:25510/v2/hist/option/eod?end_date=20250619&root=AAPL&use_csv=true&exp=20241220&right=C&start_date=20240101&strike=220000",
        }
        proxy_url = os.environ["PROXY_URL"]
        response = requests.post(proxy_url, headers=headers, json=payload)
        return response.status_code == 200
    except Exception as e:  # noqa
        return False


if get_proxy_url() is None:
    logger.info("No Proxy URL found. ThetaData API will default to direct access")
else:
    logger.info(f"Using Proxy URL: {get_proxy_url()}")
