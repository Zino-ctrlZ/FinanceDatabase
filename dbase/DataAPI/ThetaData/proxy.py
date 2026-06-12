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

ping_proxy_v2() / ping_proxy_v3() -> PingProxyResult
    Test proxy POST wrapper (HTTP 200) and Theta ``status_code`` (200) on a list endpoint.

ping_proxy() (via switcher, default) -> bool
    True only when both proxy wrapper and Theta return success. Use ``detail=True`` for PingProxyResult.

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

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import requests

from trade.helpers.Logging import setup_logger

logger = setup_logger("dbase.DataAPI.ThetaData.proxy", stream_log_level="INFO")

# Lightweight Theta endpoints used for health checks (not portfolio quote paths).
V2_PING_THETA_URL = "http://127.0.0.1:25510/v2/list/roots/stock?use_csv=true"
V3_PING_THETA_URL = "http://127.0.0.1:25503/v3/stock/list/symbols"


@dataclass(frozen=True)
class PingProxyResult:
    """Result of a proxy + ThetaData health check."""

    ok: bool
    proxy_http_status: Optional[int]
    proxy_ok: bool
    theta_status_code: Optional[int]
    theta_ok: bool
    theta_url: str
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok

    def message(self) -> str:
        proxy_part = (
            f"proxy HTTP {self.proxy_http_status}"
            if self.proxy_http_status is not None
            else "proxy HTTP (no response)"
        )
        proxy_state = "OK" if self.proxy_ok else "FAIL"
        if self.theta_status_code is not None:
            theta_part = f"Theta status {self.theta_status_code}"
        else:
            theta_part = "Theta status (unavailable)"
        theta_state = "OK" if self.theta_ok else "FAIL"
        base = f"{proxy_part} [{proxy_state}]; {theta_part} [{theta_state}]"
        if self.error:
            return f"{base} — {self.error}"
        if self.ok:
            return f"{base} — connected"
        return f"{base} — not connected"
_SHOULD_SCHEDULE = True
proxy_url = None  ## Initial initiation


def set_should_schedule(should_schedule):
    global _SHOULD_SCHEDULE
    _SHOULD_SCHEDULE = should_schedule
    logger.info(f"Set should_schedule to {should_schedule}")


def get_should_schedule():
    return _SHOULD_SCHEDULE


def schedule_kwargs(**kwargs):
    logger.warning("Schedule kwargs is not implemented. Returning None. Please stop using this function.")
    return None


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


def _evaluate_proxy_ping(response: requests.Response, theta_url: str) -> PingProxyResult:
    """Parse proxy POST wrapper and embedded Theta status_code."""
    proxy_http_status = response.status_code
    proxy_ok = proxy_http_status == 200
    if not proxy_ok:
        return PingProxyResult(
            ok=False,
            proxy_http_status=proxy_http_status,
            proxy_ok=False,
            theta_status_code=None,
            theta_ok=False,
            theta_url=theta_url,
            error=f"Proxy wrapper returned HTTP {proxy_http_status}",
        )

    try:
        body = response.json()
    except ValueError as exc:
        return PingProxyResult(
            ok=False,
            proxy_http_status=proxy_http_status,
            proxy_ok=True,
            theta_status_code=None,
            theta_ok=False,
            theta_url=theta_url,
            error=f"Proxy response is not JSON: {exc}",
        )

    theta_status_code = body.get("status_code")
    theta_ok = theta_status_code == 200
    ok = proxy_ok and theta_ok
    error = None
    if not theta_ok:
        data_msg = body.get("data")
        if data_msg is not None:
            error = str(data_msg).strip()[:300] or None
        if error is None and theta_status_code is not None:
            error = f"Theta returned status {theta_status_code}"

    return PingProxyResult(
        ok=ok,
        proxy_http_status=proxy_http_status,
        proxy_ok=proxy_ok,
        theta_status_code=theta_status_code,
        theta_ok=theta_ok,
        theta_url=theta_url,
        error=error,
    )


def _ping_proxy(theta_url: str, label: str) -> PingProxyResult:
    instance_url = get_proxy_url() or get_proxy_url_from_env()
    if not instance_url:
        return PingProxyResult(
            ok=False,
            proxy_http_status=None,
            proxy_ok=False,
            theta_status_code=None,
            theta_ok=False,
            theta_url=theta_url,
            error="PROXY_URL is not configured",
        )

    logger.info("Pinging proxy %s via %s", label, instance_url)
    try:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"method": "GET", "url": theta_url}
        response = requests.post(instance_url, headers=headers, json=payload, timeout=30)
        result = _evaluate_proxy_ping(response, theta_url)
        if result.ok:
            logger.info("Proxy ping OK: %s", result.message())
        else:
            logger.warning("Proxy ping failed: %s", result.message())
        return result
    except requests.RequestException as exc:
        logger.warning("Proxy ping request failed: %s", exc)
        return PingProxyResult(
            ok=False,
            proxy_http_status=None,
            proxy_ok=False,
            theta_status_code=None,
            theta_ok=False,
            theta_url=theta_url,
            error=str(exc),
        )


def ping_proxy_v2() -> PingProxyResult:
    return _ping_proxy(V2_PING_THETA_URL, "v2")


def ping_proxy_v3() -> PingProxyResult:
    return _ping_proxy(V3_PING_THETA_URL, "v3")


if get_proxy_url() is None:
    logger.info("No Proxy URL found. ThetaData API will default to direct access")
else:
    logger.info(f"Using Proxy URL: {get_proxy_url()}")
