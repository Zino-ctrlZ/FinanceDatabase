"""
ThetaData API Request Latency Logging Module
=============================================

This module provides automatic logging of API request latencies for ThetaData API calls.
It tracks response times and saves them to CSV files for performance monitoring and analysis.

Overview
--------
The logging system works transparently in the background:
1. API requests automatically submit latency data via _submit_log()
2. Latency entries are collected in an in-memory LOGS_BUCKET
3. On program exit (or signal), all entries are written to CSV
4. Logs are automatically archived when they exceed size threshold

Features
--------
- Automatic latency tracking for all ThetaData API requests
- CSV export with URL and response time in seconds
- Automatic log rotation and archiving at 50,000 rows
- Signal handling for graceful shutdown (SIGTERM, SIGINT)
- Archive management to prevent unbounded log growth

Log Structure
-------------
CSV Format:
    url : str
        The full URL that was requested
    latency_seconds : float
        Request latency in seconds (includes network + processing time)

File Locations:
    - Active log: $GEN_CACHE_PATH/theta_latency/latency_log.csv
    - Archives: $GEN_CACHE_PATH/theta_latency/archive/latency_log_YYYYMMDD_HHMMSS.csv

Usage
-----
Logging happens automatically via the _fetch_data() function in utils.py:

.. code-block:: python

    # In utils.py _fetch_data function
    response = requests.get(theta_url, params=params)
    _submit_log(url, response)  # Automatically logs latency

    # On exit, logs are automatically saved via signal handlers

Manual Logging (if needed):
.. code-block:: python

    from dbase.DataAPI.ThetaData.log import _submit_log, _log_latency
    import requests

    # Make request
    response = requests.get('http://example.com/api')

    # Submit to log bucket
    _submit_log(url='http://example.com/api', response=response)

    # Force write to disk (normally happens on exit)
    _log_latency()

Analyzing Logs
--------------
.. code-block:: python

    import pandas as pd

    # Load current log
    log_df = pd.read_csv('.cache/theta_latency/latency_log.csv')

    # Calculate statistics
    print(f"Mean latency: {log_df['latency_seconds'].mean():.4f}s")
    print(f"95th percentile: {log_df['latency_seconds'].quantile(0.95):.4f}s")
    print(f"Max latency: {log_df['latency_seconds'].max():.4f}s")

    # Find slow endpoints
    slow_requests = log_df[log_df['latency_seconds'] > 1.0]
    print(f"\\nSlow requests (>1s): {len(slow_requests)}")

    # Group by endpoint
    log_df['endpoint'] = log_df['url'].str.extract(r'(v[23]/[^?]+)')
    endpoint_stats = log_df.groupby('endpoint')['latency_seconds'].agg(['count', 'mean', 'max'])
    print(endpoint_stats)

Configuration
-------------
Environment Variables:
    GEN_CACHE_PATH : str
        Base directory for cache files (default: ".cache")

Constants:
    LOGS_BUCKET : list
        In-memory buffer for pending log entries (URL, response pairs)

    archive_threshold : int
        Number of rows before archiving (default: 50,000)

Signal Handling
---------------
The module automatically registers cleanup handlers:
    - Normal exit: via register_signal("exit")
    - SIGTERM: via register_signal(signal.SIGTERM)
    - SIGINT (Ctrl+C): via register_signal(signal.SIGINT)

When any of these triggers, _log_latency() is called to flush LOGS_BUCKET to disk.

Performance Considerations
--------------------------
- Latency logging adds minimal overhead (<1ms per request)
- In-memory buffering prevents I/O on every request
- Archiving prevents unbounded log file growth
- CSV format enables easy analysis with pandas/Excel

Notes
-----
- Logs are only written on program exit/termination
- If program crashes unexpectedly, buffered logs may be lost
- Archive files are never deleted automatically
- Latency includes total response time (network + ThetaData processing)

See Also
--------
- utils._fetch_data : Main API request function that submits logs
- proxy.py : Proxy configuration that affects latency
"""

import requests
import pandas as pd
import os
from pathlib import Path
import signal
from trade import register_signal
from trade.helpers.Logging import setup_logger

LOGS_BUCKET = []
logger = setup_logger("dbase.ThetaData.log", stream_log_level="INFO")


def _submit_log(url: str, response: requests.Response) -> None:
    """Submits a log entry for the request latency.

    Args:
        url (str): The URL that was requested.
        response (requests.Response): The response object from the request.
    """
    LOGS_BUCKET.append((url, response))


def _log_latency(archive_threshold: int = 50_000) -> None:
    """Logs the latency of a request to a CSV file.

    Args:
        url (str): The URL that was requested.
        response (requests.Response): The response object from the request.
        archive_threshold (int, optional): The number of rows after which to archive the log file. Defaults to 50,000.
    """
    cache_path = Path(os.environ.get("GEN_CACHE_PATH", ".cache"))
    cache_path.mkdir(parents=True, exist_ok=True)
    csv_path = cache_path / "theta_latency" / "latency_log.csv"
    archive_path = cache_path / "theta_latency" / "archive"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.mkdir(parents=True, exist_ok=True)
    for url, response in LOGS_BUCKET:
        elapsed_time = response.elapsed.total_seconds()
        log_entry = pd.DataFrame(
            {
                "url": [url],
                "latency_seconds": [elapsed_time],
            }
        )
        if csv_path.exists():
            existing_log = pd.read_csv(csv_path)
            log_entry = pd.concat([existing_log, log_entry], ignore_index=True)
            if len(log_entry) >= archive_threshold:
                archive_file = archive_path / f"latency_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                log_entry.to_csv(archive_file, index=False)
                log_entry = pd.DataFrame(columns=["url", "latency_seconds"])
            log_entry.to_csv(csv_path, mode="w", header=True, index=False)
        else:
            log_entry.to_csv(csv_path, mode="w", header=True, index=False)

    logger.info(f"Logged {len(LOGS_BUCKET)} latency entries.")
    LOGS_BUCKET.clear()
    logger.info("Cleared LOGS_BUCKET after logging.")


# Register the cleanup function to run on exit using register_signal
register_signal("exit", _log_latency)  # Handles normal program exit
register_signal(signal.SIGTERM, _log_latency)  # Handles termination signal
register_signal(signal.SIGINT, _log_latency)  # Handles Ctrl+C
