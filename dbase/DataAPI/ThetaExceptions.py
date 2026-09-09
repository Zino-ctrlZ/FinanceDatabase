"""ThetaData HTTP/status-code exceptions and response mapping.

Vendor docs list 47x/57x codes plus 404/429. Generic HTTP 400 is undocumented;
we only specialize it when the body is the future-session rejection.

Core Classes:
    ThetaDataContainsFutureDateError: 400 whose payload says the date is in the future.
    ThetaDataUnknownError: Unmapped status codes, including other 400s.

Core Functions:
    raise_thetadata_exception: Map a terminal or proxy response to a domain error.
    is_thetadata_exception: True for ThetaData domain errors.

Usage:
    >>> raise_thetadata_exception(response, params={}, proxy=None)
"""

from typing import Any, Optional

from trade.helpers.Logging import setup_logger

logger = setup_logger("dbase.DataAPI.ThetaExceptions")

## Vendor text for the undocumented 400 (not 473 INVALID_PARAMS).
_FUTURE_DATE_SNIPPET = "future date"

class MissingColumnError(Exception):
    """Exception raised when a required column is missing in the data."""

    pass
class ThetDataNoImplementation(Exception):
    """Exception raised when the data is not available/not implemented."""

    pass


class ThetaDataOSLimit(Exception):
    """Exception raised when OS limits are reached.
    Best to retry again.
    """

    pass


class ThetaDataGeneral(Exception):
    """General exception for ThetaData."""

    pass


class ThetaDataPermission(Exception):
    """Permission error for ThetaData."""

    pass


class ThetaDataNotFound(Exception):
    """Exception raised when the data is not found."""

    pass


class ThetaDataInvalidParameter(Exception):
    """Exception raised when the parameter is invalid."""

    pass


class ThetaDataDisconnected(Exception):
    """Exception raised when the data is disconnected."""

    pass


class ThetaDataParseError(Exception):
    """Exception raised when there is a parsing error."""

    pass


class ThetaDataWrongIP(Exception):
    """Exception raised when the IP address is wrong."""

    pass


class ThetaDataNoPageFound(Exception):
    """Exception raised when the page is not found."""

    pass


class ThetaDataLargeData(Exception):
    """Exception raised when the data is too large."""

    pass


class ThetaDataServerRestart(Exception):
    """Exception raised when the server is restarting."""

    pass


class ThetaDataUncaughtException(Exception):
    """Exception raised for uncaught errors."""

    pass


class ThetaDataUnknownError(Exception):
    """Exception raised for unknown errors."""

    pass


class ThetaDataContainsFutureDateError(Exception):
    """Raised when ThetaData rejects a session that is still in the future.

    Vendor HTTP 400 with a body containing ``future date``. Not every 400:
    other bad requests stay ``ThetaDataUnknownError``.
    """

    pass


def _thetadata_response_message(response: Any, proxy: Optional[str]) -> str:
    """Collect error text from a direct terminal response or proxy JSON wrapper.

    Args:
        response: ``requests.Response`` (or test double) from the terminal or proxy.
        proxy: Proxy base URL when the call went through the proxy, else None.

    Returns:
        Concatenated body text used to classify undocumented 400s.
    """
    chunks = []
    text = getattr(response, "text", None)
    if text:
        chunks.append(str(text))
    if proxy is not None:
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, dict) and body.get("data") is not None:
            chunks.append(str(body.get("data")))
    return " ".join(chunks)


def _is_future_date_rejection(code: int, message: str) -> bool:
    """Return True for the undocumented future-session 400.

    Args:
        code: Inner ThetaData status code (proxy) or HTTP status (direct).
        message: Combined response body text.

    Returns:
        True only when ``code`` is 400 and the body mentions a future date.
    """
    if code != 400:
        return False
    return _FUTURE_DATE_SNIPPET in (message or "").lower()


def raise_thetadata_exception(response, params=None, proxy=None):
    """
    Raise an exception if the response indicates an error.
    """
    if params is None:
        params = {}

    if proxy is None:
        code = response.status_code
        params["url"] = response.url
    else:
        code = response.json()["status_code"]
        params["url"] = response.json()["url"]
    message = _thetadata_response_message(response, proxy)
    if code == 404:
        raise ThetDataNoImplementation(f"The requested feature is not implemented. Parameters: {params}")
    elif code == 429:
        raise ThetaDataOSLimit("OS Limit reached. Please try again.")
    elif code == 470:
        raise ThetaDataGeneral(f"General error occurred. Parameters: {params}")
    elif code == 471:
        raise ThetaDataPermission("Permission denied.")
    elif code == 472:
        raise ThetaDataNotFound(f"Data not found for the given parameters: {params}")
    elif code == 473:
        raise ThetaDataInvalidParameter(f"Invalid parameter provided: {params}, if error persists, update terminal.")
    elif code == 474:
        raise ThetaDataDisconnected("Disconnected from the server.")
    elif code == 475:
        raise ThetaDataParseError(f"Error parsing the response. Parameters: {params}")
    elif code == 476:
        raise ThetaDataWrongIP("Wrong IP address provided.")
    elif code == 477:
        raise ThetaDataNoPageFound(f"No page found for the given request. Parameters: {params}")
    elif code == 570:
        raise ThetaDataLargeData(f"Data size is too large. Parameters: {params}")
    elif code == 571:
        raise ThetaDataServerRestart("Server is restarting. Retry in a few seconds.")
    elif code == 572:
        raise ThetaDataUncaughtException(f"Uncaught exception occurred. Parameters: {params}")
    elif _is_future_date_rejection(code, message):
        ## Do not map every HTTP 400; only the undocumented future-session body.
        raise ThetaDataContainsFutureDateError(
            f"Request date is in the future. Status code: {code}, Message: `{message}`, Parameters: {params}"
        )
    elif code == 200:
        return
    else:
        raise ThetaDataUnknownError(
            f"Unknown error occurred. Status code: {code}, Message: `{response.text}`, Parameters: {params}"
        )


def is_thetadata_exception(e):
    """
    Check if the exception is a ThetaData exception.
    """
    return isinstance(
        e,
        (
            ThetDataNoImplementation,
            ThetaDataOSLimit,
            ThetaDataGeneral,
            ThetaDataPermission,
            ThetaDataNotFound,
            ThetaDataInvalidParameter,
            ThetaDataDisconnected,
            ThetaDataParseError,
            ThetaDataWrongIP,
            ThetaDataNoPageFound,
            ThetaDataLargeData,
            ThetaDataServerRestart,
            ThetaDataUncaughtException,
            ThetaDataUnknownError,
            ThetaDataContainsFutureDateError,
        ),
    )
