from trade.helpers.Logging import setup_logger

logger = setup_logger("dbase.DataAPI.ThetaExceptions")

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
        ),
    )
