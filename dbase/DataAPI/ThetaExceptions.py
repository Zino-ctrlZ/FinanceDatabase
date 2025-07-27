import backoff
from trade.helpers.Logging import setup_logger
import requests
logger = setup_logger('dbase.DataAPI.ThetaExceptions')

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


def raise_thetadata_exception(response:requests.Response, params = {}, proxy = None):
    """
    Raise an exception if the response indicates an error.
    """
    if proxy is None:
        code = response.status_code
        params['url'] = response.url
        message = response.reason
    else:
        code = response.json()['status_code']
        params['url'] = response.json()['url']
        message = response.json().get('message', 'No message provided')
    if code == 404:
        raise ThetDataNoImplementation(f"The requested feature is not implemented. Parameters: {params}. Messsage: {message}")
    elif code == 429:
        raise ThetaDataOSLimit("OS Limit reached. Please try again. Messsage: {message}")
    elif code == 470:
        raise ThetaDataGeneral(f"General error occurred. Parameters: {params} Messsage: {message}")
    elif code == 471:
        raise ThetaDataPermission(f"Permission denied. Messsage: {message}")
    elif code == 472:
        raise ThetaDataNotFound(f"Data not found for the given parameters: {params} Messsage: {message}")
    elif code == 473:
        raise ThetaDataInvalidParameter(f"Invalid parameter provided: {params}, if error persists, update terminal. Messsage: {message}")
    elif code == 474:
        raise ThetaDataDisconnected("Disconnected from the server. Messsage: {message}")
    elif code == 475:
        raise ThetaDataParseError(f"Error parsing the response. Parameters: {params} Messsage: {message}")
    elif code == 476:
        raise ThetaDataWrongIP("Wrong IP address provided. Messsage: {message}")
    elif code == 477:
        raise ThetaDataNoPageFound(f"No page found for the given request. Parameters: {params} Messsage: {message}")
    elif code == 570:
        raise ThetaDataLargeData(f"Data size is too large. Parameters: {params} Messsage: {message}")
    elif code == 571:
        raise ThetaDataServerRestart(f"Server is restarting. Retry in a few seconds. Messsage: {message}")
    elif code == 572:
        raise ThetaDataUncaughtException(f"Uncaught exception occurred. Parameters: {params} Messsage: {message}")
    elif code == 200:
        return
    else:
        raise ThetaDataUnknownError(f"Unknown error occurred. Status code: {code}, Parameters: {params} Messsage: {message}")
    
def is_thetadata_exception(e):
    """
    Check if the exception is a ThetaData exception.
    """
    return isinstance(e, (
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
        ThetaDataUnknownError
    ))