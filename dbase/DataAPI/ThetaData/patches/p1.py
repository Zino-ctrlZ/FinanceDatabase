from typing import List, Union
import numpy as np
from .main import ThetaDataPatchProcessor

LIST_DATES_REMOVAL_BUCKET = {
    ("AAPL", "2020-10-16", "C", 96.25): "2020-08-25",
    ("AAPL", "2020-10-16", "C", 100.0): "2020-08-25",
}


def _remove_incorrect_date(d: Union[List[str], np.ndarray], dt: str) -> List[str]:
    """
    Remove the incorrect date from the list of dates.
    Parameters    ----------
    d : List[str] or np.ndarray
        List of dates to be filtered.
    dt : str
        The date to be removed from the list.
    Returns    -------
    List[str]
        The list of dates with the incorrect date removed."""
    if dt in d:
        if isinstance(d, np.ndarray):
            d = d[d != dt]

        else:

            d.remove(dt)


    return d


def _remove_incorrect_date_for_option(
     result: Union[List[str], np.ndarray], symbol: str, exp: str, right: str, strike: float, *args, **kwargs
) -> Union[List[str], np.ndarray]:
    """
    Remove the incorrect date from the list of dates for a specific option contract.
    Parameters    ----------
    result : List[str] or np.ndarray
        List of dates to be filtered.
    symbol : str
        The symbol of the option contract.
    exp : str
        The expiration date of the option contract.
    right : str
        The right of the option contract (C or P).
    strike : float
        The strike price of the option contract.
    Returns    -------
    List[str] or np.ndarray
        The list of dates with the incorrect date removed."""

    key = (symbol, exp, right, strike)
    if not len(result):
        return result

    if key in LIST_DATES_REMOVAL_BUCKET:
        dt = LIST_DATES_REMOVAL_BUCKET[key]
        result =  _remove_incorrect_date(result, dt)
    return result

# Register the patch function for the list_dates API function
ThetaDataPatchProcessor.register_patch("list_dates", _remove_incorrect_date_for_option)