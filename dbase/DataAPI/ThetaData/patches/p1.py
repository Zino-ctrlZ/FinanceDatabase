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


def _remove_aapl_split_artifact(
    result: Union[List[str], np.ndarray], symbol: str, *args, **kwargs
) -> Union[List[str], np.ndarray]:
    """Remove known AAPL split-era artifact date from list_dates responses.

    The vendor occasionally returns an out-of-sequence "2020-08-25" before
    valid dates beginning at "2020-08-31" around the split period.
    """
    if symbol != "AAPL":
        return result

    dates = result.tolist() if isinstance(result, np.ndarray) else list(result)
    if not dates:
        return result

    marker_date = "2020-08-25"
    first_valid_after_split = "2020-08-31"
    missing_gap_days = ("2020-08-26", "2020-08-27", "2020-08-28")
    date_set = set(dates)

    if marker_date not in date_set or first_valid_after_split not in date_set:
        return result

    marker_before_valid = dates.index(marker_date) < dates.index(first_valid_after_split)
    has_expected_gap = all(day not in date_set for day in missing_gap_days)

    if marker_before_valid and has_expected_gap:
        return _remove_incorrect_date(result, marker_date)

    return result


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
        result = _remove_incorrect_date(result, dt)

    # Keep symbol-specific guardrail here so direct calls to this function
    # still get split-era cleanup even without patch-processor orchestration.
    result = _remove_aapl_split_artifact(result, symbol)

    return result


# Register the patch function for the list_dates API function
ThetaDataPatchProcessor.register_patch("list_dates", _remove_incorrect_date_for_option)
ThetaDataPatchProcessor.register_patch("list_dates", _remove_aapl_split_artifact)
