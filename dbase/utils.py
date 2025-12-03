import pandas as pd
import numpy as np
from datetime import time as dtTime
from trade import HOLIDAY_SET, PRICING_CONFIG


def add_eod_timestamp(index):
    """
    Adds the EOD timestamp to the index

    If the index is a datetime object and the time is 00:00:00, it adds PRICING_CONFIG['MARKET_CLOSE_TIME'] hours to the index.
    If not, it returns the index as is.
    """
    if len(index) == 0:
        return index
    str_time = PRICING_CONFIG["MARKET_CLOSE_TIME"]
    str_time = int(str_time.split(":")[0])
    index = pd.to_datetime(index)
    if np.unique(index.time)[0] == dtTime(0, 0):
        return index + pd.DateOffset(hours=str_time)
    else:
        return index


def default_timestamp(index):
    """
    Default timestamp to 00:00:00 if the index is a datetime object.
    """
    index = pd.to_datetime(index)
    idx_name, idx_freq = index.name, index.freq
    index = pd.DatetimeIndex(
        [x.replace(hour=0, minute=0, second=0) for x in index],
        name=idx_name,
        freq=idx_freq,
    )
    return index


def enforce_bus_hours(series: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    """
    Enforce business hours between two dates.
    """
    bus_hourse = series.index.indexer_between_time(
        PRICING_CONFIG["MARKET_OPEN_TIME"], PRICING_CONFIG["MARKET_CLOSE_TIME"]
    )
    if len(bus_hourse) == 0:
        return series
    else:
        return series.iloc[bus_hourse]


def bus_range(
    start: str, end: str, freq: str, filter_holiday: bool = True
) -> pd.DatetimeIndex:
    """
    Generate a range of business days between two dates.
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    _range = pd.bdate_range(start=start_date, end=end_date, freq=freq).to_list()
    _range = (
        [x for x in _range if x.date().strftime("%Y-%m-%d") not in HOLIDAY_SET]
        if filter_holiday
        else _range
    )
    if "min" in freq.lower():
        return enforce_bus_hours(pd.Series(index=_range)).index.to_list()
    return _range
