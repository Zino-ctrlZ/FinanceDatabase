"""Tests for ticker-split missing-root raises."""

import pandas as pd
import pytest

from dbase.DataAPI.ThetaData.v2 import (
    _filter_ticker_history_window,
    resolve_ticker_history,
)
from dbase.DataAPI.ThetaExceptions import ThetaDataNotFound
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS


def test_resolve_ticker_history_raises_when_old_root_missing() -> None:
    """FB window before META rename raises with old_missing and requested params."""
    change_date = TICK_CHANGE_ALIAS["META"][-1]
    old_tick = TICK_CHANGE_ALIAS["META"][0]
    new_tick = TICK_CHANGE_ALIAS["META"][1]

    def _callable(**kwargs):
        raise ThetaDataNotFound("Data not found")

    kwargs = {
        "symbol": "META",
        "start_date": "2018-01-01",
        "end_date": "2022-01-21",
        "exp": "2022-01-21",
        "strike": 405.0,
        "right": "C",
    }
    with pytest.raises(ThetaDataNotFound, match="old_missing=True") as exc:
        resolve_ticker_history(kwargs, _callable, _type="historical")
    msg = str(exc.value)
    assert f"old_symbol={old_tick}" in msg
    assert f"new_symbol={new_tick}" in msg
    assert f"change_date={change_date}" in msg
    assert "new_attempted=False" in msg or "new_missing=False" in msg
    assert "params=" in msg


def test_resolve_ticker_history_raises_when_both_missing() -> None:
    """Span across the rename with empty both sides names FB and META as missing."""
    def _callable(**kwargs):
        raise ThetaDataNotFound("Data not found")

    kwargs = {
        "symbol": "META",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "exp": "2022-12-16",
        "strike": 100.0,
        "right": "C",
    }
    with pytest.raises(ThetaDataNotFound, match="old_missing=True") as exc:
        resolve_ticker_history(kwargs, _callable, _type="historical")
    assert "new_missing=True" in str(exc.value)


def test_resolve_ticker_history_returns_the_successful_side() -> None:
    """One successful root is returned without concatenating an empty sibling."""
    df_new = pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2022-08-01")])

    def _callable(**kwargs):
        if kwargs["symbol"] == "META":
            return df_new
        raise ThetaDataNotFound("Data not found")

    kwargs = {
        "symbol": "META",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "exp": "2022-12-16",
        "strike": 100.0,
        "right": "C",
    }
    out = resolve_ticker_history(kwargs, _callable, _type="historical")
    pd.testing.assert_frame_equal(out, df_new)


def test_filter_ticker_history_window_keeps_eod_close_on_end_date() -> None:
    """16:00 EOD stamps on the end date are kept (midnight end_date must not drop them)."""
    frame = pd.DataFrame(
        {"Midpoint": [52.45, 95.25]},
        index=[
            pd.Timestamp("2026-09-18 16:00:00"),
            pd.Timestamp("2026-09-21 16:00:00"),
        ],
    )
    same_day = _filter_ticker_history_window(frame, "2026-09-21", "2026-09-21")
    assert list(same_day.index) == [pd.Timestamp("2026-09-21 16:00:00")]
    window = _filter_ticker_history_window(frame, "2026-09-18", "2026-09-21")
    assert len(window) == 2


def test_resolve_ticker_history_keeps_same_day_eod_bar() -> None:
    """META post-rename same-day EOD at 16:00 survives the window filter."""
    df_new = pd.DataFrame(
        {"Midpoint": [95.25]},
        index=[pd.Timestamp("2026-09-21 16:00:00")],
    )

    def _callable(**kwargs):
        if kwargs["symbol"] == "META":
            return df_new
        raise ThetaDataNotFound("Data not found")

    kwargs = {
        "symbol": "META",
        "start_date": "2026-09-21",
        "end_date": "2026-09-21",
        "exp": "2027-03-19",
        "strike": 750.0,
        "right": "C",
    }
    out = resolve_ticker_history(kwargs, _callable, _type="historical")
    pd.testing.assert_frame_equal(out, df_new)
