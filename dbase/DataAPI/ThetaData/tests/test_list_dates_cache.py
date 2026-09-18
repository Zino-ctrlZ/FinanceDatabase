"""Tests for dbase-owned expired-only list_dates cache."""

from datetime import datetime

from dbase.DataAPI.ThetaData.list_dates_cache import (
    LIST_DATE_CACHE,
    get_listed_option_dates,
)


def test_expired_list_dates_are_cached(monkeypatch) -> None:
    """Second call for an expired contract does not hit the vendor."""
    calls = {"n": 0}
    dates = [datetime(2020, 1, 2), datetime(2020, 1, 3)]

    def _fake_list_dates(**kwargs):
        calls["n"] += 1
        return dates

    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.switcher.list_dates",
        _fake_list_dates,
    )
    opttick_dates = get_listed_option_dates("AAPL", 150.0, "C", "2020-01-17")
    assert opttick_dates == dates
    again = get_listed_option_dates("AAPL", 150.0, "C", "2020-01-17")
    assert again == dates
    assert calls["n"] == 1
    key = "AAPL20200117C150"
    if key in LIST_DATE_CACHE:
        del LIST_DATE_CACHE[key]


def test_live_list_dates_are_not_cached(monkeypatch) -> None:
    """Non-expired contracts call the vendor every time."""
    calls = {"n": 0}

    def _fake_list_dates(**kwargs):
        calls["n"] += 1
        return [datetime(2026, 9, 1)]

    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.switcher.list_dates",
        _fake_list_dates,
    )
    get_listed_option_dates("AAPL", 150.0, "C", "2099-01-15")
    get_listed_option_dates("AAPL", 150.0, "C", "2099-01-15")
    assert calls["n"] == 2
