"""Tests for 472 omit-or-raise on vendor-listed sessions."""

from typing import Any, Dict, Optional, Set
from datetime import datetime

import pandas as pd
import pytest

from dbase.DataAPI.ThetaExceptions import (
    ThetaDataContainsFutureDateError,
    ThetaDataNotFound,
    ThetaDataUnknownError,
    raise_thetadata_exception,
)
from dbase.DataAPI.ThetaData.v3.utils import (
    _frame_for_future_date,
    _frame_for_listed_not_found,
    _format_coverage_location,
    _iso_session_date,
    _listed_dates_in_request_window,
    _new_dataframe_formatting,
    _expected_listed_sessions,
    _today_session_iso,
    enforce_listed_quote_coverage,
)
from dbase.DataAPI.ThetaData.utils import bootstrap_ohlc
from dbase.DataAPI.ThetaData.v3.vars import SETTINGS, ListedSessionNotFoundPolicy, EOD_OHLC, HISTORICAL_QUOTE


@pytest.fixture(autouse=True)
def _quote_472_csv_under_tmp(tmp_path, monkeypatch):
    """Keep 472 CSV writes out of the real QuantTools .cache during tests."""
    monkeypatch.setattr("trade.GEN_CACHE_PATH", tmp_path)
    return tmp_path


class _DummyResponse:
    """Minimal response double for ``raise_thetadata_exception``."""

    def __init__(
        self,
        *,
        status_code: int,
        text: str,
        url: str = "http://localhost:25503/v3/option/history/quote",
        json_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store status, body text, and optional proxy JSON."""
        self.status_code = status_code
        self.text = text
        self.url = url
        self._json_body = json_body

    def json(self) -> Dict[str, Any]:
        """Return the proxy wrapper body."""
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body


def test_iso_session_date_accepts_yyyymmdd() -> None:
    """YYYYMMDD request params match list_dates ISO strings."""
    assert _iso_session_date("20260824") == "2026-08-24"


def test_listed_dates_clip_to_request_window() -> None:
    """Expected sessions ignore listed days outside the request window."""
    listed: Set[str] = {"2020-08-28", "2020-12-14", "2021-08-02"}
    assert _listed_dates_in_request_window(listed, "2020-12-01", "2021-08-02") == [
        "2020-12-14",
        "2021-08-02",
    ]


def test_listed_472_returns_empty_frame() -> None:
    """A 472 on a list_dates session omits the day."""
    params = {
        "symbol": "FXI",
        "expiration": "20261120",
        "strike": "33.00",
        "right": "P",
        "date": "20260824",
    }
    exc = ThetaDataNotFound("Data not found")
    frame = _frame_for_listed_not_found(params, {"2026-08-24", "2026-08-25"}, exc)
    assert frame.empty


def test_listed_472_writes_quote_csv(tmp_path) -> None:
    """Quote-history 472s append option metadata, date, and URL to .cache CSV."""
    params = {
        "symbol": "AMD",
        "expiration": "20251017",
        "strike": "220.00",
        "right": "C",
        "date": "20240221",
        "interval": "30m",
    }
    exc = ThetaDataNotFound("No data found for your request")
    _frame_for_listed_not_found(params, {"2024-02-21"}, exc, url=HISTORICAL_QUOTE)
    csv_path = tmp_path / "thetadata" / "quote_472.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "AMD" in text
    assert "2024-02-21" in text
    assert "history/quote" in text
    assert "20240221" in text


def test_unlisted_472_omits_session() -> None:
    """A 472 for an unlisted weekday must not abort the range pool."""
    params = {"date": "20260824", "symbol": "FXI"}
    exc = ThetaDataNotFound("Data not found")
    frame = _frame_for_listed_not_found(params, {"2026-08-25"}, exc)
    assert frame.empty


def test_unknown_calendar_472_omits_session() -> None:
    """When list_dates was not prefetched, 472 still omits that worker date."""
    params = {"date": "20260824"}
    exc = ThetaDataNotFound("Data not found")
    frame = _frame_for_listed_not_found(params, None, exc)
    assert frame.empty


def test_enforce_drops_extra_days_not_on_list_dates() -> None:
    """Rows outside list_dates ∩ request window are dropped."""
    idx = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-12-14"])
    df = pd.DataFrame({"Midpoint": [1.0, 2.0, 3.0]}, index=idx)
    out = enforce_listed_quote_coverage(
        df,
        start_date="2020-08-01",
        end_date="2020-12-31",
        symbol="TSLA",
        exp="2021-09-17",
        right="C",
        strike=1020.0,
        listed_dates={"2020-08-28", "2020-12-14"},
    )
    assert list(pd.to_datetime(out.index).strftime("%Y-%m-%d")) == [
        "2020-08-28",
        "2020-12-14",
    ]


def test_enforce_raise_on_missing_listed_session() -> None:
    """Default policy raises when a listed session has no EOD row."""
    old = SETTINGS.listed_session_not_found
    SETTINGS.listed_session_not_found = ListedSessionNotFoundPolicy.RAISE
    try:
        df = pd.DataFrame({"Midpoint": [1.0]}, index=pd.to_datetime(["2026-08-24"]))
        with pytest.raises(ThetaDataNotFound, match="url=") as exc:
            enforce_listed_quote_coverage(
                df,
                start_date="2026-08-24",
                end_date="2026-08-25",
                symbol="FXI",
                exp="2026-11-20",
                right="P",
                strike=33.0,
                listed_dates={"2026-08-24", "2026-08-25"},
                endpoint=EOD_OHLC,
            )
        msg = str(exc.value)
        assert "option/history/eod" in msg
        assert "FXI" in msg
        assert "20260825" in msg or "2026-08-25" in msg
    finally:
        SETTINGS.listed_session_not_found = old


def test_enforce_omit_returns_without_missing_dates() -> None:
    """OMIT policy returns the frame without the missing listed sessions."""
    old = SETTINGS.listed_session_not_found
    SETTINGS.listed_session_not_found = ListedSessionNotFoundPolicy.OMIT
    try:
        df = pd.DataFrame({"Midpoint": [1.0]}, index=pd.to_datetime(["2026-08-24"]))
        out = enforce_listed_quote_coverage(
            df,
            start_date="2026-08-24",
            end_date="2026-08-25",
            symbol="FXI",
            exp="2026-11-20",
            right="P",
            strike=33.0,
            listed_dates={"2026-08-24", "2026-08-25"},
        )
        assert list(pd.to_datetime(out.index).strftime("%Y-%m-%d")) == ["2026-08-24"]
        assert "Midpoint" in out.columns
    finally:
        SETTINGS.listed_session_not_found = old


def test_enforce_omit_empty_frame_keeps_columns() -> None:
    """OMIT on a fully empty fetch returns the empty frame when listed days are missing."""
    old = SETTINGS.listed_session_not_found
    SETTINGS.listed_session_not_found = ListedSessionNotFoundPolicy.OMIT
    try:
        df = pd.DataFrame(columns=["Midpoint"])
        df.index = pd.DatetimeIndex([])
        out = enforce_listed_quote_coverage(
            df,
            start_date="2022-01-03",
            end_date="2022-01-21",
            symbol="META",
            exp="2022-01-21",
            right="C",
            strike=405.0,
            listed_dates={"2022-01-03", "2022-01-21"},
        )
        assert out.empty
        assert "Midpoint" in out.columns
    finally:
        SETTINGS.listed_session_not_found = old


def test_formatting_empty_frame_skips_timestamp_check() -> None:
    """An all-omit concat must not raise MissingColumnError."""
    out = _new_dataframe_formatting(pd.DataFrame(), interval="30m")
    assert out.empty


def test_future_date_400_maps_to_contains_future_date() -> None:
    """Proxy 400 with a future-date body is not ThetaDataUnknownError."""
    body = {
        "data": "Date range contains future date; end must be before or equal to today",
        "url": "http://localhost:25503/v3/option/history/quote?date=20260909",
        "status_code": 400,
    }
    response = _DummyResponse(status_code=200, text=str(body), json_body=body)
    with pytest.raises(ThetaDataContainsFutureDateError):
        raise_thetadata_exception(response, params={}, proxy="http://proxy")


def test_other_400_stays_unknown() -> None:
    """A 400 without a future-date body stays unmapped."""
    body = {
        "data": "malformed query",
        "url": "http://localhost:25503/v3/option/history/quote",
        "status_code": 400,
    }
    response = _DummyResponse(status_code=200, text=str(body), json_body=body)
    with pytest.raises(ThetaDataUnknownError):
        raise_thetadata_exception(response, params={}, proxy="http://proxy")


def test_future_date_omit_returns_empty_frame() -> None:
    """A future-date 400 omits the session from the range concat."""
    params = {"symbol": "SLV", "date": "20260909", "expiration": "20261218"}
    frame = _frame_for_future_date(params, ThetaDataContainsFutureDateError("future"))
    assert frame.empty


def test_coverage_location_uses_reconstructed_url() -> None:
    """EOD coverage messages include a pasteable v3 URL."""
    loc = _format_coverage_location(
        url=None,
        endpoint=EOD_OHLC,
        symbol="FXI",
        start_date="2026-08-24",
        end_date="2026-08-25",
        exp="2026-11-20",
        right="P",
        strike=33.0,
    )
    assert loc.startswith("url=")
    assert "option/history/eod" in loc
    assert "FXI" in loc


def test_coverage_location_quote_pins_first_missing_date() -> None:
    """Quote holes report the per-session URL, not a range."""
    loc = _format_coverage_location(
        url=None,
        endpoint=HISTORICAL_QUOTE,
        symbol="META",
        start_date="2022-01-03",
        end_date="2022-01-21",
        exp="2022-01-21",
        right="C",
        strike=405.0,
        missing=["2022-01-03", "2022-01-21"],
    )
    assert "option/history/quote" in loc
    assert "date=20220103" in loc
    assert "start_date" not in loc


def test_coverage_location_falls_back_to_endpoint() -> None:
    """When no URL can be built, the endpoint path is still in the message."""
    loc = _format_coverage_location(
        url=None,
        endpoint=None,
        symbol="FXI",
        start_date="2026-08-24",
        end_date="2026-08-25",
        exp="2026-11-20",
        right="P",
        strike=33.0,
    )
    assert loc == "endpoint=unknown"


def test_bootstrap_ohlc_empty_frame_does_not_raise() -> None:
    """Empty quote concat has no Midpoint; bootstrap must not KeyError."""
    empty = pd.DataFrame()
    out = bootstrap_ohlc(empty)
    assert out.empty
    timestamp_only = pd.DataFrame(columns=["timestamp"])
    out2 = bootstrap_ohlc(timestamp_only)
    assert out2.empty
    assert list(out2.columns) == ["timestamp"]


def test_expected_listed_sessions_pads_today(monkeypatch) -> None:
    """Unexpired weekday today inside the window is always an expected session."""
    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.v3.utils.ny_now",
        lambda: datetime(2026, 9, 18, 21, 27),
    )
    out = _expected_listed_sessions(
        {"2026-09-04", "2026-09-17"},
        "2026-09-04",
        "2026-09-18",
        "2026-11-20",
    )
    assert out[-1] == "2026-09-18"
    assert "2026-09-17" in out
    assert _today_session_iso("2026-09-04", "2026-09-18", "2026-11-20") == "2026-09-18"


def test_today_session_iso_skips_expired_and_weekend(monkeypatch) -> None:
    """No pad after expiration or on Saturday."""
    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.v3.utils.ny_now",
        lambda: datetime(2026, 9, 18, 12, 0),
    )
    assert _today_session_iso("2026-09-04", "2026-09-18", "2026-09-17") is None
    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.v3.utils.ny_now",
        lambda: datetime(2026, 9, 19, 12, 0),
    )
    assert _today_session_iso("2026-09-04", "2026-09-19", "2026-11-20") is None


def test_eod_coverage_keeps_unlisted_today_print(monkeypatch) -> None:
    """End-of-day coverage must not drop today's print when list_dates lags."""
    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.v3.utils.ny_now",
        lambda: datetime(2026, 9, 18, 21, 27),
    )
    idx = pd.to_datetime(["2026-09-17", "2026-09-18"])
    df = pd.DataFrame({"Midpoint": [1.64, 2.25]}, index=idx)
    out = enforce_listed_quote_coverage(
        df,
        start_date="2026-09-04",
        end_date="2026-09-18",
        symbol="AAPL",
        exp="2026-11-20",
        right="C",
        strike=380.0,
        listed_dates={"2026-09-17"},
        endpoint="http://127.0.0.1:25510/v2/hist/option/eod",
    )
    assert list(pd.to_datetime(out.index).strftime("%Y-%m-%d")) == [
        "2026-09-17",
        "2026-09-18",
    ]


def test_quote_coverage_keeps_unlisted_today_print(monkeypatch) -> None:
    """Quote-to-EOD must not drop today's print when list_dates lags."""
    monkeypatch.setattr(
        "dbase.DataAPI.ThetaData.v3.utils.ny_now",
        lambda: datetime(2026, 9, 18, 21, 27),
    )
    idx = pd.to_datetime(["2026-09-17", "2026-09-18"])
    df = pd.DataFrame({"Midpoint": [1.64, 2.25]}, index=idx)
    out = enforce_listed_quote_coverage(
        df,
        start_date="2026-09-04",
        end_date="2026-09-18",
        symbol="AAPL",
        exp="2026-11-20",
        right="C",
        strike=380.0,
        listed_dates={"2026-09-17"},
        endpoint=HISTORICAL_QUOTE,
    )
    assert list(pd.to_datetime(out.index).strftime("%Y-%m-%d")) == [
        "2026-09-17",
        "2026-09-18",
    ]
