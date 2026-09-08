"""Tests for 472 omit-or-raise on vendor-listed sessions."""

from typing import Any, Dict, Optional

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
    _iso_session_date,
    _new_dataframe_formatting,
)


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


def test_unlisted_472_reraise() -> None:
    """A 472 for a session the vendor never listed stays fatal."""
    params = {"date": "20260824", "symbol": "FXI"}
    exc = ThetaDataNotFound("Data not found")
    with pytest.raises(ThetaDataNotFound):
        _frame_for_listed_not_found(params, {"2026-08-25"}, exc)


def test_unknown_calendar_reraise() -> None:
    """When list_dates was not prefetched, 472 still aborts the range."""
    params = {"date": "20260824"}
    exc = ThetaDataNotFound("Data not found")
    with pytest.raises(ThetaDataNotFound):
        _frame_for_listed_not_found(params, None, exc)


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
