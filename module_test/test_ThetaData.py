import pytest
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock

# Import the whole ThetaData module with an alias
import FinanceDatabase.dbase.DataAPI.ThetaData as td


# -------------------------------
# Fixtures for mocks
# -------------------------------
@pytest.fixture
def setup_env(monkeypatch):
    # Mock TICK_CHANGE_ALIAS
    monkeypatch.setattr(td, "TICK_CHANGE_ALIAS", {
        "ABC": ("OLDABC", "NEWABC", "2023-01-01")
    })

    # Mock ThetaDataNotFound
    class DummyError(Exception):
        pass
    monkeypatch.setattr(td, "ThetaDataNotFound", DummyError)

    # Mock logger
    mock_logger = MagicMock()
    monkeypatch.setattr(td, "logger", mock_logger)

    # Mock compare_dates helpers
    mock_compare = MagicMock()
    mock_compare.is_before = MagicMock()
    mock_compare.is_on_or_after = MagicMock()
    monkeypatch.setattr(td, "compare_dates", mock_compare)

    return mock_logger, mock_compare


# -------------------------------
# resolve_ticker_history tests
# -------------------------------
def test_resolve_ticker_history_historical_returns_combined_data(setup_env):
    mock_logger, mock_compare = setup_env
    mock_compare.is_before.return_value = True
    mock_compare.is_on_or_after.return_value = True

    df_old = pd.DataFrame({'price': [100]}, index=[pd.Timestamp('2022-12-31')])
    df_new = pd.DataFrame({'price': [110]}, index=[pd.Timestamp('2023-01-02')])
    mock_callable = MagicMock(side_effect=[df_old, df_new])

    kwargs = {"symbol": "ABC", "start_date": "2022-12-01", "exp": "2023-02-01"}
    result = td.resolve_ticker_history(kwargs, mock_callable, _type="historical")

    assert isinstance(result, pd.DataFrame)
    assert list(result['price'].values) == [110]
    assert mock_callable.call_count == 2
    mock_logger.info.assert_not_called()


# def test_resolve_ticker_history_historical_handles_missing_data(setup_env):
#     mock_logger, mock_compare = setup_env
#     mock_compare.is_before.return_value = True
#     mock_compare.is_on_or_after.return_value = True

#     def callable_side_effect(**kwargs):
#         if kwargs["symbol"] == "OLDABC":
#             raise td.ThetaDataNotFound("No data")
#         return pd.DataFrame({'price': [120]}, index=[pd.Timestamp('2023-01-02')])

#     mock_callable = MagicMock(side_effect=callable_side_effect)
#     kwargs = {"symbol": "ABC", "start_date": "2022-12-01", "exp": "2023-02-01"}
#     result = td.resolve_ticker_history(kwargs, mock_callable, _type="historical")

#     assert isinstance(result, pd.DataFrame)
#     assert not result.empty
#     assert 'price' in result.columns
#     assert result.iloc[0]['price'] == 120
#     assert mock_callable.call_count == 2
#     mock_logger.error.assert_called()


def test_resolve_ticker_history_snapshot_uses_old_symbol_before_change(setup_env):
    mock_logger, mock_compare = setup_env
    mock_compare.is_before.return_value = True
    mock_callable = MagicMock(return_value="old_data")

    kwargs = {"symbol": "ABC", "start_date": "2022-12-01"}
    result = td.resolve_ticker_history(kwargs, mock_callable, _type="snapshot")

    assert result == "old_data"
    mock_callable.assert_called_once()
    called_args = mock_callable.call_args[1]
    assert called_args["symbol"] == "OLDABC"


def test_resolve_ticker_history_snapshot_uses_new_symbol_after_change(setup_env):
    mock_logger, mock_compare = setup_env
    mock_compare.is_before.return_value = False
    mock_callable = MagicMock(return_value="new_data")

    kwargs = {"symbol": "ABC", "start_date": "2023-02-01"}
    result = td.resolve_ticker_history(kwargs, mock_callable, _type="snapshot")

    assert result == "new_data"
    mock_callable.assert_called_once()
    called_args = mock_callable.call_args[1]
    assert called_args["symbol"] == "NEWABC"


# -------------------------------
# request_from_proxy tests
# -------------------------------
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.requests.request")
def test_request_from_proxy_success(mock_request):
    thetaUrl = "https://api.thetadata.com/historical"
    queryparam = {"symbol": "AAPL", "date": "2025-01-01"}
    instanceUrl = "https://proxy.service.com/request"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "ok"}'
    mock_request.return_value = mock_response

    response = td.request_from_proxy(thetaUrl, queryparam, instanceUrl)

    expected_url = f"{thetaUrl}?symbol=AAPL&date=2025-01-01"
    expected_payload = td.json.dumps({"url": expected_url, "method": "GET"})
    expected_headers = {'Content-Type': 'application/json'}

    mock_request.assert_called_once_with(
        "POST",
        instanceUrl,
        headers=expected_headers,
        data=expected_payload
    )
    assert response.status_code == 200


SNAPSHOT_CSV = "symbol,open,high,low,close\nAAPL,150,155,149,154\nMSFT,250,255,248,253"
SNAPSHOT_CSV_STRIKE = "symbol,strike\nAAPL,150\nMSFT,250"

@pytest.mark.parametrize("method_name, url_path", [
    ("ohlc_snapshot", "v2/bulk_snapshot/option/ohlc"),
    ("open_interest_snapshot", "v2/bulk_snapshot/option/quote"),
    ("quote_snapshot", "v2/snapshot/option/quote"),
])
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.request_from_proxy")
def test_snapshot_methods_with_proxy(mock_request_proxy, method_name, url_path):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": SNAPSHOT_CSV}
    mock_request_proxy.return_value = mock_response

    method = getattr(td, method_name)
    df = method("AAPL", proxy="http://proxy.local")

    mock_request_proxy.assert_called_once()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["symbol", "open", "high", "low", "close"]
    assert df.shape[0] == 2
    assert df.iloc[0]["symbol"] == "AAPL"


@patch("FinanceDatabase.dbase.DataAPI.ThetaData.requests.get")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.get_proxy_url")
@pytest.mark.parametrize("method_name, url_path", [
    ("ohlc_snapshot", "v2/bulk_snapshot/option/ohlc"),
    ("open_interest_snapshot", "v2/bulk_snapshot/option/quote"),
    ("quote_snapshot", "v2/snapshot/option/quote"),
])
def test_snapshot_methods_no_proxy(mock_get_proxy, mock_requests_get, method_name, url_path):
    mock_get_proxy.return_value = None
    mock_response = MagicMock()
    mock_response.text = SNAPSHOT_CSV
    mock_requests_get.return_value = mock_response

    method = getattr(td, method_name)
    df = method("AAPL")

    mock_get_proxy.assert_called_once()
    mock_requests_get.assert_called_once()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["symbol", "open", "high", "low", "close"]
    assert df.shape[0] == 2
    assert df.iloc[0]["symbol"] == "AAPL"

# -------------------------------
# list_contracts test
# -------------------------------

@patch("FinanceDatabase.dbase.DataAPI.ThetaData.request_from_proxy")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.requests.get")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.get_proxy_url")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.resolve_ticker_history")
def test_list_contracts_proxy_and_snapshot(mock_resolve, mock_get_proxy, mock_requests_get, mock_request_proxy):
    mock_get_proxy.return_value = "http://proxy.local"
    
    # Mock response with required keys for raise_thetadata_exception
    mock_request_proxy.return_value = MagicMock(
        json=lambda: {
            "status_code": 200,
            "data": SNAPSHOT_CSV_STRIKE,
            "url": "http://proxy.local/fake_endpoint"
        }
    )

    # Also mock resolve_ticker_history if snapshot branch triggers it
    mock_resolve.return_value = pd.DataFrame({"symbol": ["AAPL"], "strike": [150]})

    df = td.list_contracts("AAPL", "2023-01-01", proxy="http://proxy.local")

    assert isinstance(df, pd.DataFrame)
    assert "strike" in df.columns
    # after division by 1000
    assert df['strike'].iloc[0] == 0.15
    assert df['strike'].iloc[1] == 0.25



# -------------------------------
# identify_length tests
# -------------------------------

@pytest.mark.parametrize("string, integer, rt, expected", [
    ("d", 5, False, 5),
    ("w", 2, False, 10),
    ("m", 3, False, 90),
    ("y", 1, False, 252),
    ("q", 2, False, 182),
    ("m", 1, True, 1),
    ("h", 2, True, 120),
    ("d", 3, True, 4320),
])
def test_identify_length(string, integer, rt, expected):
    result = td.identify_length(string, integer, rt=rt)
    assert result == expected


@patch("FinanceDatabase.dbase.DataAPI.ThetaData.request_from_proxy")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.requests.get")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.get_proxy_url")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.resolve_ticker_history")
def test_list_contracts_proxy_branch(mock_resolve, mock_get_proxy, mock_requests_get, mock_request_proxy):
    # Setup mocks
    mock_get_proxy.return_value = "http://proxy.local"
    
    # Proxy response with correct JSON keys
    mock_request_proxy.return_value = MagicMock(
        json=lambda: {
            "status_code": 200,
            "data": SNAPSHOT_CSV_STRIKE,
            "url": "http://proxy.local/fake_endpoint"
        }
    )

    # If snapshot branch triggers resolve_ticker_history
    mock_resolve.return_value = pd.DataFrame({"symbol": ["AAPL"], "strike": [150]})
    
    # Call the function
    df = td.list_contracts("AAPL", "2023-01-01", proxy="http://proxy.local")
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert "strike" in df.columns
    assert df['strike'].iloc[0] == 0.15
    mock_request_proxy.assert_called_once()
    mock_resolve.assert_not_called()  # adjust if snapshot branch triggers

@patch("FinanceDatabase.dbase.DataAPI.ThetaData.requests.get")
@patch("FinanceDatabase.dbase.DataAPI.ThetaData.get_proxy_url")
def test_list_contracts_no_proxy(mock_get_proxy, mock_requests_get):
    # No proxy provided
    mock_get_proxy.return_value = None

    # Create a proper mock response
    mock_response = MagicMock()
    mock_response.text = SNAPSHOT_CSV_STRIKE
    mock_response.status_code = 200  # important for raise_thetadata_exception
    mock_response.url = "http://fake_endpoint"

    mock_requests_get.return_value = mock_response

    # Call the function
    df = td.list_contracts("AAPL", "2023-01-01")

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert "strike" in df.columns
    # strike is divided by 1000 in the function
    assert df['strike'].iloc[0] == 0.15



