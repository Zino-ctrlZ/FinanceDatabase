"""Tests for proxy ping wrapper + Theta status_code evaluation."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_proxy_module():
    for name in (
        "trade",
        "trade.helpers",
        "trade.helpers.Logging",
        "requests",
    ):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["trade.helpers.Logging"].setup_logger = lambda *a, **k: MagicMock()
    sys.modules["requests"].RequestException = Exception

    path = _REPO_ROOT / "dbase/DataAPI/ThetaData/proxy.py"
    spec = importlib.util.spec_from_file_location("theta_proxy_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_proxy = _load_proxy_module()
V2_PING_THETA_URL = _proxy.V2_PING_THETA_URL
PingProxyResult = _proxy.PingProxyResult
_evaluate_proxy_ping = _proxy._evaluate_proxy_ping
ping_proxy_v2 = _proxy.ping_proxy_v2


def _mock_response(status_code: int, json_body=None, json_raises=False):
    response = MagicMock()
    response.status_code = status_code
    if json_raises:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = json_body or {}
    return response


class TestEvaluateProxyPing:
    def test_both_layers_ok(self):
        response = _mock_response(200, {"status_code": 200, "data": "AAPL"})
        result = _evaluate_proxy_ping(response, V2_PING_THETA_URL)
        assert result.ok is True
        assert result.proxy_ok is True
        assert result.theta_ok is True
        assert result.proxy_http_status == 200
        assert result.theta_status_code == 200
        assert bool(result) is True

    def test_proxy_http_fails(self):
        response = _mock_response(502)
        result = _evaluate_proxy_ping(response, V2_PING_THETA_URL)
        assert result.ok is False
        assert result.proxy_ok is False
        assert result.theta_ok is False
        assert "502" in (result.error or "")

    def test_proxy_ok_theta_disconnected(self):
        response = _mock_response(
            200,
            {"status_code": 474, "data": "Connection lost to Theta Data MDDS."},
        )
        result = _evaluate_proxy_ping(response, V2_PING_THETA_URL)
        assert result.ok is False
        assert result.proxy_ok is True
        assert result.theta_ok is False
        assert result.theta_status_code == 474
        assert "MDDS" in (result.error or "")
        assert bool(result) is False

    def test_proxy_ok_wrong_theta_route(self):
        response = _mock_response(
            200,
            {"status_code": 404, "data": "<h1>404 Not Found</h1>"},
        )
        result = _evaluate_proxy_ping(response, V2_PING_THETA_URL)
        assert result.ok is False
        assert result.proxy_ok is True
        assert result.theta_status_code == 404

    def test_message_includes_both_layers(self):
        response = _mock_response(
            200,
            {"status_code": 474, "data": "disconnected"},
        )
        result = _evaluate_proxy_ping(response, V2_PING_THETA_URL)
        msg = result.message()
        assert "proxy HTTP 200" in msg
        assert "Theta status 474" in msg


class TestPingProxyV2:
    @patch.object(_proxy, "requests")
    @patch.object(_proxy, "get_proxy_url", return_value="http://proxy.test/thetadata")
    def test_ping_proxy_v2_uses_list_roots_url(self, _mock_url, mock_requests):
        mock_requests.post.return_value = _mock_response(
            200, {"status_code": 200, "data": "ok"}
        )
        result = ping_proxy_v2()
        assert result.ok is True
        call_payload = mock_requests.post.call_args.kwargs["json"]
        assert call_payload["url"] == V2_PING_THETA_URL
        assert "/v2/list/roots/stock" in call_payload["url"]

    @patch.object(_proxy, "get_proxy_url", return_value=None)
    @patch.object(_proxy, "get_proxy_url_from_env", return_value=None)
    def test_missing_proxy_url(self, _env, _url):
        result = ping_proxy_v2()
        assert result.ok is False
        assert "PROXY_URL" in (result.error or "")
