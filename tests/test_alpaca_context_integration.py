"""Integration tests: Alpaca.py reads AlpacaContext (no live API calls)."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("alpaca")

from dbase.DataAPI import alpaca_context as ac

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CTX_KEY = "ctx-api-key"
_CTX_SECRET = "ctx-api-secret"
_LEGACY_LIVE_KEY = "legacy-live-key"
_LEGACY_LIVE_SECRET = "legacy-live-secret"
_LEGACY_PAPER_KEY = "legacy-paper-key"
_LEGACY_PAPER_SECRET = "legacy-paper-secret"


def _load_alpaca_module():
    """Load Alpaca.py without pulling dbase.database SQLHelpers."""
    db_utils = types.ModuleType("dbase.database.db_utils")
    db_utils.get_current_environment = MagicMock(return_value="test")

    database_pkg = types.ModuleType("dbase.database")
    database_pkg.db_utils = db_utils

    sys.modules.setdefault("dbase.database.db_utils", db_utils)
    sys.modules.setdefault("dbase.database", database_pkg)

    path = _REPO_ROOT / "dbase/DataAPI/Alpaca.py"
    spec = importlib.util.spec_from_file_location(
        "alpaca_module_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_alpaca = _load_alpaca_module()
LIVE_TRADING_BASE_URL = _alpaca.LIVE_TRADING_BASE_URL
PAPER_TRADING_BASE_URL = _alpaca.PAPER_TRADING_BASE_URL
get_alpaca_key = _alpaca.get_alpaca_key
get_alpaca_secret = _alpaca.get_alpaca_secret
get_base_url = _alpaca.get_base_url
get_headers = _alpaca.get_headers
get_trading_client = _alpaca.get_trading_client


@pytest.fixture(autouse=True)
def _clean_context():
    ac.clear_alpaca_context()
    yield
    ac.clear_alpaca_context()


@pytest.fixture
def legacy_env(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_KEY", _LEGACY_LIVE_KEY)
    monkeypatch.setenv("ALPACA_LIVE_SECRET", _LEGACY_LIVE_SECRET)
    monkeypatch.setenv("ALPACA_PAPER_KEY", _LEGACY_PAPER_KEY)
    monkeypatch.setenv("ALPACA_PAPER_SECRET", _LEGACY_PAPER_SECRET)


def test_get_alpaca_key_uses_context_not_legacy(legacy_env):
    ctx = ac.AlpacaContext(api_key=_CTX_KEY, api_secret=_CTX_SECRET, paper=True)
    token = ac.set_alpaca_context(ctx)
    try:
        assert get_alpaca_key() == _CTX_KEY
    finally:
        ac.reset_alpaca_context(token)


def test_missing_context_raises_prod(legacy_env):
    with pytest.raises(
        RuntimeError, match=r"AlpacaContext is not set.*set_alpaca_context"
    ):
        get_alpaca_key()
    with pytest.raises(
        RuntimeError, match=r"AlpacaContext is not set.*set_alpaca_context"
    ):
        get_alpaca_secret()


def test_missing_context_raises_non_prod(legacy_env):
    with pytest.raises(
        RuntimeError, match=r"AlpacaContext is not set.*set_alpaca_context"
    ):
        get_alpaca_key()
    with pytest.raises(
        RuntimeError, match=r"AlpacaContext is not set.*set_alpaca_context"
    ):
        get_alpaca_secret()


def test_get_base_url_from_context():
    custom_url = "https://custom.example/v2"
    ctx = ac.AlpacaContext(
        api_key=_CTX_KEY,
        api_secret=_CTX_SECRET,
        paper=True,
        base_url=custom_url,
    )
    token = ac.set_alpaca_context(ctx)
    try:
        assert get_base_url() == custom_url
    finally:
        ac.reset_alpaca_context(token)


def test_get_base_url_paper_default_from_context():
    ctx = ac.AlpacaContext(api_key=_CTX_KEY, api_secret=_CTX_SECRET, paper=True)
    token = ac.set_alpaca_context(ctx)
    try:
        assert get_base_url() == PAPER_TRADING_BASE_URL
    finally:
        ac.reset_alpaca_context(token)


def test_get_base_url_live_from_context():
    ctx = ac.AlpacaContext(api_key=_CTX_KEY, api_secret=_CTX_SECRET, paper=False)
    token = ac.set_alpaca_context(ctx)
    try:
        assert get_base_url() == LIVE_TRADING_BASE_URL
    finally:
        ac.reset_alpaca_context(token)


def test_get_headers_from_context(legacy_env):
    ctx = ac.AlpacaContext(api_key=_CTX_KEY, api_secret=_CTX_SECRET, paper=True)
    token = ac.set_alpaca_context(ctx)
    try:
        headers = get_headers()
        assert headers == {
            "accept": "application/json",
            "APCA-API-KEY-ID": _CTX_KEY,
            "APCA-API-SECRET-KEY": _CTX_SECRET,
        }
    finally:
        ac.reset_alpaca_context(token)


def test_get_trading_client_from_context():
    ctx = ac.AlpacaContext(api_key=_CTX_KEY, api_secret=_CTX_SECRET, paper=True)
    token = ac.set_alpaca_context(ctx)
    mock_client = MagicMock()
    try:
        with patch.object(_alpaca, "TradingClient", return_value=mock_client) as tc:
            client = get_trading_client(raw_data=True)
            assert client is mock_client
            tc.assert_called_once_with(
                _CTX_KEY, _CTX_SECRET, paper=True, raw_data=True
            )
    finally:
        ac.reset_alpaca_context(token)


def test_explicit_context_overrides_contextvar(legacy_env):
    slot_ctx = ac.AlpacaContext(
        api_key="slot-key", api_secret="slot-secret", paper=True
    )
    explicit = ac.AlpacaContext(
        api_key="explicit-key", api_secret="explicit-secret", paper=False
    )
    token = ac.set_alpaca_context(slot_ctx)
    try:
        assert get_alpaca_key() == "slot-key"
        assert get_alpaca_key(context=explicit) == "explicit-key"
        assert get_alpaca_secret(context=explicit) == "explicit-secret"
        assert get_base_url(context=explicit) == LIVE_TRADING_BASE_URL
        headers = get_headers(context=explicit)
        assert headers["APCA-API-KEY-ID"] == "explicit-key"
        assert headers["APCA-API-SECRET-KEY"] == "explicit-secret"

        with patch.object(_alpaca, "TradingClient") as tc:
            get_trading_client(context=explicit, paper=True)
            tc.assert_called_once_with(
                "explicit-key", "explicit-secret", paper=True, raw_data=False
            )
    finally:
        ac.reset_alpaca_context(token)
