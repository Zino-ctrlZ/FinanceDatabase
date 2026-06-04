"""Tests for Alpaca contextvars (no live API calls)."""

import os

import pytest

from dbase.DataAPI import alpaca_context as ac


def test_from_env():
    os.environ["TEST_ALPACA_KEY"] = "key-id"
    os.environ["TEST_ALPACA_SECRET"] = "secret"
    try:
        ctx = ac.AlpacaContext.from_env(
            "TEST_ALPACA_KEY", "TEST_ALPACA_SECRET", paper=False
        )
        assert ctx.api_key == "key-id"
        assert ctx.paper is False
        assert ctx.trading_base_url == ac.LIVE_TRADING_BASE_URL
    finally:
        os.environ.pop("TEST_ALPACA_KEY", None)
        os.environ.pop("TEST_ALPACA_SECRET", None)


def test_from_env_missing_raises():
    with pytest.raises(ValueError, match="unset or empty"):
        ac.AlpacaContext.from_env("MISSING_KEY_X", "MISSING_SECRET_X")


def test_contextvar_roundtrip():
    ac.clear_alpaca_context()
    ctx = ac.AlpacaContext(api_key="ctx-key", api_secret="ctx-secret", paper=True)
    token = ac.set_alpaca_context(ctx)
    try:
        assert ac.get_alpaca_context() is ctx
        assert ac.resolve_context() is ctx
    finally:
        ac.reset_alpaca_context(token)
    assert ac.get_alpaca_context() is None


def test_resolve_explicit_override():
    ac.clear_alpaca_context()
    slot_ctx = ac.AlpacaContext(api_key="slot", api_secret="s", paper=True)
    explicit = ac.AlpacaContext(api_key="explicit", api_secret="s", paper=True)
    token = ac.set_alpaca_context(slot_ctx)
    try:
        assert ac.resolve_context(explicit).api_key == "explicit"
        assert ac.resolve_context().api_key == "slot"
    finally:
        ac.reset_alpaca_context(token)


def test_custom_base_url():
    ctx = ac.AlpacaContext(
        api_key="k",
        api_secret="s",
        paper=True,
        base_url="https://custom.example/v2",
    )
    assert ctx.trading_base_url == "https://custom.example/v2"


def test_empty_key_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ac.AlpacaContext(api_key="", api_secret="s")
