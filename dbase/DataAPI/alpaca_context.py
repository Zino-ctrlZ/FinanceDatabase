"""
Active Alpaca credentials and trading API host for the current task.

FinanceDatabase does not resolve bot personas or read ``bot_identity``. Callers
(TFP-Algo jobs or one-off scripts) build an :class:`AlpacaContext` and either:

- call :func:`set_alpaca_context` once per job (``contextvars``), then use
  :mod:`dbase.DataAPI.Alpaca` with no extra arguments; or
- pass ``context=`` on individual Alpaca entrypoints (tests, overrides).

Only TFP-Algo maps env vars to a persona; scripts may set context from any
``.env`` keys they choose via :meth:`AlpacaContext.from_env`.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional

PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets/v2"
LIVE_TRADING_BASE_URL = "https://api.alpaca.markets/v2"

_alpaca_context_var: ContextVar[Optional["AlpacaContext"]] = ContextVar(
    "alpaca_context", default=None
)


@dataclass(frozen=True)
class AlpacaContext:
    """API key, secret, and trading REST host for one Alpaca account."""

    api_key: str
    api_secret: str
    paper: bool = True
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("AlpacaContext requires non-empty api_key and api_secret")

    @property
    def trading_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return PAPER_TRADING_BASE_URL if self.paper else LIVE_TRADING_BASE_URL

    @classmethod
    def from_env(
        cls,
        key_var: str,
        secret_var: str,
        *,
        paper: bool = True,
        base_url: Optional[str] = None,
    ) -> "AlpacaContext":
        """Build context from arbitrary env var names (scripts / notebooks)."""
        key = os.environ.get(key_var)
        secret = os.environ.get(secret_var)
        if not key or not secret:
            raise ValueError(
                f"AlpacaContext.from_env: {key_var!r} and/or {secret_var!r} "
                "are unset or empty"
            )
        return cls(api_key=key, api_secret=secret, paper=paper, base_url=base_url)


def set_alpaca_context(context: Optional[AlpacaContext]) -> Token:
    """Set active Alpaca context for this task; returns token for :func:`reset_alpaca_context`."""
    return _alpaca_context_var.set(context)


def get_alpaca_context() -> Optional[AlpacaContext]:
    """Return the active context for this task, if any."""
    return _alpaca_context_var.get()


def reset_alpaca_context(token: Token) -> None:
    """Restore previous context after :func:`set_alpaca_context`."""
    _alpaca_context_var.reset(token)


def clear_alpaca_context() -> None:
    """Clear active context (equivalent to ``set_alpaca_context(None)`` without token)."""
    _alpaca_context_var.set(None)


def resolve_context(override: Optional[AlpacaContext] = None) -> Optional[AlpacaContext]:
    """Explicit ``override`` wins; else return the contextvar slot."""
    if override is not None:
        return override
    return get_alpaca_context()
