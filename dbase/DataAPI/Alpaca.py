from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionIntent
from alpaca.trading.requests import OptionLegRequest
import json
import logging
import os
import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import requests

from dbase.database.db_utils import get_current_environment
from dbase.DataAPI.alpaca_context import (
    LIVE_TRADING_BASE_URL,
    PAPER_TRADING_BASE_URL,
    AlpacaContext,
    resolve_context,
)

# Set up logging - fallback if trade package is not available
try:
    from trade.helpers.Logging import setup_logger

    logger = setup_logger("dbase.DataApi.Alpaca")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dbase.DataApi.Alpaca")

# --- replace_order: PATCH vs cancel+poll+POST (see replace_order docstring) ---
PATCH_FALLBACK_HTTP_STATUSES = frozenset({403})
PATCH_FALLBACK_JSON_CODES = frozenset({40310000})


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _patch_fallback_json_codes() -> frozenset:
    merged = set(PATCH_FALLBACK_JSON_CODES)
    extra = os.environ.get("ALPACA_REPLACE_PATCH_FALLBACK_JSON_CODES", "")
    for part in extra.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            merged.add(int(part))
        except ValueError:
            logger.warning(
                "Ignoring invalid ALPACA_REPLACE_PATCH_FALLBACK_JSON_CODES entry: %s",
                part,
            )
    return frozenset(merged)


def _replace_mode() -> str:
    v = (os.environ.get("ALPACA_REPLACE_MODE") or "patch_first").strip().lower()
    if v in ("patch_first", "cancel_post_only", "patch_only"):
        return v
    return "patch_first"


def _enum_to_api_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    s = str(value)
    if "::" in s:
        s = s.split("::")[-1]
    return s.lower()


def _normalize_order_class(order: Any) -> str:
    oc = getattr(order, "order_class", None)
    if oc is None and isinstance(order, dict):
        oc = order.get("order_class")
    if hasattr(oc, "value"):
        oc = oc.value
    return str(oc or "").lower()


def _order_class_is_mleg(order: Any) -> bool:
    return _normalize_order_class(order) == "mleg"


def _replace_recreate_kind(order: Any) -> str:
    """Classify cancel+POST recreate: mleg vs simple (single symbol)."""
    if _order_class_is_mleg(order):
        return "mleg"
    oc = _normalize_order_class(order)
    if oc in ("oco", "bracket"):
        raise ValueError(
            f"replace_order cancel+post does not support order_class={str(oc)}"
        )
    legs = getattr(order, "legs", None) or []
    if oc == "simple":
        if legs:
            raise ValueError(
                "replace_order: order_class=simple but snapshot has legs; refusing to recreate"
            )
        return "simple"
    symbol = getattr(order, "symbol", None)
    if symbol and not legs:
        return "simple"
    raise ValueError(
        f"replace_order cancel+post unsupported order_class={str(oc)} (supported: mleg, simple)"
    )


def _order_legs_to_payload(order: Any) -> list[dict]:
    legs = getattr(order, "legs", None)
    if legs is None and isinstance(order, dict):
        legs = order.get("legs") or []
    out: list[dict] = []
    for leg in legs or []:
        d = leg if isinstance(leg, dict) else vars(leg)
        ratio = d.get("ratio_qty", 1) or 1
        try:
            ratio_f = float(ratio)
        except (TypeError, ValueError):
            ratio_f = 1.0
        out.append(
            {
                "symbol": d.get("symbol"),
                "ratio_qty": ratio_f,
                "side": _enum_to_api_str(d.get("side")),
                "position_intent": _enum_to_api_str(d.get("position_intent")),
            }
        )
    return out


def _order_remaining_qty(order: Any) -> int:
    try:
        q = Decimal(str(getattr(order, "qty", None) or "0"))
        f = Decimal(str(getattr(order, "filled_qty", None) or "0"))
    except (InvalidOperation, TypeError, ValueError):
        q = Decimal(0)
        f = Decimal(0)
    rem = q - f
    try:
        return int(rem)
    except (ValueError, OverflowError):
        return int(rem.to_integral_value())


def _order_snapshot_time_in_force(order: Any) -> Optional[str]:
    tif = getattr(order, "time_in_force", None)
    if tif is None and isinstance(order, dict):
        tif = order.get("time_in_force")
    if hasattr(tif, "value"):
        return str(tif.value)
    if tif is None:
        return None
    return str(tif)


def _default_replacement_client_order_id(order_id: str) -> str:
    suffix = uuid.uuid4().hex[:12]
    raw = f"repl-{order_id}-{suffix}"
    return raw if len(raw) <= 128 else raw[:128]


def _patch_triggers_fallback(response: requests.Response) -> bool:
    if response.status_code in PATCH_FALLBACK_HTTP_STATUSES:
        return True
    try:
        body = response.json()
        code = body.get("code")
        if code is None:
            return False
        return int(code) in _patch_fallback_json_codes()
    except (ValueError, TypeError, json.JSONDecodeError):
        return False


def _replace_cancel_post_kw(kw: dict) -> dict:
    """Subset of replace kwargs applicable to cancel+POST recreate (mleg or simple)."""
    if kw.get("limit_price") is None:
        raise ValueError("replace_order requires limit_price for cancel+post path")
    out: dict[str, Any] = {"limit_price": kw["limit_price"]}
    if kw.get("qty") is not None:
        out["qty"] = kw["qty"]
    if kw.get("time_in_force") is not None:
        out["time_in_force"] = kw["time_in_force"]
    return out


def _get_order_json_http(order_id: str) -> tuple[int, Optional[dict], str]:
    url = get_base_url() + f"/orders/{order_id}"
    r = requests.get(url, headers=get_headers(), timeout=30)
    text = r.text or ""
    if r.status_code != 200:
        return r.status_code, None, text
    try:
        return r.status_code, r.json(), text
    except json.JSONDecodeError:
        return r.status_code, None, text


def _require_alpaca_context() -> RuntimeError:
    """
    Fail fast when callers use Alpaca APIs without setting AlpacaContext.

    Phase 5 cutover removes legacy env==prod fallbacks; callers must either:
    - set contextvars once per job via set_alpaca_context(...)
    - or pass `context=` explicitly to Alpaca entrypoints (scripts/tests)
    """

    return RuntimeError(
        "AlpacaContext is not set. Call dbase.DataAPI.alpaca_context.set_alpaca_context(...) "
        "for the current task (or pass `context=` / AlpacaContext.from_env(...) "
        "for one-off calls) before using Alpaca credentials."
    )


def get_alpaca_key(context: AlpacaContext | None = None) -> str:
    ctx = resolve_context(context)
    if ctx is None:
        raise _require_alpaca_context()
    return ctx.api_key


def get_alpaca_secret(context: AlpacaContext | None = None) -> str:
    ctx = resolve_context(context)
    if ctx is None:
        raise _require_alpaca_context()
    return ctx.api_secret


def generate_option_symbol(
    symbol: str, expiration_date: str, right: str, strike: float
) -> str:
    """
    Generate Alpaca option symbol format.

    Args:
        root_symbol: Stock symbol (e.g., 'AAPL')
        expiration_date: Date in 'YYYY-MM-DD' format
        option_type: 'call' or 'put' (or 'C'/'P')
        strike_price: Strike price as float

    Returns:
        Option symbol in Alpaca format
    """

    # Parse expiration date
    date_obj = datetime.strptime(expiration_date, "%Y-%m-%d")
    date_str = date_obj.strftime("%y%m%d")  # YYMMDD format

    # Convert option type to single letter
    option_letter = right.upper()[0]  # 'call' -> 'C', 'put' -> 'P'

    # Format strike price (remove decimal, pad to 8 digits)
    # Multiply by 1000 to get the correct format (e.g., 120.0 -> 120000)
    strike_str = f"{int(strike * 1000):08d}"

    return f"{symbol}{date_str}{option_letter}{strike_str}"


def parse_option_symbol(option_symbol: str) -> dict | None:
    """
    Parse Alpaca option symbol format into its components.

    Args:
        option_symbol: Option symbol in Alpaca format (e.g., 'AAPL250724C01200000')

    Returns:
        A dictionary with root_symbol, expiration_date, option_type, and strike_price.
    """
    try:
        # Extract root symbol (letters before the date)
        root_symbol = "".join(filter(str.isalpha, option_symbol[:-15]))

        # Extract expiration date (YYMMDD format)
        date_str = option_symbol[len(root_symbol) : len(root_symbol) + 6]
        expiration_date = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")

        # Extract option type ('C' or 'P')
        option_type = option_symbol[len(root_symbol) + 6]
        option_type = "C" if option_type == "C" else "P"

        # Extract strike price (last 8 digits, divide by 1000 to get float)
        strike_str = option_symbol[-8:]
        strike_price = int(strike_str) / 1000.0

        return {
            "symbol": root_symbol,
            "expiration_date": expiration_date,
            "right": option_type,
            "strike": strike_price,
        }
    except Exception as e:
        print(f"Error parsing option symbol: {e}")
        return None


def collect_params(args_dict, exclude: list[str] = []):
    """
    Collect all arguments except 'symbol' from a function's arguments dict, and add them to a params dict if their value is not None.

    Args:
        args_dict (dict): Dictionary of function arguments (e.g., from locals()).

    Returns:
        dict: Dictionary of parameters (excluding 'symbol') with non-None values.

    Example:
        def example_func(symbol, limit=None, side=None, qty=None):
            params = collect_params(locals())
            # params will include only non-None values for limit, side, qty
    """
    params = {}
    for k, v in args_dict.items():
        if k not in exclude and v is not None:
            if isinstance(v, list):
                params[k] = ",".join(v)
            else:
                params[k] = v
    return params


def get_headers(context: AlpacaContext | None = None) -> dict[str, str]:
    return {
        "accept": "application/json",
        "APCA-API-KEY-ID": get_alpaca_key(context),
        "APCA-API-SECRET-KEY": get_alpaca_secret(context),
    }


def get_trading_client(
    paper: bool | None = None,
    raw_data: bool = False,
    context: AlpacaContext | None = None,
) -> TradingClient:
    ctx = resolve_context(context)
    if ctx is not None:
        return TradingClient(
            ctx.api_key,
            ctx.api_secret,
            paper=ctx.paper if paper is None else paper,
            raw_data=raw_data,
        )
    return TradingClient(
        get_alpaca_key(),
        get_alpaca_secret(),
        paper=True if paper is None else paper,
        raw_data=raw_data,
    )


def get_base_url(paper: bool | None = None, context: AlpacaContext | None = None) -> str:
    ctx = resolve_context(context)
    if ctx is not None:
        if paper is None or paper == ctx.paper:
            return ctx.trading_base_url
        return PAPER_TRADING_BASE_URL if paper else LIVE_TRADING_BASE_URL
    use_paper = True if paper is None else paper
    return PAPER_TRADING_BASE_URL if use_paper else LIVE_TRADING_BASE_URL


def add_query_param(url: str, param: str, value: str) -> str:
    """Add a single query parameter to a URL."""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{param}={value}"


def add_query_params(url: str, params: dict) -> str:
    """Add multiple query parameters to a URL."""
    for param, value in params.items():
        url = add_query_param(url, param, value)
    return url


def get_account(trading_client: TradingClient):
    return trading_client.get_account()


def create_buy_option_leg_request(
    symbol: str,
    ratio_qty: float = 1.0,
    side: OrderSide = OrderSide.BUY,
    position_intent: PositionIntent | None = None,
) -> OptionLegRequest:
    """
    Create a buy option leg request for multi-leg orders.

    Args:
        symbol (str): The option contract symbol (e.g., 'AAPL250718C00090000')
        ratio_qty (float): The proportional quantity of this leg in relation to the overall multi-leg order quantity
        side (OrderSide): The side of the order (BUY or SELL), defaults to BUY
        position_intent (PositionIntent): The position strategy for this leg (optional)

    Returns:
        OptionLegRequest: A configured option leg request for buying options

    Example:
        # Create a simple buy call option leg
        leg = create_buy_option_leg_request('AAPL250718C00090000')

        # Create a buy put option leg with custom ratio
        leg = create_buy_option_leg_request('AAPL250718P00085000', ratio_qty=2.0)
    """
    return OptionLegRequest(
        symbol=symbol, ratio_qty=ratio_qty, side=side, position_intent=position_intent
    )


def create_sell_option_leg_request(
    symbol: str,
    ratio_qty: float = 1.0,
    side: OrderSide = OrderSide.SELL,
    position_intent: PositionIntent | None = None,
) -> OptionLegRequest:
    """
    Create a sell option leg request for multi-leg orders.

    Args:
        symbol (str): The option contract symbol (e.g., 'AAPL250718C00090000')
        ratio_qty (float): The proportional quantity of this leg in relation to the overall multi-leg order quantity
        side (OrderSide): The side of the order (BUY or SELL), defaults to SELL
        position_intent (PositionIntent): The position strategy for this leg (optional)

    Returns:
        OptionLegRequest: A configured option leg request for selling options

    Example:
        # Create a simple sell call option leg
        leg = create_sell_option_leg_request('AAPL250718C00090000')

        # Create a sell put option leg with custom ratio
        leg = create_sell_option_leg_request('AAPL250718P00085000', ratio_qty=1.0)
    """
    return OptionLegRequest(
        symbol=symbol, ratio_qty=ratio_qty, side=side, position_intent=position_intent
    )


def get_option_chain(symbol: str, type: str, **kwargs):
    """
    Get option chain from Alpaca
    ref: https://docs.alpaca.markets/reference/optionchain
    base url: https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}
    params:
    - symbol: str
    - type: str
    - expiration_date: str
    - expiration_date_gte: str
    - expiration_date_lte: str
    - strike_price_gte: str
    - strike_price_lte: str
    - updated_since: str
    - limit: int
    - page_token: str

    returns : {next_page_token: str, snapshots: dict} | None
    """
    params = collect_params({**kwargs, "type": type}, exclude=["symbol"])
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(
            f"Error getting option chain: {response.status_code} {response.text}"
        )
        print(f"Error: {response.text} {response.status_code}")
        return None
    return response.json()


def get_option_chain_all(symbol: str, type: str, **kwargs):
    """
    Get all option chains from Alpaca
    ref: https://docs.alpaca.markets/reference/optionchain
    base url: https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}
    params:
    - symbol: str
    - type: str
    - expiration_date: str
    - expiration_date_gte: str
    - expiration_date_lte: str
    - strike_price_gte: str
    - strike_price_lte: str
    - updated_since: str
    - limit: int
    - page_token: str

    returns : dict | None
    """
    res = get_option_chain(symbol, type, **kwargs)
    snapshots = {}
    if res is None:
        return snapshots
    elif res is not None:
        snapshots.update(res["snapshots"])
        while res is not None and res.get("next_page_token") is not None:
            res = get_option_chain(
                symbol, type, **kwargs, page_token=res["next_page_token"]
            )
            if res is not None:
                snapshots.update(res.get("snapshots", {}))
    return snapshots


def get_option_contracts(underlying_symbols: str, type: str, **kwargs):
    """
    Get option contracts from Alpaca
    ref: https://docs.alpaca.markets/reference/get-options-contracts
    params:
    - underlying_symbols: str
    - type: str
    - expiration_date: str
    - expiration_date_gte: str
    - expiration_date_lte: str
    - strike_price_gte: str
    - strike_price_lte: str
    - updated_since: str
    - limit: int
    - page_token: str

    returns : {option_contracts: list[dict], next_page_token: str | None} | None
    """
    params = collect_params(
        {**kwargs, "type": type, "underlying_symbols": underlying_symbols}
    )
    url = get_base_url() + "/options/contracts"
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(
            f"Error getting option contracts: {response.status_code} {response.text}"
        )
        print(f"Error: {response.text} {response.status_code}")
        return None
    return response.json()


def get_option_contracts_all(underlying_symbols: str, type: str, **kwargs):
    """
    Get all option contracts from Alpaca
    ref: https://docs.alpaca.markets/reference/get-options-contracts
    params:
    - underlying_symbols: str
    - type: str
    - expiration_date: str
    - expiration_date_gte: str
    - expiration_date_lte: str
    - strike_price_gte: str
    - strike_price_lte: str
    - updated_since: str
    - limit: int
    - page_token: str

    returns : list[dict] | None
    """
    res = get_option_contracts(underlying_symbols, type, **kwargs)
    option_contracts = []
    if res is None:
        return option_contracts
    else:
        option_contracts.extend(res.get("option_contracts", []))
        while res is not None and res.get("next_page_token") is not None:
            res = get_option_contracts(
                underlying_symbols, type, **kwargs, page_token=res["next_page_token"]
            )
            if res is not None:
                option_contracts.extend(res.get("option_contracts", []))
    return option_contracts


def get_option_by_id_or_symbol(symbol_or_id: str):
    """
    Get option by id or symbol from Alpaca
    ref: https://docs.alpaca.markets/reference/get-option-contract-symbol_or_id
    params:
    - symbol_or_id: str

    returns : dict | None
    """
    url = get_base_url() + f"/options/contracts/{symbol_or_id}"
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(
            f"Error getting option by id or symbol: {response.status_code} {response.text}"
        )
        print(f"Error: {response.text} {response.status_code}")
        raise Exception(
            f"Error getting option contracts: {response.status_code} {response.text}"
        )
    return response.json()


def get_orders(status: str, **kwargs):
    """
    Get all orders
    ref: https://docs.alpaca.markets/reference/getallorders-1
    params:
    - status: str 'closed' | 'open' | 'all'
    - limit: int
    - after: str timestamp (YYYY-MM-DDTHH:MM:SSZ)
    - until: str timestamp (YYYY-MM-DDTHH:MM:SSZ)
    - direction: str 'asc' | 'desc'
    - nested: bool
    - symbols: list[str]
    - asset_class: str 'us_option' | 'us_equity' | 'crypto' | 'all'

    returns : list[dict] | None
    """
    params = collect_params({**kwargs, "status": status})
    url = get_base_url() + "/orders"
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f"Error getting orders: {response.status_code} {response.text}")
        print(f"Error: {response.text} {response.status_code}")
        raise Exception(f"Error getting orders: {response.status_code} {response.text}")
    return response.json()


def create_order(symbol: str, **kwargs):
    """
    Create order for a single asset (or pass legs for multi-leg when building payload manually).
    ref: https://docs.alpaca.markets/reference/postorder
    params:
    - symbol: str
    - qty: int
    - side: str
    - type: str 'market' | 'limit' | 'stop' | 'stop_limit' | 'trailing_stop'
    - time_in_force: str 'gtc' | 'ioc' | 'fok' | 'day' | 'opg'
    - limit_price: string
    - stop_price: dict
    - take_profit: dict
    - legs: list[dict]  (passed through to JSON; not run through collect_params)
    - client_order_id: str
    - trail_percent: float
    - trail_price: string

    returns : dict | None
    """
    kw = dict(kwargs)
    legs = kw.pop("legs", None)
    payload = collect_params({**kw, "symbol": symbol})
    if legs is not None:
        payload["legs"] = legs
    url = get_base_url() + "/orders"
    response = requests.post(url, headers=get_headers(), json=payload, timeout=30)
    if response.status_code != 200:
        logger.error(f"Error creating order: {response.status_code} {response.text}")
        print(f"Error: {response.text} {response.status_code}")
        raise Exception(f"Error creating order: {response.status_code} {response.text}")
    return response.json()


def delete_order(order_id: str):
    """
    Cancel an open order (DELETE /v2/orders/{id}).
    ref: https://docs.alpaca.markets/reference/deleteorderbyorderid-1

    Returns empty string on 204. Raises on failure; 422 means order is not cancelable.
    """
    url = get_base_url() + f"/orders/{order_id}"
    response = requests.delete(url, headers=get_headers(), timeout=30)
    if response.status_code == 204:
        return response.text or ""
    logger.error(f"Error deleting order: {response.status_code} {response.text}")
    print(f"Error: {response.text} {response.status_code}")
    if response.status_code == 422:
        raise Exception(f"Error deleting order (not cancelable): 422 {response.text}")
    raise Exception(f"Error deleting order: {response.status_code} {response.text}")


def create_multi_leg_limit_order(
    legs: list[dict], qty: int, limit_price: float, **kwargs
):
    """
    Create multi-leg limit order
    ref: https://docs.alpaca.markets/docs/options-level-3-trading
    params:
    - legs: list[dict] {
        'symbol': str,
        'ratio_qty': float,
        'side': str, # 'buy' | 'sell'
        'position_intent': str, # 'buy_to_open' | 'buy_to_close'
    }
    - qty: int
    - limit_price: float
    - time_in_force: str 'gtc' | 'ioc' | 'fok' | 'day' | 'opg'
    - trail_percent: float
    - trail_price: string
    returns : dict | None
    """

    kw = dict(kwargs)
    tif = kw.pop("time_in_force", None) or "day"
    params = collect_params(
        {
            **kw,
            "qty": qty,
            "limit_price": limit_price,
            "order_class": "mleg",
            "type": "limit",
            "time_in_force": tif,
        }
    )
    payload = {**params, "legs": legs}

    print(payload)
    url = get_base_url() + "/orders"
    response = requests.post(url, headers=get_headers(), json=payload)
    if response.status_code != 200:
        logger.error(
            f"Error creating multi-leg limit order: {response.status_code} {response.text}"
        )
        print(f"Error: {response.text} {response.status_code}")
        raise Exception(
            f"Error creating multi-leg limit order: {response.status_code} {response.text}"
        )
    return response.json()


def _replace_poll_after_delete(order_id: str) -> None:
    """Poll GET /orders/{id} until canceled/expired; raise on filled/rejected/404/timeout."""
    max_wait = _env_float("ALPACA_REPLACE_CANCEL_MAX_WAIT_SEC", 30.0)
    poll_interval = _env_float("ALPACA_REPLACE_CANCEL_POLL_INTERVAL_SEC", 0.35)
    deadline = time.monotonic() + max_wait
    terminal_ok = frozenset({"canceled", "expired"})

    while time.monotonic() < deadline:
        status_code, data, raw_text = _get_order_json_http(order_id)
        if status_code == 404:
            logger.error(
                "replace_order poll: GET order returned 404 (unexpected) | order_id=%s env=%s",
                order_id,
                get_current_environment(),
            )
            raise Exception(
                f"replace_order: order {order_id} not found (404) during cancel poll — check API host/account"
            )
        if status_code != 200 or not isinstance(data, dict):
            raise Exception(
                f"replace_order: poll GET order failed {status_code} {raw_text[:500]}"
            )
        st = str(data.get("status") or "")
        if st == "filled":
            raise Exception(
                f"replace_order: order {order_id} filled during cancel; not posting replacement"
            )
        if st == "rejected":
            raise Exception(
                f"replace_order: order {order_id} rejected during cancel poll; not posting replacement"
            )
        if st in terminal_ok:
            return
        time.sleep(poll_interval)
    raise Exception(
        f"replace_order: timeout waiting for cancel confirmation on order {order_id}"
    )


def _replace_order_cancel_and_post(
    order_id: str,
    *,
    client_order_id: str,
    limit_price: float,
    qty: Optional[int] = None,
    time_in_force: Optional[str] = None,
) -> dict:
    """
    Cancel order, poll until canceled/expired, then POST a replacement.

    Supports order_class **mleg** (same legs) and **simple** (single symbol limit).
    """
    tc = get_trading_client()
    try:
        order = tc.get_order_by_id(order_id)
    except Exception as e:
        raise Exception(f"replace_order: could not load order {order_id}: {e}") from e

    kind = _replace_recreate_kind(order)

    remaining = _order_remaining_qty(order)
    if remaining <= 0:
        raise ValueError(
            f"replace_order cancel+post: no remaining qty on order {order_id} (fully filled or flat)"
        )

    caller_qty_i = int(qty) if qty is not None else remaining
    new_qty = min(caller_qty_i, remaining)
    if new_qty < caller_qty_i:
        logger.warning(
            "replace_order qty clamped to remaining | order_id=%s requested=%s remaining=%s new_qty=%s",
            order_id,
            caller_qty_i,
            remaining,
            new_qty,
        )

    snap_tif = _order_snapshot_time_in_force(order)
    tif = (time_in_force or snap_tif or "day").strip()

    if len(client_order_id) > 128:
        client_order_id = client_order_id[:128]

    post_extras: dict[str, Any] = {}
    eh = getattr(order, "extended_hours", None)
    if eh is not None:
        post_extras["extended_hours"] = bool(eh)

    logger.info(
        "replace_order path=cancel_poll_post kind=%s | order_id=%s client_order_id=%s new_qty=%s tif=%s",
        kind,
        order_id,
        client_order_id,
        new_qty,
        tif,
    )

    try:
        delete_order(order_id)
        _replace_poll_after_delete(order_id)
    except Exception as e:
        logger.error(
            "replace_order cancel+post: cancel or poll failed | order_id=%s kind=%s "
            "(if delete returned 204, order may be canceled or still settling; no replacement POST sent) err=%s",
            order_id,
            kind,
            e,
            exc_info=True,
        )
        raise

    lp = float(limit_price)

    if kind == "mleg":
        legs = _order_legs_to_payload(order)
        if not legs:
            raise ValueError(
                f"replace_order cancel+post: no legs on mleg order {order_id}"
            )
        return create_multi_leg_limit_order(
            legs,
            new_qty,
            lp,
            client_order_id=client_order_id,
            time_in_force=tif,
            **post_extras,
        )

    # simple: single-symbol limit recreate
    ot = _enum_to_api_str(getattr(order, "type", None)) or "limit"
    if ot != "limit":
        raise ValueError(
            f"replace_order cancel+post for simple orders only supports type=limit; got {str(ot)}"
        )
    symbol = getattr(order, "symbol", None)
    if not symbol:
        raise ValueError(
            f"replace_order cancel+post: simple order {order_id} missing symbol"
        )
    side = _enum_to_api_str(getattr(order, "side", None))
    if not side:
        raise ValueError(
            f"replace_order cancel+post: simple order {order_id} missing side"
        )

    return create_order(
        str(symbol),
        qty=new_qty,
        limit_price=lp,
        side=side,
        type="limit",
        time_in_force=tif,
        client_order_id=client_order_id,
        **post_extras,
    )


def replace_order(order_id: str, **kwargs):
    """
    Replace an order: PATCH when mode allows, else cancel → poll → POST.

    Cancel+POST recreate supports **mleg** (same legs) and **simple** (single-symbol limit).
    OCO/bracket and simple non-limit types are rejected on the recreate path.

    Environment (optional):
    - ALPACA_REPLACE_MODE: patch_first (default) | cancel_post_only | patch_only
    - ALPACA_REPLACE_PATCH_MAX_RETRIES: default 2 (attempts after first = retries)
    - ALPACA_REPLACE_PATCH_RETRY_DELAY_SEC: default 0.5
    - ALPACA_REPLACE_PATCH_FALLBACK_JSON_CODES: comma-separated extra JSON integer codes
      merged with default {40310000}; HTTP 403 always triggers fallback when patch_first.
    - ALPACA_REPLACE_CANCEL_MAX_WAIT_SEC: default 30
    - ALPACA_REPLACE_CANCEL_POLL_INTERVAL_SEC: default 0.35

    Kwargs:
    - limit_price, qty, time_in_force, etc. passed to PATCH; client_order_id is stripped from PATCH
      and used only on cancel+POST recreate.
    - cancel+POST clamps qty to remaining Alpaca qty and logs a warning if clamped.

    ref: https://docs.alpaca.markets/reference/patchorderbyorderid-1
    """
    mode = _replace_mode()
    kw = dict(kwargs)
    client_order_id = kw.pop("client_order_id", None)

    if mode == "cancel_post_only":
        cid = client_order_id or _default_replacement_client_order_id(order_id)
        return _replace_order_cancel_and_post(
            order_id, client_order_id=cid, **_replace_cancel_post_kw(kw)
        )

    max_retries = _env_int("ALPACA_REPLACE_PATCH_MAX_RETRIES", 2)
    delay = _env_float("ALPACA_REPLACE_PATCH_RETRY_DELAY_SEC", 0.5)
    patch_params = collect_params(dict(kw))
    url = get_base_url() + f"/orders/{order_id}"

    last: Optional[requests.Response] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.patch(
                url, headers=get_headers(), json=patch_params, timeout=30
            )
            last = response
            if response.status_code == 200:
                logger.info(
                    "replace_order path=patch | mode=%s order_id=%s", mode, order_id
                )
                return response.json()
            if _patch_triggers_fallback(response):
                break
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(delay)
                    continue
            break
        except requests.RequestException as e:
            last = None
            if attempt < max_retries:
                logger.warning(
                    "replace_order PATCH retry | order_id=%s attempt=%s err=%s",
                    order_id,
                    attempt,
                    e,
                )
                time.sleep(delay)
                continue
            raise Exception(f"Error replacing order (network): {e}") from e

    if mode == "patch_first" and last is not None and _patch_triggers_fallback(last):
        try:
            body = last.json()
            if isinstance(body, dict) and body.get("related_orders"):
                logger.info(
                    "replace_order PATCH fallback related_orders=%s | order_id=%s",
                    body.get("related_orders"),
                    order_id,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        cid = client_order_id or _default_replacement_client_order_id(order_id)
        logger.info(
            "replace_order path=cancel_poll_post after PATCH fallback | order_id=%s status=%s",
            order_id,
            last.status_code,
        )
        return _replace_order_cancel_and_post(
            order_id, client_order_id=cid, **_replace_cancel_post_kw(kw)
        )

    if last is not None:
        logger.error("Error replacing order: %s %s", last.status_code, last.text)
        print(f"Error: {last.text} {last.status_code}")
        raise Exception(f"Error replacing order: {last.status_code} {last.text}")
    raise Exception("Error replacing order: no response")
