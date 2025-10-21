from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionIntent
from alpaca.trading.requests import OptionLegRequest
import os
import requests
import logging
from datetime import datetime

# Set up logging - fallback if trade package is not available
try:
    from trade.helpers.Logging import setup_logger
    logger = setup_logger('dbase.DataApi.Alpaca')
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('dbase.DataApi.Alpaca')
    
trading_client = TradingClient(os.environ.get('ALPACA_PAPER_KEY'), os.environ.get('ALPACA_PAPER_SECRET'))


def generate_option_symbol(symbol: str, expiration_date: str, right: str, strike: float) -> str:
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
    date_obj = datetime.strptime(expiration_date, '%Y-%m-%d')
    date_str = date_obj.strftime('%y%m%d')  # YYMMDD format
    
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
        root_symbol = ''.join(filter(str.isalpha, option_symbol[:-15]))
        
        # Extract expiration date (YYMMDD format)
        date_str = option_symbol[len(root_symbol):len(root_symbol) + 6]
        expiration_date = datetime.strptime(date_str, '%y%m%d').strftime('%Y-%m-%d')
        
        # Extract option type ('C' or 'P')
        option_type = option_symbol[len(root_symbol) + 6]
        option_type = 'C' if option_type == 'C' else 'P'
        
        # Extract strike price (last 8 digits, divide by 1000 to get float)
        strike_str = option_symbol[-8:]
        strike_price = int(strike_str) / 1000.0
        
        return {
            'symbol': root_symbol,
            'expiration_date': expiration_date,
            'right': option_type,
            'strike': strike_price
        }
    except Exception as e:
        print(f'Error parsing option symbol: {e}')
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
                params[k] = ','.join(v)
            else:
                params[k] = v
    return params

def get_headers(): 
    return {
        'accept': 'application/json',
        'APCA-API-KEY-ID': os.environ.get('ALPACA_PAPER_KEY'),
        'APCA-API-SECRET-KEY': os.environ.get('ALPACA_PAPER_SECRET')
    }

def get_trading_client(paper: bool = True): 
    return TradingClient(os.environ.get('ALPACA_PAPER_KEY'), os.environ.get('ALPACA_PAPER_SECRET'), paper=paper)

def get_base_url(paper: bool = True): 
    return 'https://paper-api.alpaca.markets/v2' if paper else 'https://api.alpaca.markets/v2'

def add_query_param(url: str, param: str, value: str) -> str:
    """Add a single query parameter to a URL."""
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}{param}={value}"

def add_query_params(url: str, params: dict) -> str:
    """Add multiple query parameters to a URL."""
    for param, value in params.items():
        url = add_query_param(url, param, value)
    return url

def get_account(trading_client: TradingClient): 
    return trading_client.get_account()

def create_buy_option_leg_request(symbol: str, ratio_qty: float = 1.0, side: OrderSide = OrderSide.BUY, position_intent: PositionIntent | None = None) -> OptionLegRequest:
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
        symbol=symbol,
        ratio_qty=ratio_qty,
        side=side,
        position_intent=position_intent
    )

def create_sell_option_leg_request(symbol: str, ratio_qty: float = 1.0, side: OrderSide = OrderSide.SELL, position_intent: PositionIntent | None = None) -> OptionLegRequest:
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
        symbol=symbol,
        ratio_qty=ratio_qty,
        side=side,
        position_intent=position_intent
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
    params = collect_params({**kwargs, 'type': type}, exclude=['symbol'])
    url = f'https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}'
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting option chain: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
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
        snapshots.update(res['snapshots'])
        while res is not None and res.get('next_page_token') is not None: 
            res = get_option_chain(symbol, type, **kwargs, page_token=res['next_page_token'])
            if res is not None:
                snapshots.update(res.get('snapshots', {}))
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
    params = collect_params({**kwargs, 'type': type, 'underlying_symbols': underlying_symbols})
    url = get_base_url() + '/options/contracts'
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting option contracts: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
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
        option_contracts.extend(res.get('option_contracts', []))
        while res is not None and res.get('next_page_token') is not None: 
            res = get_option_contracts(underlying_symbols, type, **kwargs, page_token=res['next_page_token'])
            if res is not None:
                option_contracts.extend(res.get('option_contracts', []))
    return option_contracts



def get_option_by_id_or_symbol(symbol_or_id: str): 
    """
    Get option by id or symbol from Alpaca
    ref: https://docs.alpaca.markets/reference/get-option-contract-symbol_or_id
    params: 
    - symbol_or_id: str

    returns : dict | None
    """
    url = get_base_url() + f'/options/contracts/{symbol_or_id}'
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting option by id or symbol: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error getting option contracts: {response.status_code} {response.text}')
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
    params = collect_params({**kwargs, 'status': status})
    url = get_base_url() + '/orders'
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting orders: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error getting orders: {response.status_code} {response.text}')
    return response.json()


def create_order(symbol: str, **kwargs): 
    """
    Create order for a single asset 
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
    - legs: list[dict]
    - client_order_id: str
    - trail_percent: float
    - trail_price: string

    returns : dict | None
    """
    payload = collect_params({**kwargs, 'symbol': symbol})
    url = get_base_url() + '/orders'
    response = requests.post(url, headers=get_headers(), json=payload)
    if response.status_code != 200:
        logger.error(f'Error creating order: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error creating order: {response.status_code} {response.text}')
    return response.json()
    

def delete_order(order_id: str): 
    """
    Delete order
    ref: https://docs.alpaca.markets/reference/deleteallorders-1
    param:
    - order_id: str

    returns : dict | None
    """
    url = get_base_url() + f'/orders/{order_id}'
    response = requests.delete(url, headers=get_headers())
    if response.status_code != 204:
        logger.error(f'Error deleting order: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error deleting order: {response.status_code} {response.text}')
    return response.text

def create_multi_leg_limit_order( legs: list[dict], qty: int, limit_price: float, **kwargs): 
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
   
    
    params = collect_params({**kwargs, 'qty': qty, 'limit_price': limit_price, "order_class": "mleg", "type": "limit", "time_in_force": "day"})
    payload = {**params, 'legs': legs}

    print(payload)
    url = get_base_url() + '/orders'
    response = requests.post(url, headers=get_headers(), json=payload)
    if response.status_code != 200:
        logger.error(f'Error creating multi-leg limit order: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error creating multi-leg limit order: {response.status_code} {response.text}')
    return response.json()


def replace_order(order_id: str, **kwargs): 
    """
    Replace order
    ref: https://docs.alpaca.markets/reference/patchorderbyorderid-1
    params: 
    - order_id: str
    - qty: int
    - time_in_force: str 'gtc' | 'ioc' | 'fok' | 'day' | 'opg'
    - limit_price: str
    - stop_price: str
    - trail str
    """
    params = collect_params({**kwargs})
    url = get_base_url() + f'/orders/{order_id}'
    response = requests.patch(url, headers=get_headers(), json=params)
    if response.status_code != 200:
        logger.error(f'Error replacing order: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error replacing order: {response.status_code} {response.text}')
    return response.json()
