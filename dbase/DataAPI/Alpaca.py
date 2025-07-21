from trade.helpers.Logging import setup_logger
from alpaca.trading.client import TradingClient
import os
import requests

trading_client = TradingClient(os.environ.get('ALPACA_PAPER_KEY'), os.environ.get('ALPACA_PAPER_SECRET'))
logger = setup_logger('dbase.DataApi.Alpaca')

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

def add_query_param(url: str, param: str, value: str):
    if '?' in url: 
        return f'{url}&{param}={value}'
    else: 
        return f'{url}?{param}={value}'

def add_query_params(url: str, params: dict):
    for param, value in params.items():
        url = add_query_param(url, param, value)
    return url

def get_account(trading_client: TradingClient): 
    return trading_client.get_account()



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
    Create order
    ref: https://docs.alpaca.markets/reference/postorder
    params: 
    - symbol: str
    - qty: int
    - side: str
    - type: str
    - time_in_force: str
    - limit_price: float
    - stop_price: float
    - client_order_id: str

    returns : dict | None
    """
    payload = collect_params({**kwargs, 'symbol': symbol}, exclude=['qty', 'side', 'type', 'time_in_force', 'limit_price', 'stop_price', 'client_order_id'])
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
    if response.status_code != 200:
        logger.error(f'Error deleting order: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        raise Exception(f'Error deleting order: {response.status_code} {response.text}')
    return response.json()

