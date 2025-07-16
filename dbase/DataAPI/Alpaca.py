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
def get_option_chain(symbol: str, type: str, **kwargs): 
    params = collect_params({**kwargs, 'type': type}, exclude=['symbol'])
    url = f'https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}'
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting option chain: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        return None
    return response.json()

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

def get_option_chain_all(symbol: str, type: str, **kwargs): 
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
def get_option_contracts(underlying_symbols: str, type: str, **kwargs): 
    params = collect_params({**kwargs, 'type': type, 'underlying_symbols': underlying_symbols})
    url = get_base_url(paper=True) + '/options/contracts'
    url = add_query_params(url, params)
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        logger.error(f'Error getting option contracts: {response.status_code} {response.text}')
        print(f'Error: {response.text} {response.status_code}')
        return None
    return response.json()

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
def get_option_contracts_all(underlying_symbols: str, type: str, **kwargs): 
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