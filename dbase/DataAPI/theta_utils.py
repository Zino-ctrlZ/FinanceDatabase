from datetime import datetime
from trade.helpers.helper import parse_option_tick
from .ThetaData import retrieve_quote

def get_option_data_theta_data(opttick:str,
                               start:str|datetime,
                               end: str|datetime):
    """
    Retrieve option price data from ThetaData
    start, end: str or datetime
    1. str: 'YYYY-MM-DD'
    2. datetime: datetime object
    If str, it will be converted to datetime
    If datetime, it will be used directly
    Return: DataFrame
    """
    meta = parse_option_tick(opttick)
    return retrieve_quote(
        symbol=meta['ticker'],
        start_date=start,
        end_date=end,
        strike=meta['strike'],
        right= meta['put_call'],
        exp=meta['exp_date'],
        print_url = True
        
    )