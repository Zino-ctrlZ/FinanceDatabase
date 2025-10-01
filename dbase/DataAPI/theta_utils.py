import pandas as pd
from datetime import datetime
from trade.helpers.helper import parse_option_tick
from .ThetaData import retrieve_quote, retrieve_quote_rt

def get_option_data_theta_data(opttick:str,
                               start:str|datetime,
                               end: str|datetime,
                               rt:bool=False,
                               interval: str = '1h',
                               print_url:bool=False) -> pd.DataFrame:
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
    func = retrieve_quote_rt if rt else retrieve_quote
    data =  func(
        symbol=meta['ticker'],
        start_date=start,
        end_date=end,
        strike=meta['strike'],
        right= meta['put_call'],
        exp=meta['exp_date'],
        interval=interval,
        print_url = print_url
    )

    ## Formatting because column names are inconsistent
    check_cols = {
        'Bid': 'Closebid',
        'Ask': 'Closeask',
    }
    data.rename(columns={x: check_cols[x] for x in check_cols if x in data.columns}, inplace=True)
    data.columns = [x.capitalize() for x in data.columns]
    return data


    