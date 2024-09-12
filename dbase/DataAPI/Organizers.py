
from trade.assets.Calculate import Calculate
from trade.assets.rates import get_risk_free_rate_helper
from trade.assets.Stock import Stock
from dbase.DataAPI.ThetaData import *
import numpy as np
import pandas as pd
from trade.helpers.helper import *
from dotenv import load_dotenv
load_dotenv()
sys.path.append(
    os.environ.get('DBASE_DIR'))

sys.path.append(
    os.environ.get('WORK_DIR'))  # type: ignore
"""
This module carries out all function related to organizing market retrieved from various vendors

"""


def generate_optionData_to_save(symbol,
                                end_date: str,
                                exp: str,
                                right: str,
                                start_date: int,
                                strike: int,
                                print_url=False,
                                rt=True,
                                range_filters=None

                                ):
    """
    Responsible for generating and organizing the data to be saved in the SQL database. Used in ThetaData EOD & Intraday API

    Parameters:
    end_date: Timeseries end date
    exp: Contract Expiration Date
    right: Put/Call
    start_date: Timeseries Start Date
    Strike: Option Strike Price
    print_url: Print URL used to query data
    rt: True to query intraday, False to query EOD
    range_filters: A list of lists. Each list has 2 dates. This serves as the range to pick up in OHLC
    """

    # Start by organizing necessary data points
    data = retrieve_eod_ohlc(symbol=symbol, end_date=end_date, exp=exp, right=right,
                             start_date=start_date, strike=strike, print_url=print_url, rt=rt)
    if range_filters:
        mask = pd.Series([False] * len(data), index=data.index)
        for start, end in range_filters:
            mask |= (mask.index > start) & (mask.index < end)
        data = data[mask]
    if len(data.columns) > 2 and len(data) != 0:
        data[['Strike', 'Expiration', 'Put/Call']] = strike, exp, right
        stock = Stock(symbol)
        close = stock.spot(ts=True, ts_timeframe='day', ts_timewidth='1', ts_start=start_date, ts_end=end_date)[
            'close']  # Ensure to put if statement for intraday
        data['Underlier_price'] = close
        RF_rate = get_risk_free_rate_helper()['daily']
        RF_name = get_risk_free_rate_helper()['name']
        data['RF_rate'] = RF_rate
        data['RF_rate_name'] = RF_name
        data['dividend'] = stock.div_yield_history(start='2023-01-01')
        tick_date = pd.to_datetime(exp).strftime('%Y%m%d')
        data['OptionTick'] = generate_option_tick(symbol, right, exp, strike)
        data['Underlier'] = symbol
        data['Datetime'] = data.index.strftime('%Y-%m-%d')
        data['BS_IV'] = data.apply(lambda x: IV_handler(
            price=x['Close'],
            S=x['Underlier_price'],
            K=x['Strike'],
            t=time_distance_helper(exp=x['Expiration'], strt=x['Datetime']),
            r=x['RF_rate'],
            q=x['dividend'],
            flag=x['Put/Call'].lower()
        ), axis=1)
        # data['Binomial_IV'] = data.apply(lambda x: implied_vol_custom(
        #                                                         market_price = x['Close'],
        #                                                         S0 = x['Underlier_price'],
        #                                                         K = x['Strike'],
        #                                                         exp_date = x['Expiration'],
        #                                                         model = 'bt',
        #                                                         r = x['RF_rate'],
        #                                                         y = x['dividend'],
        #                                                         flag = x['Put/Call'].lower(),
        #                                                         start = x['Expiration']
        # ), axis = 1)

        data['Binomial_IV'] = data.apply(lambda x: binomial_implied_vol(
            price=x['Close'],
            S=x['Underlier_price'],
            K=x['Strike'],
            T=x['Expiration'],
            r=x['RF_rate'],
            dividend_yield=x['dividend'],
            option_type=x['Put/Call'].lower(),
            pricing_date=x['Datetime']
        ), axis=1)
        greeks = data.apply(lambda x: Calculate.greeks(
            S=x['Underlier_price'],
            K=x['Strike'],
            r=x['RF_rate'],
            sigma=x['BS_IV'],
            start=x['Datetime'],
            flag=x['Put/Call'],
            exp=x['Expiration']
        ), axis=1, result_type='expand')
        data = pd.concat([data, greeks], axis=1)
        data['Dollar_Delta'] = data['Delta'] * data['Underlier_price']

        return data