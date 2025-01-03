
load_dotenv()
sys.path.append(
    os.environ.get('DBASE_DIR'))

sys.path.append(
    os.environ.get('WORK_DIR'))  # type: ignore
from trade.assets.Calculate import Calculate
from trade.assets.rates import get_risk_free_rate_helper
from trade.assets.Stock import Stock
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData import *
import numpy as np
import pandas as pd
from trade.helpers.helper import *
from dotenv import load_dotenv


logger = setup_logger('dbase.DataAPI.Organizers')
"""
This module carries out all function related to organizing market retrieved from various vendors

"""
def generate_optionData_to_save(symbol, 
                            end_date:str, 
                            exp:str, 
                            right: str, 
                            start_date: int, 
                            strike: float,
                            print_url = False, 
                            rt = True,
                            range_filters = None,
                            timeAggFlag ='EOD'
    
):
    from pathos.multiprocessing import ProcessingPool as Pool
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
    timeAggFlag: 'EOD' to query EOD data or 'INTRA' to query intra day data
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    ## Start by organizing necessary data points
    if timeAggFlag.upper() == 'EOD':
        data = retrieve_eod_ohlc(symbol = symbol, end_date = end_date, exp = exp, right = right, start_date = start_date, strike = strike, print_url = print_url, rt = rt)
    elif timeAggFlag.upper() == 'INTRA':
        data = retrieve_ohlc(symbol = symbol, end_date = end_date, exp = exp, right = right, start_date = start_date, strike = strike, print_url = print_url)
        

    if range_filters:
        data = data[data.index.isin(range_filters)]

    
    if len(data.columns) > 2 and len(data) != 0:
        data[['Strike','Expiration', 'Put/Call']] = strike, exp, right
        stock = Stock(symbol)
        if timeAggFlag.upper() =='EOD':
            close = stock.spot(ts= True, ts_timeframe='day', ts_timewidth='1', ts_start=start_date, ts_end = end_date)['close'] ## Ensure to put if statement for intraday
            dividend = stock.div_yield_history(start = start_date)
            rates_ts = get_risk_free_rate_helper('1d')
            RF_rate = rates_ts['annualized']
            RF_name = rates_ts['name']
            
        elif timeAggFlag.upper() == 'INTRA':
            close = stock.spot(ts= True, ts_timeframe='hour', ts_timewidth='1', ts_start=start_date, ts_end = end_date)['close'] ## Ensure to put if statement for intraday
            dividend = stock.div_yield_history(start = start_date)
            dividend = resample(dividend, '1h') if isinstance(dividend, pd.Series) else dividend
            rates_ts = get_risk_free_rate_helper('1h')
            RF_rate = rates_ts['annualized']
            RF_name = rates_ts['name']


        data['Underlier_price'] = close
        data['RF_rate'] = RF_rate
        data['RF_rate_name'] = RF_name
        data['dividend'] = dividend
        tick_date = pd.to_datetime(exp).strftime('%Y%m%d')
        data['OptionTick'] = generate_option_tick(symbol, right, exp, strike)
        data['Underlier'] = symbol.upper()
        data['Datetime'] = data.index.strftime('%Y-%m-%d') if timeAggFlag.upper() == 'EOD' else data.index.strftime('%Y-%m-%d %H:%M:%S')
        data.ffill(inplace = True)


        
        return data


def Calc_Risks(data, timeAggFlag, symbol, end_date,exp, right, start_date, strike,):
    data['BS_IV'] = data.apply(lambda x: IV_handler(
                                                            price = x['Close'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            t = time_distance_helper(exp = x['Expiration'], strt = x['Datetime']),
                                                            r = x['RF_rate'],
                                                            q = x['dividend'],
                                                            flag = x['Put/Call'].lower()
    ), axis = 1)


    data['Binomial_IV'] = data.apply(lambda x: binomial_implied_vol(
                                                            price = x['Close'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            T = x['Expiration'],
                                                            r = x['RF_rate'],
                                                            dividend_yield = x['dividend'],
                                                            option_type = x['Put/Call'].lower(),
                                                            pricing_date = x['Datetime']
    ), axis = 1)
    greeks = data.apply(lambda x: Calculate.greeks(
        S = x['Underlier_price'],
        K = x['Strike'],
        r = x['RF_rate'],
        sigma = x['BS_IV'],
        start = x['Datetime'],
        flag = x['Put/Call'],
        exp = x['Expiration'],
        y = x['dividend']
    ), axis = 1, result_type = 'expand')
    data = pd.concat([data, greeks], axis = 1)
    data['Dollar_Delta'] = data['Delta'] * data['Underlier_price']

    ## Calculate IV for midpoint & weighted
    data['midpoint_BS_IV'] = data.apply(lambda x: IV_handler(
                                                            price = x['Midpoint'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            t = time_distance_helper(exp = x['Expiration'], strt = x['Datetime']),
                                                            r = x['RF_rate'],
                                                            q = x['dividend'],
                                                            flag = x['Put/Call'].lower()
    ), axis = 1)

    data['midpoint_Binomial_IV'] = data.apply(lambda x: binomial_implied_vol(
                                                            price = x['Midpoint'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            T = x['Expiration'],
                                                            r = x['RF_rate'],
                                                            dividend_yield = x['dividend'],
                                                            option_type = x['Put/Call'].lower(),
                                                            pricing_date = x['Datetime']
                                                        ), axis = 1)

    data['weighted_midpoint_BS_IV'] = data.apply(lambda x: IV_handler(
                                                            price = x['Weighted_midpoint'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            t = time_distance_helper(exp = x['Expiration'], strt = x['Datetime']),
                                                            r = x['RF_rate'],
                                                            q = x['dividend'],
                                                            flag = x['Put/Call'].lower()
    ), axis = 1)

    data['weighted_midpoint_Binomial_IV'] = data.apply(lambda x: binomial_implied_vol(
                                                            price = x['Weighted_midpoint'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            T = x['Expiration'],
                                                            r = x['RF_rate'],
                                                            dividend_yield = x['dividend'],
                                                            option_type = x['Put/Call'].lower(),
                                                            pricing_date = x['Datetime']
                                                            ), axis = 1)


    data['bid_IV'] = data.apply(lambda x: binomial_implied_vol(
                                                    price = x['CloseBid'],
                                                    S = x['Underlier_price'],
                                                    K = x['Strike'],
                                                    T = x['Expiration'],
                                                    r = x['RF_rate'],
                                                    dividend_yield = x['dividend'],
                                                    option_type = x['Put/Call'].lower(),
                                                    pricing_date = x['Datetime']
                                                    ), axis = 1)

    data['ask_IV'] = data.apply(lambda x: binomial_implied_vol(
                                                            price = x['CloseAsk'],
                                                            S = x['Underlier_price'],
                                                            K = x['Strike'],
                                                            T = x['Expiration'],
                                                            r = x['RF_rate'],
                                                            dividend_yield = x['dividend'],
                                                            option_type = x['Put/Call'].lower(),
                                                            pricing_date = x['Datetime']
                                                            ), axis = 1)
        

    
    greeks_mid = data.apply(lambda x: Calculate.greeks(
        S = x['Underlier_price'],
        K = x['Strike'],
        r = x['RF_rate'],
        sigma = x['midpoint_BS_IV'],
        start = x['Datetime'],
        flag = x['Put/Call'],
        exp = x['Expiration'],
        y = x['dividend']
    ), axis = 1, result_type = 'expand')
    greeks_mid.columns = [f'midpoint_{x}' for x in greeks_mid.columns]
    data = pd.concat([data, greeks_mid], axis = 1)
    data['midpoint_Dollar_Delta'] = data['midpoint_Delta'] * data['Underlier_price']

    greeks_weigh = data.apply(lambda x: Calculate.greeks(
        S = x['Underlier_price'],
        K = x['Strike'],
        r = x['RF_rate'],
        sigma = x['weighted_midpoint_BS_IV'],
        start = x['Datetime'],
        flag = x['Put/Call'],
        exp = x['Expiration'],
        y = x['dividend']
    ), axis = 1, result_type = 'expand')
    greeks_weigh.columns = [f'weighted_midpoint_{x}' for x in greeks_weigh.columns]
    data = pd.concat([data, greeks_weigh], axis = 1)
    data['weighted_midpoint_Dollar_Delta'] = data['weighted_midpoint_Delta'] * data['Underlier_price']


    if timeAggFlag.upper() == "EOD":
        openInterest = retrieve_openInterest(symbol, end_date,exp, right, start_date, strike, print_url=False)
        data.index.name = 'Date'
        data['Datetime'] = pd.to_datetime(data.Datetime)
        data = data.merge(openInterest[['Datetime', 'Open_interest']], on = 'Datetime', how = 'left')
        data.rename(columns= {'Open_interest': 'OpenInterest'}, inplace = True)
    elif timeAggFlag.upper() == 'INTRA':
        data['Datetime'] = pd.to_datetime(data.Datetime)
        data['OpenInterest'] = 0
        data.index.name = 'Date'

    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ## Check for inf
    data.replace([np.inf, -np.inf], 0, inplace = True)


    ## Fillna with 0
    data.fillna(0, inplace = True)


    ## Rename open interest
    data.rename(columns = {'Open_interest':'OpenInterest'})
        
    return data