import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(
    os.environ['WORK_DIR'])
sys.path.append(os.environ['DBASE_DIR'])
from trade.helpers.Logging import setup_logger

import requests
import re
import time
from io import StringIO
import pandas as pd
import os
import json
from datetime import time as dtTime
import numpy as np
from dbase.utils import add_eod_timestamp, enforce_bus_hours, PRICING_CONFIG
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS, verify_ticker
from copy import deepcopy

    ##To-Do: Add a data cleaning function to remove zeros and inf and check for other anomalies. 
    ## In the function, add a logger to log the anomalies
"""
This Module is responsible for organizing all functions related to accessing data from ThetaData Vendor

"""


logger = setup_logger('dbase.DataAPI.ThetaData')
proxy_url = os.environ.get('PROXY_URL') if os.environ.get('PROXY_URL') else None

if proxy_url is None:
    print('No Proxy URL found. ThetaData API will default to direct access')
else:
    print(f'Using Proxy URL: {proxy_url}')



def resolve_ticker_history(kwargs, _callable, _type = 'historical'):
    if _type == 'historical':
        tick = kwargs['symbol']
        change_date = TICK_CHANGE_ALIAS[tick][-1]
        old_tick = TICK_CHANGE_ALIAS[tick][0]
        new_tick = TICK_CHANGE_ALIAS[tick][1]
        old_tick_kwargs = deepcopy(kwargs)
        new_tick_kwargs = deepcopy(kwargs)
        old_tick_kwargs['symbol'] = old_tick
        new_tick_kwargs['symbol'] = new_tick

        ## Retrieve the data for the old tick
        old_tick_data = _callable(**old_tick_kwargs) if pd.Timestamp(change_date) > pd.Timestamp(kwargs['start_date']) else None
        new_tick_data = _callable(**new_tick_kwargs) if pd.Timestamp(change_date) <= pd.Timestamp(kwargs['end_date']) else None

        ## If no data is found for the old tick, then we will just return the new tick data. Change to dataframe to avoid errors when concatenating
        if old_tick_data is None:
            logger.info(f'No data found for Old_tick {old_tick}')
            old_tick_data = pd.DataFrame()
        if new_tick_data is None:
            logger.info(f'No data found for new_tick {new_tick}')
            new_tick_data = pd.DataFrame()

        full_data = pd.concat([old_tick_data, new_tick_data])
        return full_data
    elif _type == 'snapshot':
        tick = kwargs['symbol']
        change_date = TICK_CHANGE_ALIAS[tick][-1]
        old_tick = TICK_CHANGE_ALIAS[tick][0]
        new_tick = TICK_CHANGE_ALIAS[tick][1]
        new_tick_kwargs = deepcopy(kwargs)
        new_tick_kwargs['symbol'] = old_tick if pd.Timestamp(kwargs['start_date']) <= pd.Timestamp(change_date) else new_tick
        return _callable(**new_tick_kwargs)




def request_from_proxy(thetaUrl, queryparam, instanceUrl, print_url = False): 
    request_string = f"{thetaUrl}?{'&'.join([f'{key}={value}' for key, value in queryparam.items()])}" 
    print(request_string) if print_url else None
    payload = json.dumps({
    "url": request_string,
    "method": "GET",
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", instanceUrl, headers=headers, data=payload)
    return response

def greek_snapshot(symbol, proxy=proxy_url):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))


def ohlc_snapshot(symbol, proxy = proxy_url):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/ohlc"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))


def open_interest_snapshot(symbol, proxy = proxy_url):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))


def quote_snapshot(symbol, proxy = proxy_url):
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))

def list_contracts(symbol, start_date, print_url = False, proxy = proxy_url, **kwargs):
    pass_kwargs = {'start_date': start_date, 'symbol': symbol, 'print_url': print_url}
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0) 
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    url = "http://127.0.0.1:25510/v2/list/contracts/option/quote"
    querystring = {"start_date": start_date ,"root": symbol,  "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, list_contracts, _type = 'snapshot')

    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}" 
        print(response_url) if print_url else None
    else:
        response = requests.get(url, headers=headers, params=querystring)
        print(response.url) if print_url else None

    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if data.shape[0] == 0:
        logger.error(f'No contracts found for {symbol} on {start_date}')
        logger.error(f'response: {response.text}')
        logger.info(f'Kwargs: {locals()}')
        return
    data['strike'] = data.strike/1000
    return data


def identify_length(string, integer, rt=False):
    if rt:
        TIMEFRAMES_VALUES = {'m': 1, 'h': 60, 'd': 60*24, 'w': 60*24*7}
    else:
        TIMEFRAMES_VALUES = {'d': 1, 'w': 5, 'm': 30, 'y': 252, 'q': 91}
    assert string in TIMEFRAMES_VALUES.keys(
    ), f'Available timeframes are {TIMEFRAMES_VALUES.keys()}, recieved "{string}"'
    return integer * TIMEFRAMES_VALUES[string]


def extract_numeric_value(timeframe_str):
    match = re.findall(r'(\d+)([a-zA-Z]+)', timeframe_str)
    integers = [int(num) for num, _ in match][0]
    strings = [str(letter) for _, letter in match][0]
    return strings, integers


def retrieve_ohlc(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float, start_time: str = PRICING_CONFIG['MARKET_OPEN_TIME'], print_url=False, proxy: str = proxy_url):
    """
    Interval size in miliseconds. 1 minute is 6000
    proxy the endpoint to the proxy server http://<ip>:<port>/thetadata
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    interval = PRICING_CONFIG['INTRADAY_AGG']
    strike_og, start_og, end_og, exp_og, start_time_og = strike, start_date, end_date, exp, start_time
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl,
                   "right": right, "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False}
    headers = {"Accept": "application/json"}

    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}" 
        print(response_url) if print_url else None
    else:
        response = requests.get(url, headers=headers, params=querystring)
        print(response.url) if print_url else None


    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')

    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_ohlc')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
        logger.error('Nothing returned at all')
        logger.error('Column mismatch. Check log')
        logger.info(f'Kwargs: {locals()}')
        return
    else:

        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        data['time'] = data.Ms_of_day.apply(lambda c: convert_milliseconds(c))
        #use proxy option
        if proxy:
            quote_data = retrieve_quote(symbol, end_og, exp_og, right,start_og ,strike_og, start_time = start_time_og, proxy = proxy)
        else: 
            quote_data = retrieve_quote(symbol, end_og, exp_og, right,start_og ,strike_og, start_time = start_time_og,)

        ## Merging data into quote data, because quote data has complete dates, whereas OHLC only has dates when traded
        quote_data = quote_data[['Date', 'time', 'Ask_size', 'Ask', 'Bid', 'Bid_size', 'Weighted_midpoint','Midpoint']]
        data = quote_data.merge(data, on = ['Date', 'time'], how = 'left')
        data.rename(columns = {
            'Ask': 'CloseAsk',
            'Bid': 'CloseBid'
        }, inplace = True)
        data['Date'] = data['Date'].astype(str) + ' ' + data['time']
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'
                    }, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume',
         'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        data = enforce_bus_hours(resample(data, PRICING_CONFIG['INTRADAY_AGG']))

    return data


def retrieve_eod_ohlc(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float, print_url=False, rt=True, proxy = proxy_url, **kwargs):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    pass_kwargs = {'symbol': symbol, 'end_date': end_date, 'exp': exp, 'right': right, 'start_date': start_date, 'strike': strike, 'print_url': print_url}
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_eod_ohlc, _type = 'historical')
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true",
                   "exp": exp, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}" 
        print(response_url) if print_url else None
    else:
        response = requests.get(url, headers=headers, params=querystring)
        print(response.url) if print_url else None


    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')

        
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_eod_ohlc')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
        logger.error('Nothing returned at all')
        return
    else:

        data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        
        data['time'] = '16:00:00' if rt else ''
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime(f'%Y-%m-%d {PRICING_CONFIG["MARKET_CLOSE_TIME"]}'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'
                    }, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume',
         'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        # data = data[data[['Open', 'High', 'Low', 'Close', 'Bid_size',
        #                   'CloseBid', 'Ask_size', 'CloseAsk']].sum(axis=1) != 0]
        

    return data


async def retrieve_eod_ohlc_async(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float, print_url=False, rt=True, proxy = proxy_url, **kwargs):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    pass_kwargs = {'symbol': symbol, 'end_date': end_date, 'exp': exp, 'right': right, 'start_date': start_date, 'strike': strike}
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_eod_ohlc_async, _type = 'historical')
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true",
                   "exp": exp, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    
    start_timer = time.time()


    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)


    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')

        
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_eod_ohlc')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
        logger.error('Nothing returned at all')
        return
    else:

        data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        
        data['time'] = '16:00:00' if rt else ''
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data['datetime'].hour = 16
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'
                    }, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume',
         'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        

    return data

  

def retrieve_bulk_eod(
    symbol,
    exp,
    start_date,
    end_date,
    proxy = proxy_url,
    print_url = False,
    **kwargs
):

    pass_kwargs = {
        'symbol': symbol,
        'exp': exp,
        'start_date': start_date,
        'end_date': end_date,
        'print_url': print_url
    }
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_bulk_eod, _type = 'historical')

    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    url = 'http://127.0.0.1:25510/v2/bulk_hist/option/eod'
    querystring = {
    'root': symbol,
    'exp': exp,
    'start_date': start_date,
    'end_date': end_date,
    'use_csv': 'true'
    }

    headers = {"Accept": "application/json"}
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
            logger.info('')
            logger.info(f'Long response time for Bulk EOD {symbol}, {exp}')
            logger.info(f'Response time: {end_timer - start_timer}')
            logger.info(f'Response URL: {response.url}')

    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_bulk_eod')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
        logger.error('Nothing returned at all')
    else:

        data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime(f'%Y-%m-%d {PRICING_CONFIG["MARKET_CLOSE_TIME"]}'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'}, inplace=True)
        data['Strike'] = data.Strike/1000
        data['Expiration'] = pd.to_datetime(data.Expiration, format = '%Y%m%d')
        columns = ['Root', 'Strike', 'Expiration', 'Right', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'

    return data


  
def retrieve_quote_rt(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float, start_time: str = PRICING_CONFIG['MARKET_OPEN_TIME'], print_url=False, end_time=PRICING_CONFIG['MARKET_CLOSE_TIME'], ts = False, proxy = proxy_url, **kwargs):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    interval = '1h'
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    pass_kwargs = {'symbol': symbol, 'end_date': end_date, 'exp': exp, 'right': right, 'start_date': start_date, 
                   'strike': strike, 'start_time': start_time, 'end_time': end_time, 'interval': interval,
                   'print_url': print_url, 'ts': ts}
    
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1

        return resolve_ticker_history(pass_kwargs, retrieve_quote_rt, _type = 'historical')
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote" if not ts else "http://127.0.0.1:25510/v2/hist/option/quote"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl, "right": right,
                   "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    end_timer = time.time()

    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')

    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_quote_rt')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
    else:
        data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
            data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        # # print(data.columns)
        data['time'] = data['Ms_of_day'].apply(lambda c: convert_milliseconds(c))
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime('%Y-%m-%d'))  ## Change this to "%Y-%m-%d %H:%M:%S"
        data['Date3'] = data.Date2 + ' ' + data.time
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.drop(columns=[ 'Date2', 'Date3', 'Ms_of_day', 'time', 'Date'], inplace=True)

    return data

def retrieve_quote(symbol, 
                   end_date: str, 
                   exp: str, 
                   right: str, 
                   start_date: str, 
                   strike: float, 
                   start_time: str = PRICING_CONFIG['MARKET_OPEN_TIME'], 
                   print_url=False, 
                   end_time=PRICING_CONFIG['MARKET_CLOSE_TIME'],
                   interval = '30m', 
                   proxy = proxy_url,
                   **kwargs):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    pass_kwargs = {'symbol': symbol, 'end_date': end_date, 'exp': exp, 'right': right, 'start_date': start_date, 
                'strike': strike, 'start_time': start_time, 'end_time': end_time, 'interval': interval,
                'print_url': print_url}
    
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1

        return resolve_ticker_history(pass_kwargs, retrieve_quote, _type = 'historical')
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike = round(strike * 1000, 0)
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/hist/option/quote"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl, "right": right,
                   "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    

    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}" 
        print(response_url) if print_url else None
    else:
        response = requests.get(url, headers=headers, params=querystring)
        print(response.url) if print_url else None
    end_timer = time.time()


    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')
    
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_quote function')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'EOD OHLC mismatching dataframe size. Response: {data.columns[0]}')
        logger.error(f'No data returned at all')
        logger.info(f'Kwargs: {locals()}')
        print('Column mismatch. Check log')
        return
    data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
    data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
    data.rename(columns={x: x.capitalize()
                for x in data.columns}, inplace=True)
    # # print(data.columns)
    data['time'] = data['Ms_of_day'].apply(lambda c: convert_milliseconds(c))
    data['Date2'] = pd.to_datetime(data.Date.astype(
        str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2 + ' ' + data.time
    data['datetime'] = pd.to_datetime(data.Date3)
    data.set_index('datetime', inplace=True)
    data.drop(columns=['Date2', 'Date3', 'Ms_of_day'], inplace=True)

    return data



def retrieve_openInterest(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float,  print_url=False, proxy = proxy_url, **kwargs):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    pass_kwargs = {'symbol': symbol, 'end_date': end_date, 'exp': exp, 'right': right, 'start_date': start_date, 
            'strike': strike, 'print_url': print_url}
    
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_openInterest
                                      , _type = 'historical')
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp,"right": right,
                   "start_date": start_date, "strike": strike,'rth': False}
    headers = {"Accept": "application/json"}
    
    start_timer = time.time()
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')



    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url=print_url)
    else:
        response = requests.get(url, headers=headers, params=querystring)

        
    if not __isSuccesful(response.status_code):
        logger.error('') 
        logger.error(f'Error in retrieve_openInterest')
        logger.error(f'Following error for: {locals()}')
        logger.error(f'Error in retrieving data: {response.text}')
        logger.error('Nothing returned at all')
        logger.info(f'Kwargs: {locals()}')
        return

    try:
        print(response.url) if print_url else None
        data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        # # # print(data.columns)
        data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime(f'%Y-%m-%d {PRICING_CONFIG["MARKET_CLOSE_TIME"]}'))
        data['Date3'] = data.Date2
        data['Datetime'] = pd.to_datetime(data.Date3)
        data.drop(columns=[ 'Date2', 'Date3', 'Ms_of_day'], inplace=True)
    except Exception as e:
        logger.error('') 
        logger.error(f'Error in retrieve_openInterest. Error: {e}')
        logger.error(f'Error in retrieving data: {response.text}')
        logger.error('Nothing returned at all')
        logger.info(f'Kwargs: {locals()}')
        raise e
    return data


def retrieve_bulk_open_interest(
    symbol,
    exp,
    start_date,
    end_date,
    proxy = proxy_url,
    print_url = False,
    **kwargs
):

    pass_kwargs = {
        'symbol': symbol,
        'exp': exp,
        'start_date': start_date,
        'end_date': end_date,
        'print_url': print_url
    }
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0)
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_bulk_open_interest, _type = 'historical')

    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    url = 'http://127.0.0.1:25510/v2/bulk_hist/option/open_interest'
    querystring = {
    'root': symbol,
    'exp': exp,
    'start_date': start_date,
    'end_date': end_date,
    'use_csv': 'true'
    }

    headers = {"Accept": "application/json"}
    start_timer = time.time()
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
            logger.info('')
            logger.info(f'Long response time for Bulk EOD {symbol}, {exp}')
            logger.info(f'Response time: {end_timer - start_timer}')
            logger.info(f'Response URL: {response.url}')

    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if not __isSuccesful(response.status_code):
        logger.error('') 
        logger.error(f'Error in retrieve_openInterest')
        logger.error(f'Following error for: {locals()}')
        logger.error(f'Error in retrieving data: {response.text}')
        logger.error('Nothing returned at all')
        logger.info(f'Kwargs: {locals()}')
        return
    try:
        print(response.url) if print_url else None
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        # # # print(data.columns)
        data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date3'] = data.Date2
        data['Datetime'] = pd.to_datetime(data.Date3)
        data.drop(columns=[ 'Date2', 'Date3', 'Ms_of_day'], inplace=True)
        data.Expiration = pd.to_datetime(data.Expiration, format = '%Y%m%d')
        data.Strike = data.Strike/1000
    except Exception as e:
        logger.error('') 
        logger.error(f'Error in retrieve_openInterest. Error: {e}')
        logger.error(f'Error in retrieving data: {response.text}')
        logger.error('Nothing returned at all')
        logger.info(f'Kwargs: {locals()}')
        raise e
    return data



async def retrieve_openInterest_async(symbol, end_date: str, exp: str, right: str, start_date: str, strike: float,  print_url=False, proxy = proxy_url):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp,"right": right,
                   "start_date": start_date, "strike": strike,'rth': False}
    headers = {"Accept": "application/json"}
    
    start_timer = time.time()
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp}, {right}, {strike}')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response.url}')



    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)

        
    if not __isSuccesful(response.status_code):
        logger.error('') 
        logger.error(f'Error in retrieve_openInterest')
        logger.error(f'Following error for: {locals()}')
        logger.error(f'Error in retrieving data: {response.text}')
        logger.error('Nothing returned at all')
        logger.info(f'Kwargs: {locals()}')
        return

    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    data.rename(columns={x: x.capitalize()
                for x in data.columns}, inplace=True)
    # # # print(data.columns)
    data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
    data['Date2'] = pd.to_datetime(data.Date.astype(
        str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2
    data['Datetime'] = pd.to_datetime(data.Date3)
    data.drop(columns=[ 'Date2', 'Date3', 'Ms_of_day'], inplace=True)
    return data



def resample(data, interval, custom_agg_columns = None, method = 'ffill', **kwargs):

    """
    Resamples to a specific interval size
    ps: ffills missing values

    parameters
    ----------
    data : pd.DataFrame or pd.Series
        Data to be resampled

    interval : str
        Interval to resample to. Can be a string like '1m', '5m', '1h', '1d', etc.
        or a number followed by a letter, e.g. '5m' or '1h'

    custom_agg_columns : dict, optional
        Custom aggregation dictionary to use for resampling. The default is None, which uses the default aggregation functions.
        The dictionary should have the format {'column_name': 'agg_func'} where 'agg_func' is a string representing the aggregation function to use.

    Returns
    -------
    pd.DataFrame or pd.Series
        Resampled data.
    """

    ## Will allow for custom aggregation functions, using this resample function
    if isinstance(data.index, pd.MultiIndex):
        logger.info('Resampling a MultiIndex')
        if len(data.index.names) == 2:
            try:
                datetime_col_name = kwargs['datetime_col_name']
            except KeyError:
                logger.critical('`datetime_col_name` not provided for multi index resample, setting to `Datetime`')
                datetime_col_name = 'Datetime'
            
            return _handle_multi_index_resample(data, datetime_col_name, interval, resample_col = custom_agg_columns)
        
        else:
            raise NotImplementedError('Currently only supports multi index with 2 levels')
        
    string, integer = extract_numeric_value(interval)
    TIMEFRAME_MAP = {'d': 'B', 'h': 'BH', 'm': 'MIN',
                     'M': 'BME', 'w': 'W-FRI', 'q': 'BQE', 'y': 'BYS'}
    
    if custom_agg_columns:
        columns = custom_agg_columns
    else:

        columns = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
            'Bid_size':'last', 'CloseBid': 'last', 'Ask_size': 'last', 'CloseAsk': 'last', 
            'Midpoint':'last', 'Weighted_midpoint':'last'}

    assert string in TIMEFRAME_MAP.keys(
    ), f"Available Timeframe Alias are {TIMEFRAME_MAP.keys()}, recieved '{string}'"
    
    ## Add EOD time is DateTimeIndex is EOD Series
    data.index = add_eod_timestamp(data.index)
    if isinstance(data, pd.DataFrame):
        resampled = []

        for col in data.columns:
            if col in columns.keys():
                resampled.append(resample(data[col], interval, method=columns[col]))
            else:
                resampled.append(resample(data[col], interval, method=method))
        data = pd.concat(resampled, axis=1)
        data.columns = [col for col in data.columns]
        return enforce_bus_hours(data.fillna(0))

    elif isinstance(data, pd.Series):
        if string == 'h':
            data = data.resample(f'{integer*60}T', origin=PRICING_CONFIG['MARKET_OPEN_TIME']).__getattr__(method)()
        else:
            data = data.resample(f'{integer}{TIMEFRAME_MAP[string]}').__getattr__(method)()
        return enforce_bus_hours(data.fillna(0))


def _handle_multi_index_resample(
        data: pd.DataFrame,
        datetime_col_name: str,
        interval: str,
        resample_col: str,
) -> pd.DataFrame:
    """
    Handle the edge case where the data is a multi index and we need to resample it

    Args:
        data: The data to resample
        datetime_col_name: The name of the datetime column
        interval: The interval to resample to
        resample_col: The column to resample
    """
    assert len(data.index.names) == 2, f"Currently only supports multi index with 2 levels, got {len(data.index.names)}"

    ## Save the order of the MultiIndex
    idx_names = list(data.index.names)

    ## Provide a split by to resample off
    split_by = deepcopy(idx_names)
    split_by.remove(datetime_col_name)

    ## Break down the data into smaller chunks
    pack_data = dict(tuple(data.groupby(level = split_by)))
    resampled_data_list = []

    ## Resample Individual Data
    for k, v in pack_data.items():
        data = v.copy()
        data = data.reset_index().set_index(datetime_col_name)
        data = resample( data, interval, resample_col)
        data[split_by] = data[split_by].replace(0, k)
        data.set_index(split_by, inplace = True, append = True)
        resampled_data_list.append(data)

    resampled_data = pd.concat(resampled_data_list, axis=0)
    return resampled_data





def convert_milliseconds(ms):
    hours = ms // 3600000
    ms = ms % 3600000
    minutes = ms // 60000
    ms = ms % 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_time_to_miliseconds(time):
    time_obj = pd.to_datetime(time)
    hour = time_obj.hour * 3_600_000
    minute = time_obj.minute * 60_000
    secs = time_obj.second * 1_000
    mili = time_obj.microsecond
    return hour + minute + secs + mili


def retrieve_option_ohlc(symbol: str, exp:str, strike : float, right:str, start_date:str, end_date:str, proxy = proxy_url ): 
    """
        returns eod ohlc for all the days between start_date and end_date 
        Interval is default to 3600000
    """
    strike = strike * 1000
    strike = int(strike) if strike.is_integer() else strike
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "ivl": 3600000, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if(__isSuccesful(response.status_code)): 
        if (len(data.columns)) > 1: 
            data['mean_volume'] = data.groupby('date')['volume'].transform('mean')
            data = data.loc[data.groupby('date')['volume'].apply(lambda x: (x - x.mean()).abs().idxmin())]
            data = data.drop_duplicates(subset='date', keep='last')
            data = data.drop(columns=['mean_volume'])
            data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
            return data
        else: 
            print('Error in retrieving data: ', data) 
            return response
    else: 
        return response
    
    
def __isSuccesful(status_code: int): 
    return status_code >= 200 and status_code < 300

def is_theta_data_retrieval_successful(response): 
    return type(response) != str 


def retrieve_chain_bulk(symbol, 
                        exp, 
                        start_date, 
                        end_date, 
                        end_time,
                        right = None ,
                        proxy = proxy_url,
                        print_url = False,
                        **kwargs) -> pd.DataFrame:
    pass_kwargs = {
        'symbol': symbol,
        'exp': exp,
        'start_date': start_date,
        'end_date': end_date,
        'end_time': end_time,
        'print_url': print_url
    }
    depth = pass_kwargs['depth'] = kwargs.get('depth', 0) 
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d')) if exp else 0
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = 'http://127.0.0.1:25510/v2/bulk_at_time/option/quote'
    querystring = {
        'root': symbol,
        'exp': exp,
        'start_date': start_date,
        'end_date': end_date,
        'ivl': end_time,
        'use_csv': 'true'
    }
    if right:
        querystring['right'] = right
    headers = {"Accept": "application/json"}
    if symbol in TICK_CHANGE_ALIAS.keys() and depth < 1:
        pass_kwargs['depth'] += 1
        return resolve_ticker_history(pass_kwargs, retrieve_chain_bulk, _type = 'snapshot')


    start_timer = time.time()


    #use proxy option
    if proxy:
        response = request_from_proxy(url, querystring, proxy, print_url)
        response_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in querystring.items()])}" 
    else:
        response = requests.get(url, headers=headers, params=querystring)
        response_url = response.url
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    end_timer = time.time()
    if (end_timer - start_timer) > 4:
        logger.info('')
        logger.info(f'Long response time for {symbol}, {exp} in Bulk Chain')
        logger.info(f'Response time: {end_timer - start_timer}')
        logger.info(f'Response URL: {response_url}')
    if len(data.columns) == 1:
        logger.error('')
        logger.error('Error in retrieve_bulk_chain')
        logger.error(f'Following error for: {locals()}')
        logger.error(
            f'ThetaData Response: {data.columns[0]}')
        logger.error('Nothing returned at all')

    else:
        data.columns = data.columns.str.lower()
        data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        
        data['Date2'] = pd.to_datetime(data.Date.astype(
            str)).apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date3'] = data.Date2
        data['Expiration'] = pd.to_datetime(data.Expiration, format = '%Y%m%d')
        data['Strike'] = data.Strike/1000
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'
                    }, inplace=True)
        columns = ['Root', 'Expiration', 'Strike', 'Right', 'Bid_size', 'CloseBid',  'Ask_size', 
            'CloseAsk', 'Date', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        
    return data
