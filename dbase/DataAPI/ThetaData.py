import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(
    os.environ.get('WORK_DIR'))
from trade.helpers.helper import pad_string
from trade.helpers.Logging import setup_logger

import requests
import time
import re
from io import StringIO
import numpy as np
import pandas as pd
import os


logger = setup_logger('ThetaData')


"""
This Module is responsible for organizing all functions related to accessing data from ThetaData Vendor

"""

def greek_snapshot(symbol):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    return requests.get(url, headers=headers, params=querystring)


def ohlc_snapshot(symbol):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/ohlc"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    return requests.get(url, headers=headers, params=querystring)


def open_interest_snapshot(symbol):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    return pd.read_csv(StringIO(requests.get(url, headers=headers, params=querystring)))


def quote_snapshot(symbol):
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    return pd.read_csv(StringIO(requests.get(url, headers=headers, params=querystring)))

def list_contracts(symbol, start_date, print_url = False):
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    url = "http://127.0.0.1:25510/v2/list/contracts/option/trade"
    querystring = {"start_date": start_date ,"root": symbol,  "use_csv": "true"}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    if data.shape[0] == 0:
        logger.error(f'No contracts found for {symbol} on {start_date}')
        logger.error(f'response: {response.text}')
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


def retrieve_ohlc(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    interval = '1h'
    strike_og, start_og, end_og, exp_og = strike, start_date, end_date, exp
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
    response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(
            f'Intraday OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        logger.error('Nothing returned at all')
        logger.error('Column mismatch. Check log')
        return
    else:
        

        data.rename(columns={x: x.capitalize()
                    for x in data.columns}, inplace=True)
        data['time'] = data.Ms_of_day.apply(lambda c: convert_milliseconds(c))
        quote_data = retrieve_quote(symbol, end_og, exp_og, right,start_og ,strike_og)
        data = data.merge(quote_data[['Date', 'time', 'Ask_size', 'Ask', 'Bid', 'Bid_size', 'Weighted_midpoint','Midpoint']], on = ['Date', 'time'], how = 'left')
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
        data = resample(data, '1h')
 

    return data


def retrieve_eod_ohlc(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, print_url=False, rt=True):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true",
                   "exp": exp, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(
            f'EOD OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        logger.error('Nothing returned at all')
        logger.error('Column mismatch. Check log')
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
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'
                    }, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume',
         'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        data = data[data[['Open', 'High', 'Low', 'Close', 'Bid_size',
                          'CloseBid', 'Ask_size', 'CloseAsk']].sum(axis=1) != 0]
        

    return data



def retrieve_quote_rt(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00', ts = False):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    interval = '1h'
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote?" if not ts else "http://127.0.0.1:25510/v2/hist/option/quote?"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl, "right": right,
                   "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(
            f'EOD OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        print('Column mismatch. Check log')
    else:
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
        data.drop(columns=[ 'Date2', 'Date3', 'Ms_of_day', 'time', 'Date'], inplace=True)

    return data

def retrieve_quote(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00'):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    interval = '1h'
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/hist/option/quote?"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl, "right": right,
                   "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    # print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(
            f'EOD OHLC mismatching dataframe size. Response: {data.columns[0]}')
        logger.error(f'No data returned at all')
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



def retrieve_openInterest(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float,  print_url=False):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest?"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp,"right": right,
                   "start_date": start_date, "strike": strike,'rth': False}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
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




def resample(data, interval, custom_agg_columns = None):

    """
    Resamples to a specific interval size
    ps: ffills missing values
    """
    
    string, integer = extract_numeric_value(interval)
    TIMEFRAME_MAP = {'d': 'B', 'h': 'H', 'm': 'MIN',
                     'M': 'BME', 'w': 'W-FRI', 'q': 'BQE', 'y': 'BYS'}
    
    if custom_agg_columns:
        columns = custom_agg_columns
    else:

        columns = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
            'Bid_size':'last', 'CloseBid': 'last', 'Ask_size': 'last', 'CloseAsk': 'last', 
            'Midpoint':'last', 'Weighted_midpoint':'last'}

    assert string in TIMEFRAME_MAP.keys(
    ), f"Available Timeframe Alias are {TIMEFRAME_MAP.keys()}, recieved '{string}'"
    interval = f"{integer}{TIMEFRAME_MAP[string]}"
    if isinstance(data, pd.DataFrame):
        if string == 'h':
            data = data.resample(f'{integer*60}T', origin='start_day', offset = '30min').agg(
                columns).ffill()
            data = data[(data.index.time >= pd.Timestamp('9:00').time()) & (data.index.time <= pd.Timestamp('16:00').time()) ]
            return data.fillna(0)
        else:
            data = data.resample(f'{integer}{TIMEFRAME_MAP[string]}').agg(
                columns).ffill()
        return data.fillna(0)

    elif isinstance(data, pd.Series):
        if string == 'h':
            data = data.resample(f'{integer*60}T', offset='30min', origin='start_day').ffill()
        else:

            data = data.resample(f'{integer}{TIMEFRAME_MAP[string]}').agg(
                columns).ffill()
        return data.fillna(0)
# Function to convert milliseconds to hours, minutes, seconds, and milliseconds


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
    hour = time_obj.hour * 3600000
    minute = time_obj.minute * 60000
    secs = time_obj.second * 1000
    mili = time_obj.microsecond
    return hour + minute + secs + mili


# def retrieve_ohlc(symbol, end_date: int, exp: int, ivl: int, right: str, start_date: int, strike: int):
#     """
#     Interval size in miliseconds. 1 minute is 6000
#     """
#     url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
#     querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp,
#                    "ivl": ivl, "right": right, "start_date": start_date, "strike": strike}
#     headers = {"Accept": "application/json"}
#     return requests.get(url, headers=headers, params=querystring)
