import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(
    os.environ.get('WORK_DIR'))
sys.path.append(os.environ.get('TRADE_PKG_DIR'))
sys.path.append(os.environ.get('DBASE_DIR'))
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


def retrieve_ohlc(symbol, end_date: str, exp: str, interval: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float), f'strike should be type float, recieved {type(strike)}'
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
        logger.error(
            f'Following error for {symbol}, {exp}, {right}, {strike}, Interval: {interval}')
        logger.error(
            f'Retrieve OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        print('Column mismatch. Check log')
    data.rename(columns={x: x.capitalize()
                for x in data.columns}, inplace=True)
    # print(data.columns)
    data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
    data['Date2'] = pd.to_datetime(data.Date.astype(
        str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2 + ' ' + data.time
    data['datetime'] = pd.to_datetime(data.Date3)
    data.set_index('datetime', inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Count']]
    data.index.name = 'Date'
    return data


def retrieve_eod_ohlc(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, print_url=False, rt=True):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float), f'strike should be type float, recieved {type(strike)}'
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
        print('Column mismatch. Check log')
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


def retrieve_quote(symbol, end_date: str, exp: str, interval: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00'):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float), f'strike should be type float, recieved {type(strike)}'
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
    data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
    data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
    data.rename(columns={x: x.capitalize()
                for x in data.columns}, inplace=True)
    # # print(data.columns)
    data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
    data['Date2'] = pd.to_datetime(data.Date.astype(
        str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2 + ' ' + data.time
    data['datetime'] = pd.to_datetime(data.Date3)
    data.set_index('datetime', inplace=True)
    data.drop(columns=['time', 'Date2', 'Date3', 'Ms_of_day'], inplace=True)

    return data


def retrieve_openInterest(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float,  print_url=False):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest?"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "right": right,
                   "start_date": start_date, "strike": strike, 'rth': False}
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
    data.drop(columns=['time', 'Date2', 'Date3', 'Ms_of_day'], inplace=True)
    return data


def retrieve_quote_rt(symbol, end_date: str, exp: str, interval: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00'):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    assert isinstance(
        strike, float), f'strike should be type float, recieved {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True)*60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_miliseconds(start_time))
    end_time = str(convert_time_to_miliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote?"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp, "ivl": ivl, "right": right,
                   "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    # print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text))
    data['midpoint'] = data[['bid', 'ask']].sum(axis=1)/2
    data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (
        data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
    data.rename(columns={x: x.capitalize()
                for x in data.columns}, inplace=True)
    # # print(data.columns)
    data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
    data['Date2'] = pd.to_datetime(data.Date.astype(
        str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2 + ' ' + data.time
    data['datetime'] = pd.to_datetime(data.Date3)
    data.set_index('datetime', inplace=True)
    data.drop(columns=['time', 'Date2', 'Date3', 'Ms_of_day'], inplace=True)

    return data


def resample(data, interval):
    string, integer = extract_numeric_value(interval)
    TIMEFRAME_MAP = {'d': 'B', 'h': 'H', 'm': 'MIN',
                     'M': 'BME', 'w': 'W-FRI', 'q': 'BQE'}
    assert string in TIMEFRAME_MAP.keys(
    ), f"Available Timeframe Alias are {TIMEFRAME_MAP.keys()}, recieved '{string}'"
    interval = f"{integer}{TIMEFRAME_MAP[string]}"
    data = data[data.sum(axis=1) != 0]
    if string == 'h':
        data = data.resample('60T', offset='30min', origin='start_day').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Count': 'sum'})
    else:
        data = data.resample(TIMEFRAME_MAP[string]).agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Count': 'sum'})
    return data.dropna()

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
    return requests.get(url, headers=headers, params=querystring)


def quote_snapshot(symbol):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    return requests.get(url, headers=headers, params=querystring)


def list_contracts(symbol, start_date):
    url = "http://127.0.0.1:25510/v2/list/contracts/option/trade"
    querystring = {"start_date": start_date,
                   "root": symbol,  "use_csv": "true"}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    data = pd.read_csv(StringIO(response.text))
    if 'strike' in data.columns:
        data['strike'] = data.strike/1000
    return data

def retrieve_ohlc(symbol, end_date: int, exp: int, ivl: int, right: str, start_date: int, strike: int):
    """
    Interval size in miliseconds. 1 minute is 6000
    """
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {"end_date": end_date, "root": symbol,  "use_csv": "true", "exp": exp,
                   "ivl": ivl, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    requests.get(url, headers=headers, params=querystring)
    data = pd.read_csv(StringIO(response.text))
    data['strike'] = data.strike/1000
    return data


def retrieve_option_ohlc(symbol: str, exp:str, strike : float, right:str, start_date:str, end_date:str ): 
    """
        returns eod ohlc for all the days between start_date and end_date 
        Interval is default to 3600000
    """
    strike = strike * 1000
    strike = int(strike) if strike.is_integer() else strike
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "ivl": 3600000, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    if(__isSuccesful(response.status_code)): 
        data = pd.read_csv(StringIO(response.text))
        if (len(data.columns)) > 1: 
            data['mean_volume'] = data.groupby('date')['volume'].transform('mean')
            data = data.loc[data.groupby('date')['volume'].apply(lambda x: (x - x.mean()).abs().idxmin())]
            data = data.drop_duplicates(subset='date', keep='last')
            data = data.drop(columns=['mean_volume'])
            data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
            return data
        else: 
            print('Error in retrieving data: ', data) 
            return f"No data retrieved {response.status_code}  {response.text}"
    else: 
        return f"{response.status_code}  {response.text}"
    
    
def __isSuccesful(status_code: int): 
    return status_code >= 200 and status_code < 300

def is_theta_data_retrieval_successful(response): 
    return type(response) != str