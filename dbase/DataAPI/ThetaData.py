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
import re
from io import StringIO
import pandas as pd
import json

logger = setup_logger('ThetaData')

def request_from_proxy(thetaUrl, queryparam, instanceUrl): 
    request_string = f"{thetaUrl}?{'&'.join([f'{key}={value}' for key, value in queryparam.items()])}"
    payload = json.dumps({
    "url": request_string,
    "method": "GET",
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", instanceUrl, headers=headers, data=payload)
    return response

def greek_snapshot(symbol, proxy=None):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return response

def ohlc_snapshot(symbol, proxy=None):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/ohlc"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return response

def open_interest_snapshot(symbol, proxy=None):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text))

def quote_snapshot(symbol, proxy=None):
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote"
    querystring = {"root": symbol, "exp": "0", "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    return pd.read_csv(StringIO(response.text))

def list_contracts(symbol, start_date, print_url=False, proxy=None):
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    url = "http://127.0.0.1:25510/v2/list/contracts/option/trade"
    querystring = {"start_date": start_date, "root": symbol, "use_csv": "true"}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if data.shape[0] == 0:
        logger.error(f'No contracts found for {symbol} on {start_date}')
        logger.error(f'response: {response.text}')
        return
    data['strike'] = data.strike / 1000
    return data

def retrieve_ohlc(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, proxy=None):
    assert isinstance(strike, float), f'strike should be type float, received {type(strike)}'
    interval = '1h'
    strike_og, start_og, end_og, exp_og = strike, start_date, end_date, exp
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True) * 60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_milliseconds(start_time))
    url = "http://127.0.0.1:25510/v2/hist/option/ohlc"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "ivl": ivl,
                   "right": right, "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(f'Intraday OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        logger.error('Nothing returned at all')
        logger.error('Column mismatch. Check log')
        return
    else:
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data['time'] = data.Ms_of_day.apply(lambda c: convert_milliseconds(c))
        quote_data = retrieve_quote(symbol, end_og, exp_og, right, start_og, strike_og)
        data = data.merge(quote_data[['Date', 'time', 'Ask_size', 'Ask', 'Bid', 'Bid_size', 'Weighted_midpoint', 'Midpoint']], on=['Date', 'time'], how='left')
        data.rename(columns={'Ask': 'CloseAsk', 'Bid': 'CloseBid'}, inplace=True)
        data['Date'] = data['Date'].astype(str) + ' ' + data['time']
        data['Date2'] = pd.to_datetime(data.Date.astype(str)).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'}, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        data = resample(data, '1h')
    return data

def retrieve_eod_ohlc(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, print_url=False, rt=True, proxy=None):
    assert isinstance(strike, float), f'strike should be type float, received {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/eod"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "right": right, "start_date": start_date, "strike": strike}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(f'EOD OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        logger.error('Nothing returned at all')
        logger.error('Column mismatch. Check log')
        return
    else:
        data['midpoint'] = data[['bid', 'ask']].sum(axis=1) / 2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data['time'] = '16:00:00' if rt else ''
        data['Date2'] = pd.to_datetime(data.Date.astype(str)).apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date3'] = data.Date2
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.rename(columns={'Bid': 'CloseBid', 'Ask': 'CloseAsk'}, inplace=True)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk', 'Midpoint', 'Weighted_midpoint']
        data = data[columns]
        data.index.name = 'Datetime'
        data = data[data[['Open', 'High', 'Low', 'Close', 'Bid_size', 'CloseBid', 'Ask_size', 'CloseAsk']].sum(axis=1) != 0]
    return data

def retrieve_quote_rt(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00', ts=False, proxy=None):
    interval = '1h'
    assert isinstance(strike, float), f'strike should be type float, received {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True) * 60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_milliseconds(start_time))
    end_time = str(convert_time_to_milliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/snapshot/option/quote?" if not ts else "http://127.0.0.1:25510/v2/hist/option/quote?"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "ivl": ivl, "right": right, "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(f'EOD OHLC mismatching dataframe size. Column says: {data.columns[0]}')
        print('Column mismatch. Check log')
    else:
        data['midpoint'] = data[['bid', 'ask']].sum(axis=1) / 2
        data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
        data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
        data['time'] = data['Ms_of_day'].apply(lambda c: convert_milliseconds(c))
        data['Date2'] = pd.to_datetime(data.Date.astype(str)).apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date3'] = data.Date2 + ' ' + data.time
        data['datetime'] = pd.to_datetime(data.Date3)
        data.set_index('datetime', inplace=True)
        data.drop(columns=['Date2', 'Date3', 'Ms_of_day', 'time', 'Date'], inplace=True)
    return data

def retrieve_quote(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, start_time: str = '9:30', print_url=False, end_time='16:00', proxy=None):
    interval = '1h'
    assert isinstance(strike, float), f'strike should be type float, received {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    ivl = identify_length(*extract_numeric_value(interval), rt=True) * 60000
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    start_time = str(convert_time_to_milliseconds(start_time))
    end_time = str(convert_time_to_milliseconds(end_time))
    url = "http://127.0.0.1:25510/v2/hist/option/quote?"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "ivl": ivl, "right": right, "start_date": start_date, "strike": strike, "start_time": start_time, 'rth': False, 'end_time': end_time}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    if len(data.columns) == 1:
        logger.error('')
        logger.error(f'Following error for {symbol}, {exp}, {right}, {strike}')
        logger.error(f'EOD OHLC mismatching dataframe size. Response: {data.columns[0]}')
        logger.error(f'No data returned at all')
        print('Column mismatch. Check log')
        return
    data['midpoint'] = data[['bid', 'ask']].sum(axis=1) / 2
    data['weighted_midpoint'] = ((data['ask_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['ask'])) + ((data['bid_size'] / data[['bid_size', 'ask_size']].sum(axis=1)) * (data['bid']))
    data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
    data['time'] = data['Ms_of_day'].apply(lambda c: convert_milliseconds(c))
    data['Date2'] = pd.to_datetime(data.Date.astype(str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2 + ' ' + data.time
    data['datetime'] = pd.to_datetime(data.Date3)
    data.set_index('datetime', inplace=True)
    data.drop(columns=['Date2', 'Date3', 'Ms_of_day'], inplace=True)
    return data

def retrieve_openInterest(symbol, end_date: str, exp: str, right: str, start_date: int, strike: float, print_url=False, proxy=None):
    assert isinstance(strike, float), f'strike should be type float, received {type(strike)}'
    end_date = int(pd.to_datetime(end_date).strftime('%Y%m%d'))
    exp = int(pd.to_datetime(exp).strftime('%Y%m%d'))
    start_date = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
    strike *= 1000
    strike = int(strike)
    url = "http://127.0.0.1:25510/v2/hist/option/open_interest?"
    querystring = {"end_date": end_date, "root": symbol, "use_csv": "true", "exp": exp, "right": right, "start_date": start_date, "strike": strike, 'rth': False}
    headers = {"Accept": "application/json"}
    if proxy:
        response = request_from_proxy(url, querystring, proxy)
    else:
        response = requests.get(url, headers=headers, params=querystring)
    print(response.url) if print_url else None
    data = pd.read_csv(StringIO(response.text)) if proxy is None else pd.read_csv(StringIO(response.json()['data']))
    data.rename(columns={x: x.capitalize() for x in data.columns}, inplace=True)
    data['time'] = data['Ms_of_day'].apply(convert_milliseconds)
    data['Date2'] = pd.to_datetime(data.Date.astype(str)).apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date3'] = data.Date2
    data['Datetime'] = pd.to_datetime(data.Date3)
    data.drop(columns=['Date2', 'Date3', 'Ms_of_day'], inplace=True)
    return data

# Other functions remain unchanged
