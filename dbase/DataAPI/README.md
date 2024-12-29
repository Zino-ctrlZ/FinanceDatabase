# ThetaData Module

This module is responsible for organizing all functions related to accessing data from the ThetaData Vendor. It provides various functions to retrieve financial data such as Greek snapshots, OHLC snapshots, open interest snapshots, and more.

## Setup

Ensure that the required environment variables are set in your `.env` file:

- `WORK_DIR`
- `TRADE_PKG_DIR`
- `DBASE_DIR`
- `PROXY_URL` (optional)

## Functions

### `request_from_proxy(thetaUrl, queryparam, instanceUrl)`

Sends a request to the proxy server.

- **Parameters:**
  - `thetaUrl` (str): The URL to request data from.
  - `queryparam` (dict): The query parameters for the request.
  - `instanceUrl` (str): The URL of the proxy server.

- **Returns:**
  - `response`: The response from the proxy server.

### `greek_snapshot(symbol, proxy=proxy_url)`

Retrieves a snapshot of Greek data for a given symbol.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the Greek data.

### `ohlc_snapshot(symbol, proxy=proxy_url)`

Retrieves an OHLC snapshot for a given symbol.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the OHLC data.

### `open_interest_snapshot(symbol, proxy=proxy_url)`

Retrieves an open interest snapshot for a given symbol.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the open interest data.

### `quote_snapshot(symbol, proxy=proxy_url)`

Retrieves a quote snapshot for a given symbol.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the quote data.

### `list_contracts(symbol, start_date, print_url=False, proxy=proxy_url)`

Lists contracts for a given symbol and start date.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `start_date` (str): The start date in `YYYY-MM-DD` format.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the contract data.

### `identify_length(string, integer, rt=False)`

Identifies the length of a given timeframe.

- **Parameters:**
  - `string` (str): The timeframe string (e.g., 'd', 'w', 'm').
  - `integer` (int): The integer value for the timeframe.
  - `rt` (bool, optional): Whether to use real-time values. Defaults to `False`.

- **Returns:**
  - `int`: The length of the timeframe.

### `extract_numeric_value(timeframe_str)`

Extracts numeric values from a timeframe string.

- **Parameters:**
  - `timeframe_str` (str): The timeframe string.

- **Returns:**
  - `tuple`: A tuple containing the string and integer parts of the timeframe.

### `retrieve_ohlc(symbol, end_date, exp, right, start_date, strike, start_time='9:30', print_url=False, proxy=proxy_url)`

Retrieves OHLC data for a given symbol and date range.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (int): The start date in `YYYYMMDD` format.
  - `strike` (float): The strike price.
  - `start_time` (str, optional): The start time. Defaults to '9:30'.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the OHLC data.

### `retrieve_eod_ohlc(symbol, end_date, exp, right, start_date, strike, print_url=False, rt=True, proxy=proxy_url)`

Retrieves end-of-day OHLC data for a given symbol and date range.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (int): The start date in `YYYYMMDD` format.
  - `strike` (float): The strike price.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `rt` (bool, optional): Whether to use real-time values. Defaults to `True`.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the end-of-day OHLC data.

### `retrieve_quote_rt(symbol, end_date, exp, right, start_date, strike, start_time='9:30', print_url=False, end_time='16:00', ts=False, proxy=proxy_url)`

Retrieves real-time quote data for a given symbol and date range.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (int): The start date in `YYYYMMDD` format.
  - `strike` (float): The strike price.
  - `start_time` (str, optional): The start time. Defaults to '9:30'.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `end_time` (str, optional): The end time. Defaults to '16:00'.
  - `ts` (bool, optional): Whether to use timestamped data. Defaults to `False`.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the real-time quote data.

### `retrieve_quote(symbol, end_date, exp, right, start_date, strike, start_time='9:30', print_url=False, end_time='16:00', proxy=proxy_url)`

Retrieves quote data for a given symbol and date range.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (int): The start date in `YYYYMMDD` format.
  - `strike` (float): The strike price.
  - `start_time` (str, optional): The start time. Defaults to '9:30'.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `end_time` (str, optional): The end time. Defaults to '16:00'.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the quote data.

### `retrieve_openInterest(symbol, end_date, exp, right, start_date, strike, print_url=False, proxy=proxy_url)`

Retrieves open interest data for a given symbol and date range.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (int): The start date in `YYYYMMDD` format.
  - `strike` (float): The strike price.
  - `print_url` (bool, optional): Whether to print the URL. Defaults to `False`.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the open interest data.

### `resample(data, interval, custom_agg_columns=None)`

Resamples data to a specific interval size.

- **Parameters:**
  - `data` (DataFrame or Series): The data to resample.
  - `interval` (str): The interval size (e.g., '1h', '1d').
  - `custom_agg_columns` (dict, optional): Custom aggregation columns. Defaults to `None`.

- **Returns:**
  - `DataFrame` or `Series`: The resampled data.

### `convert_milliseconds(ms)`

Converts milliseconds to hours, minutes, seconds, and milliseconds.

- **Parameters:**
  - `ms` (int): The number of milliseconds.

- **Returns:**
  - `str`: The formatted time string.

### `convert_time_to_miliseconds(time)`

Converts a time string to milliseconds.

- **Parameters:**
  - `time` (str): The time string.

- **Returns:**
  - `int`: The number of milliseconds.

### `retrieve_option_ohlc(symbol, exp, strike, right, start_date, end_date, proxy=proxy_url)`

Retrieves end-of-day OHLC data for all days between the start date and end date.

- **Parameters:**
  - `symbol` (str): The symbol to retrieve data for.
  - `exp` (str): The expiration date in `YYYY-MM-DD` format.
  - `strike` (float): The strike price.
  - `right` (str): The option right ('C' for call, 'P' for put).
  - `start_date` (str): The start date in `YYYY-MM-DD` format.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.
  - `proxy` (str, optional): The proxy URL. Defaults to `proxy_url`.

- **Returns:**
  - `DataFrame`: A pandas DataFrame containing the OHLC data.

### `__isSuccesful(status_code)`

Checks if the status code indicates a successful response.

- **Parameters:**
  - `status_code` (int): The status code.

- **Returns:**
  - `bool`: `True` if the status code indicates success, `False` otherwise.

### `is_theta_data_retrieval_successful(response)`

Checks if the ThetaData retrieval was successful.

- **Parameters:**
  - `response`: The response from the ThetaData retrieval.

- **Returns:**
  - `bool`: `True` if the retrieval was successful, `False` otherwise.


  **Note: This README is AI generated**