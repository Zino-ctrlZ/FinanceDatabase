# test_theta_data.py
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(
    os.environ.get('WORK_DIR'))
sys.path.append(os.environ.get('DBASE_DIR'))

print(os.environ.get('DBASE_DIR'))
import unittest
from unittest.mock import patch, Mock
import pandas as pd

from dbase.DataAPI.sample_data import (
    greek_snapshot_data,
    ohlc_snapshot_data,
    open_interest_snapshot_data,
    quote_snapshot_data,
    list_contracts_data,
        retrieve_ohlc_data,
    retrieve_eod_ohlc_data,
    retrieve_quote_rt_data,
    retrieve_quote_data,
    retrieve_openInterest_data
)
from ThetaData import (
    greek_snapshot,
    ohlc_snapshot,
    open_interest_snapshot,
    quote_snapshot,
    list_contracts,
    retrieve_ohlc,
    retrieve_eod_ohlc,
    retrieve_quote_rt,
    retrieve_quote,
    retrieve_openInterest
)



class TestThetaData(unittest.TestCase):

    @patch('ThetaData.requests.get')
    def test_greek_snapshot(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=greek_snapshot_data)
        response = greek_snapshot('AAPL')
        self.assertEqual(response.status_code, 200)
        self.assertIn('AAPL', response.text)

    @patch('ThetaData.requests.get')
    def test_ohlc_snapshot(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=ohlc_snapshot_data)
        response = ohlc_snapshot('AAPL')
        self.assertEqual(response.status_code, 200)
        self.assertIn('AAPL', response.text)

    @patch('ThetaData.requests.get')
    def test_open_interest_snapshot(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=open_interest_snapshot_data)
        response = open_interest_snapshot('AAPL')
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_quote_snapshot(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=quote_snapshot_data)
        response = quote_snapshot('AAPL')
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_list_contracts(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=list_contracts_data)
        response = list_contracts('AAPL', '2022-01-01')
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    #    @patch('ThetaData.requests.get')
    def test_retrieve_ohlc(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=retrieve_ohlc_data)
        response = retrieve_ohlc('AAPL', '2022-01-21', '2022-01-21', 'C', 20220101, 150.0)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_retrieve_eod_ohlc(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=retrieve_eod_ohlc_data)
        response = retrieve_eod_ohlc('AAPL', '2022-01-21', '2022-01-21', 'C', 20220101, 150.0)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_retrieve_quote_rt(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=retrieve_quote_rt_data)
        response = retrieve_quote_rt('AAPL', '2022-01-21', '2022-01-21', 'C', 20220101, 150.0)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_retrieve_quote(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=retrieve_quote_data)
        response = retrieve_quote('AAPL', '2022-01-21', '2022-01-21', 'C', 20220101, 150.0)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

    @patch('ThetaData.requests.get')
    def test_retrieve_openInterest(self, mock_get):
        mock_get.return_value = Mock(status_code=200, text=retrieve_openInterest_data)
        response = retrieve_openInterest('AAPL', '2022-01-21', '2022-01-21', 'C', 20220101, 150.0)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn('AAPL', response.to_string())

if __name__ == '__main__':
    unittest.main()