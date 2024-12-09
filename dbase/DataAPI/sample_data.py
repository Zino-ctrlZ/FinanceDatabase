retrieve_ohlc_data = """
Date,Open,High,Low,Close,Volume,Bid_size,CloseBid,Ask_size,CloseAsk,Midpoint,Weighted_midpoint
2022-01-21 09:30:00,150,155,149,152,1000,100,150,100,151,150.5,150.25
"""

retrieve_eod_ohlc_data = """
Date,Open,High,Low,Close,Volume,Bid_size,CloseBid,Ask_size,CloseAsk,Midpoint,Weighted_midpoint
2022-01-21,150,155,149,152,1000,100,150,100,151,150.5,150.25
"""

retrieve_quote_rt_data = """
Date,Open,High,Low,Close,Volume,Bid_size,CloseBid,Ask_size,CloseAsk,Midpoint,Weighted_midpoint
2022-01-21 09:30:00,150,155,149,152,1000,100,150,100,151,150.5,150.25
"""

retrieve_quote_data = """
Date,Open,High,Low,Close,Volume,Bid_size,CloseBid,Ask_size,CloseAsk,Midpoint,Weighted_midpoint
2022-01-21 09:30:00,150,155,149,152,1000,100,150,100,151,150.5,150.25
"""

retrieve_openInterest_data = """
Date,Open,High,Low,Close,Volume,Bid_size,CloseBid,Ask_size,CloseAsk,Midpoint,Weighted_midpoint
2022-01-21 09:30:00,150,155,149,152,1000,100,150,100,151,150.5,150.25
"""

greek_snapshot_data = """
root,exp,delta,gamma,theta,vega
AAPL,20220121,0.5,0.1,-0.05,0.2
"""

ohlc_snapshot_data = """
root,exp,open,high,low,close
AAPL,20220121,150,155,149,152
"""

open_interest_snapshot_data = """
root,exp,open_interest
AAPL,20220121,1000
"""

quote_snapshot_data = """
root,exp,bid,ask
AAPL,20220121,150,151
"""

list_contracts_data = """
root,exp,strike
AAPL,20220121,150000
"""