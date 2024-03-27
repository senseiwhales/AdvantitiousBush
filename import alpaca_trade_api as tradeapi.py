import alpaca_trade_api as tradeapi
import datetime

# Alpaca API authentication
API_KEY = 'PKYM7P7LWL9V2WLADG7P'
API_SECRET = '7MZVax8cgg1wTzoUUKfocOPTyNgJrtOcmNYqIvka'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')

# Request OHLCV data for Bitcoin (BTCUSD)
symbol = 'BTCUSD'
timeframe = '1Min'  # 1 minute timeframe
start_time = (datetime.datetime.now() - datetime.timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')  # 30 minutes ago
end_time = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

try:
    barset = api.get_bars(symbol, timeframe, start=start_time, end=end_time)

    for bar in barset[symbol]:
        print(f"Timestamp: {bar.t}, Open: {bar.o}, High: {bar.h}, Low: {bar.l}, Close: {bar.c}, Volume: {bar.v}")  
    
    print("Successfully fetched OHLCV data for", symbol)
except Exception as e:
    print(f"Failed to fetch OHLCV data for {symbol}: {e}")
