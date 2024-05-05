import numpy as np
import logging
import time
import datetime
import alpaca_trade_api as tradeapi
from sklearn.svm import SVR
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

API_KEY = 'PKZYTDU16C4GW63TOV68'
API_SECRET = 'si6tzwHML9ZS2BLd0IktHkC2K6KkaZdOAACn0JhR'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, symbol, api):
        self.symbol = symbol
        self.api = api
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.models = [SVR(kernel='rbf')]  # Initialize SVM model with radial basis function kernel
        self.current_position = 'flat'  # Initialize current_position attribute
        self.set_random_seed()  # Set random seed

    def update_current_position(self):
        try:
            position = self.alpaca.get_position(self.symbol)
            if position is not None:
                self.current_position = 'long' if float(position.qty) > 0 else 'flat'
            else:
                self.current_position = 'flat'
        except tradeapi.rest.APIError as e:
            if e.status_code == 404:
                self.current_position = 'flat'  # Set position to flat if it doesn't exist
            else:
                logger.error(f"Failed to get position for {self.symbol}: {str(e)}")

    def buy_shares(self, symbol, prediction_amount):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            current_price = self.get_current_price_from_coinmarketcap()
            if current_price is not None:
                num_shares = cash_available / current_price
                num_shares *= prediction_amount
            else:
                num_shares = 0

            self.alpaca.submit_order(
                symbol=symbol,
                qty=num_shares,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            self.current_position = 'long'
            logger.info("Bought %s shares of %s", num_shares, symbol)
        except tradeapi.rest.APIError as e:
            logger.error(f"Failed to buy shares of {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

    def sell_all_shares(self, symbol):
        try:
            position = self.alpaca.get_position(symbol)

            if position:
                qty_to_sell = float(position.qty)

                self.alpaca.submit_order(
                    symbol=symbol,
                    qty=qty_to_sell,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                self.current_position = 'flat'
                logger.info("Sold all shares of %s", symbol)
            else:
                logger.info(f"No position found for {symbol}.")
        except tradeapi.rest.APIError as e:
            logger.error(f"Failed to sell all shares of {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

    def on_trading_iteration(self):
        try:
            # Detailed logging before each operation
            logger.info("Fetching current price from CoinMarketCap for ETH")
            current_price = self.get_current_price_from_coinmarketcap()
            if current_price is None:
                logger.warning("No current price found.")
                return
            
            logger.info(f"Current {self.symbol} Price: {current_price}")

            if current_price is None:
                logger.warning("No current price found.")
                return

            closing_prices = [float(candle['close']) for candle in self.get_coinmarketcap_candles(limit=30)]

            if self.current_position == 'flat':
                # Check for the best time to enter a long position
                if self.predict_buy_signal(current_price, closing_prices):
                    self.buy_shares(self.symbol, 0.3)  # Buy 30% of available cash
                    logger.info("Predicted price is higher than Current Price. Buying.")
            elif self.current_position == 'long':
                # Check for the best time to exit the long position
                if self.predict_sell_signal(current_price, closing_prices):
                    self.sell_all_shares(self.symbol)
                    logger.info("Predicted price is lower than Current Price. Selling.")

        except Exception as e:
            logger.error(f"Error in trading iteration for {self.symbol}: {e}", exc_info=True)

    def get_current_price_from_coinmarketcap(self):
        url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        parameters = {
            'symbol': self.symbol,  # Change this to self.symbol
            'convert': 'USD'
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c',
        }
    
        session = Session()
        session.headers.update(headers)

        try:
            response = session.get(url, params=parameters)
            data = json.loads(response.text)
            price = data['data'][self.symbol]['quote']['USD']['price']  # Change this to self.symbol
            return price
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            logger.error(f"Error fetching data from CoinMarketCap: {str(e)}")

    def predict_buy_signal(self, current_price, previous_closing_prices):
        # Calculate moving average
        moving_average = np.mean(previous_closing_prices)

        # Check if the current price is higher than the moving average
        if current_price > moving_average:
            return True
        else:
            return False

    def predict_sell_signal(self, current_price, previous_closing_prices):
        # Calculate moving average
        moving_average = np.mean(previous_closing_prices)

        # Check if the current price is lower than the moving average
        if current_price < moving_average:
            return True
        else:
            return False

    def backtest(self, start_date, end_date, initial_cash=10000):
        """Simulate trading over a historical period."""
        historical_data = self.load_historical_data('path/to/your/historical/data.csv')
        historical_data = historical_data[start_date:end_date]

        cash = initial_cash
        shares = 0
        portfolio_value = []

        for date, row in historical_data.iterrows():
            current_price = row['Close']
            closing_prices = historical_data.loc[:date]['Close'][-30:]  # Last 30 days of closing prices

            if shares == 0 and self.predict_buy_signal(current_price, closing_prices):
                shares = cash / current_price
                cash = 0
                print(f"Buying on {date}: {shares} shares at {current_price}")
            elif shares > 0 and self.predict_sell_signal(current_price, closing_prices):
                cash = shares * current_price
                shares = 0
                print(f"Selling on {date}: All shares at {current_price}")

            portfolio_value.append(cash + (shares * current_price))

        final_value = portfolio_value[-1]
        print(f"Final portfolio value: {final_value} (Initial: {initial_cash})")
        return portfolio_value
    
    def set_random_seed(self):
        seed_value = 143  # You can change this seed value
        np.random.seed(seed_value)

if __name__ == "__main__":
    symbol = 'ETH'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='BTC', api=alpaca_api)

    # Loop to run every 60 seconds
    while True:
        start_time = time.time()

        # Check for trading opportunities every 60 seconds
        try:
            ml_trader.on_trading_iteration()
        except Exception as e:
            if '429' in str(e):
                print("Rate limit exceeded. Sleeping for 60 seconds.")
                time.sleep(60)
            elif 'ambiguous' in str(e):
                # Replace this with the actual line of code causing the error
                ambiguous_array = np.array([True, False])
                if ambiguous_array.any():  # or ambiguous_array.all(), depending on your logic
                    pass  # Replace with your actual code
            else:
                raise e

        # Sleep for remaining time to complete 60 seconds
        elapsed_time = time.time() - start_time
        time.sleep(max(60 - elapsed_time, 0))
