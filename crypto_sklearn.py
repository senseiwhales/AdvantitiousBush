import numpy as np
import logging
import requests
import time
import datetime
import alpaca_trade_api as tradeapi
from sklearn.svm import SVR

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
            current_price = self.get_current_price_from_bitquery()
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

    def get_current_price_from_bitquery(self):
        try:
            # Fetch current trading price from Bitquery
            candles = self.get_bitquery_candles(limit=1)

            if candles:
                return candles[0]['close']
            else:
                logger.warning("No candles found in the queried data.")
                return None
        except Exception as e:
            logger.error(f"Error fetching data from Bitquery: {str(e)}")
            return None

    def get_bitquery_candles(self, limit=1):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(minutes=limit)

        since = start_time.isoformat() + 'Z'
        till = end_time.isoformat() + 'Z'

        query = """
        query {
            ethereum(network: bsc) {
                dexTrades(
                    options: { desc: "timeInterval.minute" }
                    date: { since: "%s", till: "%s" }
                    exchangeName: { in: ["Pancake"] }
                    baseCurrency: {is: "0x2170ed0880ac9a755fd29b2688956bd959f933f8"}
                    quoteCurrency: {is: "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"}
                ) {
                    volume
                    open
                    close
                    high
                    low
                    timestamp
                    numberOfTrades
                }
            }
        }
        """ % (since, till)


        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }

        try:
            response = requests.post(base_url, json={'query': query}, headers=headers)
            response.raise_for_status()  # Raise exception for non-200 status codes

            data = response.json()

            if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                dex_trades = data['data']['ethereum']['dexTrades']

                if not dex_trades:
                    logger.warning("No candles found in the queried data.")

                return dex_trades

            else:
                logger.error("Invalid query result format")
                print(query)
                return []

        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return []

        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return []

    def on_trading_iteration(self):
        try:
            current_price = self.get_current_price_from_bitquery()

            if current_price is None:
                logger.warning("No current price found.")
                return

            print("Current Price:", current_price)

            closing_prices = [float(candle['close']) for candle in self.get_bitquery_candles(limit=30)]

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
            logger.error(f"Error in trading iteration: {e}")

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


    def set_random_seed(self):
        seed_value = 143  # You can change this seed value
        np.random.seed(seed_value)

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)

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
