import numpy as np
import logging
import requests
import time
import datetime
import alpaca_trade_api as tradeapi
from sklearn.svm import SVR

CandleNumber = 1

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
        self.last_trained_time = None

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
            current_vwap = self.get_current_vwap_from_coingecko()
            if current_vwap is not None:
                num_shares = cash_available / current_vwap
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

    def get_current_vwap_from_coingecko(self):
        try:
            # Fetch historical trading volume and prices from CoinGecko API
            response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=1')
            response.raise_for_status()  # Raise exception for non-200 status codes

            data = response.json()

            # Extract volumes and prices from the response
            volumes = [entry[1] for entry in data['total_volumes']]
            prices = [entry[1] for entry in data['prices']]

            # Calculate VWAP (Volume Weighted Average Price)
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)

            # Introduce a delay of 30 seconds
            time.sleep(30)

            return vwap

        except requests.RequestException as e:
            logger.error(f"Error fetching data from CoinGecko API: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return None

    def get_bitquery_candles(self, limit=100):
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
                    timeInterval {
                        minute
                    }
                    count
                    quotePrice
                    high: quotePrice(calculate: maximum)
                    low: quotePrice(calculate: minimum)
                    open: minimum(of: block, get: quote_price)
                    close: maximum(of: block, get: quote_price)
                    volume: quoteAmount
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
                    logger.warning("No trades found in the queried data.")

                return dex_trades

            else:
                logger.error("Invalid query result format")
                logger.error(data)  # Log the response for further inspection
                return []

        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return []

        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return []


    def on_trading_iteration(self):
        try:
            current_time = datetime.datetime.now()

            # Fetch historical candle data from Bitquery
            trades = self.get_bitquery_candles(limit=1000)

            if not trades:
                logger.warning("No trades found in the queried data.")
                return

            trade_prices = []
            trade_volumes = []
            open_prices = []
            close_prices = []
            high_prices = []
            low_prices = []

            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])
                open_prices.append(float(trade['open']))  # Convert to float
                close_prices.append(float(trade['close']))  # Convert to float
                high_prices.append(float(trade['high']))  # Convert to float
                low_prices.append(float(trade['low']))  # Convert to float

            vwap = np.average(trade_prices, weights=trade_volumes)

            # Calculate differences
            open_close_diff = np.array(open_prices) - np.array(close_prices)
            high_low_diff = np.array(high_prices) - np.array(low_prices)

            # Update the training data with differences
            X_train = np.array([[trade_volume, vwap, open_close_diff[i], high_low_diff[i], trade['volume'], trade['quotePrice']] for i, trade_volume in enumerate(trade_volumes)])
            y_train = np.array(trade_prices)

            if X_train.size > 0 and y_train.size > 0:  # Check if arrays are not empty
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                for model in self.models:
                    model.fit(X_train, y_train)  # Fit the SVM model

            # Make predictions using the trained models
            X = np.array([[trade_volume, vwap, open_close_diff[i], high_low_diff[i], trade['volume'], trade['quotePrice']] for i, trade_volume in enumerate(trade_volumes)])
            if len(X) == 0:
                logger.warning("Empty feature array X.")
                return

            future_price_predictions = []
            for model in self.models:
                future_price_prediction = model.predict(X)
                future_price_predictions.append(future_price_prediction)

            future_price_predictions = np.array(future_price_predictions)
            print(future_price_predictions)
            if len(future_price_predictions) < 31:
                logger.warning("Insufficient data for predicting 30 candles from now.")
                return

            if self.current_position == 'flat':
                # Check for the best time to enter a long position
                if (future_price_predictions[:, 30] > vwap).any():
                    self.buy_shares(self.symbol, 0.3)  # Buy 30% of available cash
                    logger.info("Predicted price is higher than Current Price. Buying.")
            elif self.current_position == 'long':
                # Check for the best time to exit the long position
                if (future_price_predictions[:, 30] < vwap).any():
                    self.sell_all_shares(self.symbol)
                    logger.info("Predicted price is lower than Current Price. Selling.")

            # Print the current iteration number
            print("Current Iteration:", CandleNumber)

            # Print the prediction for 60 candles from now
            print("Prediction for 30 candles from now:", future_price_predictions[:, 30])

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def train_model(self):
        try:
            # Fetch historical candle data from Bitquery
            trades = self.get_bitquery_candles(limit=1000)

            if not trades:
                logger.warning("No trades found in the queried data for training.")
                return

            trade_prices = []
            trade_volumes = []
            open_prices = []
            close_prices = []
            high_prices = []
            low_prices = []

            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])
                open_prices.append(float(trade['open']))  # Convert to float
                close_prices.append(float(trade['close']))  # Convert to float
                high_prices.append(float(trade['high']))  # Convert to float
                low_prices.append(float(trade['low']))  # Convert to float

            vwap = np.average(trade_prices, weights=trade_volumes)

            # Calculate differences
            open_close_diff = np.array(open_prices) - np.array(close_prices)
            high_low_diff = np.array(high_prices) - np.array(low_prices)

            # Prepare the training data
            X_train = np.array([[trade_volume, vwap, open_close_diff[i], high_low_diff[i], trade['volume'], trade['quotePrice']] for i, trade_volume in enumerate(trade_volumes)])
            y_train = np.array(trade_prices)

            if X_train.size > 0 and y_train.size > 0:  # Check if arrays are not empty
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                for model in self.models:
                    model.fit(X_train, y_train)  # Fit the SVM model
                logger.info("Model trained successfully.")

        except Exception as e:
            logger.error(f"Error in training model: {e}")

    def set_random_seed(self):
        seed_value = 143  # You can change this seed value
        np.random.seed(seed_value)

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)

    # Training the model initially
    ml_trader.train_model()

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

        CandleNumber += 1

        # Sleep for remaining time to complete 60 seconds
        elapsed_time = time.time() - start_time
        time.sleep(max(60 - elapsed_time, 0))