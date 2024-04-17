import numpy as np
import pandas as pd
import os
import time
import requests
import logging
import alpaca_trade_api as tradeapi
import datetime
from sklearn.tree import DecisionTreeRegressor

API_KEY = 'PKYM7P7LWL9V2WLADG7P'
API_SECRET = '7MZVax8cgg1wTzoUUKfocOPTyNgJrtOcmNYqIvka'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, symbol, api):
        self.symbol = symbol
        self.api = api
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.models = [DecisionTreeRegressor(max_depth=100, max_features=30) for _ in range(10)]
        self.train_models()  # Train the models at initialization

    def buy_shares(self, symbol, prediction_amount):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            if self.models[0].current_price is not None:
                num_shares = cash_available / self.models[0].current_price
                num_shares *= prediction_amount
            else:
                num_shares = 0

            self.alpaca.submit_order(
                symbol=symbol,
                qty=num_shares,
                side='buy',
                type='market',
            )
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
            else:
                logger.info(f"No position found for {symbol}.")
        except tradeapi.rest.APIError as e:
            logger.error(f"Failed to sell all shares of {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

    def load_data(self, filename):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)
            data = pd.read_csv(file_path)
            data.columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']

            # Assuming the target variable is the closing price (can be modified based on your data)
            y = data['c']  # Closing price represents future price

            X = data.values[:, :-1]  # All columns except closing price for features
            return X, y
        except FileNotFoundError:
            logger.error(f"Data file '{filename}' not found.")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading data: {str(e)}")
            return None, None

    def get_bitquery_current_candle(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        now = datetime.datetime.utcnow()
        interval_minutes = 30

        start_time = now - datetime.timedelta(minutes=interval_minutes * 30)
        since = start_time.isoformat() + 'Z'
        till = now.isoformat() + 'Z'

        query = """
        query {
            ethereum(network: bsc) {
                dexTrades(
                    options: { limit: 30, desc: "timeInterval.minute" }
                    date: { since: "%s", till: "%s" }
                    exchangeName: { in: ["Pancake"] }
                    baseCurrency: {is: "0x2170ed0880ac9a755fd29b2688956bd959f933f8"}
                    quoteCurrency: {is: "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"}
                ) {
                    timeInterval {
                        minute(count: %d)
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
        """ % (since, till, interval_minutes)

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
                    return {}

                return {'ethereum': {'dexTrades': dex_trades}}
            else:
                logger.error("Invalid query result format")
                logger.error(data)  # Log the response for further inspection
                return {}  # Return an empty dictionary when no trade volumes are found

        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return {}

    def on_trading_iteration(self, iteration):
        try:
            logger.info(f"Current trading iteration: {iteration}")
            query_result = self.get_bitquery_current_candle()

            if query_result is None:
                logger.error("No query result obtained from API.")
                return

            if 'ethereum' not in query_result or 'dexTrades' not in query_result['ethereum']:
                logger.error("Invalid query result format")
                return

            trades = query_result['ethereum']['dexTrades']

            if not trades:
                logger.warning("No trades found in the queried data.")
                return

            trade_prices = []
            trade_volumes = []

            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])

            if not trade_volumes:
                logger.warning("No trade volumes found.")
                return

            vwap = np.average(trade_prices, weights=trade_volumes)

            for model in self.models:
                model.current_price = vwap

            if len(trade_volumes) < 4:
                logger.warning("Insufficient trade volumes for prediction.")
                return

            # Ensure the prediction data has the correct number of features
            X = np.array([[trade_volume, vwap, 0, 0, 0, 0, 0] for trade_volume in trade_volumes])
            if len(X) == 0:
                logger.warning("Empty feature array X.")
                return

            # Make predictions using the trained models
            future_price_predictions = []
            for model in self.models:
                future_price_prediction = model.predict(X)
                future_price_predictions.append(future_price_prediction)

            future_price_predictions = np.array(future_price_predictions)
            averaged_predictions = np.mean(future_price_predictions, axis=0)

            # Retrain the models after prediction
            X_train, y_train = self.load_data('crypto.csv')
            if X_train is not None and y_train is not None:
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                for model in self.models:
                    model.fit(X_train, y_train)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def train_models(self):
        X_train, y_train = self.load_data('crypto.csv')
        if X_train is not None and y_train is not None:
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]
            for model in self.models:
                model.fit(X_train, y_train)
            logger.info("Training finished.")

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)
    iteration = 1

    while iteration <= 3:
        ml_trader.on_trading_iteration(iteration)
        time.sleep(1)
        iteration += 1

    # Load data for prediction
    X, _ = ml_trader.load_data('crypto.csv')

    if X is not None:
        # Calculate and print the result for the next 30-minute candle after 10 iterations
        average_prediction = np.mean([model.predict(X) for model in ml_trader.models], axis=0)
        logger.info("Average Prediction for the next 30-minute candle after 3 iterations: %s", average_prediction[0])

        # Log the current price of the coin
        logger.info("Current price of the coin: %s", ml_trader.models[0].current_price)
    else:
        logger.error("Failed to load data for prediction.")
