import numpy as np
import pandas as pd
import os
import time
import requests
import logging
import alpaca_trade_api as tradeapi
import datetime

API_KEY = 'PKYM7P7LWL9V2WLADG7P'
API_SECRET = '7MZVax8cgg1wTzoUUKfocOPTyNgJrtOcmNYqIvka'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, api=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None
        self.api = api

    def fit(self, X, y):
        if self.max_features is not None:
            self.max_features = min(self.max_features, X.shape[1])
        self.tree = self._build_tree(X, y, depth=0)
        self.input_shape = X.shape  # Define input_shape attribute
    
    def make_prediction(self, features):
        if self.tree is not None:
            return self.predict(np.array([features]))[0]
        else:
            raise ValueError("Model is not trained.")

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth == self.max_depth or n_labels == 1:
            leaf_value = self._compute_leaf_value(y)
            return {'leaf': True, 'value': leaf_value}
          
        feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)

        if best_feature is None:
            leaf_value = self._compute_leaf_value(y)
            return {'leaf': True, 'value': leaf_value}

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'leaf': False, 'feature': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y, feature_indices):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                gain = self._compute_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _compute_gain(self, y, left_y, right_y):
        n = len(y)
        left_n = len(left_y)
        right_n = len(right_y)

        entropy_parent = self._compute_entropy(y)
        entropy_children = (left_n / n) * self._compute_entropy(left_y) + (right_n / n) * self._compute_entropy(right_y)

        return entropy_parent - entropy_children

    def _compute_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _compute_leaf_value(self, y):
        return np.mean(y)

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']

        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

class MLTrader:
    def __init__(self, symbol, api):
        self.symbol = symbol
        self.api = api
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.train_model()  # Train the model at initialization

    def buy_shares(self, symbol, prediction_amount):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            if self.model.current_price is not None:
                num_shares = cash_available / self.model.current_price
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
        intervals = 30
        interval_minutes = 30

        start_time = now - datetime.timedelta(minutes=interval_minutes * intervals)
        since = start_time.isoformat() + 'Z'
        till = now.isoformat() + 'Z'

        query = """
        query {
            ethereum(network: bsc) {
                dexTrades(
                    options: { limit: 100, asc: "timeInterval.minute" }
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
        """ % (since, till, interval_minutes * intervals)

        logger.info("Query sent to Bitquery API: \n" + query)  # Log the query string before making the request

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


    def on_trading_iteration(self):
        try:
            query_result = self.get_bitquery_current_candle()

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

            self.model.current_price = vwap

            if len(trade_volumes) < 4:
                logger.warning("Insufficient trade volumes for prediction.")
                return

            X = np.array([[trade_volume, vwap] for trade_volume in trade_volumes])
            logger.info(f"X array size: {X.shape[0]}")

            if len(X) == 0:
                logger.warning("Empty feature array X.")
                return

            # Manual Feature Scaling
            feature_min = np.amin(X, axis=0)
            feature_max = np.amax(X, axis=0)

            # Add a small constant to avoid division by zero
            X_scaled = (X - feature_min) / (feature_max - feature_min + 1e-10)

            logger.debug(f"X_scaled shape: {X_scaled.shape}")
            logger.debug(f"Model input shape: {self.model.input_shape[1:]}")
  
            future_price_prediction = self.model.predict(X_scaled)
            
            if len(future_price_prediction) == 0:
                logger.warning("Empty prediction array.")
                return

            logger.info("Future Price Prediction:", future_price_prediction)
            prediction_mean = np.mean(future_price_prediction)
            logger.info("Prediction Mean:", prediction_mean)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def train_model(self):
        X_train, y_train = self.load_data('crypto.csv')
        if X_train is not None and y_train is not None:
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]
            self.model.fit(X_train, y_train)
            logger.info("Training finished.")

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)

    while True:
        ml_trader.on_trading_iteration()
        time.sleep(1)
