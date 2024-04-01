import numpy as np
import pandas as pd
import os
import time
import threading
import requests
import logging
import alpaca_trade_api as tradeapi
import datetime

API_KEY = 'PKYM7P7LWL9V2WLADG7P'
API_SECRET = '7MZVax8cgg1wTzoUUKfocOPTyNgJrtOcmNYqIvka'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None
        self.current_price = None
        self.total_cash = 0.0  # Initialize total_cash
        self.api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')  # Initialize Alpaca API

    def fit(self, X, y):
        if self.max_features is not None:
            self.max_features = min(self.max_features, X.shape[1])
        self.tree = self._build_tree(X, y, depth=0)
    
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

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']

        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])


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

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']

        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

import requests
import logging
import time
import threading

API_KEY = 'PKYM7P7LWL9V2WLADG7P'
API_SECRET = '7MZVax8cgg1wTzoUUKfocOPTyNgJrtOcmNYqIvka'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, symbol: str = "ETH/USD", starting_cash: float = 10000):
        self.symbol = symbol
        self.total_cash = starting_cash
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.trades = []
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.previous_prediction = None  
        self.current_price = None

    def buy_shares(self, symbol):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            if self.model.current_price is not None:
                num_shares = cash_available / self.model.current_price
            else:
                num_shares = 0

            self.alpaca.submit_order(
                symbol=symbol,
                qty=num_shares,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        except tradeapi.rest.APIError as e:
            logger.error(f"Failed to buy shares of {symbol}: {str(e)}")

    def get_bitquery_current_candle(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        query = """
        query {
        ethereum(network: bsc) {
            dexTrades(
            baseCurrency: {is: "0x2170ed0880ac9a755fd29b2688956bd959f933f8"}
            quoteCurrency: {is: "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"}
            exchangeName: {is: "Pancake"}
            options: {desc: ["timeInterval.minute"]}
            time: {since: "2024-03-31T12:00:00", till: "2024-04-01T12:00:00"}
            ) {
            timeInterval {
                minute(count: 30)
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
        """

        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }

        try:
            response = requests.post(base_url, json={'query': query}, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                    dex_trades = data['data']['ethereum']['dexTrades']
                    formatted_trades = []
                    for trade in dex_trades:
                        formatted_trade = {
                            'timeInterval': {'minute': trade['timeInterval']['minute']},
                            'count': trade['count'],
                            'quotePrice': trade['quotePrice'],
                            'high': trade['high'],
                            'low': trade['low'],
                            'open': trade['open'],
                            'close': trade['close'],
                            'volume': trade['volume']
                        }
                        formatted_trades.append(formatted_trade)
                    return {'ethereum': {'dexTrades': formatted_trades}}
                else:
                    return "Invalid query result format"
            else:
                return f"Failed to fetch data. Status code: {response.status_code}"
        except requests.RequestException as e:
            return f"Error fetching data: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"





    def on_trading_iteration(self):
        try:
            # Get data from the Bitquery query
            query_result = self.get_bitquery_current_candle()
            
            # Check if the query result is valid
            if 'ethereum' not in query_result or 'dexTrades' not in query_result['ethereum']:
                logger.error("Invalid query result format")
                return

            # Extract relevant data for VWAP calculation
            trades = query_result['ethereum']['dexTrades']

            # Initialize lists to store trade prices and volumes
            trade_prices = []
            trade_volumes = []

            # Extract trade prices and volumes
            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])

            # Calculate VWAP manually
            vwap = np.average(trade_prices, weights=trade_volumes)

            # Log VWAP
            logger.info(f"VWAP for {self.symbol}: {vwap}")

            # Example of extracting required features from the query result
            X = np.array([[trade_volume, vwap] for trade_volume in trade_volumes])
            
            # Make predictions
            prediction = self.model.predict(X)[0]

            # Log prediction
            logger.info(f"Current prediction for {self.symbol}: {prediction}")

            # Print current price as well
            logger.info(f"Current price for {self.symbol}: {vwap}")

            # Calculate the number of shares to buy or sell based on the VWAP and available cash
            shares_to_trade = self.total_cash / vwap

            # Make trading decision based on prediction
            if prediction > vwap:
                # Buy logic
                self.buy_shares(self.symbol)
            elif prediction < vwap:
                # Sell logic
                self.sell(vwap, prediction)
            else:
                logger.info("Holding position.")
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")



if __name__ == "__main__":
    starting_cash = 10000  
    symbol = 'BTCUSDC'
    api_key = 'FDPXMTIEJIP2N4TY'  
    
    # Assuming you have training data X_train and y_train
    # Example training data
    X_train = np.array([[100], [110], [120], [130], [140]])  # Example feature: current price
    y_train = np.array([0, 1, 1, 0, 1])  # Example labels: 0 for sell, 1 for buy

    # Instantiate MLTrader object
    ml_trader = MLTrader(symbol='BTC/USDC', starting_cash=starting_cash)

    # Train the model
    ml_trader.model.fit(X_train, y_train)

    # Start trading iterations
    while True:
        ml_trader.on_trading_iteration()
        time.sleep(1)