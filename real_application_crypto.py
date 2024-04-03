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
        self.total_cash = 0
        self.api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')

    def fit(self, X, y):
        if self.max_features is not None:
            self.max_features = min(self.max_features, X.shape[1])
        self.tree = self._build_tree(X, y, depth=0)
    
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

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']

        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

class MLTrader:
    def __init__(self, symbol: str = "ETHUSD", starting_cash: float = 10000):
        self.symbol = symbol
        self.total_cash = starting_cash
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.trades = []
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.previous_prediction = None  
        self.current_price = None
        self.train_model()

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

    def load_data(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        data = pd.read_csv(file_path)
        data.columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']
        data = data[['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']]
        X = data.values
        y = np.zeros(X.shape[0])
        return X, y

    def get_bitquery_current_candle(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'
        
        now = datetime.datetime.utcnow()
        start_time = now - datetime.timedelta(minutes=30)
        since = start_time.isoformat() + 'Z'
        till = now.isoformat() + 'Z'

        query = """
        query {{
            ethereum(network: bsc) {{
                dexTrades(
                    baseCurrency: {{is: "0x2170ed0880ac9a755fd29b2688956bd959f933f8"}},
                    quoteCurrency: {{is: "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"}},
                    exchangeName: {{is: "Pancake"}},
                    options: {{desc: ["timeInterval.minute"]}},
                    time: {{since: "{}", till: "{}"}}
                ) {{
                    timeInterval {{
                        minute(count: 30)
                    }}
                    count
                    quotePrice
                    high: quotePrice(calculate: maximum)
                    low: quotePrice(calculate: minimum)
                    open: minimum(of: block, get: quote_price)
                    close: maximum(of: block, get: quote_price)
                    volume: quoteAmount
                }}
            }}
        }}
        """.format(since, till)

        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }

        try:
            response = requests.post(base_url, json={'query': query}, headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                dex_trades = data['data']['ethereum']['dexTrades']
                if not dex_trades:
                    logger.error("No trade volumes found.")
                    return None
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
                logger.error("Invalid query result format")
                logger.error(data)
                return None
        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return None

    def on_trading_iteration(self):
        try:
            query_result = self.get_bitquery_current_candle()

            if 'ethereum' not in query_result or 'dexTrades' not in query_result['ethereum']:
                logger.error("Invalid query result format")
                return

            trades = query_result['ethereum']['dexTrades']

            trade_prices = []
            trade_volumes = []

            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])

            if not trade_volumes:
                logger.error("No trade volumes found.")
                return

            vwap = np.average(trade_prices, weights=trade_volumes)

            self.model.current_price = vwap

            X = np.array([[trade_volume, vwap] for trade_volume in trade_volumes])

            future_price_prediction = self.model.predict(X[-1])

            prediction_mean = np.mean(future_price_prediction)
            if prediction_mean > vwap and self.previous_prediction is not None and self.previous_prediction <= vwap:
                prediction_amount = 1.5
                shares_to_buy = self.total_cash / vwap * prediction_amount
                self.buy_shares(self.symbol, shares_to_buy)
                logger.info("New position: Buying shares.")
            elif prediction_mean <= vwap and self.previous_prediction is not None and self.previous_prediction > vwap:
                logger.info("Holding position.")

            self.previous_prediction = prediction_mean

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def train_model(self):
        X_train, y_train = self.load_data('crypto.csv')
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        self.model.fit(X_train, y_train)
        logger.info("Training finished.")

if __name__ == "__main__":
    starting_cash = 10000
    symbol = 'ETHUSD'

    ml_trader = MLTrader(symbol='ETHUSD', starting_cash=starting_cash)

    while True:
        ml_trader.on_trading_iteration()
        time.sleep(1)
