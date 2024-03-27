import numpy as np
import pandas as pd
import os
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

    def fit(self, X, y):
        if self.max_features is not None:
            self.max_features = min(self.max_features, X.shape[1])  # Ensure max_features does not exceed total features
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

def format_datetime(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

class MLTrader:
    def __init__(self, symbol: str = "MAT", starting_cash: float = 10000):
        self.symbol = symbol
        self.total_cash = starting_cash
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')  # Initialize Alpaca
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.trades = []

    def on_trading_iteration(self):
        data = self.load_stock_data(self.symbol)
        if data is None or len(data) == 0:  # Check if data is empty
            logger.error("Failed to load stock data or data is empty.")
            return
        logger.info("Stock data loaded successfully.")
        
        X = data.values  # Use closing prices as features
        y = np.roll(X, -1)  # Predict the next closing price

        self.model.fit(X, y)
        
        if len(X) > 0:  # Check if X is not empty
            prediction = self.model.predict(X[-1].reshape(1, -1))  # Predict next closing price
            current_price = X[-1][0]  # Current closing price

            logger.info(f"Current price of {self.symbol}: {current_price}")  # Print current price

            # Calculate the number of shares to buy or sell based on the current price and available cash
            shares_to_trade = int(self.total_cash / current_price)

            # Make trading decision based on prediction
            if prediction > current_price:
                # Buy logic
                self.buy(shares_to_trade, current_price)
            elif prediction < current_price:
                # Sell logic
                self.sell(shares_to_trade, current_price)
            else:
                logger.info("Holding position.")



    def buy(self, shares_to_trade, current_price):
        if shares_to_trade > 0:
            try:
                self.alpaca.submit_order(
                    symbol='MAT',
                    qty=shares_to_trade,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Bought {shares_to_trade} shares of {self.symbol}.")
                self.total_cash -= shares_to_trade * current_price  # Update total cash after buying
            except Exception as e:
                logger.error(f"Failed to buy {shares_to_trade} shares of {self.symbol}: {e}")

    def sell(self, shares_to_trade, current_price):
        if shares_to_trade > 0:
            try:
                self.alpaca.submit_order(
                    symbol='MAT',
                    qty=shares_to_trade,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Sold {shares_to_trade} shares of {self.symbol}.")
                self.total_cash += shares_to_trade * current_price  # Update total cash after selling
            except Exception as e:
                logger.error(f"Failed to sell {shares_to_trade} shares of {self.symbol}: {e}")

    def load_stock_data(self, symbol):
        try:
            # Fetch current 30-minute candle data from Alpaca
            current_time = datetime.datetime.now()
            start_time = current_time - datetime.timedelta(minutes=30)
            end_time = current_time
            barset = self.alpaca.get_bars(symbol, '30Min', start=format_datetime(start_time), end=format_datetime(end_time))

            # Extract the closing prices from the fetched data
            closing_prices = [bar.c for bar in barset]

            # Create a DataFrame from the closing prices
            data = pd.DataFrame(closing_prices, columns=['c'])

            return data
        except Exception as e:
            logger.error(f"Error loading data for symbol {symbol} from Alpaca: {e}")
            return None



if __name__ == "__main__":
    starting_cash = 10000  # Set your desired starting cash here
    strategy = MLTrader(symbol='MAT', starting_cash=starting_cash)
    
    # Run trading iteration indefinitely
    while True:
        strategy.on_trading_iteration()
