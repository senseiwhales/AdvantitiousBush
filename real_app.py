import numpy as np
import pandas as pd
import os
import logging
import alpaca_trade_api as tradeapi
import datetime
from alpaca_trade_api.rest import TimeFrame

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

class MLTrader:
    def __init__(self, symbol: str = "MAT", starting_cash: float = 10000):
        self.symbol = symbol
        self.total_cash = starting_cash
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.trades = []
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.previous_prediction = None  # Track previous prediction
        
    def on_trading_iteration(self):
        data = self.load_data_from_csv('STOCK.csv')
        if data is None or len(data) == 0:  # Check if data is empty
            logger.error("Failed to load data or data is empty.")
            return
            
        X = data  # Use closing prices as features
        y = np.roll(X, -1)  # Predict the next closing price

        self.model.fit(X, y)
            
        if len(X) > 0:  # Check if X is not empty
            prediction = self.model.predict(X[-1].reshape(1, -1))[0]  # Predict next closing price
            current_price = X[-1][0]  # Current closing price

            if self.previous_prediction != prediction:  # Print only if prediction changes
                logger.info(f"Current prediction for {self.symbol}: {prediction}")  # Print current prediction
                print(f"Current prediction for {self.symbol}: {prediction}")  # Print current prediction
                self.previous_prediction = prediction  # Update previous prediction

            # Make trading decision based on prediction
            if prediction > current_price:
                # Buy logic
                self.buy(1, current_price, prediction)  # Only buy 1 share
            elif prediction < current_price:
                # Sell logic
                self.sell(current_price, prediction)
            else:
                logger.info("Holding position.")
        self.sell(current_price, prediction)
    def buy(self, shares_to_trade, current_price, prediction):
        try:
            account_info = self.alpaca.get_account()
            available_cash = float(account_info.buying_power)
            cost_of_purchase = shares_to_trade * current_price
            if cost_of_purchase > 0 and cost_of_purchase <= available_cash:  # Ensure sufficient cash balance for buying
                if prediction > current_price:  # Check if the prediction is higher than the current price to trigger a buy
                    order = self.alpaca.submit_order(
                        symbol=self.symbol,
                        qty=shares_to_trade,
                        side='buy',
                        type='market',
                        time_in_force='day'  # Set time_in_force to 'day'
                    )
                    logger.info(f"Bought {shares_to_trade} shares of {self.symbol} at ${current_price}.")
                    self.total_cash -= order.filled_qty * order.filled_avg_price  # Update total cash after buying
                else:
                    logger.info("Prediction is lower than current price. Not buying.")
            else:
                logger.info("Insufficient cash balance to buy.")
        except Exception as e:
            logger.error(f"Failed to buy {shares_to_trade} shares of {self.symbol}: {e}")


    def sell(self, current_price, prediction):
        try:
            positions = self.alpaca.list_positions()
            symbol_position = next((pos for pos in positions if pos.symbol == self.symbol), None)
            if symbol_position:
                qty_to_sell = float(symbol_position.qty)
                if qty_to_sell > 0 and prediction < current_price:  # Check prediction before selling
                    order = self.alpaca.submit_order(
                        symbol=self.symbol,
                        qty=qty_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"Sold {qty_to_sell} shares of {self.symbol} at ${current_price}.")
                    self.total_cash += order.filled_qty * order.filled_avg_price
                else:
                    logger.info("No shares to sell or prediction is not met.")
            else:
                logger.info("No position found for the symbol.")
        except Exception as e:
            logger.error(f"Failed to sell {self.symbol}: {e}")

    def load_data(self, symbols, timeframe):
        try:
            # Fetch current bar data for the symbol
            barset = self.alpaca.get_bars(symbols[0], timeframe, limit=1)
            if barset:
                return barset.df  # Return the bar data for the specified symbol as a DataFrame
            else:
                logger.error(f"Failed to fetch data for {symbols[0]} from Alpaca.")
                return None
        except Exception as e:
            logger.error(f"Error fetching data from Alpaca: {e}")
            return None

    def load_data_from_csv(self, filename):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
            file_path = os.path.join(script_dir, filename)  # Combine script directory with filename
            data = pd.read_csv(file_path)  # Load data from CSV file

            # Extract relevant columns for features (v, vw, o, c, h, l) from the CSV data
            X = data[['v', 'vw', 'o', 'c', 'h', 'l']].values

            return X            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            return None



if __name__ == "__main__":
    starting_cash = 10000  # Set your desired starting cash here
    strategy = MLTrader(symbol='MAT', starting_cash=starting_cash)
    
    # Run trading iteration indefinitely
    while True:
        strategy.on_trading_iteration()

