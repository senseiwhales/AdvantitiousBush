import numpy as np
import pandas as pd
import os
import logging
import pickle  # Import the pickle module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        self.save_model()  # Save the trained model after fitting

    def save_model(self, filename='decision_tree_model.pkl'):
        current_dir = os.path.dirname(__file__)
        filepath = os.path.join(current_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.tree, f)

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
    def __init__(self, symbol: str = "MAT"):
        self.symbol = symbol
        self.total_profit_percentage = 0
        self.model = DecisionTree(max_depth=10, max_features=3)  # Initialize DecisionTree model

    def on_trading_iteration(self):
        current_dir = os.path.dirname(__file__)
        data = load_stock_data(os.path.join(current_dir, f'{self.symbol}.csv'))
        if data is None:
            return
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['Close'].values
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        current_prices = data['Close'].values
        next_day_prices = np.roll(current_prices, -1)
        next_day_prices[-1] = current_prices[-1]
        profit_percentage = self.calculate_profit_percentage(predictions, current_prices, next_day_prices)
        self.total_profit_percentage += profit_percentage
        logger.info(f"Trade Profit: {profit_percentage:.2f}%, Total Profit: {self.total_profit_percentage:.2f}%")

    def calculate_profit_percentage(self, predictions, current_prices, next_day_prices):
        cash = 0  # Initially no position
        stocks = 0
        initial_cash = current_prices[0]  # Initial investment
        for i in range(len(predictions)):
            if predictions[i] > current_prices[i]:  # Predicted price is higher, buy
                if cash > 0:  # Ensure enough cash to buy
                    stocks += cash / current_prices[i]  # Buy as many stocks as possible
                    cash = 0  # Update cash
            elif predictions[i] < current_prices[i]:  # Predicted price is lower, sell
                if stocks > 0:  # Ensure holding some stocks to sell
                    cash += stocks * current_prices[i]  # Sell all stocks
                    stocks = 0  # Update stocks
        final_value = cash + stocks * next_day_prices[-1]  # Calculate final portfolio value
        profit_percentage = ((final_value - initial_cash) / initial_cash) * 100  # Calculate profit percentage
        return profit_percentage

def load_stock_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data from file {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    strategy = MLTrader(symbol='MAT')
    strategy.on_trading_iteration()
    print("Total profit (percentage):", strategy.total_profit_percentage)
