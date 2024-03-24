# advantitious_bush/advantitious_bush.py

import logging
import numpy as np
import pandas as pd
from .decision_tree import DecisionTree

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvantitiousBush:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(max_depth=self.max_depth, max_features=max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))

        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)

        return np.mean(predictions, axis=1)

def load_stock_data(file_path):
    """
    Load historical stock price data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical stock price data.
    """
    try:
        # Load data from CSV file
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data from file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    strategy = AdvantitiousBush()
    data = load_stock_data('MAT.csv')  # Load data from MAT.csv file
    if data is not None:
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['Close'].values
        strategy.fit(X, y)
        predictions = strategy.predict(X)
        print("Predictions:", predictions)
    else:
        logger.error("Failed to load data. Exiting.")
        print("Sadly it done be broke")
