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

class MLTrader:
    def __init__(self, symbol: str = "BTC/USD", starting_cash: float = 10000):
        self.symbol = symbol
        self.total_cash = starting_cash
        self.model = DecisionTree(max_depth=10, max_features=3)
        self.trades = []
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.previous_prediction = None  
        self.current_price = None
    
    def print_info_periodically(self):
        while True:
            try:
                if self.model.current_price is not None:
                    logger.info(f"Current prediction for {self.symbol}: {self.previous_prediction}")
                    logger.info(f"Current price for {self.symbol}: {self.model.current_price}")
                else:
                    logger.info("Current price is not available.")
            except Exception as e:
                logger.error(f"Error in printing information: {e}")

            time.sleep(5)  # Sleep for 5 seconds


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

    def on_trading_iteration(self):
        try:
            # Load data from CSV file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file_path = os.path.join(dir_path, 'crypto.csv')
            data = pd.read_csv(file_path)

            if data is None or len(data) == 0:
                logger.error("Failed to load crypto data or data is empty.")
                return

            # Extract features and target variable
            X = data.drop(columns=['date']).values  # Remove date column and use the remaining columns as features
            y = data['close'].shift(-1).values  # Shift the 'close' column by 1 to get the target variable

            # Remove NaN values
            X = X[:-1]  # Remove the last row of X to match the shifted y
            y = y[:-1]  # Remove the last element of y

            # Train the decision tree model
            self.model.fit(X, y)

            # Make predictions
            prediction = self.model.predict(X[-1].reshape(1, -1))[0]  # Predict next closing price
            current_price = X[-1][0]  # Current closing price
            print(prediction)
            
            # Update previous prediction
            self.previous_prediction = prediction

            # Print prediction and current price
            logger.info(f"Current prediction for {self.symbol}: {prediction}")
            logger.info(f"Current price for {self.symbol}: {current_price}")

            # Calculate the number of shares to buy or sell based on the current price and available cash
            shares_to_trade = self.total_cash / current_price

            # Make trading decision based on prediction
            if prediction > current_price:
                # Buy logic
                self.buy_shares(self.symbol)
            elif prediction < current_price:
                # Sell logic
                self.sell(current_price, prediction)
            else:
                logger.info("Holding position.")

            # Update current price
            self.update_price('BTC/USD')
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    

    def get_crypto_data(self, symbol, api_key):
        base_url = "https://www.alphavantage.co/query"
        function = "CRYPTO_INTRADAY"  # Function to fetch intraday crypto data
        interval = "30min"  # Interval for data (adjust as needed)
        
        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "apikey": api_key
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            logger.info(f"Response from Alpha Vantage: {data}")  # Log the response

            # Check if data contains the expected key for crypto data
            if "Time Series Crypto (30min)" in data:
                crypto_data = data["Time Series Crypto (30min)"]
                # Extract the latest data point
                latest_data = next(iter(crypto_data.values()))
                # Extract relevant information (e.g., price)
                price = float(latest_data.get("4. close"))  # Adjust key based on actual data structure
                # Additional data processing as needed

                # Construct the crypto data dictionary
                crypto_data = {
                    "price": price
                    # Add other relevant data here
                }

                logger.info(f"Fetched crypto data for {symbol}: {crypto_data}")
                return crypto_data
            else:
                logger.error(f"No crypto data available for symbol {symbol}")
                return None
        except Exception as e:
            error_message = f"Error fetching crypto data: {str(e)}"
            logger.error(error_message)
            return None



    def print_info(self):
        while True:
            self.buy_shares(self.symbol)  
            print(f"Total cash: {self.total_cash}")
            time.sleep(5)  

if __name__ == "__main__":
    starting_cash = 10000  
    symbol = 'BTCUSD'
    api_key = 'FDPXMTIEJIP2N4TY'  
    
    strategy = MLTrader(symbol='BTC/USD', starting_cash=starting_cash)
    stock_data = strategy.get_crypto_data(symbol, api_key)
    print(f"Stock data for {symbol}: {stock_data}")

    # Create a thread for printing information periodically
    print_thread = threading.Thread(target=strategy.print_info_periodically)
    print_thread.daemon = True  # Set the thread as daemon so it exits when the main thread exits
    print_thread.start()

    # Add this line to keep the main thread running
    while True:
        time.sleep(1)