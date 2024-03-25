import numpy as np
import pandas as pd
import os
import logging
import pickle  # Import the pickle module
from advantitious_bush import MLTrader  # Import the MLTrader class

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Create a 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/trading_log.txt',
                    filemode='w')
logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, symbol: str = "MAT"):
        self.symbol = symbol
        self.total_profit_percentage = 0
        self.model = self.load_model()  # Load model from file or initialize a new one

    def load_model(self, filename='decision_tree_model.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            return MLTrader()  # Initialize a new MLTrader object if model file doesn't exist

    def on_trading_iteration(self):
        data = load_stock_data(f'{self.symbol}.csv')
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
