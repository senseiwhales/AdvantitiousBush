import numpy as np
import pandas as pd
import os
import time
import requests
import logging
import pickle  # Import pickle module for model serialization
import alpaca_trade_api as tradeapi
import datetime
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

CandleNumber = 1
train_models_flag = False  # Set to True to train models, False to load existing models

API_KEY = 'PKZYTDU16C4GW63TOV68'
API_SECRET = 'si6tzwHML9ZS2BLd0IktHkC2K6KkaZdOAACn0JhR'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('trades.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, symbol, api):
        self.symbol = symbol
        self.api = api
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.models = [SVR(kernel='rbf')]  # Initialize SVM model with radial basis function kernel
        self.current_position = 'flat'  # Initialize current_position attribute
        self.set_random_seed()  # Set random seed
        self.last_trained_time = None
        self.check_initial_position()  # Check initial position
        if train_models_flag:
            self.train_models()  # Train the models at initialization
        else:
            self.load_trained_models()  # Load existing models

    def check_initial_position(self):
        try:
            position = self.alpaca.get_position(self.symbol)
            if position is not None:
                self.current_position = 'long' if float(position.qty) > 0 else 'flat'
            else:
                self.current_position = 'flat'
        except tradeapi.rest.APIError as e:
            if e.status_code == 404:
                self.current_position = 'flat'  # Set position to flat if it doesn't exist
            else:
                logger.error(f"Failed to get position for {self.symbol}: {str(e)}")

    def buy_shares(self, symbol, prediction_amount):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            current_vwap = self.get_current_vwap_from_coingecko()
            if current_vwap is not None:
                num_shares = cash_available / current_vwap
                num_shares *= prediction_amount
            else:
                num_shares = 0

            self.alpaca.submit_order(
                symbol=symbol,
                qty=num_shares,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            self.current_position = 'long'
            logger.info("Bought %s shares of %s", num_shares, symbol)
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
                self.current_position = 'flat'
                logger.info("Sold all shares of %s", symbol)
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
        interval_minutes = 30  # Request 1-minute candles
        num_candles = 100  # Number of candles to fetch

        start_time = now - datetime.timedelta(minutes=interval_minutes * num_candles)
        since = start_time.isoformat() + 'Z'
        till = now.isoformat() + 'Z'

        query = """
        query {
            ethereum(network: bsc) {
                dexTrades(
                    options: { desc: "timeInterval.minute" }
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
        """ % (since, till, interval_minutes * num_candles)

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


    def get_current_vwap_from_coingecko(self):
        try:
            # Fetch historical trading volume and prices from CoinGecko API
            response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=1')
            response.raise_for_status()  # Raise exception for non-200 status codes

            data = response.json()

            # Extract volumes and prices from the response
            volumes = [entry[1] for entry in data['total_volumes']]
            prices = [entry[1] for entry in data['prices']]

            # Calculate VWAP (Volume Weighted Average Price)
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
            return vwap

        except requests.RequestException as e:
            logger.error(f"Error fetching data from CoinGecko API: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return None
        finally:
            # Add a delay of 1 second between requests
            time.sleep(5)

    def on_trading_iteration(self):
        try:
            # Fetch previous 100 candles from Bitquery
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

            # Extract trade prices and volumes
            trade_prices = [trade['quotePrice'] for trade in trades]
            trade_volumes = [trade['volume'] for trade in trades]

            # Calculate VWAP for the last 100 candles
            vwap = np.average(trade_prices, weights=trade_volumes)

            # Prepare the prediction data
            # Here, you can include additional candle data features alongside VWAP
            X = np.array([[trade_volume, vwap, trade['open'], trade['close'], trade['high'], trade['low'], CandleNumber] for trade, trade_volume in zip(trades, trade_volumes)])
            
            if len(X) == 0:
                logger.warning("Empty feature array X.")
                return

            # Make predictions using the trained models
            future_price_predictions = []
            for model in self.models:
                future_price_prediction = model.predict(X)
                future_price_predictions.append(future_price_prediction)

            future_price_predictions = np.array(future_price_predictions)

            # Use the average of predictions as the signal
            averaged_predictions = np.mean(future_price_predictions)

            # Determine the action based on the prediction and current position
            current_vwap = self.get_current_vwap_from_coingecko()
            if averaged_predictions > current_vwap:
                action = 'Buy'
            elif averaged_predictions < current_vwap:
                action = 'Sell'
            else:
                action = 'Hold'

            # Determine the side based on the action
            side = 'Buy' if action == 'Buy' else 'Sell'

            # Print out the information
            print("Candle", CandleNumber)
            print("  Current Price:", current_vwap)
            print("  Predicted Price:", averaged_predictions)
            print("  Side:", side)
            print("  Action Taken:", 'Yes' if action != 'Hold' else 'No')

            # Trading logic based on prediction
            if action == 'Buy':
                self.buy_shares(self.symbol, 0.3)  # Buy 10% of available cash
                logger.info("Predicted price (%f) is higher than Current Price (%f). Buying.", averaged_predictions, current_vwap)
            elif action == 'Sell':
                self.sell_all_shares(self.symbol)
                logger.info("Predicted price (%f) is lower than Current Price (%f). Selling.", averaged_predictions, current_vwap)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def train_models(self):
        # Load historical data from the CSV file
        X, y = self.load_data('crypto.csv')

        if X is None or y is None:
            logger.error("Failed to load data.")
            return

        if len(X) == 0 or len(y) == 0:
            logger.warning("Empty data.")
            return

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Calculate weights based on risk
        current_vwap = self.get_current_vwap_from_coingecko()
        weights_train = np.abs(y_train - current_vwap)
        max_weight = np.max(weights_train)
        weights_train = max_weight / weights_train

        # Train the models with weighted samples
        for model in self.models:
            model.fit(X_train, y_train, sample_weight=weights_train)

        # Evaluate the models on the test set
        for i, model in enumerate(self.models):
            score = model.score(X_test, y_test)
            logger.info("Model %d Test Score: %.2f", i + 1, score)

        logger.info("Training finished.")

        # Save the trained models to disk
        self.save_trained_models()

    def save_trained_models(self):
        try:
            with open('trained_models.pkl', 'wb') as f:
                pickle.dump(self.models, f)
            logger.info("Trained models saved to disk.")
        except Exception as e:
            logger.error(f"Failed to save trained models: {str(e)}")

    def load_trained_models(self):
        try:
            with open('trained_models.pkl', 'rb') as f:
                self.models = pickle.load(f)
            logger.info("Trained models loaded from disk.")
        except FileNotFoundError:
            logger.warning("No trained models found on disk. Models will be trained from scratch.")
            self.train_models()
        except Exception as e:
            logger.error(f"Failed to load trained models: {str(e)}")

    def set_random_seed(self):
        seed_value = 123  # You can change this seed value
        np.random.seed(seed_value)

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)

    # Loop to run every 30 minutes
    while True:
        ml_trader.on_trading_iteration()
        CandleNumber += 1
        time.sleep(5)  # Sleep for 30 minutes
