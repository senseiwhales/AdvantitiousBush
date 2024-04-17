import numpy as np
import requests
import logging
import alpaca_trade_api as tradeapi
import datetime
import pandas as pd
from sklearn.svm import SVR

CandleNumber = 1

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
        self.set_random_seed()  # Add this line here
        self.train_models()  # Train the models at initialization

    def train_models(self):
        # Load the data
        X, y = self.load_data("crypto.csv")

        if X is None or y is None:
            logger.error("Failed to load data. Training models aborted.")
            return

        # Split the data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train each model
        for model in self.models:
            try:
                logger.info(f"Training {model.__class__.__name__}...")
                model.fit(X_train, y_train)
                logger.info(f"{model.__class__.__name__} trained successfully.")
            except Exception as e:
                logger.error(f"Error training {model.__class__.__name__}: {str(e)}")

    def get_bitquery_last_candle(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        query = """
        {
            ethereum(network: bsc) {
                dexTrades(
                    options: {limit: 1, desc: "timeInterval.minute"}
                    date: {since: "2021-01-01"}
                    exchangeName: {is: "Pancake v2"}
                    baseCurrency: {is: "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"}
                    quoteCurrency: {is: "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82"}
                ) {
                    timeInterval {
                        minute(count: 1)
                    }
                    trades: count
                    open: minimum(of: block, get: quote_price)
                    close: maximum(of: block, get: quote_price)
                    high: quotePrice(calculate: maximum)
                    low: quotePrice(calculate: minimum)
                    volume: quoteAmount
                    baseCurrency {
                        symbol
                    }
                    quoteCurrency {
                        symbol
                    }
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
            response.raise_for_status()

            data = response.json()

            if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                dex_trades = data['data']['ethereum']['dexTrades']

                if not dex_trades:
                    logger.warning("No trades found in the queried data.")
                    return None

                return dex_trades[0]  # Return the latest candle data

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

    def set_random_seed(self):
        np.random.seed(123)  # Set the random seed to a fixed value

    def buy_shares(self, symbol, prediction_amount):
        try:
            account = self.alpaca.get_account()
            cash_available = float(account.cash)
            if self.models[0].current_price is not None:
                num_shares = cash_available / self.models[0].current_price
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
            data = pd.read_csv(filename)
            data.columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']

            y = data['c']
            X = data.values[:, :-1]
            return X, y
        except FileNotFoundError:
            logger.error(f"Data file '{filename}' not found.")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading data: {str(e)}")
            return None, None

    def fetch_past_candle_data(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        query = """
        {
            ethereum(network: bsc) {
                dexTrades(
                    options: {limit: 1000, asc: "timeInterval.minute"}
                    date: {since: "2021-01-01"}
                    exchangeName: {is: "Pancake v2"}
                    baseCurrency: {is: "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"}
                    quoteCurrency: {is: "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82"}
                ) {
                    timeInterval {
                        minute(count: 5)
                    }
                    trades: count
                    open: minimum(of: block, get: quote_price)
                    close: maximum(of: block, get: quote_price)
                    high: quotePrice(calculate: maximum)
                    low: quotePrice(calculate: minimum)
                    volume: quoteAmount
                    baseCurrency {
                        symbol
                    }
                    quoteCurrency {
                        symbol
                    }
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
            response.raise_for_status()

            data = response.json()

            if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                dex_trades = data['data']['ethereum']['dexTrades']

                if not dex_trades:
                    logger.warning("No trades found in the queried data.")
                    return pd.DataFrame()

                df = pd.DataFrame(dex_trades)
                return df[['v', 'vw', 'o', 'c', 'h', 'l', 'timeInterval', 'n']]

            else:
                logger.error("Invalid query result format")
                logger.error(data)
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return pd.DataFrame()


        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return {}

    def predict(self, model, data):
        try:
            # Check if required columns are present in the DataFrame
            required_columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']
            if not all(column in data.columns for column in required_columns):
                raise ValueError(f"One or more required columns {required_columns} not found in the DataFrame.")
            
            # Extract relevant columns and convert to NumPy array
            input_data = data[required_columns].values
            logging.info(f"Input data: {input_data}")
            
            # Make prediction
            prediction = model.predict(input_data)
            
            logging.info(f"Output prediction: {prediction}")
            return prediction
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    
    def fetch_past_candle_data(self):
        base_url = 'https://graphql.bitquery.io/'
        api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'

        query = """
        {
            ethereum(network: bsc) {
                dexTrades(
                    options: {limit: 1000, asc: "timeInterval.minute"}
                    date: {since: "2021-01-01"}
                    exchangeName: {is: "Pancake v2"}
                    baseCurrency: {is: "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"}
                    quoteCurrency: {is: "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82"}
                ) {
                    time: timeInterval {
                        time
                    }
                    trades: count
                    open: minimum(of: block, get: quote_price)
                    close: maximum(of: block, get: quote_price)
                    high: quotePrice(calculate: maximum)
                    low: quotePrice(calculate: minimum)
                    volume: quoteAmount
                    baseCurrency {
                        symbol
                    }
                    quoteCurrency {
                        symbol
                    }
                }
            }
        }

        """

        
        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }

        try:
            
            print("GraphQL Query:")
            print(query)

            response = requests.post(base_url, json={'query': query}, headers=headers)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and 'ethereum' in data['data'] and 'dexTrades' in data['data']['ethereum']:
                dex_trades = data['data']['ethereum']['dexTrades']

                if not dex_trades:
                    logger.warning("No trades found in the queried data.")
                    return pd.DataFrame()

                df = pd.DataFrame(dex_trades)
                return df[['v', 'vw', 'o', 'c', 'h', 'l', 'timeInterval', 'n', 'time']]

            else:
                logger.error("Invalid query result format")
                logger.error(data)
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return pd.DataFrame()

    def on_trading_iteration(self, model):
        past_candle_data = self.fetch_past_candle_data()

        prediction = self.predict(model, past_candle_data)

        try:
            self.train_models()
            query_result = self.get_bitquery_last_candle()

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

            trade_prices = []
            trade_volumes = []

            for trade in trades:
                trade_prices.append(trade['quotePrice'])
                trade_volumes.append(trade['volume'])

            if not trade_volumes:
                logger.warning("No trade volumes found.")
                return

            vwap = np.average(trade_prices, weights=trade_volumes)

            for model in self.models:
                model.current_price = vwap

            if len(trade_volumes) < 4:
                logger.warning("Insufficient trade volumes for prediction.")
                return

            X = self.construct_feature_array(trade_prices, trade_volumes)

            if len(X) == 0:
                logger.warning("Empty feature array X.")
                return

            future_price_predictions = []
            for model in self.models:
                future_price_prediction = model.predict(X)
                future_price_predictions.append(future_price_prediction)

            future_price_predictions = np.array(future_price_predictions)

            averaged_predictions = np.mean(future_price_predictions)

            if averaged_predictions > vwap and self.current_position != 'long':
                action = 'Buy'
            elif averaged_predictions < vwap and self.current_position == 'long':
                action = 'Sell'
            else:
                action = 'Hold'

            side = 'Buy' if action == 'Buy' else 'Sell'

            print("Candle", CandleNumber)
            print("  Current Price:", vwap)
            print("  Predicted Price:", averaged_predictions)
            print("  Side:", side)
            print("  Action Taken:", 'Yes' if action != 'Hold' else 'No')

            if averaged_predictions > vwap and self.current_position != 'long':
                self.buy_shares(self.symbol, 0.1)
                logger.info("Predicted price (%f) is higher than VWAP (%f). Buying.", averaged_predictions, vwap)
            elif averaged_predictions < vwap and self.current_position == 'long':
                self.sell_all_shares(self.symbol)
                logger.info("Predicted price (%f) is lower than VWAP (%f). Selling.", averaged_predictions, vwap)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def construct_feature_array(self, trade_prices, trade_volumes):
        try:
            feature_array = []

            for price, volume in zip(trade_prices, trade_volumes):
                feature_array.append([price, volume, 0, 0, 0, 0, 0])

            if not feature_array:
                logger.warning("Empty feature array constructed.")
            else:
                logger.info("Feature array constructed successfully.")

            return np.array(feature_array)

        except Exception as e:
            logger.error(f"Error constructing feature array: {str(e)}")
            return np.array([])


    def get_past_candles(self, num_candles):
        return []

if __name__ == "__main__":
    symbol = 'ETHUSD'
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets', api_version='v2')

    ml_trader = MLTrader(symbol='ETHUSD', api=alpaca_api)

    model_index = 0
    ml_trader.on_trading_iteration(ml_trader.models[model_index])
