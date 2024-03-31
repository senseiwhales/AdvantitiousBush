import requests

def get_stock_price(symbol, api_key):
    if not api_key:
        return "API key is missing or invalid"

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        
        # Check if the response status code is OK
        if response.status_code == 200:
            data = response.json()
            # Check if the response contains the expected data
            if "Global Quote" in data:
                price_data = data["Global Quote"]
                # Access the price directly from the response
                current_price = price_data.get("05. price")
                if current_price:
                    return current_price
                else:
                    return f"Price not found for symbol {symbol}"
            else:
                return f"Price not found for symbol {symbol}"
        else:
            return f"Failed to fetch data. Status code: {response.status_code}"
    except requests.RequestException as e:
        return f"Error fetching data: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Example usage
symbol = 'BTCUSD'  # Replace with the stock symbol you want to fetch data for
api_key = '7B7J6XY6GEMAKMB2'  # Replace with your Alpha Vantage API key
stock_price = get_stock_price(symbol, api_key)
print(f"Current price of {symbol}: {stock_price}")
