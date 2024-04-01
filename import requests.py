import requests

def get_bitquery_current_candle():
    base_url = 'https://graphql.bitquery.io/'
    api_key = 'BQYiC5GWXxGq6xj1umax4GRKkUyaLc64'  # Replace 'YOUR_API_KEY' with your actual Bitquery API key
    
    query = """
    query {
      ethereum(network: bsc) {
        dexTrades(
          baseCurrency: {is: "0x2170ed0880ac9a755fd29b2688956bd959f933f8"}
          quoteCurrency: {is: "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"}
          exchangeName: {is: "Pancake"}
          options: {desc: ["timeInterval.minute"]}
          time: {since: "2024-03-31T12:00:00", till: "2024-04-01T12:00:00"}
        ) {
          timeInterval {
            minute(count: 30)
          }
          baseCurrency {
            symbol
          }
          quoteCurrency {
            symbol
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
    """

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
    }

    try:
        # Print the requested query
        print("Requested query:", query)
        
        response = requests.post(base_url, json={'query': query}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data'] and 'ethereum' in data['data'] and data['data']['ethereum'] and 'dexTrades' in data['data']['ethereum'] and data['data']['ethereum']['dexTrades']:
                return data['data']['ethereum']['dexTrades'][0]  # Return the first (and only) candlestick in the list
            else:
                return "No data found for the specified query"
        else:
            return f"Failed to fetch data. Status code: {response.status_code}"
    except requests.RequestException as e:
        return f"Error fetching data: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Example usage
current_candle = get_bitquery_current_candle()
print(f"Current candle data: {current_candle}")
