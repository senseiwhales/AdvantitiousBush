import requests
import datetime as dt

def get_bitquery_current_candle(symbol, interval='1h'):
    base_url = 'https://graphql.bitquery.io/'
    api_key = 'ory_at_w1pxPrnfJ0_73fkDGw5pTuUYzvkRkTNJ35Ve4olPxDI.mslrLWAnjZwBQX7b0AFK8yPlp08ijf7XlyOEpQ4DtjU'  # Replace 'YOUR_API_KEY' with your actual Bitquery API key
    
    # Set the start and end times for the current candle
    end_time = dt.datetime.utcnow().replace(second=0, microsecond=0)  # Round down to the nearest minute (in UTC)
    start_time = end_time - dt.timedelta(hours=1)  # 1 hour interval
    
    query = """
query {
  ethereum(network: bsc) {
    dexTrades(
      baseCurrency: {is: "%s"}
      quoteCurrency: {is: "USD"}
      interval: %s
      exchangeName: {is: "Pancake"}
      options: {desc: ["timeInterval.minute"]}
      time: {since: "%s", till: "%s"}
    ) {
      timeInterval {
        minute(count: 1)
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
      trades: count
    }
  }
}
""" % (symbol.upper(), interval, start_time.isoformat(), end_time.isoformat())

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
    }
    
    try:
        # Print the URL script
        print("Requested query:", query)
        
        response = requests.post(base_url, json={'query': query}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data'] and 'ethereum' in data['data'] and data['data']['ethereum'] and 'dexTrades' in data['data']['ethereum'] and data['data']['ethereum']['dexTrades']:
                return data['data']['ethereum']['dexTrades'][0]  # Return the first (and only) candlestick in the list
            else:
                return f"No data found for symbol {symbol} and interval {interval}"
        else:
            return f"Failed to fetch data. Status code: {response.status_code}"
    except requests.RequestException as e:
        return f"Error fetching data: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Example usage
symbol = 'ETH'  # Replace with the cryptocurrency symbol you want to fetch data for
interval = '1m'   # Replace with the desired interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')

current_candle = get_bitquery_current_candle(symbol, interval)
print(f"Current candle data for {symbol} ({interval}): {current_candle}")
