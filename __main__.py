# advantitious_bush/__main__.py

from decision_tree import AdvantitiousBush
from decision_tree import load_stock_data

def main():
    # Example usage
    strategy = AdvantitiousBush()
    data = load_stock_data()  # Load data from user-selected CSV file
    if data is not None:
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['Close'].values
        strategy.fit(X, y)
        predictions = strategy.predict(X)
        print("Predictions:", predictions)
    else:
        print("Failed to load data. Exiting.")
        return

if __name__ == "__main__":
    main()
