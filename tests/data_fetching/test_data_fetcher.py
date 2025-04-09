from utils.data_fetcher import StockDataFetcher
import pandas as pd

def test_data_fetcher():
    """Test fetching stock and news data for TCS"""
    print("\nTesting Data Fetcher")
    
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # First fetch full historical stock data
    print("\nFetching full historical stock data...")
    stock_data = fetcher.fetch_stock_data(symbol='TCS.NS', period='max')
    
    if stock_data is None or len(stock_data) == 0:
        print("Error: Failed to fetch stock data")
        return
        
    print(f"\nStock Data Summary:")
    print(f"Total days of data: {len(stock_data)}")
    print(f"Date range: from {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    print("\nFirst few rows of stock data:")
    print(stock_data.head())
    print("\nLast few rows of stock data:")
    print(stock_data.tail())
    
    # Verify saved data
    print("\nVerifying saved stock data...")
    saved_stock = pd.read_csv(f'data/TCS.NS_stock_data.csv')
    print(f"Saved stock data rows: {len(saved_stock)}")
    print(f"Saved date range: from {saved_stock['Date'].min()} to {saved_stock['Date'].max()}")
    
    # Now fetch news and combine data
    print("\nFetching and combining news data...")
    combined_data = fetcher.get_combined_data(
        symbol='TCS.NS',
        company_name='Tata Consultancy Services',
        days=30
    )
    
    if combined_data is not None:
        print(f"\nCombined Data Summary:")
        print(f"Total data points: {len(combined_data)}")
        print(f"Columns available: {', '.join(combined_data.columns)}")
        print("\nFirst few rows of combined data:")
        print(combined_data.head())
        print("\nLast few rows of combined data:")
        print(combined_data.tail())
        
        # Verify saved combined data
        print("\nVerifying saved combined data...")
        saved_combined = pd.read_csv(f'data/TCS.NS_combined_data.csv')
        print(f"Saved combined data rows: {len(saved_combined)}")
    else:
        print("\nError: Failed to fetch combined data")

if __name__ == "__main__":
    test_data_fetcher() 