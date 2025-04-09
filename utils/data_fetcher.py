import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

class StockDataFetcher:
    def __init__(self):
        # Load API key from .env file
        load_dotenv()
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            raise ValueError("NewsAPI key is required! Add it to .env file")
        
        # Initialize NewsAPI client
        self.newsapi = NewsApiClient(api_key=self.news_api_key)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

    def fetch_stock_data(self, symbol='TCS.NS', period='max', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"Fetching stock data for {symbol}...")
            stock = yf.Ticker(symbol)
            
            # Fetch data with adjusted prices
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Ensure all required columns are present
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Missing column {col} in stock data")
                    return None
            
            print(f"Fetched {len(df)} days of stock data")
            
            # Save to CSV with all data
            csv_path = f'data/{symbol}_stock_data.csv'
            df.to_csv(csv_path, index=False)
            print(f"Stock data saved to {csv_path}")
            
            # Verify the saved data
            saved_df = pd.read_csv(csv_path)
            print(f"Verified saved data: {len(saved_df)} rows")
            
            return df
        
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None

    def fetch_news(self, query="Tata Consultancy Services", days=30):
        """Fetch news articles from NewsAPI"""
        try:
            print(f"Fetching news for {query}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            news = self.newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt'
            )
            
            df = pd.DataFrame(news['articles'])
            
            # Save to CSV
            csv_path = f'data/{query}_news_data.csv'
            df.to_csv(csv_path, index=False)
            print(f"News data saved to {csv_path}")
            
            return df
        
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            return None

    def get_combined_data(self, symbol, company_name=None, days=30):
        """
        Fetch both stock and news data and create a combined dataset
        
        Parameters:
        - symbol: Stock symbol
        - company_name: Full company name (optional)
        - days: Number of days of historical data
        
        Returns:
        - combined_df: DataFrame containing stock data with news sentiment for each day
        """
        # Use symbol as company name if not provided
        if company_name is None:
            company_name = symbol.split('.')[0]  # Remove exchange suffix
            
        # Fetch stock data for the entire available period
        stock_data = self.fetch_stock_data(symbol, period='max')
        if stock_data is None:
            return None
            
        # Take only the last 'days' of stock data
        stock_data = stock_data.tail(days)
        
        # Fetch news data
        news_data = self.fetch_news(company_name, days)
        if news_data is None:
            news_data = pd.DataFrame(columns=['title', 'description', 'url', 'publishedAt'])
        
        # Convert dates to datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        news_data['date'] = pd.to_datetime(news_data['publishedAt'])
        
        # Group news by date and aggregate titles and descriptions
        news_by_date = news_data.groupby(news_data['date'].dt.date).agg({
            'title': lambda x: ' | '.join(x),
            'description': lambda x: ' | '.join(x),
            'url': lambda x: ' | '.join(x)
        }).reset_index()
        
        # Convert stock dates to date (without time) for merging
        stock_dates = stock_data['Date'].dt.date
        
        # Create combined dataframe
        combined_df = pd.DataFrame({
            'Date': stock_data['Date'],
            'Open': stock_data['Open'],
            'High': stock_data['High'],
            'Low': stock_data['Low'],
            'Close': stock_data['Close'],
            'Volume': stock_data['Volume']
        })
        
        # Add news data columns
        combined_df['date_key'] = stock_dates
        combined_df = combined_df.merge(
            news_by_date,
            left_on='date_key',
            right_on='date',
            how='left'
        )
        
        # Clean up merged dataframe
        combined_df = combined_df.drop(['date_key', 'date'], axis=1)
        
        # Fill missing news data with empty strings
        combined_df['title'] = combined_df['title'].fillna('')
        combined_df['description'] = combined_df['description'].fillna('')
        combined_df['url'] = combined_df['url'].fillna('')
        
        # Save combined data
        csv_path = f'data/{symbol}_combined_data.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"Combined data saved to {csv_path}")
        
        return combined_df
