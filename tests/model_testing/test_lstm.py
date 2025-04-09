import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_fetcher import StockDataFetcher
from utils.sentiment_analyzer import SentimentAnalyzer
from models.lstm import LSTMModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from newsapi import NewsApiClient
from dotenv import load_dotenv

def get_recent_sentiment(symbol, company_name):
    """Get sentiment from recent news headlines"""
    try:
        # Initialize components
        load_dotenv()
        news_api_key = os.getenv('NEWS_API_KEY')
        newsapi = NewsApiClient(api_key=news_api_key)
        sentiment_analyzer = SentimentAnalyzer()
        
        # Get recent news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days of news
        
        news = newsapi.get_everything(
            q=company_name,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt'
        )
        
        if not news['articles']:
            return 0.0
            
        # Analyze sentiment for each article
        sentiments = []
        for article in news['articles']:
            title = article.get('title', '')
            description = article.get('description', '')
            if title or description:
                text = f"{title} {description}".strip()
                if text:
                    sentiment = sentiment_analyzer.analyze_text(text)
                    sentiments.append(sentiment['compound'])
        
        # Return average sentiment (-1 to 1)
        if sentiments:
            return np.mean(sentiments)
        return 0.0
        
    except Exception as e:
        print(f"Error getting sentiment: {str(e)}")
        return 0.0

def test_lstm(symbol, period='10y', save_path=None):
    """
    Test LSTM model for a given company symbol
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        period (str): Data period (e.g., '5y', '10y', 'max')
        save_path (str): Path to save the prediction plot
        
    Returns:
        dict: Dictionary containing metrics and predictions
    """
    print(f"Starting LSTM test for {symbol}...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Fetch Data
    print(f"Fetching stock data for {symbol}...")
    fetcher = StockDataFetcher()
    stock_data = fetcher.fetch_stock_data(symbol, period=period)
    
    if stock_data is None:
        raise ValueError(f"Could not fetch data for symbol {symbol}")
    
    # Ensure we have a datetime index
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data['Date'])
    
    print("Stock data fetched successfully")
    
    # 2. Initialize LSTM model
    print("Initializing LSTM model...")
    lstm = LSTMModel(sequence_length=60)
    
    # 3. Prepare Data
    print("Preparing data...")
    closing_prices = stock_data['Close'].values
    
    # Split data into train and test
    train_size = int(len(closing_prices) * 0.8)
    train_data = closing_prices[:train_size]
    test_data = closing_prices[train_size:]
    
    print(f"Total data points: {len(closing_prices)}")
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")
    
    # Prepare sequences for training
    X_train, y_train = lstm.prepare_data(train_data)
    X_test, y_test = lstm.prepare_data(test_data)
    
    # 4. Train the model
    print("\nTraining LSTM model...")
    history = lstm.train(X_train, y_train, epochs=50, batch_size=32)
    
    # 5. Make predictions
    print("\nMaking predictions...")
    train_predictions = lstm.predict(X_train)
    test_predictions = lstm.predict(X_test)
    
    # 6. Get recent sentiment and predict future values
    print("\nAnalyzing recent news sentiment...")
    company_name = {
        'AAPL': 'Apple Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc',
        'AMZN': 'Amazon.com Inc',
        'CRM': 'Salesforce Inc'
    }.get(symbol, symbol)
    
    sentiment = get_recent_sentiment(symbol, company_name)
    print(f"Recent news sentiment: {sentiment:.2f}")
    
    print("\nPredicting future values...")
    last_sequence = closing_prices[-lstm.sequence_length:]
    future_predictions = lstm.predict_future(last_sequence, n_steps=15)
    
    # Adjust future predictions based on sentiment
    if abs(sentiment) > 0.1:  # Only adjust if sentiment is significant
        adjustment_factor = 1 + (sentiment * 0.01)  # 1% adjustment per 1.0 sentiment
        trend = future_predictions[-1] - future_predictions[0]
        if (trend > 0 and sentiment < 0) or (trend < 0 and sentiment > 0):
            # If trend and sentiment disagree, adjust predictions
            future_predictions = future_predictions * adjustment_factor
    
    # Generate future dates (business days only)
    last_date = stock_data.index[-1]
    print(f"Last date in data: {last_date}")
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=15, freq='B')
    print(f"Future dates range: {future_dates[0]} to {future_dates[-1]}")
    
    # 7. Calculate metrics
    print("Calculating metrics...")
    mse = mean_squared_error(test_data[lstm.sequence_length:], test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data[lstm.sequence_length:], test_predictions)
    r2 = r2_score(test_data[lstm.sequence_length:], test_predictions)
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    # 8. Generate and save plot
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    
    # Get dates for plotting
    test_dates = stock_data.index[train_size+lstm.sequence_length:]
    
    # Configure date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Plot actual prices
    plt.plot(test_dates, test_data[lstm.sequence_length:], 
             label='Actual Prices', color='blue')
    
    # Combine test predictions and future predictions
    all_predictions = np.concatenate([test_predictions, future_predictions])
    all_dates = test_dates.union(future_dates)
    
    # Plot all predictions as one continuous red dashed line
    plt.plot(all_dates, all_predictions,
             label='Predicted Prices', color='red', linestyle='--')
    
    # Add sentiment annotation
    sentiment_text = 'Bullish' if sentiment > 0.1 else 'Bearish' if sentiment < -0.1 else 'Neutral'
    plt.annotate(f'Recent News Sentiment: {sentiment_text} ({sentiment:.2f})',
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
    
    plt.title(f'Stock Price Predictions for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()
    
    # Use tight layout to prevent date labels from being cut off
    plt.tight_layout()
    
    if save_path:
        print(f"Saving plot to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("Plot saved successfully")
    plt.close()
    
    # 9. Create date range for data points
    dates = test_dates.strftime('%Y-%m-%d').tolist()
    future_dates_list = future_dates.strftime('%Y-%m-%d').tolist()
    
    print("LSTM test completed successfully")
    
    # 10. Return results
    return {
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'dates': dates,
        'actual': test_data[lstm.sequence_length:].tolist(),
        'predictions': test_predictions.tolist(),
        'future_dates': future_dates_list,
        'future_predictions': future_predictions.tolist(),
        'sentiment': float(sentiment),
        'simulation_range': {
            'start': dates[0],
            'end': future_dates_list[-1]
        }
    }

if __name__ == "__main__":
    # For testing purposes only
    results = test_lstm('AAPL', save_path='lstm_predictions.png')
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"{metric.upper()}: {value:.4f}")
