import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_fetcher import StockDataFetcher
from models.arima_model import HybridStockModel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def test_arima(symbol='GOOGL', period='1y'):
    """
    Test Hybrid ARIMA-XGBoost model for stock price prediction
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to test
    period : str
        Data period to fetch ('1mo', '3mo', '6mo', '1y', '2y', '5y', etc.)
    """
    try:
        # 1. Fetch Data
        print(f"Fetching stock data for {symbol}...")
        fetcher = StockDataFetcher()
        stock_data = fetcher.fetch_stock_data(symbol, period=period)
        
        if stock_data is None or len(stock_data) < 60:  # At least 60 days of data
            raise ValueError(f"Insufficient data for {symbol}")
        
        # 2. Prepare Data
        closing_prices = stock_data['Close'].values  # Convert to numpy array
        dates = stock_data.index
        
        # Split into train and test (80-20 split)
        train_size = int(len(closing_prices) * 0.8)
        train_data = closing_prices[:train_size]
        test_data = closing_prices[train_size:]
        
        print("\nData Summary:")
        print(f"Total data points: {len(closing_prices)}")
        print(f"Training data points: {len(train_data)}")
        print(f"Testing data points: {len(test_data)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Price range: ${closing_prices.min():.2f} to ${closing_prices.max():.2f}")
        
        # 3. Initialize and Train Hybrid model
        print("\nInitializing Hybrid ARIMA-XGBoost model...")
        model = HybridStockModel(sequence_length=60)
        model.train(train_data)
        
        # 4. Make predictions for test period
        print("\nMaking predictions...")
        forecast = model.predict(steps=len(test_data))
        
        # 5. Calculate Error Metrics
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, forecast)
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        r2 = r2_score(test_data, forecast)
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(test_data))
        pred_direction = np.sign(np.diff(forecast))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print("\nModel Performance Metrics:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        # 6. Plot Results
        model.plot_forecast(train_data, test_data, forecast)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'forecast': forecast,
            'actual': test_data
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with 1 year of data by default
    test_arima()
