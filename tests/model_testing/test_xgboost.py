import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_fetcher import StockDataFetcher
from models.xgboost_model import XGBoostStockModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def test_xgboost():
    try:
        # 1. Fetch Data
        print("Fetching stock data...")
        fetcher = StockDataFetcher()
        stock_data = fetcher.fetch_stock_data('GOOGL', period='2y')
        
        # 2. Initialize model
        print("\nInitializing XGBoost model...")
        model = XGBoostStockModel()
        
        # 3. Prepare features and target
        print("\nPreparing features...")
        X, y = model.prepare_data(stock_data)
        
        print("\nData Summary:")
        print(f"Total samples: {len(X)}")
        
        # 4. Split data using sklearn's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nData splits:")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # 5. Train model
        print("\nTraining XGBoost model...")
        model.train(X_train, y_train)
        
        # 6. Make predictions
        print("\nMaking predictions...")
        test_pred = model.predict(X_test)
        
        # 7. Evaluate performance
        print("\nModel Performance:")
        print("\nTest Set Metrics:")
        test_mse, test_r2 = model.evaluate(X_test, y_test)
        
        # 8. Plot predictions
        print("\nPlotting predictions...")
        plt.figure(figsize=(15, 6))
        
        # Sort test predictions by original index for plotting
        test_indices = np.arange(len(X))[int(len(X) * 0.8):]
        sorted_indices = np.argsort(test_indices)
        
        # Plot actual prices
        actual_prices = model.scaler.inverse_transform(y_test.reshape(-1, 1))
        plt.plot(sorted_indices, actual_prices, label='Actual', color='blue')
        
        # Plot predicted prices
        plt.plot(sorted_indices, test_pred, label='Predicted', color='red', linestyle='--')
        
        plt.title("GOOGL Stock Price Prediction (Next Day)")
        plt.xlabel("Trading Days")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return test_r2
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    test_xgboost()
