from models.hybrid_model import HybridModel
from utils.data_fetcher import StockDataFetcher
import pandas as pd
import numpy as np

def test_hybrid_model(symbol='TCS.NS', company_name='Tata Consultancy Services'):
    """Test the hybrid model combining XGBoost with sentiment analysis"""
    print(f"\nTesting Hybrid Model for {symbol}")
    
    # Load existing stock data
    print("\nLoading stock data from CSV...")
    try:
        stock_data = pd.read_csv(f'data/{symbol}_stock_data.csv', index_col='Date')
        print(f"Successfully loaded data with {len(stock_data)} data points")
    except FileNotFoundError:
        print(f"Error: Could not find stock data file data/{symbol}_stock_data.csv")
        return
        
    # Initialize model
    model = HybridModel(sequence_length=60, sentiment_weight=0.3)
    
    # Prepare data
    print("\nPreparing training data...")
    X, y = model.prepare_data(stock_data, company_name)
    
    # Split data into training and testing sets
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTraining with {len(X_train)} samples")
    print(f"Testing with {len(X_test)} samples")
    
    # Train model
    print("\nTraining hybrid model...")
    model.train(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nModel Performance Metrics:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    
    # Plot results
    actual = model.xgb_model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    model.plot_results(actual, predictions, title=f'Hybrid Model Predictions for {symbol}')
    
    print("\nHybrid model testing complete!")

if __name__ == "__main__":
    test_hybrid_model() 