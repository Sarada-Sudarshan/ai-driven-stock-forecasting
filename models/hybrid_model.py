import numpy as np
import pandas as pd
from models.xgboost_model import XGBoostStockModel
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.data_fetcher import StockDataFetcher
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

class HybridModel:
    def __init__(self, sequence_length=60, sentiment_weight=0.3):
        self.xgb_model = XGBoostStockModel(sequence_length=sequence_length)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_fetcher = StockDataFetcher()
        self.sentiment_weight = sentiment_weight
        self.price_weight = 1 - sentiment_weight
        self.sequence_length = sequence_length
        self.sentiment_scaler = MinMaxScaler()
        
    def prepare_data(self, price_data, company_name):
        """Prepare data by combining price data with sentiment analysis"""
        # Get combined data with news
        combined_data = self.data_fetcher.get_combined_data(
            symbol=None,  # We already have the price data
            company_name=company_name,
            days=30  # Get last 30 days of news
        )
        
        if combined_data is None or len(combined_data) == 0:
            print("Warning: No sentiment data available. Using price data only.")
            return self.xgb_model.prepare_data(price_data)
            
        # Analyze sentiment
        sentiment_data = self.sentiment_analyzer.analyze_combined_data(combined_data)
        
        # Convert sentiment scores to DataFrame with dates as index
        sentiment_df = pd.DataFrame({
            'date': sentiment_data['Date'],
            'sentiment': sentiment_data['sentiment_compound']
        })
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        # Merge price data with sentiment data
        price_data.index = pd.to_datetime(price_data.index)
        merged_data = price_data.join(sentiment_df, how='left')
        
        # Forward fill sentiment scores for days without news
        merged_data['sentiment'] = merged_data['sentiment'].fillna(method='ffill')
        # For the initial days without sentiment, use neutral sentiment (0.5)
        merged_data['sentiment'] = merged_data['sentiment'].fillna(0.5)
        
        # Scale sentiment scores
        merged_data['sentiment'] = self.sentiment_scaler.fit_transform(
            merged_data['sentiment'].values.reshape(-1, 1)
        ).flatten()
        
        # Prepare sequences using price data
        X, y = self.xgb_model.prepare_data(merged_data)
        
        # Add sentiment features to X
        sentiment_sequences = []
        for i in range(self.sequence_length, len(merged_data)):
            sentiment_sequences.append(merged_data['sentiment'].values[i-self.sequence_length:i])
        sentiment_sequences = np.array(sentiment_sequences)
        
        # Combine price and sentiment features
        X_combined = np.column_stack([X, sentiment_sequences])
        
        return X_combined, y
        
    def train(self, X_train, y_train):
        """Train the hybrid model"""
        return self.xgb_model.train(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using both price and sentiment data"""
        price_features = X[:, :self.sequence_length]
        sentiment_features = X[:, self.sequence_length:]
        
        # Get price-based predictions
        price_predictions = self.xgb_model.predict(price_features)
        
        # Calculate sentiment adjustment
        sentiment_adjustment = np.mean(sentiment_features, axis=1)
        sentiment_adjustment = self.sentiment_scaler.inverse_transform(
            sentiment_adjustment.reshape(-1, 1)
        ).flatten()
        
        # Combine predictions with sentiment
        final_predictions = (
            self.price_weight * price_predictions +
            self.sentiment_weight * sentiment_adjustment
        )
        
        return final_predictions
        
    def evaluate(self, X_test, y_test):
        """Evaluate the hybrid model"""
        predictions = self.predict(X_test)
        actual = self.xgb_model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
        
    def plot_results(self, actual, predictions, title='Hybrid Model Predictions vs Actual'):
        """Plot the predictions against actual values"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
