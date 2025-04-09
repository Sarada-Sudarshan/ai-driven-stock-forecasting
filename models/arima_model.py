import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
        
class HybridStockModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.arima_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()
        self.order = None
        self.last_sequence = None
    
    def prepare_xgb_data(self, residuals):
        """Prepare sequence data for XGBoost training"""
        scaled_residuals = self.scaler.fit_transform(residuals.reshape(-1, 1))
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_residuals)):
            X.append(scaled_residuals[i - self.sequence_length:i, 0])
            y.append(scaled_residuals[i, 0])
        return np.array(X), np.array(y)
    
    def train_arima(self, train_data):
        """Train ARIMA model"""
        self.order = auto_arima(train_data, seasonal=False, stepwise=True, trace=False).order
        self.arima_model = ARIMA(train_data, order=self.order).fit()
        arima_pred = self.arima_model.fittedvalues
        residuals = train_data - arima_pred
        # Store the last sequence for prediction
        self.last_sequence = residuals[-self.sequence_length:]
        return residuals
    
    def train_xgboost(self, residuals):
        """Train XGBoost on residuals"""
        X_train, y_train = self.prepare_xgb_data(residuals)
        params = {'objective': 'reg:squarederror', 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(X_train, y_train)
    
    def train(self, train_data):
        """Train both ARIMA and XGBoost models"""
        residuals = self.train_arima(train_data)
        self.train_xgboost(residuals)
    
    def predict(self, steps):
        """Make future predictions using hybrid model"""
        # Get ARIMA predictions
        arima_forecast = self.arima_model.forecast(steps)
        
        # Prepare sequence for XGBoost predictions
        xgb_predictions = []
        current_sequence = self.last_sequence.copy()
        
        # Make predictions one step at a time
        for _ in range(steps):
            # Scale the current sequence
            scaled_seq = self.scaler.transform(current_sequence.reshape(-1, 1))
            # Make prediction
            xgb_pred = self.xgb_model.predict(scaled_seq.reshape(1, -1))
            # Inverse transform the prediction
            xgb_pred = self.scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0, 0]
            xgb_predictions.append(xgb_pred)
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = xgb_pred
        
        # Combine predictions
        final_forecast = arima_forecast + np.array(xgb_predictions)
        return final_forecast
    
    def evaluate(self, test_data, predictions):
        mse = mean_squared_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        print(f'MSE: {mse:.2f}, R²: {r2:.4f}')
    
    def plot_forecast(self, train_data, test_data, predictions):
        plt.figure(figsize=(15, 6))
        plt.plot(train_data, label='Training Data', color='blue')
        plt.plot(test_data, label='Actual Test Data', color='green')
        plt.plot(predictions, label='Hybrid Predictions', linestyle='dashed', color='red')
        plt.legend()
        plt.title('Hybrid ARIMA-XGBoost Stock Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.grid()
        plt.show()
