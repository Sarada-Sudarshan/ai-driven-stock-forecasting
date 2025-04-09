import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

class LSTMModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, closing_prices):
        """Prepare data for LSTM
        
        Parameters:
        closing_prices: numpy array of closing prices
        """
        # Ensure data is reshaped properly for scaling
        values = closing_prices.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y
    
    def build_model(self, input_shape):
        """Build LSTM model with improved architecture"""
        model = Sequential([
            # First LSTM layer with more units
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for better feature extraction
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Ensure X is in the right shape
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
        # Make predictions
        scaled_predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform to get actual values
        predictions = self.scaler.inverse_transform(scaled_predictions)
        
        return predictions.flatten()  # Return flattened array for easier handling

    def predict_future(self, last_sequence, n_steps=10):
        """Predict future values
        
        Parameters:
        last_sequence: The last sequence of actual values (length should be sequence_length)
        n_steps: Number of future steps to predict
        
        Returns:
        Array of predicted future values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Scale the last sequence
        last_sequence = last_sequence.reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Initialize predictions array
        future_predictions = []
        current_sequence = scaled_sequence[-self.sequence_length:].flatten()
        
        # Predict future steps
        for _ in range(n_steps):
            # Prepare current sequence for prediction
            current_input = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Get next prediction
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, next_pred)
        
        # Convert predictions back to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()
