import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

class XGBoostStockModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Drop any NaN values
        df = df.dropna()
        
        # Scale the closing prices
        scaled_data = self.scaler.fit_transform(df[['Close']].values)
        
        # Prepare X (current price) and y (next day's price)
        X = scaled_data[:-1]  # All prices except last
        y = scaled_data[1:]   # All prices except first
        
        return X, y.ravel()  # Flatten y
    
    def train(self, X_train, y_train):
        """Train the XGBoost model"""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        predictions = self.model.predict(X)
        # Convert predictions back to original scale
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        return predictions.flatten()
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        actual = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared Score: {r2:.4f}")
        
        return mse, r2