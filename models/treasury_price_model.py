import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class TreasuryPriceModel:
    """
    A model that predicts monthly price based on US Treasury prints.
    
    This is a simplified example model that assumes a linear relationship
    between treasury printing volumes and prices.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = [
            'treasury_print_volume', 
            'interest_rate',
            'inflation_rate', 
            'previous_month_price'
        ]
    
    def preprocess_features(self, X: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the input features.
        
        Args:
            X: Dictionary of feature values
            
        Returns:
            Preprocessed features array
        """
        # Extract features in the correct order
        features = np.array([[
            X.get('treasury_print_volume', 0),
            X.get('interest_rate', 0),
            X.get('inflation_rate', 0),
            X.get('previous_month_price', 0)
        ]])
        
        # Scale if the model is trained
        if self.trained:
            return self.scaler.transform(features)
        return features
    
    def train(self, training_data: List[Dict[str, Any]], prices: List[float]):
        """
        Train the model on historical treasury data and prices.
        
        Args:
            training_data: List of dictionaries with feature values
            prices: List of corresponding prices
        """
        # Extract features
        X = np.array([[
            item.get('treasury_print_volume', 0),
            item.get('interest_rate', 0),
            item.get('inflation_rate', 0),
            item.get('previous_month_price', 0)
        ] for item in training_data])
        
        y = np.array(prices)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        self.trained = True
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict price based on given features.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Predicted price
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.preprocess_features(features)
        return float(self.model.predict(X)[0])
    
    def generate_training_data(self, n_samples: int = 100) -> tuple:
        """
        Generate synthetic training data for demonstration purposes.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (training_data, prices)
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate random features with realistic trends
        treasury_volume = np.linspace(800, 1200, n_samples) + np.random.normal(0, 50, n_samples)
        interest_rates = np.clip(np.random.normal(2.5, 1.0, n_samples), 0.1, 7.0)
        inflation_rates = np.clip(np.random.normal(2.0, 0.8, n_samples), 0.2, 6.0)
        
        # Initial price
        prices = [100.0]
        
        # Generate dependent prices with some noise
        for i in range(n_samples - 1):
            # Higher treasury volume tends to decrease prices
            # Higher interest rates tend to decrease prices
            # Higher inflation tends to increase prices
            new_price = prices[-1]
            new_price -= (treasury_volume[i] - 1000) * 0.01  # Effect of treasury volume
            new_price -= interest_rates[i] * 0.5             # Effect of interest rates
            new_price += inflation_rates[i] * 0.7            # Effect of inflation
            new_price += np.random.normal(0, 2)              # Random noise
            prices.append(float(new_price))
        
        # Create training data
        training_data = []
        for i in range(n_samples - 1):
            training_data.append({
                'treasury_print_volume': float(treasury_volume[i]),
                'interest_rate': float(interest_rates[i]),
                'inflation_rate': float(inflation_rates[i]),
                'previous_month_price': prices[i]
            })
        
        # Return training data and prices (excluding the first price used only as previous price)
        return training_data, prices[1:]
    
    def evaluate(self, test_data: List[Dict[str, Any]], actual_prices: List[float]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of dictionaries with feature values
            actual_prices: List of corresponding actual prices
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = [self.predict(features) for features in test_data]
        errors = np.array(predictions) - np.array(actual_prices)
        
        return {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(np.square(errors)))),
            'mape': float(np.mean(np.abs(errors / np.array(actual_prices)) * 100))
        }