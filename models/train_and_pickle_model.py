#!/usr/bin/env python3

import os
import sys
import pickle
import argparse
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_pickle_model(output_path: str, train_samples: int = 100) -> str:
    """Train and pickle a treasury price model.
    
    Args:
        output_path: Path to save the pickled model
        train_samples: Number of training samples to generate
        
    Returns:
        Path to the pickled model
    """
    logger.info(f"Training treasury price model with {train_samples} samples...")
    model = TreasuryPriceModel()
    training_data, prices = model.generate_training_data(n_samples=train_samples)
    
    # Split into training and validation sets
    split_idx = int(len(training_data) * 0.8)
    train_features, val_features = training_data[:split_idx], training_data[split_idx:]
    train_prices, val_prices = prices[:split_idx], prices[split_idx:]
    
    # Train the model
    logger.info("Training model...")
    model.train(train_features, train_prices)
    
    # Evaluate the model
    logger.info("Evaluating model...")
    metrics = model.evaluate(val_features, val_prices)
    logger.info(f"Validation metrics: {metrics}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model as a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
        
    logger.info(f"Model saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Train and pickle a treasury price model')
    parser.add_argument(
        '--output-path', 
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "models/pickles/treasury_price_model.pkl"),
        help='Path to save the pickled model'
    )
    parser.add_argument(
        '--train-samples', 
        type=int, 
        default=100,
        help='Number of training samples to generate'
    )
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    train_and_pickle_model(
        output_path=args.output_path,
        train_samples=args.train_samples
    )

if __name__ == "__main__":
    main()