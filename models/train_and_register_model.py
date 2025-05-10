#!/usr/bin/env python3

import os
import sys
import logging
import argparse

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel
from registry.model_registry import ModelRegistry
from registry.registry_service import KafkaModelRegistryService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_register_model(registry_path, kafka_bootstrap_servers, train_samples=100):
    """Train and register a treasury price model.
    
    Args:
        registry_path: Path to the registry storage
        kafka_bootstrap_servers: Kafka bootstrap servers
        train_samples: Number of training samples to generate
    """
    logger.info("Creating model registry...")
    model_registry = ModelRegistry(registry_path)
    
    logger.info("Creating registry service...")
    registry_service = KafkaModelRegistryService(
        model_registry=model_registry,
        kafka_bootstrap_servers=kafka_bootstrap_servers
    )
    
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
    
    # Register the model
    logger.info("Registering model...")
    model_id = registry_service.register_model(
        model_name="treasury_price_model",
        model=model,
        metadata={"metrics": metrics}
    )
    
    logger.info(f"Model registered with ID: {model_id}")
    return model_id

def main():
    parser = argparse.ArgumentParser(description='Train and register a treasury price model')
    parser.add_argument(
        '--registry-path', 
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry/storage"),
        help='Path to the registry storage'
    )
    parser.add_argument(
        '--kafka-bootstrap-servers', 
        default='localhost:29092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--train-samples', 
        type=int, 
        default=100,
        help='Number of training samples to generate'
    )
    
    args = parser.parse_args()
    
    # Create registry storage directory if it doesn't exist
    os.makedirs(args.registry_path, exist_ok=True)
    
    train_and_register_model(
        registry_path=args.registry_path,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        train_samples=args.train_samples
    )

if __name__ == "__main__":
    main()