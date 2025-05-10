#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import signal
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.application import ApplicationConsumer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def signal_handler(signal_data: Dict[str, Any]):
    """Example signal handler that processes price prediction signals.
    
    Args:
        signal_data: Signal data with prediction
    """
    model_name = signal_data.get('model_name')
    model_version = signal_data.get('model_version')
    features = signal_data.get('features', {})
    predicted_price = signal_data.get('predicted_price')
    
    # Print signal info
    logger.info(f"Received signal from model: {model_name} (version: {model_version})")
    logger.info(f"Features: treasury_volume={features.get('treasury_print_volume', 'N/A'):.2f}, "
                f"interest_rate={features.get('interest_rate', 'N/A'):.2f}, "
                f"inflation_rate={features.get('inflation_rate', 'N/A'):.2f}")
    logger.info(f"Predicted price: ${predicted_price:.2f}")
    
    # Example logic for generating alerts based on predicted price
    if predicted_price < 90.0:
        logger.warning(f"ALERT: Price prediction below critical threshold! (${predicted_price:.2f})")
    elif predicted_price > 110.0:
        logger.warning(f"ALERT: Price prediction above critical threshold! (${predicted_price:.2f})")

def main():
    parser = argparse.ArgumentParser(description='Run the application consumer')
    parser.add_argument(
        '--kafka-bootstrap-servers', 
        default='localhost:29092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--signals-topic', 
        default='signals.topic',
        help='Topic name for signal events'
    )
    parser.add_argument(
        '--consumer-group', 
        default='application_consumer',
        help='Consumer group ID'
    )
    
    args = parser.parse_args()
    
    # Create an instance of the application consumer
    app = ApplicationConsumer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        signals_topic=args.signals_topic,
        consumer_group=args.consumer_group
    )
    
    # Register signal handler
    app.register_signal_handler(signal_handler)
    
    # Handle graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        app.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start the application
    logger.info("Starting application consumer...")
    app.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        app.stop()
    
if __name__ == "__main__":
    main()