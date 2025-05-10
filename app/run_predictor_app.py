#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import signal

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.application import PredictorApplication

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run the predictor application')
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
        '--treasury-topic', 
        default='treasury.data',
        help='Topic name for treasury data events'
    )
    parser.add_argument(
        '--signals-topic', 
        default='signals.topic',
        help='Topic name for signal events'
    )
    parser.add_argument(
        '--registry-topic', 
        default='model.registry',
        help='Topic name for model registry events'
    )
    parser.add_argument(
        '--consumer-group', 
        default='predictor_app',
        help='Consumer group ID'
    )
    
    args = parser.parse_args()
    
    # Create an instance of the predictor application
    app = PredictorApplication(
        bootstrap_servers=args.kafka_bootstrap_servers,
        treasury_topic=args.treasury_topic,
        signals_topic=args.signals_topic,
        registry_topic=args.registry_topic,
        consumer_group=args.consumer_group
    )
    
    # Handle graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        app.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start the application
    logger.info("Starting predictor application...")
    app.start(args.registry_path)
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        app.stop()
    
if __name__ == "__main__":
    main()