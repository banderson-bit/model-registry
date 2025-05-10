#!/usr/bin/env python3

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.application import TreasuryDataProducer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_treasury_data(
    base_volume: float = 1000.0,
    base_interest_rate: float = 2.5,
    base_inflation_rate: float = 2.0,
    base_price: float = 100.0
) -> Dict[str, Any]:
    """Generate realistic treasury data with some random variation.
    
    Args:
        base_volume: Base treasury print volume
        base_interest_rate: Base interest rate
        base_inflation_rate: Base inflation rate
        base_price: Base price for previous month
        
    Returns:
        Dictionary with treasury data features
    """
    # Add some randomness to the base values
    volume = base_volume + np.random.normal(0, 50)
    interest_rate = np.clip(base_interest_rate + np.random.normal(0, 0.2), 0.1, 7.0)
    inflation_rate = np.clip(base_inflation_rate + np.random.normal(0, 0.15), 0.2, 6.0)
    previous_month_price = base_price + np.random.normal(0, 2)
    
    return {
        'treasury_print_volume': float(volume),
        'interest_rate': float(interest_rate),
        'inflation_rate': float(inflation_rate),
        'previous_month_price': float(previous_month_price)
    }

def run_producer(kafka_bootstrap_servers: str, topic_name: str, interval: float, iterations: int):
    """Run the treasury data producer.
    
    Args:
        kafka_bootstrap_servers: Kafka bootstrap servers
        topic_name: Treasury data topic name
        interval: Interval between events in seconds
        iterations: Number of events to produce (0 for infinite)
    """
    producer = TreasuryDataProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        topic_name=topic_name
    )
    
    # Start with reasonable base values
    base_volume = 1000.0
    base_interest_rate = 2.5
    base_inflation_rate = 2.0
    base_price = 100.0
    
    count = 0
    try:
        while iterations == 0 or count < iterations:
            # Generate and produce treasury data
            treasury_data = generate_treasury_data(
                base_volume, base_interest_rate, base_inflation_rate, base_price
            )
            
            producer.produce_treasury_data(treasury_data)
            
            # Update base values for slight drift over time
            base_volume += np.random.normal(0, 5)
            base_interest_rate += np.random.normal(0, 0.05)
            base_inflation_rate += np.random.normal(0, 0.04)
            base_price += np.random.normal(0, 0.5)
            
            # Ensure values stay within reasonable ranges
            base_volume = np.clip(base_volume, 800, 1200)
            base_interest_rate = np.clip(base_interest_rate, 0.5, 6.0)
            base_inflation_rate = np.clip(base_inflation_rate, 0.5, 5.0)
            base_price = np.clip(base_price, 70, 130)
            
            time.sleep(interval)
            count += 1
            
    except KeyboardInterrupt:
        logger.info("Producer stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Produce simulated treasury data events')
    parser.add_argument(
        '--kafka-bootstrap-servers', 
        default='localhost:29092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--topic-name', 
        default='treasury.data',
        help='Topic name for treasury data events'
    )
    parser.add_argument(
        '--interval', 
        type=float, 
        default=2.0,
        help='Interval between events in seconds'
    )
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=0,
        help='Number of events to produce (0 for infinite)'
    )
    
    args = parser.parse_args()
    
    run_producer(
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        topic_name=args.topic_name,
        interval=args.interval,
        iterations=args.iterations
    )

if __name__ == "__main__":
    main()