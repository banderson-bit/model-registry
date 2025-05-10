#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import threading
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import mlflow

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel
from app.kafka_client import KafkaClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowClient:
    """Client for interacting with MLflow."""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize the client.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def get_model(self, model_name: str, stage: str = "None") -> Optional[Dict[str, Any]]:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            stage: Model stage (e.g., "None", "Staging", "Production")
            
        Returns:
            Model information
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                logger.warning(f"No model found with name: {model_name}, stage: {stage}")
                return None
            
            model_info = versions[0]
            return {
                'name': model_info.name,
                'version': model_info.version,
                'stage': model_info.current_stage,
                'run_id': model_info.run_id
            }
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            return None

    def load_model(self, model_name: str, stage: str = "None") -> Optional[Any]:
        """Load a model from MLflow.
        
        Args:
            model_name: Name of the model
            stage: Model stage (e.g., "None", "Staging", "Production")
            
        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{stage}"
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def predict(self, model, features: Dict[str, float]) -> float:
        """Make a prediction using an MLflow model.
        
        Args:
            model: MLflow model
            features: Dictionary of feature values
            
        Returns:
            Prediction
        """
        # Convert to DataFrame which is what our MLflow wrapper expects
        input_df = pd.DataFrame({
            'treasury_print_volume': [features.get('treasury_print_volume', 0)],
            'interest_rate': [features.get('interest_rate', 0)],
            'inflation_rate': [features.get('inflation_rate', 0)],
            'previous_month_price': [features.get('previous_month_price', 0)]
        })
        
        try:
            predictions = model.predict(input_df)
            return float(predictions[0])
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None


class KafkaTreasuryDataProducer:
    """Producer for test treasury data events."""
    
    def __init__(self, bootstrap_servers: str, topic_name: str):
        """Initialize the producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_name: Topic for treasury data events
        """
        self.kafka_client = KafkaClient(bootstrap_servers)
        self.topic_name = topic_name
        
    def produce_data(self, data: Dict[str, Any]) -> bool:
        """Produce a treasury data event.
        
        Args:
            data: Treasury data with features
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            'timestamp': time.time(),
            'data': data
        }
        
        success = self.kafka_client.produce_message(
            self.topic_name,
            value=message,
            key=str(message['timestamp'])
        )
        
        if success:
            logger.info(f"Produced treasury data: {data}")
        else:
            logger.error(f"Failed to produce treasury data: {data}")
            
        return success


class KafkaTreasuryDataConsumer:
    """Consumer for treasury data events that makes predictions."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        treasury_topic: str,
        signals_topic: str,
        consumer_group: str,
        mlflow_client: MLflowClient,
        model_name: str
    ):
        """Initialize the consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            treasury_topic: Topic for treasury data events
            signals_topic: Topic for signals
            consumer_group: Consumer group ID
            mlflow_client: MLflow client
            model_name: Name of the model in MLflow
        """
        self.kafka_client = KafkaClient(bootstrap_servers)
        self.treasury_topic = treasury_topic
        self.signals_topic = signals_topic
        self.consumer_group = consumer_group
        self.mlflow_client = mlflow_client
        self.model_name = model_name
        
        self.model = None
        self._running = False
        self._consumer_thread = None
        self._producer_thread = None
        self.predictions = []
        
    def start(self) -> bool:
        """Start the consumer.
        
        Returns:
            True if successful, False otherwise
        """
        if self._running:
            return True
            
        # Load the model
        self.model = self.mlflow_client.load_model(self.model_name)
        if self.model is None:
            logger.error(f"Failed to load model: {self.model_name}")
            return False
        
        self._running = True
        
        # Start consumer thread
        self._consumer_thread = threading.Thread(target=self._consume)
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        logger.info(f"Started Kafka consumer for topic: {self.treasury_topic}")
        
        return True
        
    def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=10)
        logger.info("Stopped Kafka consumer")
        
    def _consume(self) -> None:
        """Consume messages from Kafka."""
        def handle_message(message):
            try:
                data = message.get('data', {})
                logger.info(f"Received treasury data: {data}")
                
                # Make prediction using MLflow model
                prediction = self.mlflow_client.predict(self.model, data)
                
                if prediction is not None:
                    result = {
                        'features': data,
                        'predicted_price': prediction,
                        'model_name': self.model_name,
                        'timestamp': time.time()
                    }
                    self.predictions.append(result)
                    logger.info(f"Prediction made: ${prediction:.2f}")
                    
                    # Produce prediction to signals topic
                    self._produce_signal(result)
            except Exception as e:
                logger.exception(f"Error handling message: {e}")
        
        self.kafka_client.consume_messages(
            topics=[self.treasury_topic],
            group_id=self.consumer_group,
            handler=handle_message,
            run_forever=lambda: self._running
        )
    
    def _produce_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Produce a signal event.
        
        Args:
            signal_data: Signal data with prediction
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            'timestamp': time.time(),
            'signal': signal_data
        }
        
        success = self.kafka_client.produce_message(
            self.signals_topic,
            value=message,
            key=str(message['timestamp'])
        )
        
        if success:
            logger.debug(f"Produced signal: {signal_data}")
        else:
            logger.error(f"Failed to produce signal: {signal_data}")
            
        return success


class SignalConsumer:
    """Consumer for signal events."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        signals_topic: str,
        consumer_group: str
    ):
        """Initialize the consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            signals_topic: Topic for signals
            consumer_group: Consumer group ID
        """
        self.kafka_client = KafkaClient(bootstrap_servers)
        self.signals_topic = signals_topic
        self.consumer_group = consumer_group
        
        self._running = False
        self._consumer_thread = None
        self.signals = []
        
    def start(self) -> None:
        """Start the consumer."""
        if self._running:
            return
            
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consume)
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        logger.info(f"Started Kafka consumer for topic: {self.signals_topic}")
        
    def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=10)
        logger.info("Stopped Kafka consumer")
        
    def _consume(self) -> None:
        """Consume messages from Kafka."""
        def handle_message(message):
            try:
                signal = message.get('signal', {})
                logger.info(f"Received signal: predicted_price=${signal.get('predicted_price', 'N/A'):.2f}")
                self.signals.append(signal)
            except Exception as e:
                logger.exception(f"Error handling signal message: {e}")
        
        self.kafka_client.consume_messages(
            topics=[self.signals_topic],
            group_id=self.consumer_group,
            handler=handle_message,
            run_forever=lambda: self._running
        )


def generate_test_data(n_samples: int = 5) -> List[Dict[str, float]]:
    """Generate test treasury data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        List of treasury data dictionaries
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random features with realistic values
    base_volume = 1000.0
    base_interest_rate = 2.5
    base_inflation_rate = 2.0
    base_price = 100.0
    
    test_data = []
    for i in range(n_samples):
        # Add some randomness to the base values
        volume = base_volume + np.random.normal(0, 50)
        interest_rate = np.clip(base_interest_rate + np.random.normal(0, 0.2), 0.1, 7.0)
        inflation_rate = np.clip(base_inflation_rate + np.random.normal(0, 0.15), 0.2, 6.0)
        previous_month_price = base_price + np.random.normal(0, 2)
        
        test_data.append({
            'treasury_print_volume': float(volume),
            'interest_rate': float(interest_rate),
            'inflation_rate': float(inflation_rate),
            'previous_month_price': float(previous_month_price)
        })
        
        # Update base values for drift
        base_volume += np.random.normal(0, 5)
        base_interest_rate += np.random.normal(0, 0.05)
        base_inflation_rate += np.random.normal(0, 0.04)
        base_price += np.random.normal(0, 0.5)
    
    return test_data


def check_mlflow_server(tracking_uri: str, max_retries: int = 12) -> bool:
    """Check if MLflow server is running.
    
    Args:
        tracking_uri: MLflow tracking URI
        max_retries: Maximum number of retries
        
    Returns:
        True if server is running, False otherwise
    """
    for i in range(max_retries):
        try:
            response = requests.get(f"{tracking_uri}/api/2.0/mlflow-app/experiments/list")
            if response.status_code == 200:
                logger.info("MLflow server is running")
                return True
            else:
                logger.info(f"MLflow server not ready yet, status code: {response.status_code}")
        except requests.exceptions.RequestException:
            logger.info("MLflow server not ready yet, retrying...")
        
        time.sleep(5)
    
    logger.error("MLflow server is not running")
    return False


def check_kafka_server(bootstrap_servers: str, max_retries: int = 12) -> bool:
    """Check if Kafka is running.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        max_retries: Maximum number of retries
        
    Returns:
        True if server is running, False otherwise
    """
    # Using a simple producer to check connectivity
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    
    for i in range(max_retries):
        try:
            producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
            producer.close()
            logger.info("Kafka server is running")
            return True
        except NoBrokersAvailable:
            logger.info("Kafka server not ready yet, retrying...")
        except Exception as e:
            logger.info(f"Error connecting to Kafka: {e}")
        
        time.sleep(5)
    
    logger.error("Kafka server is not running")
    return False


def run_integration_test(args):
    """Run the integration test.
    
    Args:
        args: Command line arguments
    """
    # Make sure the directories exist
    os.makedirs(os.path.dirname(args.model_pickle_path), exist_ok=True)
    
    # Step 1: Train and pickle the model
    logger.info("==== Step 1: Training and pickling model ====")
    try:
        # Run the script to train and pickle the model
        cmd = [
            "python", 
            os.path.join(args.project_root, "models/train_and_pickle_model.py"),
            "--output-path", args.model_pickle_path,
            "--train-samples", str(args.train_samples)
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Model trained and pickled at: {args.model_pickle_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training and pickling model: {e}")
        return False

    # Step 2: Wait for MLflow server to be ready
    logger.info("\n==== Step 2: Checking MLflow server ====")
    if not check_mlflow_server(args.mlflow_tracking_uri):
        logger.error("MLflow server is not available. Please start it with podman-compose.")
        return False

    # Step 3: Register the model in MLflow
    logger.info("\n==== Step 3: Registering model in MLflow ====")
    try:
        # Run the script to register the model
        cmd = [
            "python", 
            os.path.join(args.project_root, "models/register_model_in_mlflow.py"),
            "--model-path", args.model_pickle_path,
            "--mlflow-tracking-uri", args.mlflow_tracking_uri,
            "--experiment-name", args.experiment_name,
            "--model-name", args.model_name
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Model registered in MLflow with name: {args.model_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error registering model in MLflow: {e}")
        return False

    # Step 4: Wait for Kafka to be ready
    logger.info("\n==== Step 4: Checking Kafka server ====")
    if not check_kafka_server(args.kafka_bootstrap_servers):
        logger.error("Kafka server is not available. Please start it with podman-compose.")
        return False

    # Step 5: Set up Kafka consumers and producer
    logger.info("\n==== Step 5: Setting up Kafka consumers and producer ====")
    
    # Create MLflow client
    mlflow_client = MLflowClient(args.mlflow_tracking_uri)
    
    # Create consumer that will make predictions using the MLflow model
    predictor = KafkaTreasuryDataConsumer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        treasury_topic=args.treasury_topic,
        signals_topic=args.signals_topic,
        consumer_group="test_predictor",
        mlflow_client=mlflow_client,
        model_name=args.model_name
    )
    
    # Create signal consumer to verify signals are produced
    signal_consumer = SignalConsumer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        signals_topic=args.signals_topic,
        consumer_group="test_signal_consumer"
    )
    
    # Create Treasury data producer
    producer = KafkaTreasuryDataProducer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        topic_name=args.treasury_topic
    )
    
    # Start consumers
    if not predictor.start():
        logger.error("Failed to start the predictor")
        return False
    
    signal_consumer.start()
    
    # Wait for consumers to initialize
    logger.info("Waiting for consumers to initialize...")
    time.sleep(3)
    
    # Step 6: Generate and send test data
    logger.info("\n==== Step 6: Sending test data ====")
    test_data = generate_test_data(args.test_samples)
    
    for i, data in enumerate(test_data):
        logger.info(f"Sending test data {i+1}/{len(test_data)}: {data}")
        producer.produce_data(data)
        time.sleep(1)  # Wait a bit between messages
    
    # Wait for all messages to be processed
    logger.info("Waiting for all messages to be processed...")
    time.sleep(5)
    
    # Step 7: Show results
    logger.info("\n==== Step 7: Test Results ====")
    logger.info(f"Total predictions made: {len(predictor.predictions)}")
    
    for i, result in enumerate(predictor.predictions):
        features = result['features']
        predicted_price = result['predicted_price']
        
        logger.info(f"Prediction {i+1}:")
        logger.info(f"  Features: treasury_volume={features.get('treasury_print_volume', 'N/A'):.2f}, "
                    f"interest_rate={features.get('interest_rate', 'N/A'):.2f}, "
                    f"inflation_rate={features.get('inflation_rate', 'N/A'):.2f}, "
                    f"previous_month_price={features.get('previous_month_price', 'N/A'):.2f}")
        logger.info(f"  Predicted price: ${predicted_price:.2f}")
    
    logger.info(f"Total signals received: {len(signal_consumer.signals)}")
    
    # Stop consumers
    predictor.stop()
    signal_consumer.stop()
    
    logger.info("\n==== Integration test complete! ====")
    
    # Test is successful if we made predictions and received signals
    if len(predictor.predictions) > 0 and len(signal_consumer.signals) > 0:
        logger.info("Test PASSED")
        return True
    else:
        logger.error("Test FAILED: No predictions or signals were produced")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run MLflow and Kafka integration test')
    
    parser.add_argument(
        '--project-root',
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help='Path to project root directory'
    )
    parser.add_argument(
        '--model-pickle-path',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "models/pickles/treasury_price_model.pkl"),
        help='Path to save the pickled model'
    )
    parser.add_argument(
        '--mlflow-tracking-uri',
        default='http://localhost:5000',
        help='MLflow tracking URI'
    )
    parser.add_argument(
        '--experiment-name',
        default='TreasuryPriceModel',
        help='Name of the MLflow experiment'
    )
    parser.add_argument(
        '--model-name',
        default='treasury-price-model',
        help='Name to register the model as in MLflow'
    )
    parser.add_argument(
        '--kafka-bootstrap-servers',
        default='localhost:29092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--treasury-topic',
        default='treasury.test.data',
        help='Topic for treasury data events'
    )
    parser.add_argument(
        '--signals-topic',
        default='signals.topic',
        help='Topic for signal events'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=100,
        help='Number of samples for model training'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=5,
        help='Number of test samples to generate'
    )
    
    args = parser.parse_args()
    
    success = run_integration_test(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()