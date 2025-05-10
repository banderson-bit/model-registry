import os
import sys
import json
import logging
import threading
import time
import pickle
from typing import Dict, Any, Optional, List, Callable

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registry.registry_service import ModelRegistrySubscriber
from app.kafka_client import KafkaClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TreasuryDataProducer:
    """Producer for US Treasury data events."""
    
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:29092",
        topic_name: str = "treasury.data"
    ):
        """Initialize the US Treasury data producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_name: Name of the topic for treasury data events
        """
        self.topic_name = topic_name
        self.kafka_client = KafkaClient(bootstrap_servers)
        
    def produce_treasury_data(self, data: Dict[str, Any]):
        """Produce a treasury data event.
        
        Args:
            data: Treasury data event with features
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
            logger.info(f"Produced treasury data event: {data}")
        else:
            logger.error(f"Failed to produce treasury data event: {data}")


class SignalProducer:
    """Producer for price prediction signals."""
    
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:29092",
        topic_name: str = "signals.topic"
    ):
        """Initialize the signal producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_name: Name of the topic for signal events
        """
        self.topic_name = topic_name
        self.kafka_client = KafkaClient(bootstrap_servers)
        
    def produce_signal(self, signal_data: Dict[str, Any]):
        """Produce a signal event.
        
        Args:
            signal_data: Signal data with prediction
        """
        message = {
            'timestamp': time.time(),
            'signal': signal_data
        }
        
        success = self.kafka_client.produce_message(
            self.topic_name, 
            value=message,
            key=str(message['timestamp'])
        )
        
        if success:
            logger.info(f"Produced signal event: {signal_data}")
        else:
            logger.error(f"Failed to produce signal event: {signal_data}")


class PredictorApplication:
    """Main application that consumes treasury data and produces price predictions."""
    
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:29092",
        treasury_topic: str = "treasury.data",
        signals_topic: str = "signals.topic",
        registry_topic: str = "model.registry",
        consumer_group: str = "predictor_app"
    ):
        """Initialize the predictor application.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            treasury_topic: Topic for treasury data events
            signals_topic: Topic for signal events
            registry_topic: Topic for model registry events
            consumer_group: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.treasury_topic = treasury_topic
        self.signals_topic = signals_topic
        self.registry_topic = registry_topic
        self.consumer_group = consumer_group
        
        self.kafka_client = KafkaClient(bootstrap_servers)
        
        self.model_cache = {}  # Cache for models
        self.latest_model_versions = {}  # Track latest model version by name
        
        # Initialize components
        self.signal_producer = SignalProducer(bootstrap_servers, signals_topic)
        
        # Initialize registry subscriber
        self.registry_subscriber = ModelRegistrySubscriber(
            bootstrap_servers,
            registry_topic,
            f"{consumer_group}_registry"
        )
        
        # Set up model update handlers
        self.registry_subscriber.on_model_registered(self.handle_model_registered)
        
        # Start registry subscriber
        self.registry_subscriber.start()
        
        self._running = False
        self._consumer_thread = None
        
    def handle_model_registered(self, payload: Dict[str, Any]):
        """Handle a model registered event.
        
        Args:
            payload: Event payload with model information
        """
        model_id = payload.get('model_id')
        model_name = payload.get('model_name')
        version = payload.get('version')
        metadata = payload.get('metadata', {})
        
        logger.info(f"New model version available: {model_name} (version: {version})")
        
        # Track latest version
        self.latest_model_versions[model_name] = version
        
        # We don't download the model here, it will be fetched from the registry when needed
        
    def get_model(self, model_name: str, registry_path: str) -> Any:
        """Get a model from the model cache or load it from the registry.
        
        Args:
            model_name: Name of the model
            registry_path: Path to the registry storage
            
        Returns:
            Model object
        """
        version = self.latest_model_versions.get(model_name)
        if not version:
            logger.error(f"No version found for model: {model_name}")
            return None
            
        model_id = f"{model_name}_{version}"
        
        # Check if model is in cache
        if model_id in self.model_cache:
            return self.model_cache[model_id]
            
        # Load model from registry
        try:
            model_path = os.path.join(registry_path, f"{model_id}.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Cache the model
            self.model_cache[model_id] = model
            logger.info(f"Loaded model from registry: {model_id}")
            return model
            
        except Exception as e:
            logger.exception(f"Error loading model {model_id}: {e}")
            return None
        
    def start(self, registry_path: str):
        """Start consuming treasury data events.
        
        Args:
            registry_path: Path to the registry storage
        """
        if self._running:
            return
            
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_treasury_data, 
            args=(registry_path,)
        )
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        logger.info("Started predictor application")
        
    def stop(self):
        """Stop consuming events."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5.0)
        self.registry_subscriber.stop()
        logger.info("Stopped predictor application")
        
    def _consume_treasury_data(self, registry_path: str):
        """Consume treasury data events and produce signals.
        
        Args:
            registry_path: Path to the registry storage
        """
        def handle_message(message):
            try:
                treasury_data = message.get('data', {})
                # Process the treasury data
                self._process_treasury_data(treasury_data, registry_path)
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
        
        topics = [self.treasury_topic]
        self.kafka_client.consume_messages(
            topics=topics,
            group_id=self.consumer_group,
            handler=handle_message,
            run_forever=self._running
        )
            
    def _process_treasury_data(self, treasury_data: Dict[str, Any], registry_path: str):
        """Process treasury data and produce signals.
        
        Args:
            treasury_data: Treasury data with features
            registry_path: Path to the registry storage
        """
        # Get the price prediction model
        model_name = "treasury_price_model"  # This should match the name used when registering the model
        model = self.get_model(model_name, registry_path)
        
        if model is None:
            logger.warning(f"No model available for {model_name}, skipping prediction")
            return
            
        try:
            # Make prediction
            prediction = model.predict(treasury_data)
            
            # Create signal
            signal_data = {
                'model_name': model_name,
                'model_version': self.latest_model_versions.get(model_name, 'unknown'),
                'features': treasury_data,
                'predicted_price': prediction,
                'timestamp': time.time()
            }
            
            # Produce signal
            self.signal_producer.produce_signal(signal_data)
            
        except Exception as e:
            logger.exception(f"Error making prediction: {e}")


class ApplicationConsumer:
    """Consumer application that processes signals."""
    
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:29092",
        signals_topic: str = "signals.topic",
        consumer_group: str = "application_consumer"
    ):
        """Initialize the application consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            signals_topic: Topic for signal events
            consumer_group: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.signals_topic = signals_topic
        self.consumer_group = consumer_group
        
        self.kafka_client = KafkaClient(bootstrap_servers)
        
        self.signal_handlers = []
        
        self._running = False
        self._consumer_thread = None
        
    def register_signal_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for signal events.
        
        Args:
            handler: Function that takes a signal data dict as an argument
        """
        self.signal_handlers.append(handler)
        
    def start(self):
        """Start consuming signal events."""
        if self._running:
            return
            
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consume_signals)
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        logger.info("Started application consumer")
        
    def stop(self):
        """Stop consuming events."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5.0)
        logger.info("Stopped application consumer")
        
    def _consume_signals(self):
        """Consume signal events and dispatch them to handlers."""
        def handle_message(message):
            try:
                signal_data = message.get('signal', {})
                # Dispatch to handlers
                for handler in self.signal_handlers:
                    try:
                        handler(signal_data)
                    except Exception as e:
                        logger.exception(f"Error in signal handler: {e}")
            except Exception as e:
                logger.exception(f"Error processing signal message: {e}")
        
        topics = [self.signals_topic]
        self.kafka_client.consume_messages(
            topics=topics,
            group_id=self.consumer_group,
            handler=handle_message,
            run_forever=self._running
        )