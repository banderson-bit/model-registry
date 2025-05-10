import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from .model_registry import ModelRegistry
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.kafka_client import KafkaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaModelRegistryService:
    """A service that handles model registrations and notifications via Kafka."""
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        kafka_bootstrap_servers: str = "localhost:29092",
        topic_name: str = "model.registry",
    ):
        """Initialize the service.
        
        Args:
            model_registry: The model registry instance
            kafka_bootstrap_servers: Kafka bootstrap servers
            topic_name: Name of the Kafka topic for model registry events
        """
        self.model_registry = model_registry
        self.topic_name = topic_name
        self.kafka_client = KafkaClient(kafka_bootstrap_servers)
    
    def register_model(
        self, model_name: str, model, version: Optional[str] = None, metadata: Dict[str, Any] = None
    ) -> str:
        """Register a model and publish an event.
        
        Args:
            model_name: Name of the model
            model: Model object
            version: Optional version string
            metadata: Optional metadata
            
        Returns:
            ID of the registered model
        """
        model_id = self.model_registry.register_model(model_name, model, version, metadata)
        
        # Notify subscribers
        payload = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version or model_id.split('_')[-1],
            'metadata': metadata or {}
        }
        self._produce_event('model_registered', payload)
        return model_id
    
    def _produce_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Produce an event to the Kafka topic.
        
        Args:
            event_type: Type of event (e.g., 'model_registered', 'model_deleted')
            payload: Event payload data
        """
        message = {
            'event_type': event_type,
            'timestamp': time.time(),
            'payload': payload
        }
        
        key = payload.get('model_id', '')
        success = self.kafka_client.produce_message(
            topic=self.topic_name,
            value=message,
            key=key
        )
        
        if not success:
            logger.error(f"Failed to produce {event_type} event for model {key}")
        
    def delete_model(self, model_name: str, version: str) -> None:
        """Delete a model and publish an event.
        
        Args:
            model_name: Name of the model
            version: Version of the model
        """
        self.model_registry.delete_model(model_name, version)
        
        # Notify subscribers
        model_id = f"{model_name}_{version}"
        payload = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version
        }
        self._produce_event('model_deleted', payload)


class ModelRegistrySubscriber:
    """A subscriber that listens for model registry events."""
    
    def __init__(
        self, 
        kafka_bootstrap_servers: str = "localhost:29092",
        topic_name: str = "model.registry",
        consumer_group: str = "model_registry_subscriber"
    ):
        """Initialize the subscriber.
        
        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            topic_name: Name of the Kafka topic for model registry events
            consumer_group: Consumer group ID
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topic_name = topic_name
        self.consumer_group = consumer_group
        
        self.kafka_client = KafkaClient(kafka_bootstrap_servers)
        
        self._running = False
        self._consumer_thread = None
        self._event_handlers = {
            'model_registered': [],
            'model_deleted': []
        }
    
    def on_model_registered(self, handler):
        """Register a handler for model registered events.
        
        Args:
            handler: A function that takes a payload dict as an argument
        """
        self._event_handlers['model_registered'].append(handler)
        return handler
    
    def on_model_deleted(self, handler):
        """Register a handler for model deleted events.
        
        Args:
            handler: A function that takes a payload dict as an argument
        """
        self._event_handlers['model_deleted'].append(handler)
        return handler
        
    def start(self):
        """Start consuming events."""
        if self._running:
            return
            
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consume_events)
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        
    def stop(self):
        """Stop consuming events."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5.0)
            
    def _consume_events(self):
        """Consume events from the topic and dispatch them to handlers."""
        def handle_message(message):
            try:
                event_type = message.get('event_type')
                payload = message.get('payload', {})
                
                # Dispatch to handlers
                handlers = self._event_handlers.get(event_type, [])
                for handler in handlers:
                    try:
                        handler(payload)
                    except Exception as e:
                        logger.exception(f"Error in handler: {e}")
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
        
        topics = [self.topic_name]
        self.kafka_client.consume_messages(
            topics=topics,
            group_id=self.consumer_group,
            handler=handle_message,
            run_forever=self._running
        )