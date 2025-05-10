#!/usr/bin/env python3

import json
import logging
from typing import Dict, Any, List, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaClient:
    """Kafka client for producing and consuming messages using kafka-python library."""
    
    def __init__(self, bootstrap_servers: str):
        """Initialize Kafka client.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka bootstrap servers
        """
        self.bootstrap_servers = bootstrap_servers
        self._producer = None
        self._consumers = {}
        
    def get_producer(self) -> KafkaProducer:
        """Get or create a Kafka producer.
        
        Returns:
            KafkaProducer instance
        """
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
        return self._producer
    
    def produce_message(self, topic: str, value: Dict[str, Any], key: str = None) -> bool:
        """Produce a message to a Kafka topic.
        
        Args:
            topic: Kafka topic name
            value: Message value (will be serialized to JSON)
            key: Optional message key
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        producer = self.get_producer()
        try:
            future = producer.send(topic, value=value, key=key)
            producer.flush()
            future.get(timeout=10)  # Wait for message to be delivered
            logger.debug(f"Message sent to topic {topic}: {value}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message to topic {topic}: {e}")
            return False
    
    def get_consumer(self, topics: List[str], group_id: str) -> KafkaConsumer:
        """Get or create a Kafka consumer for specified topics.
        
        Args:
            topics: List of Kafka topic names
            group_id: Consumer group ID
            
        Returns:
            KafkaConsumer instance
        """
        consumer_key = f"{group_id}-{'-'.join(sorted(topics))}"
        if consumer_key not in self._consumers:
            self._consumers[consumer_key] = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
        return self._consumers[consumer_key]
    
    def consume_messages(self, topics: List[str], group_id: str, handler: Callable[[Dict[str, Any]], None], 
                         timeout_ms: int = 1000, run_forever: bool = True) -> None:
        """Consume messages from Kafka topics and process them with a handler function.
        
        Args:
            topics: List of Kafka topic names
            group_id: Consumer group ID
            handler: Callback function to process messages
            timeout_ms: Poll timeout in milliseconds
            run_forever: Whether to run consumer in an infinite loop
        """
        consumer = self.get_consumer(topics, group_id)
        
        try:
            while True:
                messages = consumer.poll(timeout_ms=timeout_ms)
                for topic_partition, message_list in messages.items():
                    for message in message_list:
                        try:
                            handler(message.value)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                
                if not run_forever:
                    break
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        finally:
            consumer.close()
            
    def close(self):
        """Close all Kafka connections."""
        if self._producer:
            self._producer.close()
            self._producer = None
            
        for consumer in self._consumers.values():
            consumer.close()
        self._consumers = {}