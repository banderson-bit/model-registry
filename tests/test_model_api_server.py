#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import argparse
import threading
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
import numpy as np
import random
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure boto/S3 to use MinIO
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Create Flask application
app = Flask(__name__)

# Global variable to store the loaded model
model = None


class MLflowModelLoader:
    """Helper class to load MLflow models."""
    
    @staticmethod
    def load_model(model_name: str, stage: str = "None", mlflow_tracking_uri: str = "http://localhost:5000",
                  s3_endpoint_url: str = "http://localhost:9000",
                  aws_access_key_id: str = "minio",
                  aws_secret_access_key: str = "minio123"):
        """Load a model from MLflow.
        
        Args:
            model_name: Name of the model
            stage: Model stage (e.g., "None", "Staging", "Production")
            mlflow_tracking_uri: MLflow tracking URI
            s3_endpoint_url: S3 endpoint URL for MinIO
            aws_access_key_id: AWS access key ID for MinIO
            aws_secret_access_key: AWS secret access key for MinIO
            
        Returns:
            Loaded MLflow model
        """
        # Set environment variables for MinIO
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            # Load the model
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model: {model_name}, stage: {stage}")
            return loaded_model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Get request data
        content = request.json
        
        if not content:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert input to DataFrame for prediction
        features = pd.DataFrame([{
            'treasury_print_volume': content.get('treasury_print_volume', 0),
            'interest_rate': content.get('interest_rate', 0),
            'inflation_rate': content.get('inflation_rate', 0),
            'previous_month_price': content.get('previous_month_price', 0)
        }])
        
        # Make prediction
        prediction = model.predict(features)
        predicted_price = float(prediction[0])
        
        # Log the prediction
        logger.info(f"Prediction for input {content}: ${predicted_price:.2f}")
        
        return jsonify({
            "predicted_price": predicted_price,
            "input_features": content
        }), 200
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


def generate_test_data(n_samples: int = 5) -> list:
    """Generate test treasury data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        List of treasury data samples
    """
    np.random.seed(int(time.time()))  # Random seed based on time
    
    base_volume = 1000.0
    base_interest_rate = 2.5
    base_inflation_rate = 2.0
    base_price = 100.0
    
    test_data = []
    for i in range(n_samples):
        # Add randomness to make each sample unique
        volume = base_volume + np.random.normal(0, 50)
        interest_rate = np.clip(base_interest_rate + np.random.normal(0, 0.3), 0.1, 7.0)
        inflation_rate = np.clip(base_inflation_rate + np.random.normal(0, 0.2), 0.1, 6.0)
        previous_price = base_price + np.random.normal(0, 3)
        
        data = {
            'treasury_print_volume': float(volume),
            'interest_rate': float(interest_rate),
            'inflation_rate': float(inflation_rate),
            'previous_month_price': float(previous_price)
        }
        test_data.append(data)
        
        # Adjust base values for trend
        base_volume += np.random.normal(0, 10)
        base_interest_rate += np.random.normal(0, 0.05)
        base_inflation_rate += np.random.normal(0, 0.03)
        base_price += np.random.normal(0, 1)
    
    return test_data


def send_test_requests(api_url: str, n_requests: int = 5, delay: float = 1.0):
    """Send test requests to the API.
    
    Args:
        api_url: URL of the API
        n_requests: Number of requests to send
        delay: Delay between requests in seconds
    """
    import requests
    
    # Generate test data
    test_data = generate_test_data(n_requests)
    
    for i, data in enumerate(test_data):
        try:
            logger.info(f"\nSending test request {i+1}/{n_requests}:")
            logger.info(f"Input: treasury_volume={data['treasury_print_volume']:.2f}, "
                      f"interest_rate={data['interest_rate']:.2f}, "
                      f"inflation_rate={data['inflation_rate']:.2f}, "
                      f"previous_price={data['previous_month_price']:.2f}")
            
            # Send request to API
            response = requests.post(f"{api_url}/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"API Response: Predicted price = ${result['predicted_price']:.2f}")
            else:
                logger.error(f"API Error: {response.status_code}, {response.text}")
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        # Wait before sending next request
        if i < n_requests - 1:
            time.sleep(delay)


def run_test(args):
    """Run the model API test.
    
    Args:
        args: Command line arguments
    """
    global model
    
    # Load model from MLflow
    logger.info(f"Loading model '{args.model_name}' from MLflow at {args.mlflow_tracking_uri}...")
    model = MLflowModelLoader.load_model(
        model_name=args.model_name,
        stage=args.model_stage,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        s3_endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key
    )
    
    if model is None:
        logger.error("Failed to load the model from MLflow. Exiting.")
        return False
    
    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=lambda: app.run(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False
        )
    )
    api_thread.daemon = True
    api_thread.start()
    
    logger.info(f"API server started at http://{args.host}:{args.port}")
    
    # Wait for server to start
    time.sleep(2)
    
    # Send test requests to the API
    api_url = f"http://{args.host}:{args.port}"
    
    logger.info(f"\n{'='*50}")
    logger.info(f"SENDING TEST REQUESTS TO MODEL API")
    logger.info(f"{'='*50}")
    
    send_test_requests(
        api_url=api_url,
        n_requests=args.n_requests,
        delay=args.delay
    )
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST COMPLETE!")
    logger.info(f"{'='*50}")
    
    # Keep the server running for a short while to ensure all logs are displayed
    time.sleep(2)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test the Treasury Price Model API')
    
    parser.add_argument(
        '--mlflow-tracking-uri',
        default='http://localhost:5000',
        help='MLflow tracking URI'
    )
    parser.add_argument(
        '--model-name',
        default='treasury-price-model',
        help='Name of the MLflow model'
    )
    parser.add_argument(
        '--model-stage',
        default='None',
        choices=['None', 'Staging', 'Production', 'Archived'],
        help='Stage of the MLflow model'
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host for the API server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the API server'
    )
    parser.add_argument(
        '--n-requests',
        type=int,
        default=5,
        help='Number of test requests to send'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between test requests in seconds'
    )
    parser.add_argument(
        '--s3-endpoint-url',
        default='http://localhost:9000',
        help='S3 endpoint URL for MinIO'
    )
    parser.add_argument(
        '--aws-access-key-id',
        default='minio',
        help='AWS access key ID for MinIO'
    )
    parser.add_argument(
        '--aws-secret-access-key',
        default='minio123',
        help='AWS secret access key for MinIO'
    )
    
    args = parser.parse_args()
    
    # Configure boto/S3 to use MinIO
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
    
    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()