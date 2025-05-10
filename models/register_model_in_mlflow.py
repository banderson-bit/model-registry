#!/usr/bin/env python3

import os
import sys
import pickle
import mlflow
import logging
import argparse
import boto3
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def register_model_in_mlflow(model_path: str, 
                            mlflow_tracking_uri: str, 
                            s3_endpoint_url: str = None,
                            aws_access_key_id: str = None,
                            aws_secret_access_key: str = None,
                            experiment_name: str = "TreasuryPriceModel", 
                            model_name: str = "treasury-price-model") -> str:
    """Register a pickled model in MLflow.
    
    Args:
        model_path: Path to the pickled model
        mlflow_tracking_uri: MLflow tracking URI
        s3_endpoint_url: S3 endpoint URL for MinIO
        aws_access_key_id: AWS access key ID for MinIO
        aws_secret_access_key: AWS secret access key for MinIO
        experiment_name: Name of the MLflow experiment
        model_name: Name to register the model as
        
    Returns:
        Model version
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Configure MinIO as S3 storage for artifacts if provided
    if s3_endpoint_url and aws_access_key_id and aws_secret_access_key:
        # Set environment variables for MinIO
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        
        # Optional: Create boto3 client to test the connection
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name='us-east-1'  # Any region works for MinIO
            )
            
            # Check if bucket exists, create if needed
            try:
                s3_client.head_bucket(Bucket='mlflow')
                logger.info("MinIO mlflow bucket exists and is accessible")
            except Exception:
                logger.warning("MinIO mlflow bucket doesn't exist yet, will be created by MinIO init")
                
        except Exception as e:
            logger.warning(f"Unable to connect to MinIO: {e}")
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        logger.error(f"Error creating/getting experiment: {e}")
        raise
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Start an MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log model parameters (if any)
        mlflow.log_params({
            "feature_names": model.feature_names,
            "model_type": "LinearRegression"
        })
        
        # Generate some synthetic data for testing
        test_data, test_prices = model.generate_training_data(n_samples=20)
        test_results = model.evaluate(test_data, test_prices)
        
        # Log metrics from the model evaluation
        mlflow.log_metrics(test_results)
        
        # Create a sample input for the model signature
        sample_input = pd.DataFrame({
            'treasury_print_volume': [1000.0],
            'interest_rate': [2.5],
            'inflation_rate': [2.0],
            'previous_month_price': [100.0]
        })
        
        # Create a sample output for the model signature
        sample_output = pd.DataFrame({'price': [100.0]})
        
        # Create ModelWrapper class to match mlflow's expected interface
        class TreasuryPriceModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model
                
            def predict(self, context, model_input):
                if isinstance(model_input, pd.DataFrame):
                    results = []
                    for _, row in model_input.iterrows():
                        features = {
                            'treasury_print_volume': row.get('treasury_print_volume', 0),
                            'interest_rate': row.get('interest_rate', 0),
                            'inflation_rate': row.get('inflation_rate', 0),
                            'previous_month_price': row.get('previous_month_price', 0)
                        }
                        prediction = self.model.predict(features)
                        results.append(prediction)
                    return pd.Series(results)
                else:
                    raise ValueError("Input must be a pandas DataFrame")
        
        # Define model signature
        signature = mlflow.models.signature.infer_signature(
            sample_input, 
            sample_output
        )
        
        # Log the model to MLflow
        wrapped_model = TreasuryPriceModelWrapper(model)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            signature=signature,
            input_example=sample_input,
            registered_model_name=model_name
        )
        
        run_id = run.info.run_id
        
        # Get the latest version of the registered model
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        
        logger.info(f"Model registered in MLflow with name: {model_name}, version: {model_version}, run_id: {run_id}")
        
        return model_version

def main():
    parser = argparse.ArgumentParser(description='Register a pickled model in MLflow')
    parser.add_argument(
        '--model-path', 
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "models/pickles/treasury_price_model.pkl"),
        help='Path to the pickled model'
    )
    parser.add_argument(
        '--mlflow-tracking-uri', 
        default='http://localhost:5000',
        help='MLflow tracking URI'
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
    parser.add_argument(
        '--experiment-name', 
        default='TreasuryPriceModel',
        help='Name of the MLflow experiment'
    )
    parser.add_argument(
        '--model-name', 
        default='treasury-price-model',
        help='Name to register the model as'
    )
    
    args = parser.parse_args()
    
    # Verify that the model file exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    register_model_in_mlflow(
        model_path=args.model_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        s3_endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        experiment_name=args.experiment_name,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()