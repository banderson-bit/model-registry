#!/usr/bin/env python3

import os
import sys
import logging
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.treasury_price_model import TreasuryPriceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pickle_to_tf_model(pickle_model_path, export_path):
    """
    Convert a scikit-learn pickle model to a TensorFlow SavedModel format
    that can be served with TensorFlow Serving.
    
    Args:
        pickle_model_path: Path to the pickled model
        export_path: Path to export the TF SavedModel
    """
    # Load the pickle model
    logger.info(f"Loading pickle model from {pickle_model_path}")
    with open(pickle_model_path, 'rb') as f:
        sklearn_model = pickle.load(f)
        
    logger.info("Creating TensorFlow model that wraps the scikit-learn model")
    
    # Create a simple Keras model that mimics our treasury price model structure
    # but we'll use the weights from our trained sklearn model
    
    class TreasuryPriceModelWrapper(tf.keras.Model):
        def __init__(self, sklearn_model):
            super(TreasuryPriceModelWrapper, self).__init__()
            self.sklearn_model = sklearn_model
            
        def call(self, inputs):
            # This function processes the input tensor and returns a prediction
            # by invoking the scikit-learn model's predict method
            
            # Our TF Serving expects batches, so handle that gracefully
            if isinstance(inputs, list):
                # If inputs are already a list of features, use as is
                features_list = inputs
            else:
                # Otherwise extract the tensor values to a list
                features_list = tf.nest.map_structure(lambda x: x.numpy(), inputs)
                
            predictions = []
            
            # Process each example in the batch
            for features in features_list:
                # Convert features to dictionary format expected by sklearn model
                feature_dict = {
                    'treasury_print_volume': float(features[0]),
                    'interest_rate': float(features[1]),
                    'inflation_rate': float(features[2]),
                    'previous_month_price': float(features[3])
                }
                prediction = self.sklearn_model.predict(feature_dict)
                predictions.append([prediction])
                
            return tf.convert_to_tensor(predictions)

    # Create a serving input function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 4], dtype=tf.float32, name='treasury_features')
    ])
    def serving_fn(treasury_features):
        # Wrapper function for TF Serving
        model_wrapper = TreasuryPriceModelWrapper(sklearn_model)
        predictions = model_wrapper(treasury_features)
        return {'price_prediction': predictions}
    
    # Create module with serving function
    logger.info("Creating TensorFlow module with serving function")
    serving_module = tf.Module()
    serving_module.serving_fn = serving_fn
    
    # Save the model in TensorFlow SavedModel format
    logger.info(f"Saving TensorFlow model to {export_path}")
    tf.saved_model.save(
        serving_module,
        export_path,
        signatures={'serving_default': serving_fn}
    )
    logger.info(f"Saved TensorFlow model to {export_path}")
    
    # Write a sample config for TF Serving
    config = """
model_config_list {
  config {
    name: "treasury_model"
    base_path: "%s"
    model_platform: "tensorflow"
    model_version_policy {
      all {}
    }
  }
}
""" % os.path.dirname(export_path)
    
    config_path = os.path.join(os.path.dirname(export_path), "models.config")
    with open(config_path, "w") as f:
        f.write(config)
        
    logger.info(f"Saved TF Serving config to {config_path}")
    
    # Generate Docker command to run TF Serving
    docker_cmd = f"""
# Run TensorFlow Serving with Docker
docker run -p 8501:8501 -p 8500:8500 \\
    --mount type=bind,source={os.path.dirname(export_path)},target=/models/treasury_model \\
    --mount type=bind,source={config_path},target=/models/models.config \\
    -e MODEL_NAME=treasury_model -t tensorflow/serving
"""
    logger.info("To serve the model with Docker, run:")
    logger.info(docker_cmd)
    
    return docker_cmd

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert pickle model to TF Serving format')
    parser.add_argument(
        '--pickle-model', 
        default='registry/storage/treasury_price_model_latest.pkl',
        help='Path to pickled model file'
    )
    parser.add_argument(
        '--export-path', 
        default='registry/tf_models/treasury_model/1',
        help='Path to export the TF SavedModel'
    )
    
    args = parser.parse_args()
    
    # Create export directory
    os.makedirs(os.path.dirname(args.export_path), exist_ok=True)
    
    # Convert the model
    pickle_to_tf_model(args.pickle_model, args.export_path)