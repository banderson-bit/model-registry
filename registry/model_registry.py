import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """A simple model registry that stores ML models as pickle files."""
    
    def __init__(self, storage_path: str):
        """Initialize the model registry with a storage path.
        
        Args:
            storage_path: Directory where models will be stored
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.models_index_path = os.path.join(storage_path, "models_index.json")
        self.models_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the models index from disk or create a new one."""
        if os.path.exists(self.models_index_path):
            with open(self.models_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save the models index to disk."""
        with open(self.models_index_path, 'w') as f:
            json.dump(self.models_index, f, indent=2)
    
    def register_model(self, model_name: str, model, version: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            model: The model object to store
            version: Optional version string (default: timestamp)
            metadata: Optional metadata about the model
            
        Returns:
            ID of the registered model
        """
        if not version:
            version = datetime.now().strftime('%Y%m%d%H%M%S')
            
        model_id = f"{model_name}_{version}"
        model_path = os.path.join(self.storage_path, f"{model_id}.pkl")
        
        # Save the model as a pickle file
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Update the index
        model_info = {
            "model_name": model_name,
            "version": version,
            "path": model_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.models_index[model_id] = model_info
        self._save_index()
        
        logger.info(f"Registered model {model_name} with version {version}")
        return model_id
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Get a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Optional version string (if None, returns latest version)
            
        Returns:
            The model object
        """
        if version:
            model_id = f"{model_name}_{version}"
            if model_id not in self.models_index:
                raise ValueError(f"Model {model_name} with version {version} not found")
            model_info = self.models_index[model_id]
        else:
            # Get the latest version
            relevant_models = [m for m_id, m in self.models_index.items() if m["model_name"] == model_name]
            if not relevant_models:
                raise ValueError(f"No models found with name {model_name}")
            
            model_info = sorted(relevant_models, key=lambda x: x["timestamp"], reverse=True)[0]
        
        # Load the model from disk
        with open(model_info["path"], 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models in the registry or filter by name.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of model information dictionaries
        """
        if model_name:
            return [m for m in self.models_index.values() if m["model_name"] == model_name]
        else:
            return list(self.models_index.values())
    
    def delete_model(self, model_name: str, version: str) -> None:
        """Delete a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model
        """
        model_id = f"{model_name}_{version}"
        if model_id not in self.models_index:
            raise ValueError(f"Model {model_name} with version {version} not found")
        
        model_info = self.models_index[model_id]
        
        # Delete the model file
        if os.path.exists(model_info["path"]):
            os.remove(model_info["path"])
        
        # Update the index
        del self.models_index[model_id]
        self._save_index()
        
        logger.info(f"Deleted model {model_name} with version {version}")