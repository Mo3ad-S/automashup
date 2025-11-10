"""
Model loader for GPU/CPU model loading.
"""
import torch
from typing import Optional, Dict, Any


class ModelLoader:
    """Model loader with GPU/CPU support."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.models = {}
    
    def load_model(self, model_name: str, model_class: type, 
                  model_path: Optional[str] = None,
                  **kwargs) -> Any:
        """
        Load model with GPU/CPU support.
        
        Args:
            model_name: Name of model
            model_class: Model class
            model_path: Path to model weights (optional)
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        # Initialize model
        model = model_class(**kwargs)
        
        # Load weights if provided
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Cache model
        self.models[model_name] = model
        
        return model
    
    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.use_gpu

