"""
Configuration loader for YAML config files.
"""
import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config.yaml in project root)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config.yaml in project root
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    
    if not os.path.exists(config_path):
        # Return default config
        return _default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config
    flattened = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                # Use subkey directly (e.g., 'sr' instead of 'audio_sr')
                flattened[subkey] = subvalue
                # Also keep nested key for backward compatibility
                flattened[f"{key}_{subkey}"] = subvalue
        else:
            flattened[key] = value
    
    # Map config keys to expected engine keys
    key_mapping = {
        'enabled': {
            'inpainting': 'enable_inpainting',
            'enhancement': 'enable_enhancement',
            'mixing': 'enable_mixing',
            'mastering': 'enable_mastering'
        }
    }
    
    # Apply key mappings
    for section, mappings in key_mapping.items():
        for subkey, mapped_key in mappings.items():
            if f"{subkey}_{section}" in flattened:
                flattened[mapped_key] = flattened[f"{subkey}_{section}"]
    
    return flattened


def _default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'sr': 44100,
        'target_lufs': -14.0,
        'use_gpu': True,
        'use_phase_vocoder': True,
        'preserve_formants': True,
        'crossfade_duration': 0.1,
        'inpainting_enabled': True,
        'inpainting_model': 'interpolation',
        'enhancement_enabled': True,
        'mixing_enabled': True,
        'mastering_enabled': True,
        'quality_threshold': 0.7
    }

