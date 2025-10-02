"""
Models Module for Multi-task Deep Learning in Oral Cancer Analysis
================================================================

This module contains the model architectures for multi-task learning:
- DenseNet-169 backbone for classification tasks
- K-Net for segmentation tasks  
- Combined multi-task model
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # DenseNet Classification Models
    from densenet_backbone import (
        DenseNet169MultiTask,
        MultiTaskClassificationHead,
        create_densenet169_model
    )
    
    # K-Net Segmentation Models  
    from knet_segmentation import (
        KNetSegmentation,
        create_knet_model
    )
    
    # Combined Multi-task Model
    from multitask_model import (
        MultiTaskModel,
        create_multitask_model
    )
    
    __all__ = [
        # DenseNet models
        'DenseNet169MultiTask',
        'MultiTaskClassificationHead',
        'create_densenet169_model',
        
        # K-Net models
        'KNetSegmentation',
        'create_knet_model',
        
        # Combined models
        'MultiTaskModel',
        'create_multitask_model',
        'get_available_models',
        'create_model'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import some model components: {e}")
    # Provide fallback empty classes to prevent crashes
    __all__ = ['get_available_models', 'create_model']

def get_available_models():
    """Get list of available model architectures"""
    return {
        'classification': [
            'densenet169_multitask',
        ],
        'segmentation': [
            'knet',
        ],
        'multitask': [
            'densenet169_knet_combined',
        ]
    }

def create_model(model_name: str, **kwargs):
    """
    Factory function to create models by name
    
    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model
    """
    try:
        if model_name == 'densenet169_multitask':
            return create_densenet169_model(**kwargs)
        
        elif model_name == 'knet':
            return create_knet_model(**kwargs)
        
        elif model_name == 'densenet169_knet_combined':
            return create_multitask_model(**kwargs)
        
        else:
            available = get_available_models()
            raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    except NameError:
        raise ImportError("Model components not properly imported. Please check model files exist.")