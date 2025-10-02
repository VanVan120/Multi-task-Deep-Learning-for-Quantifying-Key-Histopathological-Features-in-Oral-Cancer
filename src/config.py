"""
Configuration settings for Multi-task Deep Learning Oral Cancer Analysis
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Backbone settings
    default_backbone: str = 'densenet169'
    pretrained: bool = True
    
    # Classification tasks
    classification_tasks: Dict[str, Dict] = field(default_factory=lambda: {
        'tvnt': {'type': 'binary_classification', 'classes': 2, 'loss': 'cross_entropy'},
        'poi': {'type': 'multiclass_classification', 'classes': 5, 'loss': 'cross_entropy'},
        'pni': {'type': 'binary_classification', 'classes': 2, 'loss': 'cross_entropy'},
        'doi': {'type': 'regression', 'classes': 1, 'loss': 'mse'}
    })
    
    # Segmentation settings
    segmentation_tasks: List[str] = field(default_factory=lambda: [
        'background', 'tumor', 'stroma', 'necrosis', 'inflammation', 'normal'
    ])
    
    # K-Net specific settings
    knet_num_kernels: int = 100
    knet_num_stages: int = 3
    
    # Multi-task learning settings
    feature_sharing: str = 'full'  # 'full', 'partial', 'none'
    use_task_attention: bool = True
    
    # Task weights for loss balancing
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'tvnt': 1.0,
        'poi': 1.0, 
        'pni': 1.0,
        'doi': 1.0,
        'segmentation': 1.0
    })


@dataclass 
class TrainingConfig:
    """Training configuration parameters"""
    # Basic training settings
    batch_size: int = 16  # Optimized for RTX 4050
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Optimization settings
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # Mixed precision and memory optimization
    use_amp: bool = True  # Automatic Mixed Precision for RTX 4050
    gradient_checkpointing: bool = True
    max_gradient_norm: float = 1.0
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Validation and checkpointing
    val_frequency: int = 5
    save_frequency: int = 10
    early_stopping_patience: int = 15


@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Image settings
    input_size: int = 512  # Patch size - optimized for memory
    num_channels: int = 3
    
    # Augmentation settings
    use_augmentation: bool = True
    rotation_range: int = 90
    horizontal_flip: bool = True
    vertical_flip: bool = True
    color_jitter: bool = True
    
    # Normalization (ImageNet stats)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Class balance handling
    use_class_weights: bool = True
    oversample_minority: bool = False


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Device settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    gpu_memory_fraction: float = 0.8  # For RTX 4050 memory management
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging and monitoring
    log_level: str = 'INFO'
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Paths
    project_root: Optional[Path] = None
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def get_config() -> Config:
    """Get default configuration"""
    config = Config()
    
    # Auto-detect device
    if config.system.device == 'auto':
        config.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set up paths
    if config.system.project_root is None:
        config.system.project_root = Path.cwd().parent if Path.cwd().name == "Source Code" else Path.cwd()
        config.system.data_dir = config.system.project_root / "data"
        config.system.output_dir = config.system.project_root / "outputs"
        config.system.checkpoint_dir = config.system.project_root / "checkpoints"
    
    return config


def print_config_summary():
    """Print configuration summary"""
    config = get_config()
    
    print("🔧 Configuration Summary:")
    print("=" * 50)
    print(f"Model: DenseNet-169 + K-Net Multi-task")
    print(f"Device: {config.system.device}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Input Size: {config.data.input_size}x{config.data.input_size}")
    print(f"Mixed Precision: {config.training.use_amp}")
    print(f"Classification Tasks: {len(config.model.classification_tasks)}")
    print(f"Segmentation Classes: {len(config.model.segmentation_tasks)}")
    print(f"Feature Sharing: {config.model.feature_sharing}")
    print(f"Task Attention: {config.model.use_task_attention}")


# Add this cell to test the config file directly
print("🔍 Testing Config File Directly")
print("=" * 40)

import sys
from pathlib import Path

# Setup path
project_root = Path.cwd().parent if Path.cwd().name == "Source Code" else Path.cwd()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    # Import and test config
    from config import get_config, Config, ModelConfig
    
    print("✅ Config classes imported successfully")
    print(f"Config class type: {Config}")
    print(f"ModelConfig class type: {ModelConfig}")
    
    # Create config object
    config_obj = get_config()
    print(f"Config object type: {type(config_obj)}")
    print(f"Is Config instance: {isinstance(config_obj, Config)}")
    
    # Test attributes
    print(f"✅ System device: {config_obj.system.device}")
    print(f"✅ Model backbone: {config_obj.model.default_backbone}")
    print(f"✅ Training batch size: {config_obj.training.batch_size}")
    print(f"✅ Data input size: {config_obj.data.input_size}")
    
    print("\n🎉 Config is working correctly!")
    
except Exception as e:
    print(f"❌ Config test failed: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
    print_config_summary()