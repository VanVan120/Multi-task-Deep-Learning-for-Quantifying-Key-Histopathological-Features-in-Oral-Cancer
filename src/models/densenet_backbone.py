"""
DenseNet-169 Backbone for Multi-task Classification
==================================================

This module implements a DenseNet-169 backbone with multiple classification heads
for various oral cancer analysis tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from pathlib import Path

class DenseNetBackbone(nn.Module):
    """DenseNet-169 backbone with feature extraction capabilities"""
    
    def __init__(self, pretrained: bool = True, freeze_layers: int = 0):
        """
        Initialize DenseNet-169 backbone
        
        Args:
            pretrained: Whether to use pretrained weights
            freeze_layers: Number of initial layers to freeze (0-4)
        """
        super(DenseNetBackbone, self).__init__()
        
        # Load pretrained DenseNet-169
        self.densenet = models.densenet169(pretrained=pretrained)
        
        # Remove the classifier to use as feature extractor
        self.features = self.densenet.features
        self.num_features = self.densenet.classifier.in_features  # 1664 for DenseNet-169
        
        # Freeze early layers if specified
        self._freeze_layers(freeze_layers)
        
    def _freeze_layers(self, freeze_layers: int):
        """Freeze specified number of layer groups"""
        if freeze_layers <= 0:
            return
            
        layer_groups = [
            self.features.conv0,
            self.features.norm0, 
            self.features.relu0,
            self.features.pool0,
            self.features.denseblock1,
            self.features.transition1,
            self.features.denseblock2,
            self.features.transition2,
            self.features.denseblock3,
            self.features.transition3,
            self.features.denseblock4,
            self.features.norm5
        ]
        
        # Freeze specified number of layer groups
        for i in range(min(freeze_layers, len(layer_groups))):
            for param in layer_groups[i].parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, 1664, H/32, W/32]
        """
        features = self.features(x)
        return features

class MultiTaskClassificationHead(nn.Module):
    """Multi-task classification heads for DenseNet-169"""
    
    def __init__(self, backbone_features: int = 1664, tasks_config: Dict = None, dropout: float = 0.2):
        """
        Initialize multi-task classification heads
        
        Args:
            backbone_features: Number of features from backbone
            tasks_config: Dictionary with task configurations
            dropout: Dropout probability
        """
        super(MultiTaskClassificationHead, self).__init__()
        
        self.backbone_features = backbone_features
        self.dropout = dropout
        
        # Default tasks if not provided
        if tasks_config is None:
            tasks_config = {
                "TVNT": {"type": "binary_classification", "num_classes": 2},
                "POI": {"type": "multi_classification", "num_classes": 5},
                "PNI": {"type": "binary_classification", "num_classes": 2},
                "DOI": {"type": "regression", "output_dim": 1},
            }
        
        self.tasks_config = tasks_config
        self.task_heads = nn.ModuleDict()
        
        # Create task-specific heads
        for task_name, task_info in tasks_config.items():
            self.task_heads[task_name] = self._create_task_head(task_info)
    
    def _create_task_head(self, task_info: Dict) -> nn.Module:
        """Create task-specific head based on task type"""
        
        if task_info["type"] in ["binary_classification", "multi_classification"]:
            # Classification head
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.Linear(self.backbone_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(256, task_info["num_classes"])
            )
        
        elif task_info["type"] == "regression":
            # Regression head
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.Linear(self.backbone_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(256, task_info["output_dim"])
            )
        
        else:
            raise ValueError(f"Unsupported task type: {task_info['type']}")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads
        
        Args:
            features: Feature tensor from backbone [B, C, H, W]
            
        Returns:
            Dictionary of task outputs
        """
        outputs = {}
        
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)
        
        return outputs

class DenseNetMultiTask(nn.Module):
    """Complete DenseNet-169 Multi-task Model"""
    
    def __init__(
        self, 
        tasks_config: Dict = None,
        pretrained: bool = True,
        freeze_layers: int = 0,
        dropout: float = 0.2
    ):
        """
        Initialize complete multi-task model
        
        Args:
            tasks_config: Dictionary with task configurations
            pretrained: Whether to use pretrained backbone
            freeze_layers: Number of backbone layers to freeze
            dropout: Dropout probability
        """
        super(DenseNetMultiTask, self).__init__()
        
        # Backbone
        self.backbone = DenseNetBackbone(pretrained=pretrained, freeze_layers=freeze_layers)
        
        # Multi-task heads
        self.classification_heads = MultiTaskClassificationHead(
            backbone_features=self.backbone.num_features,
            tasks_config=tasks_config,
            dropout=dropout
        )
        
        self.tasks_config = self.classification_heads.tasks_config
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of task outputs
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Get predictions from all task heads
        outputs = self.classification_heads(features)
        
        return outputs
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate feature maps for visualization"""
        return self.backbone(x)

def create_densenet169_model(
    tasks_config: Optional[Dict] = None,
    pretrained: bool = True,
    freeze_layers: int = 0,
    dropout: float = 0.2
) -> DenseNetMultiTask:
    """
    Factory function to create DenseNet-169 multi-task model
    
    Args:
        tasks_config: Task configuration dictionary
        pretrained: Whether to use pretrained weights
        freeze_layers: Number of layers to freeze
        dropout: Dropout probability
        
    Returns:
        DenseNet multi-task model
    """
    model = DenseNetMultiTask(
        tasks_config=tasks_config,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        dropout=dropout
    )
    
    return model

# Model testing and utilities
def test_densenet_model():
    """Test the DenseNet model with dummy data"""
    
    # Create test model
    tasks_config = {
        "TVNT": {"type": "binary_classification", "num_classes": 2},
        "POI": {"type": "multi_classification", "num_classes": 5},
        "PNI": {"type": "binary_classification", "num_classes": 2},
        "DOI": {"type": "regression", "output_dim": 1},
    }
    
    model = create_densenet169_model(tasks_config=tasks_config)
    
    # Test with dummy input
    batch_size, channels, height, width = 4, 3, 512, 512
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("DenseNet-169 Multi-task Model Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Output shapes:")
    
    for task_name, output in outputs.items():
        print(f"  {task_name}: {output.shape}")
    
    return model, outputs

# Add this cell to see what's in your current densenet_backbone.py
print("🔍 Checking Current DenseNet File")
print("=" * 40)

project_root = Path.cwd().parent if Path.cwd().name == "Source Code" else Path.cwd()
densenet_file = project_root / "src" / "models" / "densenet_backbone.py"

# Let's see what's currently in the file
try:
    with open(densenet_file, 'r') as f:
        content = f.read()
    
    # Show first 50 lines to understand the structure
    lines = content.split('\n')
    print(f"File has {len(lines)} lines")
    print("\nFirst 50 lines:")
    for i, line in enumerate(lines[:50]):
        print(f"{i+1:3d}: {line}")
    
    # Look for class definitions and their __init__ methods
    print(f"\nClass definitions found:")
    for i, line in enumerate(lines):
        if 'class ' in line:
            print(f"Line {i+1}: {line.strip()}")
            # Look for __init__ method
            for j in range(i+1, min(i+20, len(lines))):
                if 'def __init__' in lines[j]:
                    print(f"Line {j+1}: {lines[j].strip()}")
                    break

except Exception as e:
    print(f"Error reading file: {e}")

if __name__ == "__main__":
    test_densenet_model()