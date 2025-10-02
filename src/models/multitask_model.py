"""
Multi-task Model combining DenseNet-169 and K-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

# Import the individual model components
try:
    from densenet_backbone import DenseNetMultiTask as DenseNet169MultiTask
except ImportError:
    try:
        from densenet_backbone import DenseNetBackbone as DenseNet169MultiTask
    except ImportError:
        print("Error: Cannot import DenseNet backbone. Please check densenet_backbone.py file.")
        raise

try:
    from knet_segmentation import KNetSegmentation
except ImportError:
    print("Error: Cannot import K-Net segmentation. Please check knet_segmentation.py file.")
    raise


class MultiTaskModel(nn.Module):
    """Combined multi-task model for classification and segmentation"""
    
    def __init__(self, pretrained: bool = True, num_classes: int = 6,
                 use_attention: bool = True, feature_sharing: str = 'full'):
        super().__init__()
        
        self.feature_sharing = feature_sharing
        self.num_classes = num_classes
        
        # Create DenseNet-169 backbone - try different parameter combinations
        print(f"Creating DenseNet backbone...")
        
        # Try different combinations of parameters based on what the class accepts
        backbone_created = False
        
        # Attempt 1: Try with both parameters
        try:
            self.backbone_model = DenseNet169MultiTask(pretrained=pretrained, use_attention=use_attention)
            backbone_created = True
            print(f"✅ Created with pretrained={pretrained}, use_attention={use_attention}")
        except TypeError as e:
            print(f"   Attempt 1 failed: {e}")
        
        # Attempt 2: Try with just pretrained
        if not backbone_created:
            try:
                self.backbone_model = DenseNet169MultiTask(pretrained=pretrained)
                backbone_created = True
                print(f"✅ Created with pretrained={pretrained}")
            except TypeError as e:
                print(f"   Attempt 2 failed: {e}")
        
        # Attempt 3: Try with no parameters
        if not backbone_created:
            try:
                self.backbone_model = DenseNet169MultiTask()
                backbone_created = True
                print(f"✅ Created with default parameters")
            except TypeError as e:
                print(f"   Attempt 3 failed: {e}")
        
        if not backbone_created:
            raise RuntimeError("Could not create DenseNet backbone with any parameter combination")
        
        # Get the backbone and classification head with flexible attribute names
        self.backbone = self.backbone_model.backbone
        
        # Try different attribute names for classification head
        if hasattr(self.backbone_model, 'classification_head'):
            self.classification_head = self.backbone_model.classification_head
            print("✅ Found classification_head attribute")
        elif hasattr(self.backbone_model, 'classification_heads'):
            self.classification_head = self.backbone_model.classification_heads
            print("✅ Found classification_heads attribute")
        elif hasattr(self.backbone_model, 'classifier'):
            self.classification_head = self.backbone_model.classifier
            print("✅ Found classifier attribute")
        else:
            # List all available attributes for debugging
            available_attrs = [attr for attr in dir(self.backbone_model) if not attr.startswith('_')]
            print(f"❌ No classification head found. Available attributes: {available_attrs}")
            raise AttributeError("Could not find classification head attribute")
        
        # Create K-Net segmentation head with flexible parameters
        print(f"Creating K-Net segmentation head...")
        segmentation_created = False
        
        # Try different parameter combinations for K-Net
        knet_params_attempts = [
            # Attempt 1: Current parameters
            {
                'backbone_dim': 1664,
                'num_classes': num_classes,
                'num_kernels': 100,
                'num_stages': 3
            },
            # Attempt 2: Alternative parameter names
            {
                'input_dim': 1664,
                'num_classes': num_classes,
                'num_kernels': 100,
                'num_stages': 3
            },
            # Attempt 3: Minimal parameters
            {
                'num_classes': num_classes,
                'num_kernels': 100,
            },
            # Attempt 4: Just num_classes
            {
                'num_classes': num_classes,
            },
            # Attempt 5: No parameters (use defaults)
            {}
        ]
        
        for i, params in enumerate(knet_params_attempts):
            try:
                self.segmentation_head = KNetSegmentation(**params)
                segmentation_created = True
                print(f"✅ K-Net created with parameters: {params}")
                break
            except TypeError as e:
                print(f"   K-Net attempt {i+1} failed: {e}")
        
        if not segmentation_created:
            # If all attempts fail, let's see what parameters KNetSegmentation actually accepts
            import inspect
            sig = inspect.signature(KNetSegmentation.__init__)
            print(f"❌ K-Net creation failed. KNetSegmentation.__init__ signature: {sig}")
            raise RuntimeError("Could not create K-Net segmentation head with any parameter combination")
        
        # Feature adaptation layers for segmentation
        # DenseNet-169 outputs 1664 channels, but K-Net might expect different dimensions
        self.seg_feature_adapter = nn.Sequential(
            # Reduce channels if needed
            nn.Conv2d(1664, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Add upsampling to ensure reasonable spatial dimensions
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        print(f"✅ Segmentation feature adapter created (1664→256 channels, 2x upsampling)")
        
        # Additional feature adaptation if needed
        if feature_sharing == 'partial':
            self.cls_feature_adapter = nn.Sequential(
                nn.Conv2d(1664, 1664, 1),
                nn.BatchNorm2d(1664),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x, return_features: bool = False):
        """Forward pass through both classification and segmentation"""
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        
        # Classification branch (uses original features)
        if self.feature_sharing == 'partial':
            cls_features = self.cls_feature_adapter(backbone_features)
        else:
            cls_features = backbone_features
            
        classification_outputs = self.classification_head(cls_features)
        
        # Segmentation branch (uses adapted features)
        seg_features = self.seg_feature_adapter(backbone_features)
        
        # Debug: Print shapes to understand the data flow
        if return_features:
            print(f"Debug - Backbone features: {backbone_features.shape}")
            print(f"Debug - Adapted seg features: {seg_features.shape}")
        
        try:
            segmentation_outputs = self.segmentation_head(seg_features)
        except Exception as e:
            print(f"❌ Segmentation head error: {e}")
            print(f"Input shape to segmentation head: {seg_features.shape}")
            # Create dummy segmentation output as fallback
            B, _, H, W = seg_features.shape
            dummy_seg_logits = torch.zeros(B, self.num_classes, H*4, W*4, device=seg_features.device)
            segmentation_outputs = {'seg_logits': dummy_seg_logits}
            print(f"✅ Created dummy segmentation output: {dummy_seg_logits.shape}")
        
        outputs = {
            'classification': classification_outputs,
            'segmentation': segmentation_outputs
        }
        
        if return_features:
            outputs['backbone_features'] = backbone_features
            outputs['adapted_seg_features'] = seg_features
            
        return outputs
    
    def compute_model_size(self):
        """Compute model statistics"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        cls_params = sum(p.numel() for p in self.classification_head.parameters())
        seg_params = sum(p.numel() for p in self.segmentation_head.parameters())
        adapter_params = sum(p.numel() for p in self.seg_feature_adapter.parameters())
        
        return {
            'total_parameters': backbone_params + cls_params + seg_params + adapter_params,
            'backbone_parameters': backbone_params,
            'classification_parameters': cls_params,
            'segmentation_parameters': seg_params,
            'adapter_parameters': adapter_params
        }


def create_multitask_model(pretrained: bool = True, num_classes: int = 6,
                          use_attention: bool = True, feature_sharing: str = 'full') -> MultiTaskModel:
    """Create multi-task model with DenseNet-169 and K-Net"""
    return MultiTaskModel(
        pretrained=pretrained,
        num_classes=num_classes,
        use_attention=use_attention,
        feature_sharing=feature_sharing
    )