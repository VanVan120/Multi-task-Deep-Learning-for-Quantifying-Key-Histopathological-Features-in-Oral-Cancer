import torch.nn as nn
from torchvision import models

class TVNTResNet(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        name = model_name.lower()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
