"""LResNet50E_IR — ResNet50-based backbone for face recognition."""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class LResNet50E_IR(nn.Module):
    """ResNet50-based model tailored for facial recognition tasks."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")

        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.fc(features)
