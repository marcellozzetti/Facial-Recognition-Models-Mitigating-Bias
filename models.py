import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet101_Weights, ResNet50_Weights

# Definindo a camada ArcFace
class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        # Normalize logits to get the unit vectors
        logits = F.normalize(logits, dim=1)
        cos_theta = logits @ logits.T
        cos_theta = cos_theta.clamp(-1, 1)  # Limitar para evitar NaN

        # Convert labels to one-hot encoding
        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Compute the angle for the labels
        theta = torch.acos(cos_theta)  # Arco cosseno
        theta = theta.clamp(0, 3.14159265)  # Limitar para evitar valores inválidos

        # Apply margin
        arcface_logits = cos_theta - one_hot * (self.margin * torch.cos(theta))
        arcface_logits = arcface_logits * self.scale  # Escalar

        return F.cross_entropy(arcface_logits, labels)
        

# Define o modelo (LResNet50E-IR, uma ResNet50 modificada para ArcFace)
class LResNet50E_IR(nn.Module):
    def __init__(self, num_classes):
        super(LResNet50E_IR, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = self.fc

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x
