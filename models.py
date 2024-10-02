import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet101_Weights, ResNet50_Weights


class ArcFaceLossOld(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        # Normalize logits to get the unit vectors
        logits = F.normalize(logits, dim=1)
        cos_theta = logits @ logits.T
        cos_theta = cos_theta.clamp(-1, 1)

        # Convert labels to one-hot encoding
        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Compute the angle for the labels
        theta = torch.acos(cos_theta)
        theta = theta.clamp(0, 3.14159265)

        # Apply margin
        arcface_logits = cos_theta - one_hot * (self.margin * torch.cos(theta))
        arcface_logits = arcface_logits * self.scale

        return F.cross_entropy(arcface_logits, labels)
        

class LResNet50E_IROld(nn.Module):
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


class LResNet50E_IR(nn.Module):
    def __init__(self, num_classes):
        super(LResNet50E_IR, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
            
        # Load pretrained ResNet50 without the final fully connected layer
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        
        # Remove the last fully connected layer (fc) of the ResNet
        self.backbone.fc = nn.Identity()
        
        # Define a new fully connected layer for the specific number of classes
        self.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Add dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, labels):
        # Normalize logits
        logits = F.normalize(logits, dim=1)
        
        # Compute cosine similarity between logits and weights
        cos_theta = logits @ logits.T
        cos_theta = cos_theta.clamp(-1, 1)  # Ensure values are in range [-1, 1]
        
        # Convert labels to one-hot encoding
        one_hot = torch.zeros_like(cos_theta, device=cos_theta.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin to the correct class angles
        theta = torch.acos(cos_theta)
        phi = torch.cos(theta + self.margin)
        
        # Apply condition for easy margin
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        
        # Combine cos_theta with one-hot and scale
        arcface_logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        arcface_logits = arcface_logits * self.scale
        
        return F.cross_entropy(arcface_logits, labels)
