import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class LResNet50E_IR(nn.Module):
    """
    ResNet50-based model tailored for facial recognition tasks.
    """
    def __init__(self, num_classes):
        """
        Initializes the LResNet50E_IR model.
        """
        super(LResNet50E_IR, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")

        # Load pretrained ResNet50 without the final fully connected layer
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features

        # Replace the last fully connected layer with an identity layer
        self.backbone.fc = nn.Identity()

        # Define a new fully connected layer for the specific number of classes
        self.fc = nn.Linear(in_features, num_classes)

        # Optional dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Apply dropout for regularization
        features = self.dropout(features)

        # Compute class logits
        logits = self.fc(features)

        return logits