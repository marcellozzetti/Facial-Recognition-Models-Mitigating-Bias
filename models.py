import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


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
        return F.cross_entropy(logits, labels)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m  # Margem angular

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Normalizando input e pesos
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Garantir que cosine está no intervalo [-1, 1] para evitar erros numéricos
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-7)  # Adiciona pequena constante para estabilidade
        phi = cosine * self.cos_m - sine * self.sin_m  # Ajuste angular

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Aplicando a escala

        return output
