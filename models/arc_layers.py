import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcMarginProduct(nn.Module):
    """
    Implements the ArcMargin layer for angular margin penalties in classification tasks.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m  # Angular margin

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """
        Forward pass for ArcMarginProduct.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-7)  # Stabilish
        phi = cosine * self.cos_m - sine * self.sin_m  # Angular adjust

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Applying scale

        return output