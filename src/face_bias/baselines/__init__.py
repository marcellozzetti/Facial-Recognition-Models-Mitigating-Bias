"""Baselines de mitigação — Etapa 4 do pipeline (Cap. 4 §Baselines)."""

from face_bias.baselines.adversarial_debias import (
    AdversarialDebiasLoss,
    AdversarialHead,
    GradientReverse,
    gradient_reverse,
)
from face_bias.baselines.fineface import CrossLayerMutualAttention
from face_bias.baselines.fscl_plus import FSCLPlusLoss
from face_bias.baselines.group_dro import GroupDROLoss

__all__ = [
    "AdversarialDebiasLoss",
    "AdversarialHead",
    "CrossLayerMutualAttention",
    "FSCLPlusLoss",
    "GradientReverse",
    "GroupDROLoss",
    "gradient_reverse",
]
