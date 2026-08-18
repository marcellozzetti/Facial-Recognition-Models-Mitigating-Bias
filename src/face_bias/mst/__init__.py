"""MST — Etapa 1 do pipeline mestrado (Cap. 4 §4.2).

Exporta as classes e funções públicas do subpacote MST.
"""

from face_bias.mst.classifier import (
    MST_N_CLASSES,
    SKINTONENET_PAPER_URL,
    InferenceCache,
    MSTClassifier,
    WeightsUnavailableError,
    build_mst_backbone,
)
from face_bias.mst.datasets import (
    build_mst_dataset,
    class_balance,
    load_ccv2,
    load_mste,
    load_stw,
)
from face_bias.mst.preprocessing import MSTFromRawImage, MSTResult
from face_bias.mst.sensitivity import MSTSensitivityRunner, stone_monk_predictor
from face_bias.mst.trainer import MSTTrainer, TrainResult, stratified_split
from face_bias.mst.validation import (
    HumanLabel,
    HumanLabelStore,
    bootstrap_agreement,
    categorical_accuracy,
    cohens_kappa,
    generate_report,
    label_cli,
    stratified_sample,
)

__all__ = [
    # classifier
    "MST_N_CLASSES",
    "SKINTONENET_PAPER_URL",
    "InferenceCache",
    "MSTClassifier",
    "WeightsUnavailableError",
    "build_mst_backbone",
    # datasets
    "build_mst_dataset",
    "class_balance",
    "load_ccv2",
    "load_mste",
    "load_stw",
    # preprocessing (auto-suficiente)
    "MSTFromRawImage",
    "MSTResult",
    # trainer
    "MSTTrainer",
    "TrainResult",
    "stratified_split",
    # sensitivity
    "MSTSensitivityRunner",
    "stone_monk_predictor",
    # validation
    "HumanLabel",
    "HumanLabelStore",
    "bootstrap_agreement",
    "categorical_accuracy",
    "cohens_kappa",
    "generate_report",
    "label_cli",
    "stratified_sample",
]
