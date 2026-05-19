"""Pydantic schema for the YAML configuration.

The schema is intentionally permissive (extra=ignore, optional fields with
sensible defaults) so the config files used in MBA experiments still load
unchanged. Validation only fires for fields the runtime relies on.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    name: str = "LResNet50E_IR"
    pretrained: bool = True
    num_classes: int = Field(default=7, ge=2)
    dropout: float = Field(default=0.5, ge=0.0, le=1.0)
    head: Literal["linear", "arcface", "mlp", "adaface", "magface"] = "linear"
    arcface_s: float = Field(default=30.0, gt=0)
    arcface_m: float = Field(default=0.5, ge=0.0, lt=1.5)
    arcface_easy_margin: bool = False
    # MLP head (only used when head="mlp"). Search space driven by Optuna.
    mlp_hidden_dims: list[int] = Field(default_factory=lambda: [512])
    mlp_activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    mlp_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    mlp_norm: Literal["none", "batchnorm", "layernorm"] = "none"
    # AdaFace head (only used when head="adaface"). Defaults from the paper.
    adaface_m: float = Field(default=0.4, ge=0.0, lt=1.5)
    adaface_h: float = Field(default=0.333, gt=0)
    # MagFace head (only used when head="magface"). Defaults from the paper.
    magface_l_a: float = Field(default=10.0, gt=0)
    magface_u_a: float = Field(default=110.0, gt=0)
    magface_l_m: float = Field(default=0.45, ge=0.0, lt=1.5)
    magface_u_m: float = Field(default=0.8, ge=0.0, lt=1.5)
    magface_lambda_g: float = Field(default=0.0, ge=0.0)

    @field_validator("magface_u_a")
    @classmethod
    def _mag_u_a_gt_l_a(cls, v: float, info) -> float:
        l_a = info.data.get("magface_l_a", 10.0)
        if v <= l_a:
            raise ValueError(f"magface_u_a must be > magface_l_a; got {l_a}, {v}")
        return v

    @field_validator("mlp_hidden_dims")
    @classmethod
    def _hidden_dims_positive(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError(f"mlp_hidden_dims must all be > 0; got {v}")
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    num_epochs: int = Field(default=25, ge=1)
    optimizer: Literal["sgd", "adamw"] = "adamw"
    scheduler: Literal["onecyclelr", "cosineannealingwarmrestarts"] = "onecyclelr"
    loss_function: Literal[
        "cross_entropy", "arcface", "adaface", "magface"
    ] = "cross_entropy"
    test_size: float = Field(default=0.1, gt=0, lt=1)
    val_size: Optional[float] = Field(default=0.1, gt=0, lt=1)
    num_workers: int = Field(default=4, ge=0)
    random_state: int = 42
    # Gradient-norm clipping; None disables it.
    grad_clip_norm: Optional[float] = Field(default=5.0, gt=0)
    # Model-selection / early-stopping metric. Default is the task
    # metric (macro-F1): val_loss is anti-correlated with F1 for margin
    # heads (plain-cosine eval logits) and biases the loss-factor study.
    checkpoint_metric: Literal[
        "val_loss", "val_f1_macro", "val_accuracy"
    ] = "val_f1_macro"
    # Early stopping on the checkpoint_metric; null disables it.
    early_stopping_patience: Optional[int] = Field(default=5, ge=1)
    # DataLoader prefetch_factor (only used when num_workers > 0).
    prefetch_factor: int = Field(default=4, ge=1)
    # When True, train with torch.amp.autocast(fp16) + GradScaler. Default
    # off so the 11-experiment replication remains numerically reproducible;
    # HPO and any new training recipes can opt in via the config.
    use_amp: bool = False


class ImageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    processing_type: Literal["rotate_image", "detect_adjust_image"] = "detect_adjust_image"
    num_workers: int = Field(default=4, ge=1)
    rotate_angle: int = 45
    num_rotations: int = Field(default=8, ge=1)
    max_faces_per_image: Optional[int] = Field(default=1, ge=1)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_file: Optional[str] = None
    dataset_image_input_path: Optional[str] = None
    dataset_image_output_path: Optional[str] = None
    balance: Literal["none", "undersample"] = "none"
    # When set, restrict the dataset to rows whose `race` is in this list
    # (used to reproduce MBA Exp. 7 and 8 — Black + White only).
    class_filter: Optional[list[str]] = None


class BucketConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    function: Literal["download", "upload"] = "download"
    bucket_client: Literal["s3", "blob"] = "s3"
    bucket_path_file: str = "data/raw/bucket/"
    object_file_name: str = "dataset.zip"
    bucket_name: str = "dataset"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    log_dir: str = "outputs/logs"
    log_level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_bucket_file: str = "bucket.log"
    log_preprocessing_file: str = "preprocessing.log"
    log_face_data_file: str = "face_data.log"
    log_version_file: str = "version.log"
    log_training_file: str = "training.log"
    log_evaluation_file: str = "evaluation.log"

    @field_validator("log_level")
    @classmethod
    def _level_must_be_known(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level {v!r} not one of {sorted(valid)}")
        return v.upper()


class FaceBiasConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    bucket: BucketConfig = Field(default_factory=BucketConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def validate_config(raw: dict) -> dict:
    """Validate ``raw`` against ``FaceBiasConfig`` and return a normalised dict.

    Returning a dict (not the model instance) keeps callers backwards
    compatible — every existing call to ``config['training']['batch_size']``
    still works.
    """
    parsed = FaceBiasConfig.model_validate(raw)
    return parsed.model_dump()
