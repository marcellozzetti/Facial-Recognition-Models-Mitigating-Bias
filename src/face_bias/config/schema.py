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
    head: Literal["linear", "arcface"] = "linear"
    arcface_s: float = Field(default=30.0, gt=0)
    arcface_m: float = Field(default=0.5, ge=0.0, lt=1.5)
    arcface_easy_margin: bool = False


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    num_epochs: int = Field(default=25, ge=1)
    optimizer: Literal["sgd", "adamw"] = "adamw"
    scheduler: Literal["onecyclelr", "cosineannealingwarmrestarts"] = "onecyclelr"
    loss_function: Literal["cross_entropy", "arcface"] = "cross_entropy"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    num_workers: int = Field(default=4, ge=0)
    random_state: int = 42


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
