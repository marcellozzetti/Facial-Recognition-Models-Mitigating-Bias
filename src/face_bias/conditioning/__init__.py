"""Condicionamento arquitetural — Etapa 3 do pipeline (Cap. 4 §4.6–§4.7)."""

from face_bias.conditioning.clip_prompts import (
    CLIP_EMBED_DIM,
    DEFAULT_MODEL,
    DEFAULT_TEMPLATE,
    CLIPPromptEnsembler,
    build_prompts,
)
from face_bias.conditioning.film import (
    FiLMLayer,
    MLPFilmGenerator,
    compute_film_overhead,
)
from face_bias.conditioning.injector import (
    CONVNEXT_TINY_STAGE_CHANNELS,
    FiLMConditionedConvNeXt,
    wrap_convnext_with_film,
)

__all__ = [
    "CLIP_EMBED_DIM",
    "CLIPPromptEnsembler",
    "CONVNEXT_TINY_STAGE_CHANNELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPLATE",
    "FiLMConditionedConvNeXt",
    "FiLMLayer",
    "MLPFilmGenerator",
    "build_prompts",
    "compute_film_overhead",
    "wrap_convnext_with_film",
]
