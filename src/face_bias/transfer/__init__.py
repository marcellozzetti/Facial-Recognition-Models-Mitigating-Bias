"""Transferência fair para reconhecimento — Etapa 5 (Cap. 4 §Etapa 5)."""

from face_bias.transfer.bfw_eval import (
    evaluate_pairs_intersectional,
    intersectional_gap,
)
from face_bias.transfer.bisenet_pixel_info import (
    BiSeNetPixelInfo,
    pixel_info_from_bbox,
    pixel_info_from_mask,
)
from face_bias.transfer.quality_descriptors import (
    descriptors_for_crop,
    face_resolution_pixels,
    luminance_lstar,
    sharpness_laplacian,
)
from face_bias.transfer.rfw_eval import (
    RFW_RACES,
    cosine_similarity,
    evaluate_pairs,
    find_best_threshold,
    race_gap,
)

__all__ = [
    # rfw_eval
    "RFW_RACES",
    "cosine_similarity",
    "evaluate_pairs",
    "find_best_threshold",
    "race_gap",
    # bfw_eval
    "evaluate_pairs_intersectional",
    "intersectional_gap",
    # bisenet_pixel_info
    "BiSeNetPixelInfo",
    "pixel_info_from_bbox",
    "pixel_info_from_mask",
    # quality_descriptors
    "descriptors_for_crop",
    "face_resolution_pixels",
    "luminance_lstar",
    "sharpness_laplacian",
]
