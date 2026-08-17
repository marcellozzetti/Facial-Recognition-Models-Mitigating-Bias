"""Auditoria fenotípica FairFace × MST — Etapa 2 (Cap. 4 §4.2)."""

from face_bias.audit.cross_matrix import (
    FAIRFACE_RACES,
    MST_LABELS,
    H3Result,
    build_matrix,
    coefficient_variation,
    entropy_per_race,
    spread_per_race,
    summarize,
    assess_hypothesis_h3,
    visualize,
)
from face_bias.audit.fairface_mst import (
    audit_fairface,
    audit_from_files,
    join_predictions,
    load_fairface_labels,
)

__all__ = [
    "FAIRFACE_RACES",
    "MST_LABELS",
    "H3Result",
    "build_matrix",
    "coefficient_variation",
    "entropy_per_race",
    "spread_per_race",
    "summarize",
    "assess_hypothesis_h3",
    "visualize",
    "audit_fairface",
    "audit_from_files",
    "join_predictions",
    "load_fairface_labels",
]
