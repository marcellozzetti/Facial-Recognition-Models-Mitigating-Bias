"""Métricas de fairness — Etapa 4 do pipeline (Cap. 4 §4.8)."""

from face_bias.fairness.disparity_ratio import (
    disparity_ratio,
    inequity_rate,
    per_class_f1,
    report_dr,
)
from face_bias.fairness.equal_opportunity import (
    equal_opportunity_gap,
    equal_opportunity_per_class,
    tpr_per_group,
)
from face_bias.fairness.equalized_odds import (
    equalized_odds_gap,
    equalized_odds_per_class,
    tpr_fpr_per_group,
)
from face_bias.fairness.pareto import (
    ModelResult,
    build_pareto_table,
    pareto_front,
    plot_pareto,
)
from face_bias.fairness.worst_class_f1 import (
    worst_class_f1,
    worst_class_f1_bootstrap,
)

__all__ = [
    "disparity_ratio",
    "inequity_rate",
    "per_class_f1",
    "report_dr",
    "worst_class_f1",
    "worst_class_f1_bootstrap",
    "tpr_per_group",
    "equal_opportunity_gap",
    "equal_opportunity_per_class",
    "tpr_fpr_per_group",
    "equalized_odds_gap",
    "equalized_odds_per_class",
    "ModelResult",
    "pareto_front",
    "plot_pareto",
    "build_pareto_table",
]
