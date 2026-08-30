"""F1 da pior classe (worst-class F1) + bootstrap CI 95%.

Cap. 4 §4.8. Métrica que protege o grupo mais fraco: ao invés de reportar
apenas média/macro, expõe o mínimo entre classes. Um modelo com F1 médio
alto mas F1 mínimo baixo é rejeitado nesta métrica.
"""

from __future__ import annotations

import numpy as np

from face_bias.fairness.disparity_ratio import per_class_f1


def worst_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    scores = per_class_f1(y_true, y_pred, n_classes)
    valid = scores[~np.isnan(scores)]
    return float("nan") if valid.size == 0 else float(valid.min())


def worst_class_f1_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Retorna (estatística, IC_lower, IC_upper) por percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = y_true.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = worst_class_f1(y_true, y_pred, n_classes)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = worst_class_f1(y_true[idx], y_pred[idx], n_classes)
    lo = float(np.nanquantile(stats, alpha / 2))
    hi = float(np.nanquantile(stats, 1 - alpha / 2))
    return (point, lo, hi)
