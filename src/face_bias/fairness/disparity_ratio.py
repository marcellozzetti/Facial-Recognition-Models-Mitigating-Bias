"""Disparity Ratio — razão entre a pior e a melhor classe (F1 ou acurácia).

Cap. 4 §4.8 (Triangulação de métricas). Mede a razão entre o desempenho
do grupo com pior resultado e o do grupo com melhor resultado. Quanto
mais próximo de 1, mais igual é o desempenho entre grupos.

    DR = min_c(metric_c) / max_c(metric_c)

Também exposto o "inequity rate" 1 - DR usado por parte da literatura.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """F1 por classe. NaN para classes sem suporte no y_true."""
    f1s = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn)
        f1s[c] = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1s


def disparity_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    metric: str = "f1",
) -> float:
    """DR = min/max. Retorna NaN se não há pelo menos 2 classes com suporte."""
    if metric != "f1":
        raise NotImplementedError(f"metric {metric!r} — só 'f1' hoje.")
    scores = per_class_f1(y_true, y_pred, n_classes)
    valid = scores[~np.isnan(scores)]
    if valid.size < 2 or valid.max() == 0:
        return float("nan")
    return float(valid.min() / valid.max())


def inequity_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    metric: str = "f1",
) -> float:
    dr = disparity_ratio(y_true, y_pred, n_classes, metric=metric)
    return float("nan") if np.isnan(dr) else float(1.0 - dr)


def report_dr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
) -> pd.DataFrame:
    """DataFrame com colunas: class, f1, is_worst, is_best."""
    names = list(class_names)
    scores = per_class_f1(y_true, y_pred, len(names))
    worst_i = int(np.nanargmin(scores)) if not np.all(np.isnan(scores)) else -1
    best_i = int(np.nanargmax(scores)) if not np.all(np.isnan(scores)) else -1
    return pd.DataFrame({
        "class": names,
        "f1": scores,
        "is_worst": [i == worst_i for i in range(len(names))],
        "is_best": [i == best_i for i in range(len(names))],
    })
