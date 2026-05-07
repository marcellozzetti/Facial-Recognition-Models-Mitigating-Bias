"""Classification + fairness metrics for the FairFace race-classification task.

The fairness audit reports per-class disparities (max/min ratio, max-min
difference, std, coefficient of variation, Gini) so the dissertation can
move beyond aggregate accuracy and surface the actual gap between
demographic groups (REVIEW_AND_PLAN.md §2 / Kotwal & Marcel 2025).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Aggregate classification metrics computed with macro averaging."""
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out: dict[str, float] = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    if y_proba is not None:
        labels = list(range(y_proba.shape[1]))
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))

    # `class_names` is accepted for API symmetry but not used here; the
    # per-class breakdown lives in `per_class_report`.
    del class_names
    return out


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return precision / recall / F1 / support per class as a DataFrame."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


# ---------------------------------------------------------------- fairness


def _to_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("fairness metrics require at least one value")
    return arr


def inequity_rate(per_group_score: Iterable[float]) -> float:
    """max(score) / min(score) over groups — 1.0 means perfectly fair.

    Follows Pereira & Marcel's framing for biometric verification, adapted
    here to per-class classification scores. Returns +inf if any group has
    a score of zero.
    """
    arr = _to_array(per_group_score)
    lo, hi = float(arr.min()), float(arr.max())
    if lo <= 0:
        return float("inf") if hi > 0 else 0.0
    return hi / lo


def max_min_disparity(per_group_score: Iterable[float]) -> float:
    """Absolute gap between the best- and worst-served group."""
    arr = _to_array(per_group_score)
    return float(arr.max() - arr.min())


def coefficient_of_variation(per_group_score: Iterable[float]) -> float:
    """std / mean — a scale-invariant disparity proxy (FDR-style)."""
    arr = _to_array(per_group_score)
    mean = arr.mean()
    if mean == 0:
        return float("inf")
    return float(arr.std(ddof=0) / mean)


def gini_coefficient(per_group_score: Iterable[float]) -> float:
    """Gini coefficient over per-group scores. 0 = perfect equality.

    Implemented from the closed-form sorted-values expression so the result
    matches the GARBE family in Kotwal & Marcel (2025).
    """
    arr = np.sort(_to_array(per_group_score))
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float | dict[str, float]]:
    """One-call audit: per-class scores plus aggregate disparity numbers."""
    report = per_class_report(y_true, y_pred, class_names)
    metrics: dict[str, float | dict[str, float]] = {}
    for column in ("precision", "recall", "f1"):
        scores = report[column].to_numpy()
        metrics[column] = {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "inequity_rate": inequity_rate(scores),
            "max_min_disparity": max_min_disparity(scores),
            "coefficient_of_variation": coefficient_of_variation(scores),
            "gini": gini_coefficient(scores),
        }
    return metrics


def confusion_matrix_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """sklearn confusion matrix wrapped in a DataFrame with class labels."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=list(class_names), columns=list(class_names))


def text_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> str:
    """sklearn-style text classification report."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    )
