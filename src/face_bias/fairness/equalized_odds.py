"""Equalized Odds — Hardt et al. (2016), definição formal mais estrita que EO.

Cap. 4 §4.8. EqOdds exige igualdade SIMULTANEAMENTE de TPR e FPR entre
grupos protegidos:

    P(Ŷ=1 | Y=y, A=a) == P(Ŷ=1 | Y=y, A=b)   para y ∈ {0, 1}

Em regime multi-classe (por classe): para cada classe c, medimos TPR e
FPR por grupo e o gap simultâneo (max entre gap_TPR e gap_FPR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tpr_fpr_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target_class: int,
) -> dict:
    """{grupo: (TPR, FPR)} para a classe target_class."""
    out: dict = {}
    for g in np.unique(group):
        mask = group == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tp = int(((y_t == target_class) & (y_p == target_class)).sum())
        fn = int(((y_t == target_class) & (y_p != target_class)).sum())
        fp = int(((y_t != target_class) & (y_p == target_class)).sum())
        tn = int(((y_t != target_class) & (y_p != target_class)).sum())
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        out[g] = (tpr, fpr)
    return out


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target_class: int,
) -> float:
    """max(gap_TPR, gap_FPR) — pior violação entre as duas condições."""
    stats = tpr_fpr_per_group(y_true, y_pred, group, target_class)
    tprs = [v[0] for v in stats.values() if not np.isnan(v[0])]
    fprs = [v[1] for v in stats.values() if not np.isnan(v[1])]
    if len(tprs) < 2 or len(fprs) < 2:
        return float("nan")
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def equalized_odds_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    n_classes: int,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """DataFrame com TPR/FPR por grupo + gap simultâneo, por classe."""
    names = class_names or [str(i) for i in range(n_classes)]
    rows = []
    for c in range(n_classes):
        entry = {"class": names[c]}
        for g, (tpr, fpr) in tpr_fpr_per_group(y_true, y_pred, group, c).items():
            entry[f"tpr_{g}"] = tpr
            entry[f"fpr_{g}"] = fpr
        entry["eqodds_gap"] = equalized_odds_gap(y_true, y_pred, group, c)
        rows.append(entry)
    return pd.DataFrame(rows)
