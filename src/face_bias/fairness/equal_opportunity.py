"""Equal Opportunity — Hardt et al. (2016), definição formal de fairness.

Cap. 4 §4.8. EO exige que grupos protegidos tenham TAXAS DE VERDADEIRO
POSITIVO iguais, condicionadas ao rótulo verdadeiro:

    P(Ŷ=1 | Y=1, A=a) == P(Ŷ=1 | Y=1, A=b)

Em regime multi-classe, extendemos por classe: para cada classe c, TPR
por grupo protegido A.

Uso principal do Cenário B (raça × gênero): mede o gap entre gêneros
dentro de cada classe racial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tpr_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target_class: int,
) -> dict:
    """TPR (recall) da classe ``target_class`` estratificado por grupo."""
    out: dict = {}
    for g in np.unique(group):
        mask = group == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tp = int(((y_t == target_class) & (y_p == target_class)).sum())
        fn = int(((y_t == target_class) & (y_p != target_class)).sum())
        out[g] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    return out


def equal_opportunity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target_class: int,
) -> float:
    """Gap = max - min entre TPRs por grupo. 0 = igualdade perfeita."""
    tprs = list(tpr_per_group(y_true, y_pred, group, target_class).values())
    valid = [v for v in tprs if not np.isnan(v)]
    return float("nan") if len(valid) < 2 else float(max(valid) - min(valid))


def equal_opportunity_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    n_classes: int,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """DataFrame: uma linha por classe, colunas por grupo + coluna 'gap'."""
    names = class_names or [str(i) for i in range(n_classes)]
    rows = []
    for c in range(n_classes):
        entry = {"class": names[c]}
        for g, tpr in tpr_per_group(y_true, y_pred, group, c).items():
            entry[f"tpr_{g}"] = tpr
        entry["eo_gap"] = equal_opportunity_gap(y_true, y_pred, group, c)
        rows.append(entry)
    return pd.DataFrame(rows)
