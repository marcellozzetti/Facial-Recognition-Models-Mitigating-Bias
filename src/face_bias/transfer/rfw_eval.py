"""RFW verification — Racial Faces in the Wild (Wang et al. 2019).

Cap. 4 §Etapa 5. Avaliação de reconhecimento facial 1:1 sobre pares
oficiais do RFW (Caucasian, Indian, Asian, African). Métrica canônica:
acurácia binária (mesma pessoa? sim/não) com threshold escolhido no
próprio conjunto.

Este módulo NÃO treina embeddings — recebe embeddings pré-computados
(saída do backbone fair da Etapa 3) e apenas calcula:

    similaridade coseno → threshold ótimo → acurácia por raça → gap

O código depende de um formato canônico de pares oferecido no dataset
RFW: cada linha = (path_a, path_b, is_same, race).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RFW_RACES = ("Caucasian", "Indian", "Asian", "African")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity linha a linha entre a e b (mesmo shape)."""
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1) / (na.squeeze(-1) * nb.squeeze(-1) + 1e-12)


def find_best_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """Busca threshold que maximiza acurácia binária. Retorna (thresh, acc)."""
    if similarities.size == 0:
        return (0.5, 0.0)
    lo, hi = float(similarities.min()), float(similarities.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    best_acc = -1.0
    best_thr = 0.5
    for t in thresholds:
        preds = (similarities >= t).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return (best_thr, best_acc)


def evaluate_pairs(
    pairs_df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    race_col: str = "race",
) -> pd.DataFrame:
    """Avalia pares 1:1. ``pairs_df`` precisa ter path_a, path_b, is_same.

    Retorna DataFrame com colunas: race, n_pairs, best_threshold, accuracy.
    O threshold é procurado GLOBALMENTE (todas as raças) e a acurácia
    é reportada por raça no mesmo threshold.
    """
    missing = {"path_a", "path_b", "is_same"} - set(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df sem colunas: {sorted(missing)}")

    sims: list[float] = []
    labels: list[int] = []
    races: list[str] = []
    for _, row in pairs_df.iterrows():
        emb_a = embeddings.get(row["path_a"])
        emb_b = embeddings.get(row["path_b"])
        if emb_a is None or emb_b is None:
            continue
        s = float(cosine_similarity(emb_a[None, :], emb_b[None, :])[0])
        sims.append(s)
        labels.append(int(row["is_same"]))
        races.append(str(row.get(race_col, "unknown")))
    sims_arr = np.array(sims)
    labels_arr = np.array(labels)

    best_thr, _ = find_best_threshold(sims_arr, labels_arr)

    rows = []
    for race in sorted(set(races)):
        mask = np.array([r == race for r in races])
        preds = (sims_arr[mask] >= best_thr).astype(int)
        n = int(mask.sum())
        acc = float((preds == labels_arr[mask]).mean()) if n > 0 else float("nan")
        rows.append({
            "race": race, "n_pairs": n,
            "best_threshold": best_thr, "accuracy": acc,
        })
    return pd.DataFrame(rows)


def race_gap(per_race_df: pd.DataFrame) -> float:
    """Gap = max_acc - min_acc entre raças."""
    accs = per_race_df["accuracy"].dropna()
    if accs.size < 2:
        return float("nan")
    return float(accs.max() - accs.min())
