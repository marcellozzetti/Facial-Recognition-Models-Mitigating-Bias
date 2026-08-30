"""BFW verification — Balanced Faces in the Wild (Robinson et al. 2020).

Cap. 4 §Etapa 5. BFW estende o RFW com 8 subgrupos race × gender
(4 raças × 2 gêneros), oferecendo análise interseccional formal.

Reutiliza a maquinaria de ``rfw_eval`` — a diferença é que agora
agrupamos por (race, gender) em vez de apenas race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from face_bias.transfer.rfw_eval import cosine_similarity, find_best_threshold


def evaluate_pairs_intersectional(
    pairs_df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    race_col: str = "race",
    gender_col: str = "gender",
) -> pd.DataFrame:
    """Como ``rfw_eval.evaluate_pairs`` mas agrupa por (race, gender)."""
    missing = {"path_a", "path_b", "is_same"} - set(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df sem colunas: {sorted(missing)}")

    sims, labels, races, genders = [], [], [], []
    for _, row in pairs_df.iterrows():
        emb_a = embeddings.get(row["path_a"])
        emb_b = embeddings.get(row["path_b"])
        if emb_a is None or emb_b is None:
            continue
        sims.append(float(cosine_similarity(emb_a[None, :], emb_b[None, :])[0]))
        labels.append(int(row["is_same"]))
        races.append(str(row.get(race_col, "unknown")))
        genders.append(str(row.get(gender_col, "unknown")))
    sims_arr = np.array(sims)
    labels_arr = np.array(labels)
    best_thr, _ = find_best_threshold(sims_arr, labels_arr)

    rows = []
    for race in sorted(set(races)):
        for gender in sorted(set(genders)):
            mask = np.array([r == race and g == gender for r, g in zip(races, genders)])
            n = int(mask.sum())
            if n == 0:
                continue
            preds = (sims_arr[mask] >= best_thr).astype(int)
            acc = float((preds == labels_arr[mask]).mean())
            rows.append({
                "race": race, "gender": gender, "subgroup": f"{race}_{gender}",
                "n_pairs": n, "best_threshold": best_thr, "accuracy": acc,
            })
    return pd.DataFrame(rows)


def intersectional_gap(per_subgroup_df: pd.DataFrame) -> float:
    """Gap = max_acc - min_acc entre subgrupos race×gender."""
    accs = per_subgroup_df["accuracy"].dropna()
    if accs.size < 2:
        return float("nan")
    return float(accs.max() - accs.min())
