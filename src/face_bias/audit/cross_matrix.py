"""Matriz cruzada Monk Skin Tone × classes raciais — Contribuição 2.

Cap. 3, Contribuição 2 (eixo fenotípico-empírico) e Cap. 4 §4.2 (Etapa 2).

Métricas fornecidas:
    - build_matrix         : contagens normalizadas (linha soma 1) 7×10
    - spread_per_race      : nº de tons MST com massa ≥ threshold por raça
    - entropy_per_race     : entropia de Shannon (bits) por raça
    - coefficient_variation: CV por raça (desvio-padrão / média)
    - assess_hypothesis_h3   : H3 do Cap. 3 (spread Latinx ≥ 5)
    - visualize            : heatmap matplotlib (padrão dataviz da tese)

Ver também:
    - Cap 3, H3
    - Cap 4 §4.2 (Etapa 2)
    - src/face_bias/audit/fairface_mst.py — produtor do input desta matriz
    - src/face_bias/decomposition/error_decomp.py — consumidor (Etapa 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MST_LABELS = list(range(1, 11))  # 1..10 (Monk)
FAIRFACE_RACES = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino_Hispanic",
]


def build_matrix(
    audit_df: pd.DataFrame,
    race_col: str = "race",
    mst_col: str = "mst_pred",
    normalize: bool = True,
) -> pd.DataFrame:
    """Contagens raça × MST. Se ``normalize``, cada linha soma 1.

    Índices garantidos: linhas = ``FAIRFACE_RACES``, colunas = 1..10.
    Faltantes preenchidas com 0.
    """
    grid = pd.crosstab(audit_df[race_col], audit_df[mst_col], dropna=False)
    grid = grid.reindex(index=FAIRFACE_RACES, columns=MST_LABELS, fill_value=0)
    grid.columns = [int(c) for c in grid.columns]
    if normalize:
        totals = grid.sum(axis=1).replace(0, np.nan)
        grid = grid.div(totals, axis=0).fillna(0.0)
    return grid


def spread_per_race(matrix: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """Nº de tons MST com massa relativa ≥ ``threshold`` por raça.

    Espera matriz normalizada por linha (proporção). Threshold default 5%
    é o critério declarado em Cap. 3 §H3 para "spread" da classe Latinx.
    """
    return (matrix >= threshold).sum(axis=1).rename("spread")


def entropy_per_race(matrix: pd.DataFrame) -> pd.Series:
    """Entropia de Shannon (bits) da distribuição MST por raça.

    Entropia zero = concentrada em 1 tom. Máximo = log2(10) ≈ 3.32 bits.
    """
    p = matrix.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    h = -(p * logp).sum(axis=1)
    return pd.Series(h, index=matrix.index, name="entropy_bits")


def coefficient_variation(matrix: pd.DataFrame) -> pd.Series:
    """CV = std / mean da distribuição por linha (matriz normalizada).

    CV alto → distribuição muito desigual entre tons MST.
    """
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=0)
    cv = (std / mean.replace(0, np.nan)).fillna(0.0)
    return cv.rename("cv")


@dataclass(frozen=True)
class H3Result:
    """Resultado qualitativo do teste H3 do Cap. 3 (spread fenotípico Latinx).

    H3 é avaliado qualitativamente: compara o spread Latinx ao das demais
    raças sem limiar numérico pré-fixado (conforme revisão pós-qualificação).
    """

    latinx_spread: int
    threshold: float
    per_race_spread: dict
    latinx_rank_among_races: int    # posição em ordem decrescente de spread
    num_races: int                  # total de raças na matriz
    median_other_spread: float      # mediana do spread das demais raças

    def as_dict(self) -> dict:
        return {
            "hypothesis": "H3",
            "statement": "Latinx exibe spread MST expressivo comparado às demais raças (qualitativo)",
            "threshold_prop": self.threshold,
            "latinx_spread": self.latinx_spread,
            "latinx_rank_among_races": self.latinx_rank_among_races,
            "num_races": self.num_races,
            "median_other_spread": self.median_other_spread,
            "per_race_spread": self.per_race_spread,
        }


def assess_hypothesis_h3(
    matrix: pd.DataFrame,
    threshold: float = 0.05,
    latinx_key: str = "Latino_Hispanic",
) -> H3Result:
    """H3 (Cap. 3): Latinx exibe spread MST expressivamente superior às demais raças.

    Avaliação qualitativa: reporta o spread observado, sua posição relativa
    entre as raças e a mediana das demais — sem limiar numérico pré-fixado.
    """
    if latinx_key not in matrix.index:
        raise KeyError(f"{latinx_key!r} não presente na matriz. Índices: {list(matrix.index)}")
    spreads = spread_per_race(matrix, threshold=threshold).astype(int)
    latinx_spread = int(spreads.loc[latinx_key])
    sorted_spreads = spreads.sort_values(ascending=False)
    rank = int(list(sorted_spreads.index).index(latinx_key) + 1)
    other_vals = [int(v) for k, v in spreads.items() if k != latinx_key]
    median_other = float(np.median(other_vals)) if other_vals else 0.0
    return H3Result(
        latinx_spread=latinx_spread,
        threshold=threshold,
        per_race_spread={k: int(v) for k, v in spreads.items()},
        latinx_rank_among_races=rank,
        num_races=int(len(spreads)),
        median_other_spread=median_other,
    )


def visualize(
    matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Distribuição MST por classe racial (FairFace)",
    cmap: str = "YlOrBr",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Heatmap da matriz (matplotlib), com anotação dos valores.

    Importação de matplotlib dentro da função para evitar dep no import
    do módulo (mesmo padrão dos demais dataviz do projeto).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = matrix.to_numpy(dtype=np.float64)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.0, vmax=data.max())
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"MST {c}" for c in matrix.columns], rotation=0)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Monk Skin Tone (1 = clarissimo, 10 = escurissimo)")
    ax.set_ylabel("Classe racial (FairFace)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v == 0:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if v < data.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Proporção intra-raça")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def summarize(matrix: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Resumo por raça: spread, entropia (bits) e CV, alinhados em um DataFrame."""
    return pd.concat(
        [
            spread_per_race(matrix, threshold=threshold),
            entropy_per_race(matrix),
            coefficient_variation(matrix),
        ],
        axis=1,
    )
