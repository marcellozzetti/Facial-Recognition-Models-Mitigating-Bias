"""Fronteira Pareto — visualização do trade-off acurácia × disparidade.

Cap. 4 §4.8. Para cada modelo, plotamos:
    x = disparidade (menor é melhor — usamos 1 - DR = inequity rate)
    y = acurácia agregada (F1 macro, maior é melhor)

Modelos na fronteira são aqueles em que não é possível melhorar uma
dimensão sem piorar a outra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    name: str
    macro_f1: float
    inequity: float  # 1 - DR (menor é melhor)


def pareto_front(results: list[ModelResult]) -> list[ModelResult]:
    """Retorna subset Pareto-eficiente. Assume: maximizar macro_f1,
    minimizar inequity. Complexidade O(n^2), suficiente para 20-30 modelos.
    """
    front: list[ModelResult] = []
    for r in results:
        dominated = False
        for s in results:
            if s is r:
                continue
            # s domina r se s >= r em ambas as dims e > em pelo menos uma
            better_or_equal = (s.macro_f1 >= r.macro_f1) and (s.inequity <= r.inequity)
            strictly_better = (s.macro_f1 > r.macro_f1) or (s.inequity < r.inequity)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r.inequity)


def plot_pareto(
    results: list[ModelResult],
    save_path: Optional[Path] = None,
    title: str = "Trade-off acurácia × disparidade",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    import matplotlib.pyplot as plt

    front = set(id(r) for r in pareto_front(results))
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        is_front = id(r) in front
        color = "#c0392b" if is_front else "#70768c"
        marker = "o" if is_front else "x"
        size = 90 if is_front else 60
        ax.scatter(r.inequity, r.macro_f1, s=size, c=color, marker=marker,
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(
            r.name, (r.inequity, r.macro_f1),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="#1f2a4e",
        )
    # linha da fronteira
    front_sorted = sorted(
        [r for r in results if id(r) in front], key=lambda r: r.inequity
    )
    if len(front_sorted) >= 2:
        xs = [r.inequity for r in front_sorted]
        ys = [r.macro_f1 for r in front_sorted]
        ax.plot(xs, ys, linestyle="--", color="#c0392b", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Inequidade (1 − DR) — menor é melhor")
    ax.set_ylabel("F1 macro — maior é melhor")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def build_pareto_table(results: list[ModelResult]) -> pd.DataFrame:
    front_ids = {id(r) for r in pareto_front(results)}
    return pd.DataFrame([
        {
            "model": r.name,
            "macro_f1": r.macro_f1,
            "inequity": r.inequity,
            "pareto": id(r) in front_ids,
        }
        for r in results
    ]).sort_values("inequity").reset_index(drop=True)
