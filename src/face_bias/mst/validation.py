"""Validação humana interna do classificador MST — Cap. 4, §4.9.

Escopo declarado (§4.9): subset estratificado de 200-300 imagens do
FairFace, estratificado por raça e por tom MST predito. Rotulagem feita
EXCLUSIVAMENTE por Mestrando + Orientador — sem crowdsourcing.

Este módulo cobre:
    - Amostragem estratificada por (raça, mst_pred)
    - Persistência de labels em JSONL (uma linha por imagem)
    - Cálculo de κ de Cohen pareado
    - Exatidão categórica com IC 95% via bootstrap não paramétrico
    - Geração de relatório Markdown para anexo da dissertação

Interface pública:
    def stratified_sample(predictions_df, race_col, mst_col,
                          n_target=250, seed=42) -> pd.DataFrame
    class HumanLabelStore  (persistência JSONL)
    def cohens_kappa(a, b) -> float
    def bootstrap_agreement(a, b, n_boot=10_000, seed=42) -> tuple[float, float, float]
    def generate_report(...) -> str

Ver também:
    - Cap 3 (Objetivo 1) depende deste protocolo
    - Cap 5 (Riscos): Risco 2 (dependência SkinToneNet)
    - src/face_bias/mst/skintonenet.py — fornece as predições iniciais
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

MST_LABELS = list(range(1, 11))


def stratified_sample(
    predictions: pd.DataFrame,
    race_col: str = "race",
    mst_col: str = "mst_pred",
    n_target: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Amostragem estratificada proporcional por (raça, MST predito).

    Cada célula (r, m) recebe pelo menos 1 amostra se houver material,
    e o restante é distribuído proporcionalmente à frequência da célula
    no dataset completo. Estratégia conservadora para não deixar tons
    raros sem cobertura.
    """
    rng = np.random.default_rng(seed)
    groups = predictions.groupby([race_col, mst_col], sort=True)
    sizes = groups.size()
    total = int(sizes.sum())
    if total == 0:
        return predictions.iloc[0:0].copy()

    # Alocação proporcional + piso 1 por célula não vazia.
    props = sizes / total
    raw = props * n_target
    alloc = np.maximum(1, np.floor(raw).astype(int))
    # Ajuste fino para bater exatamente n_target (ou o máximo disponível).
    while alloc.sum() > n_target:
        big = alloc.idxmax()
        alloc[big] -= 1
    while alloc.sum() < n_target:
        residuals = raw - alloc
        candidate = residuals.idxmax()
        if alloc[candidate] >= sizes[candidate]:
            residuals = residuals.drop(candidate)
            if residuals.empty:
                break
            candidate = residuals.idxmax()
        alloc[candidate] += 1

    chunks = []
    for key, size in sizes.items():
        take = int(min(alloc.get(key, 0), size))
        if take <= 0:
            continue
        subset = groups.get_group(key)
        chunks.append(subset.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(chunks, ignore_index=True) if chunks else predictions.iloc[0:0].copy()


@dataclass(frozen=True)
class HumanLabel:
    image_path: str
    annotator: str
    mst_label: int  # 1..10; -1 = "não avaliável"
    notes: str = ""


class HumanLabelStore:
    """JSONL append-only para as sessões de rotulagem humana."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, label: HumanLabel) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(label.__dict__, ensure_ascii=False) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["image_path", "annotator", "mst_label", "notes"])
        with open(self.path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        return pd.DataFrame(rows)

    def align_pairs(
        self,
        annotator_a: str,
        annotator_b: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Retorna (labels_a, labels_b, image_paths) alinhados por imagem."""
        df = self.load()
        pivot = (
            df[df["annotator"].isin([annotator_a, annotator_b])]
            .pivot_table(
                index="image_path",
                columns="annotator",
                values="mst_label",
                aggfunc="last",
            )
            .dropna()
        )
        return (
            pivot[annotator_a].to_numpy(dtype=int),
            pivot[annotator_b].to_numpy(dtype=int),
            list(pivot.index),
        )


def cohens_kappa(a: Iterable[int], b: Iterable[int], labels: list[int] = MST_LABELS) -> float:
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    idx = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b):
        if x in idx and y in idx:
            conf[idx[x], idx[y]] += 1
    n = conf.sum()
    if n == 0:
        return float("nan")
    po = np.trace(conf) / n
    pa = conf.sum(axis=1) / n
    pb = conf.sum(axis=0) / n
    pe = float((pa * pb).sum())
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return float((po - pe) / (1.0 - pe))


def categorical_accuracy(a: Iterable[int], b: Iterable[int]) -> float:
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    if a.size == 0:
        return float("nan")
    return float((a == b).mean())


def bootstrap_agreement(
    a: Iterable[int],
    b: Iterable[int],
    metric: str = "kappa",
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Retorna (estatística, IC_lower, IC_upper) por percentile bootstrap."""
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    n = a.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    fn = cohens_kappa if metric == "kappa" else categorical_accuracy
    point = fn(a, b)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = fn(a[idx], b[idx])
    lo = float(np.nanquantile(stats, alpha / 2))
    hi = float(np.nanquantile(stats, 1 - alpha / 2))
    return (point, lo, hi)


def generate_report(
    annotator_a: str,
    annotator_b: str,
    kappa: tuple[float, float, float],
    accuracy: tuple[float, float, float],
    n_pairs: int,
) -> str:
    kappa_txt = f"{kappa[0]:.3f} (IC 95%: {kappa[1]:.3f}–{kappa[2]:.3f})"
    acc_txt = f"{accuracy[0]:.1%} (IC 95%: {accuracy[1]:.1%}–{accuracy[2]:.1%})"
    return (
        f"# Concordância humana MST — Etapa 1\n\n"
        f"- Anotadores: `{annotator_a}` × `{annotator_b}`\n"
        f"- Pares avaliados: **{n_pairs}**\n"
        f"- κ de Cohen: **{kappa_txt}**\n"
        f"- Exatidão categórica: **{acc_txt}**\n\n"
        f"Metodologia: bootstrap não paramétrico (10.000 réplicas, seed=42).\n"
        f"Referência: Cap. 4 §4.9 (validação humana interna).\n"
    )


def label_cli(
    sample: pd.DataFrame,
    annotator: str,
    store: HumanLabelStore,
    path_col: str = "path",
) -> None:
    """Loop CLI simples para rotulagem. Cada imagem exibe o path — o
    anotador abre a imagem à parte (viewer do SO) e digita 1..10, ``-1``
    para "não avaliável" ou ``q`` para sair.
    """
    already = set(store.load()["image_path"].tolist())
    remaining = [p for p in sample[path_col].tolist() if p not in already]
    print(f"Restam {len(remaining)} imagens para {annotator}. 'q' encerra a sessão.")
    for path in remaining:
        print(f"\n  {path}")
        raw = input("MST [1..10 | -1 | q]: ").strip()
        if raw.lower() == "q":
            break
        try:
            value = int(raw)
        except ValueError:
            print("  entrada inválida, pulando.")
            continue
        if value not in ({-1} | set(MST_LABELS)):
            print("  fora do intervalo, pulando.")
            continue
        notes = input("nota (enter p/ pular): ").strip()
        store.append(HumanLabel(image_path=path, annotator=annotator, mst_label=value, notes=notes))
