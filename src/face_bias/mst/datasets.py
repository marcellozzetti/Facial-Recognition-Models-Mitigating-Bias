"""Loaders para datasets de tom de pele Monk (MST) — Etapa 1.

Datasets suportados:
    - MSTE (Google, Monk Skin Tone Examples): ~1500 imagens, público
    - CCv2 (Casual Conversations v2, Meta): frames MST-anotados, requer EULA
    - STW (Matias 2026): 42k imagens, aguardando publicação

Cada loader produz um pandas.DataFrame com colunas:
    file (str)      — caminho absoluto da imagem
    mst_label (int) — rótulo 1..10 na escala Monk
    source (str)    — origem: "mste" | "ccv2" | "stw"

Interface uniforme + factory ``build_mst_dataset`` para composição
(train = MSTE + CCv2 frames, por exemplo).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MST_MIN_LABEL = 1
MST_MAX_LABEL = 10
SOURCE_MSTE = "mste"
SOURCE_CCV2 = "ccv2"
SOURCE_STW = "stw"


def _validate_labels(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "mst_label" not in df.columns or "file" not in df.columns:
        raise ValueError(f"{source}: DataFrame precisa ter colunas 'file' e 'mst_label'.")
    bad = ~df["mst_label"].between(MST_MIN_LABEL, MST_MAX_LABEL)
    if bad.any():
        n = int(bad.sum())
        logger.warning("%s: %d linhas com mst_label fora de [1,10]; removendo.", source, n)
        df = df.loc[~bad].copy()
    df["source"] = source
    return df.reset_index(drop=True)


def load_mste(root: Path, labels_csv: Optional[Path] = None) -> pd.DataFrame:
    """Loader do MSTE (Google Monk Skin Tone Examples).

    Estrutura esperada (padrão do release público):
        root/
            images/*.jpg
            labels.csv        # colunas: file, mst_label (moda dos 3 anotadores)

    Se ``labels_csv`` for None, procura em ``root/labels.csv``.
    """
    root = Path(root)
    labels_csv = Path(labels_csv) if labels_csv else root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"MSTE labels não encontrados em {labels_csv}. "
            "Baixar de https://skintone.google/mste-dataset."
        )
    df = pd.read_csv(labels_csv)
    df["file"] = df["file"].apply(lambda p: str((root / p).resolve()))
    return _validate_labels(df, SOURCE_MSTE)


def load_ccv2(root: Path, labels_csv: Optional[Path] = None) -> pd.DataFrame:
    """Loader do Casual Conversations v2 (Meta).

    Espera frames pré-extraídos + CSV com mst_label anotado. A extração
    de frames dos vídeos originais é externa a este módulo (usar
    ``ffmpeg`` ou notebook de preprocess).

    Estrutura esperada:
        root/
            frames/*.jpg      # frames extraídos dos vídeos
            labels.csv        # file, mst_label
    """
    root = Path(root)
    labels_csv = Path(labels_csv) if labels_csv else root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"CCv2 labels não encontrados em {labels_csv}. "
            "Baixar de https://ai.meta.com/datasets/casual-conversations-v2/."
        )
    df = pd.read_csv(labels_csv)
    df["file"] = df["file"].apply(lambda p: str((root / p).resolve()))
    return _validate_labels(df, SOURCE_CCV2)


def load_stw(root: Path, split: str = "train") -> pd.DataFrame:
    """Loader do STW (Matias 2026). Requer acesso ao dataset — ainda não público.

    Se/quando o dataset for divulgado, atualizar este loader com o
    formato oficial anunciado. Enquanto isso, levanta erro claro.
    """
    root = Path(root)
    labels_csv = root / f"{split}_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"STW ainda não publicado. Aguardando release em "
            "https://arxiv.org/abs/2603.02475 ou contato direto com "
            "os autores (ver docs/ativo/email_skintonenet_authors.md)."
        )
    df = pd.read_csv(labels_csv)
    df["file"] = df["file"].apply(lambda p: str((root / p).resolve()))
    return _validate_labels(df, SOURCE_STW)


def build_mst_dataset(sources: Iterable[dict]) -> pd.DataFrame:
    """Concatena múltiplos datasets em um único DataFrame de treino/teste.

    Exemplo::

        df = build_mst_dataset([
            {"source": "mste", "root": "data/MSTE"},
            {"source": "ccv2", "root": "data/CCv2"},
        ])

    A ordem preserva-se; deduplicação por ``file`` é aplicada ao final.
    """
    loaders = {
        SOURCE_MSTE: load_mste,
        SOURCE_CCV2: load_ccv2,
        SOURCE_STW: load_stw,
    }
    parts: list[pd.DataFrame] = []
    for spec in sources:
        name = spec["source"]
        if name not in loaders:
            raise ValueError(f"source desconhecido: {name!r}. Válidos: {list(loaders)}")
        root = spec["root"]
        kwargs = {k: v for k, v in spec.items() if k not in {"source", "root"}}
        parts.append(loaders[name](root, **kwargs))
    if not parts:
        return pd.DataFrame(columns=["file", "mst_label", "source"])
    merged = pd.concat(parts, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["file"]).reset_index(drop=True)
    if before != len(merged):
        logger.info("Dedupe: %d → %d linhas (%d duplicatas).",
                    before, len(merged), before - len(merged))
    return merged


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Devolve a distribuição de mst_label (1..10) como Series ordenada."""
    return df["mst_label"].value_counts().reindex(range(1, 11), fill_value=0).sort_index()
