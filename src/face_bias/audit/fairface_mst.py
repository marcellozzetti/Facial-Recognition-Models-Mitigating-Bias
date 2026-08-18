"""Auditoria fenotípica do FairFace via MSTClassifier — Etapa 2.

Cap. 4 §4.2 (Etapa 2). Combina a inferência MST da Etapa 1 (parquet
gerado por ``pipelines/03_mst_inference.py``) com os rótulos raciais do
FairFace, produzindo o insumo canônico para a matriz Contribuição 2
(``cross_matrix.build_matrix``).

Também expõe ``audit_fairface`` para o caso em que se deseja rodar
inferência + junção em uma única chamada (útil para testes).

Ver também:
    - src/face_bias/audit/cross_matrix.py — consumidor
    - src/face_bias/mst/classifier.py — provedor da inferência
    - pipelines/04_fairface_audit.py — orquestração CLI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FAIRFACE_FILE_COL = "file"
FAIRFACE_RACE_COL = "race"


def load_fairface_labels(csv_path: Path) -> pd.DataFrame:
    """Lê o CSV oficial do FairFace (val ou train) e devolve DataFrame.

    Exige as colunas ``file`` (path relativo à raiz do dataset) e ``race``.
    """
    df = pd.read_csv(csv_path)
    missing = {FAIRFACE_FILE_COL, FAIRFACE_RACE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} não contém colunas obrigatórias: {sorted(missing)}")
    return df[[FAIRFACE_FILE_COL, FAIRFACE_RACE_COL]].copy()


def join_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    dataset_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Junção entre predições MST (por caminho de imagem) e rótulos FairFace.

    A coluna ``path`` do parquet de predições contém caminhos absolutos
    (foi assim que o pipeline 03 gravou). O CSV do FairFace tem paths
    relativos. Normalizamos aqui para garantir a junção correta.
    """
    if dataset_root is not None:
        dataset_root = Path(dataset_root).resolve()
        labels = labels.copy()
        labels["_abs"] = labels[FAIRFACE_FILE_COL].apply(
            lambda rel: str((dataset_root / rel).resolve())
        )
        pred = predictions.copy()
        pred["_abs"] = pred["path"].apply(lambda p: str(Path(p).resolve()))
        merged = pred.merge(labels, on="_abs", how="inner").drop(columns=["_abs"])
    else:
        # Fallback: junção por basename (imagens únicas por nome)
        pred = predictions.copy()
        pred["_key"] = pred["path"].apply(lambda p: Path(p).name)
        lab = labels.copy()
        lab["_key"] = lab[FAIRFACE_FILE_COL].apply(lambda p: Path(p).name)
        merged = pred.merge(lab, on="_key", how="inner").drop(columns=["_key"])
    if merged.empty:
        raise RuntimeError(
            "Junção resultou em zero linhas. Verifique dataset_root ou nomes de arquivo."
        )
    return merged


def audit_from_files(
    mst_predictions_parquet: Path,
    fairface_labels_csv: Path,
    dataset_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Etapa 2 padrão: carrega predições MST + labels FairFace + junta."""
    predictions = pd.read_parquet(mst_predictions_parquet)
    labels = load_fairface_labels(fairface_labels_csv)
    merged = join_predictions(predictions, labels, dataset_root=dataset_root)
    logger.info(
        "Audit: %d predições × %d labels → %d linhas casadas.",
        len(predictions), len(labels), len(merged),
    )
    return merged


def audit_fairface(
    classifier,
    fairface_val_dir: Path,
    fairface_labels_csv: Path,
    batch_size: int = 32,
    limit: int = 0,
) -> pd.DataFrame:
    """Roda inferência MSTClassifier sobre imagens do FairFace + junta labels.

    Útil quando se quer rodar Etapa 1 + Etapa 2 em um único processo
    (por exemplo, em smoke tests). Para produção use o parquet gerado
    pelo pipeline 03 + ``audit_from_files``.
    """
    labels = load_fairface_labels(fairface_labels_csv)
    if limit > 0:
        labels = labels.head(limit).copy()
    paths = [Path(fairface_val_dir) / rel for rel in labels[FAIRFACE_FILE_COL].tolist()]
    predictions = classifier.infer_batch(paths, batch_size=batch_size)
    return join_predictions(predictions, labels, dataset_root=fairface_val_dir)
