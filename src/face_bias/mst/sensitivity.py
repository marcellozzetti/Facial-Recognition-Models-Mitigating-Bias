"""Sensitivity analysis — classificadores MST alternativos ao SkinToneNet.

Salvaguarda contra propagação de viés herdado de um único classificador
(Cap. 4, §4.2 Etapa 1 e Cap. 3 Objetivo 2).

Backends suportados:
    - "skintonenet"  : wrapper principal (src/face_bias/mst/skintonenet.py)
    - "stone_monk"   : ChenglongMa/SkinToneClassifier (pypi: skin-tone-classifier),
                       palette Monk 10-classes — abordagem CV clássica
                       (detecção facial + segmentação + k-means).
    - callable custom : qualquer f(path) -> int em 1..10.

O objetivo é medir concordância entre backends (Cohen's κ pairwise) e
estabilidade de ranking. Este módulo NÃO reimplementa MST-KD (Caldeira
2024) nem baselines proprietários; o slot ``stone_monk`` está disponível
localmente hoje e mais candidatos podem ser plugados via ``register``.

Interface pública:
    class MSTSensitivityRunner:
        register(name, callable)
        run(image_paths) -> pd.DataFrame     # (N, n_backends) com labels
        pairwise_kappa(df) -> pd.DataFrame   # (n_backends, n_backends)

Ver também:
    - src/face_bias/mst/skintonenet.py — modelo principal
    - Cap 5, Risco 2 (mitigação declarada)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MST_LABELS = list(range(1, 11))  # 1..10 (escala Monk)

Predictor = Callable[[Path], int]


def stone_monk_predictor() -> Predictor:
    """Backend baseado em ``skin-tone-classifier`` (ChenglongMa) com paleta ``monk``.

    Instalação: ``pip install skin-tone-classifier``. Import é lazy para
    não onerar quem só usa o wrapper principal.
    """
    try:
        import stone  # type: ignore
    except ImportError as e:
        raise ImportError(
            "sensitivity backend 'stone_monk' requer `pip install skin-tone-classifier`."
        ) from e

    def _predict(path: Path) -> int:
        result = stone.process(str(path), image_type="color", palette="monk")
        faces = result.get("faces", []) if isinstance(result, dict) else []
        if not faces:
            return -1  # sem face detectada
        label = faces[0].get("tone_label", "")  # e.g. "monk_03"
        try:
            return int(str(label).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return -1

    return _predict


def _cohens_kappa(a: np.ndarray, b: np.ndarray, labels: list[int]) -> float:
    """Cohen's κ categórico — implementação direta para evitar dep de sklearn aqui.

    Ignora amostras com valor ``-1`` (backend falhou em detectar face).
    """
    mask = (a != -1) & (b != -1)
    a, b = a[mask], b[mask]
    n = a.size
    if n == 0:
        return float("nan")
    k = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    conf = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b):
        if x in idx and y in idx:
            conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    pa = conf.sum(axis=1) / n
    pb = conf.sum(axis=0) / n
    pe = float((pa * pb).sum())
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return float((po - pe) / (1.0 - pe))


class MSTSensitivityRunner:
    def __init__(self) -> None:
        self._backends: dict[str, Predictor] = {}

    def register(self, name: str, predictor: Predictor) -> None:
        self._backends[name] = predictor

    def register_stone_monk(self, name: str = "stone_monk") -> None:
        self.register(name, stone_monk_predictor())

    def register_skintonenet(
        self,
        name: str,
        infer,
        preprocess,
    ) -> None:
        """Adapta um ``SkinToneNetInference.infer`` para a interface path->int."""
        import torch

        def _predict(path: Path) -> int:
            with torch.inference_mode():
                tensor = preprocess(path).unsqueeze(0)
                probs = infer(tensor).cpu().numpy()[0]
            return int(np.argmax(probs)) + 1

        self.register(name, _predict)

    def run(
        self,
        image_paths: list[Path],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        if not self._backends:
            raise RuntimeError("Nenhum backend registrado.")
        rows: list[dict] = []
        total = len(image_paths)
        for i, path in enumerate(image_paths, 1):
            row: dict = {"path": str(path)}
            for name, fn in self._backends.items():
                try:
                    row[name] = int(fn(path))
                except Exception as e:  # noqa: BLE001
                    logger.warning("backend %s falhou em %s: %s", name, path, e)
                    row[name] = -1
            rows.append(row)
            if progress is not None:
                progress(i, total)
        return pd.DataFrame(rows)

    def pairwise_kappa(self, predictions: pd.DataFrame) -> pd.DataFrame:
        backends = [c for c in predictions.columns if c != "path"]
        n = len(backends)
        mat = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                k = _cohens_kappa(
                    predictions[backends[i]].to_numpy(),
                    predictions[backends[j]].to_numpy(),
                    MST_LABELS,
                )
                mat[i, j] = mat[j, i] = k
        return pd.DataFrame(mat, index=backends, columns=backends)
