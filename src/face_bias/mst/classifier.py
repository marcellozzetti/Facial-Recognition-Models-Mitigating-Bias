"""Classificador MST 10-classes para a Etapa 1.

Cap. 4 §4.2 (Etapa 1) e §4.9 (validação humana interna).

O classificador principal desta pesquisa é treinado internamente sobre
MSTE (Google) + Casual Conversations v2 (Meta); ver ``trainer.py`` e
``pipelines/03a_train_mst_classifier.py``. Esta classe é o wrapper de
inferência: aceita qualquer state_dict compatível (nosso classificador,
o SkinToneNet oficial quando disponível, ou qualquer variante treinada
externamente).

Interface pública:
    class MSTClassifier:
        __init__(weights_path, device="cuda", cache_dir=None,
                 backbone="vit_b_16", allow_imagenet_only=False)
        infer(images: torch.Tensor) -> torch.Tensor            # (N, 10)
        infer_batch(image_paths: list[Path]) -> pd.DataFrame   # com cache
        preprocess(pil_or_path) -> torch.Tensor                # (3, 224, 224)

Ver também:
    - src/face_bias/mst/trainer.py       — treinamento do modelo próprio
    - src/face_bias/mst/preprocessing.py — camada auto-suficiente
    - src/face_bias/mst/validation.py    — protocolo humano interno
    - src/face_bias/mst/sensitivity.py   — comparação com MST alternativos
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

logger = logging.getLogger(__name__)

MST_N_CLASSES = 10
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SKINTONENET_PAPER_URL = "https://arxiv.org/abs/2603.02475"
CONTACT_HINT = (
    "Weights não encontrados. Treine o classificador próprio via "
    "``pipelines/03a_train_mst_classifier.py`` ou aponte para um "
    "state_dict externo compatível (10 classes)."
)


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _build_head(embed_dim: int) -> nn.Module:
    return nn.Linear(embed_dim, MST_N_CLASSES)


def build_mst_backbone(
    backbone: str = "vit_b_16",
    pretrained_imagenet: bool = True,
) -> tuple[nn.Module, int]:
    """Instancia o backbone + head 10-classes MST.

    Usado tanto pela inferência (``MSTClassifier``) quanto pelo treino
    (``MSTTrainer``). ViT-B/16 é o default por estar em ``torchvision``;
    trocar para ``timm`` ``vit_small_patch16_224`` quando se quiser
    reproduzir exatamente a arquitetura ViT-S do SkinToneNet.
    """
    if backbone != "vit_b_16":
        raise NotImplementedError(
            f"backbone={backbone!r} pendente. Suporte inicial só a vit_b_16."
        )
    weights = ViT_B_16_Weights.DEFAULT if pretrained_imagenet else None
    net = vit_b_16(weights=weights)
    embed_dim = net.heads.head.in_features  # 768
    net.heads = _build_head(embed_dim)
    return net, embed_dim


class WeightsUnavailableError(RuntimeError):
    """Levantado quando MSTClassifier é chamado sem weights nem opt-in ImageNet."""


class InferenceCache:
    """Cache SQLite por hash SHA-256 da imagem.

    Chave: sha256(bytes do arquivo). Valor: 10 floats (softmax MST).
    Evita re-inferir a mesma imagem entre execuções repetidas.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS mst_predictions (
            sha256 TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            probs BLOB NOT NULL
        )
    """

    def __init__(self, path: Path, model_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    @staticmethod
    def hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, sha: str) -> Optional[np.ndarray]:
        row = self._conn.execute(
            "SELECT probs FROM mst_predictions WHERE sha256=? AND model_id=?",
            (sha, self.model_id),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def put(self, sha: str, probs: np.ndarray) -> None:
        assert probs.shape == (MST_N_CLASSES,) and probs.dtype == np.float32
        self._conn.execute(
            "INSERT OR REPLACE INTO mst_predictions VALUES (?, ?, ?)",
            (sha, self.model_id, probs.tobytes()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class MSTClassifier:
    """Inferência MST 10-classes com cache opcional.

    Parâmetros
    ----------
    weights_path:
        Caminho para state_dict do SkinToneNet fine-tuned em STW. Se ``None``,
        exige ``allow_imagenet_only=True`` (modo smoke).
    device:
        ``"cuda"``, ``"cpu"`` ou ``torch.device``.
    cache_dir:
        Diretório para o cache SQLite; se ``None``, cache é desabilitado.
    backbone:
        Ver ``build_mst_backbone``.
    allow_imagenet_only:
        Opt-in explícito para rodar sem weights STW (útil só para testes).
    """

    def __init__(
        self,
        weights_path: Optional[Path],
        device: str | torch.device = "cuda",
        cache_dir: Optional[Path] = None,
        backbone: str = "vit_b_16",
        allow_imagenet_only: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.backbone_name = backbone
        self.transform = _default_transform()

        net, _ = build_mst_backbone(backbone=backbone, pretrained_imagenet=True)

        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"weights_path não encontrado: {weights_path}")
            self._load_state_dict(net, weights_path)
            self.model_id = f"mst_trained::{weights_path.name}"
        else:
            if not allow_imagenet_only:
                raise WeightsUnavailableError(CONTACT_HINT)
            logger.warning(
                "MSTClassifier rodando com head aleatória (ImageNet backbone only). "
                "Uso permitido APENAS para smoke tests. %s",
                CONTACT_HINT,
            )
            self.model_id = f"imagenet_only::{backbone}"

        net.eval()
        self.model = net.to(self.device)

        self.cache = (
            InferenceCache(Path(cache_dir) / "mst_cache.sqlite", self.model_id)
            if cache_dir is not None
            else None
        )

    def _load_state_dict(self, net: nn.Module, weights_path: Path) -> None:
        # torch>=2.11 torna weights_only=True o default; explícito para clareza.
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = net.load_state_dict(state, strict=False)
        head_shape = net.heads.weight.shape if hasattr(net.heads, "weight") else None
        if head_shape is not None and head_shape[0] != MST_N_CLASSES:
            raise ValueError(
                f"head shape {tuple(head_shape)} incompatível com MST ({MST_N_CLASSES})"
            )
        if missing or unexpected:
            logger.info(
                "load_state_dict: %d missing / %d unexpected keys (esperado ao "
                "adaptar de checkpoints do paper).",
                len(missing),
                len(unexpected),
            )

    def preprocess(self, source: str | Path | Image.Image) -> torch.Tensor:
        """Aplica o pipeline 224x224 + ImageNet norm. Retorna (3, 224, 224)."""
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            raise TypeError(f"source deve ser path ou PIL.Image, recebido {type(source)}")
        return self.transform(img)

    @torch.inference_mode()
    def infer(self, images: torch.Tensor) -> torch.Tensor:
        """Inferência sobre batch pré-processado. Espera (N, 3, 224, 224)."""
        if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"esperado (N, 3, {IMAGE_SIZE}, {IMAGE_SIZE}); recebido {tuple(images.shape)}"
            )
        logits = self.model(images.to(self.device))
        return F.softmax(logits, dim=-1)

    def infer_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Inferência com cache SQLite. Devolve DataFrame com colunas:
        ``path``, ``sha256``, ``mst_pred`` (argmax 1..10), ``p_1..p_10``.
        """
        paths = [Path(p) for p in image_paths]
        results: list[dict] = []
        pending: list[tuple[int, Path, str]] = []

        for idx, path in enumerate(paths):
            sha = InferenceCache.hash_file(path) if self.cache is not None else ""
            cached = self.cache.get(sha) if self.cache is not None else None
            if cached is not None:
                results.append(self._row(idx, path, sha, cached))
            else:
                pending.append((idx, path, sha))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            tensors = torch.stack([self.preprocess(p) for _, p, _ in chunk])
            probs = self.infer(tensors).cpu().numpy().astype(np.float32)
            for (idx, path, sha), row_probs in zip(chunk, probs):
                if self.cache is not None:
                    self.cache.put(sha, row_probs)
                results.append(self._row(idx, path, sha, row_probs))

        results.sort(key=lambda r: r["_order"])
        for r in results:
            r.pop("_order")
        return pd.DataFrame(results)

    @staticmethod
    def _row(order: int, path: Path, sha: str, probs: np.ndarray) -> dict:
        row = {
            "_order": order,
            "path": str(path),
            "sha256": sha,
            "mst_pred": int(np.argmax(probs)) + 1,  # 1..10 conforme escala Monk
        }
        for i, p in enumerate(probs, start=1):
            row[f"p_{i}"] = float(p)
        return row

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

    def __enter__(self) -> "MSTClassifier":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
