"""Camada auto-suficiente para MST em datasets crus — Etapa 1 + Etapa 5.

Cap. 4 §4.2 (Etapa 1) e §4.2 (Etapa 5, transferência RFW/BFW).

Motivação (feedback do orientador, reunião Ago/2026):
    RFW e BFW não têm anotação MST. Para rodar o modelo condicionado
    da Etapa 3 nesses datasets, precisamos gerar o vetor MST na hora,
    a partir da imagem crua. Esta camada é o que torna o classificador
    MST auto-suficiente: detecta rosto → alinha → normaliza → classifica.

Fluxo interno:
    raw image (path | PIL | ndarray)
        ↓ face_detector.detect()  (MTCNN default, plugável)
        ↓ validate (size, confidence)
        ↓ crop + align (eye landmarks)
        ↓ resize 224x224 + ImageNet normalize
        ↓ MSTClassifier.infer
    MSTResult { probs, mst_argmax, bbox, confidence, alignment_angle, ... }

Interface:
    class MSTFromRawImage:
        process(source) -> MSTResult
        process_batch(sources, workers=4) -> list[MSTResult]
        process_dataset(image_dir, out_parquet, recursive=True) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from face_bias.mst.classifier import IMAGE_SIZE, MST_N_CLASSES, MSTClassifier

logger = logging.getLogger(__name__)

SourceLike = Union[str, Path, Image.Image, np.ndarray]

FAILURE_NO_FACE = "no_face"
FAILURE_LOW_CONFIDENCE = "low_confidence"
FAILURE_TOO_SMALL = "too_small"
FAILURE_LOAD_ERROR = "load_error"
FAILURE_ALIGNMENT = "alignment_failed"


# --------------------------------------------------------------------------- #
# Resultado                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class MSTResult:
    """Resultado da inferência auto-suficiente sobre uma imagem crua."""

    source: str
    success: bool
    probs: Optional[list[float]] = None       # softmax 10-dim ou None
    mst_argmax: Optional[int] = None          # 1..10 ou None
    failure_reason: Optional[str] = None
    n_faces_detected: int = 0
    detected_bbox: Optional[tuple[int, int, int, int]] = None
    detection_confidence: Optional[float] = None
    alignment_angle_deg: Optional[float] = None

    def as_row(self) -> dict:
        row = asdict(self)
        if self.probs is not None:
            for i, p in enumerate(self.probs, start=1):
                row[f"p_{i}"] = float(p)
            row.pop("probs")
        else:
            for i in range(1, MST_N_CLASSES + 1):
                row[f"p_{i}"] = None
            row.pop("probs")
        if self.detected_bbox is not None:
            x1, y1, x2, y2 = self.detected_bbox
            row["bbox_x1"] = x1
            row["bbox_y1"] = y1
            row["bbox_x2"] = x2
            row["bbox_y2"] = y2
        row.pop("detected_bbox")
        return row


# --------------------------------------------------------------------------- #
# Detector — interface plugável                                               #
# --------------------------------------------------------------------------- #
class _MTCNNAdapter:
    """Adaptador do MTCNN (facenet-pytorch) para a interface esperada."""

    def __init__(self, device: str | torch.device = "cpu"):
        from facenet_pytorch import MTCNN

        self._mtcnn = MTCNN(keep_all=True, device=str(device))

    def detect(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes, probs, landmarks = self._mtcnn.detect(img_rgb, landmarks=True)
        return boxes, probs, landmarks


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
class MSTFromRawImage:
    """Pipeline: imagem crua → vetor MST (ou MSTResult com failure_reason)."""

    def __init__(
        self,
        classifier: MSTClassifier,
        face_detector=None,
        min_face_size: int = 40,
        min_confidence: float = 0.9,
        multi_face_policy: str = "largest",
    ):
        if multi_face_policy not in {"largest", "highest_conf", "first"}:
            raise ValueError(f"multi_face_policy inválida: {multi_face_policy!r}")
        self.classifier = classifier
        self.detector = face_detector or _MTCNNAdapter(device=str(classifier.device))
        self.min_face_size = int(min_face_size)
        self.min_confidence = float(min_confidence)
        self.multi_face_policy = multi_face_policy

    # ---- ingestão de fontes heterogêneas ----
    @staticmethod
    def _to_rgb(source: SourceLike) -> tuple[str, np.ndarray]:
        if isinstance(source, (str, Path)):
            with Image.open(source) as raw:
                img = raw.convert("RGB")
            return str(source), np.asarray(img)
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
            return "<PIL>", np.asarray(img)
        if isinstance(source, np.ndarray):
            arr = source
            if arr.ndim == 3 and arr.shape[2] == 3:
                return "<ndarray>", arr.astype(np.uint8)
        raise TypeError(f"source deve ser path, PIL.Image ou ndarray HxWx3; recebido {type(source)}")

    # ---- seleção quando há múltiplas faces ----
    def _pick_face(
        self,
        boxes: np.ndarray,
        confs: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, float]:
        if self.multi_face_policy == "first":
            i = 0
        elif self.multi_face_policy == "highest_conf":
            i = int(np.argmax(confs))
        else:  # largest
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            i = int(np.argmax(areas))
        return i, boxes[i], landmarks[i], float(confs[i])

    # ---- alinhamento via olhos ----
    @staticmethod
    def _align_and_crop(
        img_rgb: np.ndarray,
        bbox: np.ndarray,
        landmark: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float]:
        import cv2

        x1, y1, x2, y2 = bbox.astype(int).tolist()
        border = 40
        h, w = img_rgb.shape[:2]
        cx1, cy1 = max(0, x1 - border), max(0, y1 - border)
        cx2, cy2 = min(w, x2 + border), min(h, y2 + border)
        cropped = img_rgb[cy1:cy2, cx1:cx2]
        if cropped.size == 0 or min(cropped.shape[:2]) < 10:
            return None, 0.0

        # olhos em landmarks 0 (esq) e 1 (dir), coords no espaço original
        lex, ley = float(landmark[0][0]) - cx1, float(landmark[0][1]) - cy1
        rex, rey = float(landmark[1][0]) - cx1, float(landmark[1][1]) - cy1
        dx, dy = rex - lex, rey - ley
        angle = float(np.degrees(np.arctan2(dy, dx)))
        center = ((lex + rex) / 2.0, (ley + rey) / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(
            cropped, M, (cropped.shape[1], cropped.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        resized = cv2.resize(rotated, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return resized, angle

    # ---- API pública ----
    def process(self, source: SourceLike) -> MSTResult:
        try:
            src_id, img_rgb = self._to_rgb(source)
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao carregar %r: %s", source, e)
            return MSTResult(source=str(source), success=False, failure_reason=FAILURE_LOAD_ERROR)

        boxes, confs, landmarks = self.detector.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return MSTResult(source=src_id, success=False,
                             failure_reason=FAILURE_NO_FACE, n_faces_detected=0)
        n = int(len(boxes))
        i, bbox, landmark, conf = self._pick_face(boxes, confs, landmarks)

        if conf < self.min_confidence:
            return MSTResult(
                source=src_id, success=False,
                failure_reason=FAILURE_LOW_CONFIDENCE,
                n_faces_detected=n,
                detected_bbox=tuple(bbox.astype(int).tolist()),
                detection_confidence=conf,
            )
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        if min(w, h) < self.min_face_size:
            return MSTResult(
                source=src_id, success=False,
                failure_reason=FAILURE_TOO_SMALL,
                n_faces_detected=n,
                detected_bbox=tuple(bbox.astype(int).tolist()),
                detection_confidence=conf,
            )

        aligned, angle = self._align_and_crop(img_rgb, bbox, landmark)
        if aligned is None:
            return MSTResult(
                source=src_id, success=False,
                failure_reason=FAILURE_ALIGNMENT,
                n_faces_detected=n,
                detected_bbox=tuple(bbox.astype(int).tolist()),
                detection_confidence=conf,
            )

        pil = Image.fromarray(aligned)
        tensor = self.classifier.transform(pil).unsqueeze(0)
        probs = self.classifier.infer(tensor).cpu().numpy()[0]

        return MSTResult(
            source=src_id, success=True,
            probs=probs.astype(np.float32).tolist(),
            mst_argmax=int(np.argmax(probs)) + 1,
            n_faces_detected=n,
            detected_bbox=tuple(bbox.astype(int).tolist()),
            detection_confidence=conf,
            alignment_angle_deg=angle,
        )

    def process_batch(
        self,
        sources: list[SourceLike],
        workers: int = 4,
    ) -> list[MSTResult]:
        # I/O-bound (leitura de arquivo, MTCNN CPU), portanto ThreadPool cabe
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self.process, sources))

    def process_dataset(
        self,
        image_dir: Path,
        out_parquet: Path,
        recursive: bool = True,
        pattern: str = "*.jpg",
        workers: int = 4,
    ) -> pd.DataFrame:
        image_dir = Path(image_dir)
        globber = image_dir.rglob if recursive else image_dir.glob
        paths = sorted(list(globber(pattern)) + list(globber("*.png")))
        if not paths:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {image_dir}.")
        logger.info("Auto-preprocess sobre %d imagens em %s.", len(paths), image_dir)
        results = self.process_batch(paths, workers=workers)
        df = pd.DataFrame([r.as_row() for r in results])
        out_parquet = Path(out_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        n_ok = int(df["success"].sum())
        logger.info("Auto-preprocess: %d/%d sucesso; parquet em %s.",
                    n_ok, len(df), out_parquet)
        return df
