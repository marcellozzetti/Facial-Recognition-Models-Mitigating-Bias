"""Unit tests para src/face_bias/mst/preprocessing.py — Etapa 1 + Etapa 5."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from face_bias.mst import MSTClassifier, MSTFromRawImage, MSTResult
from face_bias.mst.preprocessing import (
    FAILURE_LOW_CONFIDENCE,
    FAILURE_NO_FACE,
    FAILURE_TOO_SMALL,
)


class FakeDetector:
    """Detector fake determinístico para testes offline."""

    def __init__(
        self,
        boxes: np.ndarray | None = None,
        confs: np.ndarray | None = None,
        landmarks: np.ndarray | None = None,
    ):
        self.boxes = boxes
        self.confs = confs
        self.landmarks = landmarks

    def detect(self, img_rgb: np.ndarray):
        return self.boxes, self.confs, self.landmarks


@pytest.fixture()
def classifier():
    return MSTClassifier(weights_path=None, device="cpu", allow_imagenet_only=True)


def _fake_landmarks(x1, y1, x2, y2):
    """Landmarks fake com olhos horizontais para bbox dada."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array([
        [cx - 20, cy - 10],  # left eye
        [cx + 20, cy - 10],  # right eye
        [cx, cy],            # nose
        [cx - 15, cy + 20],  # left mouth
        [cx + 15, cy + 20],  # right mouth
    ], dtype=np.float32)


@pytest.mark.unit
def test_result_as_row_success():
    r = MSTResult(
        source="x.jpg", success=True,
        probs=[0.1] * 10, mst_argmax=1,
        n_faces_detected=1,
        detected_bbox=(10, 20, 100, 200),
        detection_confidence=0.99,
        alignment_angle_deg=0.5,
    )
    row = r.as_row()
    assert row["p_1"] == pytest.approx(0.1)
    assert row["p_10"] == pytest.approx(0.1)
    assert row["bbox_x1"] == 10
    assert row["bbox_y2"] == 200
    assert "probs" not in row
    assert "detected_bbox" not in row


@pytest.mark.unit
def test_result_as_row_failure_has_null_probs():
    r = MSTResult(source="x.jpg", success=False, failure_reason=FAILURE_NO_FACE)
    row = r.as_row()
    for i in range(1, 11):
        assert row[f"p_{i}"] is None


@pytest.mark.unit
def test_no_face_returns_failure(classifier, tmp_path):
    img = Image.new("RGB", (300, 300), color=(120, 90, 60))
    p = tmp_path / "empty.jpg"
    img.save(p)
    pipeline = MSTFromRawImage(classifier, face_detector=FakeDetector(boxes=None))
    result = pipeline.process(p)
    assert result.success is False
    assert result.failure_reason == FAILURE_NO_FACE
    assert result.n_faces_detected == 0


@pytest.mark.unit
def test_low_confidence_returns_failure(classifier, tmp_path):
    img = Image.new("RGB", (300, 300))
    p = tmp_path / "low.jpg"
    img.save(p)
    pipeline = MSTFromRawImage(
        classifier,
        face_detector=FakeDetector(
            boxes=np.array([[50, 50, 250, 250]], dtype=np.float32),
            confs=np.array([0.6], dtype=np.float32),
            landmarks=_fake_landmarks(50, 50, 250, 250)[None, ...],
        ),
        min_confidence=0.9,
    )
    result = pipeline.process(p)
    assert result.success is False
    assert result.failure_reason == FAILURE_LOW_CONFIDENCE
    assert result.detection_confidence == pytest.approx(0.6)


@pytest.mark.unit
def test_too_small_face_returns_failure(classifier, tmp_path):
    img = Image.new("RGB", (300, 300))
    p = tmp_path / "tiny.jpg"
    img.save(p)
    pipeline = MSTFromRawImage(
        classifier,
        face_detector=FakeDetector(
            boxes=np.array([[10, 10, 30, 30]], dtype=np.float32),  # 20x20 < 40
            confs=np.array([0.99], dtype=np.float32),
            landmarks=_fake_landmarks(10, 10, 30, 30)[None, ...],
        ),
        min_face_size=40,
    )
    result = pipeline.process(p)
    assert result.success is False
    assert result.failure_reason == FAILURE_TOO_SMALL


@pytest.mark.unit
def test_successful_processing_returns_softmax(classifier, tmp_path):
    img = Image.new("RGB", (400, 400), color=(120, 100, 80))
    p = tmp_path / "face.jpg"
    img.save(p)
    pipeline = MSTFromRawImage(
        classifier,
        face_detector=FakeDetector(
            boxes=np.array([[100, 100, 300, 300]], dtype=np.float32),
            confs=np.array([0.98], dtype=np.float32),
            landmarks=_fake_landmarks(100, 100, 300, 300)[None, ...],
        ),
    )
    result = pipeline.process(p)
    assert result.success is True
    assert result.probs is not None
    assert len(result.probs) == 10
    assert sum(result.probs) == pytest.approx(1.0, abs=1e-4)
    assert result.mst_argmax in range(1, 11)
    assert result.detection_confidence == pytest.approx(0.98)


@pytest.mark.unit
def test_multi_face_policy_largest(classifier, tmp_path):
    img = Image.new("RGB", (400, 400))
    p = tmp_path / "multi.jpg"
    img.save(p)
    # duas faces: pequena (40x40) e grande (150x150)
    boxes = np.array([[10, 10, 50, 50], [200, 200, 350, 350]], dtype=np.float32)
    lms = np.stack([_fake_landmarks(*b) for b in boxes])
    pipeline = MSTFromRawImage(
        classifier,
        face_detector=FakeDetector(
            boxes=boxes,
            confs=np.array([0.99, 0.95], dtype=np.float32),
            landmarks=lms,
        ),
        multi_face_policy="largest",
    )
    result = pipeline.process(p)
    assert result.success is True
    assert result.n_faces_detected == 2
    # deve ter escolhido a grande (a segunda)
    assert result.detected_bbox == (200, 200, 350, 350)


@pytest.mark.unit
def test_load_error_returns_failure(classifier):
    pipeline = MSTFromRawImage(classifier, face_detector=FakeDetector())
    result = pipeline.process("caminho/inexistente.jpg")
    assert result.success is False
    assert result.failure_reason == "load_error"


@pytest.mark.unit
def test_invalid_multi_face_policy_raises(classifier):
    with pytest.raises(ValueError):
        MSTFromRawImage(classifier, face_detector=FakeDetector(),
                        multi_face_policy="bogus")


@pytest.mark.unit
def test_process_dataset_writes_parquet(classifier, tmp_path):
    # 3 imagens: 2 com face fake, 1 sem
    img = Image.new("RGB", (400, 400))
    for i in range(3):
        img.save(tmp_path / f"img_{i}.jpg")

    class MixedDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, img_rgb):
            self.calls += 1
            if self.calls == 3:
                return None, None, None
            boxes = np.array([[80, 80, 320, 320]], dtype=np.float32)
            return (
                boxes,
                np.array([0.99], dtype=np.float32),
                _fake_landmarks(80, 80, 320, 320)[None, ...],
            )

    pipeline = MSTFromRawImage(classifier, face_detector=MixedDetector())
    out = tmp_path / "out.parquet"
    df = pipeline.process_dataset(tmp_path, out, recursive=False, workers=1)
    assert out.exists()
    assert len(df) == 3
    assert int(df["success"].sum()) == 2
