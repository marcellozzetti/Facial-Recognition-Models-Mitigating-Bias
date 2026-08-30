"""Unit + integration tests para src/face_bias/transfer/*.py — Etapa 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_bias.transfer import (
    BiSeNetPixelInfo,
    cosine_similarity,
    descriptors_for_crop,
    evaluate_pairs,
    evaluate_pairs_intersectional,
    face_resolution_pixels,
    find_best_threshold,
    intersectional_gap,
    luminance_lstar,
    pixel_info_from_bbox,
    pixel_info_from_mask,
    race_gap,
    sharpness_laplacian,
)


# ---------------------------------------------------------------------------
# rfw_eval
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cosine_similarity_identical_returns_one():
    a = np.array([[1.0, 0.0, 0.0]])
    assert cosine_similarity(a, a)[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_similarity_orthogonal_returns_zero():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert cosine_similarity(a, b)[0] == pytest.approx(0.0)


@pytest.mark.unit
def test_find_best_threshold_perfect_separation():
    sims = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr, acc = find_best_threshold(sims, labels)
    assert acc == pytest.approx(1.0)
    assert 0.3 < thr < 0.7


@pytest.mark.unit
def test_find_best_threshold_random_returns_finite():
    rng = np.random.default_rng(0)
    sims = rng.uniform(size=100)
    labels = rng.integers(0, 2, size=100)
    thr, acc = find_best_threshold(sims, labels)
    assert 0 < acc <= 1
    assert np.isfinite(thr)


@pytest.mark.unit
def test_evaluate_pairs_end_to_end():
    # 2 raças, 4 pares (2 same, 2 different)
    emb = {
        "a1": np.array([1.0, 0.0]),
        "a2": np.array([0.98, 0.19]),  # similar a a1
        "b1": np.array([0.0, 1.0]),
        "b2": np.array([0.2, 0.98]),   # similar a b1
    }
    pairs = pd.DataFrame({
        "path_a": ["a1", "a1", "b1", "b1"],
        "path_b": ["a2", "b1", "b2", "a2"],
        "is_same": [1, 0, 1, 0],
        "race": ["A", "A", "B", "B"],
    })
    result = evaluate_pairs(pairs, emb)
    assert set(result["race"]) == {"A", "B"}
    assert all(0 <= a <= 1 for a in result["accuracy"])


@pytest.mark.unit
def test_race_gap_zero_when_equal():
    df = pd.DataFrame({"race": ["A", "B"], "accuracy": [0.9, 0.9]})
    assert race_gap(df) == pytest.approx(0.0)


@pytest.mark.unit
def test_race_gap_computes_max_min_diff():
    df = pd.DataFrame({"race": ["A", "B", "C"], "accuracy": [0.9, 0.7, 0.8]})
    assert race_gap(df) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# bfw_eval
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_pairs_intersectional_creates_subgroups():
    emb = {
        f"{r}_{g}_{i}": np.random.default_rng(i).normal(size=8)
        for r in ("A", "B")
        for g in ("M", "F")
        for i in range(2)
    }
    pairs = []
    for r in ("A", "B"):
        for g in ("M", "F"):
            pairs.append({
                "path_a": f"{r}_{g}_0", "path_b": f"{r}_{g}_1",
                "is_same": 1, "race": r, "gender": g,
            })
    result = evaluate_pairs_intersectional(pd.DataFrame(pairs), emb)
    assert set(result["subgroup"]) == {"A_M", "A_F", "B_M", "B_F"}


@pytest.mark.unit
def test_intersectional_gap_returns_valid_number():
    df = pd.DataFrame({"subgroup": ["a", "b"], "accuracy": [0.8, 0.6]})
    assert intersectional_gap(df) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# quality_descriptors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_luminance_lstar_range():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    l = luminance_lstar(rgb)
    assert 0 <= l <= 100


@pytest.mark.unit
def test_luminance_black_and_white_extremes():
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert luminance_lstar(black) == pytest.approx(0.0, abs=1e-3)
    assert luminance_lstar(white) == pytest.approx(100.0, abs=1e-3)


@pytest.mark.unit
def test_sharpness_higher_for_edges():
    smooth = np.ones((50, 50, 3), dtype=np.uint8) * 128
    edged = smooth.copy()
    edged[:, 25:] = 255  # borda vertical brusca
    assert sharpness_laplacian(edged) > sharpness_laplacian(smooth)


@pytest.mark.unit
def test_face_resolution_pixels_multiplies_hw():
    assert face_resolution_pixels((10, 20)) == 200


@pytest.mark.unit
def test_descriptors_for_crop_returns_all_keys():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    d = descriptors_for_crop(rgb)
    assert set(d) == {"luminance_lstar", "sharpness_laplacian", "face_area_pixels"}


# ---------------------------------------------------------------------------
# bisenet_pixel_info
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pixel_info_from_mask_fraction():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:8, 3:8] = 1  # 25 pixels
    assert pixel_info_from_mask(mask) == pytest.approx(25 / 100)


@pytest.mark.unit
def test_pixel_info_from_bbox_fraction():
    frac = pixel_info_from_bbox((10, 20, 30, 40), (100, 100))
    # bbox = 20x20 = 400 pixels; imagem = 10000 → 0.04
    assert frac == pytest.approx(0.04)


@pytest.mark.unit
def test_bisenet_wrapper_fallback_to_bbox_when_no_weights(tmp_path):
    bisenet = BiSeNetPixelInfo(weights_path=None)
    assert not bisenet.available()
    frac = bisenet.compute(
        np.zeros((100, 100, 3), dtype=np.uint8),
        bbox_fallback=(20, 20, 50, 50),
    )
    assert frac == pytest.approx((30 * 30) / (100 * 100))
