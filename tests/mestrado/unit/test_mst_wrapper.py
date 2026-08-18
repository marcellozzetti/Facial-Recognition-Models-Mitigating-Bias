"""Unit tests para src/face_bias/mst/*.py — Etapa 1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from face_bias.mst import (
    HumanLabel,
    HumanLabelStore,
    InferenceCache,
    MSTSensitivityRunner,
    MSTClassifier,
    WeightsUnavailableError,
    bootstrap_agreement,
    build_mst_backbone,
    categorical_accuracy,
    cohens_kappa,
    stratified_sample,
)


# ---------------------------------------------------------------------------
# skintonenet.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_mst_backbone_head_shape():
    net, embed_dim = build_mst_backbone(pretrained_imagenet=False)
    assert embed_dim == 768
    assert net.heads.out_features == 10  # 10 classes MST


@pytest.mark.unit
def test_skintonenet_raises_without_weights_and_optout():
    with pytest.raises(WeightsUnavailableError):
        MSTClassifier(weights_path=None, device="cpu")


@pytest.mark.unit
def test_skintonenet_smoke_mode_produces_softmax(tmp_path):
    """Modo ImageNet-only precisa gerar softmax válido para 10 classes."""
    infer = MSTClassifier(
        weights_path=None,
        device="cpu",
        allow_imagenet_only=True,
    )
    x = torch.randn(3, 3, 224, 224)
    probs = infer.infer(x)
    assert probs.shape == (3, 10)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


@pytest.mark.unit
def test_skintonenet_infer_bad_shape():
    infer = MSTClassifier(
        weights_path=None,
        device="cpu",
        allow_imagenet_only=True,
    )
    with pytest.raises(ValueError):
        infer.infer(torch.randn(3, 224, 224))  # sem batch dim


@pytest.mark.unit
def test_skintonenet_preprocess_shape(tmp_path):
    infer = MSTClassifier(
        weights_path=None,
        device="cpu",
        allow_imagenet_only=True,
    )
    img = Image.new("RGB", (300, 400), color=(120, 90, 60))
    path = tmp_path / "sample.png"
    img.save(path)
    tensor = infer.preprocess(path)
    assert tensor.shape == (3, 224, 224)


@pytest.mark.unit
def test_infer_batch_cache_hit(tmp_path):
    infer = MSTClassifier(
        weights_path=None,
        device="cpu",
        cache_dir=tmp_path,
        allow_imagenet_only=True,
    )
    img = Image.new("RGB", (256, 256), color=(80, 60, 40))
    p = tmp_path / "img.png"
    img.save(p)
    df1 = infer.infer_batch([p])
    df2 = infer.infer_batch([p])  # deveria vir do cache
    assert df1.iloc[0]["sha256"] == df2.iloc[0]["sha256"]
    for i in range(1, 11):
        assert df1.iloc[0][f"p_{i}"] == pytest.approx(df2.iloc[0][f"p_{i}"], abs=1e-6)


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inference_cache_roundtrip(tmp_path):
    cache = InferenceCache(tmp_path / "cache.sqlite", model_id="test")
    sha = hashlib.sha256(b"payload").hexdigest()
    probs = np.arange(10, dtype=np.float32) / 45.0  # soma = 1
    assert cache.get(sha) is None
    cache.put(sha, probs)
    got = cache.get(sha)
    assert got is not None
    np.testing.assert_allclose(got, probs)
    cache.close()


@pytest.mark.unit
def test_inference_cache_isolates_by_model_id(tmp_path):
    cache_a = InferenceCache(tmp_path / "c.sqlite", model_id="A")
    cache_b = InferenceCache(tmp_path / "c.sqlite", model_id="B")
    sha = "abc"
    cache_a.put(sha, np.zeros(10, dtype=np.float32))
    assert cache_a.get(sha) is not None
    assert cache_b.get(sha) is None
    cache_a.close()
    cache_b.close()


# ---------------------------------------------------------------------------
# validation.py — métricas
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cohens_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5]
    assert cohens_kappa(a, a) == pytest.approx(1.0)


@pytest.mark.unit
def test_cohens_kappa_chance_agreement():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 11, size=10_000)
    b = rng.integers(1, 11, size=10_000)
    k = cohens_kappa(a.tolist(), b.tolist())
    assert abs(k) < 0.05  # próximo de zero por chance


@pytest.mark.unit
def test_categorical_accuracy():
    assert categorical_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert categorical_accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_bootstrap_agreement_ci_covers_point():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 11, size=200)
    b = a.copy()
    # perturba 10% para κ ficar alto mas não 1.0
    idx = rng.choice(200, size=20, replace=False)
    b[idx] = rng.integers(1, 11, size=20)
    point, lo, hi = bootstrap_agreement(a.tolist(), b.tolist(), n_boot=500, seed=1)
    assert lo <= point <= hi
    assert 0.5 < point < 1.0


# ---------------------------------------------------------------------------
# validation.py — amostragem estratificada
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stratified_sample_respects_cells():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "race": rng.choice(["White", "Black", "Latinx/Hispanic"], size=n),
            "mst_pred": rng.integers(1, 11, size=n),
        }
    )
    sub = stratified_sample(df, n_target=150, seed=0)
    assert 100 <= len(sub) <= 200  # tolera oscilação por piso 1
    # cada célula presente no dataset deve ter >=1 amostra se n_target permitir
    non_empty_cells = df.groupby(["race", "mst_pred"]).size()
    subset_cells = sub.groupby(["race", "mst_pred"]).size()
    assert subset_cells.index.isin(non_empty_cells.index).all()


# ---------------------------------------------------------------------------
# validation.py — HumanLabelStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_human_label_store_roundtrip_and_pair_alignment(tmp_path):
    store = HumanLabelStore(tmp_path / "labels.jsonl")
    store.append(HumanLabel(image_path="img1.png", annotator="mestrando", mst_label=5))
    store.append(HumanLabel(image_path="img1.png", annotator="orientador", mst_label=4))
    store.append(HumanLabel(image_path="img2.png", annotator="mestrando", mst_label=8))
    # img2 sem orientador — deve ser descartada no pareamento
    a, b, paths = store.align_pairs("mestrando", "orientador")
    assert paths == ["img1.png"]
    assert a.tolist() == [5] and b.tolist() == [4]


# ---------------------------------------------------------------------------
# sensitivity.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sensitivity_runner_pairwise_kappa(tmp_path):
    runner = MSTSensitivityRunner()

    def fake_a(p: Path) -> int:
        return int(p.stem)

    def fake_b(p: Path) -> int:
        return int(p.stem)

    def fake_c(p: Path) -> int:
        # concorda 50% do tempo
        return int(p.stem) if int(p.stem) % 2 == 0 else (int(p.stem) % 10) + 1

    runner.register("A", fake_a)
    runner.register("B", fake_b)
    runner.register("C", fake_c)
    paths = [tmp_path / f"{i}.png" for i in range(1, 11)]
    for p in paths:
        p.touch()
    preds = runner.run(paths)
    assert list(preds.columns) == ["path", "A", "B", "C"]
    mat = runner.pairwise_kappa(preds)
    assert mat.loc["A", "B"] == pytest.approx(1.0)
    assert mat.loc["A", "A"] == 1.0


@pytest.mark.unit
def test_sensitivity_runner_empty_fails():
    runner = MSTSensitivityRunner()
    with pytest.raises(RuntimeError):
        runner.run([Path("x.png")])
