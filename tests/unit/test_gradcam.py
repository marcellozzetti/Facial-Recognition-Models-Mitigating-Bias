"""Unit tests for the Grad-CAM helpers."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from face_bias.interpretability.gradcam import (
    GradCAM,
    denormalise_image,
    find_target_layer,
    overlay_heatmap,
    plot_grid,
)


class _ToyConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        feats = torch.relu(self.conv(x))
        return self.fc(self.pool(feats).flatten(1))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.unit
def test_gradcam_returns_heatmap_and_class() -> None:
    model = _ToyConvNet()
    image = torch.randn(3, 16, 16)
    with GradCAM(model, model.conv) as cam_op:
        cam, predicted = cam_op(image)
    assert cam.shape == (16, 16)
    assert 0 <= cam.min() <= cam.max() <= 1.0
    assert 0 <= predicted < 3


@pytest.mark.unit
def test_gradcam_target_class_overrides_argmax() -> None:
    model = _ToyConvNet()
    image = torch.randn(3, 16, 16)
    with GradCAM(model, model.conv) as cam_op:
        _, predicted = cam_op(image, target_class=2)
    assert predicted == 2


@pytest.mark.unit
def test_gradcam_hooks_removed_on_exit() -> None:
    model = _ToyConvNet()
    cam_op = GradCAM(model, model.conv)
    assert len(cam_op._handles) == 2
    cam_op.remove_hooks()
    assert cam_op._handles == []


@pytest.mark.unit
def test_find_target_layer_falls_back() -> None:
    """Without backbone.layer4 the helper raises informatively."""
    model = _ToyConvNet()
    with pytest.raises(ValueError, match="auto-locate"):
        find_target_layer(model)


@pytest.mark.unit
def test_denormalise_image_round_trip() -> None:
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    img = torch.zeros(3, 8, 8)  # normalised "grey" image
    rgb = denormalise_image(img, mean, std)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    # all-zero normalised tensor inverts to mean*255 = 127
    assert abs(int(rgb.mean()) - 127) <= 1


@pytest.mark.unit
def test_overlay_heatmap_shape() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cam = np.linspace(0, 1, 16 * 16).reshape(16, 16).astype(np.float32)
    blend = overlay_heatmap(img, cam)
    assert blend.shape == (16, 16, 3)
    assert blend.dtype == np.uint8


@pytest.mark.unit
def test_plot_grid_writes_png(tmp_path) -> None:
    examples = [
        {
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "overlay": np.full((16, 16, 3), 64, dtype=np.uint8),
            "true_class": 0,
            "predicted_class": 1,
        },
        {
            "image": np.full((16, 16, 3), 200, dtype=np.uint8),
            "overlay": np.full((16, 16, 3), 100, dtype=np.uint8),
            "true_class": 2,
            "predicted_class": 2,
        },
    ]
    out = tmp_path / "gradcam.png"
    plot_grid(examples, ["a", "b", "c"], out)
    assert out.exists()
    assert out.with_suffix(".pdf").exists()
