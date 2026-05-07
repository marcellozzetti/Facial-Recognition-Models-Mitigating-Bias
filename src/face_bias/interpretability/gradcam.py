"""Grad-CAM (Selvaraju et al., 2017) for face-recognition models.

Highlights image regions whose activations most influence the predicted
class. Used in the dissertation to inspect *what* the model is looking
at when it classifies a race — a qualitative complement to the
fairness audit.

Implementation is self-contained (no external Grad-CAM library) so the
package stays light. Hooks are registered on the chosen target layer
(typically the last conv block of the backbone), the heatmap is
computed from gradients-weighted activations, and finally overlaid on
the de-normalised input image.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Forward + backward hooks on ``target_layer`` to capture activations and gradients."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_module, _inp, output):
            self.activations = output.detach()

        def bwd_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()

    def __call__(
        self, image: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Return ``(heatmap_2d, predicted_class)`` for a single CHW image."""
        self.model.eval()
        was_training = self.model.training
        try:
            x = image.unsqueeze(0) if image.dim() == 3 else image
            x = x.requires_grad_(False)
            logits = self.model(x)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            self.model.zero_grad()
            score = logits[:, target_class].sum()
            score.backward(retain_graph=False)

            assert self.activations is not None
            assert self.gradients is not None
            # Average pool gradients over spatial dims to get per-channel weights.
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam, target_class
        finally:
            if was_training:
                self.model.train()


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return the LResNet50E_IR's last conv block (``backbone.layer4``)."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4
    raise ValueError("Could not auto-locate a Grad-CAM target. Pass target_layer explicitly.")


def denormalise_image(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Recover an HxWx3 RGB uint8 array from a normalised CHW tensor."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    img = img * std_arr + mean_arr
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend ``cam`` (2-D, [0, 1]) onto ``image_rgb`` (HxWx3 uint8) with JET colormap."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap, alpha, 0)


def plot_grid(
    examples: list[dict],
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "Grad-CAM examples",
) -> Path:
    """Render N rows × 2 columns (original | overlay) into one figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.0 * n))
    if n == 1:
        axes = np.array([axes])

    for row, ex in enumerate(examples):
        for col in range(2):
            axes[row, col].axis("off")

        axes[row, 0].imshow(ex["image"])
        axes[row, 0].set_title(
            f"true: {class_names[ex['true_class']]}\npred: {class_names[ex['predicted_class']]}",
            fontsize=9,
        )
        axes[row, 1].imshow(ex["overlay"])
        axes[row, 1].set_title(f"Grad-CAM ({class_names[ex['predicted_class']]})", fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Wrote Grad-CAM grid to {output_path}")
    return output_path
