"""t-SNE projection of feature embeddings, coloured by demographic class.

Used to inspect how well the model separates races in feature space.
The dissertation cites this as a qualitative complement to the per-class
F1 / fairness audit: tight, well-separated clusters mean the embedding
geometry already encodes race; overlapping clusters often correlate with
the worst-performing classes in the confusion matrix.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

DEFAULT_PALETTE = [
    "#332288",  # indigo
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # cyan
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]


@torch.no_grad()
def compute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the backbone over ``dataloader`` and return ``(features, labels)``.

    ``model`` must expose ``extract_features(x)`` returning the
    pre-classifier embedding (LResNet50E_IR does). When ``max_samples`` is
    set, sampling stops after that many examples — useful for t-SNE
    plots, which become unreadable above ~5k points.
    """
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen = 0
    for images, batch_labels in dataloader:
        images = images.to(device, non_blocking=True)
        embeddings = model.extract_features(images)
        feats.append(embeddings.cpu().numpy())
        labels.append(batch_labels.numpy())
        seen += images.size(0)
        if max_samples is not None and seen >= max_samples:
            break

    features = np.concatenate(feats)
    targets = np.concatenate(labels)
    if max_samples is not None and len(features) > max_samples:
        features = features[:max_samples]
        targets = targets[:max_samples]
    return features, targets


def run_tsne(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Project ``features`` to 2-D via sklearn's TSNE."""
    perp = min(perplexity, max(5.0, (len(features) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    return tsne.fit_transform(features)


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "t-SNE of feature embeddings",
) -> Path:
    """Scatter plot with one colour per class. Returns ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, name in enumerate(class_names):
        mask = labels == cls_idx
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            color=DEFAULT_PALETTE[cls_idx % len(DEFAULT_PALETTE)],
            label=name,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(loc="best", frameon=True, fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist features / labels / projected coords for later re-plotting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": output_dir / "embeddings.npy",
        "labels": output_dir / "labels.npy",
        "coords": output_dir / "tsne_coords.npy",
        "metadata": output_dir / "tsne_metadata.json",
    }
    np.save(paths["features"], features)
    np.save(paths["labels"], labels)
    np.save(paths["coords"], coords)
    paths["metadata"].write_text(
        json.dumps(
            {
                "class_names": list(class_names),
                "num_samples": int(features.shape[0]),
                "embedding_dim": int(features.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info(f"Saved t-SNE artefacts under {output_dir}")
    return paths
