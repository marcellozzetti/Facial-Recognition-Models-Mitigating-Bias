"""Pipeline Etapa 1: treino do extrator MST (ViT-B/16) no dataset STW.

Cap. 4 §4.3. Fine-tune do ViT-B/16 (ImageNet-21k) em classificação 10 classes
(Monk Skin Tone 1-10) usando o dataset STW (CC BY 4.0).

Dataset STW: ~14k imagens disponíveis de 28k totais (CASIA requer registro
separado na plataforma IdealTest). Distribuição altamente desequilibrada —
classe 10 tem apenas 22 imagens — exige class weighting.

Uso:
    python pipelines/04_train_mst.py \\
        --images-root data/raw/stw/images/all_cropped_images_by_class \\
        --train-csv   data/raw/stw/stw_train.csv \\
        --val-csv     data/raw/stw/stw_val.csv \\
        --output      outputs/etapa1/mst_extractor
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("pipelines.04_train_mst")

MST_CLASSES = 10
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class STWDataset(Dataset):
    """STW images organizadas por classe MST (1-10 → index 0-9)."""

    def __init__(self, stw_root: Path, manifest_df: pd.DataFrame, transform: transforms.Compose):
        self.stw_root  = Path(stw_root)
        self.paths     = manifest_df["path"].tolist()
        # class column é 1-10; converter para 0-9
        self.labels    = [int(c) - 1 for c in manifest_df["class"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_path = self.stw_root / self.paths[idx]
        with Image.open(img_path) as raw:
            img = raw.convert("RGB")
        return self.transform(img), int(self.labels[idx])


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_weighted_sampler(labels: list[int], n_classes: int) -> WeightedRandomSampler:
    """Sampler que equilibra classes via over/undersampling."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def build_vit_mst(num_classes: int = MST_CLASSES) -> nn.Module:
    """ViT-B/16 ImageNet-21k com head substituído para 10 classes MST."""
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #
EARLY_STOPPING_PATIENCE = 7


def train_mst_extractor(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    max_epochs:   int,
    lr_backbone:  float,
    lr_head:      float,
    weight_decay: float,
    seed:         int,
    class_weights: Optional[torch.Tensor] = None,
    patience:     int = EARLY_STOPPING_PATIENCE,
) -> dict:
    _set_seed(seed)
    model = build_vit_mst().to(device)

    head_ids = {id(p) for p in model.heads.parameters()}
    head_params     = [p for p in model.parameters() if id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    warmup_epochs  = max(1, int(0.05 * max_epochs))
    cosine_epochs  = max(1, max_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_acc  = -1.0
    best_mae  = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_acc, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss   = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_acc += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = train_loss_acc / max(n, 1)

        model.eval()
        preds_list, tgts_list, val_loss_acc, n = [], [], 0.0, 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss   = loss_fn(logits, y)
                val_loss_acc += loss.item() * x.size(0)
                n += x.size(0)
                preds_list.append(logits.argmax(-1).cpu().numpy())
                tgts_list.append(y.cpu().numpy())
        preds = np.concatenate(preds_list)
        tgts  = np.concatenate(tgts_list)
        val_loss = val_loss_acc / max(n, 1)
        val_acc  = float((preds == tgts).mean())
        # MAE entre classes ordinais (1-10 escala)
        val_mae  = float(np.abs(preds - tgts).mean())

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
            "val_mae":    round(val_mae, 6),
        })
        logger.info(
            "ep=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_mae=%.4f",
            epoch, train_loss, val_loss, val_acc, val_mae,
        )

        scheduler.step()

        # Checkpoint por acurácia (primary) + MAE (tiebreak)
        improved = val_acc > best_acc or (val_acc == best_acc and val_mae < best_mae)
        if improved:
            best_acc  = val_acc
            best_mae  = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping ep=%d (patience=%d)", epoch, patience)
                break

    return {"history": history, "best_val_acc": best_acc, "best_val_mae": best_mae, "best_state": best_state}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 1 — treino extrator MST ViT-B/16 no STW.")
    p.add_argument("--images-root", type=Path, required=True,
                   help="Pasta all_cropped_images_by_class do STW.")
    p.add_argument("--train-csv",   type=Path, required=True)
    p.add_argument("--val-csv",     type=Path, required=True)
    p.add_argument("--output",      type=Path, required=True)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max-epochs",    type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--lr-backbone",   type=float, default=1e-5)
    p.add_argument("--lr-head",       type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=0.05)
    p.add_argument("--patience",        type=int,   default=EARLY_STOPPING_PATIENCE,
                   help="Paciência do early stopping. Use 999 para desativar.")
    p.add_argument("--no-class-weight", action="store_true",
                   help="Desabilita weighted sampler (não recomendado — classes muito desequilibradas).")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    # CSV paths são relativos a data/raw/stw (parent.parent de images_root)
    stw_root = args.images_root.parent.parent
    train_ds = STWDataset(stw_root, train_df, _train_transform())
    val_ds   = STWDataset(stw_root, val_df,   _eval_transform())

    if args.no_class_weight:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
        class_weights = None
    else:
        sampler = make_weighted_sampler(train_ds.labels, MST_CLASSES)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
        counts = np.bincount(train_ds.labels, minlength=MST_CLASSES).astype(float)
        counts = np.maximum(counts, 1.0)
        cw = torch.tensor(1.0 / counts, dtype=torch.float32)
        cw = cw / cw.sum()
        class_weights = cw
        logger.info("Class weights: %s", [f"{w:.4f}" for w in cw.tolist()])

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = train_mst_extractor(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=args.max_epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        seed=args.seed,
        class_weights=class_weights,
        patience=args.patience,
    )
    elapsed = time.time() - t0

    ckpt = args.output / f"mst_vit_b16_seed{args.seed}.pt"
    torch.save({
        "state_dict":   result["best_state"],
        "best_val_acc": result["best_val_acc"],
        "best_val_mae": result["best_val_mae"],
        "seed":         args.seed,
        "num_classes":  MST_CLASSES,
    }, ckpt)
    (args.output / f"mst_vit_b16_seed{args.seed}_history.json").write_text(
        json.dumps(result["history"], indent=2), encoding="utf-8",
    )
    logger.info(
        "seed=%d best_val_acc=%.4f best_val_mae=%.4f tempo=%.1fs ckpt=%s",
        args.seed, result["best_val_acc"], result["best_val_mae"], elapsed, ckpt,
    )

    _plot_mst_curves(result["history"], args.output / f"mst_vit_b16_seed{args.seed}_curves.png")
    return 0


def _plot_mst_curves(history: list[dict], out_path: Path) -> None:
    """Gera curvas de treino MST: Loss (train+val) | Val Acc | Val MAE."""
    GREY_BG   = "#F9F9F7"
    BLUE_MAIN = "#1F2A4E"
    BLUE_MID  = "#3D68BB"
    ORANGE    = "#C4792A"
    GREEN     = "#2A7A4F"
    GREY_TEXT = "#6B7585"

    plt.rcParams.update({
        "figure.facecolor": GREY_BG, "axes.facecolor": GREY_BG,
        "axes.edgecolor": GREY_TEXT, "axes.labelcolor": BLUE_MAIN,
        "xtick.color": GREY_TEXT, "ytick.color": GREY_TEXT,
        "text.color": BLUE_MAIN, "font.family": "sans-serif",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    })

    epochs     = [h["epoch"]     for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_acc    = [h["val_acc"]    for h in history]
    val_mae    = [h["val_mae"]    for h in history]

    best_idx = val_acc.index(max(val_acc))
    best_ep  = epochs[best_idx]

    fig, (ax_loss, ax_acc, ax_mae) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(GREY_BG)

    # Loss
    ax_loss.plot(epochs, train_loss, color=BLUE_MID,  linewidth=2.2, linestyle="-",  label="train")
    ax_loss.plot(epochs, val_loss,   color=ORANGE,    linewidth=2.2, linestyle="--", label="val")
    ax_loss.set_title("Loss por época")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend(fontsize=10)

    # Val Acc
    ax_acc.plot(epochs, val_acc, color=GREEN, linewidth=2.2)
    ax_acc.axvline(best_ep, color=ORANGE, linestyle=":", linewidth=1.5, alpha=0.8)
    ax_acc.scatter([best_ep], [max(val_acc)], color=ORANGE, s=90, zorder=5,
                   label=f"Best ep.{best_ep} ({max(val_acc):.3f})")
    ax_acc.set_title("Val Accuracy por época")
    ax_acc.set_ylabel("Accuracy (top-1)")
    ax_acc.legend(fontsize=10)

    # Val MAE
    ax_mae.plot(epochs, val_mae, color=ORANGE, linewidth=2.2)
    ax_mae.axvline(best_ep, color=ORANGE, linestyle=":", linewidth=1.5, alpha=0.8)
    ax_mae.scatter([best_ep], [val_mae[best_idx]], color=ORANGE, s=90, zorder=5,
                   label=f"Best ep.{best_ep} (MAE {val_mae[best_idx]:.3f})")
    ax_mae.set_title("Val MAE por época")
    ax_mae.set_ylabel("MAE (escala 0–9)")
    ax_mae.legend(fontsize=10)

    for ax in (ax_loss, ax_acc, ax_mae):
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.25, linestyle=":")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("MST Extrator ViT-B/16 — dinâmica de treinamento (STW val)",
                 fontsize=14, fontweight="bold", color=BLUE_MAIN)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("pipelines.04_train_mst").info("Saved training curves → %s", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
