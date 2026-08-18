"""Treinamento do classificador MST próprio — Etapa 1.

Cap. 4 §4.2 (Etapa 1). Substitui a dependência do SkinToneNet como
método principal, treinando internamente sobre MSTE + CCv2 (ver
``datasets.py``). O state_dict resultante é diretamente compatível
com ``MSTClassifier(weights_path=...)``.

Receita default (alinhada à receita reportada por Matias 2026 para o
SkinToneNet, para permitir comparação):
    - Backbone: ViT-B/16 pretrained ImageNet (ou ConvNeXt-T opcional)
    - Head: linear 10 classes (escala Monk)
    - Loss: cross-entropy
    - Optimizer: AdamW, lr 1e-4 backbone / 1e-3 head
    - Scheduler: ReduceLROnPlateau (patience 3, factor 0.5)
    - Early stopping: 5 épocas sem melhora do F1 macro em val
    - Split: 80/20 estratificado por mst_label, 3 sementes {42, 1, 2}

Interface:
    class MSTTrainer:
        train(train_df, val_df) -> TrainResult
        save_checkpoint(path)

    def stratified_split(df, val_frac=0.2, seed=42) -> (train_df, val_df)
    def build_dataloaders(df, ...) -> DataLoader
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from face_bias.mst.classifier import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MST_N_CLASSES,
    build_mst_backbone,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset PyTorch                                                             #
# --------------------------------------------------------------------------- #
class MSTImageDataset(Dataset):
    """Dataset PyTorch minimalista: (image_tensor, mst_label_zero_indexed)."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.files = df["file"].tolist()
        # rótulos MST são 1..10; PyTorch cross-entropy espera 0..9
        self.labels = (df["mst_label"].astype(int) - 1).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.files[idx]) as raw:
            img = raw.convert("RGB")
        return self.transform(img), int(self.labels[idx])


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado por mst_label. Garante pelo menos 1 amostra por
    classe em cada split quando possível."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, group in df.groupby("mst_label", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MSTImageDataset(train_df, _train_transform())
    val_ds = MSTImageDataset(val_df, _eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    best_val_acc: float
    best_val_f1_macro: float
    history: list[dict] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
    seed: int = 42


def _f1_macro_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """F1 macro implementado direto para evitar dep circular de sklearn aqui."""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MSTTrainer:
    """Treinamento MST 10-classes com early stopping por F1 macro em val."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        backbone: str = "vit_b_16",
        lr_backbone: float = 1e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 0.05,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        scheduler_patience: int = 3,
        seed: int = 42,
    ):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.backbone_name = backbone
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_patience = scheduler_patience
        self.seed = seed

        _set_seed(seed)
        self.model, _ = build_mst_backbone(backbone=backbone, pretrained_imagenet=True)
        self.model.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # LR distinto entre backbone e head; head aprende mais rápido
        head_params = list(self.model.heads.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.model.parameters() if id(p) not in head_ids]
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )

    @torch.inference_mode()
    def _evaluate(self, loader: DataLoader, loss_fn: nn.Module) -> dict:
        self.model.eval()
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            targets.append(y.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(targets)
        acc = float((y_pred == y_true).mean())
        f1 = _f1_macro_per_class(y_true, y_pred, MST_N_CLASSES)
        return {"val_loss": total_loss / max(n, 1), "val_acc": acc, "val_f1_macro": f1}

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
        checkpoint_dir: Optional[Path] = None,
    ) -> TrainResult:
        train_loader, val_loader = build_dataloaders(
            train_df, val_df, batch_size=batch_size, num_workers=num_workers
        )
        optimizer = self._build_optimizer()
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.scheduler_patience
        )
        loss_fn = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_state: Optional[dict] = None
        patience_left = self.early_stopping_patience
        history: list[dict] = []

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running = 0.0
            seen = 0
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.item()) * x.size(0)
                seen += x.size(0)
            train_loss = running / max(seen, 1)

            metrics = self._evaluate(val_loader, loss_fn)
            entry = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(entry)
            logger.info(
                "seed=%d ep=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                self.seed, epoch, train_loss,
                metrics["val_loss"], metrics["val_acc"], metrics["val_f1_macro"],
            )
            scheduler.step(metrics["val_f1_macro"])

            if metrics["val_f1_macro"] > best_f1:
                best_f1 = metrics["val_f1_macro"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience_left = self.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping @ epoch %d (best_f1=%.4f)", epoch, best_f1)
                    break

        result = TrainResult(
            best_val_acc=max((h["val_acc"] for h in history), default=0.0),
            best_val_f1_macro=best_f1 if best_f1 >= 0 else 0.0,
            history=history,
            seed=self.seed,
        )
        if checkpoint_dir is not None and best_state is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"best_seed{self.seed}.pt"
            payload = {
                "state_dict": best_state,
                "seed": self.seed,
                "backbone": self.backbone_name,
                "val_f1_macro": best_f1,
                "n_classes": MST_N_CLASSES,
            }
            torch.save(payload, path)
            result.checkpoint_path = path
            logger.info("Best checkpoint salvo em %s (F1 macro=%.4f)", path, best_f1)
        return result
