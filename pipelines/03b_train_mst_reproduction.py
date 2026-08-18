"""Plano B da Etapa 1 — reprodução do treinamento SkinToneNet a partir do STW.

Usar SOMENTE se os pesos oficiais (Matias 2026, arXiv 2603.02475) não
forem publicados a tempo do cronograma. Requer o dataset STW disponível
localmente. Ver docs/ativo/etapa1_report.md §1 para status atual e
docs/tese/tex/cronograma.tex §Risco~2 para a mitigação declarada.

Receita seguida (do artigo):
    - Backbone ViT (torchvision vit_b_16 como stand-in do ViT-Small)
    - Head linear 10-classes (escala Monk)
    - Loss: cross-entropy
    - Optimizer: Adam ou SGD
    - Scheduler: ReduceLROnPlateau (validation performance)
    - Splits: 80/20 holdout, 3 sementes (42, 1, 2)

Uso:
    python pipelines/03b_train_mst_reproduction.py \\
        --stw-root data/STW \\
        --output outputs/etapa1_repro/ \\
        --seed 42

Saídas:
    outputs/etapa1_repro/seed{S}/best.pt        (state_dict compatível com wrapper)
    outputs/etapa1_repro/seed{S}/history.csv    (loss + acc por época)
    outputs/etapa1_repro/seed{S}/summary.json   (métricas finais)

Compatibilidade com o wrapper:
    O ``best.pt`` gerado aqui carrega diretamente em
    ``MSTClassifier(weights_path=".../best.pt")`` — o wrapper aceita
    dicts com chave "state_dict" e faz load com strict=False.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.mst.skintonenet import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    MST_N_CLASSES,
    build_mst_backbone,
)

logger = logging.getLogger("pipelines.03b_train_mst_reproduction")

SEEDS = (42, 1, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class STWDataset(Dataset):
    """Dataset STW. Espera um CSV ``labels.csv`` na raiz com colunas
    ``file`` (path relativo) e ``mst`` (int 1..10).
    """

    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = Path(root)
        csv = self.root / "labels.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"labels.csv não encontrado em {self.root}. "
                "Formato esperado: colunas 'file' (relativo à raiz) e 'mst' (1..10)."
            )
        df = pd.read_csv(csv)
        if not {"file", "mst"}.issubset(df.columns):
            raise ValueError(f"labels.csv deve ter colunas 'file' e 'mst'; got {df.columns.tolist()}")
        self.samples = list(zip(df["file"].tolist(), df["mst"].astype(int).tolist()))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel, label = self.samples[idx]
        img = Image.open(self.root / rel).convert("RGB")
        return self.transform(img), int(label) - 1  # 1..10 -> 0..9


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if not augment:
        return val_tf, val_tf
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, val_tf


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"optimizer {name!r} não suportado (use adam ou sgd).")


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += labels.size(0)
    return {"val_loss": loss_sum / max(total, 1), "val_acc": correct / max(total, 1)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, loss_sum = 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
    return loss_sum / max(total, 1)


def save_best(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reprodução do SkinToneNet (plano B da Etapa 1).")
    p.add_argument("--stw-root", type=Path, required=True, help="Raiz do dataset STW (com labels.csv).")
    p.add_argument("--output", type=Path, required=True, help="Diretório raiz de saída.")
    p.add_argument("--seed", type=int, default=42, choices=SEEDS, help="Semente (42/1/2).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fração de validação (default 0.2).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=("adam", "sgd"))
    p.add_argument("--patience", type=int, default=5, help="Paciência do ReduceLROnPlateau.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_tf, val_tf = build_transforms(augment=not args.no_augment)

    full_dataset_val = STWDataset(args.stw_root, transform=val_tf)
    n = len(full_dataset_val)
    n_val = max(1, int(args.val_frac * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        STWDataset(args.stw_root, transform=train_tf),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # random_split preserva os transforms do dataset base; o val_ds fica com
    # augmentation. Refaço o val com transforms de eval mantendo os índices.
    val_ds.dataset = full_dataset_val

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, _ = build_mst_backbone(pretrained_imagenet=True)
    model = model.to(device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.patience)

    out_dir = args.output / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: list[dict] = []
    best_acc, best_epoch = -1.0, 0
    logger.info(
        "STW n=%d (train=%d val=%d) | classes=%d | device=%s | seed=%d",
        n,
        n_train,
        n_val,
        MST_N_CLASSES,
        device.type,
        args.seed,
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["val_acc"])
        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "lr": optimizer.param_groups[0]["lr"]}
        hist.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f lr=%.2e t=%.1fs",
            epoch, train_loss, metrics["val_loss"], metrics["val_acc"], row["lr"], time.time() - t_ep,
        )
        if metrics["val_acc"] > best_acc:
            best_acc, best_epoch = metrics["val_acc"], epoch
            save_best(
                out_dir / "best.pt",
                model,
                meta={"epoch": epoch, "val_acc": best_acc, "seed": args.seed, "recipe": "vit_b_16+ce+adam"},
            )

    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    summary = {
        "seed": args.seed,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "total_time_s": time.time() - t0,
        "recipe": {
            "backbone": "vit_b_16",
            "loss": "cross_entropy",
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "ReduceLROnPlateau",
            "batch_size": args.batch_size,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Melhor val_acc=%.4f na época %d. Checkpoint em %s.", best_acc, best_epoch, out_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
