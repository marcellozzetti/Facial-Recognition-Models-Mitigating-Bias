"""Pipeline stage — Etapa 3: treinamento das 3 configurações A/B/C.

Cap. 4 §4.7. Todas as configurações compartilham backbone (ConvNeXt-T),
dataset (FairFace) e protocolo de treinamento — o que varia é apenas
o sinal de condicionamento.

Config A: ConvNeXt-T puro (baseline)
Config B: ConvNeXt-T + FiLM com sinal MST direto (10-dim)
Config C: ConvNeXt-T + FiLM com sinal via CLIP-text (512-dim)

Uso:
    python pipelines/05_train_ablation.py --config-id A --seed 42
    python pipelines/05_train_ablation.py --config-id B --seed 42 \\
        --mst-predictions outputs/etapa1/fairface_val_mst.parquet
    python pipelines/05_train_ablation.py --config-id C --seed 42 \\
        --mst-predictions outputs/etapa1/fairface_val_mst.parquet \\
        --clip-cache outputs/etapa3/clip_bank.npy

Rigor experimental: rodar 3 sementes (42, 1, 2) por configuração.
Total: 3 configs × 3 sementes = 9 treinos.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.conditioning import (  # noqa: E402
    CLIPPromptEnsembler,
    wrap_convnext_with_film,
)

logger = logging.getLogger("pipelines.05_train_ablation")

FAIRFACE_CLASSES = 7
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Dataset com sinal MST opcional                                              #
# --------------------------------------------------------------------------- #
class FairFaceMSTDataset(Dataset):
    """FairFace com vetor MST 10-dim opcional (para configs B e C)."""

    def __init__(
        self,
        images_root: Path,
        labels_df: pd.DataFrame,
        mst_df: Optional[pd.DataFrame],
        transform: transforms.Compose,
    ):
        self.images_root = Path(images_root)
        self.files = labels_df["file"].tolist()
        # FairFace: 7 raças; converter para índice 0..6
        class_names = sorted(labels_df["race"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.labels = [self.class_to_idx[r] for r in labels_df["race"]]
        self.transform = transform
        self.mst_lookup: dict[str, np.ndarray] = {}
        if mst_df is not None:
            for _, row in mst_df.iterrows():
                key = Path(row["path"]).name
                probs = np.array(
                    [row[f"p_{i}"] for i in range(1, 11)], dtype=np.float32
                )
                self.mst_lookup[key] = probs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        path = self.images_root / rel
        from PIL import Image
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        x = self.transform(img)
        y = int(self.labels[idx])
        if self.mst_lookup:
            z = self.mst_lookup.get(Path(rel).name)
            if z is None:
                # fallback: uniforme
                z = np.ones(10, dtype=np.float32) / 10.0
            return x, y, torch.from_numpy(z)
        return x, y


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Model builder por configuração                                              #
# --------------------------------------------------------------------------- #
def build_model_for_config(config_id: str, num_classes: int = FAIRFACE_CLASSES) -> nn.Module:
    """A: ConvNeXt-T puro. B: + FiLM 10-dim. C: + FiLM 512-dim (CLIP)."""
    bb = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    if config_id == "A":
        bb.classifier[-1] = nn.Linear(bb.classifier[-1].in_features, num_classes)
        return bb
    if config_id == "B":
        return wrap_convnext_with_film(bb, cond_dim=10, num_classes=num_classes)
    if config_id == "C":
        return wrap_convnext_with_film(bb, cond_dim=512, num_classes=num_classes)
    raise ValueError(f"config_id inválido: {config_id!r}. Use A, B ou C.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Loop de treino                                                              #
# --------------------------------------------------------------------------- #
def train_one_config(
    config_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    seed: int,
    clip_ensembler: Optional[CLIPPromptEnsembler] = None,
) -> dict:
    _set_seed(seed)
    model = build_model_for_config(config_id).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if config_id == "A":
        params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=lr_backbone, weight_decay=weight_decay)
    else:
        film_ids = {id(p) for p in model.film_layers.parameters()}
        head_ids = {id(p) for p in model.head.parameters()}
        film_params = [p for p in model.parameters() if id(p) in film_ids]
        head_params = [p for p in model.parameters() if id(p) in head_ids]
        backbone_params = [
            p for p in model.parameters()
            if id(p) not in film_ids and id(p) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": film_params, "lr": lr_head},
                {"params": head_params, "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )

    def _step(batch, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
        if len(batch) == 3:
            x, y, z = batch
            z = z.to(device, non_blocking=True)
            if config_id == "C" and clip_ensembler is not None:
                z = clip_ensembler.encode(z)
        else:
            x, y = batch
            z = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if config_id == "A":
            logits = model(x)
        else:
            logits = model(x, z=z)
        loss = loss_fn(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        return float(loss.item()), logits.argmax(-1).detach().cpu().numpy(), y.cpu().numpy()

    history: list[dict] = []
    best_acc = -1.0
    best_state: Optional[dict] = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_acc, n = 0.0, 0
        for batch in train_loader:
            loss, _, _ = _step(batch, training=True)
            bs = batch[0].size(0)
            train_loss_acc += loss * bs
            n += bs
        train_loss = train_loss_acc / max(n, 1)

        model.eval()
        preds, tgts, val_loss_acc, n = [], [], 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                loss, p, t = _step(batch, training=False)
                bs = batch[0].size(0)
                val_loss_acc += loss * bs
                n += bs
                preds.append(p)
                tgts.append(t)
        preds = np.concatenate(preds)
        tgts = np.concatenate(tgts)
        val_loss = val_loss_acc / max(n, 1)
        val_acc = float((preds == tgts).mean())
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        logger.info(
            "cfg=%s seed=%d ep=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            config_id, seed, epoch, train_loss, val_loss, val_acc,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {"history": history, "best_val_acc": best_acc, "best_state": best_state}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 3 — treino ablation A/B/C.")
    p.add_argument("--config-id", choices=["A", "B", "C"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--val-labels", type=Path, required=True)
    p.add_argument("--mst-predictions", type=Path,
                   help="Parquet do pipeline 03 (necessário para B e C).")
    p.add_argument("--clip-cache", type=Path,
                   help="Cache .npy do prompt bank CLIP (Config C).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    mst_df = None
    if args.config_id in {"B", "C"}:
        if args.mst_predictions is None or not args.mst_predictions.exists():
            logger.error("Configs B/C exigem --mst-predictions apontando para parquet válido.")
            return 2
        mst_df = pd.read_parquet(args.mst_predictions)

    train_labels = pd.read_csv(args.train_labels)
    val_labels = pd.read_csv(args.val_labels)
    train_ds = FairFaceMSTDataset(args.dataset_root, train_labels, mst_df, _train_transform())
    val_ds = FairFaceMSTDataset(args.dataset_root, val_labels, mst_df, _eval_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    clip_ensembler = None
    if args.config_id == "C":
        clip_ensembler = CLIPPromptEnsembler(cache_path=args.clip_cache, device=str(device))
        clip_ensembler.build_prompt_bank()

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = train_one_config(
        config_id=args.config_id,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=args.max_epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        seed=args.seed,
        clip_ensembler=clip_ensembler,
    )
    elapsed = time.time() - t0

    ckpt = args.output / f"{args.config_id}_seed{args.seed}.pt"
    torch.save({
        "state_dict": result["best_state"],
        "config_id": args.config_id,
        "seed": args.seed,
        "best_val_acc": result["best_val_acc"],
    }, ckpt)
    (args.output / f"{args.config_id}_seed{args.seed}_history.json").write_text(
        json.dumps(result["history"], indent=2), encoding="utf-8"
    )
    logger.info(
        "config=%s seed=%d best_val_acc=%.4f tempo=%.1fs ckpt=%s",
        args.config_id, args.seed, result["best_val_acc"], elapsed, ckpt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
