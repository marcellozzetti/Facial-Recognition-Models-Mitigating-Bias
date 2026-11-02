"""Pipeline Etapa 1b: inferência MST sobre o FairFace para gerar z_mst.

Cap. 4 §4.4. Carrega o checkpoint ViT-B/16 treinado no STW e roda inferência
sobre todas as imagens do FairFace (train + val), salvando as probabilidades
softmax (p_1..p_10) em um parquet. Esse arquivo é o pré-requisito para
Configs B e C do pipeline 05_train_ablation.py.

Uso:
    python -I pipelines/04b_infer_mst_fairface.py \\
        --checkpoint   outputs/etapa1/mst_extractor_v2/mst_vit_b16_seed42.pt \\
        --dataset-root data/raw/fairface \\
        --train-labels data/raw/fairface/fairface_label_train.csv \\
        --val-labels   data/raw/fairface/fairface_label_val.csv \\
        --output       outputs/etapa1/fairface_train_mst.parquet

Saída:
    Parquet com colunas: path, p_1, p_2, ..., p_10
    path = caminho relativo tal como aparece nas labels do FairFace (coluna "file")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("pipelines.04b_infer_mst_fairface")

MST_CLASSES = 10
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class FairFaceInferDataset(Dataset):
    """FairFace dataset em modo inferência — devolve (image_tensor, file_path)."""

    def __init__(self, images_root: Path, files: list[str], transform: transforms.Compose):
        self.images_root = Path(images_root)
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        from PIL import Image
        with Image.open(self.images_root / rel) as raw:
            img = raw.convert("RGB")
        return self.transform(img), rel


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Carrega o modelo MST                                                         #
# --------------------------------------------------------------------------- #
def load_mst_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Carrega ViT-B/16 com head FC(768→10) a partir do checkpoint salvo em 04_train_mst."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", MST_CLASSES)

    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info(
        "Checkpoint carregado: val_acc=%.4f val_mae=%.4f seed=%s",
        ckpt.get("best_val_acc", float("nan")),
        ckpt.get("best_val_mae", float("nan")),
        ckpt.get("seed", "?"),
    )
    return model


# --------------------------------------------------------------------------- #
# Inferência                                                                   #
# --------------------------------------------------------------------------- #
def run_inference(
    model: nn.Module,
    files: list[str],
    images_root: Path,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> pd.DataFrame:
    ds = FairFaceInferDataset(images_root, files, _eval_transform())
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    all_paths: list[str] = []
    all_probs: list[np.ndarray] = []
    total = len(files)
    processed = 0

    with torch.inference_mode():
        for imgs, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_paths.extend(paths)
            processed += len(paths)
            if processed % 10000 == 0 or processed == total:
                logger.info("  %d / %d imagens processadas", processed, total)

    probs_arr = np.concatenate(all_probs, axis=0)
    df = pd.DataFrame({"path": all_paths})
    for i in range(1, MST_CLASSES + 1):
        df[f"p_{i}"] = probs_arr[:, i - 1]
    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferência MST sobre FairFace — gera parquet com z_mst."
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Checkpoint .pt do extrator MST (pipeline 04).")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Raiz do FairFace (contém train/ e val/).")
    p.add_argument("--train-labels", type=Path, required=True,
                   help="CSV de labels de treino do FairFace (coluna 'file').")
    p.add_argument("--val-labels", type=Path, required=True,
                   help="CSV de labels de validação do FairFace (coluna 'file').")
    p.add_argument("--output", type=Path, required=True,
                   help="Caminho do parquet de saída.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    if not args.checkpoint.exists():
        logger.error("Checkpoint não encontrado: %s", args.checkpoint)
        return 1

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    logger.info("Device: %s", device)

    model = load_mst_model(args.checkpoint, device)

    train_df = pd.read_csv(args.train_labels)
    val_df   = pd.read_csv(args.val_labels)
    all_files = train_df["file"].tolist() + val_df["file"].tolist()
    logger.info("Total de imagens FairFace: %d (train=%d val=%d)",
                len(all_files), len(train_df), len(val_df))

    logger.info("Iniciando inferência…")
    result_df = run_inference(
        model, all_files, args.dataset_root, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(args.output, index=False)
    logger.info("Parquet salvo: %s  (%d linhas)", args.output, len(result_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
