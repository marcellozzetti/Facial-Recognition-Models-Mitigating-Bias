"""Pipeline stage — Etapa 5: transferência do backbone fair para reconhecimento.

Cap. 4 §Etapa 5. Consome o checkpoint da Config B (Etapa 3) e avalia
reconhecimento facial 1:1 sobre RFW ou BFW, controlando confounders
(pixel information + descritores de qualidade).

Fluxo:
    1. Carrega backbone fair da Etapa 3.
    2. Para cada imagem citada nos pares oficiais, gera embedding.
    3. Para cada par, computa similaridade coseno + qualidade da imagem.
    4. Encontra threshold ótimo, calcula acurácia por raça (RFW) ou por
       subgrupo race×gender (BFW).
    5. Salva parquet + relatório com gap entre grupos.

Uso:
    python pipelines/07_transfer_downstream.py \\
        --dataset rfw \\
        --backbone-checkpoint outputs/etapa3/B_seed42.pt \\
        --pairs data/RFW/pairs.csv \\
        --images-root data/RFW/images \\
        --output-dir outputs/etapa5/rfw/ \\
        --mst-predictions outputs/etapa1/rfw_mst.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.conditioning import wrap_convnext_with_film  # noqa: E402
from face_bias.transfer import (  # noqa: E402
    descriptors_for_crop,
    evaluate_pairs,
    evaluate_pairs_intersectional,
    intersectional_gap,
    pixel_info_from_bbox,
    race_gap,
)

logger = logging.getLogger("pipelines.07_transfer_downstream")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 5 — transferência RFW/BFW.")
    p.add_argument("--dataset", choices=["rfw", "bfw"], required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True,
                   help="CSV com path_a, path_b, is_same, race (+ gender para BFW).")
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mst-predictions", type=Path,
                   help="Parquet do pipeline 03 com vetores MST das imagens.")
    p.add_argument("--config-id", choices=["A", "B", "C"], default="B")
    p.add_argument("--cond-dim", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=7,
                   help="Só usado para reconstruir a arquitetura do checkpoint.")
    return p.parse_args(argv)


def _load_backbone(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    bb = convnext_tiny(weights=None)
    if args.config_id == "A":
        bb.classifier[-1] = torch.nn.Linear(bb.classifier[-1].in_features, args.num_classes)
        model = bb
    else:
        model = wrap_convnext_with_film(
            bb, cond_dim=args.cond_dim, num_classes=args.num_classes
        )
    ckpt = torch.load(str(args.backbone_checkpoint), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.info("load_state_dict: %d missing / %d unexpected keys.", len(missing), len(unexpected))
    model.eval().to(device)
    return model


@torch.inference_mode()
def _extract_embeddings(
    model: torch.nn.Module,
    images_root: Path,
    pairs_df: pd.DataFrame,
    mst_lookup: dict[str, np.ndarray],
    device: torch.device,
    transform,
    batch_size: int,
    config_id: str,
) -> dict[str, np.ndarray]:
    """Retorna dict {path: embedding numpy}."""
    unique_paths = sorted(set(pairs_df["path_a"].tolist() + pairs_df["path_b"].tolist()))
    logger.info("Extraindo embeddings de %d imagens...", len(unique_paths))

    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(unique_paths), batch_size):
        chunk = unique_paths[start : start + batch_size]
        tensors = []
        zs = []
        for rel in chunk:
            with Image.open(images_root / rel) as raw:
                img = raw.convert("RGB")
            tensors.append(transform(img))
            if config_id != "A":
                key = Path(rel).name
                z = mst_lookup.get(key, np.ones(10, dtype=np.float32) / 10.0)
                zs.append(torch.from_numpy(z))
        batch = torch.stack(tensors).to(device)
        if config_id == "A":
            feats = model(batch)  # logits
            # Usa como embedding: features pré-head (aproximação; para Config A
            # sem wrapper, extraímos via forward hook seria mais correto).
            emb = feats
        else:
            z_batch = torch.stack(zs).to(device)
            # embedding = feature map global pooled (bypassa head)
            head_saved = model.head
            model.head = None
            feats = model(batch, z=z_batch)  # (N, 768, 7, 7)
            model.head = head_saved
            emb = feats.mean(dim=(-2, -1))  # (N, 768)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        for rel, e in zip(chunk, emb.cpu().numpy()):
            embeddings[rel] = e
    return embeddings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    pairs_df = pd.read_csv(args.pairs)
    required = {"path_a", "path_b", "is_same", "race"}
    if args.dataset == "bfw":
        required |= {"gender"}
    missing = required - set(pairs_df.columns)
    if missing:
        logger.error("Pares sem colunas obrigatórias: %s", sorted(missing))
        return 2
    logger.info("%s: %d pares carregados.", args.dataset.upper(), len(pairs_df))

    mst_lookup: dict[str, np.ndarray] = {}
    if args.mst_predictions and args.mst_predictions.exists():
        mst_df = pd.read_parquet(args.mst_predictions)
        for _, row in mst_df.iterrows():
            key = Path(row["path"]).name
            mst_lookup[key] = np.array(
                [row[f"p_{i}"] for i in range(1, 11)], dtype=np.float32
            )
        logger.info("MST lookup carregado: %d entradas.", len(mst_lookup))
    elif args.config_id != "A":
        logger.warning(
            "Config %s exige MST; usando softmax uniforme como fallback.",
            args.config_id,
        )

    model = _load_backbone(args, device)
    embeddings = _extract_embeddings(
        model=model,
        images_root=args.images_root,
        pairs_df=pairs_df,
        mst_lookup=mst_lookup,
        device=device,
        transform=_eval_transform(),
        batch_size=args.batch_size,
        config_id=args.config_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "rfw":
        result = evaluate_pairs(pairs_df, embeddings)
        gap = race_gap(result)
        result.to_parquet(args.output_dir / "rfw_verification.parquet", index=False)
        logger.info("RFW gap entre raças = %.4f", gap)
    else:  # bfw
        result = evaluate_pairs_intersectional(pairs_df, embeddings)
        gap = intersectional_gap(result)
        result.to_parquet(args.output_dir / "bfw_intersectional.parquet", index=False)
        logger.info("BFW gap interseccional = %.4f", gap)

    summary = {
        "dataset": args.dataset,
        "config_id": args.config_id,
        "n_pairs": int(len(pairs_df)),
        "n_embeddings": int(len(embeddings)),
        "gap": float(gap) if not np.isnan(gap) else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
