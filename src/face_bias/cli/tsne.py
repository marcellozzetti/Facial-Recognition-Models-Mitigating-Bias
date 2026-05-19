"""CLI: project feature embeddings into 2-D via t-SNE."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.interpretability.tsne import (
    compute_embeddings,
    plot_tsne,
    run_tsne,
    save_embeddings,
)
from face_bias.models.resnet import LResNet50E_IR
from face_bias.utils.logging import setup_logging
from face_bias.utils.reproducibility import seed_from_config


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _build_model(config: dict, num_classes: int) -> torch.nn.Module:
    model_cfg = config["model"]
    return LResNet50E_IR(
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        head=model_cfg["head"],
        pretrained=model_cfg.get("pretrained", True),
        arcface_s=model_cfg["arcface_s"],
        arcface_m=model_cfg["arcface_m"],
        arcface_easy_margin=model_cfg["arcface_easy_margin"],
        mlp_hidden_dims=model_cfg.get("mlp_hidden_dims", [512]),
        mlp_activation=model_cfg.get("mlp_activation", "relu"),
        mlp_dropout=model_cfg.get("mlp_dropout", 0.3),
        mlp_norm=model_cfg.get("mlp_norm", "none"),
        adaface_m=model_cfg.get("adaface_m", 0.4),
        adaface_h=model_cfg.get("adaface_h", 0.333),
        magface_l_a=model_cfg.get("magface_l_a", 10.0),
        magface_u_a=model_cfg.get("magface_u_a", 110.0),
        magface_l_m=model_cfg.get("magface_l_m", 0.45),
        magface_u_m=model_cfg.get("magface_u_m", 0.8),
        magface_lambda_g=model_cfg.get("magface_lambda_g", 0.0),
        contrastive_proj_dim=(
            (config["training"].get("contrastive") or {}).get("proj_dim", 128)
            if (config["training"].get("contrastive") or {}).get("enabled")
            else None
        ),
        contrastive_proj_hidden=(
            (config["training"].get("contrastive") or {}).get("proj_hidden", 512)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="t-SNE projection of model embeddings.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt)")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which dataloader to embed",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Cap the number of points (t-SNE blows up above ~5k)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (auto-clamped for small samples)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tsne",
        help="Directory where the figure and arrays go",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_id = setup_logging(config, "log_evaluation_file")
    seed_from_config(config)
    device = _resolve_device(args.device)
    logging.info(
        f"t-SNE run_id={run_id} device={device} checkpoint={args.checkpoint} "
        f"split={args.split} max_samples={args.max_samples}"
    )

    dataloaders, label_encoder, num_classes = setup_dataset(config)
    class_names = list(label_encoder.classes_)

    model = _build_model(config, num_classes=num_classes)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)

    features, labels = compute_embeddings(
        model,
        dataloaders[args.split],
        device,
        max_samples=args.max_samples,
    )
    logging.info(f"Computed {features.shape[0]} embeddings (dim={features.shape[1]})")

    coords = run_tsne(
        features,
        perplexity=args.perplexity,
        random_state=config["training"]["random_state"],
    )

    output_dir = Path(args.output_dir) / run_id
    figure_path = output_dir / "tsne.png"
    plot_tsne(
        coords,
        labels,
        class_names,
        figure_path,
        title=f"t-SNE — {args.split} split, {features.shape[0]} samples",
    )
    save_embeddings(features, labels, coords, class_names, output_dir)
    logging.info(f"t-SNE figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
