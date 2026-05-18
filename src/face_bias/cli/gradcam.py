"""CLI: render Grad-CAM heatmaps over a sample of test images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.interpretability.gradcam import (
    GradCAM,
    denormalise_image,
    find_target_layer,
    overlay_heatmap,
    plot_grid,
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
    )


def _select_examples(dataloader, max_samples: int) -> list[tuple[torch.Tensor, int]]:
    """Iterate the loader and collect up to ``max_samples`` (image, label) pairs."""
    out: list[tuple[torch.Tensor, int]] = []
    for images, labels in dataloader:
        for i in range(images.size(0)):
            out.append((images[i], int(labels[i].item())))
            if len(out) >= max_samples:
                return out
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualisation over a sample of evaluation images."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt)")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which dataloader to sample from",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of (image, overlay) pairs to render",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gradcam",
        help="Directory where the figure goes",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_id = setup_logging(config, "log_evaluation_file")
    seed_from_config(config)
    device = _resolve_device(args.device)
    logging.info(
        f"Grad-CAM run_id={run_id} device={device} checkpoint={args.checkpoint} "
        f"split={args.split} num_samples={args.num_samples}"
    )

    dataloaders, label_encoder, num_classes = setup_dataset(config)
    class_names = list(label_encoder.classes_)

    model = _build_model(config, num_classes=num_classes)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)

    samples = _select_examples(dataloaders[args.split], max_samples=args.num_samples)
    image_mean = config["image"]["image_mean"]
    image_std = config["image"]["image_std"]

    target_layer = find_target_layer(model)
    examples = []
    with GradCAM(model, target_layer) as cam_op:
        for tensor, true_label in samples:
            tensor_dev = tensor.to(device)
            cam, predicted = cam_op(tensor_dev)
            rgb = denormalise_image(tensor, image_mean, image_std)
            overlay = overlay_heatmap(rgb, cam)
            examples.append(
                {
                    "image": rgb,
                    "overlay": overlay,
                    "true_class": true_label,
                    "predicted_class": predicted,
                }
            )

    output_dir = Path(args.output_dir) / run_id
    plot_grid(
        examples,
        class_names,
        output_dir / "gradcam.png",
        title=f"Grad-CAM — {args.split} split, {len(examples)} samples",
    )
    logging.info(f"Grad-CAM artefacts under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
