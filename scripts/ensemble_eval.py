"""Ensemble de seeds — media de logits sobre N checkpoints.

Carrega N checkpoints (mesmo arquitetura/config, diferentes seeds),
roda inferencia sobre o mesmo test set, soma os logits (em fp32) e
emite a predicao via argmax. Reporta metricas do ensemble vs cada
seed individual + razao de disparidade interseccional.

Uso:
  python scripts/ensemble_eval.py \\
    --checkpoints "outputs/.../convnext_s01/.../best.pt" \\
                  "outputs/.../convnext_s02/.../best.pt" \\
                  "outputs/.../convnext_s42/.../best.pt" \\
    --config configs/anchor_hassanpour/exp_anc_hass_convnext_s42.yaml \\
    --output-dir outputs/ensemble \\
    --run-label convnext_hassanpour_ensemble3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_bias.config.loader import load_config  # noqa: E402
from face_bias.data.dataset import setup_dataset  # noqa: E402
from face_bias.evaluation.metrics import disparity_ratio  # noqa: E402
from face_bias.models.resnet import LResNet50E_IR  # noqa: E402


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Reconstrói o modelo conforme config e carrega pesos do best.pt."""
    model = LResNet50E_IR(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.2),
        head=cfg["model"]["head"],
        pretrained=False,
        backbone_arch=cfg["model"].get("backbone_arch", "resnet50"),
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_logits(model: torch.nn.Module, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Inferência: retorna (logits, labels) em ordem do dataloader."""
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images, labels=labels.to(device)) if hasattr(model, "head_type") else model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """F1 score per class (sem average)."""
    return f1_score(y_true, y_pred, labels=range(num_classes), average=None, zero_division=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Lista de caminhos para best.pt (1 por seed)")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--output-dir", default="outputs/ensemble")
    parser.add_argument("--run-label", default="ensemble")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    print("Reconstruindo dataloaders (split_protocol oficial)")
    dataloaders, label_encoder, num_classes = setup_dataset(cfg)
    class_names = list(label_encoder.classes_)
    test_loader = dataloaders["test"]
    print(f"Test set: {len(test_loader.dataset)} imagens; {num_classes} classes")

    all_logits = []
    individual_results = []
    y_true_ref = None

    for ckpt_path in args.checkpoints:
        print(f"\nCarregando {ckpt_path}")
        model = load_model(ckpt_path, cfg, device)
        logits, y_true = predict_logits(model, test_loader, device)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true_ref, y_true), "y_true difere entre checkpoints — dataloader inconsistente"

        y_pred = logits.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_class = per_class_f1(y_true, y_pred, num_classes)
        ir = disparity_ratio(per_class)
        print(f"  Individual: acc={acc:.4f}, f1_macro={f1:.4f}, IR={ir:.3f}")
        individual_results.append({
            "checkpoint": ckpt_path,
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "disparity_ratio": float(ir),
        })
        all_logits.append(logits)

        # libera memória do modelo
        del model
        torch.cuda.empty_cache()

    # Ensemble: média dos logits (equivalente a média de probabilidades para softmax)
    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    y_pred_ens = ensemble_logits.argmax(axis=1)
    acc_ens = accuracy_score(y_true_ref, y_pred_ens)
    f1_ens = f1_score(y_true_ref, y_pred_ens, average="macro", zero_division=0)
    per_class_ens = per_class_f1(y_true_ref, y_pred_ens, num_classes)
    ir_ens = disparity_ratio(per_class_ens)

    print(f"\n=== ENSEMBLE (media de logits de {len(args.checkpoints)} seeds) ===")
    print(f"  acc       = {acc_ens:.4f}")
    print(f"  f1_macro  = {f1_ens:.4f}")
    print(f"  IR (max/min F1 per class) = {ir_ens:.3f}")
    print(f"  per-class F1: {dict(zip(class_names, [round(float(x), 4) for x in per_class_ens]))}")

    # Comparação com média individual
    accs_ind = [r["accuracy"] for r in individual_results]
    f1s_ind = [r["f1_macro"] for r in individual_results]
    irs_ind = [r["disparity_ratio"] for r in individual_results]
    print(f"\n=== Vs media dos individuais (n={len(individual_results)}) ===")
    print(f"  acc       individual mean = {np.mean(accs_ind):.4f}, ensemble = {acc_ens:.4f}, delta = {acc_ens - np.mean(accs_ind):+.4f}")
    print(f"  f1_macro  individual mean = {np.mean(f1s_ind):.4f}, ensemble = {f1_ens:.4f}, delta = {f1_ens - np.mean(f1s_ind):+.4f}")
    print(f"  IR        individual mean = {np.mean(irs_ind):.3f},  ensemble = {ir_ens:.3f},  delta = {ir_ens - np.mean(irs_ind):+.3f}")

    # Save
    summary = {
        "run_label": args.run_label,
        "n_checkpoints": len(args.checkpoints),
        "checkpoints": args.checkpoints,
        "individual": individual_results,
        "ensemble": {
            "accuracy": float(acc_ens),
            "f1_macro": float(f1_ens),
            "disparity_ratio": float(ir_ens),
            "per_class_f1": {c: float(x) for c, x in zip(class_names, per_class_ens)},
        },
        "delta_vs_individual_mean": {
            "accuracy": float(acc_ens - np.mean(accs_ind)),
            "f1_macro": float(f1_ens - np.mean(f1s_ind)),
            "disparity_ratio": float(ir_ens - np.mean(irs_ind)),
        },
    }
    summary_path = out_dir / f"ensemble_{args.run_label}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
