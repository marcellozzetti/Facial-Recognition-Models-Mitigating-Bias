"""Test-Time Augmentation (TTA) com ensemble opcional.

Aplica augmentations SEGURAS (sem alteracao de cor/brilho — apenas
geometricas) sobre cada imagem do test set, roda inferencia, e calcula
a media dos logits ao final. Combinacao com ensemble: se multiplos
checkpoints fornecidos, cada um eh rodado em cada augmentation e
todos os logits sao medidados juntos.

Augmentations TTA usadas (todas seguras para preservar features raciais):
  - Original (resize 224x224)
  - HorizontalFlip
  - Resize 256 + CenterCrop 224 (zoom out gentil)
  - Resize 256 + HFlip + CenterCrop 224
  - (opcional) 4-corner crops

Uso:
  python scripts/tta_eval.py \\
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \\
    --config configs/.../exp_*.yaml \\
    --output-dir outputs/tta \\
    --run-label ensemble_tta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_bias.config.loader import load_config  # noqa: E402
from face_bias.data.dataset import setup_dataset  # noqa: E402
from face_bias.evaluation.metrics import disparity_ratio  # noqa: E402
from face_bias.models.resnet import LResNet50E_IR  # noqa: E402


def build_tta_transforms(image_size: tuple[int, int], image_mean, image_std) -> list:
    """Augmentations geometricas seguras para TTA."""
    norm = transforms.Normalize(image_mean, image_std)
    base_size = image_size[0]
    larger = int(base_size * 256 / 224)  # mantem aspect ratio

    return [
        # 1. Original (resize direto)
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            norm,
        ]),
        # 2. HorizontalFlip
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=1.0),  # sempre flipa
            transforms.ToTensor(),
            norm,
        ]),
        # 3. Resize maior + center crop
        transforms.Compose([
            transforms.Resize(larger),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ]),
        # 4. Resize maior + HFlip + center crop
        transforms.Compose([
            transforms.Resize(larger),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ]),
    ]


class TTADataset(Dataset):
    """Dataset que retorna a mesma imagem aplicando um TTA transform."""

    def __init__(self, base_dataset_files: list[str], base_dataset_labels: list[int],
                 img_dir: str, tta_transform):
        self.files = base_dataset_files
        self.labels = base_dataset_labels
        self.img_dir = img_dir
        self.transform = tta_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import os
        img_path = os.path.join(self.img_dir, self.files[idx])
        with Image.open(img_path) as raw:
            img = raw.convert("RGB")
        img = self.transform(img)
        return img, int(self.labels[idx])


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
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
def predict_logits_with_loader(model, loader, device):
    all_logits, all_targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images, labels=labels.to(device)) if hasattr(model, "head_type") else model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/tta")
    parser.add_argument("--run-label", default="tta")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Obter informacoes do test set
    print("Reconstruindo dataloaders para extrair test file list")
    dataloaders, label_encoder, num_classes = setup_dataset(cfg)
    class_names = list(label_encoder.classes_)
    test_dataset = dataloaders["test"].dataset
    test_files = test_dataset.img_paths
    test_labels = test_dataset.encoded_labels
    img_dir = test_dataset.img_dir
    print(f"Test set: {len(test_files)} imagens; {num_classes} classes")
    del dataloaders  # free

    # TTA transforms
    image_size = tuple(cfg["image"]["image_size"])
    tta_transforms = build_tta_transforms(image_size, cfg["image"]["image_mean"], cfg["image"]["image_std"])
    print(f"TTA: {len(tta_transforms)} augmentations seguras (original, HFlip, ResizeCrop, ResizeCrop+HFlip)")

    batch_size = cfg["training"]["batch_size"]
    n_ckpts = len(args.checkpoints)
    n_tta = len(tta_transforms)
    print(f"Total inferencias: {n_ckpts} ckpts x {n_tta} TTA = {n_ckpts * n_tta} passes pelo test set")

    # Acumula logits: media sobre (ckpt, tta) pares
    all_logits = []
    y_true_ref = None
    individual_summary = []

    for ckpt_idx, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{ckpt_idx+1}/{n_ckpts}] Carregando {ckpt_path}")
        model = load_model(ckpt_path, cfg, device)

        ckpt_logits_per_tta = []
        for tta_idx, tta_transform in enumerate(tta_transforms):
            tta_dataset = TTADataset(test_files, test_labels, img_dir, tta_transform)
            tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
            logits, y_true = predict_logits_with_loader(model, tta_loader, device)
            ckpt_logits_per_tta.append(logits)
            if y_true_ref is None:
                y_true_ref = y_true

        ckpt_logits = np.mean(np.stack(ckpt_logits_per_tta, axis=0), axis=0)
        y_pred = ckpt_logits.argmax(axis=1)
        acc = accuracy_score(y_true_ref, y_pred)
        f1 = f1_score(y_true_ref, y_pred, average="macro", zero_division=0)
        from sklearn.metrics import f1_score as f1_per
        per_class = f1_per(y_true_ref, y_pred, labels=range(num_classes), average=None, zero_division=0)
        ir = disparity_ratio(per_class)
        print(f"  Ckpt {Path(ckpt_path).parent.parent.parent.parent.name} + TTA: acc={acc:.4f}, f1={f1:.4f}, IR={ir:.3f}")
        individual_summary.append({
            "checkpoint": ckpt_path,
            "acc_with_tta": float(acc),
            "f1_with_tta": float(f1),
            "ir_with_tta": float(ir),
        })

        # Adiciona TODOS os logits (todas as TTAs deste checkpoint) ao ensemble global
        all_logits.extend(ckpt_logits_per_tta)
        del model
        torch.cuda.empty_cache()

    # Ensemble: media sobre TODAS as combinacoes (ckpt x TTA)
    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    y_pred_ens = ensemble_logits.argmax(axis=1)
    acc_ens = accuracy_score(y_true_ref, y_pred_ens)
    f1_ens = f1_score(y_true_ref, y_pred_ens, average="macro", zero_division=0)
    from sklearn.metrics import f1_score as f1_per
    per_class_ens = f1_per(y_true_ref, y_pred_ens, labels=range(num_classes), average=None, zero_division=0)
    ir_ens = disparity_ratio(per_class_ens)

    print(f"\n=== ENSEMBLE FINAL ({n_ckpts} ckpts x {n_tta} TTA = {n_ckpts*n_tta} logits) ===")
    print(f"  acc       = {acc_ens:.4f}")
    print(f"  f1_macro  = {f1_ens:.4f}")
    print(f"  IR        = {ir_ens:.3f}")
    print(f"  per-class F1: {dict(zip(class_names, [round(float(x), 4) for x in per_class_ens]))}")

    summary = {
        "run_label": args.run_label,
        "n_checkpoints": n_ckpts,
        "n_tta_augmentations": n_tta,
        "tta_augmentations": ["original", "hflip", "resize_centercrop", "resize_hflip_centercrop"],
        "individual_with_tta": individual_summary,
        "ensemble_with_tta": {
            "accuracy": float(acc_ens),
            "f1_macro": float(f1_ens),
            "disparity_ratio": float(ir_ens),
            "per_class_f1": {c: float(x) for c, x in zip(class_names, per_class_ens)},
        },
    }
    summary_path = out_dir / f"tta_{args.run_label}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
