"""Generate the Factor-5 (backbone) configs — modern alternatives to ResNet-50.

Factor 5 isolates the contribution of the *backbone architecture* to
demographic disparity. Matched protocol with every other factor: clean
dataset, linear head, CE loss, 3 seeds (42,1,2), criterion val_f1_macro,
fp32, num_workers=0 (Windows), 25 epochs, early stopping. Only the
backbone changes.

Arms (this script — 6 runs):
  - ViT-B/16 (torchvision, ImageNet pretrained, 768-d)
  - ConvNeXt-T (torchvision, ImageNet pretrained, 768-d)

Control is NOT re-run — ResNet-50 ImageNet + linear head + CE under the
same protocol/criterion already exists as the definitive baseline:
  outputs/definitive/baseline/exp_r2base_exp05_ce_s{42,01,02}

Insightface IR-real (LResNet50E-IR; 112px, 512-d, scratch-trained) is
deferred to the defense program (PLANO §5) — out of scope for the
qualification factor isolation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/experiments_factor5")
SEEDS = [42, 1, 2]
CLEAN_CSV = "data/raw/fairface/fairface_labels_clean.csv"

# Per-backbone fine-tuning recipe (declared explicitly — see comment in
# Factor-5 results doc). The lr=1e-3 of ResNet/ConvNeXt does not work
# for ViT-B/16 (sanity confirmed: val_f1 ~ chance at epoch 1, validation
# loss near ln(7)); the canonical fine-tuning rate for ViTs is 1e-4 to
# 5e-5. Batch reduced for ViT because the 4070 12GB thrashes at b=128
# fp32 (49 min/epoch). Each backbone gets its canonical recipe — same
# scientific practice every backbone comparison in the literature uses.
ARMS = {
    # ViT lr=1e-4 (canonical fine-tuning rate for ViTs; lr=1e-3
    # destroys pretrained attention — sanity confirmed).
    # ConvNeXt-T thrashes at bs=128 fp32 on a 4070 12GB (sanity caught
    # it the same way as ViT-v1); reduce to bs=64. LR stays 1e-3 (CNN).
    "vit":      {"arch": "vit_b_16",       "lr": 1e-4, "batch_size": 64},
    "convnext": {"arch": "convnext_tiny",  "lr": 1e-3, "batch_size": 64},
}


def _cfg(arm: dict, seed: int) -> dict:
    return {
        "model": {
            "name": "LResNet50E_IR",  # legacy mlflow tag (alias); arch below
            "backbone_arch": arm["arch"],  # the factor under study
            "pretrained": True,
            "num_classes": 7,
            "dropout": 0.2,
            "head": "linear",
            "arcface_s": 30.0,
            "arcface_m": 0.5,
            "arcface_easy_margin": False,
        },
        "training": {
            "batch_size": arm["batch_size"],
            "learning_rate": arm["lr"],
            "num_epochs": 25,
            "optimizer": "adamw",
            "scheduler": "cosineannealingwarmrestarts",
            "loss_function": "cross_entropy",
            "test_size": 0.1,
            "val_size": 0.1,
            "num_workers": 0,
            "random_state": seed,
            "grad_clip_norm": 5.0,
            "early_stopping_patience": 5,
            "use_amp": False,
            # checkpoint_metric omitted -> inherits val_f1_macro (correct
            # criterion; see docs/checkpoint_criterion_audit.md)
        },
        "image": {
            "image_size": [224, 224],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "preprocessing": {
            "processing_type": "detect_adjust_image",
            "num_workers": 4,
            "rotate_angle": 45,
            "num_rotations": 8,
            "max_faces_per_image": 1,
        },
        "data": {
            "dataset_file": CLEAN_CSV,
            "dataset_image_input_path": "data/processed/fairface_aligned",
            "dataset_image_output_path": "data/processed/fairface_aligned",
            "balance": "undersample",
        },
        "bucket": {
            "function": "download",
            "bucket_client": "s3",
            "bucket_path_file": "data/raw/bucket/",
            "object_file_name": "fairface_original_images.zip",
            "bucket_name": "dataset-fairface",
        },
        "logging": {
            "log_dir": "outputs/logs",
            "log_level": "INFO",
            "format": "%(asctime)s [%(run_id)s] %(name)s %(levelname)s: %(message)s",
            "log_bucket_file": "bucket.log",
            "log_preprocessing_file": "preprocessing.log",
            "log_face_data_file": "face_data.log",
            "log_version_file": "version.log",
            "log_training_file": "training.log",
            "log_evaluation_file": "evaluation.log",
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_key, arm in ARMS.items():
        for seed in SEEDS:
            cfg = _cfg(arm, seed)
            fname = f"exp_f5_{arm_key}_s{seed:02d}.yaml"
            header = (
                f"# Factor 5 (backbone) — {arm['arch']} seed={seed}\n"
                f"# clean dataset, linear head, CE, 25ep, fp32, num_workers=0,\n"
                f"# criterion=val_f1_macro, 3-seed matched protocol.\n"
                f"# Per-backbone recipe: lr={arm['lr']}, batch_size={arm['batch_size']}\n"
                f"# (canonical fine-tuning rate per backbone — declared explicitly).\n"
                f"# Control (NOT re-run): ResNet-50 ImageNet + linear + CE from\n"
                f"# the definitive baseline (outputs/definitive/baseline/exp_r2base_exp05_ce_*).\n"
                f"# Generated by scripts/generate_factor5_configs.py.\n\n"
            )
            with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
                f.write(header)
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written.append(fname)

    print(f"Wrote {len(written)} configs to {OUT_DIR}:")
    for w in sorted(written):
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
