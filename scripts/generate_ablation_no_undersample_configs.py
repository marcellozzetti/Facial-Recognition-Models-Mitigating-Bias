"""Generate the No-Undersample ABLATION configs.

Ablation 🅑 — robustness of the central finding (ConvNeXt-T as the
single lever moving both F1 and IR significantly) to the class-balance
decision. All Fatores 1-5 and 3 anchors use `balance: undersample`. This
ablation removes that, keeping natural FairFace class imbalance, on the
two arms that matter for the central claim:

  1. Controle CE+linear ResNet-50 (matched recipe with definitive
     baseline, only `balance: none` differs)
  2. ConvNeXt-T + linear (matched recipe with Factor-5 ConvNeXt-T,
     only `balance: none` differs)

Why only these two arms? Because the central thesis claim is
"ConvNeXt-T vs controle moves F1 and IR significantly". The ablation
tests whether that Δ survives without undersample. The remaining
fatores (3, 4) and anchors are NOT re-run — they were nulls under the
matched protocol, and re-running them under a different protocol would
open a new matrix instead of closing a vulnerability.

Diagnostic outcomes (cf. docs/sota_7class_race_audit.md §6.3):
  - A (theory): acc ↑ 2-3pp, F1 ~0, IR ↑ → undersample trades acc for fairness
  - B: acc ↑ 2-3pp, F1 ↑, IR ~0 → undersample was cost without benefit
  - C: acc ~0, F1 ↓, IR ↑ → undersample was net helpful
  - D: anomalous (would warrant investigation)

The interesting metric is IR, not acc.

Matched-comparison nota: the split (80/10/10 estratificado, seed-fixo)
remains identical to F1-F5 — even without undersample, the train/val/test
images are the same set per seed (just with natural imbalance present in
training instead of artificially balanced).
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/ablation_no_undersample")
SEEDS = [42, 1, 2]
CLEAN_CSV = "data/raw/fairface/fairface_labels_clean.csv"

# Two arms only — control RN-50 + ConvNeXt-T, both LayerNorm vs BN
# distinction tested under no-undersample. Recipe per backbone identical
# to the matched-protocol experiments — only the `data.balance` flips.
ARMS = {
    "control": {
        "arch": "resnet50",
        "lr": 1e-3,
        "batch_size": 128,
        "label": "Controle CE+linear ResNet-50 (no undersample)",
    },
    "convnext": {
        "arch": "convnext_tiny",
        "lr": 1e-4,
        "batch_size": 64,
        "label": "ConvNeXt-T + linear (no undersample)",
    },
}


def _cfg(arm: dict, seed: int) -> dict:
    return {
        "model": {
            "name": "LResNet50E_IR",
            "backbone_arch": arm["arch"],
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
            "balance": "none",  # <<< the single Δ for this ablation
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
            fname = f"exp_abl_nous_{arm_key}_s{seed:02d}.yaml"
            header = (
                f"# Ablation 🅑 — {arm['label']} seed={seed}\n"
                f"# Same pipeline as F1-F5 EXCEPT data.balance: none (was undersample).\n"
                f"# Tests robustness of central claim (ConvNeXt-T vs controle) to\n"
                f"# the class-balance decision. Recipe per backbone identical to the\n"
                f"# matched-protocol equivalents (control=baseline; convnext=Factor 5).\n"
                f"# Ver docs/sota_7class_race_audit.md.\n"
                f"# Generated by scripts/generate_ablation_no_undersample_configs.py.\n\n"
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
