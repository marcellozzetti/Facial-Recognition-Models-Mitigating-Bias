"""Generate the Phase-4 refit configs for the R2 Pareto winners.

Phase 4 confirms that the HPO Round 2 winners survive the full 25-epoch
budget (the HPO itself only ran 8 epochs per trial). Two topologies ×
three seeds = six YAMLs, written to ``configs/experiments_clean/refit/``.

Design decisions baked in here:

- **fp32 (`use_amp: false`)** so the refit numbers are directly
  comparable to the R2 baseline (Exp 5 clean, also fp32: F1=0.668,
  IR=1.737). The HPO that *found* these topologies ran in AMP; the
  refit *confirms* them in the same precision as the baseline they are
  being compared against. This makes the headline thesis claim
  ("MLP winner vs Linear baseline") a perfectly matched comparison.
- **Three seeds (42, 1, 2)** each producing an independent
  stratified split + initialisation, so the reported mean ± std is an
  honest estimate of generalisation variance (conservative — varies
  split AND init, not just init).
- Everything else inherited from
  ``configs/experiments_clean/exp05_ce_adamw_cosine.yaml`` (CE + AdamW +
  Cosine + dropout=0.2 backbone + clean dataset + early stopping).

The winning params come from ``outputs/hpo/round2/best_params.json``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/experiments_clean/refit")
SEEDS = [42, 1, 2]

# From outputs/hpo/round2/best_params.json (Pareto front, 2 trials).
WINNERS = {
    "t4": {
        "comment": "R2 Pareto winner trial 4 — F1=0.6935, IR=1.638 @ 8 epochs",
        "mlp_hidden_dims": [256],
        "mlp_activation": "gelu",
        "mlp_dropout": 0.5178620555253561,
        "mlp_norm": "none",
        "learning_rate": 0.0003375589571206087,
    },
    "t10": {
        "comment": "R2 Pareto winner trial 10 — F1=0.6886, IR=1.591 @ 8 epochs",
        "mlp_hidden_dims": [1024, 1024, 2048],
        "mlp_activation": "silu",
        "mlp_dropout": 0.08710958014964179,
        "mlp_norm": "layernorm",
        "learning_rate": 0.00018879886412408647,
    },
}


def _build_config(winner_key: str, seed: int) -> dict:
    w = WINNERS[winner_key]
    return {
        "model": {
            "name": "LResNet50E_IR",
            "pretrained": True,
            "num_classes": 7,
            "dropout": 0.2,
            "head": "mlp",
            "arcface_s": 30.0,
            "arcface_m": 0.5,
            "arcface_easy_margin": False,
            "mlp_hidden_dims": w["mlp_hidden_dims"],
            "mlp_activation": w["mlp_activation"],
            "mlp_dropout": w["mlp_dropout"],
            "mlp_norm": w["mlp_norm"],
        },
        "training": {
            "batch_size": 128,
            "learning_rate": w["learning_rate"],
            "num_epochs": 25,
            "optimizer": "adamw",
            "scheduler": "cosineannealingwarmrestarts",
            "loss_function": "cross_entropy",
            "test_size": 0.1,
            "val_size": 0.1,
            "num_workers": 4,
            "random_state": seed,
            "grad_clip_norm": 5.0,
            "early_stopping_patience": 5,
            "use_amp": False,  # fp32 — exact comparison vs R2 baseline
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
            "dataset_file": "data/raw/fairface/fairface_labels_clean.csv",
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
    for key, w in WINNERS.items():
        for seed in SEEDS:
            cfg = _build_config(key, seed)
            # exp* prefix so run_all_experiments.py's glob picks it up.
            fname = f"exp_r4_{key}_s{seed:02d}.yaml"
            path = OUT_DIR / fname
            header = (
                f"# Phase 4 refit — {w['comment']}\n"
                f"# Topology: hidden={w['mlp_hidden_dims']} act={w['mlp_activation']} "
                f"drop={w['mlp_dropout']:.4f} norm={w['mlp_norm']} lr={w['learning_rate']:.2e}\n"
                f"# Seed: {seed} · 25 epochs · fp32 (use_amp=false) · clean dataset\n"
                f"# Generated by scripts/generate_refit_configs.py — do not edit by hand.\n\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written.append(str(path))

    print(f"Wrote {len(written)} refit configs to {OUT_DIR}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
