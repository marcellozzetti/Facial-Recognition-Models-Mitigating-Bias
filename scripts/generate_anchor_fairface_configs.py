"""Generate the FairFace-recipe ANCHOR configs (baseline-positioning).

These are NOT one of the 5 attribution factors. They reproduce the
*recipe of the FairFace-paper itself* (Kärkkäinen & Joo, WACV 2021) on
our exact task (FairFace race 7-class in-domain) so we can anchor
"under the paper's own recipe, our pipeline lands here" against the
banca question *"vocês não reproduziram o paper?"*.

Recipe extracted from the paper §4.2 (verbatim):
  - Backbone: ResNet-34 (not 50)
  - Optimizer: ADAM
  - Learning rate: 0.0001
  - No augmentation / batch / epochs / image-size explicitly stated;
    we hold those at our matched-protocol defaults so the comparison
    isolates {backbone, optimizer, LR}.

Everything else MATCHED with the rest of the project (clean dataset,
3 seeds {42,1,2}, criterion val_f1_macro, fp32, num_workers=0,
25 epochs with early stopping). Control reused = our CE+linear
ResNet-50 definitive baseline.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/anchor_fairface_recipe")
SEEDS = [42, 1, 2]
CLEAN_CSV = "data/raw/fairface/fairface_labels_clean.csv"


def _cfg(seed: int) -> dict:
    return {
        "model": {
            "name": "LResNet50E_IR",  # mlflow tag (alias); arch below
            "backbone_arch": "resnet34",  # FairFace-paper backbone
            "pretrained": True,
            "num_classes": 7,
            "dropout": 0.2,
            "head": "linear",
            "arcface_s": 30.0,
            "arcface_m": 0.5,
            "arcface_easy_margin": False,
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 0.0001,  # FairFace paper
            "weight_decay": 0.0,  # FairFace paper does not specify -> 0
            "num_epochs": 25,
            "optimizer": "adam",  # FairFace paper
            "scheduler": "cosineannealingwarmrestarts",
            "loss_function": "cross_entropy",
            "test_size": 0.1,
            "val_size": 0.1,
            "num_workers": 0,
            "random_state": seed,
            "grad_clip_norm": 5.0,
            "early_stopping_patience": 5,
            "use_amp": False,
            # checkpoint_metric omitted -> inherits val_f1_macro
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
    for seed in SEEDS:
        cfg = _cfg(seed)
        fname = f"anchor_fairface_recipe_s{seed:02d}.yaml"
        header = (
            f"# FairFace-recipe ANCHOR — seed={seed}\n"
            f"# ResNet-34 + ADAM lr=1e-4 (FairFace paper recipe) on race 7-class\n"
            f"# in-domain. NOT one of the 5 attribution factors — anchor for\n"
            f"# baseline-positioning vs the dataset-paper recipe.\n"
            f"# See docs/baseline_positioning.md.\n\n"
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
