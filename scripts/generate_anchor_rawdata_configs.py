"""Generate the Raw-Data ANCHOR configs.

Anchor 🅓 — reproduces our best-recipe (CE+linear ResNet-50 AdamW
lr=1e-3 @224) on **raw FairFace data** (no multi-face cleaning, no
MTCNN re-alignment). Isolates the contribution of OUR preprocessing
pipeline (multi-face filter + MTCNN realign) to absolute F1/IR.

**Não toca o pipeline dos 5 fatores.** Tudo num path SEPARADO:
  - CSV:    data/raw_anchor/fairface_original/fairface_labels.csv (raw, 97k)
  - imagens: data/raw_anchor/fairface_original/images/{train,val}/*.jpg
             (FairFace publicado original, sem MTCNN realign do nosso projeto)

Matched-comparison nota: o split (80/10/10 estratificado, seed-fixo) é
o MESMO usado em F1-F5, rodando sobre o CSV raw. Isto isola APENAS o
efeito do data-preprocessing (cleaning + realign), mantendo o split
casado. A diferença "split oficial FairFace train/val" fica para um
anchor adicional (não escopo desta peça).

Veja docs/anchor_rawdata_README.md sobre como adquirir os dados raw.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/anchor_rawdata")
SEEDS = [42, 1, 2]

# Paths das imagens RAW originais (read-only — não alterar):
#   - CSV raw (97k linhas, sem multi-face cleaning) já existe localmente
#   - Imagens FairFace publicado original em
#     data/raw/bucket/fairface-img-margin125/{train,val}/*.jpg
# F1-F5 NÃO usam estes paths — eles usam fairface_labels_clean.csv +
# data/processed/fairface_aligned. Os dois pipelines coexistem.
# Updated 2026-05-22: bucket reorganized — margin125 (existing) e margin025
# (NEW, baixado para o anchor 🅔 Hassanpour-protocol).
RAW_CSV = "data/raw/fairface/fairface_labels.csv"
RAW_IMAGES = "data/raw/bucket/fairface-img-margin125"


def _cfg(seed: int) -> dict:
    return {
        "model": {
            "name": "LResNet50E_IR",
            "backbone_arch": "resnet50",  # mesmo do nosso controle CE+linear
            "pretrained": True,
            "num_classes": 7,
            "dropout": 0.2,
            "head": "linear",
            "arcface_s": 30.0,
            "arcface_m": 0.5,
            "arcface_easy_margin": False,
        },
        "training": {
            "batch_size": 128,         # mesmo do controle
            "learning_rate": 0.001,    # AdamW lr=1e-3 (nosso recipe)
            "num_epochs": 25,
            "optimizer": "adamw",
            "scheduler": "cosineannealingwarmrestarts",
            "loss_function": "cross_entropy",
            "test_size": 0.1,
            "val_size": 0.1,
            "num_workers": 0,           # Windows deadlock-proof (matched)
            "random_state": seed,
            "grad_clip_norm": 5.0,
            "early_stopping_patience": 5,
            "use_amp": False,           # fp32 — matched
            # checkpoint_metric default = val_f1_macro (matched, correto)
        },
        "image": {
            "image_size": [224, 224],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "preprocessing": {
            # NB: estes campos existem por compatibilidade do schema; o
            # anchor NÃO re-executa MTCNN — lê direto de RAW_IMAGES.
            "processing_type": "detect_adjust_image",
            "num_workers": 4,
            "rotate_angle": 45,
            "num_rotations": 8,
            "max_faces_per_image": 1,
        },
        "data": {
            "dataset_file": RAW_CSV,
            "dataset_image_input_path": RAW_IMAGES,
            "dataset_image_output_path": RAW_IMAGES,
            "balance": "undersample",   # mesmo controle (raça undersample)
        },
        "bucket": {
            # NB: dados originais ja estao no disco em data/raw/bucket/ —
            # nao re-baixar nada. Bloco mantido apenas para compat de schema.
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
        fname = f"exp_anc_rawdata_s{seed:02d}.yaml"
        header = (
            f"# Raw-Data ANCHOR — seed={seed}\n"
            f"# Mesmo recipe do controle CE+linear (RN-50 AdamW 1e-3 @224)\n"
            f"# Diferenca: CSV raw (97k, sem multi-face cleaning) +\n"
            f"#            imagens raw (sem MTCNN re-alignment do nosso projeto)\n"
            f"# NAO toca o pipeline F1-F5; data em data/raw_anchor/.\n"
            f"# Ver docs/anchor_rawdata_README.md.\n"
            f"# Generated by scripts/generate_anchor_rawdata_configs.py.\n\n"
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
