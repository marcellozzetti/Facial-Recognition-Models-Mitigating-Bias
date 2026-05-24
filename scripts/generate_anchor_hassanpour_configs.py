"""Generate Anchor 🅔 (Hassanpour-protocol) configs — combined Caminho 2 + 3.

Reproduz integralmente o setup metodológico de Hassanpour et al. 2024
(arXiv 2410.24148) sobre nossas duas arquiteturas-chave (controle RN-50
+ ConvNeXt-T do Fator 5), fechando simultaneamente os 3 confundidores
metodológicos que separam nosso pipeline do SOTA:

  1. **Padding=0.25** (224×224 tight crops nativos do FairFace publication)
     em vez de padding=1.25 (resize de 448→224).
  2. **Split oficial do FairFace** (86,744 train + 10,954 val) em vez do
     nosso 80/10/10 estratificado próprio. Test = val oficial completo.
  3. **No-undersample** — imbalance natural (~1.8x maior/menor classe)
     em vez do nosso undersample por raça.

Pergunta da auditoria: *"Sob protocolo idêntico ao de Hassanpour, nosso
ConvNeXt-T bate o ResNet-34 deles (0.720 acc) e/ou se aproxima da
FaceScanPaliGemma SOTA (0.757 acc)?"*

Predição a priori (com base na decomposição de gap em
`docs/baseline_positioning.md §4.1`):
  - Controle RN-50 nosso sob protocolo Hassanpour: ~0.70-0.72 acc
    (vs Hassanpour RN-34 = 0.72)
  - ConvNeXt-T nosso sob protocolo Hassanpour: ~0.73-0.75 acc
    (vs SOTA VLM = 0.757; ConvNeXt ainda 1-2pp abaixo por ser CNN, não VLM)

Veja:
  - `docs/sota_7class_race_audit.md` §5 (decomposição do gap)
  - `docs/THESIS_STATEMENT.md` §3.3 (escopo da ablação)
  - `configs/anchor_hassanpour/README.md` (dispatch)

CONFUNDIDORES REMANESCENTES (este anchor NÃO fecha):
  - Multi-face cleaning (continuamos com CSV clean 72k vs raw 97k); este
    confundidor favorece nós em ~+0.7pp, então o gap residual após 🅔
    será conservadoramente atribuível a "recipe deles é melhor-HPO" e/ou
    diferenças de test set (10,954 val oficial vs nosso test split).
  - MTCNN re-align (continuamos com nosso preprocessing); Anchor 🅓 já
    isolou esse fator (cost ~0.7pp F1).

Para o user-friendly: o objetivo é NÃO carregar TODOS os confundidores
remanescentes. 🅔 fecha os 3 dominantes; o restante fica como caveat
declarado.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs/anchor_hassanpour")
SEEDS = [42, 1, 2]

# Dados Hassanpour-protocol:
#   - CSV raw oficial (97k linhas com prefixos train/ e val/ no campo `file`)
#   - Imagens padding=0.25 (224×224 native tight crops)
RAW_CSV = "data/raw/fairface/fairface_labels.csv"
RAW_IMAGES_P025 = "data/raw/bucket/fairface-img-margin025"

# Duas arquiteturas-chave — replicam o pareamento da ablação 🅑.
# Tudo idêntico aos correspondentes definitivos exceto:
#   - data path (padding=0.25)
#   - split_protocol = "official"
#   - balance = "none"
#   - CSV raw (não clean)
ARMS = {
    "control": {
        "arch": "resnet50",
        "lr": 1e-3,
        "batch_size": 128,
        "label": "Controle CE+linear ResNet-50 (Hassanpour-protocol)",
    },
    "convnext": {
        "arch": "convnext_tiny",
        "lr": 1e-4,
        "batch_size": 64,
        "label": "ConvNeXt-T + linear (Hassanpour-protocol)",
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
            # test_size é IGNORADO quando split_protocol=official (test = val
            # oficial). val_size é interpretado como fração DO train_pool.
            "test_size": 0.1,
            "val_size": 0.25,  # 25% do train/ oficial como val (matching Hassanpour)
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
            # mantido por compatibilidade do schema; pipeline NÃO re-executa
            # MTCNN — lê direto de RAW_IMAGES_P025.
            "processing_type": "detect_adjust_image",
            "num_workers": 4,
            "rotate_angle": 45,
            "num_rotations": 8,
            "max_faces_per_image": 1,
        },
        "data": {
            "dataset_file": RAW_CSV,
            "dataset_image_input_path": RAW_IMAGES_P025,
            "dataset_image_output_path": RAW_IMAGES_P025,
            "balance": "none",  # <<< Hassanpour não faz undersample
            "split_protocol": "official",  # <<< split oficial FairFace
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
            fname = f"exp_anc_hass_{arm_key}_s{seed:02d}.yaml"
            header = (
                f"# Anchor 🅔 (Hassanpour-protocol) — {arm['label']} seed={seed}\n"
                f"# Caminho 2+3 combinado: fecha padding + split oficial + no-undersample\n"
                f"# de uma vez, replicando o setup Hassanpour et al. 2024 (arXiv 2410.24148).\n"
                f"# Imagens: padding=0.25 (224x224 native tight crops).\n"
                f"# CSV: raw 97k com prefixos train/ e val/.\n"
                f"# Split: train/ pool (65k) sub-dividido 75/25; test = val/ oficial (11k).\n"
                f"# Recipe matched aos definitivos correspondentes (control=baseline; convnext=Fator 5).\n"
                f"# Ver docs/sota_7class_race_audit.md §5 e configs/anchor_hassanpour/README.md.\n"
                f"# Generated by scripts/generate_anchor_hassanpour_configs.py.\n\n"
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
