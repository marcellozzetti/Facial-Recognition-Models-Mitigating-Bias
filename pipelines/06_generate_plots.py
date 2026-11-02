"""Pipeline stage — geração de assets visuais (gráficos e matrizes).

Executa após 05_train_ablation.py para cada config+seed.
Pode rodar em modo multi-seed (todos os seeds de uma config) ou single-seed.

Saídas em outputs/etapa2/<config>/<seed>/plots/:
  - training_curves.png  — F1-macro + DR por época
  - confusion_matrix.png — normalizada por linha
  - per_class_f1.png     — barra horizontal com F1 por classe

Uso (single config, todos os seeds):
    python -I pipelines/06_generate_plots.py --config-id A

Uso (seed específico):
    python -I pipelines/06_generate_plots.py --config-id A --seed 42

Uso (curvas multi-seed sobrepostas — útil para slides comparativos):
    python -I pipelines/06_generate_plots.py --config-id A --multi-seed-curves
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.conditioning import wrap_convnext_with_film  # noqa: E402
from face_bias.fairness.disparity_ratio import per_class_f1  # noqa: E402

logger = logging.getLogger("pipelines.06_generate_plots")

FAIRFACE_CLASSES = 7
# Ordem ALFABÉTICA — idêntica ao sorted() do pipeline de treino
# sorted(["White","Black","Latino_Hispanic","East Asian","Southeast Asian","Indian","Middle Eastern"])
RACE_LABELS = [
    "Black", "East Asian", "Indian", "Latino_Hispanic",
    "Middle Eastern", "Southeast Asian", "White"
]
# Nomes para exibição nos gráficos
RACE_DISPLAY = [
    "Black", "East Asian", "Indian", "Latino/Hispanic",
    "Middle Eastern", "SE Asian", "White"
]
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── Paleta consistente com os slides (IBM design) ────────────────────────────
BLUE_MAIN   = "#1F2A4E"
BLUE_MID    = "#3D68BB"
BLUE_LIGHT  = "#C8D8F0"
RED_MAIN    = "#8B3030"
RED_LIGHT   = "#FDE8E8"
GREEN_MAIN  = "#2A7A4F"
GREEN_LIGHT = "#E8F5EE"
GREY_BG     = "#F9F9F7"
GREY_TEXT   = "#6B7585"
ORANGE      = "#C4792A"
SEED_COLORS = {42: BLUE_MID, 1: GREEN_MAIN, 2: ORANGE, 3: "#8B3030"}


# ────────────────────────────────────────────────────────────────────────────
# Dataset (idêntico ao 05_train_ablation.py, sem augmentation)
# ────────────────────────────────────────────────────────────────────────────
class FairFaceDataset(Dataset):
    def __init__(self, csv_path: Path, img_root: Path, transform):
        df = pd.read_csv(csv_path)
        # Replicar exatamente o mapeamento do pipeline de treino: sorted(unique(race))
        class_names = sorted(df["race"].unique().tolist())
        race_map = {c: i for i, c in enumerate(class_names)}
        df["label"] = df["race"].map(race_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        self.df = df.reset_index(drop=True)
        self.class_names = class_names
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        img_path = self.img_root / row["file"]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), int(row["label"])


# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.facecolor": GREY_BG,
        "axes.facecolor":   GREY_BG,
        "axes.edgecolor":   GREY_TEXT,
        "axes.labelcolor":  BLUE_MAIN,
        "xtick.color":      GREY_TEXT,
        "ytick.color":      GREY_TEXT,
        "text.color":       BLUE_MAIN,
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
    })


def plot_training_curves(histories: dict[int, list], out_path: Path, config_id: str = "A"):
    """Curvas de treino: Loss (train+val) | F1-macro | DR — uma linha por seed."""
    _style()
    fig, (ax_loss, ax_f1, ax_dr) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(GREY_BG)

    for seed, hist in histories.items():
        color = SEED_COLORS.get(seed, BLUE_MID)
        epochs      = [h["epoch"] for h in hist]
        train_loss  = [h["train_loss"] for h in hist]
        val_loss    = [h["val_loss"] for h in hist]
        f1          = [h["val_f1_macro"] for h in hist]
        dr          = [h["val_disparity_ratio"] for h in hist]
        best_ep     = epochs[f1.index(max(f1))]

        # Loss: sólido = treino, tracejado = val
        ax_loss.plot(epochs, train_loss, color=color, linewidth=2.0,
                     linestyle="-",  label=f"Seed {seed} train")
        ax_loss.plot(epochs, val_loss,   color=color, linewidth=2.0,
                     linestyle="--", label=f"Seed {seed} val", alpha=0.8)

        # F1-macro
        ax_f1.plot(epochs, f1, color=color, linewidth=2.2, label=f"Seed {seed}")
        ax_f1.axvline(best_ep, color=color, linestyle=":", linewidth=1.2, alpha=0.6)
        ax_f1.scatter([best_ep], [max(f1)], color=color, s=80, zorder=5)

        # DR
        ax_dr.plot(epochs, dr, color=color, linewidth=2.2, label=f"Seed {seed}")
        ax_dr.axvline(best_ep, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    # Loss: legenda compacta com ícones de linha
    solid_patch  = plt.Line2D([0], [0], color=GREY_TEXT, linewidth=2, linestyle="-",  label="train")
    dashed_patch = plt.Line2D([0], [0], color=GREY_TEXT, linewidth=2, linestyle="--", label="val")
    seed_patches = [plt.Line2D([0], [0], color=SEED_COLORS.get(s, BLUE_MID), linewidth=2, label=f"Seed {s}")
                    for s in histories]
    ax_loss.legend(handles=[solid_patch, dashed_patch] + seed_patches, fontsize=9, ncol=2)

    for ax, ylabel, title in [
        (ax_loss, "Loss (cross-entropy)",          "Loss por época"),
        (ax_f1,   "F1-macro (val)",                "F1-macro por época"),
        (ax_dr,   "Disparity Ratio (min/max F1)",  "DR por época"),
    ]:
        ax.set_xlabel("Época", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linestyle=":")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    ax_f1.legend(fontsize=10)
    ax_dr.legend(fontsize=10)

    fig.suptitle(f"Config {config_id} — dinâmica de treinamento (FairFace val)",
                 fontsize=14, fontweight="bold", color=BLUE_MAIN)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves → %s", out_path)


def plot_per_class_f1(per_class: dict[str, float], seed: int, out_path: Path):
    """Barra horizontal de F1 por classe."""
    _style()
    classes = RACE_DISPLAY
    values  = [per_class.get(c, 0.0) for c in classes]
    colors  = [RED_MAIN if v == min(values) else BLUE_MID for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(GREY_BG)
    bars = ax.barh(classes, values, color=colors, edgecolor="none", height=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 por classe", fontsize=12)
    ax.set_title(f"Config A seed {seed} — F1 por classe racial (FairFace val)")
    ax.axvline(np.mean(values), color=GREY_TEXT, linestyle="--", linewidth=1.5, label=f"Média {np.mean(values):.3f}")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10, color=BLUE_MAIN)
    ax.legend(fontsize=11)
    ax.grid(True, axis="x", alpha=0.2, linestyle=":")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class F1 → %s", out_path)


def plot_confusion_matrix(cm: np.ndarray, seed: int, out_path: Path):
    """Matriz de confusão normalizada por linha (proporção de predições)."""
    _style()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "bluewhite", [GREY_BG, BLUE_LIGHT, BLUE_MID, BLUE_MAIN]
    )

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(GREY_BG)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    n = len(RACE_LABELS)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else BLUE_MAIN
            # Highlight Latinx row
            if i == 2:
                color = "white" if val > 0.35 else ORANGE
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    # Mesma ordem alfabética do treino: Black, East Asian, Indian, Latino_Hispanic, Middle Eastern, SE Asian, White
    short = ["Black", "E. Asian", "Indian", "Latinx", "M. East", "SE Asian", "White"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Config A seed {seed} — Confusão normalizada por linha (FairFace val)", fontsize=12, fontweight="bold")

    # Contorno na linha Latinx (índice 3 na ordem alfabética)
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, 2.5), n, 1, linewidth=2.5, edgecolor=ORANGE, facecolor="none")
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


# ────────────────────────────────────────────────────────────────────────────
# Eval (inference on val set)
# ────────────────────────────────────────────────────────────────────────────
def run_eval(
    checkpoint: Path,
    dataset_root: Path,
    val_labels: Path,
    config_id: str,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, float]]:
    """Roda inferência no val set. Retorna (confusion_matrix, per_class_f1_dict)."""
    from sklearn.metrics import confusion_matrix as sk_cm, f1_score

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = FairFaceDataset(val_labels, dataset_root, transform)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, FAIRFACE_CLASSES)

    if config_id != "A":
        dim_z = 10 if config_id == "B" else 512
        model = wrap_convnext_with_film(model, dim_z=dim_z)

    ckpt_data = torch.load(checkpoint, map_location="cpu")
    state = ckpt_data["state_dict"] if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state)
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], (list, tuple)):
                imgs, labels = batch[0], batch[1]
            else:
                imgs, labels = batch
            imgs = imgs.to(dev)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = sk_cm(all_labels, all_preds, labels=list(range(FAIRFACE_CLASSES)))
    f1s = f1_score(all_labels, all_preds, average=None, labels=list(range(FAIRFACE_CLASSES)))
    # Usa RACE_DISPLAY para nomes legíveis; ordem igual ao sorted() do treino
    per_class = {RACE_DISPLAY[i]: float(f1s[i]) for i in range(FAIRFACE_CLASSES)}
    return cm, per_class


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Gera plots para ablação")
    parser.add_argument("--config-id", required=True, choices=["A", "B", "C"])
    parser.add_argument("--seed", type=int, default=None, help="Seed específica; omitir para rodar todas disponíveis")
    parser.add_argument("--output-dir", default="outputs/etapa2")
    parser.add_argument("--dataset-root", default="data/processed/fairface_aligned")
    parser.add_argument("--val-labels", default="data/raw/fairface/val_labels.csv")
    parser.add_argument("--multi-seed-curves", action="store_true", help="Curvas com todos os seeds sobrepostos num único gráfico")
    parser.add_argument("--skip-eval", action="store_true", help="Pular inferência (só curvas de treino)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    config_dir = output_dir / f"config_{args.config_id}"
    plots_dir  = config_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Descobre seeds disponíveis
    available_seeds = []
    for hist_file in sorted(config_dir.glob(f"{args.config_id}_seed*_history.json")):
        s = int(hist_file.stem.split("seed")[1].split("_")[0])
        available_seeds.append(s)

    if args.seed is not None:
        seeds_to_process = [args.seed]
    else:
        seeds_to_process = available_seeds

    if not seeds_to_process:
        logger.error("Nenhum arquivo de histórico encontrado em %s", config_dir)
        sys.exit(1)

    # Carrega histórias
    histories = {}
    for s in seeds_to_process:
        hist_path = config_dir / f"{args.config_id}_seed{s}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[s] = json.load(f)

    # Curvas multi-seed sobrepostas
    if args.multi_seed_curves or len(histories) > 1:
        plot_training_curves(histories, plots_dir / "training_curves_all_seeds.png", config_id=args.config_id)
        logger.info("Curvas multi-seed salvas.")

    # Por seed: curva individual + eval
    for seed, hist in histories.items():
        seed_plots = plots_dir / f"seed{seed}"
        seed_plots.mkdir(exist_ok=True)

        # Curva individual
        plot_training_curves({seed: hist}, seed_plots / "training_curves.png", config_id=args.config_id)

        if args.skip_eval:
            continue

        checkpoint = config_dir / f"{args.config_id}_seed{seed}.pt"
        if not checkpoint.exists():
            logger.warning("Checkpoint não encontrado: %s — pulando eval.", checkpoint)
            continue

        logger.info("Rodando eval seed %d ...", seed)
        cm, per_class = run_eval(
            checkpoint,
            REPO_ROOT / args.dataset_root,
            REPO_ROOT / args.val_labels,
            args.config_id,
        )
        plot_per_class_f1(per_class, seed, seed_plots / "per_class_f1.png")
        plot_confusion_matrix(cm, seed, seed_plots / "confusion_matrix.png")

        # Salva JSON de per_class_f1 para uso posterior
        with open(seed_plots / "per_class_f1.json", "w") as f:
            json.dump(per_class, f, indent=2)

    logger.info("Pronto. Plots em %s", plots_dir)


if __name__ == "__main__":
    main()
