"""Generate the experiment figures from a train + evaluate run.

Reproduces the four MBA-Cap. 4 figures (Metricas, Matriz) and adds two
fairness charts that did not exist in the MBA. All artefacts are written
as PNG and (when relevant) PDF next to the source JSON/CSV files.

Usage:
    python scripts/plot_experiment.py \
        --train-dir   outputs/<train_run_id> \
        --eval-dir    outputs/evaluation/<eval_run_id> \
        --out-dir     outputs/figures/<some_label>

If --out-dir is omitted, figures land beside the inputs (history.json
neighbours metricas.png; confusion_matrix.csv neighbours matriz.png).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------- styling -------------------------------------

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

CMAP = "viridis"
BAR_COLOR = "#4477AA"
HIGHLIGHT = "#CC3311"


def _save(fig: plt.Figure, path: Path, *, also_pdf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------- training history ------------------------------


def plot_training_metrics(history_path: Path, out_path: Path) -> None:
    """4-quadrant: train_loss/val_loss, accuracy, F1 macro, log_loss-proxy IR.

    Mirrors MBA Cap. 4 figures `experimento{N}Metricas` (Loss / Accuracy /
    Precision / Log Loss). Substitutes Precision-by-class (which doesn't
    exist in our history) for the macro F1, and replaces the standalone
    Log Loss quadrant with the inequity-rate trace -- the dissertation's
    headline fairness signal.
    """
    history = pd.read_json(history_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    epoch = history["epoch"]

    # (1) Loss
    ax = axes[0, 0]
    ax.plot(epoch, history["train_loss"], label="train", color=BAR_COLOR)
    ax.plot(epoch, history["val_loss"], label="val", color=HIGHLIGHT, linestyle="--")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # (2) Accuracy
    ax = axes[0, 1]
    ax.plot(epoch, history["val_accuracy"], color=BAR_COLOR, marker="o", markersize=3)
    ax.set_title("Validation Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    # (3) F1 macro
    ax = axes[1, 0]
    ax.plot(epoch, history["val_f1_macro"], color=BAR_COLOR, marker="o", markersize=3)
    ax.set_title("Validation F1 (macro) over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)

    # (4) Inequity rate
    ax = axes[1, 1]
    ax.plot(epoch, history["val_f1_inequity_rate"], color=HIGHLIGHT, marker="o", markersize=3)
    ax.axhline(1.0, color="gray", linestyle=":", label="perfect fairness (IR=1)")
    ax.set_title("Inequity Rate (F1 across races) over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IR = max(F1) / min(F1)")
    ax.legend()

    fig.suptitle(f"Training metrics — {history_path.parent.name}", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path, also_pdf=True)


# ---------------------------- confusion matrix -----------------------------


def plot_confusion_matrix(cm_path: Path, out_path: Path) -> None:
    """Heatmap with row-normalised cells and absolute counts annotated."""
    cm = pd.read_csv(cm_path, index_col=0)
    counts = cm.values
    norm = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts[i, j]
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=colour, fontsize=8)

    fig.colorbar(im, ax=ax, label="Row-normalised proportion")
    ax.set_title(f"Confusion matrix — {cm_path.parent.name}")
    fig.tight_layout()
    _save(fig, out_path, also_pdf=True)


# ---------------------------- per-class bars -------------------------------


def plot_per_class(per_class_path: Path, out_path: Path) -> None:
    """Grouped bar chart of precision / recall / F1 per class."""
    df = pd.read_csv(per_class_path)
    classes = df["class"].tolist()
    width = 0.27
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, df["precision"], width=width, label="Precision", color="#4477AA")
    ax.bar(x, df["recall"], width=width, label="Recall", color="#EE7733")
    ax.bar(x + width, df["f1"], width=width, label="F1", color="#228833")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_path, also_pdf=True)


# ---------------------------- fairness audit -------------------------------


def plot_fairness(audit_path: Path, per_class_path: Path, out_path: Path) -> None:
    """Per-class F1 bars with mean line + textual IR/Gini/gap annotations."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    df = pd.read_csv(per_class_path).sort_values("f1", ascending=False).reset_index(drop=True)

    f1 = audit["f1"]
    summary = (
        f"Inequity rate (F1) = {f1['inequity_rate']:.3f}\n"
        f"max - min          = {f1['max_min_disparity']:.3f}\n"
        f"std                = {f1['std']:.3f}\n"
        f"coef. of variation = {f1['coefficient_of_variation']:.3f}\n"
        f"Gini               = {f1['gini']:.3f}"
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(df["class"], df["f1"], color=BAR_COLOR)
    ax.axhline(f1["mean"], color=HIGHLIGHT, linestyle="--", label=f"mean F1 = {f1['mean']:.3f}")
    ax.axhline(f1["min"], color="gray", linestyle=":", linewidth=1)
    ax.axhline(f1["max"], color="gray", linestyle=":", linewidth=1)

    for bar, value in zip(bars, df["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max(1.0, f1["max"] * 1.15))
    ax.set_ylabel("F1 score")
    ax.set_title("Fairness audit — F1 by class")
    ax.legend(loc="upper right")
    ax.text(
        0.99,
        0.05,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontfamily="monospace",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#999999"},
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out_path, also_pdf=True)


# ------------------------------ main ---------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot experiment figures.")
    parser.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="outputs/<train_run_id> directory (contains history.json)",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=Path,
        help="outputs/evaluation/<eval_run_id> directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, all figures go into this folder; otherwise next to the source files",
    )
    args = parser.parse_args(argv)

    history = args.train_dir / "history.json"
    cm = args.eval_dir / "confusion_matrix.csv"
    per_class = args.eval_dir / "per_class.csv"
    audit = args.eval_dir / "fairness_audit.json"

    for path in (history, cm, per_class, audit):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.out_dir is None:
        out_metricas = args.train_dir / "metricas.png"
        out_matriz = args.eval_dir / "matriz.png"
        out_per_class = args.eval_dir / "per_class.png"
        out_fairness = args.eval_dir / "fairness.png"
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_metricas = args.out_dir / "metricas.png"
        out_matriz = args.out_dir / "matriz.png"
        out_per_class = args.out_dir / "per_class.png"
        out_fairness = args.out_dir / "fairness.png"

    plot_training_metrics(history, out_metricas)
    plot_confusion_matrix(cm, out_matriz)
    plot_per_class(per_class, out_per_class)
    plot_fairness(audit, per_class, out_fairness)

    print("Wrote:")
    for path in (out_metricas, out_matriz, out_per_class, out_fairness):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
