"""Gera figura do pipeline experimental em 3 etapas (Cap. 4 §4.2).

Layout top-down, 1 coluna, com:
    - 2 metas na barra lateral esquerda (META 1 / META 2)
    - 3 caixas de etapa empilhadas com eyebrow, título, subtítulo
      e grade 3 colunas: Insumo | Método | Saída
    - Setas com rótulo de dado transferido entre etapas
    - Etapa 2 (proposta principal) recebe destaque visual

Saida: docs/tese/images/fig_pipeline_6etapas.png
  (mantém o nome para não quebrar a referência LaTeX existente)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY      = (31/255,  42/255,  78/255)
NAVY_SOFT = (61/255, 104/255, 187/255)
BLUE_TINT = (233/255, 238/255, 248/255)
GRAY_DK   = (61/255,  66/255,  78/255)
GRAY_MD   = (112/255, 118/255, 130/255)
GRAY_LT   = (232/255, 234/255, 237/255)
PAPER     = (250/255, 250/255, 248/255)
ACCENT    = (192/255,  57/255,  43/255)
GREEN     = (46/255,  125/255,  50/255)
WHITE     = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Stage:
    number: int
    title: str
    subtitle: str
    insumo: str
    metodo: str
    saida: str
    highlight: bool = False


STAGES = [
    Stage(
        1,
        "Extrator MST e diagnóstico fenotípico",
        "ViT-B/16 treinado em MSTE + CCv2 → vetor de 10 tons de pele",
        insumo="MSTE (~1.5k imgs)\nCCv2 (~55k frames)\nFairFace val set",
        metodo="Fine-tuning ViT-B/16\nSoftmax 10 classes\nValidação humana cega",
        saida="Extrator MST z∈ℝ¹⁰\nMatriz MST × 7 raças\n(FairFace)",
    ),
    Stage(
        2,
        "Classificador racial condicionado + ablation",
        "ConvNeXt-T + FiLM condicionado em z — ablation A / B / C e 3 baselines",
        insumo="FairFace train split\n(86 744 imgs, 7 classes)\nvetor z da Etapa 1",
        metodo="ConvNeXt-T + FiLM\nAblation A / B / C\n3 seeds · bootstrap CI",
        saida="Classificador racial\nF1-macro · DR\nWorst-class F1",
        highlight=True,
    ),
    Stage(
        3,
        "Transferência para verificação facial",
        "Feature-freeze do encoder + FiLM → cabeça de verificação em RFW e BFW",
        insumo="RFW (~11k pares)\nBFW (~50k pares)\nencoder + FiLM congelado",
        metodo="Feature-freeze\n(LAFTR, Madras 2018)\nNova cabeça de verif.",
        saida="TAR@FAR=10⁻⁴\npor grupo racial\nRFW · BFW",
    ),
]

METAS = [
    ("META 1", (1, 2), NAVY_SOFT),
    ("META 2", (3, 3), GRAY_DK),
]

ARROW_LABELS = {
    (1, 2): "z ∈ ℝ¹⁰",
    (2, 3): "encoder + FiLM\n(congelado)",
}


def _box(ax, x, y, w, h, fc, ec, lw=1.2, rounding=0.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw,
    ))


def _arrow_with_label(ax, cx, y_from, y_to, label: str, col=NAVY):
    ax.add_patch(FancyArrowPatch(
        (cx, y_from), (cx, y_to),
        arrowstyle="-|>", color=col, lw=1.8,
        mutation_scale=14, shrinkA=1, shrinkB=1,
    ))
    mid = (y_from + y_to) / 2
    ax.text(cx + 2.5, mid, label,
            ha="left", va="center",
            color=col, fontsize=8.5, style="italic",
            bbox=dict(fc=GRAY_LT, ec="none", pad=1.5, boxstyle="round"))


def _draw_stage(ax, stage: Stage, x: float, y: float, w: float, h: float) -> None:
    highlighted = stage.highlight
    body_fc = BLUE_TINT if highlighted else PAPER
    body_ec = NAVY if highlighted else GRAY_MD
    body_lw = 2.4 if highlighted else 1.1
    _box(ax, x, y, w, h, fc=body_fc, ec=body_ec, lw=body_lw, rounding=1.2)

    strip_w = 7.5
    strip_fc = NAVY if highlighted else GRAY_DK
    _box(ax, x, y, strip_w, h, fc=strip_fc, ec=strip_fc, lw=0, rounding=1.2)
    ax.text(x + strip_w / 2, y + h / 2, str(stage.number),
            ha="center", va="center", color=WHITE, fontsize=28, fontweight="bold")

    tx = x + strip_w + 3.0
    inner_w = w - strip_w - 4.0
    ax.text(tx, y + h * 0.88, stage.title,
            ha="left", va="center", color=NAVY, fontsize=12.5, fontweight="bold")
    ax.text(tx, y + h * 0.76, stage.subtitle,
            ha="left", va="center", color=GRAY_MD, fontsize=8.5, style="italic")

    sep_y = y + h * 0.65
    ax.plot([tx, x + w - 1.5], [sep_y, sep_y], color=GRAY_LT, lw=0.8)

    col_w = inner_w / 3
    headers = ["Insumo", "Método", "Saída"]
    contents = [stage.insumo, stage.metodo, stage.saida]
    header_colors = [GRAY_MD, NAVY if highlighted else GRAY_DK, GRAY_MD]

    for ci, (hdr, content, hcol) in enumerate(zip(headers, contents, header_colors)):
        cx_col = tx + ci * col_w + col_w / 2
        ax.text(cx_col, y + h * 0.57, hdr,
                ha="center", va="center", color=hcol, fontsize=8.0,
                fontweight="bold")
        ax.text(cx_col, y + h * 0.30, content,
                ha="center", va="center", color=GRAY_DK, fontsize=7.5,
                linespacing=1.35)
        if ci < 2:
            vx = tx + (ci + 1) * col_w
            ax.plot([vx, vx], [y + 1, sep_y - 0.5], color=GRAY_LT, lw=0.7)


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 97, "Pipeline experimental — 3 etapas",
            ha="center", va="center", color=NAVY, fontsize=17, fontweight="bold")
    ax.text(50, 93.5,
            "Meta 1: extrator MST + classificador condicionado  ·  Meta 2: transferência para verificação",
            ha="center", va="center", color=GRAY_MD, fontsize=9.5)

    box_w = 78.0
    box_h = 22.0
    box_x = 14.5
    top_y = 91.0
    gap   = 5.0

    positions: dict[int, tuple[float, float]] = {}
    for i, stage in enumerate(STAGES):
        y = top_y - (i + 1) * box_h - i * gap
        _draw_stage(ax, stage, box_x, y, box_w, box_h)
        positions[stage.number] = (box_x, y)

    arrow_cx = box_x + box_w / 2
    for i in range(1, len(STAGES)):
        y_from = positions[i][1]
        y_to   = positions[i + 1][1] + box_h
        label  = ARROW_LABELS.get((i, i + 1), "")
        _arrow_with_label(ax, arrow_cx, y_from - 0.3, y_to + 0.3, label)

    for label, (start, end), color in METAS:
        y_top = positions[start][1] + box_h
        y_bot = positions[end][1]
        bar_x, bar_w = 5.0, 6.5
        _box(ax, bar_x, y_bot, bar_w, y_top - y_bot, fc=color, ec=color, lw=0, rounding=1.0)
        ax.text(bar_x + bar_w / 2, (y_top + y_bot) / 2, label,
                rotation=90, ha="center", va="center",
                color=WHITE, fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = repo_root / "docs" / "tese" / "images" / "fig_pipeline_6etapas.png"
    build_figure(out)


if __name__ == "__main__":
    main()
