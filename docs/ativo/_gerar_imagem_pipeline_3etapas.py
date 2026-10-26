"""Gera figura do pipeline experimental em 3 etapas.

figsize=(6.5, 9.8) ≈ 16.5 cm — tamanho real de inclusão LaTeX (width=16cm).
Conteúdo das linhas calibrado para ≤ 50 chars @ 8.5pt, sem overflow.
GAP=7 entre caixas para acomodar label das setas sem sobreposição.

Saida: docs/tese/images/fig_pipeline_6etapas.png
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY      = (31/255,  42/255,  78/255)
NAVY_SOFT = (61/255, 104/255, 187/255)
BLUE_TINT = (228/255, 236/255, 252/255)
GRAY_DK   = (61/255,  66/255,  78/255)
GRAY_MD   = (112/255, 118/255, 130/255)
GRAY_LT   = (210/255, 215/255, 224/255)
PAPER     = (252/255, 252/255, 250/255)
WHITE     = (1.0, 1.0, 1.0)

FIG_W, FIG_H = 6.5, 9.8
DPI = 200

# ── grid 100 × 100 ────────────────────────────────────────────────────────
# 1 unidade W = 6.5/100 = 0.065 pol = 4.68 pt
# 1 unidade H = 9.8/100 = 0.098 pol = 7.06 pt
# char a 8.5 pt ≈ 4.8 pt de largura → 4.8/4.68 ≈ 1.03 unidades/char
# → 50 chars ocupam ≈ 51.5 unidades; VALUE_X=33, max_x=91 → 58 unidades disp. ✓

BOX_X  = 11.0
BOX_W  = 82.0          # max_x = 93; texto termina em 93-2 = 91
BOX_H  = 25.0
GAP    = 7.0           # gap entre caixas; acomoda label da seta
TOP_Y  = 92.0

# posição y do fundo de cada caixa
#   box[0] = 92-25 = 67
#   box[1] = 67-7-25 = 35
#   box[2] = 35-7-25 = 3
BOX_Y = [TOP_Y - BOX_H,
         TOP_Y - BOX_H - GAP - BOX_H,
         TOP_Y - BOX_H - 2*GAP - 2*BOX_H]

STRIP_W = 9.0
TEXT_X  = BOX_X + STRIP_W + 3.5   # = 23.5
LABEL_W = 11.0
VALUE_X = TEXT_X + LABEL_W         # = 34.5   → 91-34.5 = 56.5 un. disponíveis

# posições verticais DENTRO da caixa (relativas ao fundo y)
TITLE_Y = BOX_H - 4.5    # 20.5
SUB_Y   = BOX_H - 9.5    # 15.5
SEP_Y   = BOX_H - 13.5   # 11.5
ROW_YS  = [BOX_H - 17.0,  #  8.0   primeira linha de dados
            BOX_H - 20.0,  #  5.0
            BOX_H - 23.0]  #  2.0   última linha — fundo do texto ≈ y+0.9 > y ✓


@dataclass
class Stage:
    number: int
    title: str
    subtitle: str
    rows: List[tuple]          # (label, value) — value ≤ 50 chars
    highlight: bool = False


# Conteúdo calibrado: valor ≤ 50 chars cada linha
STAGES = [
    Stage(
        1,
        "Extrator MST e diagnóstico fenotípico",
        "ViT-B/16 treinado em MSTE + CCv2",
        rows=[
            ("Insumo", "MSTE  ·  CCv2 (~55 000 frames)  ·  FairFace val"),
            ("Método", "ViT-B/16 fine-tuning  ·  10 classes  ·  val. cega"),
            ("Saída",  "z ∈ ℝ¹⁰  ·  Matriz MST × 7 raças (FairFace)"),
        ],
    ),
    Stage(
        2,
        "Classificador racial condicionado + ablation",
        "ConvNeXt-T + FiLM condicionado em z — ablation A / B / C",
        rows=[
            ("Insumo", "FairFace train (86 744 imgs)  ·  vetor z da Etapa 1"),
            ("Método", "ConvNeXt-T + FiLM  ·  Ablation A/B/C  ·  3 seeds"),
            ("Saída",  "F1-macro  ·  Disparity Ratio  ·  Worst-class F1"),
        ],
        highlight=True,
    ),
    Stage(
        3,
        "Transferência para verificação facial",
        "Feature-freeze encoder + FiLM → nova cabeça em RFW e BFW",
        rows=[
            ("Insumo", "Encoder + FiLM congelado  ·  RFW  ·  BFW"),
            ("Método", "Feature-freeze (LAFTR 2018)  ·  nova cabeça de verif."),
            ("Saída",  "TAR@FAR = 10⁻⁴  por grupo racial  ·  RFW e BFW"),
        ],
    ),
]

METAS = [
    ("META 1", (0, 1), NAVY_SOFT),
    ("META 2", (2, 2), GRAY_DK),
]

ARROWS = {
    (0, 1): "z ∈ ℝ¹⁰",
    (1, 2): "encoder + FiLM  (congelado)",
}


def _box(ax, x, y, w, h, fc, ec, lw=1.0, r=0.5, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z,
    ))


def _draw_stage(ax, s: Stage, y: float):
    hl = s.highlight
    _box(ax, BOX_X, y, BOX_W, BOX_H,
         fc=BLUE_TINT if hl else PAPER,
         ec=NAVY if hl else GRAY_MD,
         lw=2.2 if hl else 1.0, r=0.8, z=2)
    _box(ax, BOX_X, y, STRIP_W, BOX_H,
         fc=NAVY if hl else GRAY_DK,
         ec="none", lw=0, r=0.8, z=3)
    ax.text(BOX_X + STRIP_W/2, y + BOX_H/2, str(s.number),
            ha="center", va="center",
            color=WHITE, fontsize=30, fontweight="bold", zorder=4)

    tx = TEXT_X
    ax.text(tx, y + TITLE_Y, s.title,
            ha="left", va="top", color=NAVY,
            fontsize=10.0, fontweight="bold", zorder=4, clip_on=False)
    ax.text(tx, y + SUB_Y, s.subtitle,
            ha="left", va="top", color=GRAY_MD,
            fontsize=8.0, style="italic", zorder=4, clip_on=False)
    ax.plot([tx, BOX_X + BOX_W - 2], [y + SEP_Y, y + SEP_Y],
            color=GRAY_LT, lw=0.8, zorder=3)

    lc = NAVY_SOFT if hl else GRAY_DK
    for (lbl, val), ry in zip(s.rows, ROW_YS):
        ax.text(tx, y + ry, f"{lbl}:",
                ha="left", va="top", color=lc,
                fontsize=8.5, fontweight="bold", zorder=4)
        ax.text(VALUE_X, y + ry, val,
                ha="left", va="top", color=GRAY_DK,
                fontsize=8.5, zorder=4, clip_on=False)


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # cabeçalho
    ax.text(50, 98.0, "Pipeline experimental — 3 etapas",
            ha="center", va="center",
            color=NAVY, fontsize=13, fontweight="bold")
    ax.text(50, 94.8,
            "Meta 1: extrator MST + classificador condicionado   ·   "
            "Meta 2: transferência para verificação facial",
            ha="center", va="center", color=GRAY_MD, fontsize=8.0)

    for s, y in zip(STAGES, BOX_Y):
        _draw_stage(ax, s, y)

    # setas
    cx = BOX_X + BOX_W / 2
    for i in range(len(STAGES) - 1):
        y_from = BOX_Y[i]                  # fundo da caixa i
        y_to   = BOX_Y[i + 1] + BOX_H     # topo da caixa i+1
        mid_y  = (y_from + y_to) / 2
        label  = ARROWS.get((i, i + 1), "")

        ax.add_patch(FancyArrowPatch(
            (cx, y_from - 0.4), (cx, y_to + 0.4),
            arrowstyle="-|>", color=NAVY, lw=1.8,
            mutation_scale=13, shrinkA=1, shrinkB=1, zorder=5,
        ))
        # label centrado na seta
        ax.text(cx, mid_y, label,
                ha="center", va="center",
                color=NAVY, fontsize=8.5, style="italic",
                bbox=dict(fc="white", ec=GRAY_LT, lw=0.8,
                          pad=2.5, boxstyle="round"),
                zorder=6)

    # barras META (i_start, i_end = índices em BOX_Y)
    BAR_X, BAR_W = 3.0, 5.5
    for label, (i0, i1), color in METAS:
        y_bot = BOX_Y[i1]
        y_top = BOX_Y[i0] + BOX_H
        _box(ax, BAR_X, y_bot, BAR_W, y_top - y_bot,
             fc=color, ec=color, lw=0, r=0.7, z=3)
        ax.text(BAR_X + BAR_W/2, (y_top + y_bot)/2, label,
                rotation=90, ha="center", va="center",
                color=WHITE, fontsize=8.5, fontweight="bold", zorder=4)

    plt.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = repo_root / "docs" / "tese" / "images" / "fig_pipeline_6etapas.png"
    build_figure(out)


if __name__ == "__main__":
    main()
