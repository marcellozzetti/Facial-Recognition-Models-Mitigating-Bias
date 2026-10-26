"""Gera figura do pipeline experimental em 3 etapas (Cap. 4 §4.2).

Layout corrigido: math explícita para posicionamento dentro das caixas,
sem overflow. Texto left-aligned, fontes legíveis.

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
BLUE_TINT = (230/255, 236/255, 248/255)
GRAY_DK   = (61/255,  66/255,  78/255)
GRAY_MD   = (112/255, 118/255, 130/255)
GRAY_LT   = (215/255, 220/255, 228/255)
PAPER     = (252/255, 252/255, 250/255)
WHITE     = (1.0, 1.0, 1.0)

# ── coordenadas: 100 × 100 mapeado sobre FIG_W × FIG_H polegadas ─────────
FIG_W, FIG_H = 16, 18        # polegadas
UNIT_W = FIG_W / 100         # 0.16 polegadas por unidade horizontal
UNIT_H = FIG_H / 100         # 0.18 polegadas por unidade vertical

# ── geometria das caixas ─────────────────────────────────────────────────
BOX_W   = 82.0
BOX_H   = 25.0   # calculado: 3×25 + 2×5 (gap) = 85, sobram 6 unidades
GAP     = 5.0
BOX_X   = 11.0
TOP_Y   = 91.0   # topo do box 1

# posições das 3 caixas (y = canto inferior esquerdo)
#   box1_top = TOP_Y → box1_bottom = TOP_Y - BOX_H = 66
#   box2_top = 66 - GAP = 61  → box2_bottom = 36
#   box3_top = 36 - GAP = 31  → box3_bottom = 6

# ── layout INTERNO de cada caixa (relativo ao canto inferior y) ──────────
# Reserva simétrica: INNER_TOP = 4, INNER_BOT = 1.5 (va="top" → ok)
TITLE_Y   = BOX_H - 4.0      # 21.0  topo do título
SUB_Y     = BOX_H - 9.5      # 15.5  topo do subtítulo
SEP_Y     = BOX_H - 13.0     # 12.0  y da linha separadora
ROW0_Y    = BOX_H - 16.5     #  8.5  topo da 1ª linha de dados
ROW1_Y    = BOX_H - 20.0     #  5.0
ROW2_Y    = BOX_H - 23.5     #  1.5  (ainda acima do fundo: 1.5 unidades)

STRIP_W   = 9.0
TEXT_X    = BOX_X + STRIP_W + 4.0          # início do texto à direita da faixa
LABEL_W   = 13.0                            # largura da coluna do rótulo
VALUE_X   = TEXT_X + LABEL_W               # início do valor


@dataclass
class Stage:
    number: int
    title: str
    subtitle: str
    rows: List[tuple]    # (label, value)
    highlight: bool = False


STAGES = [
    Stage(
        1,
        "Extrator MST e diagnóstico fenotípico",
        "ViT-B/16 treinado em MSTE + CCv2 — saída: vetor softmax de 10 tons de pele",
        rows=[
            ("Insumo", "MSTE (~1 500 imgs)  ·  Casual Conversations v2 (~55 000 frames)  ·  FairFace val set"),
            ("Método", "Fine-tuning ViT-B/16  ·  Softmax 10 classes  ·  Validação humana cega e independente"),
            ("Saída",  "Extrator MST  z ∈ ℝ¹⁰  +  Matriz de distribuição MST × 7 raças (FairFace)"),
        ],
    ),
    Stage(
        2,
        "Classificador racial condicionado + ablation",
        "ConvNeXt-T + FiLM condicionado em z — ablation A / B / C e 3 baselines",
        rows=[
            ("Insumo", "FairFace train split  (86 744 imgs, 7 classes)  ·  vetor z da Etapa 1"),
            ("Método", "ConvNeXt-T + FiLM  ·  Ablation A/B/C  ·  3 baselines  ·  3 seeds  ·  bootstrap CI 95 %"),
            ("Saída",  "Classificador racial:  F1-macro  ·  Disparity Ratio  ·  Worst-class F1"),
        ],
        highlight=True,
    ),
    Stage(
        3,
        "Transferência para verificação facial",
        "Feature-freeze encoder + FiLM → nova cabeça treinada em RFW e BFW",
        rows=[
            ("Insumo", "Encoder + FiLM congelado (Etapa 2)  ·  RFW (~11 000 pares)  ·  BFW (~50 000 pares)"),
            ("Método", "Feature-freeze (LAFTR, Madras 2018)  ·  Nova cabeça de verificação  ·  Threshold único fixo"),
            ("Saída",  "TAR@FAR = 10⁻⁴  por grupo racial em RFW e BFW  +  formalização do mecanismo"),
        ],
    ),
]

METAS = [
    ("META 1", (1, 2), NAVY_SOFT),
    ("META 2", (3, 3), GRAY_DK),
]

ARROW_LABELS = {
    (1, 2): "z ∈ ℝ¹⁰",
    (2, 3): "encoder + FiLM  (congelado)",
}


def _box(ax, x, y, w, h, fc, ec, lw=1.2, rounding=0.6, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
    ))


def _draw_stage(ax, stage: Stage, x: float, y: float):
    w, h = BOX_W, BOX_H
    highlighted = stage.highlight
    body_fc = BLUE_TINT if highlighted else PAPER
    body_ec = NAVY       if highlighted else GRAY_MD
    body_lw = 2.8        if highlighted else 1.2

    # caixa externa
    _box(ax, x, y, w, h, fc=body_fc, ec=body_ec, lw=body_lw, rounding=1.0)

    # faixa do número
    fc_strip = NAVY if highlighted else GRAY_DK
    _box(ax, x, y, STRIP_W, h, fc=fc_strip, ec=fc_strip, lw=0, rounding=1.0, zorder=3)
    ax.text(x + STRIP_W / 2, y + h / 2, str(stage.number),
            ha="center", va="center",
            color=WHITE, fontsize=36, fontweight="bold", zorder=4)

    tx = TEXT_X    # x do texto
    mx = x + w - 2 # x máximo

    # título
    ax.text(tx, y + TITLE_Y, stage.title,
            ha="left", va="top",
            color=NAVY, fontsize=13, fontweight="bold", zorder=4)

    # subtítulo
    ax.text(tx, y + SUB_Y, stage.subtitle,
            ha="left", va="top",
            color=GRAY_MD, fontsize=9.5, style="italic", zorder=4)

    # separador
    ax.plot([tx, mx], [y + SEP_Y, y + SEP_Y], color=GRAY_LT, lw=0.9, zorder=4)

    # linhas de dados
    row_ys = [ROW0_Y, ROW1_Y, ROW2_Y]
    lbl_col = NAVY_SOFT if highlighted else GRAY_DK

    for (lbl, val), ry in zip(stage.rows, row_ys):
        ax.text(tx, y + ry, f"{lbl}:",
                ha="left", va="top",
                color=lbl_col, fontsize=9.5, fontweight="bold", zorder=4)
        ax.text(VALUE_X, y + ry, val,
                ha="left", va="top",
                color=GRAY_DK, fontsize=9.5, zorder=4)


def _arrow_between(ax, cx, y_top_prev, y_bot_next, label: str):
    """Seta do fundo de uma caixa para o topo da próxima."""
    y_from = y_top_prev        # fundo da caixa anterior
    y_to   = y_bot_next        # topo da caixa seguinte
    mid_y  = (y_from + y_to) / 2

    ax.add_patch(FancyArrowPatch(
        (cx, y_from - 0.3), (cx, y_to + 0.3),
        arrowstyle="-|>", color=NAVY, lw=2.2,
        mutation_scale=16, shrinkA=1, shrinkB=1, zorder=5,
    ))
    # rótulo do dado transferido — à direita da seta
    ax.text(cx + 3.5, mid_y, label,
            ha="left", va="center",
            color=NAVY, fontsize=10.5, style="italic", fontweight="medium",
            bbox=dict(fc=GRAY_LT, ec="none", pad=3.5, boxstyle="round"),
            zorder=6)


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # cabeçalho
    ax.text(50, 97.5, "Pipeline experimental — 3 etapas",
            ha="center", va="center",
            color=NAVY, fontsize=18, fontweight="bold")
    ax.text(50, 94.2,
            "Meta 1: extrator MST + classificador condicionado     ·     Meta 2: transferência para verificação facial",
            ha="center", va="center", color=GRAY_MD, fontsize=10)

    # caixas
    # box_y[i] = canto inferior esquerdo da caixa i (0-indexed)
    box_y = [TOP_Y - BOX_H]  # box 0 (Etapa 1): y = 66
    for _ in range(1, len(STAGES)):
        box_y.append(box_y[-1] - BOX_H - GAP)
    # box_y = [66, 36, 6]  verificado: 6 + 25 = 31 < 91 ✓

    for stage, y in zip(STAGES, box_y):
        _draw_stage(ax, stage, BOX_X, y)

    # setas entre caixas (do fundo de i para o topo de i+1)
    arrow_cx = BOX_X + BOX_W / 2
    for i in range(len(STAGES) - 1):
        y_from = box_y[i]                    # fundo da caixa i
        y_to   = box_y[i + 1] + BOX_H       # topo da caixa i+1
        label  = ARROW_LABELS.get((i + 1, i + 2), "")
        _arrow_between(ax, arrow_cx, y_from, y_to, label)

    # barra lateral META 1 / META 2
    BAR_X, BAR_W = 3.0, 5.5
    meta_spans = {
        1: (box_y[0], box_y[0] + BOX_H),    # Meta 1 — Etapa 1
        2: (box_y[1], box_y[1] + BOX_H),    # Meta 1 — Etapa 2
        3: (box_y[2], box_y[2] + BOX_H),    # Meta 2 — Etapa 3
    }
    for label, (start_stage, end_stage), color in METAS:
        y_bot = meta_spans[end_stage][0]    # fundo da última etapa da meta
        y_top = meta_spans[start_stage][1]  # topo da primeira etapa da meta
        _box(ax, BAR_X, y_bot, BAR_W, y_top - y_bot,
             fc=color, ec=color, lw=0, rounding=0.8)
        ax.text(BAR_X + BAR_W / 2, (y_top + y_bot) / 2, label,
                rotation=90, ha="center", va="center",
                color=WHITE, fontsize=11, fontweight="bold", zorder=4)

    plt.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = repo_root / "docs" / "tese" / "images" / "fig_pipeline_6etapas.png"
    build_figure(out)


if __name__ == "__main__":
    main()
