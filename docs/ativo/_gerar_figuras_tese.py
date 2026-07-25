"""Gera as 4 figuras da qualificacao como PNG de alta qualidade.

Substituem os blocos TikZ que davam erro no Overleaf (conflito
provavel com o pacote xy carregado pela classe icmc.cls).

Saidas em docs/tese/images/:
  fig_disparidade_racial.png  (Cap 1, Introducao)
  fig_tripe_latinx.png        (Cap 2, Revisao Bibliografica)
  fig_pipeline_6etapas.png    (Cap 4, Metodologia)
  fig_configs_abcd.png        (Cap 4, Metodologia)

Uso:
    python _gerar_figuras_tese.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Diretorio de saida
OUT_DIR = Path(__file__).parent.parent / "tese" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta consistente com o padrao academico sobrio
NAVY = "#1F2A4E"
BLUE_LIGHT = "#B4C4E8"
BLUE_MID = "#3D68BB"
GRAY_DK = "#3D424E"
GRAY_MD = "#707682"
GRAY_LT = "#E8EAED"
RED = "#C0392B"
GREEN = "#2E7D32"
WHITE = "#FFFFFF"

# Configuracao geral
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": GRAY_DK,
})

DPI = 200


# ============================================================
# FIGURA 1 - Disparidade racial (Cap 1)
# ============================================================
def gerar_fig_disparidade():
    """Barras horizontais com F1 por classe racial no SOTA."""
    fig, ax = plt.subplots(figsize=(9, 4), dpi=DPI)

    classes = ["Latinx / Hispanic", "Southeast Asian", "White", "Black"]
    f1 = [60, 67, 80, 90]
    cores = [RED, BLUE_LIGHT, BLUE_MID, GREEN]

    # Barras horizontais
    bars = ax.barh(classes, f1, color=cores, edgecolor=GRAY_DK, linewidth=0.8, height=0.6)

    # Rotulos nas barras
    for bar, valor in zip(bars, f1):
        ax.text(valor + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{valor} %", va="center", ha="left",
                fontsize=10, fontweight="bold", color=GRAY_DK)

    # Anotacao do gap
    y_pior = bars[0].get_y() + bars[0].get_height() / 2
    y_melhor = bars[3].get_y() + bars[3].get_height() / 2
    x_gap = 96

    ax.annotate("", xy=(x_gap, y_pior), xytext=(x_gap, y_melhor),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(x_gap + 1, (y_pior + y_melhor) / 2,
            "gap\n≈ 30 pp", va="center", ha="left",
            fontsize=9, fontweight="bold", color=RED)

    # Linhas de referencia verticais
    ax.axvline(60, color=RED, linestyle=":", alpha=0.4, lw=0.8)
    ax.axvline(90, color=GREEN, linestyle=":", alpha=0.4, lw=0.8)

    # Formatacao dos eixos
    ax.set_xlabel("F1 (%)", fontsize=10, color=GRAY_DK)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", colors=GRAY_DK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Destacar rotulo Latinx (pior)
    labels = ax.get_yticklabels()
    labels[0].set_fontweight("bold")
    labels[0].set_color(RED)

    plt.tight_layout()
    out = OUT_DIR / "fig_disparidade_racial.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 2 - Tripe empirico Latinx (Cap 2)
# ============================================================
def gerar_fig_tripe():
    """Trees disciplinas convergindo em uma conclusao."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Tres caixas superiores (disciplinas)
    disciplinas = [
        {
            "x": 1.2, "titulo": "Antropologia biológica",
            "linhas": ["Telles (2014) — PERLA",
                       "4 países LatAm",
                       "pigmentocracia"],
        },
        {
            "x": 5.0, "titulo": "Genética populacional",
            "linhas": ["Bryc et al. (2015) — AJHG",
                       "162.721 indivíduos",
                       "ancestralidade variável"],
        },
        {
            "x": 8.8, "titulo": "Sociologia identitária",
            "linhas": ["Lopez et al. (2017) — Pew",
                       "4 gerações EUA",
                       "97 % → 50 %"],
        },
    ]

    for d in disciplinas:
        box = FancyBboxPatch(
            (d["x"], 3.5), 2.7, 1.7,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor=GRAY_DK, facecolor=BLUE_LIGHT,
        )
        ax.add_patch(box)
        # Titulo (bold)
        ax.text(d["x"] + 1.35, 4.95, d["titulo"],
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=NAVY)
        # Corpo
        for i, linha in enumerate(d["linhas"]):
            ax.text(d["x"] + 1.35, 4.55 - i * 0.3, linha,
                    ha="center", va="center", fontsize=9, color=GRAY_DK)

    # Elipse inferior (conclusao)
    from matplotlib.patches import Ellipse
    elipse = Ellipse((6.0, 0.9), 8.5, 1.8,
                     linewidth=1.8, edgecolor=NAVY, facecolor=BLUE_MID, alpha=0.85)
    ax.add_patch(elipse)
    ax.text(6.0, 1.15, "Heterogeneidade fenotípica intra-Latinx",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.text(6.0, 0.65, "spread MST amplo (≥ 5 das 10 classes)",
            ha="center", va="center", fontsize=9, color="white", style="italic")

    # Setas convergentes
    for d in disciplinas:
        arrow = FancyArrowPatch(
            (d["x"] + 1.35, 3.45), (6.0, 1.9),
            arrowstyle="-|>", mutation_scale=15,
            color=GRAY_DK, linewidth=1.2, alpha=0.8,
        )
        ax.add_patch(arrow)

    plt.tight_layout()
    out = OUT_DIR / "fig_tripe_latinx.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 3 - Pipeline em 6 etapas (Cap 4)
# ============================================================
def gerar_fig_pipeline():
    """Pipeline 3x2 com destaque das etapas 3 e 6."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    etapas = [
        # (col, row, titulo, sub, destaque)
        (0.5, 3.5, "Etapa 1", "Classificador MST\n(SkinToneNet)", False),
        (5.0, 3.5, "Etapa 2", "Auditoria FairFace\nMST × raça", False),
        (9.5, 3.5, "Etapa 3", "Classificador racial\nConvNeXt-T + FiLM", True),
        (0.5, 0.5, "Etapa 4", "Comparação\nvs 6 baselines", False),
        (5.0, 0.5, "Etapa 5", "Transferência fair\nRFW / BFW", False),
        (9.5, 0.5, "Etapa 6", "Síntese decompositiva\nfenotípico × algorítmico", True),
    ]

    dx, dy = 3.5, 2.0  # largura e altura das caixas

    for x, y, titulo, sub, destaque in etapas:
        face = BLUE_MID if destaque else BLUE_LIGHT
        text_color = "white" if destaque else NAVY
        sub_color = "white" if destaque else GRAY_DK
        lw = 1.8 if destaque else 1.2

        box = FancyBboxPatch(
            (x, y), dx, dy,
            boxstyle="round,pad=0.05",
            linewidth=lw, edgecolor=NAVY if destaque else GRAY_DK, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x + dx / 2, y + dy - 0.4, titulo,
                ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
        ax.text(x + dx / 2, y + dy / 2 - 0.15, sub,
                ha="center", va="center", fontsize=9, color=sub_color)

    # Setas horizontais linha superior
    for x_start, x_end in [(4.0, 5.0), (8.5, 9.5)]:
        arrow = FancyArrowPatch((x_start, 4.5), (x_end, 4.5),
                                arrowstyle="-|>", mutation_scale=18,
                                color=GRAY_DK, linewidth=1.2)
        ax.add_patch(arrow)

    # Setas horizontais linha inferior
    for x_start, x_end in [(4.0, 5.0), (8.5, 9.5)]:
        arrow = FancyArrowPatch((x_start, 1.5), (x_end, 1.5),
                                arrowstyle="-|>", mutation_scale=18,
                                color=GRAY_DK, linewidth=1.2)
        ax.add_patch(arrow)

    # Seta vertical (Etapa 3 -> Etapa 4)
    arrow = FancyArrowPatch((11.25, 3.5), (2.25, 2.5),
                            arrowstyle="-|>", mutation_scale=18,
                            connectionstyle="arc3,rad=-0.3",
                            color=GRAY_DK, linewidth=1.2)
    ax.add_patch(arrow)

    plt.tight_layout()
    out = OUT_DIR / "fig_pipeline_6etapas.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 4 - 4 configuracoes A/B/C/D (Cap 4)
# ============================================================
def gerar_fig_configs():
    """4 blocos comparativos com destaque da Config B."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=DPI)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    configs = [
        {"x": 0.3, "id": "A", "titulo": "Baseline", "corpo": "ConvNeXt-T",
         "sinal": "sem conditioning", "destaque": False},
        {"x": 3.5, "id": "B", "titulo": "FiLM linear", "corpo": "ConvNeXt-T + FiLM",
         "sinal": "MST 10-dim", "destaque": True},
        {"x": 6.7, "id": "C", "titulo": "Gated FiLM", "corpo": "ConvNeXt-T + Gated FiLM",
         "sinal": "MST 10-dim", "destaque": False},
        {"x": 9.9, "id": "D", "titulo": "FiLM CLIP", "corpo": "ConvNeXt-T + FiLM",
         "sinal": "CLIP-text 512-dim", "destaque": False},
    ]

    dx, dy = 2.9, 2.5

    for c in configs:
        face = BLUE_MID if c["destaque"] else BLUE_LIGHT
        text_color = "white" if c["destaque"] else NAVY
        sub_color = "white" if c["destaque"] else GRAY_DK
        lw = 2.0 if c["destaque"] else 1.2

        # Cabecalho (ID + titulo)
        ax.text(c["x"] + dx / 2, dy + 0.4,
                f"{c['id']} — {c['titulo']}",
                ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)

        # Caixa principal
        box = FancyBboxPatch(
            (c["x"], 1.2), dx, dy,
            boxstyle="round,pad=0.05",
            linewidth=lw, edgecolor=NAVY if c["destaque"] else GRAY_DK, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(c["x"] + dx / 2, 1.2 + dy / 2, c["corpo"],
                ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

        # Sinal condicionante (label italico)
        ax.text(c["x"] + dx / 2, 0.85, c["sinal"],
                ha="center", va="center", fontsize=9, style="italic", color=sub_color)

    # Rodape com constantes do experimento
    ax.text(6.5, 0.25,
            "Dataset: FairFace  |  Protocolo: 3 sementes  |  Métricas: DR + F1 macro + worst-class F1 + EO",
            ha="center", va="center", fontsize=8, style="italic", color=GRAY_MD)

    plt.tight_layout()
    out = OUT_DIR / "fig_configs_abcd.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


def main():
    print("Gerando figuras da qualificacao...")
    gerar_fig_disparidade()
    gerar_fig_tripe()
    gerar_fig_pipeline()
    gerar_fig_configs()
    print("\nTodas as 4 figuras geradas em docs/tese/images/")


if __name__ == "__main__":
    main()
