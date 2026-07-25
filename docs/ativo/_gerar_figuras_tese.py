"""Gera as figuras da qualificacao como PNG de alta qualidade.

Substituem os blocos TikZ que davam erro no Overleaf (conflito
provavel com o pacote xy carregado pela classe icmc.cls).

Saidas em docs/tese/images/:
  fig_disparidade_racial.png     (Cap 1, Introducao)
  fig_tripe_latinx.png           (Cap 2, Revisao Bibliografica)
  fig_pipeline_6etapas.png       (Cap 4, Metodologia)
  fig_configs_abcd.png           (Cap 4, Metodologia)
  fig_fitzpatrick_vs_mst.png     (Cap 2, sec 2.4)
  fig_timeline_mitigacoes.png    (Cap 2, sec 2.2)
  fig_f1_sota_comparativo.png    (Cap 2, sec 2.3)
  fig_film_conceitual.png        (Cap 2, sec 2.7)

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


# ============================================================
# FIGURA 5 - Fitzpatrick vs MST (Cap 2, sec 2.4)
# ============================================================
# Cores MST oficiais publicadas pelo Google em skintone.google
# (Monk Skin Tone Scale, desenvolvida por Ellis Monk 2019 e formalizada
#  por Schumann et al. NeurIPS 2023).
MST_HEX = ["#F6EDE4", "#F3E7DB", "#F7EAD0", "#EADABA", "#D7BD96",
           "#A07E56", "#825C43", "#604134", "#3A312A", "#292420"]

# Cores Fitzpatrick aproximadas de referencias dermatologicas
# (Fitzpatrick 1988; disponivel em multiplas fontes publicas)
FITZ_HEX = ["#F6E4D8", "#EDD3B6", "#D8B48A", "#B98E63", "#855832", "#322422"]


def gerar_fig_fitzpatrick_vs_mst():
    """Compara escalas Fitzpatrick (6, assimetrica) e MST (10, simetrica)."""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=DPI)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ---- Fitzpatrick (linha superior) ----
    ax.text(0.3, 5.2, "Escala Fitzpatrick (1988)",
            fontsize=11, fontweight="bold", color=NAVY, ha="left")
    ax.text(0.3, 4.75, "6 fototipos - dermatologia (fototerapia UV)",
            fontsize=9, style="italic", color=GRAY_MD, ha="left")

    fitz_w = 2.5
    fitz_x0 = 4.0
    fitz_y = 3.5
    fitz_h = 1.0
    for i, hexcolor in enumerate(FITZ_HEX):
        x = fitz_x0 + i * fitz_w
        rect = Rectangle((x, fitz_y), fitz_w, fitz_h,
                         facecolor=hexcolor, edgecolor=GRAY_DK, linewidth=0.6)
        ax.add_patch(rect)
        # Rotulo I-VI
        roman = ["I", "II", "III", "IV", "V", "VI"][i]
        text_color = "white" if i >= 4 else GRAY_DK
        ax.text(x + fitz_w / 2, fitz_y + fitz_h / 2, roman,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=text_color)

    # Colchetes de assimetria
    # 4 claros (I-IV)
    ax.annotate("", xy=(fitz_x0, 3.2), xytext=(fitz_x0 + 4 * fitz_w, 3.2),
                arrowprops=dict(arrowstyle="-", color=BLUE_MID, lw=1.5))
    ax.text(fitz_x0 + 2 * fitz_w, 2.85, "4 fototipos claros",
            ha="center", va="center", fontsize=9, fontweight="bold", color=BLUE_MID)
    # 2 escuros (V-VI)
    ax.annotate("", xy=(fitz_x0 + 4 * fitz_w, 3.2),
                xytext=(fitz_x0 + 6 * fitz_w, 3.2),
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.5))
    ax.text(fitz_x0 + 5 * fitz_w, 2.85, "apenas 2 escuros",
            ha="center", va="center", fontsize=9, fontweight="bold", color=RED)

    # ---- MST (linha inferior) ----
    ax.text(0.3, 2.15, "Monk Skin Tone Scale (2023)",
            fontsize=11, fontweight="bold", color=NAVY, ha="left")
    ax.text(0.3, 1.7, "10 tons - auditoria de fairness em IA",
            fontsize=9, style="italic", color=GRAY_MD, ha="left")

    mst_w = 1.5
    mst_x0 = 4.0
    mst_y = 0.5
    mst_h = 1.0
    for i, hexcolor in enumerate(MST_HEX):
        x = mst_x0 + i * mst_w
        rect = Rectangle((x, mst_y), mst_w, mst_h,
                         facecolor=hexcolor, edgecolor=GRAY_DK, linewidth=0.6)
        ax.add_patch(rect)
        text_color = "white" if i >= 6 else GRAY_DK
        ax.text(x + mst_w / 2, mst_y + mst_h / 2, str(i + 1),
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=text_color)

    # Barra de distribuicao simetrica
    ax.annotate("", xy=(mst_x0, 0.2), xytext=(mst_x0 + 10 * mst_w, 0.2),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.5))
    ax.text(mst_x0 + 5 * mst_w, -0.05, "distribuicao perceptualmente simetrica",
            ha="center", va="center", fontsize=9, fontweight="bold", color=GREEN)

    plt.tight_layout()
    out = OUT_DIR / "fig_fitzpatrick_vs_mst.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 6 - Timeline evolutiva das mitigacoes (Cap 2, sec 2.2)
# ============================================================
def gerar_fig_timeline_mitigacoes():
    """Linha do tempo 2018-2025 das 6 tecnicas de mitigacao algoritmica."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    ax.set_xlim(2017.5, 2025.5)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    # Linha do tempo
    ax.axhline(0, color=GRAY_DK, linewidth=1.5, zorder=1)

    # Marcadores de ano (grid)
    for ano in range(2018, 2026):
        ax.plot(ano, 0, "|", color=GRAY_DK, markersize=12, mew=1.5, zorder=2)
        ax.text(ano, -0.5, str(ano), ha="center", va="top",
                fontsize=9, color=GRAY_DK, fontweight="bold")

    # Categorias de tecnica (por cor)
    CAT_ADVERSARIAL = RED
    CAT_OPTIMIZATION = BLUE_MID
    CAT_CONTRASTIVE = GREEN
    CAT_PRUNING = "#E67E22"
    CAT_ARCHITECTURE = "#8E44AD"

    # Publicacoes (ano, autor, tecnica, cor, y_offset)
    pubs = [
        (2018, "Zhang et al.", "Adversarial\ndebiasing", CAT_ADVERSARIAL, 1.5),
        (2020, "Sagawa et al.", "DRO\n(worst-case group)", CAT_OPTIMIZATION, 1.5),
        (2022, "Park et al.", "FSCL+\n(contrastivo)", CAT_CONTRASTIVE, 2.0),
        (2022, "Lin et al.", "FairGRAPE\n(pruning)", CAT_PRUNING, -2.0),
        (2024, "Manzoor et al.", "FineFACE\n(arquitetura)", CAT_ARCHITECTURE, 1.5),
        (2025, "Liu et al.", "Bayesian Meta\nReweighting", CAT_OPTIMIZATION, 1.5),
    ]

    for ano, autor, tecnica, cor, y in pubs:
        # Bolinha do marcador
        ax.plot(ano, 0, "o", color=cor, markersize=16, zorder=3,
                markeredgecolor="white", markeredgewidth=2)

        # Linha conectando marcador a caixa de texto
        y_box = y
        y_line_end = y_box - 0.35 if y > 0 else y_box + 0.35
        ax.plot([ano, ano], [0, y_line_end], color=cor, lw=1.2,
                linestyle="--", alpha=0.6, zorder=2)

        # Caixa de texto
        va = "bottom" if y > 0 else "top"
        ax.text(ano, y_box, f"{autor}\n{tecnica}", ha="center", va=va,
                fontsize=9, color=NAVY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=cor, linewidth=1.2))

    # Legenda de categorias
    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor=CAT_ADVERSARIAL, label="Adversarial"),
        Patch(facecolor=CAT_OPTIMIZATION, label="Otimizacao (DRO / meta)"),
        Patch(facecolor=CAT_CONTRASTIVE, label="Contrastivo"),
        Patch(facecolor=CAT_PRUNING, label="Pruning"),
        Patch(facecolor=CAT_ARCHITECTURE, label="Arquitetura"),
    ]
    ax.legend(handles=legenda, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "fig_timeline_mitigacoes.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 7 - F1 SOTA vs baseline por raca (Cap 2, sec 2.3)
# ============================================================
def gerar_fig_f1_sota_comparativo():
    """Compara ResNet-34 (Lin 2022) vs FaceScanPaliGemma (AlDahoul 2024) por raca."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 5), dpi=DPI)

    # Dados diretos:
    # - Lin 2022 (FairGRAPE), Tab. 2, ResNet-34 no-pruning baseline: accuracy per race
    # - AlDahoul 2024 (FaceScanPaliGemma), Tab. 16, F1 per race
    # Comparacao ilustrativa: ambos avaliados sobre FairFace race 7-class
    classes = ["White", "Black", "Latinx", "E-Asian", "SE-Asian", "Indian", "Mid-East"]
    resnet34 = [73.9, 83.2, 59.6, 77.6, 66.9, 75.4, 66.2]  # Lin 2022 Tab.2
    paligemma = [80, 90, 60, 74, 67, 78, 72]  # AlDahoul 2024 Tab.16 (aproximado)

    x = np.arange(len(classes))
    largura = 0.38

    barras1 = ax.bar(x - largura / 2, resnet34, largura,
                     label="ResNet-34 baseline\n(Lin et al. 2022, agregado 72%)",
                     color=BLUE_LIGHT, edgecolor=GRAY_DK, linewidth=0.8)
    barras2 = ax.bar(x + largura / 2, paligemma, largura,
                     label="FaceScanPaliGemma\n(AlDahoul et al. 2024, agregado 75.7%)",
                     color=BLUE_MID, edgecolor=GRAY_DK, linewidth=0.8)

    # Valores em cima de cada barra
    for bars in (barras1, barras2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8, color=GRAY_DK)

    # Destaque visual da coluna Latinx
    ax.axvspan(2 - 0.5, 2 + 0.5, color=RED, alpha=0.08)
    ax.text(2, 100, "gap persistente\nem Latinx (~60 %)",
            ha="center", va="top", fontsize=9, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=RED, linewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Desempenho por raca (%)", fontsize=10, color=GRAY_DK)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", colors=GRAY_DK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    out = OUT_DIR / "fig_f1_sota_comparativo.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


# ============================================================
# FIGURA 8 - Diagrama conceitual do FiLM (Cap 2, sec 2.7)
# ============================================================
def gerar_fig_film_conceitual():
    """Diagrama adaptado de Perez et al. (2018), Fig. 2 do paper original."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 1. Bloco de entrada: Feature map x (do backbone)
    box_x = FancyBboxPatch((0.2, 2.2), 2.4, 1.6,
                           boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=GRAY_DK, facecolor=GRAY_LT)
    ax.add_patch(box_x)
    ax.text(1.4, 3.3, "Feature map", ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(1.4, 2.8, r"$\mathbf{x} \in \mathbb{R}^{C \times H \times W}$",
            ha="center", va="center", fontsize=11, color=NAVY)
    ax.text(1.4, 2.35, "(do ConvNeXt-T)", ha="center", va="center",
            fontsize=8, style="italic", color=GRAY_MD)

    # 2. Bloco FiLM (centro)
    box_film = FancyBboxPatch((4.5, 1.8), 3.6, 2.4,
                              boxstyle="round,pad=0.05",
                              linewidth=1.8, edgecolor=NAVY, facecolor=BLUE_MID)
    ax.add_patch(box_film)
    ax.text(6.3, 3.65, "Camada FiLM", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(6.3, 3.15, r"$\mathrm{FiLM}(\mathbf{x} | \gamma, \beta)$",
            ha="center", va="center", fontsize=12, color="white")
    ax.text(6.3, 2.7, r"$= \gamma \odot \mathbf{x} + \beta$",
            ha="center", va="center", fontsize=13, color="white")
    ax.text(6.3, 2.15, "modulacao afim canal a canal",
            ha="center", va="center", fontsize=8, style="italic", color="white")

    # 3. Sinal condicionante MST (topo)
    box_mst = FancyBboxPatch((4.5, 4.7), 3.6, 1.0,
                             boxstyle="round,pad=0.05",
                             linewidth=1.2, edgecolor=GRAY_DK, facecolor="#FFE8C7")
    ax.add_patch(box_mst)
    ax.text(6.3, 5.3, "Sinal condicionante", ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(6.3, 4.9, r"$\mathbf{z}_{\mathrm{MST}} \in \mathbb{R}^{10}$",
            ha="center", va="center", fontsize=10, color=NAVY)

    # Seta MST -> FiLM (vertical)
    arrow_mst = FancyArrowPatch((6.3, 4.65), (6.3, 4.25),
                                arrowstyle="-|>", mutation_scale=15,
                                color=GRAY_DK, linewidth=1.5)
    ax.add_patch(arrow_mst)
    ax.text(6.8, 4.45, r"$\gamma, \beta = f(\mathbf{z})$",
            ha="left", va="center", fontsize=9, style="italic", color=GRAY_MD)

    # 4. Bloco de saida: Feature map modulado
    box_y = FancyBboxPatch((10.2, 2.2), 3.4, 1.6,
                           boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=GRAY_DK, facecolor=GRAY_LT)
    ax.add_patch(box_y)
    ax.text(11.9, 3.3, "Feature map modulado", ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(11.9, 2.8, r"$\mathbf{y} \in \mathbb{R}^{C \times H \times W}$",
            ha="center", va="center", fontsize=11, color=NAVY)
    ax.text(11.9, 2.35, "(para prox. camada)", ha="center", va="center",
            fontsize=8, style="italic", color=GRAY_MD)

    # Seta x -> FiLM
    arrow_in = FancyArrowPatch((2.7, 3.0), (4.4, 3.0),
                               arrowstyle="-|>", mutation_scale=18,
                               color=GRAY_DK, linewidth=1.5)
    ax.add_patch(arrow_in)

    # Seta FiLM -> y
    arrow_out = FancyArrowPatch((8.2, 3.0), (10.1, 3.0),
                                arrowstyle="-|>", mutation_scale=18,
                                color=GRAY_DK, linewidth=1.5)
    ax.add_patch(arrow_out)

    # Anotacao inferior
    ax.text(7.0, 0.7, "Cada estagio do ConvNeXt-T recebe uma camada FiLM (4 no total), "
                     "com overhead paramtrico < 1% sobre o backbone.",
            ha="center", va="center", fontsize=9, style="italic", color=GRAY_MD,
            wrap=True)

    plt.tight_layout()
    out = OUT_DIR / "fig_film_conceitual.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"OK: {out}")


def main():
    print("Gerando figuras da qualificacao...")
    gerar_fig_disparidade()
    gerar_fig_tripe()
    gerar_fig_pipeline()
    gerar_fig_configs()
    gerar_fig_fitzpatrick_vs_mst()
    gerar_fig_timeline_mitigacoes()
    gerar_fig_f1_sota_comparativo()
    gerar_fig_film_conceitual()
    print("\nTodas as 8 figuras geradas em docs/tese/images/")


if __name__ == "__main__":
    main()
