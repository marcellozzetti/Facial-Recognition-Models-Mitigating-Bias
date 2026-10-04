"""Gera diagrama SIMPLIFICADO do mecanismo FiLM aplicado ao nosso pipeline.

Decisao pos-reuniao (orientador, Out/2026): adotar a instanciacao mais
simples possivel do FiLM — UMA UNICA camada FiLM apos o backbone,
imediatamente antes do classificador. Essa escolha:

  (a) minimiza overhead parametrico (~14 k parametros, ~0.05 % do
      backbone, contra ~380 k / ~1.3 % da variante de 4 insercoes);
  (b) oferece o mental model mais limpo para a defesa — 'o backbone
      extrai features genericas; o FiLM modula-as antes da classificacao';
  (c) e defensavel na banca como a 'minima instanciacao suficiente'
      do mecanismo.

Topologia:

    Imagem --> MST classifier (congelado) --> z in R^10 --> MLPs f_g, f_b --> (gamma, beta)
                                                                                    |
                                                                                    v
    Imagem --> [ConvNeXt-T backbone (28M params, 768 canais)] --> FiLM --> Classif. 7 racas

Saida: docs/ativo/imagens/film_pipeline.png (alta resolucao 300 DPI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Paleta consistente com o restante do repo
NAVY = (31 / 255, 42 / 255, 78 / 255)
GRAY_DK = (61 / 255, 66 / 255, 78 / 255)
GRAY_MD = (112 / 255, 118 / 255, 130 / 255)
GRAY_LT = (232 / 255, 234 / 255, 237 / 255)
ACCENT = (192 / 255, 57 / 255, 43 / 255)
GREEN = (46 / 255, 125 / 255, 50 / 255)
WHITE = (1.0, 1.0, 1.0)


def _box(ax, x, y, w, h, text, fc=GRAY_LT, ec=NAVY, fs=12, fw="normal",
         tc=GRAY_DK, lw=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, wrap=True,
            linespacing=1.2)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.0, style="-|>", ms=18):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=ms, shrinkA=2, shrinkB=2,
    ))


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis("off")

    # ---------- Titulo ----------
    ax.text(50, 51, "FiLM — Feature-wise Linear Modulation",
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)
    ax.text(50, 47.5,
            "Instanciação mínima: uma camada FiLM após o backbone, "
            "antes do classificador",
            ha="center", va="center",
            fontsize=14, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA SUPERIOR — geracao do sinal condicionante z e MLPs
    # =========================================================
    y_top = 32
    box_h = 7.5

    # Imagem (superior)
    _box(ax, 2, y_top, 11, box_h,
         "Imagem\n224 × 224 × 3",
         fc=GRAY_LT, ec=NAVY, fs=13, fw="bold")
    _arrow(ax, 13, y_top + box_h / 2, 18, y_top + box_h / 2,
           color=GREEN, lw=2.2)

    # Classificador MST (verde — congelado)
    _box(ax, 18, y_top, 18, box_h,
         "Classificador MST\ncongelado",
         fc=GREEN, ec=GREEN, fs=14, fw="bold", tc=WHITE)
    ax.text(27, y_top + box_h + 1.3,
            "treinado na Etapa 1  (MSTE + CCv2)",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")
    _arrow(ax, 36, y_top + box_h / 2, 41, y_top + box_h / 2,
           color=NAVY, lw=2.2)

    # Vetor z (10-dim)
    _box(ax, 41, y_top, 18, box_h, "",
         fc=WHITE, ec=NAVY, fs=14, fw="bold", lw=1.5)
    ax.text(50, y_top + box_h / 2 + 1.4,
            "Vetor  z  ∈  ℝ¹⁰",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=NAVY)
    ax.text(50, y_top + box_h / 2 - 1.6,
            "softmax sobre os 10 tons Monk",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")
    _arrow(ax, 59, y_top + box_h / 2, 64, y_top + box_h / 2,
           color=NAVY, lw=2.2)

    # MLPs f_g, f_b
    _box(ax, 64, y_top, 24, box_h,
         "MLPs   f_γ  ,   f_β\num único par",
         fc=WHITE, ec=ACCENT, fs=14, fw="bold", tc=ACCENT, lw=2.0)
    ax.text(76, y_top + box_h + 1.3,
            "≈ 14 k parâmetros  (≈ 0,05 % do backbone)",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA INFERIOR — backbone + 1 FiLM + classificador
    # =========================================================
    y_bb = 10
    bb_h = 8.5

    # Imagem (duplicada visualmente na esquerda inferior para marcar
    # que a mesma imagem entra no backbone)
    _box(ax, 2, y_bb, 11, bb_h,
         "Imagem\n224 × 224 × 3",
         fc=GRAY_LT, ec=NAVY, fs=13, fw="bold")
    _arrow(ax, 13, y_bb + bb_h / 2, 18, y_bb + bb_h / 2,
           color=NAVY, lw=2.4)

    # ConvNeXt-T backbone (bloco unico)
    _box(ax, 18, y_bb, 36, bb_h,
         "ConvNeXt-T backbone\n(≈ 28 M parâmetros · 768 canais na saída)",
         fc=GRAY_LT, ec=NAVY, fs=14, fw="bold", lw=1.8)
    ax.text(36, y_bb + bb_h + 1.3,
            "não modificado — recebe apenas a imagem",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")
    _arrow(ax, 54, y_bb + bb_h / 2, 59, y_bb + bb_h / 2,
           color=NAVY, lw=2.4)

    # FiLM (unico — ponto de insercao)
    film_x = 59
    film_w = 14
    _box(ax, film_x, y_bb, film_w, bb_h,
         "FiLM\n(única inserção)",
         fc=NAVY, ec=NAVY, fs=16, fw="bold", tc=WHITE, lw=2.4)
    ax.text(film_x + film_w / 2, y_bb + bb_h + 1.3,
            "modula as 768 features finais via (γ, β)",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")
    _arrow(ax, film_x + film_w, y_bb + bb_h / 2,
           film_x + film_w + 5, y_bb + bb_h / 2, color=NAVY, lw=2.4)

    # Classificador 7 racas
    class_x = film_x + film_w + 5
    class_w = 13
    _box(ax, class_x, y_bb, class_w, bb_h,
         "Classificador\n7 raças",
         fc=ACCENT, ec=ACCENT, fs=14, fw="bold", tc=WHITE, lw=2.0)
    class_center = class_x + class_w / 2

    # Seta para fora do bloco (predicao)
    _arrow(ax, class_center, y_bb, class_center, y_bb - 4, color=ACCENT)
    ax.text(class_center, y_bb - 5.5,
            "Predição de raça",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=GRAY_DK)

    # =========================================================
    # Seta vertical conectando MLPs -> FiLM (sinal γ, β)
    # =========================================================
    mlp_bottom_x = 76   # centro do bloco MLPs
    mlp_bottom_y = y_top
    film_top_x = film_x + film_w / 2
    film_top_y = y_bb + bb_h

    # Linha em L: desce da MLP ate uma altura intermediaria, depois
    # cruza horizontalmente ate o FiLM.
    mid_y = (mlp_bottom_y + film_top_y) / 2  # ~24.75

    # Descida da MLP ate a altura intermediaria
    ax.plot([mlp_bottom_x, mlp_bottom_x], [mlp_bottom_y, mid_y],
            color=ACCENT, lw=2.2, zorder=4)
    # Trajeto horizontal da MLP ate acima do FiLM
    ax.plot([mlp_bottom_x, film_top_x], [mid_y, mid_y],
            color=ACCENT, lw=2.2, zorder=4)
    # Descida final ate o topo do FiLM, com arrowhead
    _arrow(ax, film_top_x, mid_y, film_top_x, film_top_y + 0.3,
           color=ACCENT, lw=2.2, ms=18)

    # Rotulo (gamma, beta)
    ax.text(film_top_x + 1.5, mid_y + 1.2,
            "(γ, β)",
            ha="left", va="center",
            fontsize=15, fontweight="bold",
            color=ACCENT, style="italic")

    # =========================================================
    # RODAPE — equacao canonica + legenda
    # =========================================================
    # Caixa da equacao (esquerda)
    _box(ax, 4, 0.5, 44, 5, "",
         fc=WHITE, ec=NAVY, lw=1.8)
    ax.text(26, 3.9,
            "Equação canônica  (Perez et al., 2018)",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=NAVY)
    ax.text(26, 1.7,
            "F′  =  γ  ⊙  F  +  β",
            ha="center", va="center",
            fontsize=19, fontweight="bold", color=ACCENT, style="italic")

    # Legenda de cores (direita)
    legend_items = [
        (GREEN,  "Congelado  (classificador MST, Etapa 1)"),
        (NAVY,   "Treinado end-to-end  (backbone + FiLM)"),
        (ACCENT, "Camadas novas  (~0,05 % dos params)"),
    ]
    legend_x = 55
    for i, (color, text) in enumerate(legend_items):
        y_leg = 4.3 - i * 1.5
        ax.add_patch(mpatches.Rectangle(
            (legend_x, y_leg), 2.5, 1.0, fc=color, ec=color))
        ax.text(legend_x + 3.5, y_leg + 0.5, text,
                ha="left", va="center",
                fontsize=11, color=GRAY_DK)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    out = Path(__file__).resolve().parent / "imagens" / "film_pipeline.png"
    build_figure(out)
    dest = (Path(__file__).resolve().parents[2]
            / "docs" / "tese" / "images" / "film_pipeline.png")
    if dest.parent.exists():
        import shutil
        shutil.copy2(out, dest)
        print(f"Copiado para: {dest}")


if __name__ == "__main__":
    main()
