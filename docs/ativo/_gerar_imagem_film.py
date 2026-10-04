"""Gera diagrama do mecanismo FiLM aplicado ao final dos 4 estagios
do ConvNeXt-T.

Decisao (Out/2026): aplicar UMA UNICA camada FiLM ao final da pilha
de 4 estagios do ConvNeXt-T, imediatamente antes do classificador.
E a instanciacao mais simples possivel:

  (a) overhead parametrico minimo (~14 k, ~0,05 % do backbone);
  (b) mental model limpo para a defesa — o backbone extrai features
      hierarquicas; o FiLM modula a saida final antes da classificacao;
  (c) defensavel como a 'minima instanciacao suficiente' do mecanismo.

Topologia:

    Imagem -> MST classifier (congelado) -> z in R^10 -> MLPs f_g, f_b
                                                                    |
                                                                    v  (gamma, beta)
    Imagem -> [Est 1] -> [Est 2] -> [Est 3] -> [Est 4] -> FiLM -> Classif. 7 racas

Saida: docs/ativo/imagens/film_pipeline.png (300 DPI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = (31 / 255, 42 / 255, 78 / 255)
GRAY_DK = (61 / 255, 66 / 255, 78 / 255)
GRAY_MD = (112 / 255, 118 / 255, 130 / 255)
GRAY_LT = (232 / 255, 234 / 255, 237 / 255)
BLUE_LIGHT = (180 / 255, 196 / 255, 232 / 255)
BLUE_MID = (61 / 255, 104 / 255, 187 / 255)
ACCENT = (192 / 255, 57 / 255, 43 / 255)
GREEN = (46 / 255, 125 / 255, 50 / 255)
WHITE = (1.0, 1.0, 1.0)


def _box(ax, x, y, w, h, text, fc=GRAY_LT, ec=NAVY, fs=15, fw="normal",
         tc=GRAY_DK, lw=1.8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, wrap=True,
            linespacing=1.2)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.5, style="-|>", ms=22):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=ms, shrinkA=2, shrinkB=2,
    ))


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # ========================================================
    # Titulo
    # ========================================================
    ax.text(50, 56.5, "FiLM — Feature-wise Linear Modulation",
            ha="center", va="center",
            fontsize=30, fontweight="bold", color=NAVY)
    ax.text(50, 52.5,
            "Aplicação única ao final dos 4 estágios do ConvNeXt-T, "
            "antes do classificador",
            ha="center", va="center",
            fontsize=17, color=GRAY_MD, style="italic")

    # ========================================================
    # LINHA SUPERIOR — Geracao do sinal condicionante
    # ========================================================
    y_top = 36
    box_h = 8

    _box(ax, 2, y_top, 12, box_h,
         "Imagem\n224 × 224 × 3",
         fc=GRAY_LT, ec=NAVY, fs=15, fw="bold")
    _arrow(ax, 14, y_top + box_h / 2, 18, y_top + box_h / 2,
           color=GREEN, lw=2.6)

    # Classificador MST (verde — congelado)
    _box(ax, 18, y_top, 18, box_h,
         "Classificador MST\ncongelado",
         fc=GREEN, ec=GREEN, fs=16, fw="bold", tc=WHITE)
    ax.text(27, y_top + box_h + 1.6,
            "treinado na Etapa 1  (MSTE + CCv2)",
            ha="center", va="center",
            fontsize=13, color=GRAY_MD, style="italic")
    _arrow(ax, 36, y_top + box_h / 2, 40, y_top + box_h / 2,
           color=NAVY, lw=2.6)

    # Vetor z (10-dim)
    _box(ax, 40, y_top, 20, box_h, "",
         fc=WHITE, ec=NAVY, fs=16, fw="bold", lw=1.8)
    ax.text(50, y_top + box_h / 2 + 1.6,
            "Vetor  z  ∈  ℝ¹⁰",
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=NAVY)
    ax.text(50, y_top + box_h / 2 - 1.9,
            "softmax sobre os 10 tons Monk",
            ha="center", va="center",
            fontsize=13, color=GRAY_MD, style="italic")
    _arrow(ax, 60, y_top + box_h / 2, 64, y_top + box_h / 2,
           color=NAVY, lw=2.6)

    # MLPs f_g, f_b
    mlp_x = 64
    mlp_w = 24
    _box(ax, mlp_x, y_top, mlp_w, box_h,
         "MLPs   f_γ  ,   f_β",
         fc=WHITE, ec=ACCENT, fs=17, fw="bold", tc=ACCENT, lw=2.4)
    ax.text(mlp_x + mlp_w / 2, y_top + box_h + 1.6,
            "≈ 14 k parâmetros  (≈ 0,05 % do backbone)",
            ha="center", va="center",
            fontsize=13, color=GRAY_MD, style="italic")

    # ========================================================
    # LINHA INFERIOR — Backbone ConvNeXt-T (4 estagios) + FiLM + Classif
    # ========================================================
    y_bb = 11
    bb_h = 9

    # Imagem (entrada do backbone)
    _box(ax, 2, y_bb, 10, bb_h,
         "Imagem\n224 × 224 × 3",
         fc=GRAY_LT, ec=NAVY, fs=15, fw="bold")
    _arrow(ax, 12, y_bb + bb_h / 2, 15, y_bb + bb_h / 2,
           color=NAVY, lw=2.8)

    # Container "ConvNeXt-T backbone"
    stages_x0 = 15
    stage_w = 10
    stage_gap = 1.0
    container_w = 4 * stage_w + 3 * stage_gap + 1.6
    container = FancyBboxPatch(
        (stages_x0 - 0.8, y_bb - 0.8), container_w, bb_h + 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        fc=(0.96, 0.97, 0.99), ec=NAVY, lw=1.4,
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(container)
    # Rotulo ACIMA do container (fora dele), afastado dos estagios
    ax.text(stages_x0 - 0.8 + container_w / 2, y_bb + bb_h + 2.3,
            "ConvNeXt-T backbone  (4 estágios hierárquicos  ·  ≈ 28 M parâmetros)",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=NAVY, style="italic")

    # 4 estagios
    stage_data = [
        ("Est. 1", "56×56×96",   BLUE_LIGHT,                           NAVY),
        ("Est. 2", "28×28×192",  (155/255, 180/255, 220/255),           NAVY),
        ("Est. 3", "14×14×384",  BLUE_MID,                              WHITE),
        ("Est. 4", "7×7×768",    (46/255, 79/255, 138/255),             WHITE),
    ]
    cur_x = stages_x0
    for i, (nome, dims, fc, tc) in enumerate(stage_data):
        _box(ax, cur_x, y_bb, stage_w, bb_h, "",
             fc=fc, ec=NAVY, lw=1.5)
        ax.text(cur_x + stage_w / 2, y_bb + bb_h / 2 + 1.5,
                nome, ha="center", va="center",
                fontsize=14, fontweight="bold", color=tc)
        ax.text(cur_x + stage_w / 2, y_bb + bb_h / 2 - 1.5,
                dims, ha="center", va="center",
                fontsize=11, color=tc, style="italic")
        if i < 3:
            dx = cur_x + stage_w
            _arrow(ax, dx, y_bb + bb_h / 2,
                   dx + stage_gap, y_bb + bb_h / 2,
                   color=NAVY, lw=2.0, ms=16)
        cur_x += stage_w + stage_gap

    # Fim do backbone -> seta -> FiLM
    bb_end_x = cur_x
    _arrow(ax, bb_end_x, y_bb + bb_h / 2,
           bb_end_x + 2.5, y_bb + bb_h / 2,
           color=NAVY, lw=2.8)

    # FiLM (unico — ao final dos 4 estagios)
    film_x = bb_end_x + 2.5
    film_w = 11
    _box(ax, film_x, y_bb, film_w, bb_h,
         "FiLM",
         fc=NAVY, ec=NAVY, fs=24, fw="bold", tc=WHITE, lw=2.6)
    ax.text(film_x + film_w / 2, y_bb + bb_h + 0.4,
            "modula as 768 features finais",
            ha="center", va="center",
            fontsize=12, color=GRAY_MD, style="italic")
    _arrow(ax, film_x + film_w, y_bb + bb_h / 2,
           film_x + film_w + 2.5, y_bb + bb_h / 2,
           color=NAVY, lw=2.8)

    # Classificador 7 racas
    class_x = film_x + film_w + 2.5
    class_w = 10
    _box(ax, class_x, y_bb, class_w, bb_h,
         "Classif.\n7 raças",
         fc=ACCENT, ec=ACCENT, fs=15, fw="bold", tc=WHITE, lw=2.2)
    class_center = class_x + class_w / 2
    # (Removida a seta e rotulo "Predicao de raca" para evitar
    #  sobreposicao com a legenda no rodape. O proprio bloco
    #  "Classif. 7 racas" ja comunica a saida.)

    # ========================================================
    # Conector MLPs -> FiLM (sinal gamma, beta)
    # ========================================================
    mlp_bottom_x = mlp_x + mlp_w / 2
    mlp_bottom_y = y_top
    film_top_x = film_x + film_w / 2
    film_top_y = y_bb + bb_h

    mid_y = (mlp_bottom_y + film_top_y) / 2

    ax.plot([mlp_bottom_x, mlp_bottom_x], [mlp_bottom_y, mid_y],
            color=ACCENT, lw=2.6, zorder=4)
    ax.plot([mlp_bottom_x, film_top_x], [mid_y, mid_y],
            color=ACCENT, lw=2.6, zorder=4)
    _arrow(ax, film_top_x, mid_y, film_top_x, film_top_y + 0.3,
           color=ACCENT, lw=2.6, ms=22)

    # Rotulo (gamma, beta)
    ax.text(film_top_x + 1.8, mid_y + 1.6,
            "(γ, β)",
            ha="left", va="center",
            fontsize=18, fontweight="bold",
            color=ACCENT, style="italic")

    # ========================================================
    # Rodape — equacao canonica + legenda
    # ========================================================
    _box(ax, 4, 0.5, 44, 5.5, "",
         fc=WHITE, ec=NAVY, lw=2.0)
    ax.text(26, 4.2,
            "Equação canônica  (Perez et al., 2018)",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=NAVY)
    ax.text(26, 1.8,
            "F′  =  γ  ⊙  F  +  β",
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=ACCENT, style="italic")

    legend_items = [
        (GREEN,  "Congelado  (classificador MST, Etapa 1)"),
        (NAVY,   "Treinado end-to-end  (backbone + FiLM)"),
        (ACCENT, "Camadas novas  (~0,05 % dos params)"),
    ]
    legend_x = 55
    for i, (color, text) in enumerate(legend_items):
        y_leg = 4.5 - i * 1.6
        ax.add_patch(mpatches.Rectangle(
            (legend_x, y_leg), 2.6, 1.1, fc=color, ec=color))
        ax.text(legend_x + 3.6, y_leg + 0.55, text,
                ha="left", va="center",
                fontsize=13, color=GRAY_DK)

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
