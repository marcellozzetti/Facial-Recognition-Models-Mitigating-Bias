"""Gera diagrama SIMPLIFICADO do mecanismo FiLM aplicado ao nosso pipeline.

Objetivo: comunicar sem ambiguidade à banca de qualificação, com
tipografia legível e caixas dimensionadas para o conteúdo.

Saída: docs/ativo/imagens/film_pipeline.png (alta resolução 300 DPI).
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
            linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.0, style="-|>", ms=18):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=ms, shrinkA=2, shrinkB=2,
    ))


def build_figure(out_path: Path) -> None:
    # Figura mais larga para acomodar 4 pares (Estágio + FiLM) com folga
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ---------- Título ----------
    ax.text(50, 58.5, "FiLM — Feature-wise Linear Modulation",
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)
    ax.text(50, 55,
            "Uma camada FiLM aplicada após cada um dos 4 estágios do ConvNeXt-T",
            ha="center", va="center",
            fontsize=14, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA SUPERIOR — geração do sinal condicionante z e MLPs
    # =========================================================
    y_top = 40
    box_h = 8

    # Imagem
    _box(ax, 2, y_top, 11, box_h,
         "Imagem\n224 × 224 × 3",
         fc=GRAY_LT, ec=NAVY, fs=13, fw="bold")
    _arrow(ax, 13, y_top + box_h / 2, 18, y_top + box_h / 2, color=GREEN, lw=2.2)

    # Classificador MST (verde — congelado)
    _box(ax, 18, y_top, 18, box_h,
         "Classificador MST\ncongelado",
         fc=GREEN, ec=GREEN, fs=14, fw="bold", tc=WHITE)
    ax.text(27, y_top + box_h + 1.2,
            "treinado na Etapa 1  (MSTE + CCv2)",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")
    _arrow(ax, 36, y_top + box_h / 2, 41, y_top + box_h / 2, color=NAVY, lw=2.2)

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
    _arrow(ax, 59, y_top + box_h / 2, 64, y_top + box_h / 2, color=NAVY, lw=2.2)

    # MLPs f_γ, f_β
    _box(ax, 64, y_top, 22, box_h,
         "MLPs   f_γ  ,   f_β\num par por estágio",
         fc=WHITE, ec=ACCENT, fs=14, fw="bold", tc=ACCENT, lw=2.0)
    ax.text(75, y_top + box_h + 1.2,
            "≈ 380 k parâmetros  (≈ 1,3 % do backbone)",
            ha="center", va="center",
            fontsize=11, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA INFERIOR — backbone ConvNeXt-T com 4 estágios + FiLM
    # =========================================================
    y_bb = 15
    stage_w = 9.5
    film_w = 7.5
    gap = 0.7
    channels = [96, 192, 384, 768]

    # Geometria dos 4 pares
    total_w = 4 * stage_w + 4 * film_w + 7 * gap + 13
    x0 = (100 - total_w) / 2 + 2.5

    # ---------- Container "ConvNeXt-T backbone" ----------
    container_x = x0 - 1.2
    container_w = 4 * stage_w + 4 * film_w + 7 * gap + 2.4
    container_y = y_bb - 2.0
    container_h = 10.5
    container = FancyBboxPatch(
        (container_x, container_y), container_w, container_h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        fc=(0.96, 0.97, 0.99), ec=NAVY, lw=1.2,
        linestyle=(0, (4, 3)),  # tracejado
    )
    ax.add_patch(container)
    ax.text(container_x + container_w / 2, container_y + container_h - 1.0,
            "ConvNeXt-T backbone   (4 estágios hierárquicos)",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY, style="italic")

    # Seta da imagem para o primeiro estágio (entra no container)
    _arrow(ax, 7.5, y_top, 7.5, y_bb + 6, color=NAVY, lw=2.2)
    _arrow(ax, 7.5, y_bb + 6, x0 - 0.5, y_bb + 3, color=NAVY, lw=2.2)

    cur_x = x0
    film_centers_x = []

    for i in range(4):
        _box(ax, cur_x, y_bb, stage_w, 6,
             f"Estágio {i + 1}\n{channels[i]} canais",
             fc=GRAY_LT, ec=NAVY, fs=12, fw="bold")
        _arrow(ax, cur_x + stage_w, y_bb + 3,
               cur_x + stage_w + gap, y_bb + 3)
        cur_x += stage_w + gap

        # FiLM i — só o rótulo
        _box(ax, cur_x, y_bb - 0.5, film_w, 7,
             f"FiLM {i + 1}",
             fc=NAVY, ec=NAVY, fs=15, fw="bold", tc=WHITE)
        film_centers_x.append(cur_x + film_w / 2)
        cur_x += film_w

        if i < 3:
            _arrow(ax, cur_x, y_bb + 3, cur_x + gap, y_bb + 3)
            cur_x += gap

    # Seta para o classificador final (sai do container)
    _arrow(ax, cur_x, y_bb + 3, cur_x + gap + 1.2, y_bb + 3)
    cur_x += gap + 1.2

    _box(ax, cur_x, y_bb - 0.5, 11, 7,
         "Classificador\n7 raças",
         fc=ACCENT, ec=ACCENT, fs=13, fw="bold", tc=WHITE)
    class_center = cur_x + 5.5

    _arrow(ax, class_center, y_bb - 0.5, class_center, y_bb - 4, color=ACCENT)
    ax.text(class_center, y_bb - 5.5,
            "Predição de raça",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=GRAY_DK)

    # =========================================================
    # BARRAMENTO de (γᵢ, βᵢ) — uma linha horizontal + 4 quedas curtas
    # =========================================================
    # Sai do bloco MLPs (bordo inferior) para o barramento
    mlp_out_x = 75
    mlp_out_y = y_top

    bus_y = 30  # altura do barramento (linha horizontal comum)

    # Descida da MLP até o barramento
    _arrow(ax, mlp_out_x, mlp_out_y, mlp_out_x, bus_y + 0.4,
           color=ACCENT, lw=2.0)

    # Linha horizontal do barramento indo até o primeiro FiLM (esquerda)
    # e até o último FiLM (direita)
    bus_left = film_centers_x[0]
    bus_right = film_centers_x[-1]
    bus_start = min(bus_left, mlp_out_x)
    bus_end = max(bus_right, mlp_out_x)
    ax.add_patch(mpatches.Rectangle(
        (bus_start, bus_y - 0.15), bus_end - bus_start, 0.3,
        fc=ACCENT, ec=ACCENT))

    # Quedas curtas do barramento até cada FiLM
    for fx in film_centers_x:
        _arrow(ax, fx, bus_y, fx, y_bb + 6.5,
               color=ACCENT, lw=1.6, style="-|>", ms=14)

    # Rótulo (γᵢ, βᵢ) acima do barramento, alinhado à esquerda
    ax.text(bus_start - 0.5, bus_y + 1.4,
            "(γᵢ , βᵢ)  ·  i = 1, 2, 3, 4",
            ha="left", va="center",
            fontsize=13, fontweight="bold",
            color=ACCENT, style="italic")

    # =========================================================
    # RODAPÉ — equação canônica + legenda
    # =========================================================
    # Caixa da equação (esquerda)
    _box(ax, 4, 4, 44, 6, "",
         fc=WHITE, ec=NAVY, lw=1.8)
    ax.text(26, 7.8,
            "Equação canônica  (Perez et al., 2018)",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=NAVY)
    ax.text(26, 5.3,
            "F′  =  γ  ⊙  F  +  β",
            ha="center", va="center",
            fontsize=20, fontweight="bold", color=ACCENT, style="italic")

    # Legenda de cores (direita)
    legend_items = [
        (GREEN, "Congelado  (Classificador MST)"),
        (NAVY,  "Treinado end-to-end  (backbone + FiLM)"),
        (ACCENT, "Camadas novas  (~1,3 % dos params)"),
    ]
    legend_x = 55
    for i, (color, text) in enumerate(legend_items):
        y_leg = 8.5 - i * 1.9
        ax.add_patch(mpatches.Rectangle(
            (legend_x, y_leg), 2.5, 1.2, fc=color, ec=color))
        ax.text(legend_x + 3.5, y_leg + 0.6, text,
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
