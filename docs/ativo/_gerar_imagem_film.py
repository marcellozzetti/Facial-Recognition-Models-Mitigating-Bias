"""Gera diagrama do mecanismo FiLM aplicado ao nosso pipeline.

Saída: docs/ativo/imagens/film_pipeline.png (alta resolução 300 DPI).

Mostra:
- SkinToneNet (frozen) extraindo vetor de contexto z da imagem
- MLPs f_gamma, f_beta gerando parâmetros de modulação
- Backbone ConvNeXt-T com 4 blocos FiLM inseridos por estágio
- Modulação F' = gamma * F + beta por canal
- Classificador final sobre 7 raças do FairFace
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Paleta consistente com o PPTX
NAVY = (31/255, 42/255, 78/255)
GRAY_DK = (61/255, 66/255, 78/255)
GRAY_MD = (112/255, 118/255, 130/255)
GRAY_LT = (232/255, 234/255, 237/255)
ACCENT = (192/255, 57/255, 43/255)
GREEN = (46/255, 125/255, 50/255)
WHITE = (1.0, 1.0, 1.0)


def _box(ax, x, y, w, h, text, fc=GRAY_LT, ec=NAVY, fs=11, fw="normal", tc=GRAY_DK, lw=1.5, rounding=0.05):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.0, style="-|>", mutation_scale=18):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=mutation_scale,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def _label(ax, x, y, text, color=GRAY_MD, fs=9, italic=True, ha="center", va="center"):
    style = "italic" if italic else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, style=style)


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # ====== Título ======
    ax.text(50, 57, "FiLM — Feature-wise Linear Modulation",
            ha="center", va="center", fontsize=20, fontweight="bold", color=NAVY)
    ax.text(50, 54, "Como o tom de pele é injetado como contexto na rede de classificação racial",
            ha="center", va="center", fontsize=12, color=GRAY_MD, style="italic")

    # ====== Linha 1 — Pipeline da imagem (input) ======
    # Foto / Imagem
    _box(ax, 2, 35, 9, 8, "Imagem\n224x224x3", fc=GRAY_LT, ec=NAVY, fs=10, fw="bold")

    # Branch 1 — saída para SkinToneNet (cima)
    _arrow(ax, 11, 41, 16, 46, color=GREEN, lw=2.0)
    # Branch 2 — saída para ConvNeXt-T (baixo)
    _arrow(ax, 11, 37, 16, 32, color=NAVY, lw=2.0)

    # ====== Linha 2 (superior) — SkinToneNet + contexto z + MLPs ======
    # SkinToneNet
    _box(ax, 16, 44, 12, 6, "SkinToneNet\n(ViT-Small)\nCONGELADO", fc=GREEN, ec=GREEN, fs=10, fw="bold", tc=WHITE)
    _label(ax, 22, 51.5, "Pereira et al. 2026", fs=8)

    _arrow(ax, 28, 47, 33, 47)

    # Vetor z (com bargraph mini)
    _box(ax, 33, 44, 11, 6, "", fc=WHITE, ec=NAVY, fs=10)
    ax.text(38.5, 49, "Vetor z (MST)", ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY)
    # Mini histograma representando o vetor MST 10-dim
    bar_x = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    bar_h = [0.4, 0.7, 1.2, 1.5, 1.0, 0.6, 0.3, 0.2, 0.15, 0.1]
    for bx, bh in zip(bar_x, bar_h):
        ax.add_patch(mpatches.Rectangle((bx + 0.05, 44.7), 0.55, bh, fc=NAVY, ec=None))
    _label(ax, 38.5, 44.3, "MST 1...........10", fs=7, italic=False, color=GRAY_MD)

    _arrow(ax, 44, 47, 49, 47)

    # MLPs f_gamma e f_beta
    _box(ax, 49, 47, 14, 4.5, "MLP  f_γ  (gera γ)", fc=GRAY_LT, ec=NAVY, fs=10)
    _box(ax, 49, 42, 14, 4.5, "MLP  f_β  (gera β)", fc=GRAY_LT, ec=NAVY, fs=10)
    _label(ax, 56, 52.2, "treináveis (~380k params total)", fs=8)

    # Setas das MLPs para baixo (apontando para os blocos FiLM)
    _arrow(ax, 56, 42, 56, 36, color=ACCENT, lw=2.0)
    ax.text(57, 38.5, "γ, β", ha="left", va="center", fontsize=11,
            fontweight="bold", color=ACCENT)

    # ====== Linha 3 (inferior) — ConvNeXt-T blocks com FiLM inserido ======
    # ConvNeXt stage 1
    _box(ax, 16, 27, 11, 6, "ConvNeXt-T\nEstágio 1\n96 canais", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, 27, 30, 31, 30)

    # FiLM block 1
    _box(ax, 31, 26.5, 9, 7, "FiLM\nF' = γ ⊙ F + β", fc=NAVY, ec=NAVY, fs=10, fw="bold", tc=WHITE)
    _arrow(ax, 40, 30, 44, 30)

    # ConvNeXt stage 2
    _box(ax, 44, 27, 11, 6, "ConvNeXt-T\nEstágio 2\n192 canais", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, 55, 30, 59, 30)

    # FiLM block 2
    _box(ax, 59, 26.5, 9, 7, "FiLM\nF' = γ ⊙ F + β", fc=NAVY, ec=NAVY, fs=10, fw="bold", tc=WHITE)
    _arrow(ax, 68, 30, 72, 30)

    # Reticências indicando blocos 3 e 4
    _box(ax, 72, 27, 11, 6, "... Estágios\n3 e 4 com FiLM\n(384, 768 ch)", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, 83, 30, 87, 30)

    # Classifier head
    _box(ax, 87, 26, 11, 8, "Classificador\n(7 raças)", fc=ACCENT, ec=ACCENT, fs=10, fw="bold", tc=WHITE)

    # ====== Saída final ======
    _arrow(ax, 92.5, 26, 92.5, 21, color=ACCENT)
    ax.text(92.5, 19.5, "Predição de raça\n(W, B, Ind, EA, SEA, ME, Lat)",
            ha="center", va="center", fontsize=9, color=GRAY_DK, fontweight="bold")

    # ====== Segunda seta para o segundo bloco FiLM ======
    # γ, β também alimentam o FiLM 2 (linha curva implícita)
    _arrow(ax, 56, 42, 63.5, 33.5, color=ACCENT, lw=1.2, style="-|>")
    _arrow(ax, 56, 42, 77.5, 33.5, color=ACCENT, lw=1.2, style="-|>")

    # ====== Legenda de cores no rodapé ======
    legend_y = 8
    legend_items = [
        (GREEN, "Pré-treinado, congelado (SkinToneNet)"),
        (NAVY, "Treinável end-to-end (ConvNeXt-T)"),
        (ACCENT, "Camadas novas (MLPs FiLM, ~1% dos params)"),
    ]
    for i, (color, text) in enumerate(legend_items):
        x_leg = 5 + i * 30
        ax.add_patch(mpatches.Rectangle((x_leg, legend_y), 2, 1.5, fc=color, ec=color))
        ax.text(x_leg + 3, legend_y + 0.75, text, ha="left", va="center", fontsize=9, color=GRAY_DK)

    # ====== Caixa explicativa "Em palavras simples" ======
    _box(ax, 3, 1.5, 94, 4.5, "", fc=GRAY_LT, ec=NAVY, lw=1.0)
    ax.text(50, 4.5, "Em palavras simples",
            ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.text(50, 2.7,
            "A rede de raça 'consulta' o tom de pele antes de decidir. Cada bloco FiLM recebe (γ, β) gerados a partir do vetor MST "
            "e ajusta as features intermediárias da rede — sem trocar o backbone.",
            ha="center", va="center", fontsize=9, color=GRAY_DK, style="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    out = Path(__file__).resolve().parent / "imagens" / "film_pipeline.png"
    build_figure(out)


if __name__ == "__main__":
    main()
