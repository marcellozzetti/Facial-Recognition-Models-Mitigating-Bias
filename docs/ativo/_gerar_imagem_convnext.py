"""Gera diagrama do backbone ConvNeXt-T mostrando os 4 estagios hierarquicos.

Objetivo: comunicar visualmente a arquitetura do backbone escolhido
na metodologia (Etapa 3). Destaca:
  - entrada (imagem 224x224x3)
  - stem conv 4x4 stride 4
  - 4 estagios hierarquicos (3, 3, 9, 3 blocos) com canais crescentes
    (96 -> 192 -> 384 -> 768) e resolucao decrescente (56 -> 28 -> 14 -> 7)
  - GAP + LayerNorm + classificador final
  - pontos naturais de insercao de FiLM (apos cada estagio)

Saida: docs/ativo/imagens/convnext_backbone.png + copia em
docs/tese/images/convnext_backbone.png (300 DPI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = (31 / 255, 42 / 255, 78 / 255)
BLUE_LIGHT = (180 / 255, 196 / 255, 232 / 255)
BLUE_MID = (61 / 255, 104 / 255, 187 / 255)
GRAY_DK = (61 / 255, 66 / 255, 78 / 255)
GRAY_MD = (112 / 255, 118 / 255, 130 / 255)
GRAY_LT = (232 / 255, 234 / 255, 237 / 255)
ACCENT = (192 / 255, 57 / 255, 43 / 255)
WHITE = (1.0, 1.0, 1.0)


def _rbox(ax, x, y, w, h, fc, ec=NAVY, lw=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)
    return box


def _text(ax, x, y, s, size=12, bold=False, italic=False,
          color=GRAY_DK, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal",
            color=color)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.0, ms=18):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", color=color, lw=lw,
        mutation_scale=ms, shrinkA=2, shrinkB=2,
    ))


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    # ---------- Titulo ----------
    ax.text(50, 52, "ConvNeXt-T — backbone convolucional moderno",
            ha="center", va="center",
            fontsize=22, fontweight="bold", color=NAVY)
    ax.text(50, 48.3,
            "4 estágios hierárquicos  ·  28 M parâmetros  ·  "
            "LayerNorm  ·  Liu et al. (2022, CVPR)",
            ha="center", va="center",
            fontsize=13, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA PRINCIPAL — fluxo horizontal
    # =========================================================
    y_main = 24
    stage_h = 10

    # Imagem de entrada
    _rbox(ax, 2, y_main, 10, stage_h, GRAY_LT, ec=NAVY)
    _text(ax, 7, y_main + stage_h / 2 + 0.6, "Imagem", size=13, bold=True)
    _text(ax, 7, y_main + stage_h / 2 - 1.0, "224 × 224 × 3",
          size=11, color=GRAY_DK)
    _arrow(ax, 12, y_main + stage_h / 2, 16, y_main + stage_h / 2)

    # Stem (patchify conv 4x4 stride 4)
    _rbox(ax, 16, y_main + 1, 10, stage_h - 2, WHITE, ec=NAVY)
    _text(ax, 21, y_main + stage_h / 2 + 0.7, "Stem", size=12, bold=True,
          color=NAVY)
    _text(ax, 21, y_main + stage_h / 2 - 0.9, "conv 4×4", size=10,
          color=GRAY_DK)
    _text(ax, 21, y_main + stage_h / 2 - 2.1, "stride 4", size=10,
          color=GRAY_DK, italic=True)
    _arrow(ax, 26, y_main + stage_h / 2, 29, y_main + stage_h / 2)

    # Os 4 estagios
    stage_data = [
        ("Estágio 1", "3 blocos",  "56 × 56 × 96",   BLUE_LIGHT),
        ("Estágio 2", "3 blocos",  "28 × 28 × 192",  (155 / 255, 180 / 255, 220 / 255)),
        ("Estágio 3", "9 blocos",  "14 × 14 × 384",  BLUE_MID),
        ("Estágio 4", "3 blocos",  "7 × 7 × 768",    (46 / 255, 79 / 255, 138 / 255)),
    ]
    stage_w = 11
    gap = 1.0
    cur_x = 29.5
    stage_centers = []
    for i, (nome, blocos, dims, fc) in enumerate(stage_data):
        _rbox(ax, cur_x, y_main, stage_w, stage_h, fc, ec=NAVY, lw=1.8)
        # cor do texto (branco se fundo escuro)
        tc = WHITE if i >= 2 else NAVY
        _text(ax, cur_x + stage_w / 2, y_main + stage_h - 1.5,
              nome, size=13, bold=True, color=tc)
        _text(ax, cur_x + stage_w / 2, y_main + stage_h / 2 - 0.3,
              blocos, size=11, color=tc)
        _text(ax, cur_x + stage_w / 2, y_main + 1.5,
              dims, size=10, italic=True,
              color=tc if i >= 2 else GRAY_DK)
        stage_centers.append(cur_x + stage_w / 2)
        if i < 3:
            # Downsample seta + rotulo
            dx = cur_x + stage_w
            _arrow(ax, dx, y_main + stage_h / 2,
                   dx + gap - 0.1, y_main + stage_h / 2)
        cur_x += stage_w + gap

    # Seta para GAP + LN
    _arrow(ax, cur_x - gap + 0.1, y_main + stage_h / 2,
           cur_x + 1, y_main + stage_h / 2)

    # GAP + LayerNorm (pos-ultimo estagio)
    _rbox(ax, cur_x + 1, y_main + 1.5, 7, stage_h - 3, WHITE, ec=NAVY)
    _text(ax, cur_x + 4.5, y_main + stage_h / 2 + 0.6, "GAP", size=11, bold=True,
          color=NAVY)
    _text(ax, cur_x + 4.5, y_main + stage_h / 2 - 1.3, "+ LayerNorm", size=10,
          color=GRAY_DK, italic=True)
    _arrow(ax, cur_x + 8, y_main + stage_h / 2, cur_x + 10, y_main + stage_h / 2)

    # Classificador 7 racas
    cls_x = cur_x + 10
    _rbox(ax, cls_x, y_main + 1, 8, stage_h - 2, ACCENT, ec=ACCENT, lw=2.0)
    _text(ax, cls_x + 4, y_main + stage_h / 2 + 0.5, "Linear",
          size=13, bold=True, color=WHITE)
    _text(ax, cls_x + 4, y_main + stage_h / 2 - 1.1, "768 → 7",
          size=11, color=WHITE)
    _text(ax, cls_x + 4, y_main + stage_h / 2 - 2.4, "7 raças",
          size=10, color=WHITE, italic=True)

    # Seta para fora (predicao)
    _arrow(ax, cls_x + 4, y_main + 1, cls_x + 4, y_main - 3.5, color=ACCENT)
    _text(ax, cls_x + 4, y_main - 5, "Predição racial",
          size=11, bold=True, color=GRAY_DK)

    # =========================================================
    # BAIXO — marcadores de pontos de insercao FiLM
    # =========================================================
    y_mark = y_main - 2.0
    for i, cx in enumerate(stage_centers):
        # Pequena marca em accent abaixo de cada estagio
        marker = mpatches.Circle(
            (cx, y_mark), 0.6,
            fc=WHITE, ec=ACCENT, lw=1.6,
        )
        ax.add_patch(marker)
        ax.text(cx, y_mark, str(i + 1), ha="center", va="center",
                fontsize=9, fontweight="bold", color=ACCENT)
        _text(ax, cx, y_mark - 2.0,
              "ponto FiLM",
              size=8.5, italic=True, color=ACCENT)

    # Linha conectora e rotulo
    ax.plot([stage_centers[0], stage_centers[-1]],
            [y_mark - 3.2, y_mark - 3.2],
            color=ACCENT, lw=1.0, linestyle=":")
    _text(ax, (stage_centers[0] + stage_centers[-1]) / 2, y_mark - 4.3,
          "4 pontos naturais de inserção para FiLM "
          "(após cada estágio, sem modificar blocos internos)",
          size=10.5, italic=True, bold=True, color=ACCENT)

    # =========================================================
    # ALTO — propriedades-chave
    # =========================================================
    y_prop = 43
    propriedades = [
        ("82 % top-1\nImageNet",  "paridade com ViTs"),
        ("~ 1/3  dos\nparâmetros", "vs. Swin-T"),
        ("LayerNorm\n(não BN)",   "estável com batch pequeno"),
        ("~ 300 h  GPU",          "3 sementes × 3 configs"),
    ]
    pw = 15
    gap = 1.5
    total_w = len(propriedades) * pw + (len(propriedades) - 1) * gap
    px0 = (100 - total_w) / 2
    for i, (big, small) in enumerate(propriedades):
        x = px0 + i * (pw + gap)
        _rbox(ax, x, y_prop - 2, pw, 3.5, WHITE, ec=NAVY, lw=1.2)
        _text(ax, x + pw / 2, y_prop + 0.3, big,
              size=11, bold=True, color=NAVY)
        _text(ax, x + pw / 2, y_prop - 1.2, small,
              size=9, italic=True, color=GRAY_MD)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    out = Path(__file__).resolve().parent / "imagens" / "convnext_backbone.png"
    build_figure(out)
    dest = (Path(__file__).resolve().parents[2]
            / "docs" / "tese" / "images" / "convnext_backbone.png")
    if dest.parent.exists():
        import shutil
        shutil.copy2(out, dest)
        print(f"Copiado para: {dest}")


if __name__ == "__main__":
    main()
