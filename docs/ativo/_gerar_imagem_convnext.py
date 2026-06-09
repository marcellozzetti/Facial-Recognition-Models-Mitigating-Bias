"""Gera diagrama da arquitetura ConvNeXt-T (Liu et al. 2022, CVPR).

Saída: docs/ativo/imagens/convnext_t_architecture.png (alta resolução 300 DPI).

Mostra:
- Pipeline hierárquico em 4 estágios (Stem + Stages 1-4 + Head)
- Tensor shapes em cada nível
- Detalhe expandido de UM bloco ConvNeXt
- Total de parâmetros e mapeamento para nossa tarefa (7 raças do FairFace)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Paleta consistente com o PPTX e diagrama FiLM
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


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis("off")

    # ====== Título ======
    ax.text(50, 72.5, "ConvNeXt-T (Tiny) — arquitetura",
            ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(50, 69.5, "Liu et al. 2022 (CVPR) — backbone moderno escolhido para nossa tese (~28M parâmetros)",
            ha="center", va="center", fontsize=12, color=GRAY_MD, style="italic")

    # ====== PARTE 1 — Pipeline hierárquico horizontal ======
    pipeline_y = 50
    box_h = 5.5
    arrow_gap = 1.5

    # Input
    _box(ax, 2, pipeline_y, 8, box_h, "Imagem\n224 x 224 x 3", fc=GREEN, ec=GREEN, fs=10, fw="bold", tc=WHITE)
    _arrow(ax, 10, pipeline_y + box_h/2, 11.5, pipeline_y + box_h/2)

    # Stem
    _box(ax, 11.5, pipeline_y, 9, box_h, "Stem\nConv 4x4 stride 4\n56 x 56 x 96", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, 20.5, pipeline_y + box_h/2, 22, pipeline_y + box_h/2)

    # Stage 1 — 3 blocks @ 96
    _box(ax, 22, pipeline_y, 9, box_h, "Estágio 1\n3 blocos ConvNeXt\n56 x 56 x 96", fc=NAVY, ec=NAVY, fs=9, fw="bold", tc=WHITE)
    _arrow(ax, 31, pipeline_y + box_h/2, 32.5, pipeline_y + box_h/2)

    # Downsample 1
    _box(ax, 32.5, pipeline_y, 6, box_h, "Down\nLN + 2x2\nstride 2", fc=GRAY_LT, ec=ACCENT, fs=8)
    _arrow(ax, 38.5, pipeline_y + box_h/2, 40, pipeline_y + box_h/2)

    # Stage 2 — 3 blocks @ 192
    _box(ax, 40, pipeline_y, 9, box_h, "Estágio 2\n3 blocos ConvNeXt\n28 x 28 x 192", fc=NAVY, ec=NAVY, fs=9, fw="bold", tc=WHITE)
    _arrow(ax, 49, pipeline_y + box_h/2, 50.5, pipeline_y + box_h/2)

    # Downsample 2
    _box(ax, 50.5, pipeline_y, 6, box_h, "Down\nLN + 2x2\nstride 2", fc=GRAY_LT, ec=ACCENT, fs=8)
    _arrow(ax, 56.5, pipeline_y + box_h/2, 58, pipeline_y + box_h/2)

    # Stage 3 — 9 blocks @ 384 (estágio "wide" do ConvNeXt-T)
    _box(ax, 58, pipeline_y, 11, box_h, "Estágio 3\n9 blocos ConvNeXt\n14 x 14 x 384", fc=NAVY, ec=NAVY, fs=9, fw="bold", tc=WHITE)
    _arrow(ax, 69, pipeline_y + box_h/2, 70.5, pipeline_y + box_h/2)

    # Downsample 3
    _box(ax, 70.5, pipeline_y, 6, box_h, "Down\nLN + 2x2\nstride 2", fc=GRAY_LT, ec=ACCENT, fs=8)
    _arrow(ax, 76.5, pipeline_y + box_h/2, 78, pipeline_y + box_h/2)

    # Stage 4 — 3 blocks @ 768
    _box(ax, 78, pipeline_y, 9, box_h, "Estágio 4\n3 blocos ConvNeXt\n7 x 7 x 768", fc=NAVY, ec=NAVY, fs=9, fw="bold", tc=WHITE)
    _arrow(ax, 87, pipeline_y + box_h/2, 88.5, pipeline_y + box_h/2)

    # Head
    _box(ax, 88.5, pipeline_y, 9.5, box_h, "Head\nAvgPool + LN\n+ Linear(7)", fc=ACCENT, ec=ACCENT, fs=9, fw="bold", tc=WHITE)

    # Annotation abaixo: depths e channels
    ax.text(50, pipeline_y - 2.5,
            "depths = [3, 3, 9, 3]      channels = [96, 192, 384, 768]      total ≈ 28M parâmetros",
            ha="center", va="center", fontsize=11, color=GRAY_DK, fontweight="bold")

    # Linha divisória sutil
    ax.plot([3, 97], [42, 42], color=GRAY_MD, lw=0.7, linestyle="--", alpha=0.5)

    # ====== PARTE 2 — Detalhe expandido de UM bloco ConvNeXt ======
    ax.text(50, 39, "Detalhe: estrutura interna de UM bloco ConvNeXt",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(50, 36.5,
            "Aplicado dentro de cada estágio do pipeline acima. Mantém shape espacial (H x W x C → H x W x C); residual flui de fora.",
            ha="center", va="center", fontsize=10, color=GRAY_MD, style="italic")

    # Layout do bloco — fluxo vertical à esquerda, residual à direita
    block_left = 25
    block_y_top = 31.5
    block_w = 16

    # Entrada x
    _box(ax, block_left, block_y_top, block_w, 3, "Entrada x  (H x W x C)", fc=GREEN, ec=GREEN, fs=10, fw="bold", tc=WHITE)
    _arrow(ax, block_left + block_w/2, block_y_top, block_left + block_w/2, block_y_top - 1.5)

    # DWConv 7x7
    _box(ax, block_left, block_y_top - 5.5, block_w, 3, "Depthwise Conv  7x7\n(mistura espacial)", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, block_left + block_w/2, block_y_top - 5.5, block_left + block_w/2, block_y_top - 7)

    # LayerNorm
    _box(ax, block_left, block_y_top - 11, block_w, 3, "LayerNorm", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, block_left + block_w/2, block_y_top - 11, block_left + block_w/2, block_y_top - 12.5)

    # Pointwise expand
    _box(ax, block_left, block_y_top - 16.5, block_w, 3, "Pointwise Conv  1x1\n(expande para 4C)", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, block_left + block_w/2, block_y_top - 16.5, block_left + block_w/2, block_y_top - 18)

    # GELU
    _box(ax, block_left, block_y_top - 22, block_w, 3, "GELU\n(não linearidade)", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, block_left + block_w/2, block_y_top - 22, block_left + block_w/2, block_y_top - 23.5)

    # Pointwise project
    _box(ax, block_left, block_y_top - 27.5, block_w, 3, "Pointwise Conv  1x1\n(projeta de volta para C)", fc=GRAY_LT, ec=NAVY, fs=9)
    _arrow(ax, block_left + block_w/2, block_y_top - 27.5, block_left + block_w/2, block_y_top - 29)

    # Residual sum (vermelho)
    _box(ax, block_left, block_y_top - 32, block_w, 3, "y  =  x  +  bloco(x)", fc=ACCENT, ec=ACCENT, fs=10, fw="bold", tc=WHITE)

    # Residual skip curva — partindo de cima e chegando no '+' do final
    skip_x = block_left + block_w + 4
    _arrow(ax, block_left + block_w, block_y_top + 1.5, skip_x, block_y_top + 1.5, color=ACCENT, lw=1.5, style="-")
    _arrow(ax, skip_x, block_y_top + 1.5, skip_x, block_y_top - 30.5, color=ACCENT, lw=1.5, style="-")
    _arrow(ax, skip_x, block_y_top - 30.5, block_left + block_w, block_y_top - 30.5, color=ACCENT, lw=1.5)
    ax.text(skip_x + 1, block_y_top - 14, "skip\nresidual",
            ha="left", va="center", fontsize=10, color=ACCENT, fontweight="bold")

    # ====== Anotações laterais — comparação com ResNet ======
    note_x = 55
    note_y = 27
    _box(ax, note_x, note_y - 8, 38, 12, "", fc=GRAY_LT, ec=NAVY, lw=1.0)
    ax.text(note_x + 19, note_y + 2.5, "Por que ConvNeXt-T e não ResNet-34?",
            ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
    note_lines = [
        "•  Bloco ConvNeXt aproxima o desempenho de Vision Transformers",
        "    mantendo o paradigma convolucional puro.",
        "•  Depthwise 7x7 imita auto-atenção espacial em janela larga.",
        "•  LayerNorm (não BatchNorm) e GELU (não ReLU) modernizam a rede.",
        "•  Parâmetros (~28M) similares a ResNet-50 mas accuracy de ViT-B.",
        "•  Compatível com FiLM-conditioning por canal (nossa contribuição).",
    ]
    for i, line in enumerate(note_lines):
        ax.text(note_x + 0.8, note_y + 0.5 - i*1.4, line,
                ha="left", va="center", fontsize=9, color=GRAY_DK)

    # ====== Legenda de cores no rodapé ======
    legend_y = 4
    legend_items = [
        (GREEN, "Entrada / tensor inicial"),
        (NAVY, "Estágios e operações convolucionais principais"),
        (ACCENT, "Conexões residuais e head de classificação (7 raças do FairFace)"),
        (GRAY_LT, "Operações auxiliares (downsample, normalização)"),
    ]
    for i, (color, text) in enumerate(legend_items):
        x_leg = 4 + (i % 2) * 50
        y_leg = legend_y - (i // 2) * 1.8
        ax.add_patch(mpatches.Rectangle((x_leg, y_leg), 1.6, 1.2, fc=color, ec=color))
        ax.text(x_leg + 2.3, y_leg + 0.6, text, ha="left", va="center",
                fontsize=9, color=GRAY_DK)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    out = Path(__file__).resolve().parent / "imagens" / "convnext_t_architecture.png"
    build_figure(out)


if __name__ == "__main__":
    main()
