"""Gera diagrama do backbone ConvNeXt-T detalhado — foco exclusivo
na ARQUITETURA. Sem mencao a FiLM nem comparacoes com ResNet.

Mostra em 2 niveis:
  (a) Fluxo top-level: imagem -> Stem -> 4 estagios -> Head
      Com resolucoes espaciais e canais em cada etapa
      (56x56x96 -> 28x28x192 -> 14x14x384 -> 7x7x768).
  (b) Zoom detalhando o INTERIOR de um bloco ConvNeXt:
      DW 7x7 -> LN -> PW 1x1 (expand 4x) -> GELU -> PW 1x1 (reduce)
      -> LayerScale -> residual add.

Saida: docs/ativo/imagens/convnext_backbone.png + copia em
docs/tese/images/convnext_backbone.png (300 DPI).
"""

from __future__ import annotations

from pathlib import Path

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


def _rbox(ax, x, y, w, h, fc, ec=NAVY, lw=1.5, dashed=False):
    style = (0, (4, 3)) if dashed else "solid"
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        fc=fc, ec=ec, lw=lw, linestyle=style,
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


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.0, ms=18, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=ms, shrinkA=2, shrinkB=2,
    ))


def build_figure(out_path: Path) -> None:
    # figura mais larga para acomodar os 9 blocos essenciais do inset
    # com fontes legiveis
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    # ---------- Titulo ----------
    ax.text(50, 53, "ConvNeXt-T — anatomia do backbone",
            ha="center", va="center",
            fontsize=28, fontweight="bold", color=NAVY)
    ax.text(50, 49.3,
            "4 estágios hierárquicos  ·  28 M parâmetros  ·  "
            "Liu et al. (2022, CVPR)",
            ha="center", va="center",
            fontsize=16, color=GRAY_MD, style="italic")

    # =========================================================
    # LINHA PRINCIPAL — Fluxo top-level da arquitetura
    # =========================================================
    y_main = 36
    stage_h = 9

    # Imagem de entrada
    _rbox(ax, 1.5, y_main, 8, stage_h, GRAY_LT, ec=NAVY)
    _text(ax, 5.5, y_main + stage_h / 2 + 0.8, "Imagem", size=14, bold=True)
    _text(ax, 5.5, y_main + stage_h / 2 - 1.3, "224×224×3",
          size=12, color=GRAY_DK)
    _arrow(ax, 9.5, y_main + stage_h / 2, 11.5, y_main + stage_h / 2)

    # Stem (patchify conv 4x4 stride 4)
    _rbox(ax, 11.5, y_main + 0.8, 9.5, stage_h - 1.6, WHITE, ec=NAVY)
    _text(ax, 16.25, y_main + stage_h / 2 + 1.6, "Stem",
          size=14, bold=True, color=NAVY)
    _text(ax, 16.25, y_main + stage_h / 2 - 0.1, "conv 4×4",
          size=12, color=GRAY_DK)
    _text(ax, 16.25, y_main + stage_h / 2 - 1.6, "stride 4",
          size=11, italic=True, color=GRAY_MD)
    _text(ax, 16.25, y_main - 1.6, "56×56×96",
          size=12, italic=True, color=NAVY)
    _arrow(ax, 21, y_main + stage_h / 2, 23, y_main + stage_h / 2)

    # Os 4 estagios
    stage_data = [
        ("Estágio 1", "3 blocos",  "56×56×96",   BLUE_LIGHT, NAVY),
        ("Estágio 2", "3 blocos",  "28×28×192",  (155 / 255, 180 / 255, 220 / 255), NAVY),
        ("Estágio 3", "9 blocos",  "14×14×384",  BLUE_MID, WHITE),
        ("Estágio 4", "3 blocos",  "7×7×768",    (46 / 255, 79 / 255, 138 / 255), WHITE),
    ]
    stage_w = 11.5
    gap = 1.2
    cur_x = 23
    stage_centers = []
    for i, (nome, blocos, dims, fc, tc) in enumerate(stage_data):
        _rbox(ax, cur_x, y_main, stage_w, stage_h, fc, ec=NAVY, lw=1.8)
        _text(ax, cur_x + stage_w / 2, y_main + stage_h - 1.6,
              nome, size=15, bold=True, color=tc)
        _text(ax, cur_x + stage_w / 2, y_main + stage_h / 2 + 0.1,
              blocos, size=13, color=tc)
        _text(ax, cur_x + stage_w / 2, y_main - 1.6,
              dims, size=12, italic=True,
              color=NAVY)
        stage_centers.append(cur_x + stage_w / 2)
        if i < 3:
            # Downsample seta com rotulo "↓2" apenas (detalhes vao
            # para o texto do slide — manter so o essencial visual)
            dx = cur_x + stage_w
            _arrow(ax, dx, y_main + stage_h / 2,
                   dx + gap - 0.1, y_main + stage_h / 2)
            _text(ax, dx + gap / 2, y_main + stage_h + 1.5,
                  "↓ 2", size=14, bold=True, color=ACCENT)
        cur_x += stage_w + gap

    # Seta para GAP + LN
    _arrow(ax, cur_x - gap + 0.1, y_main + stage_h / 2,
           cur_x + 1, y_main + stage_h / 2)

    # Head: GAP + LN + Linear
    _rbox(ax, cur_x + 1, y_main + 0.8, 8, stage_h - 1.6, WHITE, ec=NAVY)
    _text(ax, cur_x + 5, y_main + stage_h / 2 + 1.6, "Head",
          size=14, bold=True, color=NAVY)
    _text(ax, cur_x + 5, y_main + stage_h / 2 - 0.1, "GAP + LN",
          size=12, color=GRAY_DK)
    _text(ax, cur_x + 5, y_main + stage_h / 2 - 1.6, "Linear 7 classes",
          size=11, italic=True, color=GRAY_MD)

    # =========================================================
    # LINHA 2 — Zoom detalhando o interior de um bloco ConvNeXt
    # =========================================================
    # Dashed callout do Estagio 3 para o inset
    callout_src_x = stage_centers[2]  # estagio 3 (central)
    callout_src_y = y_main
    ax.plot([callout_src_x, 50], [callout_src_y, 28],
            color=NAVY, lw=1.0, linestyle=(0, (4, 3)))
    ax.plot([callout_src_x, 50], [callout_src_y, 28],
            color=NAVY, lw=0, marker=",", zorder=1)

    # Container do inset (mais largo para acomodar 9 blocos com texto grande)
    inset_x = 3
    inset_y = 9
    inset_w = 94
    inset_h = 16.5
    inset = _rbox(ax, inset_x, inset_y, inset_w, inset_h,
                  (0.985, 0.99, 1.0), ec=NAVY, lw=1.4, dashed=True)
    _text(ax, inset_x + inset_w / 2, inset_y + inset_h - 1.3,
          "Dentro de cada bloco ConvNeXt  "
          "(repetido 3, 3, 9, 3 vezes nos estágios 1–4)",
          size=16, bold=True, italic=True, color=NAVY)

    # Sequencia ESSENCIAL de operacoes dentro do bloco — todos os 9
    # pecas sao arquitetonicas e preservadas. Removidas apenas as
    # anotacoes textuais secundarias (ex. "normaliza por amostra").
    #   Input -> DW 7x7 -> LN -> PW 1x1 up4x -> GELU -> PW 1x1 down1x
    #        -> LayerScale -> + (residual) -> Output
    ops = [
        ("Input",         GRAY_LT,     NAVY,  False),
        ("DW 7×7",        WHITE,       NAVY,  False),
        ("LayerNorm",     WHITE,       NAVY,  False),
        ("PW 1×1  ↑ 4×",  BLUE_LIGHT,  NAVY,  False),
        ("GELU",          WHITE,       NAVY,  False),
        ("PW 1×1  ↓ 1×",  BLUE_LIGHT,  NAVY,  False),
        ("LayerScale",    WHITE,       NAVY,  False),
        ("+",             ACCENT,      WHITE, True),
        ("Output",        GRAY_LT,     NAVY,  False),
    ]
    n_ops = len(ops)
    op_gap = 1.0
    usable_w = inset_w - 2.0
    op_w = (usable_w - (n_ops - 1) * op_gap) / n_ops   # ~9.3
    op_h = 6.5
    op_y = inset_y + 2.6
    op_centers = []
    cur_x = inset_x + 1.0
    import matplotlib.patches as mpatches
    for i, (lbl, fc, tc, is_circle) in enumerate(ops):
        if is_circle:
            cx = cur_x + op_w / 2
            cy = op_y + op_h / 2
            circ_r = min(op_w, op_h) / 2 * 0.85
            c = mpatches.Circle((cx, cy), circ_r,
                                 fc=fc, ec=fc, lw=2.0)
            ax.add_patch(c)
            ax.text(cx, cy, lbl,
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color=tc)
        else:
            _rbox(ax, cur_x, op_y, op_w, op_h, fc, ec=NAVY, lw=1.8)
            _text(ax, cur_x + op_w / 2, op_y + op_h / 2,
                  lbl, size=14, bold=True, color=tc)
        op_centers.append(cur_x + op_w / 2)
        if i < n_ops - 1:
            _arrow(ax, cur_x + op_w,
                   op_y + op_h / 2,
                   cur_x + op_w + op_gap - 0.1,
                   op_y + op_h / 2,
                   lw=1.8, ms=16)
        cur_x += op_w + op_gap

    # Arco de residual (Input -> +)
    input_cx = op_centers[0]
    add_cx = op_centers[7]
    arc_top_y = op_y + op_h + 1.8
    ax.plot([input_cx, input_cx], [op_y + op_h, arc_top_y],
            color=ACCENT, lw=2.2)
    ax.plot([input_cx, add_cx], [arc_top_y, arc_top_y],
            color=ACCENT, lw=2.2)
    _arrow(ax, add_cx, arc_top_y, add_cx, op_y + op_h + 0.3,
           color=ACCENT, lw=2.2, ms=18)
    _text(ax, (input_cx + add_cx) / 2, arc_top_y + 0.9,
          "conexão residual",
          size=13, italic=True, bold=True, color=ACCENT)

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
