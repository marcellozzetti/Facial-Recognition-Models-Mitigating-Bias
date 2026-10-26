"""Gera figura do pipeline experimental em 6 etapas (Cap. 4 §4.2).

Layout top-down, 1 coluna, com:
    - 3 fases metodologicas na barra lateral esquerda (Diagnostico / Metodo /
      Validacao) - agrupamento visual coerente com o discurso do Cap. 4.
    - 6 caixas de etapa empilhadas, cada uma com:
        * eyebrow "ETAPA N"
        * titulo (em NAVY)
        * descricao (1 linha)
        * meta em 4 colunas: Insumo | Metodo | Saida | Prazo
    - Chip lateral direito com o numero da Contribuicao vinculada (Cap. 3).
    - Etapa 3 (metodo proposto) recebe tratamento de destaque.
    - Setas verticais retas entre etapas.

Saida: docs/tese/images/fig_pipeline_6etapas.png (substitui a versao antiga
horizontal 2x3 com setas cruzadas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Paleta consistente com _gerar_imagem_film.py e o PPTX de reunioes
NAVY = (31 / 255, 42 / 255, 78 / 255)
NAVY_SOFT = (61 / 255, 104 / 255, 187 / 255)
BLUE_TINT = (233 / 255, 238 / 255, 248 / 255)
GRAY_DK = (61 / 255, 66 / 255, 78 / 255)
GRAY_MD = (112 / 255, 118 / 255, 130 / 255)
GRAY_LT = (232 / 255, 234 / 255, 237 / 255)
PAPER = (250 / 255, 250 / 255, 248 / 255)
ACCENT = (192 / 255, 57 / 255, 43 / 255)
GREEN = (46 / 255, 125 / 255, 50 / 255)
AMBER = (198 / 255, 138 / 255, 0 / 255)
WHITE = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# Modelo de dados                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    number: int
    title: str
    subtitle: str        # frase curta em linguagem simples
    highlight: bool = False


STAGES = [
    Stage(1, "Detectar o tom de pele",
          "Classifica cada rosto na escala Monk de 10 tons."),
    Stage(2, "Cruzar tom de pele e raça",
          "Mede a diversidade de tons dentro de cada grupo racial."),
    Stage(3, "Rede que consulta o tom de pele",
          "Classificador de raça que usa o tom como pista de contexto.",
          highlight=True),
    Stage(4, "Comparar com métodos existentes",
          "Confronta a proposta com seis abordagens já publicadas."),
    Stage(5, "Aplicar em reconhecimento facial",
          "Verifica se o ganho transfere para identificação de pessoas."),
    Stage(6, "Analisar a origem do erro",
          "Separa quanto do erro vem de rostos parecidos vs da rede."),
]

# Fases metodológicas: (label, etapas_incluidas, cor_barra)
PHASES = [
    ("A. DIAGNÓSTICO", (1, 2), GRAY_MD),
    ("B. MÉTODO", (3, 3), NAVY),
    ("C. VALIDAÇÃO", (4, 6), GRAY_MD),
]


# --------------------------------------------------------------------------- #
# Primitivas de desenho                                                       #
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, fc, ec, lw=1.2, rounding=0.8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=1.8, style="-|>", mut=16):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle=style, color=color, lw=lw,
            mutation_scale=mut, shrinkA=1, shrinkB=1,
        )
    )


def _text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)


# --------------------------------------------------------------------------- #
# Layout de uma etapa                                                         #
# --------------------------------------------------------------------------- #
def _draw_stage(ax, stage: Stage, x: float, y: float, w: float, h: float) -> None:
    highlighted = stage.highlight
    body_fc = BLUE_TINT if highlighted else PAPER
    body_ec = NAVY if highlighted else GRAY_MD
    body_lw = 2.4 if highlighted else 1.1

    # caixa principal
    _box(ax, x, y, w, h, fc=body_fc, ec=body_ec, lw=body_lw, rounding=1.2)

    # faixa lateral esquerda com numero da etapa
    strip_w = 8.0
    strip_fc = NAVY if highlighted else GRAY_DK
    _box(ax, x, y, strip_w, h, fc=strip_fc, ec=strip_fc, lw=0, rounding=1.2)
    _text(
        ax, x + strip_w / 2, y + h / 2,
        f"{stage.number}",
        color=WHITE, fontsize=30, fontweight="bold",
    )

    # titulo + linha de metodo/prazo
    title_x = x + strip_w + 3.0
    _text(
        ax, title_x, y + h * 0.62, stage.title,
        ha="left", va="center",
        color=NAVY, fontsize=15, fontweight="bold",
    )
    _text(
        ax, title_x, y + h * 0.30,
        stage.subtitle,
        ha="left", va="center",
        color=GRAY_MD, fontsize=10.5, style="italic",
    )


# --------------------------------------------------------------------------- #
# Figura completa                                                             #
# --------------------------------------------------------------------------- #
def build_figure(out_path: Path) -> None:
    # Figura mais compacta agora que cada caixa carrega menos texto
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 115)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------- Título ----------
    _text(
        ax, 50, 110,
        "Pipeline experimental em 6 etapas",
        color=NAVY, fontsize=20, fontweight="bold",
    )

    # ---------- Geometria das etapas ----------
    box_w = 76.0
    box_h = 11.0
    box_x = 16.0
    top_y = 103.0
    gap = 2.6

    positions: dict[int, tuple[float, float]] = {}
    for i, stage in enumerate(STAGES):
        y = top_y - i * (box_h + gap) - box_h
        _draw_stage(ax, stage, box_x, y, box_w, box_h)
        positions[stage.number] = (box_x, y)

    # ---------- Setas verticais entre etapas ----------
    for i in range(1, len(STAGES)):
        top_prev = positions[i][1]
        bot_next = positions[i + 1][1] + box_h
        cx = box_x + box_w / 2
        _arrow(ax, cx, top_prev - 0.3, cx, bot_next + 0.4,
               color=NAVY, lw=1.8, mut=14)

    # ---------- Barra lateral de fases (esquerda) ----------
    for label, (start, end), color in PHASES:
        y_top = positions[start][1] + box_h
        y_bot = positions[end][1]
        bar_x = 5.0
        bar_w = 6.5
        _box(ax, bar_x, y_bot, bar_w, y_top - y_bot,
             fc=color, ec=color, lw=0, rounding=1.0)
        cy = (y_top + y_bot) / 2
        ax.text(
            bar_x + bar_w / 2, cy, label,
            rotation=90, ha="center", va="center",
            color=WHITE, fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = repo_root / "docs" / "tese" / "images" / "fig_pipeline_6etapas.png"
    build_figure(out)


if __name__ == "__main__":
    main()
