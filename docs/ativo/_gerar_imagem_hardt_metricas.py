"""Gera slide explicativo sobre as métricas de Hardt (EO_h e EOD) adaptadas
ao nosso caso de race classification multi-classe.

Saída: docs/ativo/imagens/hardt_metricas_slide.png

Formato 13.33 x 7.5 inches (landscape padrão PPTX), 300 DPI.
Pode ser inserido manualmente no PowerPoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Paleta consistente com o PPTX e demais diagramas
NAVY = (31/255, 42/255, 78/255)
GRAY_DK = (61/255, 66/255, 78/255)
GRAY_MD = (112/255, 118/255, 130/255)
GRAY_LT = (232/255, 234/255, 237/255)
ACCENT = (192/255, 57/255, 43/255)
GREEN = (46/255, 125/255, 50/255)
WHITE = (1.0, 1.0, 1.0)


def _box(ax, x, y, w, h, fc=GRAY_LT, ec=NAVY, lw=1.2, rounding=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)


def build_figure(out_path: Path) -> None:
    # Slide PPTX padrão widescreen: 13.33 x 7.5 in
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=300)
    ax.set_xlim(0, 133.3)
    ax.set_ylim(0, 75)
    ax.axis("off")

    # ====== Título e linha ======
    ax.text(4, 71, "Métricas de Hardt (2016) no nosso protocolo",
            ha="left", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(4, 67, "Como Equal Opportunity (EO_h) e Equalized Odds (EOD) se encaixam em race classification 7-class",
            ha="left", va="center", fontsize=11, color=GRAY_MD, style="italic")
    ax.plot([4, 129], [64.5, 64.5], color=NAVY, lw=2.2)

    # ====== COLUNA ESQUERDA — Definições matemáticas ======
    col1_x = 4
    col1_w = 60

    # Cabeçalho coluna 1
    ax.text(col1_x, 60.5, "As definições originais (binárias)",
            ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)

    # Box EO_h
    _box(ax, col1_x, 47.5, col1_w, 11.5, fc=GRAY_LT, ec=NAVY, lw=1.3)
    ax.text(col1_x + 1, 57, "Equal Opportunity (EO_h)",
            ha="left", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(col1_x + 1, 54.2,
            "Igual True Positive Rate entre grupos demográficos.",
            ha="left", va="center", fontsize=10, color=GRAY_DK)
    ax.text(col1_x + 1, 51.5,
            "P(Ŷ = 1 | A = 0, Y = 1)   =   P(Ŷ = 1 | A = 1, Y = 1)",
            ha="left", va="center", fontsize=11, color=NAVY, fontweight="bold", family="monospace")
    ax.text(col1_x + 1, 48.8,
            "Em palavras: 'o modelo é igualmente bom em identificar quem realmente",
            ha="left", va="center", fontsize=9, color=GRAY_DK, style="italic")
    ax.text(col1_x + 1, 47.0,
            "é positivo, independente do grupo demográfico a que pertence'.",
            ha="left", va="center", fontsize=9, color=GRAY_DK, style="italic")

    # Box EOD
    _box(ax, col1_x, 33, col1_w, 12.5, fc=GRAY_LT, ec=NAVY, lw=1.3)
    ax.text(col1_x + 1, 43.5, "Equalized Odds (EOD)",
            ha="left", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(col1_x + 1, 40.5,
            "Igual TPR e igual FPR entre grupos. Mais restritivo que EO_h.",
            ha="left", va="center", fontsize=10, color=GRAY_DK)
    ax.text(col1_x + 1, 37.8,
            "P(Ŷ = 1 | A = 0, Y = y)   =   P(Ŷ = 1 | A = 1, Y = y),   ∀ y ∈ {0, 1}",
            ha="left", va="center", fontsize=11, color=NAVY, fontweight="bold", family="monospace")
    ax.text(col1_x + 1, 35.1,
            "Em palavras: 'mesma taxa de acerto E mesma taxa de erro entre",
            ha="left", va="center", fontsize=9, color=GRAY_DK, style="italic")
    ax.text(col1_x + 1, 33.4,
            "grupos demográficos'. Mais difícil de satisfazer.",
            ha="left", va="center", fontsize=9, color=GRAY_DK, style="italic")

    # Box "Por que importam"
    _box(ax, col1_x, 20, col1_w, 11, fc=WHITE, ec=GREEN, lw=1.5)
    ax.text(col1_x + 1, 28.5, "Por que essas métricas são importantes",
            ha="left", va="center", fontsize=12, fontweight="bold", color=GREEN)
    pq_lines = [
        "Substituem demographic parity, que força desvio do Bayes-optimal",
        "mesmo quando o ground truth é distribuído uniformemente.",
        "São vocabulário canônico em fair ML desde NeurIPS 2016.",
        "Toda literatura subsequente (LAFTR, FSCL+, Group DRO, U-FaTE)",
        "as adota como referência.",
    ]
    for i, ln in enumerate(pq_lines):
        ax.text(col1_x + 1, 26.5 - i*1.5, "•  " + ln,
                ha="left", va="center", fontsize=9.5, color=GRAY_DK)

    # ====== COLUNA DIREITA — Adaptação multi-classe ======
    col2_x = 69
    col2_w = 60

    ax.text(col2_x, 60.5, "Como aplicamos em race 7-class",
            ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)

    # Cenário A
    _box(ax, col2_x, 47.5, col2_w, 11.5, fc=GRAY_LT, ec=NAVY, lw=1.3)
    ax.text(col2_x + 1, 57, "Cenário A — Race como única dimensão",
            ha="left", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(col2_x + 1, 54.5,
            "Y = race (7 classes), sem outro atributo sensível separado.",
            ha="left", va="center", fontsize=10, color=GRAY_DK)
    ax.text(col2_x + 1, 51.7,
            "Reportamos:",
            ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)
    a_lines = [
        "F1 por classe + DR = max(F1) / min(F1)",
        "Worst-class F1 (a pior raça)",
        "DR é descendente direto de EO_h em multi-classe sem A separado",
    ]
    for i, ln in enumerate(a_lines):
        ax.text(col2_x + 1, 49.7 - i*1.5, "•  " + ln,
                ha="left", va="center", fontsize=9, color=GRAY_DK)

    # Cenário B
    _box(ax, col2_x, 33, col2_w, 12.5, fc=GRAY_LT, ec=NAVY, lw=1.3)
    ax.text(col2_x + 1, 43.5, "Cenário B — Race × atributo sensível separado",
            ha="left", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(col2_x + 1, 41,
            "Y = race, A = gender (ou age) → 14 subgrupos (race × gender).",
            ha="left", va="center", fontsize=10, color=GRAY_DK)
    ax.text(col2_x + 1, 38.2,
            "Aqui EO_h e EOD recuperam o significado completo:",
            ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)
    b_lines = [
        "EO_h por classe: TPR igual entre M e F dentro de cada raça",
        "EOD por classe: TPR E FPR iguais (mais estrito)",
        "Reportados em ablation com atributo sensível separado",
    ]
    for i, ln in enumerate(b_lines):
        ax.text(col2_x + 1, 36.2 - i*1.5, "•  " + ln,
                ha="left", va="center", fontsize=9, color=GRAY_DK)

    # Box exemplo numérico do FairFace atual
    _box(ax, col2_x, 20, col2_w, 11, fc=WHITE, ec=ACCENT, lw=1.5)
    ax.text(col2_x + 1, 28.5, "Exemplo numérico — estado atual",
            ha="left", va="center", fontsize=12, fontweight="bold", color=ACCENT)
    ax.text(col2_x + 1, 26,
            "FaceScanPaliGemma sobre FairFace 7-class (AlDahoul 2026, Tabela 16):",
            ha="left", va="center", fontsize=9.5, color=GRAY_DK)
    ax.text(col2_x + 1, 24,
            "F1 macro = 75%      F1 Black = 90%      F1 Latinx = 60%",
            ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.text(col2_x + 1, 22.3,
            "DR = 0.60 / 0.90 = 0.67          worst-class F1 = 60%",
            ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.text(col2_x + 1, 20.7,
            "Como sem A separado, DR é proxy direta de EO_h gap macro.",
            ha="left", va="center", fontsize=9, color=GRAY_DK, style="italic")

    # ====== FAIXA INFERIOR — Triangulação completa ======
    _box(ax, 4, 8, 125.3, 9.5, fc=NAVY, ec=NAVY, lw=1.0)
    ax.text(66.5, 15.5, "Triangulação completa que vamos reportar",
            ha="center", va="center", fontsize=13, fontweight="bold", color=WHITE)
    ax.text(66.5, 13.2,
            "F1 macro    +    DR = max/min F1    +    worst-class F1    +    EO_h por classe (Cenário B)    +    EOD por classe (Cenário B)",
            ha="center", va="center", fontsize=11, color=WHITE, fontweight="bold")
    ax.text(66.5, 10.8,
            "Reportar 5 simultaneamente é a forma honesta de comunicar trade-offs, conforme teorema da impossibilidade (Kleinberg, Mullainathan & Raghavan, ITCS 2017).",
            ha="center", va="center", fontsize=10, color=GRAY_LT, style="italic")

    # Footer
    ax.text(66.5, 4.5,
            "Hardt, Price & Srebro 2016 (NeurIPS) é fonte canônica das métricas EO_h e EOD. Nossa contribuição C4: triangulação multi-classe.",
            ha="center", va="center", fontsize=9, color=GRAY_MD, style="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    out = Path(__file__).resolve().parent / "imagens" / "hardt_metricas_slide.png"
    build_figure(out)


if __name__ == "__main__":
    main()
