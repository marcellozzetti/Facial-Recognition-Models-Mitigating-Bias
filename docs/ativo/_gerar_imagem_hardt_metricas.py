"""Gera slide-imagem standalone das métricas do nosso protocolo.

Saída: docs/ativo/imagens/hardt_metricas_slide.png

Formato 13.33 x 7.5 inches (landscape padrão PPTX), 300 DPI.
Pode ser inserido manualmente em qualquer material PowerPoint.

Estrutura:
- Cenário A (protocolo principal, race apenas): F1 macro, DR, Worst-class F1
- Cenário B (ablation race × gender): EO_h por classe, EOD por classe
- Faixa final: teorema da impossibilidade (Kleinberg 2017)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Paleta consistente com o PPTX e demais diagramas
NAVY = (31/255, 42/255, 78/255)
GRAY_DK = (61/255, 66/255, 78/255)
GRAY_MD = (112/255, 118/255, 130/255)
GRAY_LT = (232/255, 234/255, 237/255)
ACCENT = (192/255, 57/255, 43/255)
GREEN = (46/255, 125/255, 50/255)
WHITE = (1.0, 1.0, 1.0)


def _rounded_box(ax, x, y, w, h, fc=GRAY_LT, ec=NAVY, lw=1.2, rounding=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)


def _rect(ax, x, y, w, h, fc):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=fc, ec=fc))


def _metric_card(ax, x, y, w, h, color, nome, subtitulo, definicao, ref,
                 valor_titulo=None, valor=None, valor_sub=None):
    """Card de métrica com cabeçalho colorido + definição + ref + valor opcional.

    definicao deve já vir com quebras de linha (\n) manuais para caber.
    """
    # Card de fundo
    _rounded_box(ax, x, y, w, h, fc=GRAY_LT, ec=color, lw=1.5)

    # Cabeçalho colorido (60% mais espesso)
    header_h = 0.55
    _rect(ax, x, y + h - header_h, w, header_h, fc=color)
    ax.text(x + w/2, y + h - 0.20, nome,
            ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)
    ax.text(x + w/2, y + h - 0.42, subtitulo,
            ha="center", va="center", fontsize=8.5, color=GRAY_LT, style="italic")

    # Definição (texto com quebras manuais para evitar overflow)
    def_y = y + h - header_h - 0.15
    ax.text(x + 0.18, def_y, definicao,
            ha="left", va="top", fontsize=8.5, color=GRAY_DK)

    # Referência (rodapé do card)
    ref_y = y + (0.85 if valor is not None else 0.20)
    ax.text(x + 0.18, ref_y, ref,
            ha="left", va="center", fontsize=7.5, color=GRAY_MD, style="italic")

    # Valor numérico (se houver) — em bloco compacto no rodapé do card
    if valor is not None:
        # linha separadora
        ax.plot([x + 0.3, x + w - 0.3], [y + 0.78, y + 0.78],
                color=GRAY_MD, lw=0.5, alpha=0.5)
        # valor em destaque alinhado à esquerda; subtítulo do valor compacto à direita
        ax.text(x + w/2, y + 0.55, valor_titulo,
                ha="center", va="center", fontsize=8.5, color=GRAY_MD, style="italic")
        ax.text(x + w/2, y + 0.30, valor,
                ha="center", va="center", fontsize=16, fontweight="bold", color=color)
        if valor_sub:
            ax.text(x + w/2, y + 0.08, valor_sub,
                    ha="center", va="bottom", fontsize=7.5, color=GRAY_DK)


def build_figure(out_path: Path) -> None:
    # Slide PPTX padrão widescreen: 13.33 x 7.5 in
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=300)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # ====== Título ======
    ax.text(0.5, 7.20, "Métricas — triangulação em dois cenários",
            ha="left", va="center", fontsize=20, fontweight="bold", color=NAVY)
    ax.text(0.5, 6.88, "Hardt et al. 2016 (NeurIPS) é a fonte canônica. Aplicamos no nosso protocolo de race classification 7-class.",
            ha="left", va="center", fontsize=10, color=GRAY_MD, style="italic")
    ax.plot([0.5, 12.83], [6.7, 6.7], color=NAVY, lw=2.0)

    # ====== CENÁRIO A — header ======
    ax_y = 6.18
    _rect(ax, 0.5, ax_y, 12.33, 0.35, fc=NAVY)
    ax.text(0.65, ax_y + 0.175, "Cenário A — Protocolo principal",
            ha="left", va="center", fontsize=12, fontweight="bold", color=WHITE)
    ax.text(4.6, ax_y + 0.175, "Y = race (7 classes), sem outro atributo sensível separado",
            ha="left", va="center", fontsize=10, color=GRAY_LT, style="italic")

    # Cards Cenário A
    card_w_a = 3.95
    gap_a = 0.2
    total_a = 3 * card_w_a + 2 * gap_a
    x0_a = (13.33 - total_a) / 2
    y_card_a = 3.6
    h_card_a = 2.45

    metricas_a = [
        {
            "color": NAVY,
            "nome": "F1 macro",
            "subtitulo": "performance média",
            "definicao": "Média harmônica de precision\ne recall por classe, depois\nmédia simples entre as 7 raças.",
            "ref": "van Rijsbergen 1979",
            "valor_titulo": "SOTA atual",
            "valor": "75%",
            "valor_sub": "FaceScanPaliGemma",
        },
        {
            "color": ACCENT,
            "nome": "DR",
            "subtitulo": "Disparity Ratio",
            "definicao": "Razão entre F1 da pior raça e\nF1 da melhor. Em multi-classe\nsem A separado, é proxy\ndireta de EO_h gap.",
            "ref": "Hardt, Price & Srebro 2016 (NeurIPS)",
            "valor_titulo": "Estado atual",
            "valor": "0.67",
            "valor_sub": "60% Latinx ÷ 90% Black",
        },
        {
            "color": GREEN,
            "nome": "Worst-class F1",
            "subtitulo": "pior subgrupo",
            "definicao": "F1 da raça em que o modelo\nerra mais. Garante que\nninguém fique para trás.",
            "ref": "Sagawa et al. 2020 (ICLR) — Group DRO",
            "valor_titulo": "Estado atual",
            "valor": "60%",
            "valor_sub": "Latinx",
        },
    ]

    for i, m in enumerate(metricas_a):
        xc = x0_a + i * (card_w_a + gap_a)
        _metric_card(ax, xc, y_card_a, card_w_a, h_card_a,
                     color=m["color"], nome=m["nome"], subtitulo=m["subtitulo"],
                     definicao=m["definicao"], ref=m["ref"],
                     valor_titulo=m["valor_titulo"], valor=m["valor"],
                     valor_sub=m["valor_sub"])

    # ====== CENÁRIO B — header ======
    bx_y = 3.1
    _rect(ax, 0.5, bx_y, 12.33, 0.35, fc=ACCENT)
    ax.text(0.65, bx_y + 0.175, "Cenário B — Ablation com atributo sensível separado",
            ha="left", va="center", fontsize=12, fontweight="bold", color=WHITE)
    ax.text(6.6, bx_y + 0.175, "Y = race, A = gender → 14 subgrupos race × gender",
            ha="left", va="center", fontsize=10, color=GRAY_LT, style="italic")

    # Cards Cenário B (2 cards centralizados)
    card_w_b = 5.6
    gap_b = 0.35
    total_b = 2 * card_w_b + gap_b
    x0_b = (13.33 - total_b) / 2
    y_card_b = 0.95
    h_card_b = 2.05

    metricas_b = [
        {
            "color": NAVY,
            "nome": "EO_h por classe",
            "subtitulo": "Equal Opportunity",
            "definicao": "Para cada raça c, mede se TPR_c é igual entre M e F.\nA diferença é o gap EO_h da classe c.\nReportamos vetor [gap_c] para as 7 raças.",
            "ref": "Hardt, Price & Srebro 2016 (NeurIPS)",
        },
        {
            "color": ACCENT,
            "nome": "EOD por classe",
            "subtitulo": "Equalized Odds",
            "definicao": "Para cada raça c, TPR_c E FPR_c iguais entre M e F.\nMais restritivo que EO_h.\nReportamos vetor [gap_c] para as 7 raças.",
            "ref": "Hardt, Price & Srebro 2016 (NeurIPS)",
        },
    ]

    for i, m in enumerate(metricas_b):
        xc = x0_b + i * (card_w_b + gap_b)
        _metric_card(ax, xc, y_card_b, card_w_b, h_card_b,
                     color=m["color"], nome=m["nome"], subtitulo=m["subtitulo"],
                     definicao=m["definicao"], ref=m["ref"])

    # ====== Faixa final — Kleinberg ======
    _rounded_box(ax, 0.5, 0.2, 12.33, 0.55, fc=NAVY, ec=NAVY, rounding=0.05)
    ax.text(0.7, 0.55, "Por que 5 métricas?",
            ha="left", va="center", fontsize=11, fontweight="bold", color=WHITE)
    ax.text(3.10, 0.55,
            "Teorema da impossibilidade (Kleinberg, Mullainathan & Raghavan 2017, ITCS): nenhuma métrica única captura fairness — triangular é honesto.",
            ha="left", va="center", fontsize=10, color=GRAY_LT)
    ax.text(6.665, 0.27,
            "Nossa contribuição C4: triangulação multi-classe baseada em Hardt como métrica padrão para race classification 7-class.",
            ha="center", va="center", fontsize=9, color=GRAY_LT, style="italic")

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
