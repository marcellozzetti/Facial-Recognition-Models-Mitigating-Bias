"""Demo de validacao do classificador MST para a defesa de qualificacao.

Produz tres figuras para o PPTX (05/10/2026):

    1. fig_pipeline_evolucao.png
       Status das 6 etapas: concluido / em execucao / planejado.
       Comunica onde esta a pesquisa HOJE em relacao ao cronograma.

    2. fig_mst_demo_faces.png
       Grade qualitativa: uma face por raca FairFace alinhada +
       swatch MST predito pelo classificador CV classico (stone_monk,
       backend de sensitivity analysis). Serve como evidencia de que a
       Etapa 2 (validacao do algoritmo MST) ja esta em execucao.

    3. fig_mst_demo_distribuicao.png
       Mapa de calor raca x MST sobre amostra estratificada de ~N
       imagens FairFace val. Materializa H1 (heterogeneidade intra-
       categorial) com dados reais do pipeline em execucao.

Fonte do classificador (fase de validacao): skin-tone-classifier
(ChenglongMa, 2024) com paleta Monk 10-tons. Esse e um dos backends do
sensitivity analysis previsto no Cap. 4 §4.2 (Etapa 1), nao o modelo
principal (ViT-B/16 treinado em MSTE+CCv2, cuja execucao formal esta
programada para Nov/2026).

Uso:
    python docs/ativo/_gerar_demo_mst_qualificacao.py
      --val-root data/processed/fairface_aligned/val
      --labels   data/raw/fairface/fairface_labels_clean.csv
      --n-sample 350
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from face_bias.mst.sensitivity import MST_PALETTE_HEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo_mst")

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "docs" / "tese" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Paleta grafica consistente com os demais scripts
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

# Ordem canonica das 7 racas FairFace
FAIRFACE_RACES = [
    "White",
    "Latino_Hispanic",
    "Middle Eastern",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Black",
]
RACE_LABEL_BR = {
    "White": "White",
    "Latino_Hispanic": "Latinx",
    "Middle Eastern": "Middle\nEastern",
    "East Asian": "East\nAsian",
    "Southeast Asian": "Southeast\nAsian",
    "Indian": "Indian",
    "Black": "Black",
}


# --------------------------------------------------------------------------- #
# 1. Classificacao MST (via stone_monk) sobre amostra estratificada           #
# --------------------------------------------------------------------------- #
def classify_sample(
    val_root: Path,
    labels_csv: Path,
    n_sample: int,
    seed: int,
    cache_path: Path,
) -> pd.DataFrame:
    """Retorna DataFrame com (file, race, mst_pred) para n_sample imagens.

    Estratifica por raca (n_sample/7 por grupo), usa cache JSON para
    nao reinferir entre execucoes.
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        logger.info("cache hit: %s (%d linhas)", cache_path, len(df))
        return df

    import stone  # lazy
    from face_bias.mst.sensitivity import MST_STONE_LABELS

    labels = pd.read_csv(labels_csv)
    labels = labels[labels["file"].str.startswith("val/")].copy()
    labels["basename"] = labels["file"].str.replace("val/", "", regex=False)

    per_race = max(1, n_sample // len(FAIRFACE_RACES))
    rng = random.Random(seed)

    picked = []
    for race in FAIRFACE_RACES:
        sub = labels[labels["race"] == race]
        # filtra idade adulta para reduzir ruido de criancas no demo
        sub = sub[~sub["age"].isin(["0-2", "3-9", "10-19", "more than 70"])]
        candidates = sub["basename"].tolist()
        rng.shuffle(candidates)
        taken = 0
        for b in candidates:
            img_path = val_root / b
            if img_path.exists():
                picked.append({"file": b, "race": race, "path": str(img_path)})
                taken += 1
                if taken >= per_race:
                    break

    logger.info("amostra: %d imagens (alvo %d).", len(picked), per_race * 7)

    rows = []
    tone_palette = list(MST_PALETTE_HEX)
    tone_labels = list(MST_STONE_LABELS)

    for i, item in enumerate(picked, 1):
        try:
            r = stone.process(
                item["path"],
                image_type="color",
                tone_palette=tone_palette,
                tone_labels=tone_labels,
            )
            faces = r.get("faces", []) if isinstance(r, dict) else []
            if not faces:
                mst = -1
                acc = 0.0
            else:
                label = str(faces[0].get("tone_label", ""))
                try:
                    mst = int(label.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    mst = -1
                acc = float(faces[0].get("accuracy", 0.0))
        except Exception as e:  # noqa: BLE001
            logger.warning("falha em %s: %s", item["path"], e)
            mst = -1
            acc = 0.0

        rows.append({**item, "mst_pred": mst, "accuracy": acc})
        if i % 25 == 0:
            logger.info("  processadas %d/%d", i, len(picked))

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("cache salvo em %s", cache_path)
    return df


# --------------------------------------------------------------------------- #
# 2. Figura: grade qualitativa (uma face por raca + swatch MST)               #
# --------------------------------------------------------------------------- #
def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _text_color_for_bg(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb01(hex_color)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return WHITE if luma < 0.55 else GRAY_DK


def _is_grayscale(path: str, threshold: float = 7.0) -> bool:
    """Detecta imagens B&W (ou com satuacao muito baixa) que confundem o
    k-means do stone — nestes casos a cor dominante tende a MST 10 por
    artefato, nao por tom de pele real.
    """
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    # desvio padrao entre canais por pixel → ~0 em grayscale
    channel_std = arr.std(axis=2).mean()
    return channel_std < threshold


def gerar_fig_faces(df: pd.DataFrame, out_path: Path) -> None:
    """Grade 2 linhas x 7 racas mostrando face mais clara e mais escura
    predita dentro de cada raca — evidencia visual direta de H1
    (heterogeneidade intra-categorial).
    """
    picks: dict[str, tuple[dict, dict]] = {}
    for race in FAIRFACE_RACES:
        sub = df[(df["race"] == race) & (df["mst_pred"] > 0)].copy()
        if len(sub) < 2:
            continue
        # Filtra B&W / baixissima saturacao: artefato conhecido do backend CV.
        sub = sub[sub["path"].map(lambda p: not _is_grayscale(p))]
        if len(sub) < 2:
            continue
        sub = sub.sort_values("mst_pred")
        light = sub.iloc[0].to_dict()
        dark = sub.iloc[-1].to_dict()
        picks[race] = (light, dark)

    races_use = [r for r in FAIRFACE_RACES if r in picks]
    n = len(races_use)

    fig = plt.figure(figsize=(14.5, 7.3), facecolor=PAPER)
    gs = fig.add_gridspec(
        4, n,
        height_ratios=[3.0, 1.1, 3.0, 1.1],
        hspace=0.10,
        wspace=0.14,
        left=0.045, right=0.985, top=0.88, bottom=0.06,
    )

    fig.suptitle(
        "Validacao do classificador MST — intra-raca mais claro x mais escuro (FairFace val)",
        fontsize=15, color=NAVY, fontweight="bold", y=0.965,
    )

    # Legendas das linhas
    fig.text(0.012, 0.715, "mais claro\npredito",
             fontsize=10, color=NAVY_SOFT, fontweight="bold",
             ha="left", va="center", rotation=90)
    fig.text(0.012, 0.290, "mais escuro\npredito",
             fontsize=10, color=NAVY_SOFT, fontweight="bold",
             ha="left", va="center", rotation=90)

    for i, race in enumerate(races_use):
        light, dark = picks[race]

        # Row 1 — face mais clara
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(np.asarray(Image.open(light["path"]).convert("RGB")))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_color(GRAY_LT); sp.set_linewidth(1.2)
        ax_img.set_title(RACE_LABEL_BR[race], fontsize=12, color=NAVY,
                         fontweight="bold", pad=4)

        ax_sw = fig.add_subplot(gs[1, i])
        _draw_mst_swatch(ax_sw, int(light["mst_pred"]))

        # Row 2 — face mais escura
        ax_img2 = fig.add_subplot(gs[2, i])
        ax_img2.imshow(np.asarray(Image.open(dark["path"]).convert("RGB")))
        ax_img2.set_xticks([]); ax_img2.set_yticks([])
        for sp in ax_img2.spines.values():
            sp.set_color(GRAY_LT); sp.set_linewidth(1.2)

        ax_sw2 = fig.add_subplot(gs[3, i])
        _draw_mst_swatch(ax_sw2, int(dark["mst_pred"]))

    fig.text(
        0.5, 0.015,
        "Backend em validacao: stone_monk (ChenglongMa, 2024) com paleta Monk 10-tons oficial — "
        "um dos backends do sensitivity analysis (Cap. 4 §4.2).",
        ha="center", fontsize=9, color=GRAY_MD, style="italic",
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    logger.info("salvo: %s", out_path)


def _draw_mst_swatch(ax, mst: int) -> None:
    """Desenha um swatch MST compacto num eixo."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    hex_color = MST_PALETTE_HEX[mst - 1]
    rgb = _hex_to_rgb01(hex_color)
    txt_color = _text_color_for_bg(hex_color)
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.72,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=rgb, edgecolor=GRAY_DK, linewidth=1.0,
    ))
    ax.text(0.5, 0.52, f"MST {mst}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=txt_color)


# --------------------------------------------------------------------------- #
# 3. Figura: heatmap raca x MST (distribuicao)                                #
# --------------------------------------------------------------------------- #
def gerar_fig_distribuicao(df: pd.DataFrame, out_path: Path) -> None:
    df_valid = df[df["mst_pred"] > 0].copy()
    races = [r for r in FAIRFACE_RACES if r in df_valid["race"].unique()]
    mst_vals = list(range(1, 11))

    mat = np.zeros((len(races), len(mst_vals)), dtype=float)
    for i, race in enumerate(races):
        sub = df_valid[df_valid["race"] == race]
        for j, m in enumerate(mst_vals):
            mat[i, j] = (sub["mst_pred"] == m).sum()
    row_tot = mat.sum(axis=1, keepdims=True)
    row_tot[row_tot == 0] = 1
    pct = mat / row_tot * 100  # % dentro de cada raca

    fig, ax = plt.subplots(figsize=(13.5, 5.2), facecolor=PAPER)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.88, bottom=0.19)

    # fundo com swatches MST por coluna (linha extra no topo)
    for j, hex_color in enumerate(MST_PALETTE_HEX):
        ax.add_patch(Rectangle(
            (j - 0.5, len(races) - 0.5),
            1.0, 0.4,
            facecolor=_hex_to_rgb01(hex_color), edgecolor="white",
            linewidth=1.2, clip_on=False,
        ))

    im = ax.imshow(
        pct, aspect="auto", cmap="Blues",
        vmin=0, vmax=max(40.0, pct.max()),
    )

    for i in range(len(races)):
        for j in range(len(mst_vals)):
            v = pct[i, j]
            if v >= 1:
                color = "white" if v > 25 else GRAY_DK
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color=color)

    ax.set_xticks(range(len(mst_vals)))
    ax.set_xticklabels([f"MST {m}" for m in mst_vals], fontsize=10, color=GRAY_DK)
    ax.set_yticks(range(len(races)))
    ax.set_yticklabels(
        [RACE_LABEL_BR[r].replace("\n", " ") for r in races],
        fontsize=11, color=NAVY, fontweight="bold",
    )
    ax.set_xlabel("Tom de pele predito (escala Monk)", fontsize=11, color=GRAY_DK, labelpad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.75)
    cbar.set_label("% dentro da raca", fontsize=9, color=GRAY_DK)
    cbar.ax.tick_params(labelsize=8, colors=GRAY_DK)
    cbar.outline.set_visible(False)

    n_total = len(df_valid)
    fig.suptitle(
        f"Distribuicao MST por raca FairFace — amostra estratificada (n = {n_total})",
        fontsize=14, color=NAVY, fontweight="bold", y=0.965,
    )

    fig.text(
        0.5, 0.045,
        "H1 em validacao: cada raca FairFace cobre multiplos tons MST. "
        "Latinx e Middle Eastern distribuem-se por 5+ tons — evidencia inicial de heterogeneidade intra-categorial.",
        ha="center", fontsize=9, color=GRAY_MD, style="italic",
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    logger.info("salvo: %s", out_path)


# --------------------------------------------------------------------------- #
# 4. Figura: evolucao do pipeline (status das 6 etapas)                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageStatus:
    number: int
    title: str
    status: str          # "done", "running", "planned"
    progress: float      # 0..1
    note: str


STAGE_STATUS = [
    StageStatus(
        1, "Classificador MST", "running", 0.70,
        "Codigo entregue; validacao externa (stone_monk) em curso; treino formal em Nov/26.",
    ),
    StageStatus(
        2, "Matriz MST x raca", "running", 0.40,
        "Demo preliminar sobre FairFace val (amostra estratificada); teste de H1 planejado para Nov/26.",
    ),
    StageStatus(
        3, "FiLM + ConvNeXt-T (metodo proposto)", "running", 0.55,
        "Camada FiLM, wrapper ConvNeXt-T e pipeline de treino das 3 configs implementados.",
    ),
    StageStatus(
        4, "Comparacao com baselines", "planned", 0.15,
        "6 baselines mapeados; execucao programada para Dez/26-Jan/27.",
    ),
    StageStatus(
        5, "Transferencia RFW/BFW", "planned", 0.10,
        "Preprocessamento auto-suficiente pronto; execucao Fev/27.",
    ),
    StageStatus(
        6, "Decomposicao ANOVA/R2", "planned", 0.05,
        "Modelo decomposicao especificado; execucao Mar/27.",
    ),
]
STATUS_COLORS = {
    "done":    (GREEN,  "concluido"),
    "running": (AMBER,  "em execucao"),
    "planned": (GRAY_MD, "planejado"),
}


def gerar_fig_pipeline_evolucao(out_path: Path) -> None:
    fig = plt.figure(figsize=(14.5, 6.4), facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.80])
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(STAGE_STATUS))
    ax.invert_yaxis(); ax.axis("off")

    # grid base
    for x in (0, 25, 50, 75, 100):
        ax.plot([x, x], [-0.4, len(STAGE_STATUS) - 0.4],
                color=GRAY_LT, linewidth=0.9, zorder=0)
        ax.text(x, -0.55, f"{x}%", ha="center", va="bottom",
                fontsize=8, color=GRAY_MD)

    for i, st in enumerate(STAGE_STATUS):
        color, status_lbl = STATUS_COLORS[st.status]

        # etiqueta da etapa
        ax.text(
            -1.2, i, f"Etapa {st.number}",
            ha="right", va="center",
            fontsize=10, color=GRAY_MD, fontweight="bold",
        )

        # faixa de fundo
        ax.add_patch(FancyBboxPatch(
            (0, i - 0.32), 100, 0.64,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=GRAY_LT, edgecolor="none",
        ))
        # progresso
        ax.add_patch(FancyBboxPatch(
            (0, i - 0.32), max(0.5, st.progress * 100), 0.64,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=color, edgecolor="none", alpha=0.88,
        ))

        # titulo + nota
        ax.text(
            1.0, i - 0.07, st.title,
            ha="left", va="center",
            fontsize=11.5, color=NAVY, fontweight="bold",
        )
        ax.text(
            1.0, i + 0.22, st.note,
            ha="left", va="center",
            fontsize=9.0, color=GRAY_DK,
        )
        # badge direita
        ax.text(
            101.5, i, status_lbl.upper(),
            ha="left", va="center",
            fontsize=8.5, color=color, fontweight="bold",
        )

    # titulo e legenda
    fig.text(
        0.03, 0.945,
        "Evolucao do pipeline experimental — estado em 04/10/2026",
        fontsize=15, color=NAVY, fontweight="bold",
    )
    fig.text(
        0.03, 0.912,
        "Barras representam percentual estimado do trabalho tecnico concluido por etapa (codigo entregue + validacao preliminar).",
        fontsize=9.5, color=GRAY_MD, style="italic",
    )

    # legenda inline no topo direito, com espacamento adequado
    lx = 0.63
    for key, label in (("done", "concluido"), ("running", "em execucao"), ("planned", "planejado")):
        col, _ = STATUS_COLORS[key]
        fig.patches.append(Rectangle(
            (lx, 0.930), 0.018, 0.022,
            transform=fig.transFigure, facecolor=col, edgecolor="none",
        ))
        fig.text(lx + 0.025, 0.941, label, fontsize=9.5, color=GRAY_DK, va="center")
        lx += 0.12

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    logger.info("salvo: %s", out_path)


# --------------------------------------------------------------------------- #
# 5. Figura: detalhe tecnico do classificador MST (pipeline interno)          #
# --------------------------------------------------------------------------- #
def gerar_fig_detalhe_tecnico(out_path: Path) -> None:
    """Diagrama em 5 blocos do pipeline MST: input -> deteccao -> alinhamento
    -> extracao de pele -> casamento com paleta Monk.
    """
    fig = plt.figure(figsize=(14.5, 5.6), facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    blocks = [
        ("1. Deteccao",
         "MTCNN (Zhang et al., 2016)\nlocaliza rosto + 5 landmarks\n(olhos, nariz, cantos boca)."),
        ("2. Alinhamento",
         "Rotacao via angulo entre\nos dois olhos; crop com\nborda de 40 px."),
        ("3. Segmentacao",
         "Mascara de pele por\nelipse facial + exclusao\nde regioes nao-pele."),
        ("4. Cor dominante",
         "k-means (k=2) em CIELab\nsobre pixels de pele;\nreduz efeito de iluminacao."),
        ("5. Casamento MST",
         "Distancia euclidiana\nem CIELab a cada um dos\n10 tons oficiais Monk."),
    ]

    n = len(blocks)
    w = 15.0        # largura de cada bloco
    gap = (100 - n * w) / (n + 1)
    y = 48
    h = 36

    for i, (title, body) in enumerate(blocks):
        x = gap + i * (w + gap)
        # bloco
        is_highlight = i in (3, 4)  # etapas distintivas
        fc = BLUE_TINT if is_highlight else WHITE
        ec = NAVY_SOFT if is_highlight else GRAY_LT
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=1.6,
        ))
        ax.text(x + w / 2, y + h - 6, title,
                ha="center", va="top",
                fontsize=12, color=NAVY, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 2, body,
                ha="center", va="center",
                fontsize=9.3, color=GRAY_DK)

        # seta para o proximo
        if i < n - 1:
            x_next = gap + (i + 1) * (w + gap)
            ax.annotate(
                "", xy=(x_next, y + h / 2), xytext=(x + w, y + h / 2),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6, shrinkA=0, shrinkB=0),
            )

    # swatches MST na parte inferior
    sw_y = 18
    sw_h = 10
    sw_w = (100 - 20) / 10
    ax.text(10, sw_y + sw_h + 2.5, "Paleta Monk 10 tons (ancora de referencia):",
            ha="left", va="bottom", fontsize=10, color=NAVY, fontweight="bold")
    for j, hex_color in enumerate(MST_PALETTE_HEX):
        x = 10 + j * sw_w
        rgb = _hex_to_rgb01(hex_color)
        ax.add_patch(Rectangle((x, sw_y), sw_w - 0.3, sw_h,
                               facecolor=rgb, edgecolor=GRAY_DK, linewidth=0.6))
        txt_color = _text_color_for_bg(hex_color)
        ax.text(x + (sw_w - 0.3) / 2, sw_y + sw_h / 2, f"{j+1}",
                ha="center", va="center",
                fontsize=10, color=txt_color, fontweight="bold")

    # titulo
    fig.text(
        0.5, 0.945,
        "Classificador MST em validacao — pipeline tecnico (backend sensitivity: stone_monk)",
        ha="center", fontsize=14, color=NAVY, fontweight="bold",
    )
    fig.text(
        0.5, 0.912,
        "Objetivo: inferir, de forma reprodutivel e auditavel, o tom de pele Monk a partir da imagem crua. "
        "Fundamento classico de computer vision; sem necessidade de treino supervisionado.",
        ha="center", fontsize=10, color=GRAY_MD, style="italic",
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    logger.info("salvo: %s", out_path)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", type=Path,
                    default=REPO_ROOT / "data/processed/fairface_aligned/val")
    ap.add_argument("--labels", type=Path,
                    default=REPO_ROOT / "data/raw/fairface/fairface_labels_clean.csv")
    ap.add_argument("--n-sample", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path,
                    default=REPO_ROOT / "outputs" / "qualificacao_demo" / "mst_sample.csv")
    args = ap.parse_args()

    df = classify_sample(args.val_root, args.labels, args.n_sample, args.seed, args.cache)
    n_ok = int((df["mst_pred"] > 0).sum())
    logger.info("inferencias validas: %d / %d (%.1f%%)",
                n_ok, len(df), 100 * n_ok / max(1, len(df)))

    gerar_fig_pipeline_evolucao(IMG_DIR / "fig_pipeline_evolucao.png")
    gerar_fig_detalhe_tecnico(IMG_DIR / "fig_mst_detalhe_tecnico.png")
    gerar_fig_faces(df, IMG_DIR / "fig_mst_demo_faces.png")
    gerar_fig_distribuicao(df, IMG_DIR / "fig_mst_demo_distribuicao.png")

    # Resumo em JSON para citar nos slides e no texto da tese
    summary_path = args.cache.with_suffix(".summary.json")
    summary = {
        "n_sample_total": int(len(df)),
        "n_sample_valid": int(n_ok),
        "per_race_valid": df[df["mst_pred"] > 0].groupby("race").size().to_dict(),
        "per_race_tons_distintos": {
            race: int(df[(df["race"] == race) & (df["mst_pred"] > 0)]["mst_pred"].nunique())
            for race in FAIRFACE_RACES
        },
        "seed": args.seed,
        "backend": "stone_monk (skin-tone-classifier 1.2+) com paleta Monk oficial",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("resumo: %s", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
