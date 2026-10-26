"""Gera apresentação pós-qualificação para reunião de orientação (26/10/2026).

14 slides, formato 16:9, ~20-25 min de fala.
Consolida feedback da banca, pipeline 3 etapas/2 metas,
título revisado e cronograma dos próximos 6 meses.

Uso:
    python docs/ativo/_gerar_apresentacao_pos_qualificacao_2026-10-26.py
    -> produz: docs/ativo/material_pos_qualificacao_2026-10-26.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

# Paleta consistente com todos os scripts do repo
NAVY = RGBColor(0x1F, 0x2A, 0x4E)
BLUE_MID = RGBColor(0x3D, 0x68, 0xBB)
BLUE_LIGHT = RGBColor(0xB4, 0xC4, 0xE8)
GRAY_DK = RGBColor(0x3D, 0x42, 0x4E)
GRAY_MD = RGBColor(0x70, 0x76, 0x82)
GRAY_LT = RGBColor(0xE8, 0xEA, 0xED)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
AMBER = RGBColor(0xF5, 0xB7, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TOTAL_SLIDES = 14

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "docs" / "tese" / "images"
OUT_PATH = Path(__file__).parent / "material_pos_qualificacao_2026-10-26.pptx"

FOOTER_TEXT = (
    "Reunião de orientação · Pós-qualificação · Marcello Ozzetti · "
    "Orientação: Prof. Dr. Marcos G. Quiles · PPG-CC UNIFESP/ICT"
)


# ============================================================
# Helpers
# ============================================================
def _blank(prs: Presentation):
    return prs.slide_layouts[6]


def add_page_number(slide, number: int) -> None:
    tb = slide.shapes.add_textbox(Inches(12.6), Inches(7.05), Inches(0.7), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{number} / {TOTAL_SLIDES}"
    p.font.size = Pt(9)
    p.font.color.rgb = GRAY_MD
    p.font.italic = True


def add_title(slide, text: str) -> None:
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.5), Inches(1.0))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.04)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()


def add_footer(slide, text: str = FOOTER_TEXT) -> None:
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.0), Inches(0.4))
    pf = ft.text_frame.paragraphs[0]
    pf.text = text
    pf.font.size = Pt(9)
    pf.font.color.rgb = GRAY_MD
    pf.font.italic = True


def add_bullets(prs: Presentation, number: int, title: str, bullets: list) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.55), Inches(12.5), Inches(5.4))
    tf = tx.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            head, body = item
            r1 = p.add_run()
            r1.text = head + "  "
            r1.font.size = Pt(18)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(18)
            r2.font.color.rgb = GRAY_DK
        else:
            p.text = "•  " + item
            p.font.size = Pt(18)
            p.font.color.rgb = GRAY_DK
        p.space_after = Pt(10)
    add_footer(slide)
    add_page_number(slide, number)


def add_two_col(prs: Presentation, number: int, title: str,
                left_title: str, left_items: list,
                right_title: str, right_items: list) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)

    for col_x, col_title, items, color in [
        (0.5, left_title, left_items, NAVY),
        (6.9, right_title, right_items, BLUE_MID),
    ]:
        hdr = slide.shapes.add_textbox(Inches(col_x), Inches(1.55), Inches(5.8), Inches(0.5))
        ph = hdr.text_frame.paragraphs[0]
        ph.text = col_title
        ph.font.size = Pt(16)
        ph.font.bold = True
        ph.font.color.rgb = color

        tx = slide.shapes.add_textbox(Inches(col_x), Inches(2.1), Inches(5.8), Inches(4.6))
        tf = tx.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "•  " + item
            p.font.size = Pt(15)
            p.font.color.rgb = GRAY_DK
            p.space_after = Pt(8)

    add_footer(slide)
    add_page_number(slide, number)


def add_table_slide(
    prs: Presentation,
    number: int,
    title: str,
    headers: list,
    rows: list,
    col_widths: list | None = None,
    highlight_rows: list | None = None,
    font_size: int = 12,
) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)

    n_cols = len(headers)
    n_rows = len(rows) + 1
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(12.5)
    height = Inches(5.2)

    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(15)
                run.font.bold = True
                run.font.color.rgb = WHITE

    highlight_rows = highlight_rows or []
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri + 1, ci)
            cell.text = str(val)
            if ri in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = GRAY_LT
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = GRAY_DK
    add_footer(slide)
    add_page_number(slide, number)


def add_kpi_row(slide, kpis: list[tuple[str, str, RGBColor]],
                y_in: float = 2.0, box_h: float = 1.8) -> None:
    n = len(kpis)
    box_w = 12.0 / n
    for i, (label, value, color) in enumerate(kpis):
        x = 0.65 + i * box_w
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(x), Inches(y_in),
                                     Inches(box_w - 0.2), Inches(box_h))
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()

        tb_val = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y_in + 0.2),
                                          Inches(box_w - 0.4), Inches(0.8))
        pv = tb_val.text_frame.paragraphs[0]
        pv.text = value
        pv.font.size = Pt(28)
        pv.font.bold = True
        pv.font.color.rgb = WHITE
        pv.alignment = 2

        tb_lbl = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y_in + 1.0),
                                          Inches(box_w - 0.4), Inches(0.65))
        pl = tb_lbl.text_frame.paragraphs[0]
        pl.text = label
        pl.font.size = Pt(11)
        pl.font.color.rgb = WHITE
        pl.alignment = 2


# ============================================================
# Slides
# ============================================================
def slide_capa(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(2.5), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    logo = IMG_DIR / "Unifesp_completa_policromia_RGB.png"
    if logo.exists():
        slide.shapes.add_picture(str(logo), Inches(0.2), Inches(0.35), width=Inches(2.1))

    lab = slide.shapes.add_textbox(Inches(0.15), Inches(3.0), Inches(2.3), Inches(1.5))
    tf = lab.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "UNIFESP"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.text = "Instituto de\nCiência e\nTecnologia"
    p2.font.size = Pt(11)
    p2.font.color.rgb = BLUE_LIGHT

    sub = slide.shapes.add_textbox(Inches(3.0), Inches(1.1), Inches(10), Inches(0.6))
    p = sub.text_frame.paragraphs[0]
    p.text = "PÓS-QUALIFICAÇÃO · Reunião de orientação · 26/10/2026"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    tit = slide.shapes.add_textbox(Inches(3.0), Inches(1.9), Inches(10), Inches(2.5))
    tf = tit.text_frame
    tf.word_wrap = True
    for line in [
        "Caracterização do viés racial",
        "em classificação facial via",
        "condicionamento por tom de pele",
    ]:
        p = tf.paragraphs[0] if line == "Caracterização do viés racial" else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(30)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.space_after = Pt(4)

    subtit = slide.shapes.add_textbox(Inches(3.0), Inches(4.6), Inches(10), Inches(1.0))
    tf2 = subtit.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Marcello Vinicius Alves Ozzetti Cruz"
    p.font.size = Pt(18)
    p.font.color.rgb = GRAY_DK
    p2 = tf2.add_paragraph()
    p2.text = "Orientador: Prof. Dr. Marcos Gonçalves Quiles"
    p2.font.size = Pt(14)
    p2.font.color.rgb = GRAY_MD
    p3 = tf2.add_paragraph()
    p3.text = "PPG-CC · UNIFESP/ICT"
    p3.font.size = Pt(13)
    p3.font.color.rgb = GRAY_MD


def slide_aprovacao(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Aprovação na qualificação — 05/10/2026")

    add_kpi_row(
        slide,
        [
            ("Resultado", "APROVADO", GREEN),
            ("Banca", "3 membros", BLUE_MID),
            ("Fase atual", "Execução", NAVY),
            ("Prazo defesa", "~Abr/2027", AMBER),
        ],
        y_in=1.55,
        box_h=2.0,
    )

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(3.9), Inches(12.5), Inches(2.8))
    tf = tx.text_frame
    tf.word_wrap = True
    items = [
        ("Feedback da banca", "anotações completas revisadas e incorporadas nesta apresentação"),
        ("Pipeline reduzido", "6 etapas → 3 etapas / 2 metas (escopo de mestrado em 6 meses)"),
        ("Título revisado", '"mitigação" → "caracterização" — reflete melhor o escopo'),
        ("Decomposição Latinx", "retirada do pipeline principal → cap. 5 / trabalhos futuros"),
    ]
    for i, (h, b) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = h + ":  "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = b
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)

    add_footer(slide)
    add_page_number(slide, 2)


def slide_feedback_banca(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Síntese do feedback da banca")

    cats = [
        ("Abertura e definições",
         ["Começar definindo fairness e contexto de uso",
          "Não abrir com 'mitigar/remover disparidades' como objetivo",
          "Não contar história do backbone; não definir VLMs fora do escopo"]),
        ("Afirmações e rigor",
         ["Suavizar afirmações fortes: preferir 'há evidências de'",
          "Nunca usar 'primeira/primeiro' nas contribuições",
          "Retirar hipóteses com limiares numéricos arbitrários"]),
        ("Escopo e pipeline",
         ["Distinguir classificação facial vs reconhecimento facial",
          "Disparidade = medida racial, não de reconhecimento de faces",
          "Deixar transições entre etapas explícitas"]),
        ("Conteúdo faltante",
         ["Sempre citar tamanho dos datasets",
          "Formalizar matematicamente o mecanismo FiLM",
          "Explicar por que tom de pele melhora fairness e como chega ao reconhecimento",
          "Registrar que hiperparâm./pré-proc./losses anteriores não geraram ganhos",
          "Anotação é cega; revisar escopo; incluir conclusão"]),
    ]

    col_positions = [(0.5, 6.0), (6.9, 6.0)]
    for col_idx, (col_x, col_w) in enumerate(col_positions):
        for row_idx in range(2):
            overall = col_idx * 2 + row_idx
            if overall >= len(cats):
                break
            cat_title, cat_items = cats[overall]
            y_base = 1.55 + row_idx * 2.8

            hdr = slide.shapes.add_textbox(Inches(col_x), Inches(y_base), Inches(col_w), Inches(0.45))
            ph = hdr.text_frame.paragraphs[0]
            ph.text = cat_title
            ph.font.size = Pt(14)
            ph.font.bold = True
            ph.font.color.rgb = BLUE_MID

            rule = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(col_x), Inches(y_base + 0.45), Inches(col_w), Inches(0.03)
            )
            rule.fill.solid()
            rule.fill.fore_color.rgb = BLUE_LIGHT
            rule.line.fill.background()

            tx = slide.shapes.add_textbox(Inches(col_x), Inches(y_base + 0.52), Inches(col_w), Inches(2.0))
            tf = tx.text_frame
            tf.word_wrap = True
            for i, item in enumerate(cat_items):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "•  " + item
                p.font.size = Pt(12)
                p.font.color.rgb = GRAY_DK
                p.space_after = Pt(4)

    add_footer(slide)
    add_page_number(slide, 3)


def slide_titulo(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Título revisado")

    # Antigo
    old_lbl = slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(0.4))
    p = old_lbl.text_frame.paragraphs[0]
    p.text = "ANTES"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    old_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2.05),
                                     Inches(12.3), Inches(1.1))
    old_box.fill.solid()
    old_box.fill.fore_color.rgb = RGBColor(0xFA, 0xEB, 0xEB)
    old_box.line.fill.background()

    old_tx = slide.shapes.add_textbox(Inches(0.7), Inches(2.15), Inches(12.0), Inches(0.9))
    p = old_tx.text_frame.paragraphs[0]
    p.text = ("Mitigação de viés racial em classificação facial via "
              "condicionamento por tom de pele em arquiteturas profundas")
    p.font.size = Pt(17)
    p.font.italic = True
    p.font.color.rgb = ACCENT

    # Seta
    arrow = slide.shapes.add_textbox(Inches(6.0), Inches(3.3), Inches(1.5), Inches(0.5))
    p = arrow.text_frame.paragraphs[0]
    p.text = "▼"
    p.font.size = Pt(22)
    p.font.color.rgb = GRAY_MD
    p.alignment = 2

    # Novo
    new_lbl = slide.shapes.add_textbox(Inches(0.5), Inches(3.9), Inches(12.0), Inches(0.4))
    p = new_lbl.text_frame.paragraphs[0]
    p.text = "NOVO"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN

    new_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(4.35),
                                     Inches(12.3), Inches(1.1))
    new_box.fill.solid()
    new_box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    new_box.line.fill.background()

    new_tx = slide.shapes.add_textbox(Inches(0.7), Inches(4.45), Inches(12.0), Inches(0.9))
    p = new_tx.text_frame.paragraphs[0]
    p.text = ("Caracterização do viés racial em classificação facial via "
              "condicionamento por tom de pele em arquiteturas profundas")
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = GREEN

    # Justificativa
    just = slide.shapes.add_textbox(Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.2))
    tf = just.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = ("Justificativa (banca):  A pesquisa mensura e evidencia o viés — o arcabouço "
              "não foi desenhado para removê-lo. 'Caracterização' é mais preciso "
              "e defensável empiricamente.")
    p.font.size = Pt(15)
    p.font.italic = True
    p.font.color.rgb = GRAY_DK

    add_footer(slide)
    add_page_number(slide, 4)


def slide_fairness(prs: Presentation) -> None:
    add_bullets(
        prs, 5,
        "Fairness em reconhecimento facial — definição e contexto",
        [
            ("Fairness (equidade algorítmica):",
             "ausência de disparidade sistemática de desempenho entre grupos "
             "demográficos protegidos (raça, gênero, tom de pele) — Hardt et al. 2016."),
            ("Contexto desta pesquisa:",
             "sistemas de classificação / verificação facial apresentam, na literatura, "
             "diferenças de F1 de até 20 pp entre grupos raciais (Buolamwini & Gebru 2018; "
             "Wang et al. 2019 RFW)."),
            ("Questão de pesquisa:",
             "há evidências de que o condicionamento explícito por tom de pele (via FiLM) "
             "reduz a disparidade de desempenho em classificação racial no FairFace?"),
            ("Por que tom de pele?",
             "sinal fenotípico mensurável, menos dependente de raça autodeclarada; "
             "permite controlar sistematicamente a variação intra-classe."),
            ("Escopo:",
             "classificação racial em imagens estáticas (FairFace, 108k imgs, 7 classes) "
             "com transferência zero-shot para verificação (RFW + BFW)."),
        ],
    )


def slide_objetivos(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Objetivos — 2 metas, 3 etapas")

    # META 1
    m1_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.5),
                                    Inches(5.8), Inches(5.1))
    m1_box.fill.solid()
    m1_box.fill.fore_color.rgb = RGBColor(0xEB, 0xEF, 0xF8)
    m1_box.line.fill.background()

    m1h = slide.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(5.4), Inches(0.55))
    p = m1h.text_frame.paragraphs[0]
    p.text = "META 1 — Classificador racial com fairness"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    m1_items = [
        ("Etapa 1", "Extrator de tom de pele (ViT-B/16) treinado em MSTE (~1.500 imgs) + "
         "CCv2 (~5.500 vídeos) · diagnóstico MST × FairFace (108k imgs, 7 classes)"),
        ("Etapa 2", "Classificador ConvNeXt-T + FiLM condicionado no vetor MST z∈ℝ¹⁰ · "
         "comparação vs ResNet-34, FSCL+, FaceScanPaliGemma · "
         "métricas: F1-macro, Disparity Ratio, Worst-class F1"),
    ]
    tx1 = slide.shapes.add_textbox(Inches(0.7), Inches(2.2), Inches(5.4), Inches(4.0))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    for i, (h, b) in enumerate(m1_items):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        r1 = p.add_run()
        r1.text = h + ":  "
        r1.font.size = Pt(14)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = b
        r2.font.size = Pt(14)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(14)

    # META 2
    m2_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.0), Inches(1.5),
                                    Inches(5.8), Inches(5.1))
    m2_box.fill.solid()
    m2_box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    m2_box.line.fill.background()

    m2h = slide.shapes.add_textbox(Inches(7.2), Inches(1.6), Inches(5.4), Inches(0.55))
    p = m2h.text_frame.paragraphs[0]
    p.text = "META 2 — Transferência para verificação"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GREEN

    m2_items = [
        ("Etapa 3", "Congela encoder + FiLM (LAFTR · Madras 2018) · treina nova cabeça de "
         "verificação · avaliação em RFW (4 raças, Wang 2019) + BFW (4 raças × 2 gêneros, "
         "Robinson 2020) · métrica: TAR@FAR=10⁻⁴ por grupo com threshold fixo"),
    ]
    tx2 = slide.shapes.add_textbox(Inches(7.2), Inches(2.2), Inches(5.4), Inches(4.0))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    for i, (h, b) in enumerate(m2_items):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        r1 = p.add_run()
        r1.text = h + ":  "
        r1.font.size = Pt(14)
        r1.font.bold = True
        r1.font.color.rgb = GREEN
        r2 = p.add_run()
        r2.text = b
        r2.font.size = Pt(14)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(14)

    add_footer(slide)
    add_page_number(slide, 6)


def slide_pipeline(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Pipeline — 3 etapas, 2 metas")

    boxes = [
        (0.5,  "Etapa 1", "Extrator MST\n(ViT-B/16)\n\nDiagnóstico\nMST×FairFace",    BLUE_MID),
        (4.4,  "Etapa 2", "ConvNeXt-T\n+ FiLM(MST)\n\nvs baselines\n(F1, DR, WCF1)",  NAVY),
        (8.3,  "Etapa 3", "Freeze encoder\n+ nova cabeça\n\nRFW / BFW\nTAR@FAR",       GREEN),
    ]

    for bx, title, body, color in boxes:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(bx), Inches(1.55),
                                     Inches(3.5), Inches(4.5))
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()

        tb_t = slide.shapes.add_textbox(Inches(bx + 0.15), Inches(1.7), Inches(3.2), Inches(0.5))
        p = tb_t.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = 2

        rule = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(bx + 0.15), Inches(2.25), Inches(3.2), Inches(0.03)
        )
        rule.fill.solid()
        rule.fill.fore_color.rgb = WHITE
        rule.line.fill.background()

        tb_b = slide.shapes.add_textbox(Inches(bx + 0.15), Inches(2.35), Inches(3.2), Inches(3.5))
        tf = tb_b.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = body
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE
        p.alignment = 2

    # Setas entre etapas com label de transição
    for ax, transition_text in [
        (4.0, "Vetor MST z\n→ condiciona FiLM"),
        (7.9, "Encoder+FiLM\ncongelado"),
    ]:
        arr = slide.shapes.add_textbox(Inches(ax), Inches(3.4), Inches(0.35), Inches(0.6))
        p = arr.text_frame.paragraphs[0]
        p.text = "▶"
        p.font.size = Pt(20)
        p.font.color.rgb = GRAY_MD
        p.alignment = 2

        tr_lbl = slide.shapes.add_textbox(Inches(ax - 0.2), Inches(4.1), Inches(0.75), Inches(0.8))
        tf = tr_lbl.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = transition_text
        p.font.size = Pt(9)
        p.font.italic = True
        p.font.color.rgb = GRAY_MD
        p.alignment = 2

    # Bandas das metas
    for mx, mw, mt, mc in [
        (0.5, 7.3, "META 1 — Classificador racial com fairness", BLUE_MID),
        (8.3, 3.5, "META 2 — Transferência", GREEN),
    ]:
        mb = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                    Inches(mx), Inches(6.15),
                                    Inches(mw), Inches(0.45))
        mb.fill.solid()
        mb.fill.fore_color.rgb = mc
        mb.line.fill.background()
        mt_tx = slide.shapes.add_textbox(Inches(mx + 0.1), Inches(6.18), Inches(mw - 0.2), Inches(0.4))
        p = mt_tx.text_frame.paragraphs[0]
        p.text = mt
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE

    add_footer(slide)
    add_page_number(slide, 7)


def slide_etapa1(prs: Presentation) -> None:
    add_bullets(
        prs, 8,
        "Etapa 1 — Extrator de tom de pele (MST)",
        [
            ("Objetivo:",
             "treinar um classificador de tom de pele (10 tons Monk) que forneça "
             "o vetor de condicionamento z∈ℝ¹⁰ para o FiLM na Etapa 2."),
            ("Arquitetura:",
             "ViT-B/16 (torchvision), pré-treinado ImageNet-21k, cabeça linear "
             "substituída (768 → 10); backbone congelado na fase de fine-tuning."),
            ("Datasets:",
             "MSTE (Google) — ~1.500 imgs anotadas (tom Monk 1-10) + "
             "Casual Conversations v2 (Meta/Porgali 2023) — ~5.500 vídeos "
             "[extraído ~55k frames]; STW (Pereira 2026, ICMC/USP) — 42.313 imgs "
             "[uso condicional, aguardando liberação]."),
            ("Diagnóstico MST × FairFace:",
             "mapa de co-ocorrência tom de pele × raça sobre o FairFace (~108k imgs, "
             "7 classes) para evidenciar a variação fenotípica intra-racial."),
            ("Validação:",
             "anotação humana cega (2 anotadores independentes); "
             "3 seeds (42, 1, 2); acurácia top-1 e matriz de confusão entre tons adjacentes."),
        ],
    )


def slide_etapa2(prs: Presentation) -> None:
    add_bullets(
        prs, 9,
        "Etapa 2 — Classificador racial condicionado (ConvNeXt-T + FiLM)",
        [
            ("Backbone:",
             "ConvNeXt-T (~28M parâm.), pré-treinado ImageNet-1k. Inserção de uma camada "
             "FiLM após o Stage 4 (mapa 7×7×768); ~14k parâm. adicionados (<0,1% do backbone)."),
            ("Mecanismo FiLM:",
             "F′_c = γ_c · F_c + β_c   onde γ,β ∈ ℝ^C são produzidos por "
             "MLPs f_γ, f_β a partir de z∈ℝ¹⁰ (Perez et al. 2018; CBN: de Vries 2017). "
             "O vetor z vem do extrator da Etapa 1, passado como entrada no treinamento."),
            ("Por que FiLM melhora fairness?",
             "o condicionamento em z força o backbone a modular seus mapas de ativação "
             "de acordo com o tom de pele, reduzindo viés implícito de texturas de pele "
             "aprendido da distribuição de treinamento."),
            ("Baselines (comparação casada, 3 seeds cada):",
             "ResNet-34 (~72% F1-macro); FSCL+ (Qin et al. 2023); "
             "FaceScanPaliGemma (75,7% F1-macro, somente inferência)."),
            ("Datasets + métricas:",
             "FairFace (~108k imgs, 7 classes); F1-macro, Disparity Ratio, "
             "Worst-class F1. Bootstrap 95% CI; comparação casada (mesmo split por seed)."),
        ],
    )


def slide_etapa3(prs: Presentation) -> None:
    add_bullets(
        prs, 10,
        "Etapa 3 — Transferência para verificação de faces",
        [
            ("Premissa:",
             "o encoder+FiLM treinado na Etapa 2 captura representações com menor viés "
             "racial. A hipótese é que isso se traduz em menor disparidade de TAR@FAR "
             "no problema de verificação."),
            ("Protocolo:",
             "congela encoder + FiLM (LAFTR, Madras et al. 2018); treina somente "
             "nova cabeça de verificação (MLP + ArcFace loss) sobre pares do RFW."),
            ("Datasets:",
             "RFW (Wang et al. 2019) — 4 raças, ~11k pares de teste; "
             "BFW (Robinson et al. 2020) — 4 raças × 2 gêneros, ~50k pares."),
            ("Métrica:",
             "TAR@FAR=10⁻⁴ por grupo racial com threshold único fixo (definido no "
             "conjunto de validação); Disparity Ratio entre TAR máximo e mínimo."),
            ("Como a melhoria chega ao reconhecimento?",
             "representações menos enviesadas → embeddings de faces do mesmo indivíduo "
             "ficam mais próximos independentemente do tom de pele → threshold único "
             "funciona melhor em grupos de pele mais escura."),
        ],
    )


def slide_fora_do_escopo(prs: Presentation) -> None:
    add_two_col(
        prs, 11,
        "Escopo — o que entra e o que fica para depois",
        left_title="IN — escopo desta dissertação",
        left_items=[
            "Extrator MST (ViT-B/16) + diagnóstico fenotípico",
            "Classificador racial ConvNeXt-T + FiLM condicionado",
            "Comparação vs baselines (3 seeds, métricas de fairness)",
            "Transferência para verificação com feature-freeze (Etapa 3)",
            "Formalização matemática do mecanismo FiLM",
            "Evidência de redução de disparidade (não 'eliminação')",
        ],
        right_title="OUT — trabalhos futuros / cap. 5",
        right_items=[
            "Decomposição do erro Latinx (análise fenotípica vs algorítmica)",
            "Definição de VLMs e história do backbone",
            "Técnicas de pré-processamento e augmentation testadas anteriormente "
            "(hiperparâm. e loss functions não geraram ganhos — registrado na dissertação)",
            "Generalização para outros atributos protegidos (gênero, idade)",
            "Deploy em sistema de vigilância real",
        ],
    )


def slide_contribuicoes(prs: Presentation) -> None:
    add_bullets(
        prs, 12,
        "Contribuições esperadas",
        [
            ("C1 — Diagnóstico fenotípico:",
             "mapa empírico de co-ocorrência MST × raça no FairFace, "
             "evidenciando a variação intra-racial de tom de pele."),
            ("C2 — Classificador condicionado:",
             "framework ConvNeXt-T + FiLM com overhead paramétrico <0,1%, "
             "comparado experimentalmente a baselines públicos com protocolo auditável "
             "(3 seeds, bootstrap CI)."),
            ("C3 — Transferência para verificação:",
             "evidência quantitativa (ou contrafactual) de que o condicionamento "
             "MST reduz a disparidade de TAR@FAR entre grupos raciais."),
            ("C4 — Protocolo reproduzível:",
             "código, dados e configurações públicos; "
             "anotação humana cega documentada."),
            "Nota: evitamos afirmar 'primeira vez' — usamos 'até onde sabemos' se necessário.",
        ],
    )


def slide_cronograma(prs: Presentation) -> None:
    add_table_slide(
        prs, 13,
        "Cronograma — 6 meses até a defesa",
        headers=["Mês", "Meta / Etapa", "Entregável", "Risco"],
        rows=[
            ["Nov/2026", "META 1 · Etapa 1\nExtrator MST",
             "Classificador MST treinado;\ndiagnóstico MST×FairFace",
             "STW indisponível (usar MSTE+CCv2)"],
            ["Dez/2026", "META 1 · Etapa 2a\nConvNeXt-T + FiLM",
             "Modelo baseline ConvNeXt-T\n(sem FiLM) rodando",
             "Custo GPU (>200h estimadas)"],
            ["Jan/2027", "META 1 · Etapa 2b\nFiLM + ablation",
             "Tabela de ablation completa\n(3 seeds, bootstrap CI)",
             "Convergência instável (mitigada\npor LR scheduler)"],
            ["Fev/2027", "META 1 final\n+ início Cap 4 LaTeX",
             "Resultados classificação\nredação metodologia",
             "Revisão bibliográfica ainda\nincompleta (cap 2)"],
            ["Mar/2027", "META 2 · Etapa 3\nTransferência",
             "TAR@FAR por grupo\nRFW + BFW",
             "Formato de pares RFW\n(parsing manual)"],
            ["Abr/2027", "Redação final + defesa",
             "Dissertação completa;\ndefesa pública",
             "Banca com restrição de data"],
        ],
        col_widths=[1.4, 2.7, 3.4, 4.2],
        highlight_rows=[0, 2, 4],
        font_size=11,
    )


def slide_status(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Status atual do trabalho (26/10/2026)")

    sections = [
        ("✅  Concluído", GREEN, [
            "Qualificação aprovada (05/10/2026)",
            "Estrutura de código e repositório (src/, tests/, pipelines/)",
            "Download e inventário dos datasets (MSTE, CCv2, FairFace)",
            "Experimentos de ablation preliminares (hiperparâm., pré-proc., losses) "
            "— sem ganho conclusivo; registrados na dissertação",
            "Rascunho caps. 1–4 (LaTeX) — revisão pós-banca pendente",
        ]),
        ("⏳  Em andamento", AMBER, [
            "Anotação humana cega (2 anotadores independentes, ~30% concluído)",
            "Fine-tuning do extrator MST (Etapa 1) — aguardando CCv2 completo",
            "Revisão dos capítulos 1–4 conforme checklist da banca",
            "Incorporação de referências pós-2024 (survey fairness 2025, Pereira 2026 STW)",
        ]),
        ("🔜  Próximos passos", BLUE_MID, [
            "Diagnóstico MST × FairFace (Etapa 1 — Nov/2026)",
            "Implementar FiLM layer e rodar ablation (Etapa 2 — Dez/2026–Jan/2027)",
            "Transferência para verificação (Etapa 3 — Mar/2027)",
            "Redação da conclusão (ainda ausente — pedido da banca)",
        ]),
    ]

    for si, (sec_title, color, items) in enumerate(sections):
        y = 1.55 + si * 1.85
        hdr = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(12.5), Inches(0.4))
        p = hdr.text_frame.paragraphs[0]
        p.text = sec_title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = color

        tx = slide.shapes.add_textbox(Inches(0.7), Inches(y + 0.43), Inches(12.2), Inches(1.3))
        tf = tx.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "•  " + item
            p.font.size = Pt(13)
            p.font.color.rgb = GRAY_DK
            p.space_after = Pt(3)

    add_footer(slide)
    add_page_number(slide, 14)


# ============================================================
# Main
# ============================================================
def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide_capa(prs)             # 1
    slide_aprovacao(prs)        # 2
    slide_feedback_banca(prs)   # 3
    slide_titulo(prs)           # 4
    slide_fairness(prs)         # 5
    slide_objetivos(prs)        # 6
    slide_pipeline(prs)         # 7
    slide_etapa1(prs)           # 8
    slide_etapa2(prs)           # 9
    slide_etapa3(prs)           # 10
    slide_fora_do_escopo(prs)   # 11
    slide_contribuicoes(prs)    # 12
    slide_cronograma(prs)       # 13
    slide_status(prs)           # 14

    prs.save(str(OUT_PATH))
    print(f"Gerado: {OUT_PATH}")


if __name__ == "__main__":
    main()
