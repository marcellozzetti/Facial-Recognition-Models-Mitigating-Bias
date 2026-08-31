"""Gera apresentação PowerPoint da defesa da qualificação (30/09/2026).

29 slides, formato 16:9, ~15-20 min de fala.
Linguagem clara, sem jargão desnecessário, com referências acadêmicas
fortes inline (Autor Ano) e figuras integradas para tangibilizar.

Estrutura:
    1  Capa (com logo UNIFESP)
    2  Agenda (títulos objetivos, sem subexplicações)
    3  Motivação — contexto (KPIs visuais + citação de destaque)
    4  Motivação — regulação (timeline visual das regulações)
    5  Problema — disparidade racial (figura)
    6  Problema — comparativo SOTA (figura)
    7  Problema — heterogeneidade fenotípica (layout balanceado)
    8  Problema — refutação Pangelinan (2 colunas: crítica × resposta)
    9  Objetivo geral
    10 Objetivos específicos (6)
    11 Hipóteses (6)
    12 Revisão — timeline das mitigações (figura)
    13 Revisão — 6 baselines de mitigação (referências)
    14 Revisão — heterogeneidade intra-Latinx (3 disciplinas)
    15 Revisão — 5 lacunas identificadas
    16 Metodologia — pipeline 6 etapas (figura)
    17 Metodologia — Etapa 1 (classificador MST próprio)
    18 Metodologia — por que ConvNeXt-T (4 critérios vs ResNet vs ViT)  [NOVO]
    19 Metodologia — por que FiLM (comparação com 7 alternativas)      [NOVO]
    20 Metodologia — mecanismo FiLM (figura)
    21 Metodologia — 3 configurações A/B/C
    22 Metodologia — baselines + cenários
    23 Metodologia — triangulação de métricas
    24 Contribuições (3 eixos, 7 contribuições)
    25 Cronograma
    26 Riscos + mitigações
    27 Estado atual — adiantamento (KPIs visuais)
    28 Considerações finais
    29 Perguntas / obrigado

Uso:
    python docs/ativo/_gerar_apresentacao_qualificacao_2026-09.py
    -> produz: docs/ativo/material_qualificacao_2026-09-30.pptx
"""

from __future__ import annotations

from datetime import date
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

QUALIFICACAO = date(2026, 9, 30)
TOTAL_SLIDES = 29

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "docs" / "tese" / "images"


# ============================================================
# Helpers
# ============================================================
def add_page_number(slide, number: int) -> None:
    tb = slide.shapes.add_textbox(Inches(12.6), Inches(7.05), Inches(0.7), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{number} / {TOTAL_SLIDES}"
    p.font.size = Pt(9)
    p.font.color.rgb = GRAY_MD
    p.font.italic = True


def add_title(slide, text: str) -> None:
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.5), Inches(0.9))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.04)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()


def add_footer(slide, text: str = "Qualificação · Marcello Ozzetti · Prof. Marcos Quiles · UNIFESP/ICT") -> None:
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(11.5), Inches(0.4))
    pf = ft.text_frame.paragraphs[0]
    pf.text = text
    pf.font.size = Pt(9)
    pf.font.color.rgb = GRAY_MD
    pf.font.italic = True


def _blank(prs: Presentation):
    return prs.slide_layouts[6]


def add_bullets(prs: Presentation, number: int, title: str, bullets: list) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.5), Inches(5.4))
    tf = tx.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            head, body = item
            r1 = p.add_run()
            r1.text = head + "  "
            r1.font.size = Pt(15)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(15)
            r2.font.color.rgb = GRAY_DK
        else:
            p.text = "•  " + item
            p.font.size = Pt(15)
            p.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)
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
                run.font.size = Pt(13)
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


def add_image_slide(
    prs: Presentation,
    number: int,
    title: str,
    image_path: Path,
    caption: str = "",
    height_in: float = 4.8,
) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)
    if image_path.exists():
        pic = slide.shapes.add_picture(
            str(image_path), Inches(1.0), Inches(1.5), height=Inches(height_in)
        )
        pic.left = Inches((13.33 - pic.width.inches) / 2)
    else:
        tb = slide.shapes.add_textbox(Inches(2), Inches(3), Inches(9), Inches(1))
        p = tb.text_frame.paragraphs[0]
        p.text = f"[figura ausente: {image_path.name}]"
        p.font.size = Pt(14)
        p.font.italic = True
        p.font.color.rgb = ACCENT
    if caption:
        cap = slide.shapes.add_textbox(Inches(0.5), Inches(6.55), Inches(12.5), Inches(0.4))
        p = cap.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = GRAY_MD
        p.alignment = 2
    add_footer(slide)
    add_page_number(slide, number)


def add_image_plus_bullets(
    prs: Presentation,
    number: int,
    title: str,
    image_path: Path,
    bullets: list,
    image_width_in: float = 5.5,
) -> None:
    """Slide dividido: imagem à esquerda, bullets à direita."""
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, title)

    if image_path.exists():
        pic = slide.shapes.add_picture(
            str(image_path), Inches(0.5), Inches(1.6), width=Inches(image_width_in)
        )

    tx = slide.shapes.add_textbox(
        Inches(0.5 + image_width_in + 0.3),
        Inches(1.6),
        Inches(13.33 - 0.5 - image_width_in - 0.3 - 0.4),
        Inches(5.2),
    )
    tf = tx.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            head, body = item
            r1 = p.add_run()
            r1.text = head + "  "
            r1.font.size = Pt(13)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(13)
            r2.font.color.rgb = GRAY_DK
        else:
            p.text = "•  " + item
            p.font.size = Pt(13)
            p.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)
    add_footer(slide)
    add_page_number(slide, number)


# ============================================================
# SLIDE 1 — Capa
# ============================================================
def slide_capa(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(2.5), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    logo = IMG_DIR / "Unifesp_completa_policromia_RGB.png"
    if logo.exists():
        pic = slide.shapes.add_picture(str(logo), Inches(0.2), Inches(0.35), width=Inches(2.1))

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

    sub = slide.shapes.add_textbox(Inches(3.0), Inches(1.2), Inches(10), Inches(0.6))
    p = sub.text_frame.paragraphs[0]
    p.text = "EXAME DE QUALIFICAÇÃO"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    tit = slide.shapes.add_textbox(Inches(3.0), Inches(2.0), Inches(10), Inches(2.5))
    tf = tit.text_frame
    tf.word_wrap = True
    for line in [
        "Mitigação de viés racial em",
        "classificação facial com",
        "condicionamento por tom de pele",
    ]:
        p = tf.paragraphs[0] if line.startswith("Mitigação") else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = NAVY

    meta = slide.shapes.add_textbox(Inches(3.0), Inches(5.0), Inches(10), Inches(2.2))
    tf = meta.text_frame
    tf.word_wrap = True
    rows = [
        ("Mestrando:", "Marcello Vinicius Alves Ozzetti Cruz"),
        ("Orientador:", "Prof. Dr. Marcos Gonçalves Quiles"),
        ("Programa:", "Pós-Graduação em Ciência da Computação"),
        ("Data:", f"{QUALIFICACAO.strftime('%d de setembro de %Y')}"),
        ("Local:", "UNIFESP / ICT — São José dos Campos"),
    ]
    for i, (k, v) in enumerate(rows):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = k + "  "
        r1.font.size = Pt(14)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(14)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(4)


# ============================================================
# SLIDE 2 — Agenda
# ============================================================
def slide_agenda(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Roteiro")

    # 8 blocos em 2 colunas x 4 linhas
    items = [
        "Motivação",
        "Problema",
        "Objetivos e hipóteses",
        "Revisão da literatura",
        "Metodologia",
        "Contribuições esperadas",
        "Cronograma e riscos",
        "Estado atual",
    ]
    box_w = 5.8
    box_h = 1.05
    gap_x = 0.3
    gap_y = 0.15
    start_x = (13.33 - 2 * box_w - gap_x) / 2
    start_y = 1.7

    for i, label in enumerate(items):
        col = i % 2
        row = i // 2
        x = start_x + col * (box_w + gap_x)
        y = start_y + row * (box_h + gap_y)

        # número em círculo/quadrado navy
        num = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(0.9), Inches(box_h),
        )
        num.fill.solid()
        num.fill.fore_color.rgb = NAVY
        num.line.fill.background()
        tx = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(0.9), Inches(box_h))
        p = tx.text_frame.paragraphs[0]
        p.text = str(i + 1)
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = 2

        # rótulo em caixa cinza
        lab = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x + 0.9 + 0.05), Inches(y), Inches(box_w - 0.9 - 0.05), Inches(box_h),
        )
        lab.fill.solid()
        lab.fill.fore_color.rgb = GRAY_LT
        lab.line.color.rgb = GRAY_LT
        tx = slide.shapes.add_textbox(
            Inches(x + 0.9 + 0.2), Inches(y), Inches(box_w - 0.9 - 0.2), Inches(box_h),
        )
        p = tx.text_frame.paragraphs[0]
        p.text = label
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = NAVY
        # vertical centering via paragraph spacing hack não é trivial; deixamos top-aligned
    add_footer(slide)
    add_page_number(slide, 2)


# ============================================================
# SLIDE 3 — Motivação: contexto
# ============================================================
def slide_motivacao_contexto(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Reconhecimento facial já é infraestrutura social")

    # 4 cards visuais de uso
    usos = [
        ("Dispositivos", "Desbloqueio\nde celular"),
        ("Bancos", "Autenticação\nfinanceira"),
        ("Fronteiras", "Controle\nmigratório"),
        ("Segurança", "Identificação\npolicial"),
    ]
    box_w = 2.6
    gap = 0.25
    start_x = (13.33 - 4 * box_w - 3 * gap) / 2
    top_y = 1.6

    for i, (kicker, label) in enumerate(usos):
        x = start_x + i * (box_w + gap)
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(top_y), Inches(box_w), Inches(1.8),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = GRAY_LT
        box.line.color.rgb = GRAY_LT
        # kicker
        tx = slide.shapes.add_textbox(Inches(x), Inches(top_y + 0.25), Inches(box_w), Inches(0.4))
        p = tx.text_frame.paragraphs[0]
        p.text = kicker.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT
        p.alignment = 2
        # label
        tx2 = slide.shapes.add_textbox(Inches(x + 0.1), Inches(top_y + 0.75), Inches(box_w - 0.2), Inches(0.9))
        tf = tx2.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        for j, line in enumerate(label.split("\n")):
            para = p if j == 0 else tf.add_paragraph()
            para.text = line
            para.font.size = Pt(16)
            para.font.bold = True
            para.font.color.rgb = NAVY
            para.alignment = 2

    # Bloco NIST em destaque (KPI grande)
    kpi_y = 3.9
    kpi_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(kpi_y), Inches(4.5), Inches(2.6),
    )
    kpi_box.fill.solid()
    kpi_box.fill.fore_color.rgb = NAVY
    kpi_box.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(0.7), Inches(kpi_y + 0.25), Inches(4.5), Inches(1.1))
    p = tx.text_frame.paragraphs[0]
    p.text = "10 – 100 ×"
    p.font.size = Pt(46)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2
    tx2 = slide.shapes.add_textbox(Inches(0.9), Inches(kpi_y + 1.5), Inches(4.1), Inches(1.0))
    tf = tx2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "diferença na taxa de\nfalso positivo entre\ngrupos raciais"
    p.font.size = Pt(14)
    p.font.color.rgb = BLUE_LIGHT
    p.alignment = 2

    # Texto explicativo à direita
    tx3 = slide.shapes.add_textbox(Inches(5.5), Inches(kpi_y), Inches(7.4), Inches(2.6))
    tf = tx3.text_frame
    tf.word_wrap = True
    linhas = [
        ("NIST 2019 (Grother et al.):", "maior auditoria pública já feita em biometria facial — 189 algoritmos comerciais avaliados sobre 18 milhões de imagens."),
        ("Gender Shades (Buolamwini & Gebru, 2018):", "estudo fundacional que colocou o viés racial-de-gênero na agenda pública."),
        ("Conclusão:", "a tecnologia funciona bem para a maioria, mas não funciona igualmente bem para todos."),
    ]
    for i, (head, body) in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + " "
        r1.font.size = Pt(13)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(13)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(7)

    add_footer(slide)
    add_page_number(slide, 3)


# ============================================================
# SLIDE 4 — Motivação: regulação
# ============================================================
def slide_motivacao_regulacao(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "O tema virou obrigação regulatória")

    # Timeline horizontal
    marcos = [
        ("2018", "Gender Shades", "Buolamwini & Gebru colocam o viés na agenda."),
        ("2019", "NIST FRVT", "Grother et al. documentam disparidade 10–100× em escala industrial."),
        ("2024", "EU AI Act", "Auditoria de equidade vira REQUISITO FORMAL para sistemas biométricos.", True),
        ("2025", "FAccT", "Lafargue et al. propõem pipeline de auditoria com propagação de incerteza."),
    ]
    n = len(marcos)
    total_w = 12.0
    start_x = (13.33 - total_w) / 2
    line_y = 2.4
    node_r = 0.35

    # linha horizontal
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(start_x + 0.5), Inches(line_y - 0.03),
        Inches(total_w - 1.0), Inches(0.06),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = GRAY_MD
    line.line.fill.background()

    step = (total_w - 1.0) / (n - 1)
    for i, marco in enumerate(marcos):
        year, title, body = marco[0], marco[1], marco[2]
        highlight = marco[3] if len(marco) > 3 else False

        cx = start_x + 0.5 + i * step
        # nó
        node_color = ACCENT if highlight else NAVY
        node = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(cx - node_r), Inches(line_y - node_r),
            Inches(2 * node_r), Inches(2 * node_r),
        )
        node.fill.solid()
        node.fill.fore_color.rgb = node_color
        node.line.color.rgb = WHITE

        # ano acima do nó
        tx = slide.shapes.add_textbox(Inches(cx - 0.75), Inches(line_y - 1.0), Inches(1.5), Inches(0.5))
        p = tx.text_frame.paragraphs[0]
        p.text = year
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = ACCENT if highlight else NAVY
        p.alignment = 2

        # título abaixo do nó
        tx = slide.shapes.add_textbox(Inches(cx - 1.35), Inches(line_y + 0.5), Inches(2.7), Inches(0.5))
        p = tx.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.alignment = 2

        # corpo abaixo
        tx = slide.shapes.add_textbox(Inches(cx - 1.4), Inches(line_y + 1.05), Inches(2.8), Inches(2.0))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = body
        p.font.size = Pt(11)
        p.font.color.rgb = GRAY_DK
        p.alignment = 2

    # Bloco de conclusão inferior
    bottom = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.7), Inches(5.6), Inches(11.9), Inches(1.1),
    )
    bottom.fill.solid()
    bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(0.9), Inches(5.72), Inches(11.5), Inches(0.9))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = ("A pergunta científica mudou. Já não é “existe viés?” — está provado. "
              "É “como mitigar de forma defensável e auditável?”")
    p.font.size = Pt(15)
    p.font.italic = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    add_footer(slide)
    add_page_number(slide, 4)


# ============================================================
# SLIDE 5 — Problema (figura disparidade)
# ============================================================
def slide_problema_disparidade(prs: Presentation) -> None:
    add_image_slide(
        prs, 5,
        "Disparidade estável de ~30 pontos entre grupos raciais",
        IMG_DIR / "fig_disparidade_racial.png",
        caption="FaceScanPaliGemma (AlDahoul et al., 2024) sobre o dataset FairFace (Kärkkäinen & Joo, 2021). Diferença entre Black e Latinx = 30 pontos percentuais.",
        height_in=4.8,
    )


# ============================================================
# SLIDE 6 — Problema (comparativo SOTA)
# ============================================================
def slide_problema_sota_comparativo(prs: Presentation) -> None:
    add_image_slide(
        prs, 6,
        "A disparidade persiste através de múltiplas arquiteturas",
        IMG_DIR / "fig_f1_sota_comparativo.png",
        caption="Comparativo de F1 macro entre modelos publicados sobre o FairFace. A hierarquia de dificuldade entre raças é estável — problema estrutural, não artefato de um único modelo.",
        height_in=4.6,
    )


# ============================================================
# SLIDE 7 — Problema: heterogeneidade (figura MST vs Fitzpatrick)
# ============================================================
def slide_problema_heterogeneidade(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Por que “Latinx” é a classe mais difícil?")

    # Imagem alinhada à esquerda com moldura de espaço reservado
    img = IMG_DIR / "fig_fitzpatrick_vs_mst.png"
    if img.exists():
        pic = slide.shapes.add_picture(
            str(img), Inches(0.5), Inches(2.0), width=Inches(6.2)
        )
    # Caption abaixo da imagem
    cap = slide.shapes.add_textbox(Inches(0.5), Inches(6.4), Inches(6.2), Inches(0.4))
    p = cap.text_frame.paragraphs[0]
    p.text = "Fitzpatrick (esq., 6 tons) × Monk (dir., 10 tons)."
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = GRAY_MD
    p.alignment = 2

    # Coluna de bullets alinhados verticalmente com a imagem
    tx = slide.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.8), Inches(5.3))
    tf = tx.text_frame
    tf.word_wrap = True
    bullets = [
        ("Rótulo monolítico:",
         "a categoria agrupa fenótipos muito diferentes em uma única etiqueta."),
        ("Escala Monk (Ellis Monk, 2019):",
         "10 tons — sucessora moderna da escala Fitzpatrick, que sub-representa tons escuros."),
        ("Schumann et al. (2023):",
         "consolida a MST como padrão moderno de auditoria em fairness facial."),
        ("Consequência:",
         "a rotulagem monolítica de raça esconde essa diversidade e prejudica o modelo em grupos heterogêneos como Latinx."),
    ]
    for i, (head, body) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(14)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(14)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(10)

    add_footer(slide)
    add_page_number(slide, 7)


# ============================================================
# SLIDE 8 — Problema: refutação
# ============================================================
def slide_problema_refutacao(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Contra-argumento: e se o problema não for “raça”?")

    # Duas colunas com títulos coloridos: crítica (ACCENT) x resposta (NAVY)
    col_w = 6.0
    col_h = 5.4
    gap = 0.4
    start_x = (13.33 - 2 * col_w - gap) / 2
    top_y = 1.55

    # Coluna 1: A crítica (vermelho)
    header1 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(start_x), Inches(top_y), Inches(col_w), Inches(0.55),
    )
    header1.fill.solid()
    header1.fill.fore_color.rgb = ACCENT
    header1.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(start_x), Inches(top_y + 0.06), Inches(col_w), Inches(0.5))
    p = tx.text_frame.paragraphs[0]
    p.text = "A CRÍTICA DA LITERATURA"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    body1 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(start_x), Inches(top_y + 0.65), Inches(col_w), Inches(col_h - 0.65),
    )
    body1.fill.solid()
    body1.fill.fore_color.rgb = GRAY_LT
    body1.line.color.rgb = GRAY_LT

    tx = slide.shapes.add_textbox(
        Inches(start_x + 0.25), Inches(top_y + 0.85),
        Inches(col_w - 0.5), Inches(col_h - 1.0),
    )
    tf = tx.text_frame
    tf.word_wrap = True
    criticas = [
        ("Pangelinan et al. (2023, FAccT):",
         "o gap racial em face recognition é primariamente explicado pela FRAÇÃO DE FACE ÚTIL na imagem, não pelo rótulo de raça."),
        ("Matias et al. (2026):",
         "publica o SkinToneNet e reforça o tom de pele como dimensão auditável — pressiona a comunidade a abandonar rótulos nominais."),
        ("Implicação forte:",
         "se a refutação vale, condicionar por raça é injustificável — o sinal correto seria puramente fenotípico."),
    ]
    for i, (head, body) in enumerate(criticas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(12)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(12)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)

    # Coluna 2: Nossa resposta (azul)
    x2 = start_x + col_w + gap
    header2 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x2), Inches(top_y), Inches(col_w), Inches(0.55),
    )
    header2.fill.solid()
    header2.fill.fore_color.rgb = NAVY
    header2.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(x2), Inches(top_y + 0.06), Inches(col_w), Inches(0.5))
    p = tx.text_frame.paragraphs[0]
    p.text = "NOSSA POSIÇÃO"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    body2 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x2), Inches(top_y + 0.65), Inches(col_w), Inches(col_h - 0.65),
    )
    body2.fill.solid()
    body2.fill.fore_color.rgb = BLUE_LIGHT
    body2.line.color.rgb = BLUE_LIGHT

    tx = slide.shapes.add_textbox(
        Inches(x2 + 0.25), Inches(top_y + 0.85),
        Inches(col_w - 0.5), Inches(col_h - 1.0),
    )
    tf = tx.text_frame
    tf.word_wrap = True
    respostas = [
        ("Tom de pele COMPLEMENTA, não substitui a raça:",
         "usamos MST como sinal de contexto que ajuda a rede — o rótulo racial continua sendo o alvo de predição."),
        ("Incorporamos a crítica formalmente:",
         "H6 testa diretamente a tese de Pangelinan (≥ 70 % da variância do erro explicada por pixel information)."),
        ("Controle explícito de pixel information (Etapa 5):",
         "avaliamos com e sem esse confounder — se a crítica vencer, a refutação vira contribuição quantitativa."),
        ("Ganho esperado se a nossa tese estiver certa:",
         "redução de disparidade + explicação estrutural de quanto do erro é irredutível vs mitigável."),
    ]
    for i, (head, body) in enumerate(respostas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(12)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(12)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)

    add_footer(slide)
    add_page_number(slide, 8)


# ============================================================
# SLIDE 9 — Objetivo geral
# ============================================================
def slide_objetivo_geral(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Objetivo geral")

    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.5), Inches(2.0), Inches(10.3), Inches(2.8),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(2.0), Inches(2.4), Inches(9.3), Inches(2.0))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = ("Desenvolver e avaliar um pipeline de classificação racial que usa "
              "o tom de pele (escala Monk, 10 tons) como sinal de contexto para "
              "reduzir a disparidade entre grupos, sem sacrificar a acurácia geral.")
    p.font.size = Pt(20)
    p.font.color.rgb = WHITE

    nota = slide.shapes.add_textbox(Inches(1.5), Inches(5.2), Inches(10.3), Inches(1.5))
    tf = nota.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = (
        "Arquitetura escolhida: ConvNeXt-T (Liu et al., 2022) como rede base + "
        "FiLM (Perez et al., 2018) como mecanismo que “consulta” o tom de pele antes de decidir."
    )
    p.font.size = Pt(14)
    p.font.italic = True
    p.font.color.rgb = GRAY_DK
    add_footer(slide)
    add_page_number(slide, 9)


# ============================================================
# SLIDE 10 — Objetivos específicos
# ============================================================
def slide_objetivos_especificos(prs: Presentation) -> None:
    add_table_slide(
        prs, 10,
        "Seis objetivos específicos",
        ["#", "Objetivo"],
        [
            ["1", "Quantificar como o tom de pele Monk se distribui dentro de cada grupo racial do FairFace."],
            ["2", "Treinar e avaliar classificador MST próprio, com sensitivity analysis contra alternativas."],
            ["3", "Implementar o pipeline com FiLM sobre ConvNeXt-T e compará-lo contra seis baselines."],
            ["4", "Demonstrar que o ganho em classificação transfere para reconhecimento (RFW/BFW)."],
            ["5", "Formalizar a triangulação de métricas — Disparity Ratio, F1 da pior classe, Equal Opportunity."],
            ["6", "Decompor quanto do erro Latinx vem do fenótipo (irredutível) e quanto do modelo (mitigável)."],
        ],
        col_widths=[0.5, 12.0],
        highlight_rows=[2, 5],
        font_size=13,
    )


# ============================================================
# SLIDE 11 — Hipóteses
# ============================================================
def slide_hipoteses(prs: Presentation) -> None:
    add_table_slide(
        prs, 11,
        "Seis hipóteses testáveis",
        ["#", "Hipótese", "Confirma se..."],
        [
            ["H1", "O classificador MST próprio atinge concordância humana suficiente.", "κ ≥ 0,7 com anotações internas"],
            ["H2", "A classe Latinx cobre ≥ 5 dos 10 tons Monk no FairFace.", "cobertura observada ≥ 5"],
            ["H3", "Condicionar por tom reduz disparidade sem perder acurácia agregada.", "DR menor + F1 macro ≥ baseline"],
            ["H4", "≥ 50 % dos erros Latinx concentram-se em zonas de sobreposição de tom.", "concentração observada ≥ 50 %"],
            ["H5", "O ganho em classificação transfere para reconhecimento.", "redução de gap também em RFW/BFW"],
            ["H6", "Parte substancial do gap é explicada por pixel information (Pangelinan 2023).", "R² explicado ≥ 70 %"],
        ],
        col_widths=[0.7, 7.5, 4.3],
        font_size=11,
    )


# ============================================================
# SLIDE 12 — Revisão: timeline das mitigações (figura)
# ============================================================
def slide_revisao_timeline(prs: Presentation) -> None:
    add_image_slide(
        prs, 12,
        "Timeline das principais respostas ao problema (2018–2026)",
        IMG_DIR / "fig_timeline_mitigacoes.png",
        caption="Frentes cronológicas: dados balanceados → funções de perda → arquiteturas Pareto-eficientes → modelos vision-language → escalas MST → heterogeneidade fenotípica.",
        height_in=4.8,
    )


# ============================================================
# SLIDE 13 — Revisão: 6 baselines
# ============================================================
def slide_revisao_mitigacao(prs: Presentation) -> None:
    add_table_slide(
        prs, 13,
        "Seis baselines de mitigação (comparação sistemática planejada)",
        ["Método", "Referência", "Ideia central"],
        [
            ["Adversarial debiasing", "Zhang et al. (2018)", "Segundo modelo tenta “ler” o atributo sensível — o primeiro se defende."],
            ["Group DRO", "Sagawa et al. (2020)", "Otimiza o erro do pior grupo, não o erro médio."],
            ["FSCL+", "Park et al. (2022)", "Aprendizado contrastivo com controle explícito de viés."],
            ["FineFACE", "Manzoor et al. (2024)", "Atenção mútua entre camadas — operação Pareto-eficiente."],
            ["ResNet-34", "Kärkkäinen & Joo (2021)", "Baseline canônico da própria comunidade FairFace."],
            ["ConvNeXt-T puro", "Liu et al. (2022)", "Controle arquitetural moderno — efeito da rede base sem condicionamento."],
        ],
        col_widths=[3.0, 3.0, 6.5],
        font_size=11,
    )


# ============================================================
# SLIDE 14 — Revisão: heterogeneidade (3 disciplinas)
# ============================================================
def slide_revisao_heterogeneidade(prs: Presentation) -> None:
    add_bullets(prs, 14, "O rótulo racial esconde variação — evidência externa à computação", [
        ("Antropologia biológica — Telles (2014):", "projeto PERLA em 4 países latino-americanos documenta “pigmentocracia”."),
        ("Genética populacional — Bryc et al. (2015, AJHG):", "estudo com 162 mil indivíduos mostra composição ancestral altamente variável em Latinos."),
        ("Sociologia identitária — Pew Research (López et al., 2017):", "identidade Hispanic decai de 97 % para 50 % ao longo de quatro gerações nos EUA."),
        "Três disciplinas independentes convergem: o rótulo racial do FairFace não captura variação fenotípica real.",
        ("Consequência metodológica:", "condicionar por tom de pele não substitui o rótulo — dá ao modelo uma pista objetiva que a rotulagem sozinha não oferece."),
    ])


# ============================================================
# SLIDE 15 — Revisão: 5 lacunas
# ============================================================
def slide_revisao_lacunas(prs: Presentation) -> None:
    add_table_slide(
        prs, 15,
        "Cinco lacunas científicas identificadas na literatura",
        ["#", "Lacuna", "Como endereçamos"],
        [
            ["L1", "Não existe matriz pública MST × classes raciais no FairFace.", "Etapa 2 — Contribuição 2."],
            ["L2", "FiLM (Perez 2018) nunca foi aplicado em classificação racial multi-classe.", "Etapa 3 — Contribuição 3."],
            ["L3", "Métricas de fairness multi-classe estão fragmentadas na literatura.", "Etapa 4 — Contribuição 4 (triangulação)."],
            ["L4", "Transferência de fairness (Madras et al., 2018) para reconhecimento é pouco estudada.", "Etapa 5 — Contribuição 5."],
            ["L5", "Não há decomposição quantitativa do gap Latinx (fenótipo × algoritmo).", "Etapa 6 — Contribuição 6."],
        ],
        col_widths=[0.6, 6.5, 5.4],
        font_size=11,
    )


# ============================================================
# SLIDE 16 — Metodologia: pipeline (figura)
# ============================================================
def slide_metodologia_pipeline(prs: Presentation) -> None:
    add_image_slide(
        prs, 16,
        "Pipeline em 6 etapas",
        IMG_DIR / "fig_pipeline_6etapas.png",
        caption="Fluxo top-down organizado em 3 fases: diagnóstico (etapas 1-2), método proposto (etapa 3) e validação/síntese (etapas 4-6).",
        height_in=5.4,
    )


# ============================================================
# SLIDE 17 — Metodologia: Etapa 1
# ============================================================
def slide_metodologia_etapa1(prs: Presentation) -> None:
    add_bullets(prs, 17, "Etapa 1 — Classificador de tom de pele (próprio)", [
        ("O que faz:", "recebe uma foto de rosto e devolve as 10 probabilidades da escala Monk."),
        ("Decisão pós-reunião Ago/2026:", "treinar nosso próprio classificador — não depender do SkinToneNet (Matias 2026), cujos pesos e dataset STW ainda não foram liberados."),
        ("Datasets de treino:", "MSTE — Monk Skin Tone Examples (Monk, 2019, Google) e Casual Conversations v2 (Porgali et al., 2023, Meta)."),
        ("Camada auto-suficiente:", "detecta o rosto (MTCNN, Zhang et al. 2016), alinha, corta e classifica — funciona em qualquer dataset, até os que não têm anotação."),
        ("Validação:", "protocolo humano interno (~250 imagens do FairFace) + sensitivity analysis com 2 a 3 classificadores alternativos."),
    ])


# ============================================================
# SLIDE 18 — Por que ConvNeXt-T (justificativa vs ResNet e vs ViT)
# ============================================================
def slide_por_que_convnext(prs: Presentation) -> None:
    add_table_slide(
        prs, 18,
        "Por que ConvNeXt-T (Liu et al., 2022) e não ResNet ou ViT?",
        ["Critério", "ResNet-34 (Kärkkäinen 2021)", "ViT-B (Swin, DeiT)", "ConvNeXt-T (adotado)"],
        [
            [
                "Desempenho ImageNet",
                "76 % top-1 — 5 anos atrás",
                "81–83 % top-1 — SOTA moderno",
                "82 % top-1 — paridade com ViT a custo convolucional",
            ],
            [
                "Estabilidade em fine-tuning",
                "BatchNorm — sensível a batch pequeno",
                "Boa, mas exige mais dados",
                "LayerNorm — robusto a variação de batch (importante p/ 3 sementes)",
            ],
            [
                "Compatibilidade com FiLM",
                "4 estágios naturais",
                "Estrutura por tokens — inserção não trivial",
                "4 estágios hierárquicos — inserção direta após bloco principal",
            ],
            [
                "Comparabilidade científica",
                "É o baseline canônico de fairness facial (FairFace)",
                "Sem baseline consolidado em fairness facial",
                "Controle arquitetural moderno vs ResNet — isola efeito do FiLM",
            ],
            [
                "Custo computacional",
                "22 M params",
                "86 M params (ViT-B)",
                "28 M params — leve, viabiliza 3 sementes × 3 configs em GPU comum",
            ],
        ],
        col_widths=[2.2, 3.0, 3.0, 4.3],
        highlight_rows=[],
        font_size=10,
    )


# ============================================================
# SLIDE 19 — Por que FiLM (comparação com 7 alternativas de conditioning)
# ============================================================
def slide_por_que_film(prs: Presentation) -> None:
    add_table_slide(
        prs, 19,
        "Por que FiLM (Perez et al., 2018) e não outras técnicas de condicionamento?",
        ["Alternativa avaliada", "Referência / origem", "Por que descartada"],
        [
            ["Concatenação direta MST → features", "Prática comum", "Explosão paramétrica e diluição do sinal — sem modulação explícita."],
            ["Conditional Batch Normalization", "Anterior a FiLM", "Caso particular do FiLM — generalizado por Perez et al. (2018)."],
            ["Cross-attention", "Transformer decoders", "Superdimensionado para sinal 10-dim — ~3× o custo de FiLM sem ganho."],
            ["AdaIN", "Style transfer", "Projetado p/ transferência de estilo — incompatível com sinal demográfico."],
            ["SPADE", "Síntese de imagens", "Requer mapa espacial denso — incompatível com vetor MST global."],
            ["HyperNetworks", "Meta-aprendizagem", "Instabilidade documentada — sobredimensionado para nosso porte."],
            ["LoRA / Adaptadores", "Parameter-efficient fine-tuning", "Modifica pesos, não condiciona features — categoria distinta (trab. futuro)."],
            ["FiLM (adotado)", "Perez et al. (2018)", "★ 10-dim ideal + ~1 % overhead + interpretabilidade γ,β + compatível com LayerNorm."],
        ],
        col_widths=[3.5, 3.0, 6.0],
        highlight_rows=[7],
        font_size=10,
    )


# ============================================================
# SLIDE 20 — Metodologia: mecanismo FiLM (figura)
# ============================================================
def slide_metodologia_film(prs: Presentation) -> None:
    add_image_slide(
        prs, 20,
        "Mecanismo FiLM — como o tom entra como contexto",
        IMG_DIR / "film_pipeline.png",
        caption="FiLM (Perez et al., 2018) modula as features intermediárias do ConvNeXt-T canal a canal, condicionadas ao vetor MST. Overhead: ~1,3 % do backbone.",
        height_in=4.6,
    )


# ============================================================
# SLIDE 21 — Metodologia: 3 configurações
# ============================================================
def slide_metodologia_configs(prs: Presentation) -> None:
    add_table_slide(
        prs, 21,
        "Três configurações comparadas no estudo de ablation",
        ["ID", "Configuração", "O que testa"],
        [
            ["A", "ConvNeXt-T puro (baseline)", "Sem condicionamento — controle arquitetural."],
            ["B", "ConvNeXt-T + FiLM (sinal MST direto, 10-dim)", "Proposta principal — tom de pele entra como contexto."],
            ["C", "ConvNeXt-T + FiLM (sinal via CLIP-text, 512-dim)", "Alternativa moderna — Radford et al. (2021) + Dehdashtian et al. (2024)."],
        ],
        col_widths=[0.6, 5.4, 6.5],
        highlight_rows=[1],
        font_size=12,
    )


# ============================================================
# SLIDE 20 — Baselines + Cenários
# ============================================================
def slide_baselines_cenarios(prs: Presentation) -> None:
    add_bullets(prs, 22, "Como validamos — cenários e norma", [
        ("Cenário A — apenas raça:", "métricas por classe racial (7 classes do FairFace)."),
        ("Cenário B — raça × gênero:", "análise interseccional (8 subgrupos, seguindo Gender Shades — Buolamwini & Gebru, 2018)."),
        ("Norma seguida:", "ISO/IEC 19795-10:2024 — padrão internacional para reporte de desempenho biométrico entre grupos."),
        ("Rigor experimental:", "3 sementes independentes por experimento (42, 1, 2), comparação pareada, intervalo de confiança 95 % via bootstrap."),
        ("Datasets de transferência (Etapa 5):", "RFW (Wang et al., 2019) e BFW (Robinson et al., 2020) — pares oficiais 1:1."),
    ])


# ============================================================
# SLIDE 21 — Triangulação de métricas
# ============================================================
def slide_metricas(prs: Presentation) -> None:
    add_bullets(prs, 23, "Triangulação de métricas — nenhuma métrica isolada basta", [
        ("Por que triangular — Kleinberg et al. (2017):", "Teorema da Impossibilidade — não existe métrica única de equidade que satisfaça todos os critérios simultaneamente."),
        ("Disparity Ratio:", "razão entre pior e melhor F1. Mede o quão desigual é o desempenho entre grupos."),
        ("F1 da pior classe:", "protege o grupo mais fraco — garante que a melhoria não vem só no meio."),
        ("Equal Opportunity — Hardt et al. (2016, NeurIPS):", "iguala a taxa de acerto entre grupos, condicionada ao rótulo verdadeiro."),
        ("Equalized Odds — Hardt et al. (2016):", "vai além — iguala acerto e erro entre grupos."),
        ("Visualização Pareto:", "para cada modelo, plotamos acurácia × disparidade — vemos quem está na fronteira."),
    ])


# ============================================================
# SLIDE 22 — Contribuições
# ============================================================
def slide_contribuicoes(prs: Presentation) -> None:
    add_table_slide(
        prs, 24,
        "Contribuições esperadas — 3 eixos, 7 contribuições",
        ["Eixo", "Contribuições", "Foco"],
        [
            ["Fenotípico-empírico", "1, 2", "Documentar como o fenótipo (MST) se distribui dentro de cada rótulo racial."],
            ["Metodológico-arquitetural", "3, 4, 7", "Injetar tom de pele como contexto arquitetural + triangulação de métricas."],
            ["Diagnóstico-estrutural", "5, 6", "Decompor o erro em fenotípico (irredutível) e algorítmico (mitigável)."],
        ],
        col_widths=[3.5, 2.0, 7.0],
        font_size=13,
    )


# ============================================================
# SLIDE 23 — Cronograma
# ============================================================
def slide_cronograma(prs: Presentation) -> None:
    add_table_slide(
        prs, 25,
        "Cronograma",
        ["Período", "Etapa", "Entrega"],
        [
            ["Ago/2026 (hoje)", "Preparação adiantada", "Código das Etapas 1 e 2 pronto e testado."],
            ["Set/2026", "QUALIFICAÇÃO", "Marco atual."],
            ["Out/2026", "Aplicar sugestões da banca", "Ajustes de texto e escopo."],
            ["Nov/2026", "Etapa 1 formal", "Treinar classificador MST + validação humana."],
            ["Dez/2026", "Etapa 2", "Matriz pública MST × raça."],
            ["Jan–Mar/2027", "Etapa 3", "Ablation 3 configurações FiLM."],
            ["Abr/2027", "Etapa 4", "Comparação vs 6 baselines."],
            ["Mai/2027", "Etapa 5", "Transferência RFW/BFW."],
            ["Jun/2027", "Etapa 6", "Síntese decompositiva."],
            ["2º sem 2027", "Redação final + DEFESA", "Encerramento."],
        ],
        col_widths=[2.5, 3.0, 7.0],
        highlight_rows=[0, 1, 9],
        font_size=11,
    )


# ============================================================
# SLIDE 24 — Riscos
# ============================================================
def slide_riscos(prs: Presentation) -> None:
    add_table_slide(
        prs, 26,
        "Riscos identificados e mitigações",
        ["#", "Risco", "Mitigação"],
        [
            ["R1", "Alguma das 6 hipóteses pode ser refutada.", "Cada hipótese tem plano B; refutação é resultado científico válido."],
            ["R2", "Qualidade do nosso classificador de tom de pele.", "Validação humana + sensitivity com outros classificadores + benchmark externo (STW) quando disponível."],
            ["R3", "Refutação parcial pela tese Pangelinan (pixel info > tom).", "Já incluída como H6 — refutação vira contribuição quantitativa."],
            ["R4", "Custo computacional (3 sementes × 3 configs × 6 baselines).", "ConvNeXt-T é leve (~28M params); estimativa 200–400 h GPU total."],
        ],
        col_widths=[0.6, 4.5, 7.4],
        highlight_rows=[1],
        font_size=12,
    )


# ============================================================
# SLIDE 25 — Estado atual (KPIs visuais)
# ============================================================
def slide_estado_atual(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Estado atual — adiantados no cronograma")

    kpis = [
        ("+3", "meses adiantados", "sobre o cronograma proposto na qualificação"),
        ("6 de 6", "etapas com código", "estruturado, testado e pronto para execução"),
        ("120", "testes automatizados", "todos passando (unit + smoke + integração)"),
    ]
    box_w = 4.0
    gap = 0.25
    start_x = (13.33 - (3 * box_w + 2 * gap)) / 2
    top_y = 1.6

    for i, (big, mid, small) in enumerate(kpis):
        x = start_x + i * (box_w + gap)
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(top_y), Inches(box_w), Inches(2.2),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if i == 0 else GRAY_LT
        box.line.fill.background()

        tx = slide.shapes.add_textbox(Inches(x), Inches(top_y + 0.15), Inches(box_w), Inches(0.9))
        p = tx.text_frame.paragraphs[0]
        p.text = big
        p.font.size = Pt(48)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 0 else NAVY
        p.alignment = 2

        tx2 = slide.shapes.add_textbox(Inches(x), Inches(top_y + 1.15), Inches(box_w), Inches(0.5))
        p = tx2.text_frame.paragraphs[0]
        p.text = mid
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 0 else NAVY
        p.alignment = 2

        tx3 = slide.shapes.add_textbox(Inches(x + 0.1), Inches(top_y + 1.65), Inches(box_w - 0.2), Inches(0.6))
        tf = tx3.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = small
        p.font.size = Pt(10)
        p.font.italic = True
        p.font.color.rgb = BLUE_LIGHT if i == 0 else GRAY_MD
        p.alignment = 2

    subtitle = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.5), Inches(0.4))
    p = subtitle.text_frame.paragraphs[0]
    p.text = "Entregas concretas até 30/08/2026"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.5), Inches(2.2))
    tf = tx.text_frame
    tf.word_wrap = True
    entregas = [
        ("Etapas 1 e 2 (código completo):", "classificador MST próprio, camada auto-suficiente, matriz MST × raça, teste da H2."),
        ("Etapa 3 (FiLM):", "camada FiLM + wrapper ConvNeXt-T + ensembler CLIP-text + pipeline de treino das 3 configs."),
        ("Etapas 4, 5 e 6:", "métricas de fairness + 4 baselines + Pareto + verificação RFW/BFW + decomposição ANOVA/R²."),
        ("Texto da dissertação:", "5 capítulos consolidados, 3 passadas de revisão de estilo, 104 fichas bibliográficas."),
    ]
    for i, (head, body) in enumerate(entregas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "•  " + head + "  "
        r1.font.size = Pt(13)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(13)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)

    add_footer(slide)
    add_page_number(slide, 27)


# ============================================================
# SLIDE 26 — Considerações finais
# ============================================================
def slide_consideracoes(prs: Presentation) -> None:
    add_bullets(prs, 28, "Por que este trabalho é oportuno agora", [
        "Regulação europeia recém-aprovada (AI Act, 2024) exige auditoria de equidade em sistemas biométricos.",
        "Literatura de 2023–2026 converge para tom de pele como pista central (Pangelinan 2023, Matias 2026, Schumann 2023).",
        "Ferramentas necessárias amadureceram: MSTE, Casual Conversations v2, FairFace, RFW, BFW.",
        ("A proposta agrega 3 vetores:", "fenotípico (empírico), metodológico (arquitetural) e diagnóstico (decompositivo)."),
        ("Diferencial científico:", "não busca apenas “reduzir F1 médio” — busca entender quanto do erro é mitigável e quanto é limite estrutural do problema."),
    ])


# ============================================================
# SLIDE 27 — Perguntas
# ============================================================
def slide_perguntas(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.33), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.3), Inches(3))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Obrigado."
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    p2 = tf.add_paragraph()
    p2.text = "Perguntas?"
    p2.font.size = Pt(28)
    p2.font.color.rgb = BLUE_LIGHT
    p2.alignment = 2

    ft = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.5))
    pf = ft.text_frame.paragraphs[0]
    pf.text = "Marcello Ozzetti · marcello.ozzetti@gmail.com · UNIFESP/ICT"
    pf.font.size = Pt(11)
    pf.font.italic = True
    pf.font.color.rgb = BLUE_LIGHT
    pf.alignment = 2


# ============================================================
# Main
# ============================================================
def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide_capa(prs)
    slide_agenda(prs)
    slide_motivacao_contexto(prs)
    slide_motivacao_regulacao(prs)
    slide_problema_disparidade(prs)
    slide_problema_sota_comparativo(prs)
    slide_problema_heterogeneidade(prs)
    slide_problema_refutacao(prs)
    slide_objetivo_geral(prs)
    slide_objetivos_especificos(prs)
    slide_hipoteses(prs)
    slide_revisao_timeline(prs)
    slide_revisao_mitigacao(prs)
    slide_revisao_heterogeneidade(prs)
    slide_revisao_lacunas(prs)
    slide_metodologia_pipeline(prs)
    slide_metodologia_etapa1(prs)
    slide_por_que_convnext(prs)      # NOVO - slide 18
    slide_por_que_film(prs)          # NOVO - slide 19
    slide_metodologia_film(prs)
    slide_metodologia_configs(prs)
    slide_baselines_cenarios(prs)
    slide_metricas(prs)
    slide_contribuicoes(prs)
    slide_cronograma(prs)
    slide_riscos(prs)
    slide_estado_atual(prs)
    slide_consideracoes(prs)
    slide_perguntas(prs)

    return prs


def main() -> None:
    prs = build_presentation()
    out_dir = Path(__file__).parent
    out = out_dir / "material_qualificacao_2026-09-30.pptx"
    prs.save(out)
    print(f"OK: {out}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
