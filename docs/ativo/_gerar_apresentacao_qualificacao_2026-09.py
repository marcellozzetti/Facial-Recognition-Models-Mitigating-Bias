"""Gera apresentação PowerPoint da defesa da qualificação (05/10/2026).

29 slides, formato 16:9, ~20 min de fala.
Linguagem acadêmica formal, referências inline (Autor, ANO) e figuras
integradas para tangibilizar o estado do trabalho.

Estrutura:
    1  Capa (com logo UNIFESP)
    2  Motivação acadêmica — origem Buolamwini (2016/2018/2023)
    3  Motivação — contexto sociotécnico
    4  Motivação — regulação (European AI Act)
    5  Problema — disparidade racial (figura)
    6  Problema — heterogeneidade fenotípica (Latinx)
    7  Problema — refutação Pangelinan
    8  Objetivo geral
    9  Objetivos específicos + hipóteses testáveis (consolidado)
    10 Revisão — linha do tempo das mitigações
    11 Revisão — baselines de mitigação
    12 Revisão — lacunas identificadas
    13 Pipeline experimental em 6 etapas (figura)
    14 Evolução do pipeline — estado em 04/10/2026
    15 Etapa 1 — classificador MST (contexto + datasets)
    16 Classificador MST — detalhe técnico do backend
    17 Classificador MST — validação qualitativa (grade)
    18 Classificador MST — distribuição MST × raça (heatmap)
    19 Racional do backbone ConvNeXt-T
    20 Racional do mecanismo FiLM
    21 Mecanismo FiLM (figura)
    22 Configurações do ablation (A, B, C)
    23 Protocolo de validação — cenários e norma
    24 Triangulação de métricas de equidade
    25 Contribuições esperadas
    26 Cronograma
    27 Riscos e mitigações
    28 Estado atual do trabalho
    29 Obrigado / perguntas

Uso:
    python docs/ativo/_gerar_apresentacao_qualificacao_2026-09.py
    -> produz: docs/ativo/material_qualificacao_2026-10-05.pptx
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

QUALIFICACAO = date(2026, 10, 5)
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


def add_footer(slide, text: str = "Exame de Qualificação · Marcello Ozzetti · Orientação: Prof. Dr. Marcos G. Quiles · PPG-CC UNIFESP/ICT") -> None:
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
        p.font.size = Pt(14)
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
            r1.font.size = Pt(16)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(16)
            r2.font.color.rgb = GRAY_DK
        else:
            p.text = "•  " + item
            p.font.size = Pt(16)
            p.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)
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
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(16)
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

    # Column-major: item 1 topo-esquerda, item 2 abaixo, ... item 5 topo-direita
    n_rows = 4
    for i, label in enumerate(items):
        col = i // n_rows
        row = i % n_rows
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
# SLIDE 2 — Motivação acadêmica: origem do problema (Buolamwini, 2016/2018/2023)
# ============================================================
def slide_motivacao_academica(prs: Presentation) -> None:
    """Slide de motivação acadêmica — âncora histórica da agenda de pesquisa.

    Reescritura em registro acadêmico-formal do episódio relatado por
    Buolamwini (2023, Unmasking AI), com encaminhamento à consolidação
    científica em Gender Shades (Buolamwini & Gebru, 2018, PMLR).
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Motivação acadêmica: origem do problema de pesquisa")

    # Barra lateral de citação (acento visual)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.5), Inches(1.55),
        Inches(0.15), Inches(3.1),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Bloco da citação acadêmica (reescritura)
    quote = slide.shapes.add_textbox(
        Inches(0.95), Inches(1.60), Inches(11.9), Inches(2.6)
    )
    tf = quote.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = (
        "Em 2016, durante projeto acadêmico no MIT Media Lab, a então mestranda "
        "Joy Buolamwini constatou que sistemas comerciais de análise facial "
        "falhavam em detectar seu próprio rosto, enquanto detectavam, sem falha, "
        "interlocutores fenotipicamente mais claros. A conclusão do projeto "
        "exigiu o uso de uma máscara branca para que a face fosse reconhecida — "
        "episódio que converteu uma observação pessoal em programa de pesquisa."
    )
    p.font.size = Pt(17)
    p.font.italic = True
    p.font.color.rgb = NAVY

    attr = slide.shapes.add_textbox(
        Inches(0.95), Inches(4.25), Inches(11.9), Inches(0.4)
    )
    pa = attr.text_frame.paragraphs[0]
    pa.text = (
        "Reescritura acadêmica da narrativa relatada em "
        "Unmasking AI: My Mission to Protect What Is Human in a World of Machines "
        "(Buolamwini, 2023, W. W. Norton)."
    )
    pa.font.size = Pt(10)
    pa.font.italic = True
    pa.font.color.rgb = GRAY_MD
    pa.alignment = 2

    # Consequências acadêmicas (bullets)
    cons = slide.shapes.add_textbox(
        Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0)
    )
    tf = cons.text_frame
    tf.word_wrap = True
    bullets = [
        (
            "Consolidação científica:",
            "a observação foi formalizada em Gender Shades (Buolamwini & Gebru, 2018, PMLR), "
            "estudo seminal que quantificou disparidade de até 34,4 pontos percentuais em acurácia "
            "entre homens de pele clara e mulheres de pele escura em três APIs comerciais de análise facial.",
        ),
        (
            "Herança científica desta pesquisa:",
            "inscreve-se formalmente na mesma agenda — dissociar o desempenho de classificadores "
            "faciais do tom de pele —, deslocando a pergunta de \"existe viés?\" para "
            "\"como mitigá-lo de forma defensável e auditável?\".",
        ),
    ]
    for i, (head, body) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(15)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(15)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)

    add_footer(slide)
    add_page_number(slide, 2)


# ============================================================
# SLIDE 3 — Motivação: contexto
# ============================================================
def slide_motivacao_contexto(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Reconhecimento facial: adoção sociotécnica em larga escala")

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
        p.font.size = Pt(13)
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
    p.text = "disparidade na taxa de\nfalso positivo entre\ngrupos demográficos"
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
        ("Conclusão:", "desempenho agregado favorável coexiste com disparidade sistemática entre subgrupos demográficos."),
    ]
    for i, (head, body) in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + " "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(7)

    add_footer(slide)
    add_page_number(slide, 3)


# ============================================================
# SLIDE 4 — Motivação: regulação
# ============================================================
def slide_motivacao_regulacao(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Regulação formal em vigor: European AI Act (2024)")

    # Timeline horizontal
    marcos = [
        ("2018", "Gender Shades", "Buolamwini & Gebru colocam o viés na agenda."),
        ("2019", "NIST FRVT", "Grother et al. documentam disparidade 10–100× em escala industrial."),
        ("2024", "EU AI Act", "Auditoria de equidade torna-se requisito formal para sistemas biométricos.", True),
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
        tx = slide.shapes.add_textbox(Inches(cx - 0.75), Inches(line_y - 1.1), Inches(1.5), Inches(0.5))
        p = tx.text_frame.paragraphs[0]
        p.text = year
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = ACCENT if highlight else NAVY
        p.alignment = 2

        # título abaixo do nó
        tx = slide.shapes.add_textbox(Inches(cx - 1.35), Inches(line_y + 0.5), Inches(2.7), Inches(0.5))
        p = tx.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.alignment = 2

        # corpo abaixo
        tx = slide.shapes.add_textbox(Inches(cx - 1.5), Inches(line_y + 1.15), Inches(3.0), Inches(2.4))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = body
        p.font.size = Pt(13)
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
    p.text = ("A questão científica desloca-se da constatação do viés — empiricamente estabelecida — "
              "para o desenvolvimento de mecanismos de mitigação defensáveis e auditáveis.")
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
        "Disparidade sistemática de aproximadamente 30 pontos percentuais entre grupos raciais",
        IMG_DIR / "fig_disparidade_racial.png",
        caption="FaceScanPaliGemma (AlDahoul et al., 2024) avaliado sobre o dataset FairFace (Kärkkäinen & Joo, 2021). Diferencial entre Black e Latinx: 30 pontos percentuais em F1.",
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
    add_title(slide, "Heterogeneidade fenotípica intra-categorial: o caso Latinx")

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
    p.font.size = Pt(12)
    p.font.italic = True
    p.font.color.rgb = GRAY_MD
    p.alignment = 2

    # Coluna de bullets alinhados verticalmente com a imagem
    tx = slide.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.8), Inches(5.3))
    tf = tx.text_frame
    tf.word_wrap = True
    bullets = [
        ("Rótulo monolítico:",
         "a rotulagem monolítica agrega fenótipos heterogêneos sob única categoria discreta."),
        ("Escala Monk (Ellis Monk, 2019):",
         "escala de 10 tons — sucessora moderna da escala Fitzpatrick, que sub-representa tons escuros."),
        ("Schumann et al. (2023):",
         "consolida a MST como padrão moderno de auditoria de equidade em análise facial."),
        ("Consequência:",
         "a rotulagem monolítica compromete a acurácia condicional em grupos com maior variabilidade fenotípica, como Latinx."),
    ]
    for i, (head, body) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(10)

    add_footer(slide)
    add_page_number(slide, 6)


# ============================================================
# SLIDE 8 — Problema: refutação
# ============================================================
def slide_problema_refutacao(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Refutação concorrente: tom de pele como principal fator (Pangelinan, 2023)")

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
    p.font.size = Pt(15)
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
         "a disparidade racial em face recognition é primariamente explicada pela fração de área facial útil na imagem (pixel information), não pelo rótulo racial."),
        ("Matias et al. (2026):",
         "publica o SkinToneNet e reforça o tom de pele como dimensão auditável, tensionando o uso de rótulos nominais."),
        ("Implicação forte:",
         "se a refutação se sustenta, o condicionamento por rótulo racial nominal perde sustentação teórica em favor do sinal fenotípico direto."),
    ]
    for i, (head, body) in enumerate(criticas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(16)
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
    p.font.size = Pt(15)
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
        ("O tom de pele complementa o rótulo racial:",
         "o vetor MST atua como sinal condicionante; o rótulo racial permanece como alvo de predição."),
        ("Incorporamos a crítica formalmente:",
         "a hipótese H6 testa diretamente a tese de Pangelinan (R² ≥ 70 % explicado por pixel information)."),
        ("Refutação convertida em contribuição:",
         "eventual confirmação da refutação produz decomposição quantitativa do erro, resultado científico válido em si."),
    ]
    for i, (head, body) in enumerate(respostas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)

    add_footer(slide)
    add_page_number(slide, 7)


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
    add_page_number(slide, 8)


# ============================================================
# SLIDE 10 — Objetivos específicos
# ============================================================
def slide_objetivos_hipoteses(prs: Presentation) -> None:
    """Slide consolidado (era 10 + 11): 6 objetivos à esquerda, 6 hipóteses à direita."""
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Objetivos específicos e hipóteses testáveis")

    col_w = 6.0
    gap = 0.3
    start_x = (13.33 - 2 * col_w - gap) / 2
    top_y = 1.55
    header_h = 0.55
    body_top = top_y + header_h + 0.15

    # ---------- Coluna 1: Objetivos ----------
    h1 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(start_x), Inches(top_y), Inches(col_w), Inches(header_h),
    )
    h1.fill.solid()
    h1.fill.fore_color.rgb = NAVY
    h1.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(start_x), Inches(top_y + 0.05), Inches(col_w), Inches(0.5))
    p = tx.text_frame.paragraphs[0]
    p.text = "6 OBJETIVOS ESPECÍFICOS"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    objetivos = [
        "Quantificar distribuição MST × classe racial no FairFace.",
        "Treinar e avaliar classificador MST próprio + sensitivity.",
        "Implementar FiLM sobre ConvNeXt-T e comparar vs 6 baselines.",
        "Demonstrar transferência do ganho para reconhecimento (RFW/BFW).",
        "Formalizar triangulação de métricas (DR, F1 pior classe, EO).",
        "Decompor erro Latinx: fenótipo (irredutível) vs modelo (mitigável).",
    ]
    tx = slide.shapes.add_textbox(
        Inches(start_x + 0.15), Inches(body_top),
        Inches(col_w - 0.3), Inches(5.0),
    )
    tf = tx.text_frame
    tf.word_wrap = True
    for i, obj in enumerate(objetivos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = f"{i+1}.  "
        r1.font.size = Pt(15)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = obj
        r2.font.size = Pt(15)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)

    # ---------- Coluna 2: Hipóteses ----------
    x2 = start_x + col_w + gap
    h2 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x2), Inches(top_y), Inches(col_w), Inches(header_h),
    )
    h2.fill.solid()
    h2.fill.fore_color.rgb = ACCENT
    h2.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(x2), Inches(top_y + 0.05), Inches(col_w), Inches(0.5))
    p = tx.text_frame.paragraphs[0]
    p.text = "6 HIPÓTESES TESTÁVEIS"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = 2

    hipoteses = [
        ("H1", "Classificador MST atinge concordância humana", "κ ≥ 0,7"),
        ("H2", "Latinx cobre ≥ 5 dos 10 tons Monk", "cobertura ≥ 5"),
        ("H3", "Condicionamento reduz disparidade sem perder acurácia", "DR ↓  ·  F1 ≥ baseline"),
        ("H4", "≥ 50 % dos erros Latinx em zonas de sobreposição", "concentração ≥ 50 %"),
        ("H5", "Módulo FiLM/MST pré-treinado transfere via feature-freeze para RFW/BFW", "ΔTAR@FAR=1e-4 ≥ +3 pp (African)"),
        ("H6", "Pixel information explica variância do erro (Pangelinan)", "R² ≥ 70 %"),
    ]
    tx = slide.shapes.add_textbox(
        Inches(x2 + 0.15), Inches(body_top),
        Inches(col_w - 0.3), Inches(5.0),
    )
    tf = tx.text_frame
    tf.word_wrap = True
    for i, (hid, texto, criterio) in enumerate(hipoteses):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = f"{hid}.  "
        r1.font.size = Pt(15)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT
        r2 = p.add_run()
        r2.text = texto + "  "
        r2.font.size = Pt(15)
        r2.font.color.rgb = GRAY_DK
        r3 = p.add_run()
        r3.text = f"[{criterio}]"
        r3.font.size = Pt(13)
        r3.font.color.rgb = GRAY_MD
        r3.font.italic = True
        p.space_after = Pt(8)

    add_footer(slide)
    add_page_number(slide, 9)


# ============================================================
# SLIDE 12 — Revisão: timeline das mitigações (figura)
# ============================================================
def slide_revisao_timeline(prs: Presentation) -> None:
    add_image_slide(
        prs, 10,
        "Linha do tempo das principais respostas ao problema (2018–2026)",
        IMG_DIR / "fig_timeline_mitigacoes.png",
        caption="Frentes cronológicas: dados balanceados → funções de perda → arquiteturas Pareto-eficientes → modelos vision-language → escalas MST → heterogeneidade fenotípica.",
        height_in=4.8,
    )


# ============================================================
# SLIDE 13 — Revisão: 6 baselines
# ============================================================
def slide_revisao_mitigacao(prs: Presentation) -> None:
    add_table_slide(
        prs, 11,
        "Baselines de mitigação para comparação sistemática",
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
        font_size=15,
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
        prs, 12,
        "Lacunas científicas identificadas na literatura",
        ["#", "Lacuna", "Endereçamento"],
        [
            ["L1",
             "Ausência de tabulação pública MST × raça sobre FairFace. "
             "Schumann et al. (2023, NeurIPS) estabelecem o protocolo MST "
             "sem aplicá-lo ao FairFace; Pereira et al. (2026) liberam o "
             "STW sobre dataset próprio.",
             "Etapa 2 — Contribuição 2."],
            ["L2",
             "FiLM nunca instanciado em classificação racial multi-classe. "
             "Nem o mecanismo original (Perez et al., 2018) nem sua "
             "formulação equivalente Conditional Batch Normalization "
             "(de Vries et al., 2017) foram adotados com condicionamento "
             "por tom de pele contínuo.",
             "Etapa 3 — Contribuição 3."],
            ["L3",
             "Falta decomposição formal do erro Latinx. Pangelinan et al. "
             "(2023) decompõem o gender gap em verificação (FST 6-tons), "
             "mas não estendem a classificação racial multi-classe; o "
             "patamar F1 ≈ 60 % em Latinx (Lin et al., 2022; AlDahoul "
             "et al., 2024) permanece sem explicação quantitativa.",
             "Etapa 6 — Contribuição 5."],
            ["L4",
             "Transferência classificação → verificação com condicionamento "
             "preservado é lacuna. Madras et al. (2018, LAFTR) provam a "
             "transferência fair apenas em dados tabulares; FairCal (2021), "
             "MixFairFace (2022) e score normalization (2024) treinam a "
             "mitigação diretamente em RFW/BFW.",
             "Etapa 5 — Contribuição 6."],
            ["L5",
             "Reporte empírico de mitigação sem triangulação. Apesar do "
             "Teorema da Impossibilidade (Kleinberg et al., 2017) e dos "
             "surveys que o catalogam (Mehrabi et al., 2021), papers de "
             "mitigação continuam reportando um único indicador agregado.",
             "Etapa 4 — Contribuição 4."],
        ],
        col_widths=[0.5, 8.0, 4.0],
        font_size=11,
    )


# ============================================================
# SLIDE 13 — Metodologia: pipeline em 6 etapas (shapes nativos PPTX)
# ============================================================
def slide_metodologia_pipeline(prs: Presentation) -> None:
    """Pipeline em 6 etapas desenhado com shapes nativos (editavel no
    PowerPoint). Layout horizontal: 6 caixas em uma linha, com faixa de
    fases acima e setas conectoras entre caixas; Etapa 3 destacada.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Pipeline experimental em seis etapas")

    # ---- Dados das 6 etapas ----
    etapas = [
        (1, "Classificador MST",
         "Treina rede 10-tons sobre MSTE + CCv2.", False),
        (2, "Matriz MST × raça",
         "Tabula tons preditos pelas 7 classes FairFace.", False),
        (3, "FiLM + ConvNeXt-T",
         "Rede condiciona features ao vetor MST (método proposto).", True),
        (4, "Baselines e cenários",
         "Confronta a proposta com seis famílias já publicadas.", False),
        (5, "Transferência RFW/BFW",
         "Verifica herança do condicionamento em reconhecimento.", False),
        (6, "Decomposição do erro",
         "Separa componente fenotípico de componente algorítmico.", False),
    ]
    # Fases: (label, cor_fundo, cor_texto, indices_etapas 0-based)
    fases = [
        ("A. DIAGNÓSTICO", GRAY_LT,   GRAY_DK, [0, 1]),
        ("B. MÉTODO",      BLUE_LIGHT, NAVY,    [2]),
        ("C. VALIDAÇÃO",   GRAY_LT,   GRAY_DK, [3, 4, 5]),
    ]

    # ---- Dimensoes (em polegadas) ----
    SLIDE_W = 13.33
    MARGIN_X = 0.35
    BOX_W = 2.00
    BOX_GAP = 0.11
    N_BOXES = 6
    total_boxes_w = N_BOXES * BOX_W + (N_BOXES - 1) * BOX_GAP
    start_x = (SLIDE_W - total_boxes_w) / 2

    PHASE_Y = 1.45
    PHASE_H = 0.42
    BOX_Y = 2.10
    BOX_H = 4.30

    # ---- Faixa de fases (topo) ----
    for label, bg, fg, idxs in fases:
        x_left = start_x + idxs[0] * (BOX_W + BOX_GAP)
        x_right = start_x + idxs[-1] * (BOX_W + BOX_GAP) + BOX_W
        bar = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_left), Inches(PHASE_Y),
            Inches(x_right - x_left), Inches(PHASE_H),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = bg
        bar.line.color.rgb = fg
        bar.line.width = Pt(0.75)
        tf = bar.text_frame
        tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
        tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
        p = tf.paragraphs[0]
        p.alignment = 2  # center
        r = p.add_run()
        r.text = label
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = fg

    # ---- 6 cartoes das etapas ----
    for i, (num, titulo, subt, highlight) in enumerate(etapas):
        x = start_x + i * (BOX_W + BOX_GAP)
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(BOX_Y), Inches(BOX_W), Inches(BOX_H),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if highlight else WHITE
        box.line.color.rgb = NAVY
        box.line.width = Pt(2.0 if highlight else 1.2)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.15); tf.margin_bottom = Inches(0.12)

        # eyebrow "ETAPA N"
        p0 = tf.paragraphs[0]
        p0.alignment = 2
        r0 = p0.add_run()
        r0.text = f"ETAPA {num}"
        r0.font.size = Pt(11)
        r0.font.bold = True
        r0.font.color.rgb = BLUE_LIGHT if highlight else BLUE_MID

        # titulo
        p1 = tf.add_paragraph()
        p1.alignment = 2
        p1.space_before = Pt(6)
        r1 = p1.add_run()
        r1.text = titulo
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = WHITE if highlight else NAVY

        # subtitulo
        p2 = tf.add_paragraph()
        p2.alignment = 2
        p2.space_before = Pt(14)
        r2 = p2.add_run()
        r2.text = subt
        r2.font.size = Pt(12)
        r2.font.color.rgb = BLUE_LIGHT if highlight else GRAY_DK

    # ---- Setas horizontais entre caixas ----
    arrow_y = BOX_Y + BOX_H / 2
    for i in range(N_BOXES - 1):
        x_start = start_x + i * (BOX_W + BOX_GAP) + BOX_W
        x_end = start_x + (i + 1) * (BOX_W + BOX_GAP)
        conn = slide.shapes.add_connector(
            2,  # straight connector
            Inches(x_start), Inches(arrow_y),
            Inches(x_end), Inches(arrow_y),
        )
        conn.line.color.rgb = NAVY
        conn.line.width = Pt(2.5)
        # arrowhead no final (direita)
        line = conn.line
        from pptx.oxml.ns import qn
        ln = line._get_or_add_ln()
        tailEnd = ln.find(qn("a:tailEnd"))
        if tailEnd is None:
            from lxml import etree
            tailEnd = etree.SubElement(ln, qn("a:tailEnd"))
        tailEnd.set("type", "triangle")
        tailEnd.set("w", "med")
        tailEnd.set("len", "med")

    # ---- Caption inferior ----
    cap = slide.shapes.add_textbox(
        Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.4)
    )
    pc = cap.text_frame.paragraphs[0]
    pc.alignment = 2
    pc.text = (
        "Fluxo em três fases: diagnóstico (etapas 1 e 2), "
        "método proposto (etapa 3, destacada) e validação e síntese (etapas 4 a 6)."
    )
    pc.font.size = Pt(12)
    pc.font.italic = True
    pc.font.color.rgb = GRAY_MD

    add_footer(slide)
    add_page_number(slide, 13)


# ============================================================
# SLIDE 13 — Evolução do pipeline (status das 6 etapas em 04/10/2026)
# ============================================================
def slide_evolucao_pipeline(prs: Presentation) -> None:
    add_image_slide(
        prs, 14,
        "Evolução do pipeline experimental — estado em 04/10/2026",
        IMG_DIR / "fig_pipeline_evolucao.png",
        caption=(
            "Etapas 1 a 3 em execução (código implementado, validação preliminar em curso); "
            "etapas 4 a 6 planejadas conforme cronograma. Marco de qualificação assegurado."
        ),
        height_in=5.0,
    )


# ============================================================
# SLIDE 14 — Metodologia: Etapa 1
# ============================================================
def slide_metodologia_etapa1(prs: Presentation) -> None:
    add_bullets(prs, 15, "Etapa 1: classificador MST treinado internamente", [
        ("O que faz:", "produz, para cada imagem facial, o vetor softmax sobre os dez tons da escala Monk."),
        ("Decisão pós-reunião Ago/2026:", "adotar treinamento interno do classificador, reduzindo dependência do release do SkinToneNet (Matias, 2026), cujos pesos e dataset STW ainda não foram divulgados publicamente."),
        ("Datasets de treino:", "Monk Skin Tone Examples (Monk, 2019) e Casual Conversations v2 (Porgali et al., 2023)."),
        ("Camada auto-suficiente:", "detecção facial (MTCNN, Zhang et al., 2016), alinhamento, recorte e classificação — viabiliza inferência em datasets sem anotação MST."),
        ("Validação:", "protocolo humano interno (aproximadamente 250 imagens do FairFace) e sensitivity analysis com dois a três classificadores alternativos."),
    ])


# ============================================================
# SLIDE 15 — Classificador MST em validação: detalhe técnico do backend
# ============================================================
def slide_mst_detalhe_tecnico(prs: Presentation) -> None:
    add_image_slide(
        prs, 16,
        "Classificador MST em validação: detalhe técnico do backend",
        IMG_DIR / "fig_mst_detalhe_tecnico.png",
        caption=(
            "Pipeline técnico em cinco etapas: detecção MTCNN → alinhamento por landmarks → "
            "segmentação de pele → cor dominante via k-means em CIELab → distância euclidiana à paleta Monk. "
            "Backend operacional durante a fase de validação (sensitivity analysis — Cap. 4 §4.2)."
        ),
        height_in=4.6,
    )


# ============================================================
# SLIDE 16 — Validação preliminar sobre FairFace val (grade qualitativa)
# ============================================================
def slide_mst_validacao_faces(prs: Presentation) -> None:
    add_image_slide(
        prs, 17,
        "Classificador MST em validação: amostras FairFace val (qualitativo)",
        IMG_DIR / "fig_mst_demo_faces.png",
        caption=(
            "Para cada raça FairFace: face com tom MST mais claro (acima) e mais escuro (abaixo) "
            "preditos sobre amostra estratificada de 350 imagens. Evidência visual direta de H1 — "
            "heterogeneidade fenotípica intra-categorial."
        ),
        height_in=5.3,
    )


# ============================================================
# SLIDE 17 — Validação preliminar sobre FairFace val (distribuição quantitativa)
# ============================================================
def slide_mst_validacao_distribuicao(prs: Presentation) -> None:
    add_image_slide(
        prs, 18,
        "Classificador MST em validação: distribuição MST × raça (quantitativo)",
        IMG_DIR / "fig_mst_demo_distribuicao.png",
        caption=(
            "Distribuição percentual (dentro de cada raça) sobre amostra estratificada "
            "(n = 350, 50 por raça). Nenhuma raça colapsa em um único tom Monk — "
            "evidência preliminar que sustenta a formalização de H1."
        ),
        height_in=4.8,
    )


# ============================================================
# SLIDE 18 — Por que ConvNeXt-T (justificativa vs ResNet e vs ViT)
# ============================================================
def slide_por_que_convnext(prs: Presentation) -> None:
    add_bullets(prs, 19, "Racional da escolha do backbone: ConvNeXt-T (Liu et al., 2022)", [
        ("Paridade com ViTs a custo convolucional:",
         "82 % top-1 ImageNet, comparável a Swin-T, ~1/3 dos params."),
        ("Estável em fine-tuning:",
         "LayerNorm (não BatchNorm) — robusto a batch pequeno, essencial para 3 sementes."),
        ("Inserção natural de FiLM:",
         "4 estágios hierárquicos oferecem 4 pontos de inserção sem modificar blocos internos."),
        ("Comparável ao baseline canônico:",
         "ResNet-34 do FairFace serve como âncora — ConvNeXt-T isola o efeito do condicionamento."),
        ("Viável computacionalmente:",
         "28 M params permitem 3 sementes × 3 configs em GPU comum (~300 h total)."),
    ])


# ============================================================
# SLIDE 19 — Por que FiLM (comparação com 7 alternativas de conditioning)
# ============================================================
def slide_por_que_film(prs: Presentation) -> None:
    add_bullets(prs, 20, "Racional do mecanismo de condicionamento: FiLM (Perez et al., 2018)", [
        ("Adequação dimensional ao sinal MST:",
         "sinal 10-dim casa naturalmente com γ, β — sem explosão paramétrica."),
        ("Eficiência:",
         "~380 k parâmetros (~1,3 % do backbone) — muito abaixo de cross-attention (~3×)."),
        ("Interpretabilidade direta:",
         "γ e β por canal permitem inspecionar como cada tom modula as features."),
        ("Compatibilidade nativa:",
         "opera bem com LayerNorm do ConvNeXt-T; init identidade preserva backbone."),
        ("Lacuna documentada:",
         "sem aplicação prévia em fairness facial multi-classe — Contribuição 3 desta pesquisa."),
        ("Descartadas com justificativa formal (Cap 2):",
         "Concatenação, CBN, Cross-attention, AdaIN, SPADE, HyperNetworks, LoRA/Adaptadores."),
    ])


# ============================================================
# SLIDE 20 — Metodologia: mecanismo FiLM (figura)
# ============================================================
def slide_metodologia_film(prs: Presentation) -> None:
    add_image_slide(
        prs, 21,
        "Mecanismo FiLM: modulação de features condicionada ao tom de pele",
        IMG_DIR / "film_pipeline.png",
        caption="FiLM (Perez et al., 2018) modula as features intermediárias do ConvNeXt-T canal a canal, condicionadas ao vetor MST. Overhead paramétrico: aproximadamente 1,3 % do backbone.",
        height_in=4.6,
    )


# ============================================================
# SLIDE 22 — Metodologia: 3 configurações (shapes nativos PPTX)
# ============================================================
def slide_metodologia_configs(prs: Presentation) -> None:
    """Estudo de ablation arquitetural em 3 configuracoes (A, B, C)
    desenhadas como cartoes nativos lado a lado. Config B destacada
    como proposta principal. Mini-pipeline visual dentro de cada card.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Configurações do estudo de ablation arquitetural")

    # ---- Dados das 3 configuracoes ----
    # (id, nome, sinal_condicionante, oque_testa, referencia, highlight)
    configs = [
        (
            "A",
            "ConvNeXt-T puro",
            None,
            "Controle arquitetural\nsem condicionamento.",
            "Baseline",
            False,
        ),
        (
            "B",
            "ConvNeXt-T + FiLM\n(MST direto, 10-dim)",
            "MST (10-dim)",
            "Proposta principal — tom\nde pele contínuo como\ncontexto arquitetural.",
            "Perez et al. (2018) + Monk\n(Schumann et al., 2023)",
            True,
        ),
        (
            "C",
            "ConvNeXt-T + FiLM\n(CLIP-text, 512-dim)",
            "CLIP-text (512-dim)",
            "Alternativa moderna —\nembedding textual rico\ncomo sinal condicionante.",
            "Radford et al. (2021) +\nDehdashtian et al. (2024)",
            False,
        ),
    ]

    # ---- Geometria ----
    SLIDE_W = 13.33
    CARD_W = 4.00
    CARD_GAP = 0.30
    CARD_H = 5.05
    CARD_Y = 1.45
    n = len(configs)
    total_w = n * CARD_W + (n - 1) * CARD_GAP
    start_x = (SLIDE_W - total_w) / 2

    # ---- Cartoes ----
    for i, (cid, nome, sinal, oque, ref, hl) in enumerate(configs):
        x = start_x + i * (CARD_W + CARD_GAP)

        # Card background
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y), Inches(CARD_W), Inches(CARD_H),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if hl else WHITE
        box.line.color.rgb = NAVY
        box.line.width = Pt(2.4 if hl else 1.3)
        box.text_frame.margin_top = Inches(0.0)

        # Letter badge (circulo com letra no topo)
        badge_d = 0.75
        badge_x = x + (CARD_W - badge_d) / 2
        badge_y = CARD_Y + 0.25
        badge = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(badge_x), Inches(badge_y),
            Inches(badge_d), Inches(badge_d),
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = WHITE if hl else NAVY
        badge.line.color.rgb = NAVY
        badge.line.width = Pt(1.5)
        bf = badge.text_frame
        bf.margin_top = Inches(0.0); bf.margin_bottom = Inches(0.0)
        bp = bf.paragraphs[0]
        bp.alignment = 2
        br = bp.add_run()
        br.text = cid
        br.font.size = Pt(28)
        br.font.bold = True
        br.font.color.rgb = NAVY if hl else WHITE

        # Nome da configuracao (abaixo do badge)
        name_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(CARD_Y + 1.10),
            Inches(CARD_W - 0.30), Inches(0.95),
        )
        tf = name_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = nome
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = WHITE if hl else NAVY

        # Mini-pipeline visual (configuracao arquitetural)
        mini_y = CARD_Y + 2.10
        _mini_pipeline(slide, x + 0.20, mini_y, CARD_W - 0.40,
                       sinal, hl)

        # "O que testa" (corpo descritivo)
        body_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + 3.30),
            Inches(CARD_W - 0.40), Inches(1.00),
        )
        tf = body_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = oque
        r.font.size = Pt(13)
        r.font.color.rgb = BLUE_LIGHT if hl else GRAY_DK

        # Referencia / rodape do card
        ref_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + CARD_H - 0.75),
            Inches(CARD_W - 0.40), Inches(0.65),
        )
        tf = ref_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = ref
        r.font.size = Pt(10.5)
        r.font.italic = True
        r.font.color.rgb = BLUE_LIGHT if hl else GRAY_MD

    # ---- Caption ----
    cap = slide.shapes.add_textbox(
        Inches(0.5), Inches(6.60), Inches(12.3), Inches(0.4)
    )
    pc = cap.text_frame.paragraphs[0]
    pc.alignment = 2
    pc.text = (
        "Ceteris paribus: mesmo backbone (ConvNeXt-T), mesmo dataset "
        "(FairFace), mesmas três sementes, mesma receita de treino — "
        "varia-se apenas o sinal condicionante."
    )
    pc.font.size = Pt(12)
    pc.font.italic = True
    pc.font.color.rgb = GRAY_MD

    add_footer(slide)
    add_page_number(slide, 22)


def _mini_pipeline(slide, x: float, y: float, width: float,
                   signal: str | None, highlight: bool) -> None:
    """Desenha mini-pipeline topologicamente correto para cada config:

    Config A (sem sinal):
            [Imagem]  ->  [ConvNeXt-T]

    Config B / C (com sinal condicionante):
            [Imagem]  ───┐
                         ├─▶ [FiLM · ConvNeXt-T]
            [sinal]  ────┘      (sinal nao entra no backbone;
                                 modula features via gamma, beta)
    """
    pill_h = 0.42

    def _pill(px, py, pw, label, fc, ec, fg, size=10, bold=True):
        pill = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(px), Inches(py),
            Inches(pw), Inches(pill_h),
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = fc
        pill.line.color.rgb = ec
        pill.line.width = Pt(1.1)
        tf = pill.text_frame
        tf.margin_left = Inches(0.03); tf.margin_right = Inches(0.03)
        tf.margin_top = Inches(0.01); tf.margin_bottom = Inches(0.01)
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = label
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = fg

    if signal is None:
        # Config A: [Imagem] -> [ConvNeXt-T] (unica linha, centrada em y)
        row_y = y + 0.28
        w1 = width * 0.38
        w2 = width * 0.48
        gap = width - w1 - w2
        _pill(x,              row_y, w1, "Imagem",
              GRAY_LT, GRAY_DK, GRAY_DK)
        _pill(x + w1 + gap,   row_y, w2, "ConvNeXt-T",
              WHITE,  NAVY,   NAVY)
        conn = slide.shapes.add_connector(
            2,
            Inches(x + w1),       Inches(row_y + pill_h / 2),
            Inches(x + w1 + gap), Inches(row_y + pill_h / 2),
        )
        conn.line.color.rgb = GRAY_DK
        conn.line.width = Pt(2.0)
        _add_arrowhead(conn)
    else:
        # Config B / C: 2 inputs convergindo no FiLM.ConvNeXt-T
        #   [Imagem]  --->
        #                 ---->  [FiLM . ConvNeXt-T]
        #   [sinal]   --->
        w_in = width * 0.42
        w_out = width * 0.48
        gap = width - w_in - w_out
        in_x = x
        out_x = x + w_in + gap
        img_y = y + 0.05
        sig_y = y + 0.55
        target_y = y + 0.30  # centro vertical do bloco (entre img e sig)

        # Pilula [Imagem] (sempre fundo neutro, mesmo quando card highlight)
        _pill(in_x, img_y, w_in, "Imagem",
              GRAY_LT, GRAY_DK, GRAY_DK)
        # Pilula [sinal condicionante]
        signal_fc = BLUE_MID if not highlight else BLUE_LIGHT
        signal_fg = WHITE if not highlight else NAVY
        _pill(in_x, sig_y, w_in, signal,
              signal_fc, NAVY, signal_fg)
        # Pilula [FiLM . ConvNeXt-T] no destino
        target_fc = WHITE if not highlight else BLUE_LIGHT
        _pill(out_x, target_y, w_out, "FiLM · ConvNeXt-T",
              target_fc, NAVY, NAVY)

        # Seta superior: [Imagem] -> [FiLM.ConvNeXt-T]
        conn_img = slide.shapes.add_connector(
            2,
            Inches(in_x + w_in),  Inches(img_y + pill_h / 2),
            Inches(out_x),        Inches(target_y + pill_h / 2),
        )
        conn_img.line.color.rgb = GRAY_DK
        conn_img.line.width = Pt(1.8)
        _add_arrowhead(conn_img)

        # Seta inferior: [sinal] -> [FiLM.ConvNeXt-T] (condicionamento)
        conn_sig = slide.shapes.add_connector(
            2,
            Inches(in_x + w_in),  Inches(sig_y + pill_h / 2),
            Inches(out_x),        Inches(target_y + pill_h / 2),
        )
        conn_sig.line.color.rgb = BLUE_MID if not highlight else WHITE
        conn_sig.line.width = Pt(1.8)
        _add_arrowhead(conn_sig)


def _add_arrowhead(connector) -> None:
    """Adiciona arrowhead triangular ao final de um connector."""
    from pptx.oxml.ns import qn
    from lxml import etree
    ln = connector.line._get_or_add_ln()
    tailEnd = ln.find(qn("a:tailEnd"))
    if tailEnd is None:
        tailEnd = etree.SubElement(ln, qn("a:tailEnd"))
    tailEnd.set("type", "triangle")
    tailEnd.set("w", "med")
    tailEnd.set("len", "med")


# ============================================================
# SLIDE 20 — Baselines + Cenários
# ============================================================
def slide_baselines_cenarios(prs: Presentation) -> None:
    add_bullets(prs, 23, "Protocolo de validação: cenários e norma", [
        ("Cenário A — apenas raça:", "reporte estratificado por classe racial (sete classes do FairFace)."),
        ("Cenário B — raça × gênero:", "análise interseccional (oito subgrupos), conforme protocolo estabelecido por Gender Shades (Buolamwini & Gebru, 2018)."),
        ("Norma seguida:", "ISO/IEC 19795-10:2024, padrão internacional para reporte de desempenho biométrico estratificado entre grupos demográficos."),
        ("Rigor experimental:", "três sementes independentes por experimento (42, 1, 2), comparação pareada e intervalo de confiança de 95 % via bootstrap não paramétrico."),
        ("Datasets de transferência (Etapa 5):", "RFW (Wang et al., 2019) e BFW (Robinson et al., 2020), com pares oficiais de verificação 1:1."),
    ])


# ============================================================
# SLIDE 21 — Triangulação de métricas
# ============================================================
def slide_metricas(prs: Presentation) -> None:
    add_bullets(prs, 24, "Triangulação de métricas de equidade", [
        ("Por que triangular — Kleinberg et al. (2017):", "Teorema da Impossibilidade — demonstra a impossibilidade formal de satisfação simultânea de múltiplas definições de equidade quando as prevalências diferem entre grupos."),
        ("Disparity Ratio:", "razão entre F1 mínimo e máximo entre grupos; mensura a desigualdade de desempenho."),
        ("F1 da pior classe:", "protege o grupo sub-representado; evita ganhos concentrados apenas na média agregada."),
        ("Equal Opportunity — Hardt et al. (2016, NeurIPS):", "requer igualdade de taxa de verdadeiro positivo (TPR) condicionada ao rótulo verdadeiro."),
        ("Equalized Odds — Hardt et al. (2016):", "requisito estrito: igualdade simultânea de TPR e FPR entre grupos."),
        ("Visualização Pareto:", "identificação da fronteira Pareto no trade-off acurácia agregada × disparidade demográfica."),
    ])


# ============================================================
# SLIDE 22 — Contribuições
# ============================================================
def slide_contribuicoes(prs: Presentation) -> None:
    add_table_slide(
        prs, 25,
        "Contribuições esperadas",
        ["Eixo", "Contribuições", "Foco"],
        [
            ["Fenotípico-empírico", "1, 2", "Documentar como o fenótipo (MST) se distribui dentro de cada rótulo racial."],
            ["Metodológico-arquitetural", "3, 4, 7", "Injetar tom de pele como contexto arquitetural + triangulação de métricas."],
            ["Diagnóstico-estrutural", "5, 6", "Decompor o erro em fenotípico (irredutível) e algorítmico (mitigável)."],
        ],
        col_widths=[3.5, 2.0, 7.0],
        font_size=15,
    )


# ============================================================
# SLIDE 23 — Cronograma
# ============================================================
def slide_cronograma(prs: Presentation) -> None:
    add_table_slide(
        prs, 26,
        "Cronograma",
        ["Período", "Etapa", "Entrega"],
        [
            ["Ago–Set/2026", "Preparação adiantada", "Código das 6 etapas pronto (120 testes passando)."],
            ["05/10/2026", "QUALIFICAÇÃO", "Marco atual."],
            ["Out/2026", "Aplicar sugestões da banca", "Ajustes de texto e escopo."],
            ["Nov–Dez/2026", "Etapas 1 e 2 formais", "Treino MST + matriz pública MST × raça."],
            ["Jan–Mar/2027", "Etapa 3 (ablation)", "3 configurações A/B/C do FiLM."],
            ["Abr–Mai/2027", "Etapas 4 e 5", "6 baselines + transferência RFW/BFW."],
            ["Jun/2027", "Etapa 6", "Síntese decompositiva."],
            ["2º sem 2027", "Redação final + DEFESA", "Encerramento."],
        ],
        col_widths=[2.5, 3.5, 6.5],
        highlight_rows=[0, 1, 7],
        font_size=16,
    )


# ============================================================
# SLIDE 24 — Riscos
# ============================================================
def slide_riscos(prs: Presentation) -> None:
    add_table_slide(
        prs, 27,
        "Riscos identificados e estratégias de mitigação",
        ["#", "Risco", "Mitigação"],
        [
            ["R1", "Alguma das 6 hipóteses pode ser refutada.", "Cada hipótese tem plano B; refutação é resultado científico válido."],
            ["R2", "Qualidade do nosso classificador de tom de pele.", "Validação humana + sensitivity com outros classificadores + benchmark externo (STW) quando disponível."],
            ["R3", "Refutação parcial pela tese Pangelinan (pixel info > tom).", "Já incluída como H6 — refutação vira contribuição quantitativa."],
            ["R4", "Custo computacional (3 sementes × 3 configs × 6 baselines).", "ConvNeXt-T é leve (~28M params); estimativa 200–400 h GPU total."],
        ],
        col_widths=[0.6, 4.5, 7.4],
        highlight_rows=[1],
        font_size=14,
    )


# ============================================================
# SLIDE 25 — Estado atual (KPIs visuais)
# ============================================================
def slide_estado_atual(prs: Presentation) -> None:
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Estado atual do trabalho")

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
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = BLUE_LIGHT if i == 0 else GRAY_MD
        p.alignment = 2

    subtitle = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.5), Inches(0.4))
    p = subtitle.text_frame.paragraphs[0]
    p.text = "Estado das entregas em 28/09/2026"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.5), Inches(2.2))
    tf = tx.text_frame
    tf.word_wrap = True
    entregas = [
        ("Etapas 1 e 2 (código completo):", "classificador MST treinado internamente, camada de preprocessamento auto-suficiente, matriz MST × raça, teste formal de H2."),
        ("Etapa 3 (FiLM):", "camada FiLM, wrapper ConvNeXt-T, ensembler CLIP-text e pipeline de treinamento das três configurações."),
        ("Etapas 4, 5 e 6:", "métricas de equidade, quatro baselines, análise Pareto, verificação em RFW/BFW e decomposição ANOVA/R²."),
        ("Texto da dissertação:", "cinco capítulos consolidados, três passadas de revisão de estilo e 104 fichas bibliográficas."),
    ]
    for i, (head, body) in enumerate(entregas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "•  " + head + "  "
        r1.font.size = Pt(16)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)

    add_footer(slide)
    add_page_number(slide, 28)


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

    # 29 slides (era 28): adicionada ancora academica de motivacao em slide 2
    # — reescritura formal do episodio Buolamwini (MIT Media Lab, 2016),
    # consolidado em Gender Shades (2018) e Unmasking AI (2023).
    slide_capa(prs)                           # 1
    slide_motivacao_academica(prs)            # 2   [NOVO — origem academica Buolamwini]
    slide_motivacao_contexto(prs)             # 3
    slide_motivacao_regulacao(prs)            # 4
    slide_problema_disparidade(prs)           # 5
    slide_problema_heterogeneidade(prs)       # 6
    slide_problema_refutacao(prs)             # 7
    slide_objetivo_geral(prs)                 # 8
    slide_objetivos_hipoteses(prs)            # 9   [CONSOLIDADO]
    slide_revisao_timeline(prs)               # 10
    slide_revisao_mitigacao(prs)              # 11
    slide_revisao_lacunas(prs)                # 12
    slide_metodologia_pipeline(prs)           # 13
    slide_evolucao_pipeline(prs)              # 14
    slide_metodologia_etapa1(prs)             # 15
    slide_mst_detalhe_tecnico(prs)            # 16
    slide_mst_validacao_faces(prs)            # 17
    slide_mst_validacao_distribuicao(prs)     # 18
    slide_por_que_convnext(prs)               # 19
    slide_por_que_film(prs)                   # 20
    slide_metodologia_film(prs)               # 21
    slide_metodologia_configs(prs)            # 22
    slide_baselines_cenarios(prs)             # 23
    slide_metricas(prs)                       # 24
    slide_contribuicoes(prs)                  # 25
    slide_cronograma(prs)                     # 26
    slide_riscos(prs)                         # 27
    slide_estado_atual(prs)                   # 28
    slide_perguntas(prs)                      # 29

    return prs


def main() -> None:
    prs = build_presentation()
    out_dir = Path(__file__).parent
    out = out_dir / "material_qualificacao_2026-10-05.pptx"
    prs.save(out)
    print(f"OK: {out}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
