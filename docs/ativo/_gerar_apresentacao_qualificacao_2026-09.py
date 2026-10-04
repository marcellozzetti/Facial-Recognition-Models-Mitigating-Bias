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
TOTAL_SLIDES = 32

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
    """Linha do tempo em 4 fases: figura compacta no topo + 4 grupos
    de texto abaixo (foco, marcos, achado/diferencial)."""
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Linha do tempo das principais respostas ao problema (2018 – 2026)")

    # ---- Figura das 4 fases no topo ----
    img = IMG_DIR / "fig_timeline_fases.png"
    if img.exists():
        pic = slide.shapes.add_picture(
            str(img), Inches(0.5), Inches(1.30),
            width=Inches(12.3),
        )
    # Altura total da figura ~ 1.9in

    # ---- 4 grupos de texto abaixo (foco, marcos, achado) ----
    # (foco, marcos_list, achado_label, achado_text, destaque)
    grupos = [
        (
            "Rebalanceamento amostral e novos\nrepositórios biométricos.",
            [
                "RFW (Wang et al., 2019)",
                "BFW (Robinson et al., 2020)",
                "FairFace (Kärkkäinen & Joo, 2021)",
            ],
            "Limitação",
            "Equilíbrio amostral é necessário mas insuficiente em\n"
            "populações heterogêneas (Latinx). Uso restrito ao\n"
            "Fitzpatrick I–VI.",
            False,
        ),
        (
            "Intervenções na otimização e nas\nfunções de perda.",
            [
                "Group DRO (Sagawa et al., 2020)",
                "FSCL+ (Park et al., 2022)",
                "FairGRAPE (Lin et al., 2022)",
            ],
            "Limitação",
            "Conflito severo entre acurácia global e paridade\n"
            "demográfica (Teorema da Impossibilidade,\n"
            "Kleinberg et al., 2017).",
            False,
        ),
        (
            "Modelos visão-linguagem e\ntransição à escala Monk (MST).",
            [
                "FaceScanPaliGemma (AlDahoul et al., 2024)",
                "FairCLIP (Luo et al., 2024)",
                "MST (Schumann et al., 2023; Pereira, 2026)",
            ],
            "Limitação",
            "Acurácia agregada sobe (75,7 %) mas persiste gap\n"
            "de ~30 pp em Latinx (F1 ≈ 60 %); vieses herdados\n"
            "da web.",
            False,
        ),
        (
            "Injeção de MST como sinal arquitetural\nintermediário via FiLM.",
            [
                "FiLM · ConvNeXt-T (Perez et al., 2018)",
                "Triangulação ISO/IEC 19795-10:2024",
                "Decomposição fenotípico × pixel (H6)",
            ],
            "Diferencial",
            "Modulação do fluxo de features pelo tom contínuo —\n"
            "supera rótulos raciais nominais sem exigir modelos\n"
            "de bilhões de parâmetros.",
            True,
        ),
    ]

    SLIDE_W = 13.33
    MARGIN_X = 0.5
    N = 4
    GAP = 0.12
    COL_W = (SLIDE_W - 2 * MARGIN_X - (N - 1) * GAP) / N
    COL_Y = 3.45
    COL_H = 3.10

    for i, (foco, marcos, lbl_achado, achado, destaque) in enumerate(grupos):
        x = MARGIN_X + i * (COL_W + GAP)

        # Linha vertical lateral (acento colorido)
        cor_barra = NAVY if destaque else BLUE_MID
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(COL_Y),
            Inches(0.07), Inches(COL_H),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = cor_barra
        bar.line.fill.background()

        # Foco (eyebrow + conteudo)
        foco_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(COL_Y + 0.00),
            Inches(COL_W - 0.15), Inches(0.80),
        )
        tf = foco_tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = "FOCO"
        r0.font.size = Pt(9)
        r0.font.bold = True
        r0.font.color.rgb = cor_barra
        p1 = tf.add_paragraph()
        r1 = p1.add_run()
        r1.text = foco
        r1.font.size = Pt(11)
        r1.font.color.rgb = GRAY_DK

        # Marcos (eyebrow + lista)
        marcos_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(COL_Y + 0.95),
            Inches(COL_W - 0.15), Inches(1.25),
        )
        tf = marcos_tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = "MARCOS"
        r0.font.size = Pt(9)
        r0.font.bold = True
        r0.font.color.rgb = cor_barra
        for j, m in enumerate(marcos):
            p = tf.add_paragraph()
            r = p.add_run()
            r.text = "•  " + m
            r.font.size = Pt(10.5)
            r.font.color.rgb = NAVY

        # Achado / Diferencial
        ach_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(COL_Y + 2.25),
            Inches(COL_W - 0.15), Inches(0.85),
        )
        tf = ach_tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = lbl_achado.upper()
        r0.font.size = Pt(9)
        r0.font.bold = True
        r0.font.color.rgb = cor_barra
        p1 = tf.add_paragraph()
        r1 = p1.add_run()
        r1.text = achado
        r1.font.size = Pt(10.5)
        r1.font.italic = True
        r1.font.color.rgb = GRAY_DK

    # Caption inferior
    cap = slide.shapes.add_textbox(
        Inches(0.5), Inches(6.65), Inches(12.3), Inches(0.35)
    )
    pc = cap.text_frame.paragraphs[0]
    pc.alignment = 2
    pc.text = (
        "Fluxo cronológico: dados balanceados → funções de perda → "
        "modelos visão-linguagem com MST → condicionamento arquitetural."
    )
    pc.font.size = Pt(11)
    pc.font.italic = True
    pc.font.color.rgb = GRAY_MD

    add_footer(slide)
    add_page_number(slide, 10)


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
    """Etapa 1: pipeline neural horizontal + 2 colunas de contexto.

    Pipeline: Rosto -> MTCNN+Alinhamento -> ViT-S/ResNet-34 fine-tuned
    -> Softmax 10-dim (paleta Monk destacada).

    Duas colunas abaixo:
      - Decisao de Autonomia Metodologica (NAVY)
      - Arquitetura e Saida Esperada     (ACCENT)
    """
    MST_PAL = [
        RGBColor(0xF6, 0xED, 0xE4), RGBColor(0xF3, 0xE7, 0xDB),
        RGBColor(0xF7, 0xEA, 0xD0), RGBColor(0xEA, 0xDA, 0xBA),
        RGBColor(0xD7, 0xBD, 0x96), RGBColor(0xA0, 0x7E, 0x56),
        RGBColor(0x82, 0x5C, 0x43), RGBColor(0x60, 0x41, 0x34),
        RGBColor(0x3A, 0x31, 0x2A), RGBColor(0x29, 0x24, 0x20),
    ]

    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 1 — Classificador MST interno")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Para cada rosto, produz vetor softmax 10-dim sobre a escala "
        "Monk — sinal condicionante estático para a Etapa 3 (FiLM)."
    )
    ps.font.size = Pt(13)
    ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ============================================================
    # PIPELINE neural horizontal (hero)
    # ============================================================
    SLIDE_W = 13.33
    PIPE_Y = 1.80
    PIPE_H = 1.25

    passos = [
        ("Rosto\n224 × 224",            "entrada bruta",         GRAY_LT, GRAY_DK),
        ("MTCNN\ndetecção + alinhamento", "Zhang 2016",          WHITE,   NAVY),
        ("Crop à face\n(normalizada)",   "pose canônica",        WHITE,   NAVY),
        ("ViT-S  /  ResNet-34\nfine-tuned para MST", "backbone neural", BLUE_MID, WHITE),
    ]
    n = len(passos)
    PAL_SLOT_W = 3.80   # espaco reservado para paleta a direita
    MARGIN = 0.50
    ARROW_SPACE = 0.22
    total_arrows = n * ARROW_SPACE  # inclui seta final para paleta
    usable = SLIDE_W - 2 * MARGIN - total_arrows - PAL_SLOT_W
    pw = usable / n

    for i, (titulo, nota, fc, fg) in enumerate(passos):
        px = MARGIN + i * (pw + ARROW_SPACE)
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(px), Inches(PIPE_Y),
            Inches(pw), Inches(PIPE_H),
        )
        box.fill.solid(); box.fill.fore_color.rgb = fc
        box.line.color.rgb = NAVY; box.line.width = Pt(1.6)
        tf = box.text_frame
        tf.margin_top = Inches(0.12); tf.margin_bottom = Inches(0.06)
        p = tf.paragraphs[0]; p.alignment = 2
        r = p.add_run()
        r.text = titulo
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = fg
        p2 = tf.add_paragraph(); p2.alignment = 2
        r2 = p2.add_run()
        r2.text = nota
        r2.font.size = Pt(9.5)
        r2.font.italic = True
        r2.font.color.rgb = GRAY_LT if fc in (NAVY, BLUE_MID) else GRAY_MD

        # Seta
        cx0 = px + pw
        cy = PIPE_Y + PIPE_H / 2
        conn = slide.shapes.add_connector(
            2,
            Inches(cx0), Inches(cy),
            Inches(cx0 + ARROW_SPACE), Inches(cy),
        )
        conn.line.color.rgb = NAVY; conn.line.width = Pt(2.2)
        _add_arrowhead(conn)

    # ============================================================
    # Paleta Monk (saida) a direita do pipeline
    # ============================================================
    PAL_X = MARGIN + n * (pw + ARROW_SPACE)
    PAL_Y = PIPE_Y + 0.08
    PAL_W = PAL_SLOT_W - 0.10
    PAL_H = 0.70
    cell_w = PAL_W / 10
    highlight_idx = 6

    # Mini rotulo acima
    hdr = slide.shapes.add_textbox(
        Inches(PAL_X), Inches(PAL_Y - 0.32),
        Inches(PAL_W), Inches(0.28),
    )
    ph = hdr.text_frame.paragraphs[0]; ph.alignment = 2
    rh = ph.add_run()
    rh.text = "Softmax 10-dim · paleta Monk"
    rh.font.size = Pt(10)
    rh.font.bold = True
    rh.font.italic = True
    rh.font.color.rgb = ACCENT

    for i in range(10):
        cx = PAL_X + i * cell_w
        cell = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(cx), Inches(PAL_Y),
            Inches(cell_w - 0.02), Inches(PAL_H),
        )
        cell.fill.solid(); cell.fill.fore_color.rgb = MST_PAL[i]
        if i == highlight_idx:
            cell.line.color.rgb = ACCENT; cell.line.width = Pt(2.8)
        else:
            cell.line.color.rgb = GRAY_DK; cell.line.width = Pt(0.5)
        tn = slide.shapes.add_textbox(
            Inches(cx), Inches(PAL_Y + PAL_H + 0.02),
            Inches(cell_w - 0.02), Inches(0.22),
        )
        pn = tn.text_frame.paragraphs[0]; pn.alignment = 2
        rn = pn.add_run()
        rn.text = str(i + 1)
        rn.font.size = Pt(9)
        rn.font.bold = (i == highlight_idx)
        rn.font.color.rgb = ACCENT if i == highlight_idx else GRAY_DK

    # Nota: 1 vetor por imagem
    nota = slide.shapes.add_textbox(
        Inches(PAL_X), Inches(PAL_Y + PAL_H + 0.28),
        Inches(PAL_W), Inches(0.3),
    )
    pn2 = nota.text_frame.paragraphs[0]; pn2.alignment = 2
    rn2 = pn2.add_run()
    rn2.text = "1 vetor por imagem"
    rn2.font.size = Pt(9)
    rn2.font.italic = True
    rn2.font.color.rgb = GRAY_MD

    # ============================================================
    # Duas colunas de contexto (abaixo do pipeline)
    # ============================================================
    COL_Y = 3.55
    COL_H = 2.95
    COL_W = 6.00
    COL_GAP = 0.33
    col1_x = (SLIDE_W - 2 * COL_W - COL_GAP) / 2
    col2_x = col1_x + COL_W + COL_GAP

    colunas = [
        {
            "titulo": "Decisão de Autonomia Metodológica",
            "cor_titulo": NAVY,
            "itens": [
                ("Decisão:",   "treinar classificador próprio — sem dependência de modelos de terceiros (ex. SkinToneNet, cujo release ainda não é público)."),
                ("Datasets:",  "Monk Skin Tone Examples (MSTE) + Casual Conversations v2 (Meta AI)."),
                ("Validação:", "subset FairFace anotado por especialistas humanos — concordância κ ≥ 0,70 com o modelo (H1)."),
            ],
        },
        {
            "titulo": "Arquitetura e Saída Esperada",
            "cor_titulo": ACCENT,
            "itens": [
                ("Backbone:",   "ViT-Small ou ResNet-34 fine-tuned para classificação na escala MST de 10 tons."),
                ("Saída:",      "vetor de probabilidade softmax 10-dim (um por imagem)."),
                ("Função no pipeline:", "sinal condicionante estático para as camadas FiLM da Etapa 3 — também alimenta a matriz da Etapa 2."),
            ],
        },
    ]

    for col_i, col in enumerate(colunas):
        x = col1_x if col_i == 0 else col2_x
        # Header
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(COL_Y),
            Inches(COL_W), Inches(0.50),
        )
        hdr.fill.solid(); hdr.fill.fore_color.rgb = col["cor_titulo"]
        hdr.line.fill.background()
        tf = hdr.text_frame
        tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]; p.alignment = 2
        r = p.add_run()
        r.text = col["titulo"]
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = WHITE

        # Corpo
        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(COL_Y + 0.53),
            Inches(COL_W), Inches(COL_H - 0.53),
        )
        body.fill.solid(); body.fill.fore_color.rgb = WHITE
        body.line.color.rgb = col["cor_titulo"]
        body.line.width = Pt(1.2)

        # Bullets
        tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(COL_Y + 0.65),
            Inches(COL_W - 0.40), Inches(COL_H - 0.70),
        )
        tf = tb.text_frame; tf.word_wrap = True
        for j, (head, body_txt) in enumerate(col["itens"]):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            r1 = p.add_run()
            r1.text = "•  " + head + "  "
            r1.font.size = Pt(12)
            r1.font.bold = True
            r1.font.color.rgb = col["cor_titulo"]
            r2 = p.add_run()
            r2.text = body_txt
            r2.font.size = Pt(12)
            r2.font.color.rgb = GRAY_DK
            p.space_after = Pt(5)

    # ============================================================
    # Banner inferior
    # ============================================================
    bot_y = 6.75; bot_h = 0.55
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.03)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = (
        "Fundação do pipeline — alimenta diretamente a Etapa 2 (matriz MST × raça) "
        "e a Etapa 3 (FiLM)."
    )
    r1.font.size = Pt(12); r1.font.bold = True
    r1.font.color.rgb = WHITE

    add_footer(slide)
    add_page_number(slide, 15)


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
# SLIDE 20 — Etapa 3 (a): Arquitetura ConvNeXt-T (figura + texto)
# ============================================================
def slide_por_que_convnext(prs: Presentation) -> None:
    """Slide 20 — Etapa 3 (A): figura do backbone ConvNeXt-T + bullets."""
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 3 (a) — Arquitetura ConvNeXt-T (backbone)")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Backbone convolucional moderno — 4 estágios hierárquicos, "
        "28 M parâmetros, LayerNorm (Liu et al., 2022, CVPR)."
    )
    ps.font.size = Pt(13); ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ============================================================
    # LADO ESQUERDO — figura (convnext_backbone.png, aspect 2.013:1)
    # ============================================================
    fig_x = 0.40
    fig_y = 1.80
    fig_w = 7.60
    fig_h = fig_w / 2.013  # ~3.78
    slide.shapes.add_picture(
        str(IMG_DIR / "convnext_backbone.png"),
        Inches(fig_x), Inches(fig_y),
        width=Inches(fig_w), height=Inches(fig_h),
    )

    # ============================================================
    # LADO DIREITO — bullets de justificativa
    # ============================================================
    txt_x = fig_x + fig_w + 0.30
    txt_y = 1.80
    txt_w = 13.33 - txt_x - 0.40
    txt_h = 4.60

    # Header do bloco
    hdr = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(txt_x), Inches(txt_y),
        Inches(txt_w), Inches(0.42),
    )
    hdr.fill.solid(); hdr.fill.fore_color.rgb = NAVY
    hdr.line.fill.background()
    tf = hdr.text_frame
    tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "Por que ConvNeXt-T"
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE

    body = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(txt_x), Inches(txt_y + 0.46),
        Inches(txt_w), Inches(txt_h - 0.46),
    )
    body.fill.solid(); body.fill.fore_color.rgb = WHITE
    body.line.color.rgb = NAVY; body.line.width = Pt(1.2)

    bullets = [
        ("Paridade com ViTs a custo convolucional:",
         "82 % top-1 ImageNet, comparável a Swin-T, ~1/3 dos parâmetros."),
        ("Estável em fine-tuning:",
         "LayerNorm (não BatchNorm) — robusto a batch pequeno, essencial para 3 sementes."),
        ("Inserção natural de FiLM:",
         "4 estágios hierárquicos oferecem 4 pontos de inserção sem modificar blocos internos."),
        ("Comparável ao baseline canônico:",
         "ResNet-34 do FairFace serve como âncora — ConvNeXt-T isola o efeito do condicionamento."),
        ("Viável computacionalmente:",
         "28 M params permitem 3 sementes × 3 configs em GPU comum (~300 h total)."),
    ]
    tb = slide.shapes.add_textbox(
        Inches(txt_x + 0.14), Inches(txt_y + 0.56),
        Inches(txt_w - 0.28), Inches(txt_h - 0.60),
    )
    tf = tb.text_frame; tf.word_wrap = True
    for j, (head, body_txt) in enumerate(bullets):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "•  " + head + "  "
        r1.font.size = Pt(10.5); r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body_txt
        r2.font.size = Pt(10.5); r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(4)

    # Banner inferior
    bot_y = 6.60; bot_h = 0.70
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(12.33), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = (
        "Escolha arquitetural que combina estabilidade de treino, "
        "paridade de desempenho e naturalidade para condicionamento."
    )
    r1.font.size = Pt(12); r1.font.bold = True
    r1.font.color.rgb = WHITE

    add_footer(slide)
    add_page_number(slide, 20)


# ============================================================
# SLIDE 21 — Etapa 3 (b): FiLM mecanismo (figura + 2 colunas)
# ============================================================
def slide_por_que_film(prs: Presentation) -> None:
    """Slide 21 — Etapa 3 (B): figura FiLM pipeline + 2 colunas (O que
    e / Por que escolhemos). Merge do antigo slide_metodologia_film."""
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 3 (b) — Mecanismo FiLM (condicionamento)")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Feature-wise Linear Modulation (Perez et al., 2018) — "
        "4 camadas FiLM, uma após cada estágio ConvNeXt-T."
    )
    ps.font.size = Pt(13); ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ============================================================
    # LADO ESQUERDO — figura film_pipeline.png (aspect 1.808:1)
    # ============================================================
    fig_x = 0.40
    fig_y = 1.80
    fig_w = 7.20
    fig_h = fig_w / 1.808  # ~3.98
    slide.shapes.add_picture(
        str(IMG_DIR / "film_pipeline.png"),
        Inches(fig_x), Inches(fig_y),
        width=Inches(fig_w), height=Inches(fig_h),
    )

    # ============================================================
    # LADO DIREITO — 2 subsecoes empilhadas
    # ============================================================
    sub_x = fig_x + fig_w + 0.25
    sub_w = 13.33 - sub_x - 0.40
    sub_h = 2.30
    gap = 0.15

    subsecoes = [
        {
            "y": 1.80,
            "cor": NAVY,
            "titulo": "O que é FiLM",
            "itens": [
                ("Origem:", "Perez et al. 2018 (AAAI); generaliza CBN de Vries 2017."),
                ("Mecanismo:", "para cada canal, par (γ, β) aprendido do contexto."),
                ("Inserção:", "após cada bloco residual do ConvNeXt-T (4 blocos)."),
                ("Uso:", "~1 000+ papers citantes em CV, NLP, RL desde 2018."),
            ],
        },
        {
            "y": 1.80 + sub_h + gap,
            "cor": ACCENT,
            "titulo": "Por que FiLM aqui",
            "itens": [
                ("Adequação:", "sinal MST 10-dim casa com γ, β — sem explosão."),
                ("Eficiência:", "~380 k params (~1,3 % do backbone) — « cross-attention."),
                ("Interpretabilidade:", "γ, β por canal inspecionáveis."),
                ("Lacuna:", "sem aplicação prévia em fairness facial multi-classe — Contribuição 3."),
            ],
        },
    ]

    for sec in subsecoes:
        # Header
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(sub_x), Inches(sec["y"]),
            Inches(sub_w), Inches(0.38),
        )
        hdr.fill.solid(); hdr.fill.fore_color.rgb = sec["cor"]
        hdr.line.fill.background()
        tf = hdr.text_frame
        tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.03)
        p = tf.paragraphs[0]; p.alignment = 2
        r = p.add_run()
        r.text = sec["titulo"]
        r.font.size = Pt(12); r.font.bold = True
        r.font.color.rgb = WHITE

        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(sub_x), Inches(sec["y"] + 0.41),
            Inches(sub_w), Inches(sub_h - 0.41),
        )
        body.fill.solid(); body.fill.fore_color.rgb = WHITE
        body.line.color.rgb = sec["cor"]; body.line.width = Pt(1.0)

        tb = slide.shapes.add_textbox(
            Inches(sub_x + 0.12), Inches(sec["y"] + 0.48),
            Inches(sub_w - 0.24), Inches(sub_h - 0.52),
        )
        tf = tb.text_frame; tf.word_wrap = True
        for j, (head, body_txt) in enumerate(sec["itens"]):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            r1 = p.add_run()
            r1.text = "•  " + head + "  "
            r1.font.size = Pt(9.5); r1.font.bold = True
            r1.font.color.rgb = sec["cor"]
            r2 = p.add_run()
            r2.text = body_txt
            r2.font.size = Pt(9.5); r2.font.color.rgb = GRAY_DK
            p.space_after = Pt(2)

    # Banner inferior
    bot_y = 6.60; bot_h = 0.70
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(12.33), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Equação canônica:  F′ = γ ⊙ F + β"
    r1.font.size = Pt(13); r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Alternativas descartadas no Cap. 2:  Concatenação · CBN · "
        "Cross-attention · AdaIN · SPADE · HyperNetworks · LoRA/Adaptadores."
    )
    r2.font.size = Pt(10); r2.font.italic = True
    r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 21)


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
    """Protocolo de validacao em 3 pergunta-resposta grandes com um
    numero/badge hero por pergunta. Formato narrativo:
        'O QUE?'   -> 2 cenarios (hero: 7 e 8)
        'COMO?'    -> 3 sementes + IC 95 % + norma (hero: 3)
        'ONDE?'    -> RFW + BFW (hero: 20 K pares)
    Fluxo narrativo linear facilita a fala.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Protocolo de validação")

    # Subtitulo
    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Três perguntas, três respostas pré-registradas — nenhum ajuste "
        "pós-resultado."
    )
    ps.font.size = Pt(13)
    ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ---- Dados das 3 perguntas ----
    # (pergunta, hero_num, hero_sub, linha1, linha2, ref)
    # Cada hero number tem explicacao direta nos bullets abaixo, para
    # que o apresentador consiga responder "o que e isso?" em uma frase.
    perguntas = [
        (
            "O QUE AVALIO?",
            "A · B",
            "dois cenários estratificados",
            "Cenário A — reporte pelas 7 classes raciais do FairFace",
            "Cenário B — reporte pelos 8 subgrupos raça × gênero",
            "Protocolo interseccional (Buolamwini & Gebru, 2018)",
        ),
        (
            "COM QUE RIGOR?",
            "3",
            "sementes · IC 95 % · comparação pareada",
            "Sementes independentes: 42, 1 e 2",
            "Bootstrap não-paramétrico · norma ISO/IEC 19795-10:2024",
            "Padrão biométrico internacional estratificado",
        ),
        (
            "ONDE GENERALIZA?",
            "RFW · BFW",
            "dois datasets externos de verificação",
            "RFW — 4 raças, pares oficiais 1 : 1 (Wang et al., 2019)",
            "BFW — 4 raças × 2 gêneros, balanceado (Robinson et al., 2020)",
            "Transferência arquitetural via feature-freeze",
        ),
    ]

    # Geometria
    SLIDE_W = 13.33
    CARD_W = 4.10
    CARD_GAP = 0.25
    N = len(perguntas)
    total_w = N * CARD_W + (N - 1) * CARD_GAP
    start_x = (SLIDE_W - total_w) / 2
    CARD_Y = 1.75
    HEADER_H = 0.65
    CARD_BODY_H = 4.00
    CARD_H = HEADER_H + CARD_BODY_H

    for i, (pergunta, hero, hero_sub, l1, l2, ref) in enumerate(perguntas):
        x = start_x + i * (CARD_W + CARD_GAP)
        hl = (i == 0)  # destaca a primeira ("o que") para orientar o olhar

        # Header NAVY (pergunta)
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y),
            Inches(CARD_W), Inches(HEADER_H),
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        tf = hdr.text_frame
        tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = pergunta
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = WHITE

        # Corpo cinza-claro
        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y + HEADER_H + 0.05),
            Inches(CARD_W), Inches(CARD_BODY_H),
        )
        body.fill.solid()
        body.fill.fore_color.rgb = GRAY_LT
        body.line.fill.background()

        # Hero number (grande, centrado) — tamanho ajustado dinamicamente
        hero_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(CARD_Y + HEADER_H + 0.25),
            Inches(CARD_W - 0.30), Inches(1.05),
        )
        tf = hero_tb.text_frame
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = hero
        # hero curto (1-3 char) usa fonte maior; hero mais longo usa menor
        r.font.size = Pt(46) if len(hero) <= 3 else Pt(36)
        r.font.bold = True
        r.font.color.rgb = NAVY

        # Hero subtitle
        sub_tb = slide.shapes.add_textbox(
            Inches(x + 0.15), Inches(CARD_Y + HEADER_H + 1.30),
            Inches(CARD_W - 0.30), Inches(0.35),
        )
        tf = sub_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = hero_sub
        r.font.size = Pt(12)
        r.font.italic = True
        r.font.color.rgb = GRAY_MD

        # Linha separadora
        sep = slide.shapes.add_connector(
            2,
            Inches(x + 0.30), Inches(CARD_Y + HEADER_H + 1.80),
            Inches(x + CARD_W - 0.30), Inches(CARD_Y + HEADER_H + 1.80),
        )
        sep.line.color.rgb = GRAY_MD
        sep.line.width = Pt(0.75)

        # Linha 1 (bullet grande)
        l1_tb = slide.shapes.add_textbox(
            Inches(x + 0.25), Inches(CARD_Y + HEADER_H + 1.95),
            Inches(CARD_W - 0.50), Inches(0.70),
        )
        tf = l1_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = "•  " + l1
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = NAVY

        # Linha 2 (bullet grande)
        l2_tb = slide.shapes.add_textbox(
            Inches(x + 0.25), Inches(CARD_Y + HEADER_H + 2.65),
            Inches(CARD_W - 0.50), Inches(0.70),
        )
        tf = l2_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = "•  " + l2
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = NAVY

        # Referencia/nota (rodape do card)
        ref_tb = slide.shapes.add_textbox(
            Inches(x + 0.25), Inches(CARD_Y + HEADER_H + 3.50),
            Inches(CARD_W - 0.50), Inches(0.45),
        )
        tf = ref_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = ref
        r.font.size = Pt(10)
        r.font.italic = True
        r.font.color.rgb = GRAY_MD

    # ---- Banner inferior: compromisso falsificavel ----
    bot_y = 6.65
    bot_h = 0.75
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.05)
    p1 = tf.paragraphs[0]
    p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Critério definido antes do experimento"
    r1.font.size = Pt(13)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Para cada hipótese da pesquisa, o valor-limite de aprovação "
        "foi fixado antes de rodar os experimentos — o resultado só "
        "pode confirmar ou refutar, sem reinterpretar a métrica "
        "depois dos dados."
    )
    r2.font.size = Pt(11)
    r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 26)


# ============================================================
# SLIDE 21 — Triangulação de métricas
# ============================================================
def slide_metricas(prs: Presentation) -> None:
    """As metricas que vamos reportar: 3 cartoes com header NAVY,
    corpo cinza-claro contendo descricao + referencia + valor
    numerico atual, e banner inferior NAVY com justificativa
    (Kleinberg, teorema da impossibilidade).
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "As métricas que vamos reportar")

    # Subtitulo (abaixo da linha do titulo)
    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Padrão da literatura — valores reportados: SOTA FaceScanPaliGemma "
        "(AlDahoul et al., 2024) sobre FairFace 7-class in-domain."
    )
    ps.font.size = Pt(13)
    ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ---- 3 cartoes ----
    # (nome, subtitulo, body, referencia, label_estado, valor, contexto)
    # Valores referenciados em AlDahoul et al. (2024, Nature Sci. Reports)
    # Tabela 10 (F1 macro) e Tabela 16 (F1 por raca); e Kärkkäinen & Joo
    # (2021) baseline ResNet-34 do FairFace original.
    cartoes = [
        (
            "F1 macro",
            "performance média",
            "Média harmônica de precision/recall por\nclasse, depois média simples entre as\n7 raças do FairFace.",
            "Referência: van Rijsbergen 1979",
            "SOTA atual",
            "75 %",
            "FaceScanPaliGemma  ·  baseline ResNet-34 = 72 %",
        ),
        (
            "DR",
            "Disparity Ratio",
            "Razão F1_min / F1_max entre raças.\nMede o gap entre o pior e o melhor\nsubgrupo (0 = total disparidade, 1 = paridade).",
            "Referência: Hardt, Price & Srebro 2016 (NeurIPS)",
            "SOTA atual",
            "0,67",
            "F1 Latinx 60 %  ÷  F1 Black 90 %",
        ),
        (
            "Worst-class F1",
            "pior subgrupo",
            "F1 da raça em que o modelo erra mais.\nGarante que ninguém fique para trás,\nmesmo quando a média agregada é alta.",
            "Referência: Sagawa et al. 2020 (ICLR) — Group DRO",
            "SOTA atual",
            "60 %",
            "Latinx  ·  FaceScanPaliGemma",
        ),
    ]

    SLIDE_W = 13.33
    CARD_W = 4.00
    CARD_GAP = 0.25
    N = len(cartoes)
    total_w = N * CARD_W + (N - 1) * CARD_GAP
    start_x = (SLIDE_W - total_w) / 2
    CARD_Y = 1.75
    HEADER_H = 0.90
    CARD_BODY_H = 4.00
    CARD_H = HEADER_H + CARD_BODY_H

    for i, (nome, sub_nome, descr, refer, label_est, valor, contexto) in enumerate(cartoes):
        x = start_x + i * (CARD_W + CARD_GAP)

        # Header NAVY
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y),
            Inches(CARD_W), Inches(HEADER_H),
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        tf = hdr.text_frame
        tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
        tf.margin_left = Inches(0.10); tf.margin_right = Inches(0.10)
        p1 = tf.paragraphs[0]
        p1.alignment = 2
        r1 = p1.add_run()
        r1.text = nome
        r1.font.size = Pt(22)
        r1.font.bold = True
        r1.font.color.rgb = WHITE
        p2 = tf.add_paragraph()
        p2.alignment = 2
        r2 = p2.add_run()
        r2.text = sub_nome
        r2.font.size = Pt(12)
        r2.font.italic = True
        r2.font.color.rgb = BLUE_LIGHT

        # Corpo (cinza-claro)
        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y + HEADER_H + 0.05),
            Inches(CARD_W), Inches(CARD_BODY_H),
        )
        body.fill.solid()
        body.fill.fore_color.rgb = GRAY_LT
        body.line.fill.background()

        # Descricao
        d_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + HEADER_H + 0.20),
            Inches(CARD_W - 0.40), Inches(0.90),
        )
        tf = d_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = descr
        r.font.size = Pt(12)
        r.font.color.rgb = GRAY_DK

        # Referencia
        ref_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + HEADER_H + 1.35),
            Inches(CARD_W - 0.40), Inches(0.55),
        )
        tf = ref_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = refer
        r.font.size = Pt(10.5)
        r.font.italic = True
        r.font.color.rgb = GRAY_MD

        # Linha separadora
        sep = slide.shapes.add_connector(
            2,
            Inches(x + 0.30), Inches(CARD_Y + HEADER_H + 2.00),
            Inches(x + CARD_W - 0.30), Inches(CARD_Y + HEADER_H + 2.00),
        )
        sep.line.color.rgb = GRAY_MD
        sep.line.width = Pt(0.75)

        # Label do estado
        lbl_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + HEADER_H + 2.10),
            Inches(CARD_W - 0.40), Inches(0.30),
        )
        tf = lbl_tb.text_frame
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = label_est
        r.font.size = Pt(11)
        r.font.italic = True
        r.font.color.rgb = GRAY_MD

        # Valor grande
        val_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + HEADER_H + 2.45),
            Inches(CARD_W - 0.40), Inches(0.95),
        )
        tf = val_tb.text_frame
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = valor
        r.font.size = Pt(44)
        r.font.bold = True
        r.font.color.rgb = NAVY

        # Contexto do valor
        ctx_tb = slide.shapes.add_textbox(
            Inches(x + 0.20), Inches(CARD_Y + HEADER_H + 3.45),
            Inches(CARD_W - 0.40), Inches(0.40),
        )
        tf = ctx_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = contexto
        r.font.size = Pt(11)
        r.font.color.rgb = GRAY_DK

    # ---- Banner inferior: por que triangular (Kleinberg) ----
    bot_y = 6.65
    bot_h = 0.75
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.05)
    p1 = tf.paragraphs[0]
    p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Por que três métricas e não uma?"
    r1.font.size = Pt(13)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Teorema da Impossibilidade (Kleinberg et al., 2017): quando as "
        "prevalências diferem entre grupos, nenhuma métrica única satisfaz "
        "ao mesmo tempo (1) acertar a probabilidade predita por grupo, "
        "(2) mesma taxa de falso positivo entre grupos e (3) mesma taxa de "
        "falso negativo entre grupos. Reportar múltiplas é o único jeito honesto."
    )
    r2.font.size = Pt(11)
    r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 27)


# ============================================================
# SLIDE 22 — Contribuições
# ============================================================
def slide_contribuicoes(prs: Presentation) -> None:
    add_table_slide(
        prs, 28,
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
        prs, 29,
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
        prs, 30,
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
    add_page_number(slide, 31)


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
# Helper comum: slide-detalhamento de uma etapa do pipeline
# (3 cartoes Entrada / Processo / Saida + banner inferior
#  "Regra de aprovacao" com hipotese testada)
# ============================================================
def _slide_etapa_detalhe(
    prs: Presentation,
    numero: int,
    titulo: str,
    subtitulo: str,
    entrada: list,
    processo: list,
    saida: list,
    regra_titulo: str,
    regra_texto: str,
    extra_visual: callable | None = None,
) -> None:
    """Monta um slide-detalhamento de etapa com shapes nativos.

    extra_visual: funcao opcional (slide, x, y, w, h) para desenhar um
    elemento visual especifico da etapa logo abaixo dos 3 cartoes.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, titulo)

    # Subtitulo
    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = subtitulo
    ps.font.size = Pt(13)
    ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # 3 cartoes horizontais
    SLIDE_W = 13.33
    CARD_W = 4.10
    CARD_GAP = 0.25
    CARD_Y = 1.75
    HEADER_H = 0.55
    CARD_BODY_H = 3.00
    CARD_H = HEADER_H + CARD_BODY_H
    total_w = 3 * CARD_W + 2 * CARD_GAP
    start_x = (SLIDE_W - total_w) / 2

    cartoes = [
        ("ENTRADA",   NAVY,     entrada),
        ("PROCESSO",  BLUE_MID, processo),
        ("SAÍDA",     ACCENT,   saida),
    ]

    for i, (header, cor, bullets) in enumerate(cartoes):
        x = start_x + i * (CARD_W + CARD_GAP)

        # Header colorido
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y),
            Inches(CARD_W), Inches(HEADER_H),
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = cor
        hdr.line.fill.background()
        tf = hdr.text_frame
        tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.05)
        p = tf.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = header
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = WHITE

        # Corpo branco com borda colorida
        body = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(CARD_Y + HEADER_H + 0.03),
            Inches(CARD_W), Inches(CARD_BODY_H),
        )
        body.fill.solid()
        body.fill.fore_color.rgb = WHITE
        body.line.color.rgb = cor
        body.line.width = Pt(1.2)

        # Bullets dentro do corpo
        tb = slide.shapes.add_textbox(
            Inches(x + 0.18), Inches(CARD_Y + HEADER_H + 0.15),
            Inches(CARD_W - 0.36), Inches(CARD_BODY_H - 0.25),
        )
        tf = tb.text_frame
        tf.word_wrap = True
        for j, bullet in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = "•  " + bullet
            r.font.size = Pt(12)
            r.font.color.rgb = NAVY
            p.space_after = Pt(6)

    # Visual extra opcional (abaixo dos cartoes)
    if extra_visual is not None:
        extra_visual(slide, 0.5, CARD_Y + CARD_H + 0.18, 12.3, 1.05)

    # Banner inferior: Regra de aprovacao
    bot_y = 6.60
    bot_h = 0.80
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.05)
    p1 = tf.paragraphs[0]
    p1.alignment = 2
    r1 = p1.add_run()
    r1.text = regra_titulo
    r1.font.size = Pt(13)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.alignment = 2
    r2 = p2.add_run()
    r2.text = regra_texto
    r2.font.size = Pt(11)
    r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, numero)


# ============================================================
# SLIDE 19 — Etapa 2: Matriz MST × raca (detalhamento)
# ============================================================
def slide_etapa2_matriz_mst_raca(prs: Presentation) -> None:
    """Etapa 2: layout com MATRIZ 7x10 como heroi visual central.

    Metafora: a propria matriz publicada e o artefato — mostra-se o
    artefato em primeiro plano. Latinx (linha 5) destacada em vermelho
    para evidenciar o spread.
    """
    MST_PAL = [
        RGBColor(0xF6, 0xED, 0xE4), RGBColor(0xF3, 0xE7, 0xDB),
        RGBColor(0xF7, 0xEA, 0xD0), RGBColor(0xEA, 0xDA, 0xBA),
        RGBColor(0xD7, 0xBD, 0x96), RGBColor(0xA0, 0x7E, 0x56),
        RGBColor(0x82, 0x5C, 0x43), RGBColor(0x60, 0x41, 0x34),
        RGBColor(0x3A, 0x31, 0x2A), RGBColor(0x29, 0x24, 0x20),
    ]
    # Distribuicao ilustrativa (% dentro de cada raca)
    # intensidades 0-5 para representar frequencia
    #  linhas = raca (ordem FairFace); colunas = MST 1..10
    #  Latinx (idx 4) distribuida; outras com picos mais concentrados
    freq = [
        [4, 5, 5, 4, 3, 1, 0, 0, 0, 0],  # White
        [0, 0, 0, 0, 1, 3, 5, 5, 4, 2],  # Black
        [0, 0, 1, 3, 4, 5, 4, 2, 1, 0],  # Indian
        [3, 5, 5, 4, 2, 1, 0, 0, 0, 0],  # East Asian
        [1, 2, 3, 4, 4, 3, 2, 1, 0, 0],  # Latinx  <-- distribuido
        [1, 3, 4, 4, 3, 2, 1, 0, 0, 0],  # SE Asian
        [1, 3, 5, 4, 3, 2, 1, 0, 0, 0],  # Mid Eastern
    ]
    racas = [
        "White", "Black", "Indian", "East Asian",
        "Latinx", "SE Asian", "Mid East.",
    ]
    LATINX_IDX = 4

    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 2 — Matriz MST × raça")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Primeira matriz de distribuição cruzada MST × FairFace — "
        "dataset auditado completo (~108 000 imagens, 7 grupos raciais)."
    )
    ps.font.size = Pt(13)
    ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    # ============================================================
    # HEROI VISUAL: matriz 7 x 10 ao centro
    # ============================================================
    MTX_X = 2.80
    MTX_Y = 1.95
    CELL_W = 0.56
    CELL_H = 0.42

    # Rotulos de coluna (MST 1..10) acima
    for c in range(10):
        tx = slide.shapes.add_textbox(
            Inches(MTX_X + c * CELL_W), Inches(MTX_Y - 0.30),
            Inches(CELL_W), Inches(0.25),
        )
        p = tx.text_frame.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = str(c + 1)
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = GRAY_DK
    # Header "MST →"
    hdr = slide.shapes.add_textbox(
        Inches(MTX_X - 1.6), Inches(MTX_Y - 0.30),
        Inches(1.5), Inches(0.25),
    )
    ph = hdr.text_frame.paragraphs[0]
    ph.alignment = 2
    rh = ph.add_run()
    rh.text = "Monk tone →"
    rh.font.size = Pt(10)
    rh.font.bold = True
    rh.font.italic = True
    rh.font.color.rgb = GRAY_MD

    # Celulas da matriz
    for r_idx in range(7):
        # Rotulo da raca na coluna esquerda
        tr = slide.shapes.add_textbox(
            Inches(MTX_X - 1.6), Inches(MTX_Y + r_idx * CELL_H + 0.05),
            Inches(1.5), Inches(CELL_H - 0.05),
        )
        pr = tr.text_frame.paragraphs[0]
        pr.alignment = 2
        rr = pr.add_run()
        rr.text = racas[r_idx]
        rr.font.size = Pt(11)
        rr.font.bold = (r_idx == LATINX_IDX)
        rr.font.color.rgb = ACCENT if r_idx == LATINX_IDX else NAVY

        for c in range(10):
            intensity = freq[r_idx][c]
            # fundo claro quando intensidade=0; MST palette quando >0
            if intensity == 0:
                fc = RGBColor(0xF5, 0xF5, 0xF5)
            else:
                fc = MST_PAL[c]
            cell = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(MTX_X + c * CELL_W + 0.02),
                Inches(MTX_Y + r_idx * CELL_H + 0.02),
                Inches(CELL_W - 0.04), Inches(CELL_H - 0.04),
            )
            cell.fill.solid()
            cell.fill.fore_color.rgb = fc
            if r_idx == LATINX_IDX:
                cell.line.color.rgb = ACCENT
                cell.line.width = Pt(1.4)
            else:
                cell.line.color.rgb = GRAY_DK
                cell.line.width = Pt(0.4)
            # Dot de frequencia (circulo preto escalado por intensidade)
            if intensity > 0:
                dot_size = 0.06 + intensity * 0.04  # 0.10..0.26
                dot_cx = MTX_X + c * CELL_W + (CELL_W - dot_size) / 2
                dot_cy = MTX_Y + r_idx * CELL_H + (CELL_H - dot_size) / 2
                dot = slide.shapes.add_shape(
                    MSO_SHAPE.OVAL,
                    Inches(dot_cx), Inches(dot_cy),
                    Inches(dot_size), Inches(dot_size),
                )
                dot.fill.solid()
                dot.fill.fore_color.rgb = NAVY
                dot.line.fill.background()

    # Legenda da intensidade (direita)
    LEG_X = MTX_X + 10 * CELL_W + 0.40
    tl = slide.shapes.add_textbox(
        Inches(LEG_X), Inches(MTX_Y - 0.30),
        Inches(1.8), Inches(0.25),
    )
    pl = tl.text_frame.paragraphs[0]
    rl = pl.add_run()
    rl.text = "frequência"
    rl.font.size = Pt(10)
    rl.font.bold = True
    rl.font.italic = True
    rl.font.color.rgb = GRAY_MD
    for i, sz in enumerate([0.10, 0.14, 0.20, 0.26]):
        row_y = MTX_Y + 0.05 + i * 0.45
        d = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(LEG_X + 0.10), Inches(row_y),
            Inches(sz), Inches(sz),
        )
        d.fill.solid(); d.fill.fore_color.rgb = NAVY
        d.line.fill.background()
        txt = slide.shapes.add_textbox(
            Inches(LEG_X + 0.50), Inches(row_y - 0.03),
            Inches(1.3), Inches(0.30),
        )
        pt_ = txt.text_frame.paragraphs[0]
        rt_ = pt_.add_run()
        rt_.text = ["baixa", "média", "alta", "muito alta"][i]
        rt_.font.size = Pt(9)
        rt_.font.color.rgb = GRAY_DK

    # ============================================================
    # Callout destacando a linha Latinx
    # ============================================================
    CALL_X = 0.60
    CALL_Y = MTX_Y + 7 * CELL_H + 0.30
    callout = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(CALL_X), Inches(CALL_Y),
        Inches(12.1), Inches(0.95),
    )
    callout.fill.solid()
    callout.fill.fore_color.rgb = WHITE
    callout.line.color.rgb = ACCENT
    callout.line.width = Pt(1.8)
    tf = callout.text_frame
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.06)
    tf.margin_left = Inches(0.20); tf.margin_right = Inches(0.20)
    p1 = tf.paragraphs[0]
    r1 = p1.add_run()
    r1.text = "Latinx:  "
    r1.font.size = Pt(13); r1.font.bold = True
    r1.font.color.rgb = ACCENT
    r1b = p1.add_run()
    r1b.text = "variância fenotípica interna cobre MST 1 a MST 8 — "
    r1b.font.size = Pt(13); r1b.font.color.rgb = NAVY
    r1c = p1.add_run()
    r1c.text = "contraste com White (1–5) e Black (5–10)."
    r1c.font.size = Pt(13); r1c.font.italic = True
    r1c.font.color.rgb = GRAY_DK
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = (
        "Prova empírica da inadequação dos rótulos raciais discretos "
        "— justifica o condicionamento por tom de pele (Etapa 3)."
    )
    r2.font.size = Pt(11)
    r2.font.italic = True
    r2.font.color.rgb = GRAY_MD

    # ============================================================
    # Banner inferior: regra H3
    # ============================================================
    bot_y = 6.60; bot_h = 0.80
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(12.33), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Regra de aprovação — hipótese H3"
    r1.font.size = Pt(13); r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Latinx cobre ao menos 6 dos 10 tons MST (observado: 1–8) — "
        "quantifica a inadequação do rótulo racial discreto."
    )
    r2.font.size = Pt(11); r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 19)


# ============================================================
# SLIDE 24 — Etapa 4: Comparacao com baselines (detalhamento)
# ============================================================
def slide_etapa4_baselines(prs: Presentation) -> None:
    """Etapa 4: layout VS — 6 baselines publicados vs. nossa proposta.

    Metafora: duas panels, direita vs. esquerda, com simbolo "x" no meio.
    Deixa claro visualmente a estrutura de comparacao direta.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 4 — Comparação sistemática com baselines")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Confronta a proposta FiLM + MST contra seis famílias publicadas "
        "sob o mesmo protocolo pareado."
    )
    ps.font.size = Pt(13); ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    SLIDE_W = 13.33
    PANEL_Y = 1.80
    PANEL_H = 3.60

    # ============================================================
    # PAINEL ESQUERDO — 6 baselines publicados (3 x 2)
    # ============================================================
    LEFT_X = 0.50
    LEFT_W = 7.70
    left_bg = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(LEFT_X), Inches(PANEL_Y),
        Inches(LEFT_W), Inches(PANEL_H),
    )
    left_bg.fill.solid(); left_bg.fill.fore_color.rgb = GRAY_LT
    left_bg.line.color.rgb = GRAY_DK; left_bg.line.width = Pt(0.8)

    # Header
    lhdr = slide.shapes.add_textbox(
        Inches(LEFT_X), Inches(PANEL_Y + 0.08),
        Inches(LEFT_W), Inches(0.35),
    )
    plh = lhdr.text_frame.paragraphs[0]; plh.alignment = 2
    rh = plh.add_run()
    rh.text = "6 famílias de mitigação já publicadas"
    rh.font.size = Pt(13); rh.font.bold = True
    rh.font.color.rgb = NAVY

    # 6 mini-cards: 3 cols x 2 rows
    baselines = [
        ("Adv. Debiasing",   "Zhang 2018",       "loss adversarial"),
        ("Group DRO",         "Sagawa 2020",      "worst-case group"),
        ("FSCL+",             "Park 2022",        "contrastive fair"),
        ("FineFACE",          "Manzoor 2024",     "Pareto-eficiente"),
        ("Bayes. Meta",       "Liu 2025",         "component fairness"),
        ("FaceScanPaliGemma", "AlDahoul 2024",    "VLM / VQA"),
    ]
    COLS = 3; ROWS = 2
    inner_margin = 0.18
    inner_gap = 0.12
    card_w = (LEFT_W - 2 * inner_margin - (COLS - 1) * inner_gap) / COLS
    card_h = (PANEL_H - 0.55 - inner_margin - (ROWS - 1) * inner_gap) / ROWS
    grid_y0 = PANEL_Y + 0.50

    for i, (nome, ref, nota) in enumerate(baselines):
        row = i // COLS
        col = i % COLS
        cx = LEFT_X + inner_margin + col * (card_w + inner_gap)
        cy = grid_y0 + row * (card_h + inner_gap)
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(cx), Inches(cy),
            Inches(card_w), Inches(card_h),
        )
        card.fill.solid(); card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = NAVY; card.line.width = Pt(1.0)
        tf = card.text_frame
        tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
        p = tf.paragraphs[0]; p.alignment = 2
        r = p.add_run()
        r.text = nome
        r.font.size = Pt(11); r.font.bold = True
        r.font.color.rgb = NAVY
        p2 = tf.add_paragraph(); p2.alignment = 2
        r2 = p2.add_run()
        r2.text = ref
        r2.font.size = Pt(9); r2.font.italic = True
        r2.font.color.rgb = GRAY_MD
        p3 = tf.add_paragraph(); p3.alignment = 2
        r3 = p3.add_run()
        r3.text = nota
        r3.font.size = Pt(9)
        r3.font.color.rgb = GRAY_DK

    # ============================================================
    # DIVISOR "VS" central
    # ============================================================
    VS_X = LEFT_X + LEFT_W + 0.10
    VS_W = 0.90
    vs = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(VS_X), Inches(PANEL_Y + PANEL_H / 2 - 0.45),
        Inches(VS_W), Inches(VS_W),
    )
    vs.fill.solid(); vs.fill.fore_color.rgb = ACCENT
    vs.line.color.rgb = NAVY; vs.line.width = Pt(2.2)
    tf = vs.text_frame
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "VS"
    r.font.size = Pt(20); r.font.bold = True
    r.font.color.rgb = WHITE

    # ============================================================
    # PAINEL DIREITO — nossa proposta
    # ============================================================
    RIGHT_X = VS_X + VS_W + 0.15
    RIGHT_W = SLIDE_W - 0.50 - RIGHT_X
    right_bg = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(RIGHT_X), Inches(PANEL_Y),
        Inches(RIGHT_W), Inches(PANEL_H),
    )
    right_bg.fill.solid(); right_bg.fill.fore_color.rgb = NAVY
    right_bg.line.color.rgb = NAVY; right_bg.line.width = Pt(1.0)

    # Header "Nossa proposta"
    rhdr = slide.shapes.add_textbox(
        Inches(RIGHT_X), Inches(PANEL_Y + 0.08),
        Inches(RIGHT_W), Inches(0.35),
    )
    prh = rhdr.text_frame.paragraphs[0]; prh.alignment = 2
    rr = prh.add_run()
    rr.text = "Nossa proposta"
    rr.font.size = Pt(13); rr.font.bold = True
    rr.font.color.rgb = BLUE_LIGHT

    # Card grande da proposta
    OUR_X = RIGHT_X + 0.20
    OUR_Y = PANEL_Y + 0.55
    OUR_W = RIGHT_W - 0.40
    OUR_H = PANEL_H - 0.75
    our = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(OUR_X), Inches(OUR_Y),
        Inches(OUR_W), Inches(OUR_H),
    )
    our.fill.solid(); our.fill.fore_color.rgb = WHITE
    our.line.color.rgb = ACCENT; our.line.width = Pt(2.5)
    tf = our.text_frame
    tf.margin_top = Inches(0.20); tf.margin_bottom = Inches(0.10)
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "FiLM + MST"
    r.font.size = Pt(20); r.font.bold = True
    r.font.color.rgb = NAVY
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = "Config B (Etapa 3)"
    r2.font.size = Pt(12); r2.font.italic = True
    r2.font.color.rgb = ACCENT
    p3 = tf.add_paragraph(); p3.alignment = 2
    p3.space_before = Pt(10)
    r3 = p3.add_run()
    r3.text = "condicionamento\narquitetural\n(FiLM) com sinal\nde tom de pele\ncontínuo (MST)"
    r3.font.size = Pt(11)
    r3.font.color.rgb = GRAY_DK

    # ============================================================
    # Faixa inferior: protocolo pareado (3 atributos)
    # ============================================================
    PROTO_Y = PANEL_Y + PANEL_H + 0.20
    proto_items = [
        ("mesmo backbone",    "ConvNeXt-T"),
        ("mesmos dados",      "FairFace train split"),
        ("mesma receita",     "3 sementes · bootstrap IC 95 %"),
    ]
    nprot = len(proto_items)
    gap = 0.20
    pw = (SLIDE_W - 1.0 - (nprot - 1) * gap) / nprot

    # Header da faixa
    phd = slide.shapes.add_textbox(
        Inches(0.5), Inches(PROTO_Y - 0.03),
        Inches(SLIDE_W - 1.0), Inches(0.28),
    )
    pph = phd.text_frame.paragraphs[0]; pph.alignment = 2
    rh = pph.add_run()
    rh.text = "Protocolo pareado  —  o que mantemos constante entre os 7 métodos"
    rh.font.size = Pt(11); rh.font.italic = True
    rh.font.bold = True
    rh.font.color.rgb = GRAY_MD

    for i, (k, v) in enumerate(proto_items):
        bx = 0.5 + i * (pw + gap)
        by = PROTO_Y + 0.30
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(bx), Inches(by),
            Inches(pw), Inches(0.55),
        )
        card.fill.solid(); card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BLUE_MID; card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.03)
        p = tf.paragraphs[0]; p.alignment = 2
        r1 = p.add_run()
        r1.text = k + "  "
        r1.font.size = Pt(11); r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(11); r2.font.italic = True
        r2.font.color.rgb = GRAY_DK

    # ============================================================
    # Banner inferior: regra H1
    # ============================================================
    bot_y = 6.60; bot_h = 0.80
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Regra de aprovação — hipótese H1 (central)"
    r1.font.size = Pt(13); r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Supera o melhor baseline em F1 macro ≥ +2 pp E "
        "reduz Disparity Ratio em ≥ 20 % — ambos critérios juntos."
    )
    r2.font.size = Pt(11); r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 23)


# ============================================================
# SLIDE 25 — Etapa 5: Transferencia RFW/BFW (detalhamento)
# ============================================================
def slide_etapa5_transferencia(prs: Presentation) -> None:
    """Etapa 5: layout ANTES vs DEPOIS com feature-freeze visualizado.

    LADO ESQUERDO: Modelo da Etapa 3 (classificacao racial).
    SETA CENTRAL: "transferencia via feature-freeze (Madras, LAFTR 2018)".
    LADO DIREITO: Modelo da Etapa 5 (verificacao RFW/BFW) com
                  encoder congelado + nova cabeca.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 5 — Transferência arquitetural para verificação")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Testa se o condicionamento FiLM + MST herda-se para "
        "verificação sem re-treino do encoder (LAFTR, Madras 2018)."
    )
    ps.font.size = Pt(13); ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    SLIDE_W = 13.33
    PANEL_Y = 1.80
    PANEL_H = 4.00

    # Dimensoes dos paineis esquerdo e direito
    SIDE_W = 5.20
    LEFT_X = 0.50
    RIGHT_X = SLIDE_W - 0.50 - SIDE_W
    MID_X = LEFT_X + SIDE_W
    MID_W = RIGHT_X - MID_X

    # ============================================================
    # HELPER para desenhar um "stack" de camadas
    # ============================================================
    def _draw_stack(x, y, w, titulo, header_cor, camadas):
        # Header do painel
        hdr = slide.shapes.add_textbox(
            Inches(x), Inches(y),
            Inches(w), Inches(0.35),
        )
        ph = hdr.text_frame.paragraphs[0]; ph.alignment = 2
        r = ph.add_run()
        r.text = titulo
        r.font.size = Pt(13); r.font.bold = True
        r.font.color.rgb = header_cor

        # Pilha vertical de camadas
        STACK_Y = y + 0.40
        STACK_H = PANEL_H - 0.50
        n = len(camadas)
        gap_internal = 0.12
        layer_h = (STACK_H - (n - 1) * gap_internal) / n
        for i, (nome, nota, estado) in enumerate(camadas):
            # estado: "frozen" | "trained" | "discarded"
            ly = STACK_Y + i * (layer_h + gap_internal)
            box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(x + 0.20), Inches(ly),
                Inches(w - 0.40), Inches(layer_h),
            )
            if estado == "frozen":
                box.fill.solid(); box.fill.fore_color.rgb = BLUE_LIGHT
                box.line.color.rgb = NAVY; box.line.width = Pt(2.0)
                border_cor = NAVY
            elif estado == "trained":
                box.fill.solid(); box.fill.fore_color.rgb = ACCENT
                box.line.color.rgb = ACCENT; box.line.width = Pt(2.0)
                border_cor = WHITE
            elif estado == "discarded":
                box.fill.solid(); box.fill.fore_color.rgb = GRAY_LT
                box.line.color.rgb = GRAY_DK; box.line.width = Pt(1.0)
                border_cor = GRAY_MD
            else:  # "neutral"
                box.fill.solid(); box.fill.fore_color.rgb = WHITE
                box.line.color.rgb = NAVY; box.line.width = Pt(1.0)
                border_cor = NAVY
            tf = box.text_frame
            tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.04)
            p = tf.paragraphs[0]; p.alignment = 2
            r = p.add_run()
            r.text = nome
            r.font.size = Pt(12); r.font.bold = True
            if estado == "trained":
                r.font.color.rgb = WHITE
            elif estado == "discarded":
                r.font.color.rgb = GRAY_MD
            else:
                r.font.color.rgb = NAVY
            p2 = tf.add_paragraph(); p2.alignment = 2
            r2 = p2.add_run()
            r2.text = nota
            r2.font.size = Pt(10); r2.font.italic = True
            if estado == "trained":
                r2.font.color.rgb = BLUE_LIGHT
            elif estado == "discarded":
                r2.font.color.rgb = GRAY_MD
            else:
                r2.font.color.rgb = GRAY_DK
            # Badge estado (bolinha pequena no canto superior esquerdo)
            if estado in ("frozen", "trained"):
                badge_size = 0.32
                bx = x + 0.26
                by = ly + 0.08
                badge = slide.shapes.add_shape(
                    MSO_SHAPE.OVAL,
                    Inches(bx), Inches(by),
                    Inches(badge_size), Inches(badge_size),
                )
                badge.fill.solid()
                badge.fill.fore_color.rgb = NAVY if estado == "frozen" else WHITE
                badge.line.color.rgb = WHITE if estado == "frozen" else ACCENT
                badge.line.width = Pt(1.5)
                tfb = badge.text_frame
                tfb.margin_top = Inches(0.0); tfb.margin_bottom = Inches(0.0)
                pb = tfb.paragraphs[0]; pb.alignment = 2
                rb = pb.add_run()
                rb.text = "F" if estado == "frozen" else "T"
                rb.font.size = Pt(10); rb.font.bold = True
                rb.font.color.rgb = WHITE if estado == "frozen" else ACCENT

    # ============================================================
    # PAINEL ESQUERDO — Modelo Etapa 3 (classificacao racial)
    # ============================================================
    camadas_antes = [
        ("Cabeça classificador\n7 raças",  "treinada em Etapa 3",  "trained"),
        ("FiLM · condicionamento",           "treinado",             "trained"),
        ("ConvNeXt-T",                       "treinado end-to-end",  "trained"),
    ]
    _draw_stack(LEFT_X, PANEL_Y, SIDE_W,
                "Modelo da Etapa 3  (classificação de 7 raças)",
                NAVY, camadas_antes)

    # ============================================================
    # SETA central grande — transferencia arquitetural
    # ============================================================
    # Pentagona ou flecha larga
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        Inches(MID_X + 0.15), Inches(PANEL_Y + 1.20),
        Inches(MID_W - 0.30), Inches(1.60),
    )
    arrow.fill.solid(); arrow.fill.fore_color.rgb = BLUE_MID
    arrow.line.color.rgb = NAVY; arrow.line.width = Pt(1.8)
    tf = arrow.text_frame
    tf.margin_top = Inches(0.20); tf.margin_bottom = Inches(0.10)
    tf.margin_left = Inches(0.10); tf.margin_right = Inches(0.35)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "TRANSFERÊNCIA"
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = "feature-freeze"
    r2.font.size = Pt(11); r2.font.italic = True
    r2.font.color.rgb = WHITE
    p3 = tf.add_paragraph(); p3.alignment = 2
    r3 = p3.add_run()
    r3.text = "Madras 2018 · LAFTR"
    r3.font.size = Pt(9); r3.font.italic = True
    r3.font.color.rgb = BLUE_LIGHT

    # Rotulo abaixo da seta
    sublbl = slide.shapes.add_textbox(
        Inches(MID_X), Inches(PANEL_Y + 3.00),
        Inches(MID_W), Inches(0.45),
    )
    psb = sublbl.text_frame.paragraphs[0]; psb.alignment = 2
    rsb = psb.add_run()
    rsb.text = "sem re-treino do\nencoder MST"
    rsb.font.size = Pt(10); rsb.font.italic = True
    rsb.font.bold = True
    rsb.font.color.rgb = NAVY

    # ============================================================
    # PAINEL DIREITO — Modelo Etapa 5 (verificacao RFW/BFW)
    # ============================================================
    camadas_depois = [
        ("Nova cabeça\nde verificação",     "ÚNICA treinada aqui",    "trained"),
        ("FiLM · condicionamento",           "congelado",              "frozen"),
        ("ConvNeXt-T",                       "congelado",              "frozen"),
    ]
    _draw_stack(RIGHT_X, PANEL_Y, SIDE_W,
                "Modelo da Etapa 5  (verificação RFW · BFW)",
                ACCENT, camadas_depois)

    # ============================================================
    # Rodape auxiliar: datasets + metrica
    # ============================================================
    FOOT_Y = PANEL_Y + PANEL_H + 0.10
    foot_items = [
        ("Datasets",   "RFW (Wang 2019, 4 raças) · BFW (Robinson 2020, 4 raças × 2 gêneros)"),
        ("Métrica",    "TAR @ FAR = 1 × 10⁻⁴  estratificada por raça"),
    ]
    n = len(foot_items)
    gap = 0.20
    pw = (SLIDE_W - 1.0 - (n - 1) * gap) / n
    for i, (k, v) in enumerate(foot_items):
        bx = 0.5 + i * (pw + gap)
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(bx), Inches(FOOT_Y),
            Inches(pw), Inches(0.52),
        )
        card.fill.solid(); card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BLUE_MID; card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.03)
        tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.12)
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = k + ":  "
        r1.font.size = Pt(11); r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(11)
        r2.font.color.rgb = GRAY_DK

    # ============================================================
    # Banner inferior: regra H5
    # ============================================================
    bot_y = 6.60; bot_h = 0.80
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.07); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Regra de aprovação — hipótese H5 (central)"
    r1.font.size = Pt(13); r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "ΔTAR @ FAR = 1 × 10⁻⁴ ≥ +3 pp no grupo African, "
        "sobre backbone equivalente sem condicionamento."
    )
    r2.font.size = Pt(11); r2.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 24)


# ============================================================
# SLIDE 26 — Etapa 6: Decomposicao do erro (detalhamento)
# ============================================================
def slide_etapa6_decomposicao(prs: Presentation) -> None:
    """Etapa 6: FORK diagram (isolamento dos componentes do erro) + bar.

    Hero: diagrama em arvore invertida —
      Erro total (Latinx) --> H4 Fenotipico (entropia MST, irredutivel)
                          --> H6 Algoritmico (alinhamento + FiLM, mitigavel)
    Abaixo: barra de resultado ilustrativo + banner com resultado intelectual.
    """
    slide = prs.slides.add_slide(_blank(prs))
    add_title(slide, "Etapa 6 — Decomposição diagnóstica do erro")

    sub = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35)
    )
    ps = sub.text_frame.paragraphs[0]
    ps.text = (
        "Isola o componente fenotípico (irredutível) do algorítmico "
        "(mitigável) — primeiro diagnóstico quantitativo do limite teórico de equidade."
    )
    ps.font.size = Pt(13); ps.font.italic = True
    ps.font.color.rgb = GRAY_MD

    SLIDE_W = 13.33

    # ============================================================
    # HEROI VISUAL — FORK DIAGRAM: erro total -> 2 componentes
    # ============================================================
    # Caixa topo: Erro total (Latinx)
    TOP_X = 4.33
    TOP_Y = 1.80
    TOP_W = 4.67
    TOP_H = 0.80
    top_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(TOP_X), Inches(TOP_Y),
        Inches(TOP_W), Inches(TOP_H),
    )
    top_box.fill.solid(); top_box.fill.fore_color.rgb = NAVY
    top_box.line.color.rgb = NAVY; top_box.line.width = Pt(1.5)
    tf = top_box.text_frame
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "Erro total do classificador em Latinx"
    r.font.size = Pt(14); r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = "100 % da variância observada"
    r2.font.size = Pt(10); r2.font.italic = True
    r2.font.color.rgb = BLUE_LIGHT

    # Dois ramos — H4 (esquerda) e H6 (direita)
    FORK_Y_TOP = TOP_Y + TOP_H
    BRANCH_Y = 3.10
    BRANCH_H = 1.55
    LEFT_X = 0.60
    RIGHT_X = SLIDE_W - 0.60 - 5.80
    BRANCH_W = 5.80

    top_center_x = TOP_X + TOP_W / 2
    left_center_x = LEFT_X + BRANCH_W / 2
    right_center_x = RIGHT_X + BRANCH_W / 2

    # Linhas Y da conexao fork (vertical do centro + horizontal + descida)
    FORK_MID_Y = 2.85

    # Vertical descendente do centro
    v1 = slide.shapes.add_connector(
        1,
        Inches(top_center_x), Inches(FORK_Y_TOP),
        Inches(top_center_x), Inches(FORK_MID_Y),
    )
    v1.line.color.rgb = NAVY; v1.line.width = Pt(2.0)
    # Horizontal esquerda
    h1 = slide.shapes.add_connector(
        1,
        Inches(left_center_x), Inches(FORK_MID_Y),
        Inches(top_center_x), Inches(FORK_MID_Y),
    )
    h1.line.color.rgb = NAVY; h1.line.width = Pt(2.0)
    # Horizontal direita
    h2 = slide.shapes.add_connector(
        1,
        Inches(top_center_x), Inches(FORK_MID_Y),
        Inches(right_center_x), Inches(FORK_MID_Y),
    )
    h2.line.color.rgb = NAVY; h2.line.width = Pt(2.0)
    # Descida esquerda com seta
    da1 = slide.shapes.add_connector(
        2,
        Inches(left_center_x), Inches(FORK_MID_Y),
        Inches(left_center_x), Inches(BRANCH_Y),
    )
    da1.line.color.rgb = BLUE_MID; da1.line.width = Pt(2.2)
    _add_arrowhead(da1)
    # Descida direita com seta
    da2 = slide.shapes.add_connector(
        2,
        Inches(right_center_x), Inches(FORK_MID_Y),
        Inches(right_center_x), Inches(BRANCH_Y),
    )
    da2.line.color.rgb = ACCENT; da2.line.width = Pt(2.2)
    _add_arrowhead(da2)

    # ============================================================
    # BRANCH ESQUERDA — H4 Fenotipico
    # ============================================================
    left_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(LEFT_X), Inches(BRANCH_Y),
        Inches(BRANCH_W), Inches(BRANCH_H),
    )
    left_card.fill.solid(); left_card.fill.fore_color.rgb = BLUE_MID
    left_card.line.color.rgb = NAVY; left_card.line.width = Pt(1.8)
    tf = left_card.text_frame
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "H4 · Componente FENOTÍPICO / BIOLÓGICO"
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE
    p1 = tf.add_paragraph(); p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "irredutível"
    r1.font.size = Pt(11); r1.font.italic = True
    r1.font.color.rgb = BLUE_LIGHT
    p2 = tf.add_paragraph(); p2.alignment = 2
    p2.space_before = Pt(5)
    r2a = p2.add_run()
    r2a.text = "Medida:  "
    r2a.font.size = Pt(11); r2a.font.bold = True
    r2a.font.color.rgb = WHITE
    r2b = p2.add_run()
    r2b.text = "entropia das probabilidades do extrator MST"
    r2b.font.size = Pt(11); r2b.font.color.rgb = WHITE
    p3 = tf.add_paragraph(); p3.alignment = 2
    r3 = p3.add_run()
    r3.text = (
        "sobreposição biológica real + instabilidade\n"
        "dos rótulos sociais de raça em Latinx"
    )
    r3.font.size = Pt(10); r3.font.italic = True
    r3.font.color.rgb = BLUE_LIGHT

    # ============================================================
    # BRANCH DIREITA — H6 Algoritmico
    # ============================================================
    right_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(RIGHT_X), Inches(BRANCH_Y),
        Inches(BRANCH_W), Inches(BRANCH_H),
    )
    right_card.fill.solid(); right_card.fill.fore_color.rgb = ACCENT
    right_card.line.color.rgb = ACCENT; right_card.line.width = Pt(1.8)
    tf = right_card.text_frame
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.05)
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "H6 · Componente ALGORÍTMICO / ÓPTICO"
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE
    p1 = tf.add_paragraph(); p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "mitigável"
    r1.font.size = Pt(11); r1.font.italic = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    p2.space_before = Pt(5)
    r2a = p2.add_run()
    r2a.text = "Medida:  "
    r2a.font.size = Pt(11); r2a.font.bold = True
    r2a.font.color.rgb = WHITE
    r2b = p2.add_run()
    r2b.text = "variação de desempenho sob alinhamento estrito + FiLM"
    r2b.font.size = Pt(11); r2b.font.color.rgb = WHITE
    p3 = tf.add_paragraph(); p3.alignment = 2
    r3 = p3.add_run()
    r3.text = (
        "o espaço onde o condicionamento\n"
        "arquitetural pode efetivamente agir"
    )
    r3.font.size = Pt(10); r3.font.italic = True
    r3.font.color.rgb = WHITE

    # ============================================================
    # BARRA DE RESULTADO ILUSTRATIVO (abaixo do fork)
    # ============================================================
    BAR_X = 1.00
    BAR_Y = 5.00
    BAR_W = SLIDE_W - 2 * BAR_X
    BAR_H = 0.80
    FEN_FRAC = 0.70

    # Rotulo
    hdr = slide.shapes.add_textbox(
        Inches(BAR_X), Inches(BAR_Y - 0.26),
        Inches(BAR_W), Inches(0.22),
    )
    ph = hdr.text_frame.paragraphs[0]; ph.alignment = 2
    rh = ph.add_run()
    rh.text = (
        "Partição ilustrativa do R²  —  o resultado experimental "
        "da Etapa 6 determina a proporção real:"
    )
    rh.font.size = Pt(10); rh.font.italic = True
    rh.font.color.rgb = GRAY_MD

    fen = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(BAR_X), Inches(BAR_Y),
        Inches(BAR_W * FEN_FRAC), Inches(BAR_H),
    )
    fen.fill.solid(); fen.fill.fore_color.rgb = BLUE_MID
    fen.line.color.rgb = NAVY; fen.line.width = Pt(1.2)
    tf = fen.text_frame
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "FENOTÍPICO · ~70 %"
    r.font.size = Pt(14); r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = "H4 — define o LIMITE TEÓRICO de equidade atingível"
    r2.font.size = Pt(10); r2.font.italic = True
    r2.font.color.rgb = BLUE_LIGHT

    alg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(BAR_X + BAR_W * FEN_FRAC), Inches(BAR_Y),
        Inches(BAR_W * (1 - FEN_FRAC)), Inches(BAR_H),
    )
    alg.fill.solid(); alg.fill.fore_color.rgb = ACCENT
    alg.line.color.rgb = NAVY; alg.line.width = Pt(1.2)
    tf = alg.text_frame
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]; p.alignment = 2
    r = p.add_run()
    r.text = "ALGORÍTMICO · ~30 %"
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = "H6 — o teto da mitigação"
    r2.font.size = Pt(10); r2.font.italic = True
    r2.font.color.rgb = WHITE

    # ============================================================
    # Banner inferior: resultado intelectual
    # ============================================================
    bot_y = 6.20; bot_h = 1.20
    bot = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(bot_y),
        Inches(SLIDE_W - 1.0), Inches(bot_h),
    )
    bot.fill.solid(); bot.fill.fore_color.rgb = NAVY
    bot.line.fill.background()
    tf = bot.text_frame
    tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.08)
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p1 = tf.paragraphs[0]; p1.alignment = 2
    r1 = p1.add_run()
    r1.text = "Resultado intelectual da Etapa 6"
    r1.font.size = Pt(14); r1.font.bold = True
    r1.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.alignment = 2
    r2 = p2.add_run()
    r2.text = (
        "Primeiro diagnóstico quantitativo na literatura sobre o "
        "LIMITE TEÓRICO de equidade em classificadores raciais."
    )
    r2.font.size = Pt(12); r2.font.color.rgb = BLUE_LIGHT
    p3 = tf.add_paragraph(); p3.alignment = 2
    r3 = p3.add_run()
    r3.text = (
        "Regra:  R² fenotípico ≥ 70 %  →  confirma Pangelinan.   "
        "R² fenotípico < 70 %  →  evidência contrária, nova contribuição teórica."
    )
    r3.font.size = Pt(10); r3.font.italic = True
    r3.font.color.rgb = BLUE_LIGHT

    add_footer(slide)
    add_page_number(slide, 25)


# ============================================================
# Main
# ============================================================
def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # 33 slides (era 29): adicionados 4 slides de detalhamento por etapa
    # do pipeline — Etapa 2 (matriz MST x raca), Etapa 4 (baselines),
    # Etapa 5 (transferencia RFW/BFW), Etapa 6 (decomposicao do erro).
    slide_capa(prs)                           # 1
    slide_motivacao_academica(prs)            # 2
    slide_motivacao_contexto(prs)             # 3
    slide_motivacao_regulacao(prs)            # 4
    slide_problema_disparidade(prs)           # 5
    slide_problema_heterogeneidade(prs)       # 6
    slide_problema_refutacao(prs)             # 7
    slide_objetivo_geral(prs)                 # 8
    slide_objetivos_hipoteses(prs)            # 9
    slide_revisao_timeline(prs)               # 10
    slide_revisao_mitigacao(prs)              # 11
    slide_revisao_lacunas(prs)                # 12
    slide_metodologia_pipeline(prs)           # 13
    slide_evolucao_pipeline(prs)              # 14
    slide_metodologia_etapa1(prs)             # 15
    slide_mst_detalhe_tecnico(prs)            # 16
    slide_mst_validacao_faces(prs)            # 17
    slide_mst_validacao_distribuicao(prs)     # 18
    slide_etapa2_matriz_mst_raca(prs)         # 19  [NOVO — Etapa 2 detalhamento]
    slide_por_que_convnext(prs)               # 20  [Etapa 3a — ConvNeXt arquitetura]
    slide_por_que_film(prs)                   # 21  [Etapa 3b — FiLM mecanismo + figura]
    slide_metodologia_configs(prs)            # 22
    slide_etapa4_baselines(prs)               # 23  [Etapa 4 detalhamento]
    slide_etapa5_transferencia(prs)           # 24  [Etapa 5 detalhamento]
    slide_etapa6_decomposicao(prs)            # 25  [Etapa 6 detalhamento]
    slide_baselines_cenarios(prs)             # 26
    slide_metricas(prs)                       # 27
    slide_contribuicoes(prs)                  # 28
    slide_cronograma(prs)                     # 29
    slide_riscos(prs)                         # 30
    slide_estado_atual(prs)                   # 31
    slide_perguntas(prs)                      # 32

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
