"""Gera apresentacao PowerPoint da defesa da qualificacao (30/09/2026).

25 slides, formato 16:9, ~15-20 min de fala. Linguagem simples,
sem jargao, entendivel por leigos.

Estrutura:
    1  Capa
    2  Agenda
    3  Motivacao - contexto
    4  Motivacao - regulacao
    5  Problema - disparidade racial (com figura)
    6  Problema - heterogeneidade fenotipica
    7  Problema - refutacao (tom de pele > raca)
    8  Objetivo geral
    9  Objetivos especificos (6)
    10 Hipoteses (6)
    11 Revisao - visao geral (6 frentes, 104 fichas)
    12 Revisao - mitigacao algoritmica
    13 Revisao - heterogeneidade fenotipica
    14 Revisao - 5 lacunas identificadas
    15 Metodologia - pipeline 6 etapas (com figura)
    16 Metodologia - Etapa 1 (classificador MST proprio)
    17 Metodologia - Etapa 3 (FiLM + 3 configuracoes)
    18 Metodologia - baselines + cenarios
    19 Metodologia - triangulacao de metricas
    20 Contribuicoes (3 eixos, 7 contribuicoes)
    21 Cronograma
    22 Riscos + mitigacoes
    23 Estado atual - o que ja foi feito
    24 Consideracoes finais
    25 Perguntas / obrigado

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
TOTAL_SLIDES = 25


# ============================================================
# Helpers de layout
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


def add_footer(slide, text: str = "Qualificacao - Marcello Ozzetti - Prof. Marcos Quiles - UNIFESP/ICT") -> None:
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(11.5), Inches(0.4))
    pf = ft.text_frame.paragraphs[0]
    pf.text = text
    pf.font.size = Pt(9)
    pf.font.color.rgb = GRAY_MD
    pf.font.italic = True


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_bullets(prs: Presentation, number: int, title: str, bullets: list) -> None:
    slide = _blank(prs)
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
            r1.font.size = Pt(16)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(16)
            r2.font.color.rgb = GRAY_DK
        else:
            p.text = "-  " + item
            p.font.size = Pt(16)
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
    slide = _blank(prs)
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
    slide = _blank(prs)
    add_title(slide, title)
    if image_path.exists():
        pic = slide.shapes.add_picture(
            str(image_path), Inches(1.0), Inches(1.5), height=Inches(height_in)
        )
        # centralizar horizontalmente
        pic.left = Inches((13.33 - pic.width.inches) / 2)
    else:
        # placeholder
        _blank_placeholder(slide, f"[figura ausente: {image_path.name}]")
    if caption:
        cap = slide.shapes.add_textbox(Inches(0.5), Inches(6.55), Inches(12.5), Inches(0.4))
        p = cap.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = GRAY_MD
        p.alignment = 2  # center
    add_footer(slide)
    add_page_number(slide, number)


def _blank_placeholder(slide, text: str) -> None:
    tb = slide.shapes.add_textbox(Inches(2), Inches(3), Inches(9), Inches(1))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(14)
    p.font.italic = True
    p.font.color.rgb = ACCENT


# ============================================================
# SLIDE 1 - Capa
# ============================================================
def slide_capa(prs: Presentation) -> None:
    slide = _blank(prs)

    # barra lateral navy
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(2.5), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # UNIFESP na barra
    lab = slide.shapes.add_textbox(Inches(0.2), Inches(0.5), Inches(2.2), Inches(1.5))
    tf = lab.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "UNIFESP"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.text = "Instituto de\nCiencia e\nTecnologia"
    p2.font.size = Pt(11)
    p2.font.color.rgb = BLUE_LIGHT

    # subtitle "Qualificacao"
    sub = slide.shapes.add_textbox(Inches(3.0), Inches(1.2), Inches(10), Inches(0.6))
    p = sub.text_frame.paragraphs[0]
    p.text = "EXAME DE QUALIFICACAO"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    # titulo principal
    tit = slide.shapes.add_textbox(Inches(3.0), Inches(2.0), Inches(10), Inches(2.5))
    tf = tit.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Mitigacao de vies racial em"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "classificacao facial com"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph()
    p3.text = "condicionamento por tom de pele"
    p3.font.size = Pt(32)
    p3.font.bold = True
    p3.font.color.rgb = NAVY

    # meta
    meta = slide.shapes.add_textbox(Inches(3.0), Inches(5.0), Inches(10), Inches(2.2))
    tf = meta.text_frame
    tf.word_wrap = True
    rows = [
        ("Mestrando:", "Marcello Vinicius Alves Ozzetti Cruz"),
        ("Orientador:", "Prof. Dr. Marcos Goncalves Quiles"),
        ("Programa:", "Pos-Graduacao em Ciencia da Computacao"),
        ("Data:", f"{QUALIFICACAO.strftime('%d de setembro de %Y')}"),
        ("Local:", "UNIFESP / ICT - Sao Jose dos Campos"),
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
# SLIDE 2 - Agenda
# ============================================================
def slide_agenda(prs: Presentation) -> None:
    add_bullets(prs, 2, "Roteiro da apresentacao", [
        ("1.  Motivacao", "por que esse problema importa hoje"),
        ("2.  Problema", "o que exatamente esta errado"),
        ("3.  Objetivos e hipoteses", "o que vamos investigar"),
        ("4.  Revisao da literatura", "o que a comunidade ja tentou"),
        ("5.  Metodologia", "como pretendemos resolver"),
        ("6.  Contribuicoes esperadas", "o que ficara para a area"),
        ("7.  Cronograma e riscos", "quando e o que pode dar errado"),
        ("8.  Estado atual", "o que ja esta feito"),
    ])


# ============================================================
# SLIDE 3 - Motivacao: contexto
# ============================================================
def slide_motivacao_contexto(prs: Presentation) -> None:
    add_bullets(prs, 3, "Reconhecimento facial esta em toda parte", [
        "Desbloqueio de celular, autenticacao bancaria, controle de fronteiras, identificacao policial.",
        "Deixou de ser tecnologia de laboratorio: e infraestrutura social.",
        ("NIST 2019:", "maior auditoria publica ja feita em biometria facial - 189 algoritmos comerciais, 18 milhoes de imagens."),
        ("Resultado:", "diferenca de 10 a 100 vezes na taxa de falso positivo entre grupos raciais."),
        "Em resumo: a tecnologia funciona bem para a maioria, mas nao funciona igualmente bem para todos.",
    ])


# ============================================================
# SLIDE 4 - Motivacao: regulacao
# ============================================================
def slide_motivacao_regulacao(prs: Presentation) -> None:
    add_bullets(prs, 4, "O tema virou obrigacao regulatoria", [
        ("European AI Act (2024):", "primeira lei que regula sistemas de IA de alto risco na Uniao Europeia."),
        ("Auditoria de equidade:", "requisito formal para sistemas biometricos - nao mais opcional."),
        ("Consequencias praticas:", "casos documentados de prisao errada por falso match ja resultaram em processos e moratorias."),
        "A pergunta cientifica mudou. Nao e mais 'existe vies?' - isso ja esta provado. E 'como mitigar de forma defensavel?'",
    ])


# ============================================================
# SLIDE 5 - Problema (com figura)
# ============================================================
def slide_problema_disparidade(prs: Presentation) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    img = repo_root / "docs" / "tese" / "images" / "fig_disparidade_racial.png"
    add_image_slide(
        prs, 5,
        "Existe uma disparidade estavel de ~30 pontos entre grupos",
        img,
        caption="Melhor modelo publico (FaceScanPaliGemma, 2024) sobre o dataset FairFace. Diferenca entre Black e Latinx = 30 pontos.",
        height_in=4.8,
    )


# ============================================================
# SLIDE 6 - Problema: heterogeneidade
# ============================================================
def slide_problema_heterogeneidade(prs: Presentation) -> None:
    add_bullets(prs, 6, "Por que a classe Latinx e a mais dificil?", [
        "A categoria 'Latinx/Hispanic' agrupa fenotipos muito diferentes em uma so etiqueta.",
        ("Antropologia:", "estudos em quatro paises latino-americanos documentam ampla variacao de tom de pele dentro da populacao (Telles, 2014)."),
        ("Genetica:", "estudo com 162 mil individuos mostra composicao ancestral altamente variavel (Bryc, 2015)."),
        ("Sociologia:", "identidade Hispanic diminui de 97% para 50% ao longo de quatro geracoes nos EUA (Pew, 2017)."),
        "A rotulagem monolitica de 'raca' esconde essa diversidade e prejudica o modelo.",
    ])


# ============================================================
# SLIDE 7 - Problema: refutacao
# ============================================================
def slide_problema_refutacao(prs: Presentation) -> None:
    add_bullets(prs, 7, "E se o problema nao for 'raca' e sim 'tom de pele'?", [
        ("Pangelinan et al. (2023):", "argumenta que a fracao de face util na imagem (pixel information) explica melhor o gap do que o rotulo de raca."),
        ("Matias et al. (2026):", "publica o SkinToneNet - primeiro classificador de tom de pele Monk em larga escala."),
        "Se essa refutacao estiver certa, a pesquisa em equidade facial deveria olhar mais para tom de pele.",
        ("Nossa aposta:", "o tom de pele nao substitui a raca - ele funciona como sinal de contexto que ajuda a rede a decidir melhor."),
        "Vamos testar isso como hipotese formal, incluindo os cenarios em que a refutacao ganha.",
    ])


# ============================================================
# SLIDE 8 - Objetivo geral
# ============================================================
def slide_objetivo_geral(prs: Presentation) -> None:
    slide = _blank(prs)
    add_title(slide, "Objetivo geral")

    # caixa central destacada
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.5), Inches(2.2), Inches(10.3), Inches(2.6),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(2.0), Inches(2.6), Inches(9.3), Inches(2.0))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = ("Desenvolver e avaliar um pipeline de classificacao racial "
              "que usa o tom de pele (escala Monk, 10 tons) como sinal "
              "de contexto para reduzir a disparidade entre grupos, "
              "sem sacrificar a acuracia geral.")
    p.font.size = Pt(20)
    p.font.color.rgb = WHITE

    # nota abaixo
    nota = slide.shapes.add_textbox(Inches(1.5), Inches(5.2), Inches(10.3), Inches(1.5))
    tf = nota.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Arquitetura escolhida: ConvNeXt-T (rede base moderna) + FiLM (mecanismo que 'consulta' o tom de pele antes de decidir)."
    p.font.size = Pt(14)
    p.font.italic = True
    p.font.color.rgb = GRAY_DK
    add_footer(slide)
    add_page_number(slide, 8)


# ============================================================
# SLIDE 9 - Objetivos especificos
# ============================================================
def slide_objetivos_especificos(prs: Presentation) -> None:
    add_table_slide(
        prs, 9,
        "Seis objetivos especificos",
        ["#", "Objetivo"],
        [
            ["1", "Quantificar como o tom de pele Monk se distribui dentro de cada grupo racial do FairFace."],
            ["2", "Treinar e avaliar um classificador de tom de pele proprio, com sensitivity analysis contra alternativas."],
            ["3", "Implementar o pipeline com FiLM sobre ConvNeXt-T e compara-lo contra seis baselines de mitigacao."],
            ["4", "Demonstrar que o ganho em classificacao transfere para reconhecimento (RFW/BFW)."],
            ["5", "Formalizar a triangulacao de metricas - Disparity Ratio, F1 da pior classe, Equal Opportunity."],
            ["6", "Decompor quanto do erro Latinx vem do fenotipo (irredutivel) e quanto do modelo (mitigavel)."],
        ],
        col_widths=[0.5, 12.0],
        highlight_rows=[2, 5],
        font_size=13,
    )


# ============================================================
# SLIDE 10 - Hipoteses
# ============================================================
def slide_hipoteses(prs: Presentation) -> None:
    add_table_slide(
        prs, 10,
        "Seis hipoteses testaveis",
        ["#", "Hipotese", "Confirma se..."],
        [
            ["H1", "O classificador MST proprio atinge concordancia humana suficiente.", "kappa >= 0.7 com anotacoes internas"],
            ["H2", "A classe Latinx cobre >= 5 dos 10 tons Monk no FairFace.", "cobertura observada >= 5"],
            ["H3", "Condicionar por tom reduz disparidade sem perder acuracia agregada.", "DR menor + F1 macro >= baseline"],
            ["H4", ">= 50% dos erros Latinx concentram-se em zonas de sobreposicao de tom.", "concentracao observada >= 50%"],
            ["H5", "O ganho em classificacao transfere para reconhecimento.", "reducao de gap tambem em RFW/BFW"],
            ["H6", "Parte substancial do gap e explicada por pixel information (Pangelinan).", "R^2 explicado >= 70%"],
        ],
        col_widths=[0.7, 7.5, 4.3],
        font_size=11,
    )


# ============================================================
# SLIDE 11 - Revisao visao geral
# ============================================================
def slide_revisao_visao_geral(prs: Presentation) -> None:
    add_bullets(prs, 11, "Como a comunidade tem atacado o problema", [
        ("6 frentes de literatura mapeadas:", "datasets balanceados; funcoes de perda; arquiteturas; modelos vision-language; MST; heterogeneidade fenotipica."),
        ("104 fichas bibliograficas revisadas", "no periodo 2014-2026, com 56% dos trabalhos posteriores a 2022."),
        ("Fundacoes classicas:", "Gender Shades (Buolamwini 2018), Hardt (2016) - definicoes formais de equidade."),
        ("Marcos consolidados:", "FairFace (Karkkainen 2021), RFW (Wang 2019), Impossibilidade (Kleinberg 2017)."),
        ("Fronteira atual:", "FSCL+ (Park 2022), FineFACE (Manzoor 2024), SkinToneNet (Matias 2026)."),
    ])


# ============================================================
# SLIDE 12 - Revisao mitigacao algoritmica
# ============================================================
def slide_revisao_mitigacao(prs: Presentation) -> None:
    add_table_slide(
        prs, 12,
        "O que ja se tentou para mitigar o vies (usaremos como baselines)",
        ["Metodo", "Ano", "Ideia central"],
        [
            ["Adversarial debiasing (Zhang)", "2018", "Segundo modelo tenta 'ler' o atributo sensivel - o primeiro se defende."],
            ["Group DRO (Sagawa)", "2020", "Otimiza o erro do pior grupo, nao o erro medio."],
            ["FSCL+ (Park)", "2022", "Aprendizado contrastivo com controle de vies."],
            ["FineFACE (Manzoor)", "2024", "Atencao entre camadas para operar em fronteira Pareto."],
            ["ResNet-34 (Karkkainen)", "2021", "Baseline canonico da propria comunidade FairFace."],
            ["ConvNeXt-T puro", "2022", "Controle: efeito da rede moderna sem condicionamento."],
        ],
        col_widths=[3.5, 1.0, 8.0],
        font_size=12,
    )


# ============================================================
# SLIDE 13 - Revisao heterogeneidade
# ============================================================
def slide_revisao_heterogeneidade(prs: Presentation) -> None:
    add_bullets(prs, 13, "Evidencia externa a computacao: o rotulo racial esconde variacao", [
        ("Antropologia biologica (Telles, 2014):", "projeto PERLA em 4 paises latino-americanos - documenta 'pigmentocracia'."),
        ("Genetica populacional (Bryc, 2015):", "162 mil individuos - composicao ancestral variavel em Latinos."),
        ("Sociologia identitaria (Pew, 2017):", "identidade Hispanic e fluida ao longo de geracoes."),
        "Tres disciplinas independentes convergem: o rotulo racial no FairFace nao captura variacao fenotipica real.",
        ("Consequencia:", "condicionar por tom de pele nao substitui o rotulo, mas oferece ao modelo uma pista objetiva que a rotulagem sozinha nao da."),
    ])


# ============================================================
# SLIDE 14 - Revisao lacunas
# ============================================================
def slide_revisao_lacunas(prs: Presentation) -> None:
    add_table_slide(
        prs, 14,
        "5 lacunas cientificas identificadas na literatura",
        ["#", "Lacuna", "Como enderecamos"],
        [
            ["L1", "Nao existe matriz publica MST x classes raciais no FairFace.", "Etapa 2 do pipeline - Contribuicao 2."],
            ["L2", "FiLM nunca foi aplicado em classificacao racial multi-classe.", "Etapa 3 - Contribuicao 3."],
            ["L3", "Metricas de fairness multi-classe estao fragmentadas na literatura.", "Etapa 4 - Contribuicao 4 (triangulacao)."],
            ["L4", "Transferencia de fairness para reconhecimento e pouco estudada.", "Etapa 5 - Contribuicao 5."],
            ["L5", "Nao ha decomposicao quantitativa do gap Latinx.", "Etapa 6 - Contribuicao 6."],
        ],
        col_widths=[0.6, 6.5, 5.4],
        font_size=12,
    )


# ============================================================
# SLIDE 15 - Metodologia pipeline (com figura)
# ============================================================
def slide_metodologia_pipeline(prs: Presentation) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    img = repo_root / "docs" / "tese" / "images" / "fig_pipeline_6etapas.png"
    add_image_slide(
        prs, 15,
        "Pipeline em 6 etapas",
        img,
        caption="Fluxo top-down: 3 fases (diagnostico, metodo, validacao). Etapa 3 e o metodo proposto.",
        height_in=5.4,
    )


# ============================================================
# SLIDE 16 - Metodologia Etapa 1
# ============================================================
def slide_metodologia_etapa1(prs: Presentation) -> None:
    add_bullets(prs, 16, "Etapa 1 - Classificador de tom de pele (proprio)", [
        ("O que faz:", "recebe uma foto de rosto e devolve as 10 probabilidades da escala Monk."),
        ("Decisao pos-reuniao Ago/2026:", "treinar nosso proprio classificador ao inves de depender do SkinToneNet (que ainda nao foi liberado)."),
        ("Datasets de treino:", "MSTE (Google) e Casual Conversations v2 (Meta) - publicos hoje."),
        ("Camada auto-suficiente:", "detecta o rosto, alinha, corta e classifica - funciona em qualquer dataset, ate os que nao tem anotacao."),
        ("Validacao:", "protocolo humano interno (~250 imagens do FairFace) + comparacao com 2-3 classificadores alternativos."),
    ])


# ============================================================
# SLIDE 17 - Metodologia Etapa 3 (3 configuracoes)
# ============================================================
def slide_metodologia_etapa3(prs: Presentation) -> None:
    add_table_slide(
        prs, 17,
        "Etapa 3 - Classificador racial com FiLM (3 configuracoes)",
        ["ID", "Configuracao", "O que testa"],
        [
            ["A", "ConvNeXt-T puro (baseline)", "Sem condicionamento - controle."],
            ["B", "ConvNeXt-T + FiLM (sinal MST direto)", "Proposta principal - o tom de pele entra como contexto."],
            ["C", "ConvNeXt-T + FiLM (sinal via CLIP-text)", "Alternativa: sinal em espaco semantico ao inves de numerico."],
        ],
        col_widths=[0.6, 4.9, 7.0],
        highlight_rows=[1],
        font_size=13,
    )


# ============================================================
# SLIDE 18 - Baselines + Cenarios
# ============================================================
def slide_baselines_cenarios(prs: Presentation) -> None:
    add_bullets(prs, 18, "Como validamos - baselines e cenarios", [
        ("Comparamos contra 6 baselines:", "ResNet-34, ConvNeXt-T puro, FSCL+, Group DRO, FineFACE, Adversarial debiasing."),
        ("Cenario A - apenas raca:", "metricas por classe racial (7 classes do FairFace)."),
        ("Cenario B - raca X genero:", "analise interseccional (8 subgrupos, seguindo Gender Shades)."),
        ("Norma seguida:", "ISO/IEC 19795-10:2024 - padrao internacional para reporte de desempenho biometrico entre grupos."),
        ("Rigor experimental:", "3 sementes independentes por experimento (42, 1, 2), comparacao pareada, intervalo de confianca 95% por bootstrap."),
    ])


# ============================================================
# SLIDE 19 - Triangulacao de metricas
# ============================================================
def slide_metricas(prs: Presentation) -> None:
    add_bullets(prs, 19, "Triangulacao de metricas (nenhuma metrica isolada basta)", [
        ("Por que triangular:", "Teorema da Impossibilidade (Kleinberg 2017) - nao existe metrica unica de equidade que satisfaca tudo."),
        ("Disparity Ratio:", "razao entre pior e melhor F1. Mede quao desigual e o desempenho entre grupos."),
        ("F1 da pior classe:", "garante que a melhoria nao vem so no meio - protege o grupo mais fraco."),
        ("Equal Opportunity:", "iguala a taxa de acerto entre grupos, condicionada ao rotulo verdadeiro."),
        ("Equalized Odds:", "vai alem: iguala acerto e erro entre grupos."),
        ("Visualizacao Pareto:", "para cada modelo, plotamos (acuracia agregada) x (disparidade) - vemos quem esta na fronteira."),
    ])


# ============================================================
# SLIDE 20 - Contribuicoes
# ============================================================
def slide_contribuicoes(prs: Presentation) -> None:
    add_table_slide(
        prs, 20,
        "Contribuicoes esperadas (3 eixos, 7 contribuicoes)",
        ["Eixo", "Contribuicoes", "Foco"],
        [
            ["Fenotipico-empirico", "1, 2", "Documentar como o fenotipo (MST) se distribui dentro de cada rotulo racial."],
            ["Metodologico-arquitetural", "3, 4, 7", "Injetar tom de pele como contexto arquitetural + triangulacao de metricas."],
            ["Diagnostico-estrutural", "5, 6", "Decompor o erro em fenotipico (irredutivel) e algoritmico (mitigavel)."],
        ],
        col_widths=[3.5, 2.0, 7.0],
        font_size=13,
    )


# ============================================================
# SLIDE 21 - Cronograma
# ============================================================
def slide_cronograma(prs: Presentation) -> None:
    add_table_slide(
        prs, 21,
        "Cronograma",
        ["Periodo", "Etapa", "Entrega"],
        [
            ["Ago/2026 (hoje)", "Preparacao adiantada", "Codigo das Etapas 1 e 2 pronto e testado."],
            ["Set/2026", "QUALIFICACAO", "Marco atual."],
            ["Out/2026", "Aplicar sugestoes da banca", "Ajustes de texto e escopo."],
            ["Nov/2026", "Etapa 1 formal", "Treinar classificador MST + validacao humana."],
            ["Dez/2026", "Etapa 2", "Matriz publica MST x raca."],
            ["Jan-Mar/2027", "Etapa 3", "Ablation 3 configs FiLM."],
            ["Abr/2027", "Etapa 4", "Comparacao vs 6 baselines."],
            ["Mai/2027", "Etapa 5", "Transferencia RFW/BFW."],
            ["Jun/2027", "Etapa 6", "Sintese decompositiva."],
            ["2o sem 2027", "Redacao final + DEFESA", "Encerramento."],
        ],
        col_widths=[2.5, 3.0, 7.0],
        highlight_rows=[0, 1, 9],
        font_size=11,
    )


# ============================================================
# SLIDE 22 - Riscos
# ============================================================
def slide_riscos(prs: Presentation) -> None:
    add_table_slide(
        prs, 22,
        "Riscos identificados e mitigacoes",
        ["#", "Risco", "Mitigacao"],
        [
            ["R1", "Alguma das 6 hipoteses pode ser refutada.", "Cada hipotese tem plano B; refutacao e resultado cientifico valido."],
            ["R2", "Qualidade do nosso classificador de tom de pele.", "Validacao humana + sensitivity com outros classificadores + benchmark externo (STW) quando disponivel."],
            ["R3", "Refutacao parcial pela linha do Pangelinan (pixel info > tom).", "Ja incluida como hipotese H6 - refutacao vira contribuicao quantitativa."],
            ["R4", "Custo computacional do treinamento com 3 sementes.", "ConvNeXt-T e leve (~28M params); estimativa 200-400h GPU total."],
        ],
        col_widths=[0.6, 4.5, 7.4],
        highlight_rows=[1],
        font_size=12,
    )


# ============================================================
# SLIDE 23 - Estado atual (destaque visual do adiantamento)
# ============================================================
def slide_estado_atual(prs: Presentation) -> None:
    slide = _blank(prs)
    add_title(slide, "Estado atual - adiantados no cronograma")

    # Faixa superior: 3 numeros destacados
    kpis = [
        ("+3", "meses adiantados", "sobre o cronograma proposto na qualificacao"),
        ("2 de 6", "etapas com codigo pronto", "Etapas 1 e 2 funcionando ponta a ponta"),
        ("50", "testes automatizados", "todos passando (unit + smoke)"),
    ]
    box_w = 4.0
    gap = 0.25
    start_x = (13.33 - (3 * box_w + 2 * gap)) / 2
    top_y = 1.6

    for i, (big, mid, small) in enumerate(kpis):
        x = start_x + i * (box_w + gap)
        # caixa
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(top_y), Inches(box_w), Inches(2.2),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if i == 0 else GRAY_LT
        box.line.fill.background()

        # numero grande
        tx = slide.shapes.add_textbox(Inches(x), Inches(top_y + 0.15), Inches(box_w), Inches(0.9))
        p = tx.text_frame.paragraphs[0]
        p.text = big
        p.font.size = Pt(48)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 0 else NAVY
        p.alignment = 2  # center

        # label medio
        tx2 = slide.shapes.add_textbox(Inches(x), Inches(top_y + 1.15), Inches(box_w), Inches(0.5))
        p = tx2.text_frame.paragraphs[0]
        p.text = mid
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 0 else NAVY
        p.alignment = 2

        # descricao pequena
        tx3 = slide.shapes.add_textbox(Inches(x + 0.1), Inches(top_y + 1.65), Inches(box_w - 0.2), Inches(0.6))
        tf = tx3.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = small
        p.font.size = Pt(10)
        p.font.italic = True
        p.font.color.rgb = BLUE_LIGHT if i == 0 else GRAY_MD
        p.alignment = 2

    # Bloco inferior - o que ja esta pronto
    subtitle = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.5), Inches(0.4))
    p = subtitle.text_frame.paragraphs[0]
    p.text = "Entregas concretas ate 18/08/2026"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.5), Inches(2.2))
    tf = tx.text_frame
    tf.word_wrap = True
    entregas = [
        ("Etapa 1 (codigo pronto):", "classificador de tom de pele proprio + camada auto-suficiente + trainer + cache."),
        ("Etapa 2 (codigo pronto):", "matriz publica MST x raca + spread + entropia + teste da hipotese H2."),
        ("Etapas 3 a 6:", "estrutura de codigo criada, aguardando execucao formal (Nov/2026 em diante)."),
        ("Texto da dissertacao:", "5 capitulos consolidados, 3 passadas de revisao de estilo, 104 fichas bibliograficas."),
    ]
    for i, (head, body) in enumerate(entregas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "-  " + head + "  "
        r1.font.size = Pt(13)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(13)
        r2.font.color.rgb = GRAY_DK
        p.space_after = Pt(6)

    add_footer(slide)
    add_page_number(slide, 23)


# ============================================================
# SLIDE 24 - Consideracoes finais
# ============================================================
def slide_consideracoes(prs: Presentation) -> None:
    add_bullets(prs, 24, "Por que este trabalho e oportuno agora", [
        "Regulacao europeia recem-aprovada exige auditoria de equidade em sistemas biometricos.",
        "Literatura de 2023-2026 converge para tom de pele como pista central, mas nao ha aplicacao arquitetural comprovada.",
        "Ferramentas necessarias amadureceram: SkinToneNet, MSTE, Casual Conversations v2, FairFace, RFW, BFW.",
        "A proposta agrega 3 vetores: fenotipico (empirico), metodologico (arquitetural) e diagnostico (decompositivo).",
        "Diferencial cientifico: nao busca apenas 'reduzir F1 medio' - busca entender quanto do erro e mitigavel e quanto e limite do problema.",
    ])


# ============================================================
# SLIDE 25 - Perguntas
# ============================================================
def slide_perguntas(prs: Presentation) -> None:
    slide = _blank(prs)

    # fundo navy total
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
    p.alignment = 2  # center

    p2 = tf.add_paragraph()
    p2.text = "Perguntas?"
    p2.font.size = Pt(28)
    p2.font.color.rgb = BLUE_LIGHT
    p2.alignment = 2

    # rodape
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.5))
    pf = ft.text_frame.paragraphs[0]
    pf.text = "Marcello Ozzetti - marcello.ozzetti@gmail.com - UNIFESP/ICT"
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
    slide_problema_heterogeneidade(prs)
    slide_problema_refutacao(prs)
    slide_objetivo_geral(prs)
    slide_objetivos_especificos(prs)
    slide_hipoteses(prs)
    slide_revisao_visao_geral(prs)
    slide_revisao_mitigacao(prs)
    slide_revisao_heterogeneidade(prs)
    slide_revisao_lacunas(prs)
    slide_metodologia_pipeline(prs)
    slide_metodologia_etapa1(prs)
    slide_metodologia_etapa3(prs)
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
