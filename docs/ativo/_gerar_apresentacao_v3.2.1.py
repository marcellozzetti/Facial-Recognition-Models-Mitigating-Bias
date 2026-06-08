"""Gera apresentação PowerPoint v3.2.1 — narrativa pedagógica.

Estrutura em 4 partes:
1. Onde paramos (recap da última reunião)
2. O que avancei nesta semana (resposta aos 4 pedidos do orientador)
3. Como a tese está sendo construída (v3.2 com conceitos explicados)
4. Próximos passos

Princípios:
- Linguagem narrativa, não tabular dense
- Cada termo técnico explicado ANTES do uso
- Voz coletiva em 1ª pessoa do plural ("validamos", "combinamos")
- Analogias para conceitos complexos
- Máximo 5-6 bullets por slide
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# --- Paleta acadêmica sóbria ---
NAVY = RGBColor(0x1F, 0x2A, 0x4E)
GRAY_DK = RGBColor(0x3D, 0x42, 0x4E)
GRAY_MD = RGBColor(0x70, 0x76, 0x82)
GRAY_LT = RGBColor(0xE8, 0xEA, 0xED)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def add_title_slide(prs: Presentation) -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(2.2), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(2.5), Inches(1.5), Inches(10.5), Inches(2.0))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Equidade Racial em Classificação Facial"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "Atualização da pesquisa — o que evoluiu nesta semana"
    p2.font.size = Pt(22)
    p2.font.color.rgb = GRAY_DK

    meta = slide.shapes.add_textbox(Inches(2.5), Inches(4.2), Inches(10.5), Inches(2.5))
    mf = meta.text_frame
    mf.word_wrap = True

    rows = [
        ("Mestrando:", "Marcello Ozzetti"),
        ("Orientador:", "Prof. Marcos Quiles"),
        ("Programa:", "Mestrado em Ciência da Computação — Unifesp / ICT"),
        ("Reunião:", "Junho de 2026 (segunda reunião do mês)"),
    ]
    for i, (k, v) in enumerate(rows):
        p = mf.paragraphs[0] if i == 0 else mf.add_paragraph()
        p.text = f"{k}  {v}"
        p.font.size = Pt(17)
        p.font.color.rgb = GRAY_DK


def add_section_divider(prs: Presentation, num: str, title: str, subtitle: str = "") -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    tx_num = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(3), Inches(3))
    pn = tx_num.text_frame.paragraphs[0]
    pn.text = num
    pn.font.size = Pt(130)
    pn.font.bold = True
    pn.font.color.rgb = WHITE

    tx_t = slide.shapes.add_textbox(Inches(4.3), Inches(2.5), Inches(8.5), Inches(2))
    tf = tx_t.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(20)
        p2.font.color.rgb = GRAY_LT


def add_content_slide(prs: Presentation, title: str, bullets: list, footer: str = "") -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    tf = tx_t.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(12.5), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()

    tx_b = slide.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(12.3), Inches(5.4))
    bf = tx_b.text_frame
    bf.word_wrap = True

    for i, b in enumerate(bullets):
        if isinstance(b, tuple):
            text, level = b
        else:
            text, level = b, 0

        p = bf.paragraphs[0] if i == 0 else bf.add_paragraph()
        p.text = text
        p.level = level
        if level == 0:
            p.font.size = Pt(20)
            p.font.color.rgb = GRAY_DK
            p.font.bold = False
        else:
            p.font.size = Pt(17)
            p.font.color.rgb = GRAY_MD
        p.space_after = Pt(8)

    if footer:
        tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.5), Inches(0.4))
        pf = tx_f.text_frame.paragraphs[0]
        pf.text = footer
        pf.font.size = Pt(15)
        pf.font.color.rgb = GRAY_MD
        pf.font.italic = True


def add_explainer_slide(prs: Presentation, title: str, simple_def: str, why_matters: str, concrete: str = "", footer: str = "") -> None:
    """Slide explicativo: 'O que é X?' — definição simples + por que importa + exemplo concreto."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(12.5), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()

    # Em palavras simples
    box1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.6), Inches(12.1), Inches(1.7))
    box1.fill.solid()
    box1.fill.fore_color.rgb = GRAY_LT
    box1.line.fill.background()
    tf1 = box1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.3); tf1.margin_right = Inches(0.3); tf1.margin_top = Inches(0.15)
    p1 = tf1.paragraphs[0]
    p1.text = "Em palavras simples"
    p1.font.size = Pt(15)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1b = tf1.add_paragraph()
    p1b.text = simple_def
    p1b.font.size = Pt(18)
    p1b.font.color.rgb = GRAY_DK

    # Por que importa
    box2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(3.5), Inches(12.1), Inches(1.7))
    box2.fill.solid()
    box2.fill.fore_color.rgb = WHITE
    box2.line.color.rgb = NAVY
    box2.line.width = Pt(1)
    tf2 = box2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.3); tf2.margin_right = Inches(0.3); tf2.margin_top = Inches(0.15)
    p2 = tf2.paragraphs[0]
    p2.text = "Por que importa para a tese"
    p2.font.size = Pt(15)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2b = tf2.add_paragraph()
    p2b.text = why_matters
    p2b.font.size = Pt(18)
    p2b.font.color.rgb = GRAY_DK

    # Exemplo concreto
    if concrete:
        box3 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.4), Inches(12.1), Inches(1.6))
        box3.fill.solid()
        box3.fill.fore_color.rgb = WHITE
        box3.line.color.rgb = GREEN
        box3.line.width = Pt(1)
        tf3 = box3.text_frame
        tf3.word_wrap = True
        tf3.margin_left = Inches(0.3); tf3.margin_right = Inches(0.3); tf3.margin_top = Inches(0.15)
        p3 = tf3.paragraphs[0]
        p3.text = "Exemplo concreto"
        p3.font.size = Pt(15)
        p3.font.bold = True
        p3.font.color.rgb = GREEN
        p3b = tf3.add_paragraph()
        p3b.text = concrete
        p3b.font.size = Pt(17)
        p3b.font.color.rgb = GRAY_DK

    if footer:
        tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.5), Inches(0.4))
        pf = tx_f.text_frame.paragraphs[0]
        pf.text = footer
        pf.font.size = Pt(14)
        pf.font.color.rgb = GRAY_MD
        pf.font.italic = True


def add_thesis_slide(prs: Presentation, title: str, statement: str, footer: str = "") -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY

    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.4))
    box.fill.solid()
    box.fill.fore_color.rgb = GRAY_LT
    box.line.color.rgb = NAVY
    box.line.width = Pt(2)

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.5); tf.margin_right = Inches(0.5); tf.margin_top = Inches(0.4)

    p = tf.paragraphs[0]
    p.text = statement
    p.font.size = Pt(20)
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.LEFT

    if footer:
        tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.5), Inches(0.4))
        pf = tx_f.text_frame.paragraphs[0]
        pf.text = footer
        pf.font.size = Pt(14)
        pf.font.color.rgb = GRAY_MD
        pf.font.italic = True


def add_table_slide(prs: Presentation, title: str, headers: list, rows: list, footer: str = "") -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(12.5), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()

    nrows = len(rows) + 1
    ncols = len(headers)
    table = slide.shapes.add_table(nrows, ncols, Inches(0.5), Inches(1.5), Inches(12.5), Inches(5.4)).table

    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = WHITE
                r.font.size = Pt(16)

    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else GRAY_LT
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(15)
                    r.font.color.rgb = GRAY_DK

    if footer:
        tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.5), Inches(0.4))
        pf = tx_f.text_frame.paragraphs[0]
        pf.text = footer
        pf.font.size = Pt(15)
        pf.font.color.rgb = GRAY_MD
        pf.font.italic = True


def build_presentation(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ==================== TÍTULO + SUMÁRIO ====================
    add_title_slide(prs)

    add_content_slide(
        prs,
        "Principais alinhamentos e decisões",
        [
            "Parte 1 — Onde paramos na última reunião",
            ("Recap da última reunião e orientações", 1),
            "Parte 2 — O que avancei nesta semana",
            ("Resposta direta aos 4 pedidos: ampliar venues, revisar método, validar SOTA, reformular tese", 1),
            "Parte 3 — Como a tese está sendo construída agora",
            ("Linguagem clara: cada conceito explicado antes de usar", 1),
            "Parte 4 — Próximos passos",
            ("O que preciso decidir nesta reunião", 1),
        ],
    )

    # ==================== PARTE 1: ONDE PARAMOS ====================
    add_section_divider(prs, "1", "Onde paramos na última reunião", "Recap da última reunião e orientações")

    add_content_slide(
        prs,
        "Como a tese estava na última reunião",
        [
            "A tese era DIAGNÓSTICA — focada em explicar por que o erro existe:",
            "",
            "“O limite atual de 75% F1 em classificação racial sobre o FairFace não é só problema de arquitetura nem só problema de método. Tem um componente fenotípico irredutível: as raças se sobrepõem em tom de pele, especialmente Latinx. A dissertação ia construir a primeira matriz pública dessa sobreposição e quantificar o que dá para reduzir vs o que é estrutural.”",
            "",
            "Corpus naquele momento: 23 fichas catalogadas, 6 tracks temáticos.",
        ],
    )

    add_content_slide(
        prs,
        "O que validamos na última reunião",
        [
            "✓  Concordamos que a evolução metodológica melhorou significativamente",
            "",
            "✓  Validamos que o SOTA encontrado (FaceScanPaliGemma 75.7% F1) é mesmo o atual",
            "",
            "✓  Alinhamos e validamos a linha de pesquisa proposta",
            "",
            "Estas 3 validações são a base sobre a qual estou construindo a nova versão da tese.",
        ],
    )

    add_content_slide(
        prs,
        "Os 4 pedidos que combinamos",
        [
            "1. Revisar o método em profundidade de cada artigo selecionado",
            ("Queremos enxergar não só o resultado, mas o rigor metodológico", 1),
            "",
            "2. Ampliar a pesquisa para venues de ML, Redes Neurais e temas relacionados",
            ("Identificamos lacuna de papers fundadores destes campos", 1),
            "",
            "3. Double-check do SOTA (75.7% F1)",
            ("Garantir que não emergiu competidor recente", 1),
            "",
            "4. Reformular a tese de DIAGNÓSTICA para PRESCRITIVA",
            ("Treinar classifier de tom → usar para condicionar classifier de raça → estender a face recognition", 1),
        ],
    )

    # ==================== PARTE 2: O QUE AVANCEI ESTA SEMANA ====================
    add_section_divider(prs, "2", "O que avancei nesta semana", "Resposta direta aos 4 pedidos")

    add_content_slide(
        prs,
        "Resposta ao pedido 2: ampliar venues ML / Redes Neurais",
        [
            "Executei Rodada 5 de triagem com foco em ML e Redes Neurais.",
            "",
            "Adicionei 6 fichas novas, todas de venues top da área:",
            ("• Hardt, Price & Srebro (NeurIPS 2016) — origem das métricas formais de fairness", 1),
            ("• Perez et al. (AAAI 2018) — FiLM, mecanismo de condicionamento neural", 1),
            ("• Zemel et al. (ICML 2013, Test-of-Time Award 2023) — Fair Representation Learning", 1),
            ("• Madras et al. (ICML 2018) — LAFTR, prova teórica de fair transferência", 1),
            ("• Zhang, Lemoine & Mitchell (AIES 2018) — adversarial debiasing", 1),
            ("• Kleinberg, Mullainathan & Raghavan (ITCS 2017) — teorema da impossibilidade", 1),
            "",
            "Cada um destes é base de algum baseline ou mecanismo do pipeline.",
        ],
    )

    add_content_slide(
        prs,
        "Resposta ao pedido 1: revisar método em profundidade",
        [
            "Criei uma nova Seção 12 normativa em cada ficha: ‘Análise crítica do método’.",
            "",
            "Avalia cada paper em 5 dimensões:",
            ("(a) Rigor formal — matemática defensável? Pressupostos declarados?", 1),
            ("(b) Reprodutibilidade — hiperparâmetros, código, seeds?", 1),
            ("(c) Aplicabilidade ao nosso pipeline — onde funciona e onde falha?", 1),
            ("(d) Design choices — o que foi justificado vs assumido?", 1),
            ("(e) Conexão com o que aprendi na Rodada 5", 1),
            "",
            "Adicionada nas 10 fichas centrais (lista de leitura prioritária).",
            "Vamos expandir para as outras 19 após aprovação conjunta da nova versão da tese.",
        ],
    )

    add_content_slide(
        prs,
        "Resposta ao pedido 3: double-check do SOTA",
        [
            "Executei a Rodada 2.6 com janela expandida fevereiro–junho de 2026.",
            "",
            "Fonte: 12 papers que citam o FaceScanPaliGemma no Semantic Scholar.",
            "",
            "Resultado: NENHUM dos 12 papers usa FairFace race 7-class como métrica principal.",
            ("Citantes trabalham em tarefas diferentes: gender, age, emotion, detection, audit, etc.", 1),
            "",
            "Conclusão: FaceScanPaliGemma 75.7% F1 segue sendo o SOTA único em junho/2026.",
            "",
            "Achados adicionais: validação cruzada confirmou também o baseline ResNet-34 = 72%",
            ("Reportado por AlDahoul (2024/26) E por Lin (FairGRAPE 2022) independentemente", 1),
        ],
    )

    add_content_slide(
        prs,
        "Resposta ao pedido 4: reformular para PRESCRITIVA",
        [
            "Mudança central: deixa de ‘decompor o erro’ e passa a ‘construir um pipeline’ que melhora fairness em race classification E em face recognition, usando tom de pele como sinal auxiliar.",
            "",
            "Pipeline construtivo em 6 etapas:",
            ("1. Classificador de tom de pele (MST) — usar SkinToneNet pré-treinado", 1),
            ("2. Auditar FairFace e publicar a matriz pública MST × raça", 1),
            ("3. Race classifier com tom de pele como contexto (mecanismo FiLM)", 1),
            ("4. Comparar fairness COM vs SEM tom de pele, contra 6 baselines", 1),
            ("5. Aplicar pipeline a face recognition (RFW ou BFW)", 1),
            ("6. Medir melhora em Black/African e decomposição Latinx", 1),
        ],
    )

    add_content_slide(
        prs,
        "Triagem do corpus de pesquisa",
        [
            "Dos 57 papers avaliados, aprovamos 29 como fichas do corpus.",
            "",
            "Aprovação seguiu os critérios de seleção acordados:",
            ("Venue científico forte (top conferences ou journals peer-reviewed)", 1),
            ("Citações acima do threshold do ano de publicação", 1),
            ("Aplicabilidade direta a uma das 14 perguntas de pesquisa", 1),
            ("Aprovação por exceção (cobertura única) com justificativa registrada", 1),
            "",
            "Cada decisão (aprovado / standby / descartado) tem data e justificativa em _triagem.md.",
        ],
    )

    # ==================== PARTE 3: COMO A TESE ESTÁ SENDO CONSTRUÍDA ====================
    add_section_divider(prs, "3", "Como a tese está sendo construída", "Agora prescritiva, com conceitos explicados")

    add_thesis_slide(
        prs,
        "A tese em uma frase clara",
        "Treinar uma rede neural para reconhecer o tom de pele de uma foto, "
        "e usar essa informação como contexto extra ao treinar uma rede de "
        "classificação racial, melhora as métricas de fairness (a paridade "
        "entre raças). Essa melhoria também se transfere para tarefas "
        "downstream de reconhecimento facial, beneficiando especialmente "
        "grupos sub-representados (Black/African).\n\n"
        "Em outras palavras: o tom de pele é informação que o classificador "
        "de raça precisa ter explicitamente — não pode descobrir sozinho.",
    )

    # Conceitos explicados (3 slides) — ANTES do pipeline
    add_explainer_slide(
        prs,
        "O que é a escala MST (Monk Skin Tone)?",
        "Uma escala de 10 tons de pele (do tom 1 mais claro ao tom 10 mais escuro), "
        "criada em 2023 pelo sociólogo Dr. Ellis Monk (Harvard) em parceria com o Google. "
        "Foi desenhada especificamente para auditar fairness em sistemas de IA — não é "
        "a escala antiga de dermatologia (Fitzpatrick), que tem viés para pele clara.",
        "Nossa tese precisa medir tom de pele de forma confiável e granular. MST é o "
        "padrão moderno aceito pela comunidade de fairness research (publicado em "
        "NeurIPS 2023). Substituir a Fitzpatrick pela MST evita 3 problemas conhecidos "
        "da escala antiga.",
        "A Fitzpatrick (1975) tem só 6 tons, sendo 3 deles para variações de 'pele "
        "considerada branca'. Um terço dos próprios dermatologistas confunde Fitzpatrick "
        "com classificação racial — erro categorial endêmico.",
        footer="Classificador MST pré-treinado validado pela Rodada 6 (SkinToneNet, Pereira 2026)",
    )

    _add_film_math_slide(prs)

    _add_laftr_theory_slide(prs)

    # Pipeline com analogia
    add_content_slide(
        prs,
        "O pipeline em 6 etapas",
        [
            "Etapa 1 — Treinar um classificador de tom de pele (MST)",
            ("Dados: MST-E (Schumann 2023) + Casual Conversations (Meta 2021)", 1),
            "Etapa 2 — Auditar o FairFace: como tom de pele se distribui entre as 7 raças?",
            ("Aplicar o classificador MST sobre 10.954 imagens; validar manualmente um subset", 1),
            "Etapa 3 — Treinar o classificador de raça USANDO tom de pele como contexto (via FiLM)",
            ("Backbone: ConvNeXt-T (rede moderna, 28 milhões de parâmetros — leve)", 1),
            "Etapa 4 — Comparar fairness: COM tom de pele vs SEM",
            ("Baselines: ConvNeXt-T puro, FSCL+, Group DRO, Adversarial debiasing", 1),
            "Etapa 5 — Aplicar pipeline idêntico em reconhecimento facial (RFW ou BFW)",
            ("Verificar transferência da propriedade fair", 1),
            "Etapa 6 — Medir se Black/African melhora especificamente",
            ("Métrica: TAR @ FAR fixo por raça", 1),
        ],
        footer="Etapas 1, 4 e 5 reforçadas pela Rodada 6 (SkinToneNet, Adversarial debiasing, fair transfer empírico)",
    )

    add_content_slide(
        prs,
        "As métricas que vamos reportar — todas públicas e estabelecidas",
        [
            "F1 macro — média das pontuações nas 7 raças, tratando todas igualmente",
            ("Referência: van Rijsbergen 1979 (livro Information Retrieval). Implementação: sklearn.metrics.f1_score(average='macro').", 1),
            ("Quanto maior, melhor. SOTA atual = 75% (FaceScanPaliGemma).", 1),
            "",
            "DR (Disparity Ratio) — razão entre a melhor e a pior raça",
            ("Referência: Hardt, Price & Srebro 2016 (NeurIPS) — Equal Opportunity / Equalized Odds.", 1),
            ("Quanto mais perto de 1.0, mais justo. Latinx vs Black atualmente = 60% / 90% = 0.67.", 1),
            "",
            "Worst-class F1 — pontuação na raça em que o modelo se sai pior",
            ("Referência: Sagawa et al. 2020 (ICLR) — Group DRO. Métrica = min_g F1(g).", 1),
            ("Atualmente = 60% (Latinx). Quanto maior, melhor o pior caso.", 1),
            "",
            "Por que 3 métricas e não uma?",
            ("Teorema da impossibilidade (Kleinberg, Mullainathan & Raghavan 2017, ITCS): não existe métrica única de fairness — reportar as 3 é forma honesta de comunicar trade-offs.", 1),
        ],
        footer="Todas as 3 métricas + teorema da impossibilidade já têm ficha catalogada no corpus",
    )

    add_table_slide(
        prs,
        "As 5 hipóteses que vamos testar (na forma de perguntas)",
        ["#", "Pergunta", "Como saberemos se SIM"],
        [
            ["H1", "O pipeline com tom de pele melhora fairness em classificação?", "F1 sobe ≥+2pp E DR cai ≥20% vs ResNet-34"],
            ["H2", "Trocar a rede (ResNet-34 → ConvNeXt-T) por si só ajuda Latinx?", "Latinx permanece em ≈60% (não muda)"],
            ["H3", "Latinx tem tom de pele espalhado em várias categorias?", "Spread cobre ≥5 das 10 categorias MST"],
            ["H4", "A maior parte dos erros do Latinx é por sobreposição de tom?", "≥50% dos erros estão em zonas de overlap"],
            ["H5", "A melhoria transfere para reconhecimento facial?", "Black/African melhora ≥+3pp em RFW ou BFW"],
        ],
        footer="H5 em revisão após Pangelinan 2023 (Rodada 6): pixel info pode ser confounder — discussão na pauta",
    )

    # ==================== PARTE 4: PRÓXIMOS PASSOS ====================
    add_section_divider(prs, "4", "Próximos passos", "Plano experimental e calendário")

    add_table_slide(
        prs,
        "Cronograma estimado",
        ["Fase", "Duração", "O que será entregue"],
        [
            ["Aprovação conjunta da nova versão da tese", "Esta reunião", "Ajustes na tese se necessário"],
            ["Submissão para qualificação", "+2 semanas após aprovação", "Documento de qualificação ao programa"],
            ["Setup metodológico", "2 semanas", "Documentos com especificações detalhadas"],
            ["Capítulo 1 — Classificador MST + matriz", "4 semanas", "Resultado de H3"],
            ["Capítulo 2 — Race + condicionamento", "10–12 semanas", "Resultados de H1, H2, H4"],
            ["Capítulo 3 — Face recognition", "6 semanas", "Resultado de H5"],
            ["Síntese final", "4 semanas", "Decomposição do erro Latinx"],
            ["Escrita dos capítulos", "8–12 semanas (paralelo)", "Texto final da dissertação"],
            ["Defesa prevista", "~ Janeiro–Março de 2027", "Total: ~28–32 semanas ativas"],
        ],
    )

    add_content_slide(
        prs,
        "Os 3 capítulos experimentais",
        [
            "Capítulo 1 — Construir o classificador de tom",
            ("Treinar sobre MST-E + Casual Conversations", 1),
            ("Aplicar no FairFace e construir a matriz pública tom × raça", 1),
            ("Testa H3 (Latinx tem spread amplo)", 1),
            "",
            "Capítulo 2 — Classificador de raça com tom como contexto",
            ("Pipeline ConvNeXt-T + FiLM, comparado com 4 baselines independentes", 1),
            ("3 seeds para significância estatística", 1),
            ("Testa H1 (pipeline funciona), H2 (Latinx é estrutural), H4 (overlap explica erros)", 1),
            "",
            "Capítulo 3 — Aplicar a reconhecimento facial",
            ("Mesmo pipeline em RFW ou BFW (datasets que já estão catalogados)", 1),
            ("Foco em accuracy de Black/African", 1),
            ("Testa H5 (fair transferência funciona)", 1),
        ],
    )

    add_content_slide(
        prs,
        "O que precisamos decidir nesta reunião",
        [
            "1. Aprovação da nova versão da tese (prescritiva, pipeline integrado)?",
            ("Se aprovarmos, seguimos para o detalhamento metodológico.", 1),
            ("Se houver ajustes, fazemos e revisamos juntos antes de prosseguir.", 1),
            "",
            "2. Alguma das 5 hipóteses precisa ser reformulada?",
            ("São o esqueleto da dissertação — se uma estiver mal formulada, melhor descobrir agora.", 1),
            "",
            "3. Escolha definitiva entre RFW e BFW para o Capítulo 3?",
            ("Temos preferência por RFW (mais escala, mais histórico) — precisamos validar essa escolha juntos.", 1),
            "",
            "4. Alguma nova frente de literatura que ficou faltando?",
            ("Podemos fazer Rodada 6 se necessário, mas o corpus já está consistente.", 1),
        ],
    )

    # ==================== SLIDE FINAL ====================
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11.3), Inches(2.5))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Discussão e feedback"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Aprovação da nova versão da tese — decidir os 4 itens acima"
    p2.font.size = Pt(22)
    p2.font.color.rgb = GRAY_LT
    p2.alignment = PP_ALIGN.CENTER

    # ==================== SLIDES DE APOIO ====================
    add_section_divider(prs, "★", "Slides de apoio", "Material complementar para consulta durante a discussão")

    _add_perguntas_slide(prs)

    _add_paper_card_slide(
        prs,
        "Pangelinan et al. (2023) — Causas de variação demográfica em FR",
        "arXiv:2304.07175 — autores Notre Dame / Florida Tech",
        "Analisa POR QUE a accuracy de face recognition varia entre grupos demográficos. Conclui que diferenças de pixel info da face nas imagens de teste explicam mais variação do que tom de pele direto.",
        "Refutação potencial de H5. Para FR (verificação 1:1), pixel info pode ser causa primária. Motivou reformulação de H5 em discussão na pauta de hoje.",
        accent=ACCENT,
    )

    _add_paper_card_slide(
        prs,
        "Pereira et al. (2026) — SkinToneNet + dataset STW",
        "arXiv:2603.02475 — autores ICMC-USP / IMPA",
        "Propõe classificador MST (ViT-Small fine-tuned) treinado em dataset STW (42 mil imagens, 3,5 mil indivíduos, MST 10 classes). Audita FairFace e reporta ausência sistêmica de MST 6-10. NÃO publica matriz cruzada MST × raça.",
        "Insumo direto do nosso pipeline: usar SkinToneNet pré-treinado em vez de treinar do zero. Nossa contribuição C2 (matriz cruzada) segue original. Economiza ~3 semanas de cronograma.",
        accent=GREEN,
    )

    _add_paper_card_slide(
        prs,
        "Dooley et al. (2022) — Fairer architectures make for fairer FR",
        "arXiv:2210.09943 — NAS search para FR fairness",
        "Argumenta que biases são inerentes a arquiteturas neurais. Faz busca de arquitetura (NAS) jointly com hiperparâmetros e produz modelos que Pareto-dominam baselines de mitigação tradicionais.",
        "Reforça nossa H2: trocar a arquitetura por si só importa. Valida escolha de ConvNeXt-T como backbone moderno. Se H1 falhar mas ConvNeXt-T puro já reduzir disparity, parte do efeito é arquitetural.",
        accent=NAVY,
    )

    _add_paper_card_slide(
        prs,
        "Aguirre & Dredze (2023) — Transferring fairness via multi-task",
        "arXiv:2305.12671 — Johns Hopkins",
        "Adapta uma fairness loss single-task para um framework multi-task e demonstra empiricamente que objetivos de fairness demográfico SE TRANSFEREM entre tarefas que compartilham representação. Domínio dos experimentos é NLP.",
        "Reforço empírico do princípio teórico do LAFTR (Madras 2018). Fortalece as etapas 3 e 5 do pipeline (race classifier condicionado e extensão para FR).",
        accent=NAVY,
    )

    _add_paper_card_slide(
        prs,
        "Kolla & Savadamuthu (2022) — Impact of racial distribution in training",
        "arXiv:2211.14498 — WACVW 2023 (IEEE/CVF)",
        "Investiga o efeito da distribuição racial nos dados de treino sobre disparities de face recognition. Conclui que distribuição uniforme de raças NO TREINO sozinha não garante FR sem viés.",
        "Reforça necessidade de intervenção arquitetural além de balanceamento de dados. FairFace já é balanceado e o gap Latinx persiste — coerente com a tese de que precisamos do FiLM-conditioning.",
        accent=NAVY,
    )

    _add_paper_card_slide(
        prs,
        "Liu et al. (2025) — BNMR Bayesian meta-learning",
        "arXiv:2505.01699 — ACM FAccT 2025",
        "Propõe BNMR (Bayesian Network-informed Meta Reweighting): uma rede bayesiana calibra meta-learning de reweighting de amostras durante treino, tracking de viés do modelo em tempo real.",
        "Baseline competitivo recente. Mecanismo é ortogonal ao nosso (sample reweighting vs FiLM-conditioning). Candidato a baseline forte do Cap 2 para comparação justa.",
        accent=NAVY,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))


def _add_film_math_slide(prs: Presentation) -> None:
    """Slide do FiLM com formulação matemática + receita prática."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = "O que é FiLM (Feature-wise Linear Modulation)?"
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = NAVY; line.line.fill.background()

    # Caixa 1 — Formulação matemática
    box1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.5), Inches(12.1), Inches(2.3))
    box1.fill.solid(); box1.fill.fore_color.rgb = GRAY_LT; box1.line.fill.background()
    tf1 = box1.text_frame; tf1.word_wrap = True
    tf1.margin_left = Inches(0.3); tf1.margin_right = Inches(0.3); tf1.margin_top = Inches(0.15)
    p1 = tf1.paragraphs[0]; p1.text = "Formulação matemática (Perez et al. 2018, AAAI)"
    p1.font.size = Pt(14); p1.font.bold = True; p1.font.color.rgb = NAVY
    p1b = tf1.add_paragraph()
    p1b.text = "Dada uma feature map intermediária F do backbone e um vetor de contexto z (no nosso caso, vetor MST de 10 dimensões saído do SkinToneNet):"
    p1b.font.size = Pt(15); p1b.font.color.rgb = GRAY_DK
    p1c = tf1.add_paragraph()
    p1c.text = "    FiLM(F | γ, β) = γ ⊙ F + β       onde γ = fγ(z) e β = fβ(z)"
    p1c.font.size = Pt(17); p1c.font.color.rgb = NAVY; p1c.font.bold = True
    p1d = tf1.add_paragraph()
    p1d.text = "fγ e fβ são duas pequenas MLPs treináveis que mapeiam o contexto z para os parâmetros de modulação (γ, β) por canal. O símbolo ⊙ é multiplicação elemento-a-elemento."
    p1d.font.size = Pt(14); p1d.font.color.rgb = GRAY_DK

    # Caixa 2 — Como aplicar na prática
    box2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.0), Inches(12.1), Inches(2.3))
    box2.fill.solid(); box2.fill.fore_color.rgb = WHITE
    box2.line.color.rgb = NAVY; box2.line.width = Pt(1.5)
    tf2 = box2.text_frame; tf2.word_wrap = True
    tf2.margin_left = Inches(0.3); tf2.margin_right = Inches(0.3); tf2.margin_top = Inches(0.15)
    p2 = tf2.paragraphs[0]; p2.text = "Como aplicar no nosso pipeline (receita prática)"
    p2.font.size = Pt(14); p2.font.bold = True; p2.font.color.rgb = NAVY
    receita = [
        "1. Extrair vetor MST z ∈ ℝ¹⁰ da imagem usando o SkinToneNet pré-treinado.",
        "2. Para cada bloco residual do ConvNeXt-T, inserir um bloco FiLM que recebe z e produz (γ, β).",
        "3. Modular as features intermediárias F do bloco: F' = γ ⊙ F + β.",
        "4. Backbone e cabeça de classificação treinam normalmente; só os parâmetros das MLPs (fγ, fβ) são novos.",
    ]
    for ln in receita:
        pln = tf2.add_paragraph()
        pln.text = ln; pln.font.size = Pt(14); pln.font.color.rgb = GRAY_DK
        pln.space_after = Pt(2)

    # Caixa 3 — Por que importa
    box3 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(6.5), Inches(12.1), Inches(0.9))
    box3.fill.solid(); box3.fill.fore_color.rgb = WHITE
    box3.line.color.rgb = GREEN; box3.line.width = Pt(1.5)
    tf3 = box3.text_frame; tf3.word_wrap = True
    tf3.margin_left = Inches(0.3); tf3.margin_right = Inches(0.3); tf3.margin_top = Inches(0.1)
    p3 = tf3.paragraphs[0]; p3.text = "Por que importa para a tese"
    p3.font.size = Pt(13); p3.font.bold = True; p3.font.color.rgb = GREEN
    p3b = tf3.add_paragraph()
    p3b.text = "Sem FiLM, 'usar tom de pele como contexto' seria vaga. Com FiLM, é matemática e código concreto — qualquer revisor consegue reproduzir."
    p3b.font.size = Pt(13); p3b.font.color.rgb = GRAY_DK


def _add_laftr_theory_slide(prs: Presentation) -> None:
    """Slide do LAFTR com formalização teórica completa."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = "O que é fair transferência (LAFTR)?"
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = NAVY; line.line.fill.background()

    # Caixa 1 — Formalização teórica
    box1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.5), Inches(12.1), Inches(2.4))
    box1.fill.solid(); box1.fill.fore_color.rgb = GRAY_LT; box1.line.fill.background()
    tf1 = box1.text_frame; tf1.word_wrap = True
    tf1.margin_left = Inches(0.3); tf1.margin_right = Inches(0.3); tf1.margin_top = Inches(0.15)
    p1 = tf1.paragraphs[0]; p1.text = "Formalização teórica (Madras et al. 2018, ICML)"
    p1.font.size = Pt(14); p1.font.bold = True; p1.font.color.rgb = NAVY
    teorico = [
        "Encoder e: X → Z      mapeia entrada para representação latente Z",
        "Classificador c: Z → Y      prediz a tarefa de interesse",
        "Adversário a: Z → S      tenta prever atributo sensível S a partir de Z",
        "Objetivo:    min_{e,c} max_a  [ L_clf(c(e(X)), Y)  +  λ · L_adv(a(e(X)), S) ]",
        "Resultado teórico: se a falha (L_adv → 0), então Z é fair sob demographic parity / equalized odds.",
    ]
    for ln in teorico:
        pln = tf1.add_paragraph()
        pln.text = ln
        pln.font.size = Pt(13); pln.font.color.rgb = GRAY_DK
        if "Objetivo" in ln or "Resultado" in ln:
            pln.font.bold = True; pln.font.color.rgb = NAVY
        pln.space_after = Pt(2)

    # Caixa 2 — Garantia de transferência
    box2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.05), Inches(12.1), Inches(1.9))
    box2.fill.solid(); box2.fill.fore_color.rgb = WHITE
    box2.line.color.rgb = NAVY; box2.line.width = Pt(1.5)
    tf2 = box2.text_frame; tf2.word_wrap = True
    tf2.margin_left = Inches(0.3); tf2.margin_right = Inches(0.3); tf2.margin_top = Inches(0.15)
    p2 = tf2.paragraphs[0]; p2.text = "Garantia de transferência (Teorema 1, Madras 2018)"
    p2.font.size = Pt(14); p2.font.bold = True; p2.font.color.rgb = NAVY
    p2b = tf2.add_paragraph()
    p2b.text = "Para QUALQUER classificador downstream c' treinado sobre a mesma representação Z, a violação de fairness de c' é limitada superiormente pela violação de fairness em c. Em palavras simples: a propriedade fair é PROPRIEDADE DA REPRESENTAÇÃO Z, não da tarefa específica."
    p2b.font.size = Pt(14); p2b.font.color.rgb = GRAY_DK
    p2c = tf2.add_paragraph()
    p2c.text = "Implicação: ao treinar fair em race classification (Cap 2), o backbone passa a Z fair. Reaproveitamos esse mesmo Z em face recognition (Cap 3) sem re-treinar fairness do zero."
    p2c.font.size = Pt(13); p2c.font.color.rgb = GRAY_DK

    # Caixa 3 — Evidência empírica
    box3 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(6.1), Inches(12.1), Inches(1.3))
    box3.fill.solid(); box3.fill.fore_color.rgb = WHITE
    box3.line.color.rgb = GREEN; box3.line.width = Pt(1.5)
    tf3 = box3.text_frame; tf3.word_wrap = True
    tf3.margin_left = Inches(0.3); tf3.margin_right = Inches(0.3); tf3.margin_top = Inches(0.1)
    p3 = tf3.paragraphs[0]; p3.text = "Evidência empírica complementar"
    p3.font.size = Pt(13); p3.font.bold = True; p3.font.color.rgb = GREEN
    p3b = tf3.add_paragraph()
    p3b.text = "Aguirre & Dredze 2023 (arXiv:2305.12671) — confirmaram empiricamente que objetivos de fairness demográfico SE TRANSFEREM entre tarefas que compartilham representação (domínio NLP, mas princípio é independente da modalidade)."
    p3b.font.size = Pt(13); p3b.font.color.rgb = GRAY_DK

    tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.5), Inches(0.4))
    pf = tx_f.text_frame.paragraphs[0]
    pf.text = "H5 em revisão após Pangelinan 2023 (Rodada 6): pixel info pode ser confounder adicional em FR — discussão na pauta"
    pf.font.size = Pt(13); pf.font.color.rgb = GRAY_MD; pf.font.italic = True


def _add_perguntas_slide(prs: Presentation) -> None:
    """Slide das 14 perguntas de pesquisa respondidas (Q01-Q14)."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = "As 14 perguntas de pesquisa respondidas"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = NAVY; line.line.fill.background()

    perguntas = [
        ("Q01", "Existem melhores datasets para corrigir fairness em biometria facial de raças?"),
        ("Q02", "Qual o dataset mais utilizado nos estudos de fairness?"),
        ("Q03", "Qual o split oficial do FairFace e por que diferentes papers usam splits distintos?"),
        ("Q04", "Quais técnicas de mitigação algorítmica já foram testadas em FairFace race 7-class?"),
        ("Q05", "Existe consenso sobre métrica de fairness para classificação multi-classe?"),
        ("Q06", "O baseline ResNet-34 = 72% acurácia é teto da arquitetura ou da metodologia?"),
        ("Q07", "Existe pesquisa de fairness em biometria facial sem usar FairFace?"),
        ("Q08", "Por que os estudos tentam sempre fazer merge de classes raciais?"),
        ("Q09", "7 classes é realmente a taxonomia correta para fairness racial facial?"),
        ("Q10", "Existe matriz associativa Fitzpatrick/MST × FairFace 7-race?"),
        ("Q11", "As 7 categorias raciais do FairFace têm fundamento biológico ou socio-político?"),
        ("Q12", "Como antropologia forense moderna trata 'raça'?"),
        ("Q13", "Origem e propósito da escala Fitzpatrick — para que foi criada?"),
        ("Q14", "Quantos tons de pele existem cientificamente?"),
    ]

    # Duas colunas
    col1 = perguntas[:7]
    col2 = perguntas[7:]

    tx_c1 = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(6.1), Inches(5.5))
    tf1 = tx_c1.text_frame; tf1.word_wrap = True

    for i, (code, txt) in enumerate(col1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        run1 = p.add_run() if i == 0 and not p.text else None
        if i == 0:
            p.text = ""
        r_code = p.add_run()
        r_code.text = f"{code}  "
        r_code.font.bold = True
        r_code.font.color.rgb = NAVY
        r_code.font.size = Pt(13)
        r_txt = p.add_run()
        r_txt.text = txt
        r_txt.font.color.rgb = GRAY_DK
        r_txt.font.size = Pt(13)
        p.space_after = Pt(6)

    tx_c2 = slide.shapes.add_textbox(Inches(6.85), Inches(1.4), Inches(6.1), Inches(5.5))
    tf2 = tx_c2.text_frame; tf2.word_wrap = True

    for i, (code, txt) in enumerate(col2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        if i == 0:
            p.text = ""
        r_code = p.add_run()
        r_code.text = f"{code}  "
        r_code.font.bold = True
        r_code.font.color.rgb = NAVY
        r_code.font.size = Pt(13)
        r_txt = p.add_run()
        r_txt.text = txt
        r_txt.font.color.rgb = GRAY_DK
        r_txt.font.size = Pt(13)
        p.space_after = Pt(6)

    tx_f = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.5), Inches(0.4))
    pf = tx_f.text_frame.paragraphs[0]
    pf.text = "Detalhamento completo em docs/ativo/04_pesquisa_bibliografica/_perguntas.md"
    pf.font.size = Pt(13); pf.font.color.rgb = GRAY_MD; pf.font.italic = True


def _add_paper_card_slide(prs: Presentation, title: str, citation: str, summary: str, impact: str, accent: RGBColor) -> None:
    """Slide compacto de resumo de um paper: o que defende, por que entrou, impacto na tese."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # Título
    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    p = tx_t.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(24); p.font.bold = True; p.font.color.rgb = NAVY

    # Citação
    tx_c = slide.shapes.add_textbox(Inches(0.5), Inches(1.15), Inches(12.5), Inches(0.4))
    pc = tx_c.text_frame.paragraphs[0]
    pc.text = citation
    pc.font.size = Pt(14); pc.font.color.rgb = GRAY_MD; pc.font.italic = True

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.6), Inches(12.5), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = accent; line.line.fill.background()

    # O que defende
    box1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(2.0), Inches(12.1), Inches(2.3))
    box1.fill.solid(); box1.fill.fore_color.rgb = GRAY_LT
    box1.line.fill.background()
    tf1 = box1.text_frame; tf1.word_wrap = True
    tf1.margin_left = Inches(0.3); tf1.margin_right = Inches(0.3); tf1.margin_top = Inches(0.15)
    p1 = tf1.paragraphs[0]; p1.text = "O que defende"
    p1.font.size = Pt(14); p1.font.bold = True; p1.font.color.rgb = NAVY
    p1b = tf1.add_paragraph(); p1b.text = summary
    p1b.font.size = Pt(16); p1b.font.color.rgb = GRAY_DK

    # Impacto na tese
    box2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.5), Inches(12.1), Inches(2.4))
    box2.fill.solid(); box2.fill.fore_color.rgb = WHITE
    box2.line.color.rgb = accent; box2.line.width = Pt(1.5)
    tf2 = box2.text_frame; tf2.word_wrap = True
    tf2.margin_left = Inches(0.3); tf2.margin_right = Inches(0.3); tf2.margin_top = Inches(0.15)
    p2 = tf2.paragraphs[0]; p2.text = "Impacto na nossa tese"
    p2.font.size = Pt(14); p2.font.bold = True; p2.font.color.rgb = accent
    p2b = tf2.add_paragraph(); p2b.text = impact
    p2b.font.size = Pt(16); p2b.font.color.rgb = GRAY_DK


def main() -> None:
    out = Path(__file__).resolve().parent / "material_reuniao_orientador_v3.2.1.pptx"
    build_presentation(out)
    print(f"Gerado: {out}")
    print(f"Tamanho: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
