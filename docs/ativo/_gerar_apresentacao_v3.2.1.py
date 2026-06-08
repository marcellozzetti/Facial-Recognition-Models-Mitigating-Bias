"""Gera apresentação PowerPoint v3.2.1 — narrativa pedagógica.

Estrutura em 4 partes:
1. Onde paramos (recap da última reunião)
2. O que avancei nesta semana (resposta aos 4 pedidos do orientador)
3. Como a tese está sendo construída (v3.2 com conceitos explicados)
4. Próximos passos

Princípios:
- Linguagem narrativa, não tabular dense
- Cada termo técnico explicado ANTES do uso
- Voz direta ao orientador ("você validou", "você pediu")
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


def add_explainer_slide(prs: Presentation, title: str, simple_def: str, why_matters: str, concrete: str = "") -> None:
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


def add_thesis_slide(prs: Presentation, title: str, statement: str) -> None:
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
            "“O limite atual de 75% F1 em classificação racial sobre o FairFace não é só",
            "problema de arquitetura nem só problema de método. Tem um componente",
            "fenotípico irredutível: as raças se sobrepõem em tom de pele, especialmente",
            "Latinx. A dissertação ia construir a primeira matriz pública dessa",
            "sobreposição e quantificar o que dá para reduzir vs o que é estrutural.”",
            "",
            "Corpus naquele momento: 23 fichas catalogadas, 6 tracks temáticos.",
        ],
    )

    add_content_slide(
        prs,
        "O que você validou na reunião",
        [
            "✓  Concordou que a evolução metodológica melhorou significativamente",
            "",
            "✓  Validou que o SOTA encontrado (FaceScanPaliGemma 75.7% F1) é mesmo o atual",
            "",
            "✓  Entendeu e validou a linha de pesquisa proposta",
            "",
            "Estas 3 validações são a base sobre a qual estou construindo a nova versão da tese.",
        ],
    )

    add_content_slide(
        prs,
        "Os 4 pedidos que você fez",
        [
            "1. Revisar o método em profundidade de cada artigo selecionado",
            ("Quer enxergar não só o resultado, mas o rigor metodológico", 1),
            "",
            "2. Ampliar a pesquisa para venues de ML, Redes Neurais e temas relacionados",
            ("Sentiu falta de papers fundadores destes campos", 1),
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
            "Vou expandir para as outras 19 após sua aprovação da nova versão da tese.",
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
            "Bônus: validação cruzada confirmou também o baseline ResNet-34 = 72%",
            ("Reportado por AlDahoul (2024/26) E por Lin (FairGRAPE 2022) independentemente", 1),
        ],
    )

    add_content_slide(
        prs,
        "Resposta ao pedido 4: reformular para PRESCRITIVA",
        [
            "Nova versão da tese escrita e disponível.",
            "",
            "Mudança central: deixa de ser ‘decompor o erro’ e passa a ser",
            "‘construir um pipeline que melhora fairness em race classification",
            "E em face recognition, usando tom de pele como sinal auxiliar’.",
            "",
            "Q04 (mitigação) e Q10 (matriz tom × raça) deixam de ser capítulos",
            "paralelos e viram pipeline UNIFICADO.",
            "",
            "Vou detalhar a nova tese na Parte 3 desta apresentação.",
        ],
    )

    add_content_slide(
        prs,
        "Big numbers da pesquisa hoje",
        [
            "✓  29 fichas catalogadas no corpus (era 23 na semana passada)",
            "",
            "✓  ~57 papers avaliados ao todo (29 aprovados + 6 rejeitados explicitamente + outros descartados)",
            "",
            "✓  7 tracks temáticos cobertos (era 6; Track G NOVO: Mecanismos ML / Redes Neurais)",
            "",
            "✓  14 perguntas de pesquisa formalmente respondidas (Q01–Q14)",
            "",
            "✓  Cobertura temporal: 1972 a 2026 (54 anos de literatura, mediana 2020–2021)",
            "",
            "✓  Autoria verificada em fonte primária: 29 de 29 (100%)",
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

    add_table_slide(
        prs,
        "Comparação: antes (semana passada) vs agora",
        ["Dimensão", "Antes (diagnóstica)", "Agora (prescritiva)"],
        [
            ["Postura", "Explicar por que o erro existe", "Construir um pipeline que reduz o erro"],
            ["Saída prática", "Análise post-hoc", "Modelo treinado deployável"],
            ["Tarefa", "Apenas classification", "Classification + Face recognition"],
            ["Q04 e Q10", "Capítulos paralelos", "Pipeline unificado"],
            ["Critério de sucesso", "Decomposição irredutível vs redutível", "Melhora mensurável em métricas concretas"],
        ],
        footer="Aprovação da nova versão é o item principal desta reunião",
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
    )

    add_explainer_slide(
        prs,
        "O que é FiLM (Feature-wise Linear Modulation)?",
        "Uma técnica de redes neurais (Perez et al., AAAI 2018) que permite uma rede "
        "‘consultar’ uma fonte de contexto extra ao tomar decisões. Tecnicamente: ela "
        "modula as features intermediárias por uma transformação aprendida a partir do "
        "contexto. Não muda a arquitetura — é uma camada que se adiciona.",
        "É o mecanismo formal que torna nossa proposta operacional. Sem FiLM, a ideia "
        "'usar tom de pele como contexto' seria vaga. Com FiLM, é matemática e código "
        "concreto. Custo computacional: menos de 1% sobre o backbone.",
        "Analogia: imagine um médico fazendo diagnóstico. Antes de decidir, ele consulta "
        "o exame complementar (ECG, raio-X). FiLM faz isso para a rede de raça: antes de "
        "decidir, ela 'consulta' a rede de tom de pele.",
    )

    add_explainer_slide(
        prs,
        "O que é fair transferência (LAFTR)?",
        "Madras et al. (ICML 2018) provaram teoricamente e mostraram empiricamente que, "
        "se você treina uma rede para ser justa em uma tarefa A, essa propriedade de ser "
        "justa é HERDADA por outras tarefas B que usem a mesma representação como ponto "
        "de partida.",
        "É o que sustenta a EXTENSÃO da nossa tese para face recognition. Se treinarmos "
        "fairness na classificação racial (Cap 2), a propriedade fair se transfere para "
        "reconhecimento facial (Cap 3) — não precisa re-treinar do zero. Esta é a "
        "demonstração empírica que você pediu.",
        "Analogia: aprender ética profissional em medicina geral te torna mais ético "
        "também quando você se especializa em cardiologia. A propriedade 'ética' não "
        "depende da especialidade — está embutida no profissional.",
    )

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
    )

    add_content_slide(
        prs,
        "As métricas que vamos reportar — explicação simples",
        [
            "F1 macro — média das pontuações em todas as 7 raças, tratando todas igualmente.",
            ("Quanto maior, melhor. SOTA atual = 75% (FaceScanPaliGemma).", 1),
            "",
            "DR (Disparity Ratio) — razão entre a melhor e a pior raça.",
            ("Quanto mais perto de 1.0, mais justo. Latinx vs Black hoje = 60% / 90% = 0.67.", 1),
            "",
            "Worst-class F1 — pontuação na raça em que o modelo se sai pior.",
            ("Hoje = 60% (Latinx). Quanto maior, melhor o pior caso.", 1),
            "",
            "Por que 3 métricas e não uma?",
            ("Teorema da impossibilidade (Kleinberg et al. 2017): não existe métrica única", 1),
            ("de fairness. Reportar 3 simultaneamente é a forma honesta de comunicar trade-offs.", 1),
        ],
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
        footer="H1, H4 e H5 são as hipóteses centrais. Plano B documentado caso refutadas.",
    )

    # ==================== PARTE 4: PRÓXIMOS PASSOS ====================
    add_section_divider(prs, "4", "Próximos passos", "Plano experimental e calendário")

    add_content_slide(
        prs,
        "Os 3 capítulos experimentais",
        [
            "Capítulo 1 (~4 semanas) — Construir o classificador de tom",
            ("Treinar sobre MST-E + Casual Conversations", 1),
            ("Aplicar no FairFace e construir a matriz pública tom × raça", 1),
            ("Testa H3 (Latinx tem spread amplo)", 1),
            "",
            "Capítulo 2 (~10-12 semanas) — Classificador de raça com tom como contexto",
            ("Pipeline ConvNeXt-T + FiLM, comparado com 4 baselines independentes", 1),
            ("3 seeds para significância estatística", 1),
            ("Testa H1 (pipeline funciona), H2 (Latinx é estrutural), H4 (overlap explica erros)", 1),
            "",
            "Capítulo 3 (~6 semanas) — Aplicar a reconhecimento facial",
            ("Mesmo pipeline em RFW ou BFW (datasets que já estão catalogados)", 1),
            ("Foco em accuracy de Black/African", 1),
            ("Testa H5 (fair transferência funciona)", 1),
        ],
    )

    add_table_slide(
        prs,
        "Cronograma estimado",
        ["Fase", "Duração", "O que será entregue"],
        [
            ["Aprovação da nova versão da tese com você", "Esta reunião", "Ajustes na tese se necessário"],
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
        "Riscos identificados e como mitigá-los",
        [
            "Risco 1 — Alguma hipótese ser refutada",
            ("Plano B documentado. Observação negativa é resultado científico válido.", 1),
            "",
            "Risco 2 — Recrutar anotadores diversos para validação manual MST",
            ("Vou usar Prolific Academic com filtros regionais. Custo estimado: $1400 para 700 imagens × 3 anotadores.", 1),
            "",
            "Risco 3 — Adaptação multi-classe das técnicas baseline",
            ("Código open-source disponível para todos: FSCL+, Group DRO, FineFACE, Adversarial.", 1),
            "",
            "Risco 4 — Compute",
            ("ConvNeXt-T é leve (28M parâmetros). Estimativa ~200–400 GPU-horas para Cap 2 completo.", 1),
        ],
    )

    add_content_slide(
        prs,
        "O que preciso decidir nesta reunião",
        [
            "1. Aprovação da nova versão da tese (prescritiva, pipeline integrado)?",
            ("Se aprovar, sigo para o detalhamento metodológico.", 1),
            ("Se quiser ajustes, faço e te mostro antes de prosseguir.", 1),
            "",
            "2. Alguma das 5 hipóteses precisa ser reformulada?",
            ("São o esqueleto da dissertação — se uma estiver mal formulada, melhor descobrir agora.", 1),
            "",
            "3. Escolha definitiva entre RFW e BFW para o Capítulo 3?",
            ("Tenho preferência por RFW (mais escala, mais histórico) — mas quero sua opinião.", 1),
            "",
            "4. Alguma nova frente de literatura que ficou faltando?",
            ("Posso fazer Rodada 6 se necessário, mas o corpus já está consistente.", 1),
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))


def main() -> None:
    out = Path(__file__).resolve().parent / "material_reuniao_orientador_v3.2.1.pptx"
    build_presentation(out)
    print(f"Gerado: {out}")
    print(f"Tamanho: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
