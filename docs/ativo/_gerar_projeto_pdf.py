"""Gera PDF formal do projeto de pesquisa para submissao com a
Declaracao de Responsabilidade (Art. 8 da Res. 200/2021/CONSU Unifesp).

Uso:
    python _gerar_projeto_pdf.py
    -> produz: docs/ativo/projeto_declaracao_responsabilidade.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#1F2A4E")
GRAY_DK = colors.HexColor("#3D424E")
GRAY_MD = colors.HexColor("#707682")
GRAY_LT = colors.HexColor("#E8EAED")


def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cabecalho": ParagraphStyle(
            "cabecalho",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=GRAY_MD,
            alignment=TA_CENTER,
            leading=11,
        ),
        "titulo": ParagraphStyle(
            "titulo",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=NAVY,
            alignment=TA_CENTER,
            leading=20,
            spaceBefore=6,
            spaceAfter=12,
        ),
        "subtitulo": ParagraphStyle(
            "subtitulo",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10.5,
            textColor=GRAY_DK,
            alignment=TA_CENTER,
            leading=14,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=NAVY,
            leading=17,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=NAVY,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            textColor=GRAY_DK,
            alignment=TA_JUSTIFY,
            leading=15,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            textColor=GRAY_DK,
            alignment=TA_JUSTIFY,
            leading=15,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=4,
        ),
        "assinatura": ParagraphStyle(
            "assinatura",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            textColor=GRAY_DK,
            alignment=TA_CENTER,
            leading=14,
            spaceAfter=2,
        ),
    }
    return styles


def _p(text, style):
    return Paragraph(text, style)


def _table_kv(rows, col_widths=(4.5 * cm, 12 * cm)):
    tbl_style = TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("TEXTCOLOR", (1, 0), (1, -1), GRAY_DK),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, GRAY_LT),
        ]
    )
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(tbl_style)
    return tbl


def _table_header(headers, rows, col_widths=None):
    data = [headers] + rows
    tbl_style = TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 1), (-1, -1), GRAY_DK),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_LT),
        ]
    )
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(tbl_style)
    return tbl


def build_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Projeto de Pesquisa — Anexo à Declaração de Responsabilidade",
        author="Marcello Ozzetti",
        subject="Projeto de mestrado — enquadramento Art. 8º Res. 200/2021 CONSU Unifesp",
    )
    s = _build_styles()
    story = []

    # Cabeçalho institucional
    story.append(_p("UNIVERSIDADE FEDERAL DE SÃO PAULO — UNIFESP", s["cabecalho"]))
    story.append(_p("Instituto de Ciência e Tecnologia — ICT", s["cabecalho"]))
    story.append(_p("Programa de Pós-Graduação em Ciência da Computação", s["cabecalho"]))
    story.append(Spacer(1, 12))

    # Título
    story.append(
        _p(
            "Projeto de Pesquisa — Anexo à Declaração de Responsabilidade",
            s["titulo"],
        )
    )
    story.append(
        _p(
            "Documento formal do projeto para submissão junto à Declaração de "
            "Responsabilidade prevista no parágrafo único do Art. 8º da "
            "Resolução nº 200/2021/CONSELHO UNIVERSITÁRIO da Universidade "
            "Federal de São Paulo.",
            s["subtitulo"],
        )
    )

    # 1. Identificação
    story.append(_p("1. Identificação", s["h1"]))
    identificacao = [
        ["Título do projeto", "Equidade Racial em Classificação Facial: Pipeline Condicionado por Tom de Pele (Escala Monk Skin Tone) via FiLM sobre o Dataset FairFace"],
        ["Mestrando", "Marcello Ozzetti"],
        ["Orientador / Pesquisador Responsável", "Prof. Dr. Marcos Quiles"],
        ["Programa", "Mestrado em Ciência da Computação"],
        ["Instituição", "Universidade Federal de São Paulo — Instituto de Ciência e Tecnologia (Unifesp / ICT)"],
        ["Linha de pesquisa", "Visão Computacional e Equidade Algorítmica"],
        ["Período previsto", "julho de 2026 a outubro de 2026 (qualificação)"],
    ]
    story.append(_table_kv(identificacao))

    # 2. Resumo
    story.append(_p("2. Resumo", s["h1"]))
    story.append(
        _p(
            "Sistemas modernos de reconhecimento facial apresentam disparidade "
            "demográfica significativa entre grupos raciais, mesmo quando "
            "treinados sobre datasets explicitamente balanceados. No estado da "
            "arte atual, o modelo FaceScanPaliGemma sobre o dataset público "
            "FairFace atinge F1 de 90&nbsp;% para faces classificadas como "
            "Black, mas apenas 60&nbsp;% para faces classificadas como Latinx — "
            "uma diferença de trinta pontos percentuais que persiste através "
            "de múltiplos métodos e arquiteturas. Esta dissertação propõe um "
            "pipeline de classificação racial que incorpora tom de pele "
            "(escala Monk Skin Tone, dez classes) como sinal auxiliar "
            "condicionante via mecanismo arquitetural FiLM (Feature-wise "
            "Linear Modulation), e avalia sua capacidade de reduzir essa "
            "disparidade sem sacrificar acurácia agregada. <b>A pesquisa é "
            "inteiramente computacional, conduzida sobre datasets públicos "
            "secundários, sem coleta primária, intervenção ou qualquer forma "
            "de envolvimento direto ou indireto de seres humanos.</b>",
            s["body"],
        )
    )

    # 3. Justificativa
    story.append(_p("3. Justificativa e problema de pesquisa", s["h1"]))
    story.append(
        _p(
            "O reconhecimento facial deixou de ser tecnologia laboratorial e "
            "passou a integrar a infraestrutura social — desbloqueio de "
            "dispositivos, autenticação bancária, identidade digital, "
            "controle de fronteiras e identificação policial. Em escala "
            "industrial, o National Institute of Standards and Technology "
            "auditou em 2019 cento e oitenta e nove algoritmos de "
            "reconhecimento facial sobre dezoito milhões de imagens, "
            "documentando diferenciais de dez a cem vezes na taxa de falso "
            "positivo entre grupos raciais.",
            s["body"],
        )
    )
    story.append(
        _p(
            "Apesar de mais de uma década de pesquisa em equidade "
            "algorítmica, a disparidade persiste. O baseline ResNet-34 sobre "
            "FairFace, o FaceScanPaliGemma via VLM, e mesmo modelos com "
            "balanceamento explícito de dados continuam apresentando F1 "
            "aproximadamente 60&nbsp;% para a categoria Latinx / Hispanic. "
            "Trabalhos recentes sugerem que essa disparidade não é apenas "
            "artefato algorítmico, mas reflete heterogeneidade fenotípica "
            "intra-categorial que a rotulagem racial monolítica do FairFace "
            "não captura adequadamente.",
            s["body"],
        )
    )
    story.append(
        _p(
            "O problema de pesquisa desta dissertação é, portanto: "
            "<b>condicionar arquiteturalmente um classificador racial pelo "
            "tom de pele reduz a disparidade demográfica sem sacrificar "
            "acurácia agregada, e permite decompor o erro em componentes "
            "fenotípico e algorítmico?</b>",
            s["body"],
        )
    )

    # 4. Objetivos
    story.append(_p("4. Objetivos", s["h1"]))
    story.append(_p("4.1 Objetivo geral", s["h2"]))
    story.append(
        _p(
            "Desenvolver e avaliar um pipeline de classificação racial em "
            "imagens faciais que incorpora tom de pele (escala Monk Skin "
            "Tone) como sinal auxiliar condicionante via mecanismo "
            "arquitetural FiLM, com o propósito de mitigar viés racial "
            "demonstrável no estado da arte atual — particularmente a "
            "disparidade severa entre classes raciais bem representadas e "
            "classes sub-representadas documentada em modelos como "
            "FaceScanPaliGemma sobre o dataset FairFace.",
            s["body"],
        )
    )

    story.append(_p("4.2 Objetivos específicos", s["h2"]))
    oes = [
        "<b>1. Auditoria fenotípica do FairFace</b> — quantificar a distribuição cruzada MST × classes raciais sobre o FairFace via SkinToneNet pré-treinado, entregando a primeira matriz pública dessa distribuição.",
        "<b>2. Avaliação metodológica de classificadores MST</b> — comparar SkinToneNet, Casual Conversations baseline e alternativas disponíveis, com <i>sensitivity analysis</i> a dois ou três classificadores alternativos para mitigar risco de propagação de viés.",
        "<b>3. Race classifier com conditioning arquitetural</b> — treinar ConvNeXt-T <i>fine-tuned</i> em FairFace, com camadas FiLM por estágio recebendo o vetor MST como contexto.",
        "<b>4. Comparação sistemática contra baselines de mitigação</b> — avaliar o pipeline contra seis baselines (ResNet-34, ConvNeXt-T puro, FSCL+, Group DRO, FineFACE, Adversarial debiasing) via triangulação de métricas (DR + <i>worst-class</i> F1 + EO_h/EOD).",
        "<b>5. Fair transferência para face recognition downstream</b> — aplicar o backbone fair em <i>face recognition</i> em RFW ou BFW, com controle explícito de <i>pixel information</i> como <i>confounder</i>.",
        "<b>6. Síntese decompositiva</b> — combinar resultados dos objetivos 1 e 5 para quantificar componente fenotípico (irredutível) versus componente algorítmico (mitigável) do erro Latinx.",
    ]
    for oe in oes:
        story.append(_p(oe, s["bullet"]))

    story.append(PageBreak())

    # 5. Metodologia
    story.append(_p("5. Metodologia", s["h1"]))

    story.append(_p("5.1 Natureza da pesquisa", s["h2"]))
    story.append(
        _p(
            "A pesquisa é <b>puramente computacional</b>. Consiste em "
            "treinamento e avaliação de modelos de aprendizado profundo "
            "sobre datasets de imagens faciais já existentes e publicamente "
            "disponíveis; análise estatística de métricas de desempenho "
            "estratificadas por grupo demográfico; comparação sistemática "
            "entre configurações arquiteturais; e síntese analítica dos "
            "resultados.",
            s["body"],
        )
    )

    story.append(_p("5.2 Datasets utilizados — todos secundários e públicos", s["h2"]))
    datasets_headers = ["Dataset", "Origem", "Licença", "Uso na pesquisa"]
    datasets_rows = [
        ["FairFace", "Kärkkäinen & Joo (2021), UCLA", "CC BY 4.0 (uso acadêmico)", "Dataset principal — 108.501 imagens balanceadas em 7 categorias raciais"],
        ["RFW", "Wang et al. (2019), BUPT", "Research-only (mediante solicitação)", "Avaliação de fair transferência downstream"],
        ["BFW", "Robinson et al. (2020), RIT", "CC BY 4.0", "Avaliação alternativa de fair transferência"],
        ["BUPT-Balancedface", "Wang, Zhang, Deng (2022)", "Research-only", "Sensitivity analysis do backbone"],
        ["STW", "Pereira et al. (2026)", "Research-only", "Base de treino do SkinToneNet (uso apenas como modelo pré-treinado)"],
    ]
    story.append(
        _table_header(
            datasets_headers,
            datasets_rows,
            col_widths=(2.7 * cm, 4.2 * cm, 3.6 * cm, 5.5 * cm),
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        _p(
            "<b>Todas as imagens desses datasets foram coletadas, anotadas "
            "e disponibilizadas por seus autores originais anteriormente ao "
            "início desta pesquisa. Este projeto não coleta, não anota e não "
            "distribui qualquer imagem original.</b>",
            s["body"],
        )
    )

    story.append(_p("5.3 Pipeline metodológico em seis etapas", s["h2"]))
    etapas = [
        "<b>Etapa 1 — Classificador MST</b>: uso do SkinToneNet pré-treinado como insumo, mediante validação de concordância interna em subset estratificado.",
        "<b>Etapa 2 — Auditoria FairFace</b>: aplicação do SkinToneNet sobre o FairFace validation set e construção da matriz pública MST × raça.",
        "<b>Etapa 3 — Race classifier com conditioning</b>: treinamento do ConvNeXt-T em FairFace train, com camadas FiLM recebendo o vetor MST como contexto arquitetural.",
        "<b>Etapa 4 — Comparação contra baselines</b>: avaliação do pipeline contra seis baselines de mitigação, com triangulação métrica.",
        "<b>Etapa 5 — Fair transferência</b>: aplicação do backbone fair em face recognition downstream (RFW ou BFW).",
        "<b>Etapa 6 — Síntese decompositiva</b>: quantificação do componente fenotípico versus algorítmico do erro Latinx.",
    ]
    for e in etapas:
        story.append(_p(e, s["bullet"]))

    story.append(_p("5.4 Rigor experimental", s["h2"]))
    story.append(
        _p(
            "Todos os experimentos serão conduzidos com múltiplas sementes "
            "aleatórias (mínimo três: 42, 1, 2), comparação pareada entre "
            "configurações, intervalos de confiança via <i>bootstrap</i> e "
            "reporte estratificado por raça e por interseção raça × gênero. "
            "As métricas seguirão triangulação (DR + <i>worst-class</i> F1 + "
            "EO_h/EOD) para endereçar o Teorema da Impossibilidade de "
            "Kleinberg.",
            s["body"],
        )
    )

    story.append(_p("5.5 Validação humana — modalidade interna", s["h2"]))
    story.append(
        _p(
            "Para o Objetivo Específico 1, será conduzida validação de "
            "concordância entre o SkinToneNet e anotação manual em subset "
            "estratificado de aproximadamente duzentas a trezentas imagens "
            "do FairFace. <b>Esta validação será realizada exclusivamente "
            "pela equipe acadêmica do projeto</b> — Mestrando e Orientador "
            "(e Coorientador, se designado). <b>Não haverá contratação de "
            "anotadores externos, não haverá uso de plataformas de "
            "crowdsourcing, e não haverá coleta de qualquer dado pessoal de "
            "indivíduos externos à equipe.</b>",
            s["body"],
        )
    )

    story.append(PageBreak())

    # 6. Aspectos éticos
    story.append(
        _p(
            "6. Aspectos éticos — enquadramento no Art. 8º da Resolução 200/2021",
            s["h1"],
        )
    )
    story.append(
        _p(
            "Este projeto <b>não envolve, direta ou indiretamente, seres "
            "humanos como sujeitos de pesquisa</b>, conforme a definição "
            "estabelecida pela Resolução nº 200/2021 do Conselho "
            "Universitário da Universidade Federal de São Paulo e pelas "
            "Resoluções nº 466/2012 e nº 510/2016 do Conselho Nacional de "
            "Saúde. Justifica-se:",
            s["body"],
        )
    )

    story.append(_p("6.1 Ausência de coleta primária", s["h2"]))
    story.append(
        _p(
            "O projeto <b>não coleta</b>, em nenhuma etapa, imagens de "
            "rostos, dados biométricos, dados demográficos, opiniões, "
            "comportamento ou qualquer outra informação de seres humanos "
            "identificáveis ou identificados. Todas as imagens utilizadas "
            "são secundárias, provenientes de datasets construídos e "
            "disponibilizados publicamente por terceiros anteriormente ao "
            "início desta pesquisa.",
            s["body"],
        )
    )

    story.append(_p("6.2 Ausência de intervenção", s["h2"]))
    story.append(
        _p(
            "O projeto <b>não realiza intervenção</b> de qualquer natureza "
            "sobre seres humanos — não há entrevistas, questionários, "
            "experimentos, testes psicológicos, exames físicos, uso de "
            "dispositivos, alteração de ambiente ou qualquer procedimento "
            "que produza efeito sobre pessoas físicas.",
            s["body"],
        )
    )

    story.append(_p("6.3 Ausência de envolvimento indireto via crowdsourcing", s["h2"]))
    story.append(
        _p(
            "Diferentemente de projetos que empregam plataformas de "
            "crowdsourcing (Amazon Mechanical Turk, Prolific, Toloka, entre "
            "outras) para anotação de dados, este projeto <b>não contrata "
            "anotadores externos</b>. A validação manual prevista no "
            "Objetivo Específico 1 será conduzida exclusivamente por "
            "membros da equipe acadêmica do projeto, enquadrando-se como "
            "atividade de pesquisa científica interna e não como "
            "recrutamento de participantes.",
            s["body"],
        )
    )

    story.append(_p("6.4 Ausência de identificação de sujeitos", s["h2"]))
    story.append(
        _p(
            "As análises produzidas nesta pesquisa referem-se a <b>grupos "
            "demográficos agregados</b> (categorias raciais, tons de pele) "
            "e não a indivíduos identificáveis. Nenhum resultado publicado "
            "permitirá identificar pessoas específicas presentes nos "
            "datasets utilizados.",
            s["body"],
        )
    )

    story.append(_p("6.5 Ausência de uso de animais vertebrados vivos", s["h2"]))
    story.append(
        _p(
            "O projeto <b>não utiliza animais vertebrados vivos</b> de "
            "qualquer natureza, não se enquadrando na competência da "
            "Comissão de Ética no Uso de Animais (CEUA).",
            s["body"],
        )
    )

    story.append(_p("6.6 Enquadramento formal", s["h2"]))
    story.append(
        _p(
            "Pelo exposto, este projeto se enquadra no <b>Art. 8º da "
            "Resolução nº 200/2021/CONSU</b> — “Os projetos de pesquisa que "
            "não envolvem seres humanos, direta e indiretamente, nem "
            "animais vertebrados vivos, estão dispensados de cadastro” — e "
            "é objeto da <b>Declaração de Responsabilidade</b> prevista no "
            "parágrafo único do referido artigo, assinada pelo estudante, "
            "pelo orientador e pelo chefe do Departamento ao qual o "
            "orientador está vinculado.",
            s["body"],
        )
    )

    # 7. Cronograma
    story.append(_p("7. Cronograma", s["h1"]))
    cron_headers = ["Marco", "Data", "Descrição"]
    cron_rows = [
        ["Primeira revisão da qualificação ao orientador", "15/jul/2026", "Documento completo em LaTeX / Overleaf"],
        ["Pedido formal de qualificação ao Programa", "30/jul/2026", "Prazo regimental do PPG-CC / ICT"],
        ["Defesa da qualificação", "outubro/2026", "Sujeita à prorrogação já solicitada de dois meses"],
        ["Experimentos completos e escrita da dissertação", "nov/2026 – mai/2027", "—"],
        ["Defesa da dissertação", "2º semestre/2027", "—"],
    ]
    story.append(
        _table_header(
            cron_headers,
            cron_rows,
            col_widths=(6.5 * cm, 3.5 * cm, 6.0 * cm),
        )
    )

    story.append(PageBreak())

    # 8. Referências principais
    story.append(_p("8. Referências principais", s["h1"]))

    story.append(_p("Datasets e benchmarks", s["h2"]))
    refs_datasets = [
        "Kärkkäinen &amp; Joo (2021). FairFace: Face Attribute Dataset. WACV.",
        "Wang et al. (2019). Racial Faces in the Wild — RFW. ICCV.",
        "Robinson et al. (2020). Face Recognition: Too Bias, or Not Too Bias? CVPRW.",
        "Hazirbas et al. (2021). Casual Conversations. CVPRW.",
    ]
    for r in refs_datasets:
        story.append(_p(r, s["bullet"]))

    story.append(_p("Fundamentação teórica em fairness", s["h2"]))
    refs_fair = [
        "Buolamwini &amp; Gebru (2018). Gender Shades. FAT*.",
        "Hardt et al. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.",
        "Kleinberg et al. (2017). Inherent Trade-Offs in the Fair Determination of Risk Scores. ITCS.",
        "Madras et al. (2018). Learning Adversarially Fair and Transferable Representations. ICML.",
    ]
    for r in refs_fair:
        story.append(_p(r, s["bullet"]))

    story.append(_p("Skin tone e conditioning arquitetural", s["h2"]))
    refs_skin = [
        "Schumann et al. (2023). Consensus and Subjectivity of Skin Tone Annotation — MST-E. CVPR.",
        "Pereira et al. (2026). Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild — SkinToneNet. arXiv:2603.02475.",
        "Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI.",
    ]
    for r in refs_skin:
        story.append(_p(r, s["bullet"]))

    story.append(_p("Refutação e diálogo crítico", s["h2"]))
    refs_ref = [
        "Pangelinan et al. (2023). Analyzing Bias in Race Classification. FAccT.",
        "AlDahoul et al. (2024). FaceScanPaliGemma. arXiv.",
    ]
    for r in refs_ref:
        story.append(_p(r, s["bullet"]))

    story.append(_p("Diversidade fenotípica intra-Latinx", s["h2"]))
    refs_latinx = [
        "Telles (2014). Pigmentocracies: Ethnicity, Race, and Color in Latin America. UNC Press.",
        "Bryc et al. (2015). The Genetic Ancestry of African Americans, Latinos, and European Americans across the United States. AJHG.",
        "Pew Research (2017). Hispanic Identity Fades Across Generations.",
    ]
    for r in refs_latinx:
        story.append(_p(r, s["bullet"]))

    story.append(Spacer(1, 6))
    story.append(
        _p(
            "<i>Bibliografia consolidada de 104 referências disponível no "
            "repositório do projeto (arquivo referencias.bib).</i>",
            s["body"],
        )
    )

    # 9. Local
    story.append(_p("9. Local de execução", s["h1"]))
    story.append(
        _p(
            "Toda a pesquisa será conduzida em <b>ambiente computacional "
            "próprio</b> do mestrando e em infraestrutura de computação "
            "disponibilizada pela Unifesp / ICT ou por serviços de "
            "computação em nuvem contratados individualmente (Google Colab "
            "Pro, AWS educacional ou equivalente). <b>Não há uso de "
            "laboratórios institucionais que envolvam seres humanos ou "
            "animais.</b>",
            s["body"],
        )
    )

    # 10. Declaração final
    story.append(_p("10. Declaração final", s["h1"]))
    story.append(
        _p(
            "Declaro, para os devidos fins, que o projeto de pesquisa aqui "
            "descrito não envolve, direta ou indiretamente, seres humanos "
            "como sujeitos de pesquisa, nem faz uso de animais vertebrados "
            "vivos, enquadrando-se no Art. 8º da Resolução nº 200/2021 do "
            "Conselho Universitário da Universidade Federal de São Paulo. "
            "Comprometo-me a submeter novo protocolo ao Comitê de Ética em "
            "Pesquisa da Unifesp caso, no curso da pesquisa, sobrevenha "
            "qualquer alteração de escopo que envolva seres humanos ou "
            "animais.",
            s["body"],
        )
    )

    story.append(Spacer(1, 40))
    story.append(_p("São Paulo, 08 de julho de 2026.", s["assinatura"]))
    story.append(Spacer(1, 32))
    story.append(_p("_" * 55, s["assinatura"]))
    story.append(_p("Marcello Ozzetti — Mestrando", s["assinatura"]))
    story.append(Spacer(1, 24))
    story.append(_p("_" * 55, s["assinatura"]))
    story.append(_p("Prof. Dr. Marcos Quiles — Orientador / Pesquisador Responsável", s["assinatura"]))

    doc.build(story)


def main() -> None:
    here = Path(__file__).parent
    out = here / "projeto_declaracao_responsabilidade.pdf"
    build_pdf(out)
    print(f"PDF gerado: {out}")
    print(f"Tamanho: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
