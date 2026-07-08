"""Gera apresentacao PowerPoint da reuniao 2026-07-08.

Foco: material enxuto e consolidado apos 3 semanas sem reunioes.
- Plano de trabalho consolidado (3 marcos formais)
- Pendencia 1: escrita da qualificacao
- Pendencia 2: adequacao etica CEP
- Solicitacao de extensao de 2 meses
- Perguntas objetivas

Uso:
    python _gerar_apresentacao_2026-07-08.py
    -> produz: docs/ativo/material_reuniao_orientador_2026-07-08.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

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

    tx = slide.shapes.add_textbox(Inches(2.5), Inches(1.3), Inches(10.5), Inches(2.4))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Consolidação do plano de trabalho"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "Estado da escrita, adequação ética e solicitação de extensão"
    p2.font.size = Pt(20)
    p2.font.color.rgb = GRAY_DK

    p3 = tf.add_paragraph()
    p3.text = "Alinhamento após 3 semanas sem reuniões — foco em 2 pendências específicas"
    p3.font.size = Pt(15)
    p3.font.color.rgb = GRAY_MD
    p3.font.italic = True

    meta = slide.shapes.add_textbox(Inches(2.5), Inches(4.5), Inches(10.5), Inches(2.5))
    mf = meta.text_frame
    mf.word_wrap = True
    rows = [
        ("Mestrando:", "Marcello Ozzetti"),
        ("Orientador:", "Prof. Marcos Quiles"),
        ("Programa:", "Mestrado em Ciência da Computação — Unifesp / ICT"),
        ("Reunião:", "08 de julho de 2026"),
    ]
    for i, (k, v) in enumerate(rows):
        p = mf.paragraphs[0] if i == 0 else mf.add_paragraph()
        p.text = f"{k}  {v}"
        p.font.size = Pt(15)
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
    pn.font.size = Pt(140)
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
        p2.font.size = Pt(18)
        p2.font.color.rgb = GRAY_LT


def add_title(slide, text: str) -> None:
    tx_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.5), Inches(0.9))
    tf = tx_t.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(12.5), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()


def add_footer(slide, prs, text: str) -> None:
    if not text:
        return
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.5), Inches(0.4))
    pf = ft.text_frame.paragraphs[0]
    pf.text = text
    pf.font.size = Pt(10)
    pf.font.color.rgb = GRAY_MD
    pf.font.italic = True


def add_bullets(prs: Presentation, title: str, bullets: list, footer: str = "") -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    add_title(slide, title)

    tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.5), Inches(5.4))
    tf = tx.text_frame
    tf.word_wrap = True

    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            head, body = item
            run1 = p.add_run()
            run1.text = head + "  "
            run1.font.size = Pt(16)
            run1.font.bold = True
            run1.font.color.rgb = NAVY
            run2 = p.add_run()
            run2.text = body
            run2.font.size = Pt(16)
            run2.font.color.rgb = GRAY_DK
        else:
            p.text = "— " + item
            p.font.size = Pt(16)
            p.font.color.rgb = GRAY_DK
        p.space_after = Pt(8)

    add_footer(slide, prs, footer)


def add_table_slide(prs: Presentation, title: str, headers: list, rows: list, footer: str = "",
                    col_widths: list | None = None, highlight_rows: list | None = None) -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
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
                    run.font.size = Pt(11)
                    run.font.color.rgb = GRAY_DK

    add_footer(slide, prs, footer)


def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # 1. Capa
    add_title_slide(prs)

    # 2. Agenda enxuta
    add_bullets(
        prs,
        "Agenda da reunião",
        [
            ("1.", "Plano de trabalho consolidado — 3 marcos formais"),
            ("2.", "Pendência 1 — Escrita da qualificação e postura sobre uso de LLM"),
            ("3.", "Pendência 2 — Adequação ética CEP/Unifesp (Res. 200/2021)"),
            ("4.", "Solicitação pessoal — extensão de 2 meses no prazo da defesa"),
            ("5.", "Perguntas objetivas e próximos passos"),
        ],
        footer="Reunião enxuta, foco em alinhamento e desbloqueio.",
    )

    # SEÇÃO 1 — Plano consolidado
    add_section_divider(prs, "1", "Plano de trabalho consolidado",
                        "Três marcos formais no horizonte")

    add_table_slide(
        prs,
        "Três marcos formais",
        ["#", "Marco", "Data", "Observação"],
        [
            ["1", "1ª revisão da qualificação ao orientador", "15/jul/2026", "Mantida — 7 dias a partir de hoje"],
            ["2", "Pedido formal de qualificação (PPG-CC / ICT)", "30/jul/2026", "Mantido — prazo regimental do Programa"],
            ["3", "Defesa da qualificação", "outubro/2026", "Ajustada de agosto para outubro (solicitação de extensão de 2 meses)"],
        ],
        col_widths=[0.5, 5.0, 2.5, 5.0],
        highlight_rows=[2],
        footer="Marcos 1 e 2 mantidos; marco 3 depende da solicitação formal de extensão.",
    )

    # SEÇÃO 2 — Escrita
    add_section_divider(prs, "2", "Pendência 1 — Escrita da qualificação",
                        "Corpo pronto no repositório; transferência manual para Overleaf")

    add_table_slide(
        prs,
        "Estado do corpo dos capítulos no repositório",
        ["Capítulo", "Base narrativa no repositório", "Estado"],
        [
            ["1. Introdução", "_pre_qualificacao_narrativa.md v1.2", "Corpo pronto"],
            ["2. Revisão bibliográfica", "104 fichas + _mapa_citacoes_por_capitulo.md", "Corpo pronto"],
            ["3. Objetivos", "_objetivo_tese v3.6 (OG + 6 OE + 6 H + 7 C)", "Corpo pronto"],
            ["4. Metodologia", "_decisao_arquitetural_film + _checklist_etica_cep", "Corpo pronto"],
            ["5. Cronograma", "Marcos ajustados desta reunião", "A finalizar após alinhamento"],
        ],
        col_widths=[2.5, 6.5, 3.5],
        footer="Repositório GitHub pode ser aberto ao vivo durante a reunião.",
    )

    add_bullets(
        prs,
        "Postura sobre uso de LLM — este trabalho NÃO é 100 % LLM",
        [
            ("Princípio:", "cada linha do texto final é escrita, lida e apropriada conscientemente pelo mestrando"),
            ("", ""),
            ("Papel do mestrando:", "definição do problema, escolha do dataset, leitura crítica dos 104 papers, decisões arquiteturais, argumentação"),
            ("Papel da LLM (Claude Code):", "ferramenta operacional — estruturação, revisão de coerência, formatação de fichas"),
            ("Papel do NotebookLM:", "segunda opinião externa e independente (validação, não construção)"),
            ("", ""),
            ("Transferência para Overleaf:", "MANUAL, linha a linha — garante consistência ABNT + revisão obrigatória + apropriação defensável"),
        ],
        footer="Analogia: LLM é ferramenta operacional (como IDE, corretor ortográfico, mecanismo de busca).",
    )

    add_table_slide(
        prs,
        "Divisão explícita de responsabilidade autor × LLM",
        ["Atividade", "Autor", "Papel da LLM"],
        [
            ["Definição do problema e pergunta de pesquisa", "Marcello", "Nenhum"],
            ["Leitura crítica dos 104 papers", "Marcello (PDFs, VPN)", "Assistência em resumos pós-leitura"],
            ["Fichas bibliográficas (104)", "Marcello (verificação PDF a PDF)", "Formatação padronizada"],
            ["Argumentação e storytelling", "Marcello (decisões editoriais)", "Sugestão de estrutura + revisão de coerência"],
            ["Decisão arquitetural FiLM", "Marcello + Orientador (reuniões)", "Comparação sistemática entre alternativas"],
            ["Ajuste ético do OE-1 (v3.6)", "Marcello (interpretação Res. 200/2021)", "Análise da resolução"],
            ["Transferência para Overleaf", "Marcello — manual, linha a linha", "Nenhum"],
        ],
        col_widths=[4.5, 4.0, 4.0],
        footer="Se o orientador desejar, posso abrir os capítulos no repositório agora.",
    )

    # SEÇÃO 3 — Adequação ética
    add_section_divider(prs, "3", "Pendência 2 — Adequação ética CEP",
                        "Resolução 200/2021/CONSU Unifesp — enquadramento no Art. 8º")

    add_bullets(
        prs,
        "Análise da Resolução 200/2021/CONSU Unifesp",
        [
            ("Base legal:", "Resolução nº 200/2021/CONSELHO UNIVERSITÁRIO Unifesp (SEI 0719529)"),
            ("", ""),
            ("Pergunta central:", "o projeto envolve, direta ou indiretamente, seres humanos?"),
            ("Datasets utilizados:", "FairFace, RFW, BFW, BUPT — todos secundários, públicos, sem coleta primária"),
            ("Natureza da pesquisa:", "puramente computacional, sem intervenção em seres humanos"),
            ("Ajuste v3.6 aplicado:", "validação MST do OE-1 substituída por processo interno (Mestrando + Orientador)"),
            ("", ""),
            ("Enquadramento:", "Art. 8º — dispensa de cadastro no CEP; exige apenas Declaração de Responsabilidade"),
        ],
        footer="Auditoria completa em _checklist_etica_cep.md",
    )

    add_table_slide(
        prs,
        "Ajuste técnico no OE-1 (v3.5 → v3.6)",
        ["Aspecto", "v3.5 (anterior)", "v3.6 (atual)"],
        [
            ["Método", "Crowdsourcing externo via Prolific", "Validação interna pela equipe acadêmica"],
            ["Anotadores", "~3 externos pagos", "Mestrando + Orientador"],
            ["Escala", "~700 imgs estratificadas", "~200-300 imgs estratificadas"],
            ["Estratificação", "Por raça e tom MST", "Por raça e tom MST (preservada)"],
            ["Implicação ética", "Requer CEP (pesquisa indireta com humanos)", "Dispensa CEP → apenas Declaração"],
        ],
        col_widths=[2.5, 4.5, 5.5],
        highlight_rows=[4],
        footer="Rigor metodológico preservado via estratificação; escala adaptada ao processo interno.",
    )

    add_bullets(
        prs,
        "Trâmite administrativo — próximos passos",
        [
            ("1.", "Confirmar hoje quem é o(a) Chefe do Departamento (3ª assinatura obrigatória)"),
            ("2.", "Baixar modelo de Declaração em http://www.cep.unifesp.br/cep"),
            ("3.", "Preencher com dados do projeto"),
            ("4.", "Coletar 3 assinaturas via SEI Unifesp — Mestrando + Orientador + Chefe do Departamento"),
            ("5.", "Anexar Declaração assinada à entrega da qualificação"),
            ("6.", "Incluir parágrafo metodológico no Cap 4 sobre dispensa CEP (Art. 8º)"),
        ],
        footer="",
    )

    # SEÇÃO 4 — Solicitação pessoal
    add_section_divider(prs, "4", "Solicitação pessoal — extensão de 2 meses",
                        "Cuidado parental e reprogramação do prazo da defesa")

    add_bullets(
        prs,
        "Solicitação formal de extensão",
        [
            ("Motivo:", "nascimento de minha filha em 28/março/2026"),
            ("", "Período de afastamento das atividades acadêmicas nas semanas subsequentes para cuidado parental"),
            ("", ""),
            ("Solicitação:", "extensão de 2 meses no prazo da defesa da qualificação"),
            ("Prazo anterior:", "agosto/2026"),
            ("Prazo solicitado:", "outubro/2026"),
            ("", ""),
            ("Marcos preservados:", "1ª revisão em 15/jul e pedido formal em 30/jul mantidos — a extensão afeta apenas o prazo final"),
            ("Documento:", "carta formal em 2 parágrafos redigida, pronta para envio via SEI ou conforme trâmite indicado"),
        ],
        footer="",
    )

    # SEÇÃO 5 — Perguntas e próximos passos
    add_section_divider(prs, "5", "Perguntas e próximos passos",
                        "Alinhamento objetivo — 6 perguntas + plano de 7 dias")

    add_bullets(
        prs,
        "Perguntas objetivas ao orientador",
        [
            ("1.", "Estrutura dos capítulos apresentada está adequada? Sugestões?"),
            ("2.", "Chefe do Departamento — quem assina a Declaração de Responsabilidade do CEP?"),
            ("3.", "Trâmite formal da extensão de 2 meses — carta, requerimento SEI, processo?"),
            ("4.", "Template Overleaf institucional Unifesp / ICT ou padrão ABNT genérico?"),
            ("5.", "Co-orientador — alguma definição?"),
            ("6.", "Composição da banca preliminar — alguma sugestão?"),
        ],
        footer="",
    )

    add_table_slide(
        prs,
        "Próximos 7 dias — plano até 15/jul",
        ["Data", "Atividade", "Entregável"],
        [
            ["08-09/jul", "Transferência do Cap 1 (Markdown → Overleaf)", "Cap 1 em LaTeX"],
            ["10-11/jul", "Transferência do Cap 2 (Revisão bibliográfica)", "Cap 2 em LaTeX"],
            ["12-13/jul", "Transferência dos Caps 3, 4 e 5", "Caps 3-5 em LaTeX"],
            ["14/jul", "Revisão final integrada + ajustes desta reunião", "Documento consolidado"],
            ["15/jul", "Entrega da 1ª revisão da qualificação", "Marco 1 atingido"],
        ],
        col_widths=[2.0, 6.0, 4.5],
        highlight_rows=[4],
        footer="Tramitação da Declaração CEP e carta de extensão em paralelo.",
    )

    # Fechamento
    add_section_divider(prs, "", "Obrigado",
                        "Discussão livre — dúvidas e alinhamento")

    return prs


def main() -> None:
    here = Path(__file__).parent
    out = here / "material_reuniao_orientador_2026-07-08.pptx"
    prs = build_presentation()
    prs.save(out)
    print(f"Apresentacao gerada: {out}")
    print(f"Total de slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
