# docs/ativo/ — materiais em uso para a escrita da qualificação

> Diretório de trabalho ativo. Contém apenas o essencial para a fase
> de escrita da qualificação (jul-out/2026). Materiais concluídos ou
> de rodadas anteriores foram arquivados em `docs/historico/`.

## 1. Bases narrativas dos capítulos

Documentos-fonte para a escrita no Overleaf. Cada base foi transposta
manualmente para LaTeX (`docs/tese/capitulos/`) ao longo de 08-12/jul.

| Arquivo | Vira qual capítulo |
|---|---|
| `_pre_qualificacao_narrativa.md` | Cap 1 (Introdução) + estrutura geral |
| `_objetivo_tese_v3.3.md` (v3.6) | Cap 3 (Objetivos) |
| `_validacao_cientifica_pipeline.md` | Cap 4 (Metodologia — pipeline) |
| `_decisao_arquitetural_film.md` | Cap 4 (seção FiLM) |
| `_mapa_citacoes_por_capitulo.md` | Cap 2 (mapa das 104 fichas por seção) |

## 2. Adequação ética CEP (Art. 8º Res. 200/2021)

Material da tramitação da Declaração de Responsabilidade — ativo até
que a Declaração seja assinada e anexada à qualificação.

| Arquivo | Uso |
|---|---|
| `_checklist_etica_cep.md` | Checklist com 5 passos administrativos |
| `_projeto_declaracao_responsabilidade.md` | Texto-fonte do projeto formal |
| `projeto_declaracao_responsabilidade.docx` | Word editável (submissão) |
| `projeto_declaracao_responsabilidade.pdf` | PDF profissional (alternativo) |
| `_gerar_projeto_docx.py` | Regera .docx se editar o .md fonte |
| `_gerar_projeto_pdf.py` | Regera .pdf se editar o .md fonte |

## 3. Fichas bibliográficas e infraestrutura

| Arquivo | Uso |
|---|---|
| `04_pesquisa_bibliografica/` | **107 fichas verificadas** (INDEX.md com 11 tracks) |
| `_validacao_cross_reference_v3.md` | Auditoria tese × 104 fichas — consultar durante escrita |
| `_pdfs_inventario.md` | Inventário atual dos 103 PDFs no repositório |
| `_gerar_bibliografia.py` | Regera `docs/tese/referencias.bib` se fichas mudarem |

## 4. Documentos de referência (histórico ainda útil)

Documentos consolidados de rodadas anteriores (2026-05 a 2026-06) que
ainda são consultados eventualmente durante a escrita.

| Arquivo | Uso |
|---|---|
| `00_referencias.md` | Saneamento de citações — quais citar, quais evitar |
| `01_taxonomia.md` | Nomenclatura, glossário e convenções (usar durante escrita) |
| `05_landscape.md` | Síntese transversal da literatura |
| `06_gap.md` | Identificação e ranqueamento de lacunas |
| `07_thesis_statement.md` | Thesis statement v3.2 (prescritivo — âncora conceitual) |

## 5. Material da reunião de 13/jul/2026

Reunião de status com Prof. Marcos Quiles. Após a reunião, este
material será movido para `docs/historico/reuniao_2026-07-13/`.

| Arquivo | Uso |
|---|---|
| `_reuniao_2026-07-13_cola.md` | Cola de bolso (1 página, 8-10 min) |
| `_reuniao_2026-07-13_evolucao.md` | Documento técnico completo |
| `_gerar_apresentacao_2026-07-13.py` | Script gerador do PPT |
| `material_reuniao_orientador_2026-07-13.pptx` | Apresentação (21 slides) |

## 6. Geradores de figuras para a tese

Scripts que produzem imagens LaTeX-ready para inclusão nos capítulos.

| Arquivo | Figura |
|---|---|
| `_gerar_imagem_convnext.py` | Arquitetura ConvNeXt-T |
| `_gerar_imagem_film.py` | Diagrama do mecanismo FiLM |
| `_gerar_imagem_hardt_metricas.py` | Métricas de fairness (Hardt 2016) |
| `imagens/` | Diretório de saída das figuras |

---

## O que foi arquivado (docs/historico/)

- **`reuniao_2026-06-28_nao_realizada/`** — material da reunião de 28/jun que não aconteceu
- **`reuniao_2026-07-08_nao_realizada/`** — material da reunião de 08/jul que foi remarcada para 13/jul
- **`scripts_operacionais/`** — 7 scripts one-shot já executados (download PDFs, correção autoria, auditoria)
- **`analises_pontuais/`** — 6 análises intermediárias já consolidadas nos documentos ativos

Ver README em cada pasta para detalhamento.
