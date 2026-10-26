# docs/ativo/ — materiais em uso ativo

> Diretório de trabalho corrente. Fase pós-qualificação (aprovado em
> 05/10/2026) — execução das 3 etapas / 2 metas do pipeline com prazo
> de ~6 meses até a defesa. Materiais da qualificação foram arquivados
> em `docs/historico/qualificacao_2026-10-05/`.
>
> **Última revisão:** 2026-10-26.

## 1. Bases narrativas dos capítulos

Documentos-fonte para a escrita no Overleaf.

| Arquivo | Vira qual capítulo |
|---|---|
| `_validacao_cientifica_pipeline.md` | Cap 4 (Metodologia — pipeline) |
| `_decisao_arquitetural_film.md` | Cap 4 (seção FiLM) |
| `_mapa_citacoes_por_capitulo.md` | Cap 2 (mapa das fichas por seção) |
| `_mudancas_pos_defesa.md` | Checklist de edições pós-qualificação |

## 2. Adequação ética CEP (Art. 8º Res. 200/2021)

| Arquivo | Uso |
|---|---|
| `_checklist_etica_cep.md` | Checklist com passos administrativos |
| `_projeto_declaracao_responsabilidade.md` | Texto-fonte do projeto formal |

## 3. Fichas bibliográficas e infraestrutura

| Arquivo | Uso |
|---|---|
| `04_pesquisa_bibliografica/` | Fichas verificadas (INDEX.md com tracks) |
| `_pdfs_inventario.md` | Inventário dos PDFs (data: jun/2026) |
| `_gerar_bibliografia.py` | Regera `docs/tese/references.bib` |

## 4. Referências ainda consultadas

Documentos consolidados de rodadas anteriores mantidos como âncora.

| Arquivo | Uso |
|---|---|
| `00_referencias.md` | Saneamento de citações |
| `01_taxonomia.md` | Nomenclatura, glossário, convenções |
| `05_landscape.md` | Síntese transversal da literatura |
| `06_gap.md` | Identificação e ranqueamento de lacunas |
| `07_thesis_statement.md` | Thesis statement — âncora conceitual |

## 5. Estado corrente do projeto (documentos vivos)

| Arquivo | Uso |
|---|---|
| `mestrado_architecture.md` | Mapa de código × Etapas do Cap 4 |
| `etapa1_report.md` | Relatório operacional da Etapa 1 |

## 6. Apresentação pós-qualificação

| Arquivo | Uso |
|---|---|
| `_gerar_apresentacao_pos_qualificacao_2026-10-26.py` | Script gerador do PPTX pós-qualificação |
| `material_pos_qualificacao_2026-10-26.pptx` | 12–15 slides, reunião de orientação |

> Material da qualificação (05/10/2026) arquivado em
> `docs/historico/qualificacao_2026-10-05/`.

## 7. Geradores de figuras da tese

Scripts que produzem imagens LaTeX-ready para os capítulos.

| Arquivo | Figura de saída |
|---|---|
| `_gerar_figuras_tese.py` | Múltiplas figuras da qualificação |
| `_gerar_imagem_convnext.py` | Arquitetura ConvNeXt-T |
| `_gerar_imagem_film.py` | Diagrama do mecanismo FiLM |
| `_gerar_imagem_hardt_metricas.py` | Métricas de fairness (Hardt 2016) |
| `_gerar_demo_mst_qualificacao.py` | 4 figuras: evolução + detalhe + validação MST (faces e heatmap) |
| `imagens/` | Diretório de saída das figuras |

---

## O que foi arquivado em `docs/historico/`

- `reuniao_2026-07-13/` — cola + evolução da reunião de 13/jul
- `reuniao_2026-07-20/` — cola + evolução + PPTX da reunião de 20/jul
- `reuniao_2026-08-17/` — script + PPTX da reunião de 17/ago
- `qualificacao_2026-10-05/` — gerador, PPTX (32 slides), narrativa,
  resumo, demo MST, pipeline 6-etapas, gabarito conceitual, sota review
- Reuniões e apresentações anteriores (jun/2026), scripts operacionais
  one-shot e análises intermediárias consolidadas

## O que foi removido (não arquivado)

Emails já enviados não são versionados — o registro fica no cliente de
email:

- `_email_secretaria_ppgcc_2026-07.md` (qualificação agendada)
- `email_orientador_2026-08-18.md` (retorno pós-reunião)
- `email_skintonenet_authors.md` (solicitação ao ICMC/USP + IMPA)
