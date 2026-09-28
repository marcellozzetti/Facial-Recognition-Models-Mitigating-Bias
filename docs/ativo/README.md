# docs/ativo/ — materiais em uso ativo

> Diretório de trabalho corrente. Contém o essencial para (a) a defesa
> da qualificação em **05/10/2026** e (b) a execução das Etapas 1–6 do
> pipeline. Materiais concluídos, reuniões passadas e emails já
> enviados foram arquivados em `docs/historico/` ou removidos.
>
> **Última revisão:** 2026-08-30.

## 1. Bases narrativas dos capítulos

Documentos-fonte para a escrita no Overleaf.

| Arquivo | Vira qual capítulo |
|---|---|
| `_pre_qualificacao_narrativa.md` | Cap 1 (Introdução) + estrutura geral |
| `_validacao_cientifica_pipeline.md` | Cap 4 (Metodologia — pipeline) |
| `_decisao_arquitetural_film.md` | Cap 4 (seção FiLM) |
| `_mapa_citacoes_por_capitulo.md` | Cap 2 (mapa das fichas por seção) |
| `_resumo_qualificacao.md` | Elemento pré-textual (resumo/abstract) |

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

## 6. Material da qualificação — próxima entrega (05/10/2026)

| Arquivo | Uso |
|---|---|
| `_gerar_apresentacao_qualificacao_2026-09.py` | Script gerador do PPTX de defesa |
| `material_qualificacao_2026-09-30.pptx` | 25 slides, ~15–20 min de fala |

## 7. Geradores de figuras da tese

Scripts que produzem imagens LaTeX-ready para os capítulos.

| Arquivo | Figura de saída |
|---|---|
| `_gerar_figuras_tese.py` | Múltiplas figuras da qualificação |
| `_gerar_imagem_convnext.py` | Arquitetura ConvNeXt-T |
| `_gerar_imagem_film.py` | Diagrama do mecanismo FiLM |
| `_gerar_imagem_hardt_metricas.py` | Métricas de fairness (Hardt 2016) |
| `_gerar_imagem_pipeline_6etapas.py` | Pipeline em 6 etapas (Cap 4) |
| `imagens/` | Diretório de saída das figuras |

---

## O que foi arquivado em `docs/historico/`

- `reuniao_2026-07-13/` — cola + evolução da reunião de 13/jul
- `reuniao_2026-07-20/` — cola + evolução + PPTX da reunião de 20/jul
- `reuniao_2026-08-17/` — script + PPTX da reunião de 17/ago
- Reuniões e apresentações anteriores (jun/2026), scripts operacionais
  one-shot e análises intermediárias consolidadas

## O que foi removido (não arquivado)

Emails já enviados não são versionados — o registro fica no cliente de
email:

- `_email_secretaria_ppgcc_2026-07.md` (qualificação agendada)
- `email_orientador_2026-08-18.md` (retorno pós-reunião)
- `email_skintonenet_authors.md` (solicitação ao ICMC/USP + IMPA)
