---
name: racial-bias-dataset-2017
status_verificacao: VERIFIED
autores: [Rachel Hong, Tadayoshi Kohno, Jamie Morgenstern (University of Washington)]
ano: 2023
titulo: "Evaluation of Targeted Dataset Collection on Racial Equity in Face Recognition"
venue: "ACM EAAMO 2023 — Equity and Access in Algorithms, Mechanisms, and Optimization"
tipo_publicacao: conference
arxiv_id: null
doi: "10.1145/3600211.3604662"
url_primario: https://dl.acm.org/doi/10.1145/3600211.3604662
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN institucional (pdfs/racial_bias_dataset_2017.pdf — nome do arquivo herdado, mas paper é de 2023). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

> ℹ️ **Nota sobre o nome do arquivo**: o slug `racial_bias_dataset_2017`
> foi herdado de uma referência anterior; o paper de fato é
> **Hong, Kohno & Morgenstern (2023)**, publicado na **EAAMO 2023**.
> Conteúdo e metadados atualizados após leitura do PDF.

# Targeted Dataset Collection on Racial Equity (Hong, Kohno & Morgenstern — EAAMO 2023)

> **Framework empírico para avaliar impacto de coleta direcionada
> de dados em fairness de FR**. Achados confirmam intuição "adicionar
> dados do grupo com pior performance ajuda mais que adicionar dados
> de outros grupos" — mas com nuances importantes detectadas em
> análise de embeddings.

## 1-2. Resumo + método

- **Motivação**: empresas têm respondido a audits diversificando
  data collection. Quanto isso ajuda quantitativamente?
- **Framework**: avaliar impacto de adicionar dados de grupo X em
  performance de cada grupo demográfico.
- **Datasets testados**: 3 datasets benchmark racially-annotated.
- **Modelos testados**: 3 modelos standard de FR.

## 3-5. Achados empíricos

- **Hipótese confirmada**: adicionar dados do grupo com **performance
  mais baixa** beneficia esse grupo significativamente mais que
  adicionar dados de outros grupos.
- **Sem prejuízo**: introduzir dados de grupo previamente omitido
  **não degrada performance** de outros grupos.
- **Causa identificada**: aumentos de performance estão associados
  a **maior separação no espaço de embeddings entre identidades
  diferentes**.
- **Achado adicional**: em um dos datasets, treinar em **um único
  grupo racial generaliza bem para todos** — diferença importante
  vs outros datasets.

## 7. Aplicação ao pipeline v3.2

- **Cap 1** (Introdução): contexto sobre eficácia de targeted
  collection — complementa narrativa de "balanceamento não basta"
  (Pangelinan, Rethinking Assumptions).
- **Suporte indireto à H6**: achado sobre embedding separation
  sugere que **features importam**, não só dados.
- **Não muda nosso pipeline** — operamos em FairFace (já dado).

## 8. Citar

- *"Hong, Kohno & Morgenstern (EAAMO 2023, DOI
  10.1145/3600211.3604662) propõem framework empírico para avaliar
  o impacto de coleta direcionada de dados em equidade racial de
  face recognition, demonstrando em três datasets que adicionar
  dados do grupo com performance mais baixa beneficia esse grupo
  sem degradar os demais, com ganhos associados a maior separação
  entre identidades no espaço de embeddings. Os autores também
  observam variação cross-dataset relevante — em um caso, treinar
  em um único grupo racial generaliza para todos —, evidência da
  criticidade de re-aplicar avaliações empíricas a cada novo
  dataset, posição que esta dissertação adota ao reportar resultados
  específicos sobre o FairFace 7-class."*

## 9-12.

PDF: `pdfs/racial_bias_dataset_2017.pdf` (nome herdado).
Conexões: [[rethinking_assumptions_2021]] (paralelo — balanceamento
não basta), [[kolla_2022]] (também questiona distribuição uniforme),
[[pangelinan_2023]] (pixel info como confounder),
[[dataset_karkkainen_2021]] (FairFace — exemplo de targeted balanced
collection).
