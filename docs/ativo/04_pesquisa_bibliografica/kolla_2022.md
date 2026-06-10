---
name: kolla-2022
status_verificacao: OVERVIEW_ONLY
autores: [Manideep Kolla, Aravinth Savadamuthu]
ano: 2022
titulo: "The Impact of Racial Distribution in Training Data on Face Recognition Bias: A Closer Look"
venue: "IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW 2023)"
tipo_publicacao: workshop_conference
arxiv_id: "2211.14498"
doi: null
url_primario: https://arxiv.org/abs/2211.14498
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Apenas abstract via arXiv. Contexto adicional via [[../_validacao_cientifica_pipeline]] R6-5.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**
>
> Construída apenas a partir do abstract. PDF integral pendente.

# Racial Distribution in Training Data (Kolla & Savadamuthu, 2022)

> **Refuta a hipótese ingênua "balanceamento racial resolve fairness".**
> Conduz 16 experimentos com proporções variáveis e demonstra que
> uniform distribution NÃO basta. Introduz métrica "racial gradation".

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

Algoritmos de FR são úteis mas perigosos quando enviesados. É
essencial entender **como treino afeta accuracy e fairness**. O paper
investiga o **efeito da distribuição racial dos dados de treino** no
desempenho de modelos de FR.

## 2. Método

> *Fonte: abstract verbatim.*

- **16 experimentos** com distribuições raciais variáveis no treino.
- Análise multi-dimensional:
  - Accuracy metrics
  - Clustering metrics
  - UMAP projections
  - Face quality
  - Decision thresholds
- **Métrica nova introduzida**: **racial gradation** — mede
  correlação inter e intra-race em features faciais e impacto no
  aprendizado.

> **[PENDENTE PDF]** Datasets usados (MS1MV2? variantes?). FR model
> exato (ArcFace? CosFace?). Proporções específicas testadas.
> Definição matemática de racial gradation.

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Apenas as **conclusões qualitativas** estão no
> abstract:
>
> *"a uniform distribution of races in the training datasets alone
> does not guarantee bias-free face recognition algorithms"*
>
> *"factors like face image quality play a crucial role"*

## 7. Limitações que identifiquei (a partir do abstract apenas)

- **Workshop paper** (WACVW) — revisão menos profunda que main
  conference.
- **Foco em FR** (verification 1:1), não classification.
- **16 experimentos** é número fixo — sem ablation de quantos seriam
  suficientes.
- **Racial gradation** introduzida sem benchmarking contra métricas
  existentes (DSAP de Dominguez 2024, ε de FairFace).
- **Sem mencionar comparação com intervenções arquiteturais**
  (Dooley 2022 paralelo).

## 8. Relação com nossa pesquisa

### 8.1 Reforça motivação central

Kolla confirma empíricamente o que vemos no FairFace: **balanceamento
não basta**. FairFace é balanceado em raça mas Latinx F1=60% vs
White F1=80% (AlDahoul 2024 Tabela 16). Kolla fornece evidência
adicional em FR.

### 8.2 Justifica intervenção arquitetural

Se balanceamento de dados não basta, então **mecanismo arquitetural
é necessário**. Justifica nosso FiLM-conditioning como abordagem
além de balanceamento.

### 8.3 Métrica "racial gradation"

Pode ser relevante para nossa C2 (matriz MST × race). Se definida
como correlação inter/intra-race em features, é conceitualmente
próxima à nossa proposta de quantificar overlap fenotípico via MST.

> **[PENDENTE PDF]** Definição exata da métrica é necessária para
> avaliar overlap com nossa C2.

## 9. Pontos para citar

- *"Kolla & Savadamuthu (2022), em estudo apresentado no WACVW 2023,
  demonstram empíricamente, com base em 16 experimentos com
  proporções variáveis de raças no treino, que distribuição uniforme
  de raças nos dados de treinamento NÃO garante algoritmos de face
  recognition livres de viés — fatores como qualidade da imagem
  desempenham papel crucial. Esta evidência converge com a observação
  da disparidade Latinx persistente no FairFace balanceado
  (Kärkkäinen & Joo, 2021), justificando a busca por intervenções
  arquiteturais além de balanceamento de dados."*

## 10. Arquivos relacionados

- PDF: pendente em `pdfs/kolla_2022.pdf`.
- Análise R6 em [[../_validacao_cientifica_pipeline]] (R6-5).
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace
  balanceado), [[grother_2019]] (NIST escala), [[pangelinan_2023]]
  (pixel info como confounder).

## 11-12. Pendente PDF

> **[BLOQUEADO]** Future work e análise crítica detalhada requerem
> leitura integral.
