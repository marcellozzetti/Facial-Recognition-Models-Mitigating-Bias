---
name: zhao-2025-aimfair
status_verificacao: VERIFIED
autores: [Zengqun Zhao, Ziquan Liu, Yu Cao, Shaogang Gong, Ioannis Patras]
ano: 2025
titulo: "AIM-Fair: Advancing Algorithmic Fairness via Selectively Fine-Tuning Biased Models"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)"
tipo_publicacao: conference
arxiv_id: a confirmar
doi: null
url_primario: https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_AIM-Fair_Advancing_Algorithmic_Fairness_via_Selectively_Fine-Tuning_Biased_Models_with_CVPR_2025_paper.html
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/zhao_2025_aimfair.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---

> Construída a partir de busca web. PDF integral pendente.

# AIM-Fair — Selectively Fine-Tuning Biased Models (CVPR 2025)

> **Selectively fine-tuning** modelos pré-treinados enviesados —
> abordagem de fairness in-processing via fine-tuning seletivo.
> Relevante para nossa Etapa 3 (race classifier conditioning) como
> alternativa moderna (2025) a abordagens tradicionais.

## 1. Resumo

> **[PENDENTE PDF]** Abstract não extraído. Inferência a partir do
> título:
>
> Pré-trained foundation models carregam viés. Em vez de re-treinar
> do zero, fairness pode ser conseguida via **fine-tuning seletivo**
> apenas das partes do modelo que mais contribuem para o viés.

## 2. Método (inferido)

> **[PENDENTE PDF]** Inferência:
> - Identificação de camadas/parâmetros enviesados.
> - Fine-tuning seletivo apenas nessas partes.
> - Mantém eficiência computacional (não fine-tuning completo).

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Não extraídos.

## 7. Aplicação ao nosso pipeline v3.2

### 7.1 Alternativa metodológica para Etapa 3

Para race classifier (Etapa 3), AIM-Fair sugere abordagem
ortogonal ao FiLM:

- **FiLM (nosso)**: insere camadas novas (FiLM blocks) para
  modular features.
- **AIM-Fair**: fine-tuning seletivo das camadas existentes mais
  enviesadas.

### 7.2 Possível combinação

ConvNeXt-T fine-tunado seletivamente (AIM-Fair-style) + FiLM-
conditioning (nosso). Combinação testável.

### 7.3 Como baseline em Cap 2

Comparar AIM-Fair fine-tuning seletivo contra nosso pipeline em
race classification 7-class FairFace. Endereça recomendação do
orientador (incluir mecanismo moderno).

## 8. Pontos para citar

- *"Zhao et al. (2025), apresentado no CVPR 2025, propõem AIM-Fair —
  abordagem de fairness algorítmica via fine-tuning seletivo de
  modelos pré-treinados enviesados, evitando o custo computacional
  de re-treinamento integral. A presente dissertação investiga, como
  baseline competitivo no Capítulo 2, se este paradigma supera ou
  iguala o FiLM-conditioning proposto."*

## 9-12. Arquivos e análise crítica

- PDF: pendente em `pdfs/zhao_2025_aimfair.pdf`.
- Entradas relacionadas: [[park_2022]] FSCL (alternativa), 
  [[manzoor_2024]] FineFACE (Pareto-efficient).

> **[BLOQUEADO]** Análise crítica detalhada requer leitura integral.
