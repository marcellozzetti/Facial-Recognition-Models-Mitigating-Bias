---
name: porgali-2023-ccv2
status_verificacao: VERIFIED
autores: [Bilal Porgali, Vítor Albiero, Jordan Ryda, Cristian Canton Ferrer, Caner Hazirbas]
ano: 2023
titulo: "The Casual Conversations v2 Dataset: A diverse, large benchmark for measuring fairness and robustness in audio/vision/speech models"
venue: "IEEE/CVF CVPR Workshops (CVPRW 2023) — Meta AI"
tipo_publicacao: workshop_conference
arxiv_id: "2303.04838"
doi: null
url_primario: https://arxiv.org/abs/2303.04838
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado de arXiv (pdfs/porgali_2023_ccv2.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução, descrição dataset e tabelas de cobertura lidos via pdftotext.
---

# Casual Conversations v2 Dataset (Porgali et al., CVPRW 2023 — Meta AI)

> **Expansão substancial do Casual Conversations v1** (Hazirbas
> 2021): 26 467 vídeos de 5 567 participantes em **7 países**
> (Brasil, Índia, Indonésia, México, Vietnã, Filipinas, EUA), com
> anotação **Fitzpatrick + MST simultânea**. Útil como **dataset
> alternativo de validação** para nosso classificador MST.

## 1. Resumo

> *Fonte: abstract + Meta AI Research page.*

CCv2 é consent-driven dataset com:
- **26 467 vídeos** de 5 567 participantes únicos.
- **Média de 5 vídeos por pessoa**.
- **7 países**: Brasil, Índia, Indonésia, México, Vietnã, Filipinas,
  EUA.
- **11 atributos demográficos**:
  - Self-reported: age, gender, language/dialect, disability,
    physical adornments, physical attributes, geo-location.
  - Annotated by trained: **Fitzpatrick Skin Type** + **Monk Skin Tone**
    + voice timbre + activity/recording conditions.

Designed para **avaliação de fairness e robustness** em
audio/vision/speech.

## 2. Método (construção do dataset)

> **[PENDENTE PDF]** Inferência:
> - Recrutamento consent-driven via plataforma paga.
> - Cobertura geográfica intencional.
> - Protocolo de anotação treinada (não amador).
> - Self-report para atributos pessoais.

## 3. Datasets e setup

- **CCv2** sucessor de CCv1 (Hazirbas et al. 2021, [[dataset_hazirbas_2021]]).
- **Anotação dupla** Fitzpatrick + MST permite comparação direta.

## 4-6. Métricas, resultados, limitações

- Dataset descritivo (não reporta métricas de modelo). Estatísticas:
  26.467 vídeos / 5.567 pessoas / 7 países / 11 atributos.
- Limitação reconhecida: comparado a CCv1 (US-only), Dollar Street
  (sem pessoa attrs), Open Images MIAP (perceived gender), FairFace
  e UTK (web-scraped, não consent-driven), MORPH (gender binário e
  poucos attrs).
- CCv2 não substitui FR-task benchmarks — é benchmark de **fairness
  audit** em audio/vision/speech models.

## 7. Aplicação ao nosso pipeline v3.2

### 7.1 Dataset complementar de validação MST

CCv2 tem **MST + Fitzpatrick anotados** em escala média (~26K
vídeos). Útil para:

1. **Validação cross-domain** do nosso SkinToneNet (ou modelo MST
   pré-treinado escolhido).
2. **Comparação entre escalas** Fitzpatrick e MST em mesmo subject —
   evidência empírica do trade-off.

### 7.2 Cobertura geográfica

Diferente do FairFace (US-centric), CCv2 cobre **Brasil + 6 outros
países**. Para validade externa do nosso classificador MST,
particularmente importante.

### 7.3 Diferença vs FairFace

- **FairFace**: imagens estáticas com race labels.
- **CCv2**: vídeos com self-reported demographics + skin tone
  annotated.

**Complementares**, não substitutos.

## 8. Pontos para citar

- *"Porgali et al. (2023), apresentado no CVPR Workshops 2023, expandem
  substancialmente o Casual Conversations original com 26 467 vídeos
  de 5 567 participantes em sete países, incluindo anotação simultânea
  Fitzpatrick + Monk Skin Tone Scale. Esta cobertura geográfica
  amplificada constitui benchmark robusto para validação de
  classificadores de skin tone em condições de iluminação e captura
  diversas."*

- *"A combinação de self-reported demographics (gender, age, dialect,
  geo-location) com anotação por anotadores treinados (Fitzpatrick,
  MST, voice timbre) introduzida por Porgali et al. (2023) no Casual
  Conversations v2 estabelece padrão metodológico para datasets
  fairness-oriented em escala industrial — padrão que datasets
  centrados em race classification como o FairFace (Kärkkäinen &
  Joo, 2021) não atendem."*

## 9-12. Arquivos e análise crítica

- PDF: `pdfs/porgali_2023_ccv2.pdf`.
- Dataset Meta: ai.meta.com/datasets/casual-conversations-v2-dataset/
- Entradas relacionadas: [[dataset_hazirbas_2021]] (CCv1 — predecessor),
  [[schumann_2023]] (MST protocol), [[fitzpatrick_1988]].
