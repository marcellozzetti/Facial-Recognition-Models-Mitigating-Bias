---
name: enhancing-visual-attributes-2022
status_verificacao: VERIFIED
autores: [Tobias Hänel, Nishant Kumar, Dmytro Schienle (Eslami), Stefan Gumhold (TU Dresden) — Carl Zeiss Meditec AG]
ano: 2022
titulo: "Enhancing Fairness of Visual Attribute Predictors"
venue: "arXiv preprint (jul 2022) — TU Dresden + Carl Zeiss Meditec"
tipo_publicacao: preprint
arxiv_id: "2207.05727"
doi: null
url_primario: https://arxiv.org/abs/2207.05727
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/enhancing_visual_attributes_2022.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract lido via pdftotext.
---

# Enhancing Fairness of Visual Attribute Predictors (Hänel et al., 2022)

> **Primeiro trabalho** a incorporar losses de fairness baseadas em
> batch estimates de Demographic Parity, Equalized Odds e uma nova
> **medida Intersection-over-Union (IoU)** em training end-to-end para
> mitigar viés em visual attribute predictors.

## 1-2. Resumo + método

- Performance de DNN para image recognition (e.g., predicting smiling
  face) **degrada com classes sub-representadas** de atributos
  sensíveis.
- **Contribuição**: 3 losses de regularização **fairness-aware**:
  1. **Demographic Parity loss** (batch estimate)
  2. **Equalized Odds loss** (batch estimate)
  3. **Intersection-over-Union (IoU) loss** — novel
- **First attempt** a incorporar essas losses em training end-to-end
  para visual attribute prediction.

## 3-5. Datasets, métricas, resultados

- **Datasets**: CelebA + UTKFace + SIIM-ISIC melanoma challenge.
- **Resultado**: melhora fairness mantendo high classification
  performance.

## 7. Aplicação ao pipeline v3.2

- **Baseline alternativo** para Cap 2 — losses fairness-regularized.
- **Diferença vs FiLM-conditioning**: Hänel modula loss durante treino;
  FiLM modula features. Mecanismos ortogonais.
- **CelebA/UTKFace** — não FairFace race 7-class — generalização
  indireta.

## 8. Citar

- *"Hänel et al. (2022, arXiv:2207.05727) introduzem o primeiro
  framework end-to-end com losses de regularização fairness-aware
  baseadas em batch estimates de Demographic Parity, Equalized Odds
  e uma medida novel Intersection-over-Union para visual attribute
  prediction. A demonstração empírica de redução de viés mantendo
  performance em CelebA, UTKFace e SIIM-ISIC reforça a viabilidade
  de mitigação algorítmica complementar ao balanceamento de dados —
  estratégia adotada nesta dissertação."*

## 9-12.

PDF: `pdfs/enhancing_visual_attributes_2022.pdf`. Conexões:
[[hardt_2016]] (origem das métricas DP, EOD), [[zhang_2018]]
(Adversarial debiasing — paradigma alternativo), [[park_2022]]
(FSCL — fair contrastive alternativo).
