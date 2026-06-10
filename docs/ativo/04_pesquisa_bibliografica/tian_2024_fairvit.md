---
name: tian-2024-fairvit
status_verificacao: OVERVIEW_ONLY
autores: [Bowei Tian, Ruijie Du, Yanning Shen]
ano: 2024
titulo: "FairViT: Fair Vision Transformer via Adaptive Masking"
venue: "European Conference on Computer Vision (ECCV 2024)"
tipo_publicacao: conference
arxiv_id: "2407.14799"
doi: null
url_primario: https://arxiv.org/abs/2407.14799
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# FairViT — Fair Vision Transformer via Adaptive Masking (ECCV 2024)

> **Fair ViT framework** com **distance loss + adaptive
> fairness-aware masks** em attention layers. Track J — fairness
> via arquitetura ViT específica.

## 1-2. Resumo + método

- **Distance loss** durante treino.
- **Adaptive fairness-aware masks** em attention layers que atualizam
  com parâmetros do modelo.
- Atinge melhor accuracy que alternativas mantendo fairness razoável.

## 7. Aplicação ao pipeline v3.2

- **Alternativa arquitetural ViT** ao nosso ConvNeXt-T.
- Para Cap 2 ablation: **trocar backbone para ViT-FairViT** vs
  ConvNeXt-T + FiLM.
- Endereça orientador (testar mecanismos modernos arquiteturais).

## 8. Citar

- *"FairViT (Tian, Du & Shen, ECCV 2024, arXiv:2407.14799) propõe
  Vision Transformer fair via adaptive masking em attention layers,
  demonstrando que masks fairness-aware atualizadas com parâmetros
  do modelo conseguem manter accuracy superior. A presente
  dissertação considera FairViT como referência arquitetural
  alternativa ao backbone ConvNeXt-T adotado, particularmente para
  ablation sobre o impacto da escolha de paradigma arquitetural
  (convolutional vs transformer) na efetividade do conditioning
  via tom de pele."*

## 9-12.

PDF pendente. Conexões: [[manzoor_2024]] FineFACE, ConvNeXt-T
(nosso backbone), [[perez_2018]] FiLM.
