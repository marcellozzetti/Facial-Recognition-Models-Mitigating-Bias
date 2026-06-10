---
name: reliable-demo-inference-2025
status_verificacao: OVERVIEW_ONLY
autores: [a verificar]
ano: 2025
titulo: "Reliable and Reproducible Demographic Inference for Fairness in Face Analysis"
venue: "arXiv preprint (out 2025)"
tipo_publicacao: preprint
arxiv_id: "2510.20482"
doi: null
url_primario: https://arxiv.org/abs/2510.20482
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# Reliable & Reproducible Demographic Inference (2025)

> **Modular DAI pipeline** (Demographic Attribute Inference) usando
> pre-trained FR encoders + non-linear classification heads.
> **Intra-identity consistency** como dimensão de robustez.
> Diretamente relevante para nossa Etapa 1 (audit FairFace).

## 1-2. Resumo + método

- Fairness eval em face analysis depende de DAI confiável.
- **Pipeline modular**:
  - Encoder: pre-trained FR (ArcFace?).
  - Head: non-linear classification.
- **3 dimensões de audit**: accuracy + fairness + robustness (via
  intra-identity consistency).

## 7. Aplicação ao pipeline v3.2

- **Diretamente relevante** para nossa Etapa 2 (auditar FairFace).
- Sugere protocolo de validação MST classifier robusto.
- **Intra-identity consistency**: se aplicarmos SkinToneNet sobre
  múltiplas imagens da mesma identidade no FairFace, devemos
  esperar consistência.

## 8. Citar

- *"Pesquisa recente sobre Demographic Attribute Inference (DAI)
  para auditoria de fairness em face analysis (2025,
  arXiv:2510.20482) estabelece pipeline modular com encoder de
  face recognition pré-treinado e head não-linear, auditado em
  três dimensões: accuracy, fairness e intra-identity consistency.
  A última dimensão é particularmente relevante para validação do
  classificador de skin tone no Capítulo 1 desta dissertação."*

## 9-12.

PDF pendente.
