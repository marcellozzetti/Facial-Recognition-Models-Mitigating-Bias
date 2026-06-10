---
name: bendvlm-2024
status_verificacao: OVERVIEW_ONLY
autores: [a verificar]
ano: 2024
titulo: "BendVLM: Test-Time Debiasing of Vision-Language Embeddings"
venue: "arXiv preprint (a verificar venue final)"
tipo_publicacao: preprint
arxiv_id: "2411.04420"
doi: null
url_primario: https://arxiv.org/abs/2411.04420
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# BendVLM — Test-Time Debiasing VL Embeddings (2024)

> **Debiasing em inference time** — sem retraining. Track I.
> Endereça custo computacional de FairCLIP-style fine-tuning.

## 1-2. Resumo + método

- **Test-time intervention** em VL embeddings.
- Sem retraining (vantagem operacional sobre FairCLIP).
- Provavelmente baseado em projeção de embeddings para subespaço
  "neutro" demograficamente.

## 7. Aplicação ao pipeline v3.2

- **Track I** — baseline de baixo custo computacional.
- Para nossa Cap 2 ablation: BendVLM vs FairCLIP vs FiLM-conditioning.
- Útil para argumentar custo-benefício do nosso FiLM (que também
  é parameter-efficient).

## 8. Citar

- *"Abordagens de debiasing em test-time como BendVLM (2024,
  arXiv:2411.04420) oferecem alternativa de baixo custo computacional
  ao fine-tuning baseado em FairCLIP, ao custo de generalização
  potencialmente menos robusta. Esta dimensão de custo-benefício
  é parte da comparação metodológica do Capítulo 2 desta
  dissertação."*

## 9-12.

PDF pendente. Conexões: [[luo_2024_fairclip]], [[dehdashtian_2024_fairerclip]].
