---
name: score-normalization-2024
status_verificacao: OVERVIEW_ONLY
autores: [a verificar]
ano: 2024
titulo: "Score Normalization for Demographic Fairness in Face Recognition"
venue: "IEEE Winter Conference on Applications of Computer Vision (WACV 2024)"
tipo_publicacao: conference
arxiv_id: "2407.14087"
doi: null
url_primario: https://arxiv.org/abs/2407.14087
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# Score Normalization for Demographic Fairness (WACV 2024)

> **Regularized score calibration** post-hoc. Computa **local
> thresholds** com clusters de identidades similares. Optimal
> threshold varia 0.3-0.7 por cluster.

## 1-2. Resumo

- **Calibração regularizada** post-training.
- **Local thresholds** vs single global threshold.
- Identifica clusters de identidades similares.
- Achado: optimal threshold varia significativamente (0.3-0.7).

## 7. Aplicação ao pipeline v3.2

- **Alternativa ao threshold global** em Cap 3 (FR).
- Conceitualmente alinhado com [[dataset_robinson_2020]] BFW
  (thresholds adaptativos per-subgroup).
- Para Cap 3: comparação contra abordagem global.

## 8. Citar

- *"Score normalization regularizada (WACV 2024, arXiv:2407.14087)
  demonstra que thresholds locais por cluster de identidades
  similares variam significativamente entre 0.3 e 0.7, sustentando
  empíricamente a necessidade de abandonar threshold único global
  em face recognition fairness-aware."*

## 9-12.

PDF pendente. Conexões: [[dataset_robinson_2020]] BFW, [[faircal_2021]].
