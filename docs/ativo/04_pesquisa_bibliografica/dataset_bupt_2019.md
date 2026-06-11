---
name: dataset-bupt-2019
status_verificacao: OVERVIEW_ONLY
autores: [Mei Wang, Weihong Deng, et al.]
ano: 2019
titulo: "BUPT-Balancedface: A Large-Scale Race-Balanced Face Dataset"
venue: "BUPT (Beijing University of Posts and Telecommunications) — público para pesquisa"
tipo_publicacao: dataset
arxiv_id: null
doi: null
url_primario: http://www.whdeng.cn/RFW/index.html
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: Busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF/site pendente.

# BUPT-Balancedface (Wang & Deng, 2019)

> **1.3M imagens** em 28.000 indivíduos, **igualmente balanceado**
> entre 4 raças (African, Asian, Caucasian, Indian). Dataset de
> **treino** para FR fairness — complementar ao RFW (teste).

## 1-2. Resumo

- **1.3 milhões de imagens** de 28.000 indivíduos.
- **4 grupos demográficos**: African, Asian, Caucasian, Indian.
- **Balanceamento estrito**: cada raça com mesmo número de
  identidades e imagens.
- **Propósito**: treino balanceado para FR (RFW é teste).

## 7. Aplicação ao pipeline v3.2

- **Dataset de treino alternativo** para Cap 3 (FR baselines).
- Permite **treinar fair** vs FairFace que tem 7-class.
- Útil para audit "balanced training não basta" — Kolla 2022.

## 8. Citar

- *"BUPT-Balancedface (Wang & Deng, 2019) fornece 1.3 milhões de
  imagens estritamente balanceadas entre 4 grupos raciais (African,
  Asian, Caucasian, Indian), constituindo conjunto de treino
  padrão para fairness em face recognition em complementaridade
  com o RFW como conjunto de teste."*

## 9-12.

PDF pendente. Conexões: [[dataset_wang_2019]] RFW, [[kolla_2022]].
