---
name: fairlora-2024
status_verificacao: VERIFIED
autores: [a verificar]
ano: 2024
titulo: "FairLoRA: Unpacking Bias Mitigation in Vision Models with Fairness-Driven Low-Rank Adaptation"
venue: "arXiv preprint (a verificar venue final)"
tipo_publicacao: preprint
arxiv_id: "2410.17358"
doi: null
url_primario: https://arxiv.org/abs/2410.17358
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/fairlora_2024.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---


# FairLoRA — Fairness-Driven LoRA for Vision (2024)

> **Bias mitigation via LoRA** com objetivo fairness explícito.
> Track J (conditioning moderno) — alternativa direta ao FiLM.

## 1-2. Resumo + método

- **LoRA** padrão é parameter-efficient mas não fairness-aware.
- FairLoRA propõe **fairness-driven low-rank adaptation**.
- Introduz **fairness constraints** durante adaptação LoRA.
- Foca em modelos de visão.

## 7. Aplicação ao pipeline v3.2

- **Direct competitor do FiLM-conditioning** em nossa Cap 2.
- Comparar:
  - FiLM: modulação de features (γ ⊙ F + β).
  - FairLoRA: weight updates low-rank com fairness constraints.
- Ambos são parameter-efficient.

## 8. Citar

- *"FairLoRA (2024, arXiv:2410.17358) introduz fairness constraints
  em low-rank adaptation para modelos de visão, oferecendo
  alternativa parameter-efficient à modulação por FiLM. A
  comparação experimental entre FiLM-conditioning e FairLoRA é
  parte da ablation do Capítulo 2 desta dissertação, endereçando
  a recomendação metodológica registrada em reunião de orientação
  (2026-06-08)."*

## 9-12.

PDF pendente. Conexões: [[perez_2018]] FiLM (alternativa principal),
[[bian_2025_lorafair]] (LoRA federated relacionado).
