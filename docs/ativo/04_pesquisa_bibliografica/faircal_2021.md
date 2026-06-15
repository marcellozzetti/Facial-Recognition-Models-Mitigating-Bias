---
name: faircal-2021
status_verificacao: VERIFIED
autores: [Tiago Salvador, et al.]
ano: 2021
titulo: "FairCal: Fairness Calibration for Face Verification"
venue: "ICLR 2022 (a confirmar)"
tipo_publicacao: conference
arxiv_id: "2106.03761"
doi: null
url_primario: https://arxiv.org/abs/2106.03761
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/faircal_2021.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---


# FairCal — Fairness Calibration for Face Verification

> **Post-training calibration** — não requer retraining nem
> knowledge of sensitive attribute. Reduz gap FPR + aumenta accuracy.

## 1-2. Resumo

- **Calibração post-hoc** de probabilidades.
- Aumenta accuracy do modelo simultaneamente.
- **Reduz gap em FPR** entre grupos.
- **Não requer**: knowledge of sensitive attribute, retraining.

## 7. Aplicação ao pipeline v3.2

- **Post-processing alternativo** para Cap 2/3.
- Particularmente útil para Cap 3 (FR verification).
- Endereça quality-fairness trade-off de Hardt 2016 via calibration.

## 8. Citar

- *"FairCal (Salvador et al., arXiv:2106.03761) introduz fairness
  calibration post-training para face verification, demonstrando
  que aumentar accuracy e reduzir gap em false positive rates entre
  grupos demográficos é alcançável sem retraining nem conhecimento
  prévio do atributo sensível — abordagem complementar ao
  in-processing FiLM-conditioning desta dissertação."*

## 9-12.

PDF pendente. Conexões: [[hardt_2016]], [[deng_2019_arcface]].
