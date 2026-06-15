---
name: post-comparison-2020
status_verificacao: VERIFIED
autores: [Philipp Terhörst (Fraunhofer IGD + TU Darmstadt), Jan Niklas Kolf (TU Darmstadt), Naser Damer (Fraunhofer IGD), Florian Kirchbuchner (Fraunhofer IGD + TU Darmstadt), Arjan Kuijper (Fraunhofer IGD + TU Darmstadt)]
ano: 2020
titulo: "Post-comparison mitigation of demographic bias in face recognition using fair score normalization"
venue: "arXiv preprint 2002.03592v3 (nov 2020) — Fraunhofer Institute for Computer Graphics Research IGD + TU Darmstadt"
tipo_publicacao: preprint
arxiv_id: "2002.03592"
doi: null
url_primario: https://arxiv.org/abs/2002.03592
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/post_comparison_2020.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e motivação (GDPR/EU Convention) lidos via pdftotext.
---

# Post-Comparison Fair Score Normalization (Terhörst, Kolf, Damer et al. — Fraunhofer IGD + TU Darmstadt, 2020)

> **Post-processing unsupervised fair score normalization** para
> reduzir viés em FR. Pioneiro em mitigação **pós-comparison** sem
> degradar performance overall — supera abordagens de "less-biased
> representation".

## 1-2. Resumo + método

- **Problema**: trabalhos anteriores focaram em **learning
  less-biased representations**, que vem ao custo de **performance
  overall degradada** + **template-replacement caro** se sistema é
  atualizado.
- **Contribuição**: **fair score normalization** — abordagem
  unsupervised específica para reduzir bias em FR e
  simultaneamente **boost performance overall**.
- **Hipótese**: notion de **individual fairness** — designing
  normalization para tratar similares similarmente.
- **Motivação legal**: GDPR (art. 71) + EU Convention on Human
  Rights (right to non-discrimination).

## 3-5. Resultados

- **Achado central**: post-processing **boost overall performance**
  enquanto **reduz bias** — contrário ao trade-off típico do campo.
- Aplicável a sistemas existentes sem retreinamento.

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — post-hoc / calibration) — paralelo a
  [[faircal_2021]], [[score_normalization_2024]], [[fair_sight_2025]].
- **Não usamos post-processing** — operamos em classification
  com FiLM-conditioning durante treino.
- **Cap 2** (Revisão): citação para "post-processing alternatives"
  em pipeline de FR fairness.

## 8. Citar

- *"Terhörst, Kolf, Damer, Kirchbuchner & Kuijper (arXiv
  2002.03592, 2020), do Fraunhofer Institute for Computer Graphics
  Research IGD em colaboração com a TU Darmstadt, propõem
  abordagem unsupervised de fair score normalization aplicada
  pós-comparison em face recognition, demonstrando que mitigação
  pode ser obtida sem o trade-off típico de degradação de
  performance overall — alternativa metodológica relevante a
  pipelines com conditioning em treino, como o adotado nesta
  dissertação."*

## 9-12.

PDF: `pdfs/post_comparison_2020.pdf`. Conexões: [[faircal_2021]]
(post-hoc calibration), [[score_normalization_2024]],
[[fair_sight_2025]] (família Track L — post-hoc/calibration),
[[hardt_2016]] (post-processing como categoria fundadora).
