---
name: face4fairshifts-2025
status_verificacao: VERIFIED
autores: [Yumeng Lin (Tianjin University), Dong Li (Baylor), Xintao Wu (Univ Arkansas), Minglai Shao (Tianjin), Xujiang Zhao (NEC Labs America), Zhong Chen (Southern Illinois Univ), Chen Zhao (Baylor)]
ano: 2025
titulo: "Face4FairShifts: A Large Image Benchmark for Fairness and Robust Learning across Visual Domains"
venue: "arXiv preprint 2509.00658 (ago 2025)"
tipo_publicacao: preprint
arxiv_id: "2509.00658"
doi: null
url_primario: https://arxiv.org/abs/2509.00658
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado de arXiv (pdfs/face4fairshifts_2025.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução com taxonomia de shifts lidos via pdftotext.
---

# Face4FairShifts — Cross-Domain Fairness Benchmark (Lin, Li, Wu et al., arXiv 2025)

> **Benchmark large-scale (100K imagens) em 4 domínios visualmente
> distintos** com 42 anotações em 15 atributos demográficos e
> faciais. Para fairness-aware learning + domain generalization.

## 1-2. Resumo + composição

- **100.000 imagens** em **4 domínios visualmente distintos**.
- **42 anotações** em **15 atributos** cobrindo:
  - Demographic features
  - Facial features
- **Objetivo**: avaliar sistematicamente fairness-aware learning
  + domain generalization sob distribution shifts.
- **Taxonomia de shifts** identificada pelos autores:
  - Covariate shifts
  - Semantic shifts
  - Demographic shifts (marginal distribution de race/gender
    muda cross domains)
- Disponível em https://meviuslab.github.io/Face4FairShifts/.

## 3-5. Achados

- Análise via experimentos extensivos identifica **gaps
  significativos** de fairness sob distribution shifts.
- **Limitações de datasets existentes** reportadas — motivação
  para benchmark consolidado.
- Necessidade de técnicas mais efetivas de **fairness-aware domain
  adaptation**.

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — cross-domain) — benchmark possível para
  validação cross-domain.
- **Não usamos cross-domain** nesta dissertação — FairFace
  in-domain.
- **Cap 4** (Discussão): direção futura para generalização.

## 8. Citar

- *"Lin et al. (arXiv 2509.00658, 2025), em colaboração entre Tianjin
  University, Baylor, Arkansas e NEC Labs America, propõem
  Face4FairShifts, benchmark large-scale com 100 mil imagens em
  quatro domínios visualmente distintos anotadas em 42 atributos
  demográficos e faciais. O benchmark identifica taxonomia de shifts
  (covariate, semantic, demographic) e documenta gaps significativos
  de fairness em modelos sob distribution shifts — direção
  complementar para trabalho futuro em generalização cross-domain
  do classificador racial proposto nesta dissertação."*

## 9-12.

PDF: `pdfs/face4fairshifts_2025.pdf`. Conexões: [[fairdomain_2024]]
(cross-domain medical fairness — Track L), [[dataset_karkkainen_2021]]
(FairFace — in-domain), [[gras_2025]] (benchmark VLM).
