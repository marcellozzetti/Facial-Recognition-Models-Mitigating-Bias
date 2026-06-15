---
name: fairdomain-2024
status_verificacao: VERIFIED
autores: [Yu Tian (Harvard), Congcong Wen (NYU AD), Min Shi (Harvard), Muhammad Muneeb Afzal (NYU), Hao Huang (NYU AD), Muhammad Osama Khan (NYU), Yan Luo (Harvard), Yi Fang (NYU AD), Mengyu Wang (Harvard)]
ano: 2024
titulo: "FairDomain: Achieving Fairness in Cross-Domain Medical Image Segmentation and Classification"
venue: "arXiv preprint 2407.08813v2 (jul 2024) — Harvard Ophthalmology AI Lab + NYU Abu Dhabi (mesmo grupo do FairCLIP)"
tipo_publicacao: preprint
arxiv_id: "2407.08813"
doi: null
url_primario: https://arxiv.org/abs/2407.08813
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/fairdomain_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract com FIA module e dataset Harvard-FairDomain20k lidos via pdftotext.
---

# FairDomain — Cross-Domain Medical Fairness (Tian, Wen, Shi et al., arXiv 2024)

> **Pioneer study sobre fairness sob domain shifts em medical AI**.
> Mesmo grupo do [[luo_2024_fairclip]] (Harvard Ophthalmology AI Lab +
> NYU AD). Introduz **FIA module** (Fair Identity Attention),
> plug-and-play para DA/DG algorithms.

## 1-2. Resumo + método

- **Problema**: fairness em **domain transfer** em medical AI é
  inexplorado. Clínicas usam diferentes imaging technologies (e.g.,
  retinal imaging modalities) para mesmo diagnóstico.
- **Contribuições**:
  1. **Estudo sistemático** de fairness sob domain shifts com
     algoritmos SOTA de domain adaptation (DA) e generalization (DG).
  2. **FIA module (Fair Identity Attention)** — plug-and-play module
     usando self-attention para ajustar feature importance baseado em
     atributos demográficos.
  3. **Harvard-FairDomain20k** — primeiro dataset com 2 imaging
     modalities pareadas para o mesmo paciente em segmentation e
     classification.

## 3-5. Resultados

- **FIA significativamente melhora performance + fairness** em todos
  os settings de domain shift (DA e DG).
- Supera métodos existentes em segmentation e classification.
- Code/data: https://ophai.hms.harvard.edu/datasets/harvard-fairdomain20k

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — cross-domain) — paralelo a
  [[face4fairshifts_2025]].
- **Não médico** — nossa tese é face recognition.
- **Conexão técnica relevante**: FIA usa self-attention sobre
  features demográficas — paradigma análogo ao FiLM (modulação
  feature-wise condicionada em demographic info).
- **Cap 2** (Revisão): citação para "demographic-aware attention"
  como mecanismo paralelo ao FiLM.

## 8. Citar

- *"Tian, Wen, Shi et al. (arXiv 2407.08813, 2024), em colaboração
  Harvard Ophthalmology AI Lab + NYU Abu Dhabi (mesmo grupo do
  FairCLIP de Luo et al.), apresentam FairDomain — estudo pioneer
  sobre fairness sob domain shifts em medical AI. Introduzem o
  módulo plug-and-play Fair Identity Attention (FIA) que usa
  self-attention para ajustar feature importance baseada em
  atributos demográficos, mecanismo conceitualmente análogo ao
  FiLM-conditioning aplicado nesta dissertação."*

## 9-12.

PDF: `pdfs/fairdomain_2024.pdf`. Dataset: Harvard-FairDomain20k.
Conexões: [[luo_2024_fairclip]] (mesmo grupo Harvard + NYU AD),
[[face4fairshifts_2025]] (cross-domain fairness paralelo),
[[perez_2018]] (FiLM — paradigma conceitualmente similar via
self-attention vs feature-wise affine).
