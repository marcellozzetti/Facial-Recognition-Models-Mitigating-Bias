---
name: rethinking-assumptions-2021
status_verificacao: VERIFIED
autores: [J. Alex Hanson, Chris Dulhanty, Alexander Wong, et al.]
ano: 2021
titulo: "Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets"
venue: "IEEE/CVF International Conference on Computer Vision Workshops (ICCV 2021)"
tipo_publicacao: conference
arxiv_id: "2109.03229"
doi: null
url_primario: https://arxiv.org/abs/2109.03229
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/rethinking_assumptions_2021.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e Section 4.2 lidos via pdftotext.
---

# Rethinking Common Assumptions FR Datasets (ICCV 2021)

> **Crítica empírica a duas premissas comuns**: (1) datasets
> balanceados por raça produzem menos viés; (2) treinar apenas em
> uma raça é intrinsecamente prejudicial. **Os autores refutam
> ambas com experimentos**.

## 1-2. Resumo + achados centrais

- **Refutação 1**: treinar **apenas com faces africanas** induziu
  **menos viés** que treinar com distribuição balanceada.
- **Refutação 2**: distribuições **enviesadas para incluir mais faces
  africanas** produziram modelos **mais equitativos** que datasets
  balanceados (Figura 1 do paper).
- **Achado complementar**: adicionar mais imagens de **identidades
  já existentes** pode aumentar accuracy cross-race mais que
  adicionar **novas identidades**.
- Crítica a BUPT-Balancedface/RFW (Wang 2019) e FairFace
  (Kärkkäinen 2021) — premissas centrais dos seus designs questionadas.

## 7. Aplicação ao pipeline v3.2

- **Suporta H1** da tese: balanceamento simples não é suficiente.
- **Reforça motivação** para mitigação algorítmica (FiLM-conditioning)
  além de balanceamento de dados.
- **Cap 1** (Introdução): citação direta para fundamentar narrativa
  "balanceamento não basta".
- **Limitação**: focados em verification (FR), não em classification
  7-class — generalização indireta.

## 8. Citar

- *"Hanson et al. (ICCV 2021 Workshops, arXiv:2109.03229)
  demonstram empiricamente que duas premissas comuns sobre datasets
  de FR não se sustentam: distribuições enviesadas para incluir mais
  faces africanas produzem modelos mais equitativos que distribuições
  balanceadas, e adicionar mais imagens de identidades existentes
  pode superar adicionar novas identidades em ganho de equidade.
  Este achado fortalece o argumento de que mitigação algorítmica é
  necessária além de balanceamento de dados — direção adotada na
  presente dissertação."*

## 9-12.

PDF: `pdfs/rethinking_assumptions_2021.pdf`. Código:
github.com/j-alex-hanson/rethinking-race-face-datasets.
Conexões: [[dataset_wang_2019]] (RFW — premissa criticada),
[[dataset_karkkainen_2021]] (FairFace — premissa criticada),
[[pangelinan_2023]] (também conclui que balanceamento não basta),
[[kolla_2022]] (impacto da distribuição racial).
