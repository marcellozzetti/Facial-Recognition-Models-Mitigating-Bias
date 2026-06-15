---
name: mllm-face-verification-2026
status_verificacao: VERIFIED
autores: [a verificar]
ano: 2026
titulo: "Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification"
venue: "arXiv preprint (mar 2026)"
tipo_publicacao: preprint
arxiv_id: "2603.25613"
doi: null
url_primario: https://arxiv.org/abs/2603.25613
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/mllm_face_verification_2026.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---


# Demographic Fairness in MLLMs (March 2026)

> **Benchmark de 9 MLLMs (2B-8B params)** em IJB-C + RFW através
> de **4 etnias × 2 gêneros**. Achado: **mais accurate ≠ mais fair**.

## 1-2. Resumo

- **9 MLLMs open-source** avaliados em FR via visual prompting.
- Datasets: **IJB-C e RFW**.
- **4 grupos étnicos × 2 gêneros**.
- **Achado central**: modelos mais accurate não são necessariamente
  mais fair; modelos com accuracy pobre podem **parecer fair**
  porque produzem erros uniformemente altos.

## 7. Aplicação ao pipeline v3.2

- **Track I (VLM fairness)** evidência empírica recente (2026).
- Para Cap 1: argumento para abandonar single-metric evaluation.
- Reforça **triangulação** (nossa Contribuição C4): não basta uma
  métrica.

## 8. Citar

- *"Estudo recente de demographic fairness em multimodal LLMs
  (arXiv:2603.25613, março/2026) avaliou 9 modelos open-source de
  2B a 8B parâmetros sobre IJB-C e RFW, demonstrando empíricamente
  que os modelos mais accurate não são necessariamente os mais
  fair, e que modelos com performance overall pobre podem aparentar
  fairness via uniform high error rates. Esta observação reforça a
  necessidade de triangulação de métricas (Disparity Ratio +
  worst-class F1 + métricas de Hardt) adotada nesta dissertação."*

## 9-12.

PDF pendente. Conexões: [[hardt_2016]], [[aldahoul_2024]],
[[kleinberg_2017]].
