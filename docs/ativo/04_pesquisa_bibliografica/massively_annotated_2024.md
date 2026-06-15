---
name: massively-annotated-2024
status_verificacao: VERIFIED
autores: [Pedro C. Neto (Univ. Porto + INESC TEC), Rafael M. Mamede (Univ. Porto + INESC TEC), Carolina Albuquerque (Univ. Porto + INESC TEC), Tiago Gonçalves (Univ. Porto + INESC TEC), Ana F. Sequeira (Univ. Porto + INESC TEC)]
ano: 2024
titulo: "Massively Annotated Datasets for Assessment of Synthetic and Real Data in Face Recognition"
venue: "arXiv preprint 2404.15234v1 (abr 2024) — Faculty of Engineering of the University of Porto + INESC TEC"
tipo_publicacao: preprint
arxiv_id: "2404.15234"
doi: null
url_primario: https://arxiv.org/abs/2404.15234
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado de arXiv (pdfs/massively_annotated_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract com MAC (Massive Attribute Classifier) e setup KL divergence lidos via pdftotext.
---

# Massively Annotated Datasets (Neto, Mamede, Albuquerque, Gonçalves & Sequeira — Univ. Porto 2024)

> **MAC (Massive Attribute Classifier)** para anotar 4 datasets de FR
> (2 reais + 2 sintéticos) e estudar drift entre modelos treinados em
> dados reais vs sintéticos. **Pesquisa portuguesa** — Univ. Porto +
> INESC TEC.

## 1-2. Resumo + método

- **Problema**: datasets reais de FR estão sendo **retirados de
  acesso público** (privacy/ethical concerns); GenAI fornece
  alternativa mas performance ainda não comparável a real.
- **Contribuição**:
  - **MAC** — Massive Attribute Classifier que anota datasets em
    larga escala.
  - **Anotação de 4 datasets**: 2 reais + 2 sintéticos.
  - **Análise de distribuição** de cada atributo cross 4 datasets.
  - **KL divergence** quantifica diferenças entre real e sintético.

## 3-5. Achados

- **KL divergence positiva**: diferenças sistemáticas entre real
  e synthetic em distribuição de atributos.
- Síntese ainda não captura plenamente diversidade de samples
  reais (especialmente em atributos demográficos).

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — synthetic data) — paralelo a
  [[synthetic_face_2024]], [[variface_2024]], [[frcsyn_2024]],
  [[fairimagen_neurips2025]], [[fairer_datasets_2024]].
- **Não usamos synthetic data** — operamos em FairFace (real).
- **Cap 2** (Revisão): citação para discussão sobre alternativas
  privacy-preserving de dados de treino.

## 8. Citar

- *"Neto, Mamede, Albuquerque, Gonçalves & Sequeira (arXiv
  2404.15234, 2024), pesquisadores da Faculdade de Engenharia da
  Universidade do Porto e do INESC TEC, apresentam o classificador
  MAC (Massive Attribute Classifier) para anotação automática em
  larga escala de datasets de face recognition, demonstrando via
  divergência de Kullback-Leibler que datasets sintéticos atuais
  ainda apresentam drift significativo em distribuição de atributos
  cross datasets reais — limitação relevante para a viabilidade
  prática de substituição completa por GenAI em FR fairness."*

## 9-12.

PDF: `pdfs/massively_annotated_2024.pdf`. Conexões:
[[synthetic_face_2024]], [[variface_2024]], [[frcsyn_2024]],
[[fairimagen_neurips2025]], [[fairer_datasets_2024]] (família
synthetic data — Track L).
