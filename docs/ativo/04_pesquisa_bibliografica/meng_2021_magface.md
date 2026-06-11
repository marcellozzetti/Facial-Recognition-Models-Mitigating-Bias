---
name: meng-2021-magface
status_verificacao: OVERVIEW_ONLY
autores: [Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou]
ano: 2021
titulo: "MagFace: A Universal Representation for Face Recognition and Quality Assessment"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021 — Oral)"
tipo_publicacao: conference
arxiv_id: "2103.06627"
doi: null
url_primario: https://arxiv.org/abs/2103.06627
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# MagFace — Universal Embedding with Quality (Meng et al., CVPR 2021 Oral)

> **Magnitude do embedding = qualidade da imagem**. Predecessor
> direto do AdaFace. Aprende within-class feature distribution
> bem-estruturada via mecanismo adaptativo.

## 1-2. Resumo + método

- **Loss**: aprende embeddings cuja **magnitude mede qualidade** da face.
- **Mecanismo adaptativo**: melhor estruturação within-class.
- **Aplicação dupla**: face recognition + face quality assessment
  (FQA) em modelo único.

## 7. Aplicação ao pipeline v3.2

- **Predecessor histórico do AdaFace**.
- Quality-aware sem módulo dedicado — análogo conceitual ao FiLM
  que injeta MST sem rede dedicada por estágio.
- **Baseline FR opcional** para Cap 3.

## 8. Citar

- *"MagFace (Meng et al., CVPR 2021 Oral) estabeleceu que a
  magnitude do embedding pode ser usada como proxy de qualidade
  da imagem, aprendendo simultaneamente representação para face
  recognition e quality assessment. Esta abordagem precede AdaFace
  (Kim et al., 2022) e consolida quality-awareness como dimensão
  necessária em face recognition robusto."*

## 9-12.

PDF pendente. Conexões: [[deng_2019_arcface]], [[kim_2022_adaface]].
