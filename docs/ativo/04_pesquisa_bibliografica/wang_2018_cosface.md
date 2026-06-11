---
name: wang-2018-cosface
status_verificacao: OVERVIEW_ONLY
autores: [Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu]
ano: 2018
titulo: "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018), pp. 5265-5274"
tipo_publicacao: conference
arxiv_id: "1801.09414"
doi: null
url_primario: https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf
citacoes_semantic_scholar: 3212
data_verificacao_citacoes: 2026-06-10
n_paginas: 10
lente_disrupcao: paradigma
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# CosFace — Large Margin Cosine Loss (Wang et al., CVPR 2018)

> **Reformulação do softmax como cosine loss** com L2 normalization +
> margin term aditivo no cosseno. Predecessor direto do ArcFace.
> >3000 citações.

## 1-2. Resumo + método

- **Problema**: softmax tradicional lacks discriminative power para
  face recognition.
- **LMCL (Large Margin Cosine Loss)**: reformula softmax como cosine
  loss via L2 normalization de features e weights.
- Margin term aditivo no cosseno: cos(θ) − m em vez de cos(θ + m).
- **Resultado**: maximiza inter-class variance, minimiza intra-class.

## 7. Aplicação ao pipeline v3.2

- **Loss alternativo** para Cap 3 (FR baselines).
- Para classification (Cap 2): adaptável mas não testado em race
  classification 7-class.
- **Referência histórica** entre FaceNet 2015 → CosFace 2018 →
  ArcFace 2019.

## 8. Citar

- *"CosFace (Wang et al., CVPR 2018) reformulou a softmax tradicional
  como large margin cosine loss via L2 normalization, estabelecendo
  o paradigma de margin-based losses que dominou face recognition
  pós-2018 e culminou em ArcFace (Deng et al., 2019) como referência
  canônica."*

## 9-12.

PDF pendente. Conexões: [[schroff_2015_facenet]], [[deng_2019_arcface]].
