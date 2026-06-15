---
name: schroff-2015-facenet
status_verificacao: VERIFIED
autores: [Florian Schroff, Dmitry Kalenichenko, James Philbin]
ano: 2015
titulo: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2015)"
tipo_publicacao: conference
arxiv_id: "1503.03832"
doi: null
url_primario: https://arxiv.org/abs/1503.03832
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/schroff_2015_facenet.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução e seções de método/resultados lidos via pdftotext.
---

# FaceNet — Triplet Loss Embedding (Schroff et al., CVPR 2015)

> **Paper fundador moderno** de face recognition. Triplet loss
> introduz embeddings em hipersfera unitária para similaridade
> euclidiana. Base para todos os métodos margin-based posteriores
> (CosFace, ArcFace, AdaFace).

## 1-2. Resumo + método

- **Embedding f: imagem → ℝ^d** sobre hipersfera unitária ‖f(x)‖₂ = 1.
- **Triplet loss**: (anchor, positive, negative) com margin α.
- ‖f(a) − f(p)‖² + α < ‖f(a) − f(n)‖².
- Aprende espaço onde **distância euclidiana = similaridade facial**.
- **128 bytes per face** — extremamente compacto.
- **Online triplet mining** novel para gerar pares balanceados durante
  o treino.

## 3-5. Datasets e resultados

- **LFW (Labeled Faces in the Wild)**: **99.63 % accuracy** —
  novo recorde absoluto na época (cuts error 30 % vs SOTA anterior).
- **YouTube Faces DB**: 95.12 %.
- Treino em dataset Google interno (centenas de milhões de faces).
- Hipóteses para verificação, recognition (k-NN), clustering
  (k-means/agglomerative) — tudo via mesmo embedding.

## 7. Aplicação ao pipeline v3.2

- **Referência fundadora** para Cap 1 (Introdução) e Cap 3 (FR).
- **Cap 3**: nossos baselines de FR (RFW/BFW) operam em embedding
  space — paradigma de FaceNet.
- **Triplet loss**: alternativa a softmax para face verification.

## 8. Citar

- *"O paradigma moderno de face recognition foi estabelecido por
  Schroff, Kalenichenko & Philbin (CVPR 2015, FaceNet), que
  introduziu embeddings em hipersfera unitária via triplet loss,
  permitindo verificação facial via distância euclidiana no
  espaço aprendido. Esta arquitetura é a base para todos os
  métodos margin-based posteriores (CosFace, ArcFace, AdaFace)."*

## 9-12.

Arquivo PDF: `pdfs/schroff_2015_facenet.pdf`. Fundador moderno do FR
deep-learning. Conexões: [[wang_2018_cosface]] (substituição da
triplet por margin softmax), [[deng_2019_arcface]] (angular margin),
[[meng_2021_magface]] (quality + magnitude),
[[kim_2022_adaface]] (quality-adaptive),
[[dataset_wang_2019]] RFW, [[dataset_robinson_2020]] BFW.
