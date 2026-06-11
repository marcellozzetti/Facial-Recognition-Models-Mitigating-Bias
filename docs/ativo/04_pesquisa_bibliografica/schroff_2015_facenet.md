---
name: schroff-2015-facenet
status_verificacao: OVERVIEW_ONLY
autores: [Florian Schroff, Dmitry Kalenichenko, James Philbin]
ano: 2015
titulo: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2015)"
tipo_publicacao: conference
arxiv_id: "1503.03832"
doi: null
url_primario: https://arxiv.org/abs/1503.03832
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# FaceNet — Triplet Loss Embedding (Schroff et al., CVPR 2015)

> **Paper fundador moderno** de face recognition. Triplet loss
> introduz embeddings em hipersfera unitária para similaridade
> euclidiana. Base para todos os métodos margin-based posteriores
> (CosFace, ArcFace, AdaFace).

## 1-2. Resumo + método

- **Embedding f: imagem → ℝ^d** sobre hipersfera ‖f(x)‖₂ = 1.
- **Triplet loss**: (anchor, positive, negative) com margin α.
- ‖f(a) − f(p)‖² + α < ‖f(a) − f(n)‖².
- Aprende espaço onde distância euclidiana = similaridade facial.

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

PDF pendente. Fundador FR. Conexões: [[wang_2018_cosface]],
[[deng_2019_arcface]], [[dataset_wang_2019]] RFW, [[dataset_robinson_2020]] BFW.
