---
name: provable-adversarial-ssl-2024
status_verificacao: VERIFIED
autores: [Qi Qi (University of Iowa), Quanqi Hu (Texas A&M), Qihang Lin (University of Iowa), Tianbao Yang (Texas A&M)]
ano: 2024
titulo: "Provable Optimization for Adversarial Fair Self-supervised Contrastive Learning"
venue: "arXiv preprint (jun 2024) — University of Iowa + Texas A&M"
tipo_publicacao: preprint
arxiv_id: "2406.05686"
doi: null
url_primario: https://arxiv.org/abs/2406.05686
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/provable_adversarial_ssl_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract lido via pdftotext.
---

# Provable Adversarial Fair SSL Contrastive (2024)

> **Garantias formais** de otimização para fair SSL contrastive.
> Extensão teórica para [[ramachandran_2024]] (SSL fair empírico).

## 1-2. Resumo + método

- **Fair encoders em setting SSL**: todos os dados unlabeled, apenas
  pequena porção anotada com atributo sensível.
- **Adversarial fair representation learning** ideal para este
  cenário: minimiza contrastive loss sobre unlabeled data E maximiza
  adversarial loss de predição do atributo sensível sobre data com
  atributo.
- **Desafio matemático**: non-convex non-concave minimax game.
- **Contribuição**: garantias formais de otimização (provable
  convergence) para este minimax problem.
- Generaliza Park 2022 (FSCL+) para SSL setting com bases teóricas
  rigorosas.

## 7. Aplicação ao pipeline v3.2

- **Track de SSL fair**: alternativa ao nosso supervised setup.
- Para Cap 2 como **baseline SSL avançado** (junto com
  ramachandran_2024).
- Importante para responder "por que supervised e não SSL?".

## 8. Citar

- *"Trabalhos recentes em fair self-supervised contrastive learning
  com garantias formais de otimização (2024, arXiv:2406.05686)
  estendem o paradigma supervisedo de Park et al. (FSCL+, CVPR
  2022) para settings auto-supervisionados. A presente dissertação
  adota o paradigma supervised pelo acesso a race labels do
  FairFace, registrando SSL como alternativa relevante quando
  labels demográficos não estão disponíveis."*

## 9-12.

PDF pendente. Conexões: [[park_2022]] FSCL+, [[ramachandran_2024]] SSL.
