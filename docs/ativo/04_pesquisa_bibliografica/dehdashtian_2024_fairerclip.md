---
name: dehdashtian-2024-fairerclip
status_verificacao: VERIFIED
autores: [Sepehr Dehdashtian, Lan Wang, Vishnu Naresh Boddeti]
ano: 2024
titulo: "FairerCLIP: Debiasing CLIP's Zero-Shot Predictions using Functions in RKHSs"
venue: "International Conference on Learning Representations (ICLR 2024)"
tipo_publicacao: conference
arxiv_id: "2403.15593"
doi: null
url_primario: https://arxiv.org/abs/2403.15593
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/dehdashtian_2024_fairerclip.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---


# FairerCLIP — Debiasing CLIP via RKHSs (Dehdashtian et al., ICLR 2024)

> **Debiasing zero-shot CLIP** via funções em Reproducing Kernel
> Hilbert Spaces. Track I (VLM fairness) — alternativa metodológica
> ao FairCLIP (Luo 2024).

## 1-2. Resumo + método

- **Debiasing pós-treino** de CLIP, foco em zero-shot.
- Funciona via funções em **RKHS** (Reproducing Kernel Hilbert Spaces).
- Mesmo autores do U-FaTE ([[dehdashtian_2024]]) — **autoria
  importante de reconhecer**.

## 7. Aplicação ao pipeline v3.2

- **Track I** complementar a FairCLIP (Luo 2024).
- Para nossa Cap 2 ablation FiLM vs CLIP-conditioning, FairerCLIP
  oferece versão zero-shot, sem fine-tuning custoso.
- Reportar como **baseline mais leve** que FairCLIP.

## 8. Citar

- *"Dehdashtian, Wang e Boddeti (2024), apresentado no ICLR 2024,
  propõem FairerCLIP — debiasing zero-shot de CLIP via funções em
  espaços de Hilbert com kernel reprodutor. O paradigma RKHS
  oferece alternativa metodológica leve ao optimal transport
  baseado em Sinkhorn proposto por Luo et al. (2024, FairCLIP)
  para o mesmo propósito."*

## 9-12.

PDF pendente. Conexões: [[luo_2024_fairclip]] (FairCLIP),
[[dehdashtian_2024]] (U-FaTE mesmos autores), [[aldahoul_2024]].

> **[BLOQUEADO]** Análise crítica completa requer PDF.
