---
name: lin-2025-aiface
status_verificacao: VERIFIED
autores: [Li Lin, Santosh, Mingyang Wu, Xin Wang, Shu Hu]
ano: 2025
titulo: "AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)"
tipo_publicacao: conference
arxiv_id: "2406.00783"
doi: null
url_primario: https://arxiv.org/abs/2406.00783
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/lin_2025_aiface.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---

> Construída a partir de abstract e busca web. PDF integral pendente.

# AI-Face — Million-Scale Demographically Annotated Dataset (CVPR 2025)

> **Primeiro dataset million-scale** demograficamente anotado de
> faces AI-generated (GAN + Diffusion + deepfake) + benchmark de
> fairness para detecção. Relevante como **dataset alternativo** e
> evidência da expansão de fairness para AI-generated content.

## 1. Resumo

> *Fonte: abstract + busca web.*

Detecção de AI-generated faces (GAN, Diffusion, deepfakes) é
crítica para integridade midiática. Mas **fairness em detectores**
de AI-generated content é problema sub-explorado. AI-Face fornece:
- Million-scale dataset.
- Anotação demográfica detalhada.
- Faces reais + GAN + Diffusion + deepfake.
- Benchmark de fairness para classificadores de "real vs AI-generated".

## 2. Método (inferido)

> **[PENDENTE PDF]** Inferência:
> - Coleta de faces de múltiplas fontes (real + sintéticas).
> - Anotação demográfica (race, gender, age, possibly skin tone).
> - Protocolo de benchmark fairness.

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Não extraídos.

## 7. Aplicação ao nosso pipeline v3.2

### 7.1 Dataset complementar (não substitui FairFace)

AI-Face é para **detecção de deepfakes**, não classification racial.
Não compete com FairFace para nossa pesquisa.

### 7.2 Relevância contextual

- Demonstra que **viés racial em AI-generated detection** é problema
  ativo em 2025.
- Reforça narrativa do Cap 1 (Introdução): viés facial é tema
  abrangente além de race classification clássica.

### 7.3 Possível extensão

Se tempo permitir após qualificação: testar pipeline (Cap 2) em
AI-Face para verificar generalização.

## 8. Pontos para citar

- *"Lin et al. (2025), apresentado no CVPR 2025, introduzem AI-Face —
  primeiro dataset million-scale de faces AI-generated com anotação
  demográfica detalhada, junto com benchmark de fairness para
  detecção. Esta cobertura demográfica em escala industrial
  consolida fairness em facial vision como linha de pesquisa
  ativa em 2025."*

## 9-12. Arquivos e análise crítica

- PDF: pendente em `pdfs/lin_2025_aiface.pdf`.
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace), 
  [[dataset_wang_2019]] (RFW).

> **[BLOQUEADO]** Análise crítica detalhada requer leitura integral.
