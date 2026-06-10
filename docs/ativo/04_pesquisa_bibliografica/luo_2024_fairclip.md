---
name: luo-2024-fairclip
status_verificacao: OVERVIEW_ONLY
autores: [Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, Yi Fang, Mengyu Wang]
ano: 2024
titulo: "FairCLIP: Harnessing Fairness in Vision-Language Learning"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)"
tipo_publicacao: conference
arxiv_id: "2403.19949"
doi: null
url_primario: https://arxiv.org/abs/2403.19949
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Apenas abstract via arXiv + extração de resultados via WebSearch. PDF integral pendente.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**
>
> Construída a partir do abstract + sumarização via busca web.
> PDF integral pendente; promover a VERIFIED requer leitura completa.

# FairCLIP — Harnessing Fairness in Vision-Language Learning (Luo et al., CVPR 2024)

> **Primeiro fair vision-language medical dataset** (Harvard-FairVLMed)
> + método baseado em **optimal transport / Sinkhorn distance** para
> alinhar distribuições demográficas. Documenta vieses substanciais em
> CLIP e BLIP-2. Diretamente relevante para nossa Contribuição C7
> (comparativo FiLM vs CLIP-conditioning).

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

Fairness em vision-language models (VLMs) é problema crítico em
healthcare — modelos influenciam diagnóstico e tratamento. Embora
fairness tenha sido investigado no domínio vision-only, fairness em
**medical VL models** permanece inexplorada por escassez de datasets
médicos VL para estudar fairness.

## 2. Método

### 2.1 Dataset Harvard-FairVLMed

- **Primeiro fair vision-language medical dataset**.
- Atributos demográficos detalhados.
- Ground-truth labels.
- Clinical notes.

### 2.2 FairCLIP — abordagem central

- **Optimal-transport-based approach**.
- Reduz **Sinkhorn distance** entre:
  - Distribuição amostral overall.
  - Distribuições por grupo demográfico.
- Trade-off favorável performance ↔ fairness.

### 2.3 Análise de fairness em CLIP e BLIP-2

Auditoria comprehensiva sobre 4 atributos protegidos:
- Race
- Gender
- Ethnicity
- Language

**Achados qualitativos**: vieses significativos em todos os VLMs;
**grupos preferidos**: Asian, Male, Non-Hispanic, Spanish.

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Números específicos do trade-off. Magnitude da
> redução de bias por Sinkhorn alignment. Comparação contra outras
> abordagens (adversarial, contrastive).

## 7. Aplicação ao nosso pipeline v3.2

### 7.1 Centralidade para Contribuição C7

C7 (comparativo FiLM vs CLIP-conditioning) requer entender:
- Como CLIP pode ser tornado fair (FairCLIP fornece o método).
- Sinkhorn distance como alternativa ao FiLM-conditioning.

### 7.2 Aplicação no nosso pipeline

1. **Como baseline alternativo**: ConvNeXt-T + FiLM (nossa proposta)
   vs CLIP + FairCLIP fine-tuning (Luo et al. 2024) em race
   classification FairFace.
2. **Sinkhorn como métrica auxiliar**: além de DR + worst-class F1,
   podemos reportar Sinkhorn distance entre distribuições raciais.
3. **Dataset diferente do nosso**: Harvard-FairVLMed é médico
   ophthalmology. **Não compete** com FairFace.

### 7.3 Risco identificado

CLIP pré-treinado tem **14% misclass rate em faces Black** vs <8%
em outros grupos. FairCLIP mitiga, mas o baseline CLIP herda esse
viés. Para nossa comparação justa em race classification, precisamos
fine-tunar CLIP em FairFace antes da ablation.

## 8. Pontos para citar

- *"Luo et al. (2024), em paper publicado no CVPR 2024, introduzem
  FairCLIP — primeira abordagem baseada em optimal transport para
  fairness em vision-language models — junto com o dataset
  Harvard-FairVLMed, primeiro benchmark VL médico com atributos
  demográficos. A análise comparativa de CLIP e BLIP-2 documenta
  vieses substanciais em ambos os modelos cross-attribute."*

- *"A abordagem FairCLIP de Luo et al. (2024), que reduz Sinkhorn
  distance entre distribuição overall e distribuições por grupo
  demográfico, representa o estado-da-arte atual em fairness para
  VLMs em domínio médico. A presente dissertação investiga, em
  ablation arquitetural, se mecanismo análogo aplicado a race
  classification multi-classe sobre FairFace supera o FiLM-
  conditioning baseado em SkinToneNet — endereçando a recomendação
  metodológica registrada em reunião de orientação (2026-06-08)."*

## 9-12. Arquivos e análise crítica

- PDF: pendente em `pdfs/luo_2024_fairclip.pdf`.
- Código: github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP.
- Entradas relacionadas: [[aldahoul_2024]] (FaceScanPaliGemma), 
  [[perez_2018]] (FiLM — alternativa arquitetural), [[hardt_2016]].

> **[BLOQUEADO]** Análise crítica detalhada requer leitura integral
> do PDF.
