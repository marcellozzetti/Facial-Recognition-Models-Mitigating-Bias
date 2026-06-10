---
name: bian-2025-lorafair
status_verificacao: OVERVIEW_ONLY
autores: [Jieming Bian, et al.]
ano: 2025
titulo: "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement"
venue: "IEEE/CVF International Conference on Computer Vision (ICCV 2025)"
tipo_publicacao: conference
arxiv_id: a confirmar
doi: null
url_primario: https://openaccess.thecvf.com/content/ICCV2025/papers/Bian_LoRA-FAIR_Federated_LoRA_Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.pdf
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Apenas título + venue + sumário via busca web. PDF integral pendente.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**

# LoRA-FAIR — Federated LoRA Fine-Tuning (ICCV 2025)

> **LoRA com fairness via federated learning**. Aggregation +
> initialization refinement para fairness em distributed setup.
> Track J (conditioning moderno) — alternativa ao FiLM via LoRA.

## 1. Resumo (inferido do título)

> **[PENDENTE PDF]** Federated learning com LoRA tem desafios de
> fairness (heterogeneity entre clients, viés de dados locais).
> LoRA-FAIR propõe técnica de aggregation + initialization refinement
> para mitigar.

## 2. Método (inferido)

> **[PENDENTE PDF]** Inferência:
> - LoRA padrão (low-rank decomposition de gradient updates).
> - Aggregation step modificado para fairness.
> - Initialization refinement para reduzir viés inicial.

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Não extraídos.

## 7. Aplicação ao nosso pipeline v3.2

### 7.1 Track J (conditioning moderno) — alternativa ao FiLM

LoRA é abordagem mais moderna que FiLM para fine-tuning eficiente.
**Diferença**: LoRA opera em low-rank decomposition de weights;
FiLM modula features intermediárias.

### 7.2 Aplicação como ablation

No Cap 2, ablation:
- ConvNeXt-T + FiLM (nosso) — modulação de features.
- ConvNeXt-T + LoRA (LoRA-FAIR) — modificação de weights via
  low-rank.

Endereça recomendação do orientador de testar mecanismo moderno.

### 7.3 Federated não se aplica diretamente

Nossa pesquisa não é federated. LoRA-FAIR é citação para **LoRA
em fairness contexts**, não para federated specifically.

## 8. Pontos para citar

- *"Bian et al. (2025), apresentado no ICCV 2025, propõem LoRA-FAIR —
  framework de federated LoRA fine-tuning com aggregation e
  initialization refinement para fairness em settings distribuídos.
  Embora a presente dissertação não opere em federated learning,
  o paradigma LoRA é considerado como ablation arquitetural
  alternativa ao FiLM-conditioning no Capítulo 2."*

## 9-12. Arquivos e análise crítica

- PDF: pendente em `pdfs/bian_2025_lorafair.pdf`.
- Entradas relacionadas: [[perez_2018]] FiLM, mecanismos de
  conditioning modernos (Track J).

> **[BLOQUEADO]** Análise crítica detalhada requer leitura integral.
