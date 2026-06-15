---
name: bian-2025-lorafair
status_verificacao: VERIFIED
autores: [Jieming Bian (University of Florida), Lei Wang (Univ Florida), Letian Zhang (Middle Tennessee State University), Jie Xu (Univ Florida)]
ano: 2025
titulo: "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement"
venue: "IEEE/CVF International Conference on Computer Vision (ICCV 2025) — Open Access"
tipo_publicacao: conference
arxiv_id: null
doi: null
url_primario: https://openaccess.thecvf.com/content/ICCV2025/papers/Bian_LoRA-FAIR_Federated_LoRA_Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.pdf
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de OpenAccess CVF (pdfs/bian_2025_lorafair.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, motivação técnica e figura LoRA-FAIR lidos via pdftotext.
---

# LoRA-FAIR — Federated LoRA Fine-Tuning (Bian, Wang, Zhang & Xu, ICCV 2025)

> **Federated Learning + LoRA** com correção de dois problemas técnicos
> em FL: **Server-Side Aggregation Bias** e **Client-Side Initialization
> Lag**. Universidade da Flórida + Middle Tennessee State.

## 1-2. Resumo + método

- **Problema técnico**: combinar LoRA com Federated Learning enfrenta
  dois desafios:
  1. **Server-Side Aggregation Bias**: média server-side de matrices
     LoRA diverge do ideal global update.
  2. **Client-Side Initialization Lag**: inicialização inconsistente
     entre rounds.
- **Método LoRA-FAIR**: introduz correction term no server que
  reconstrói o ideal global update via Eq. (6), encontra residual
  LoRA module via Eq. (8), e substitui B̄ por B̄' = B̄ + ΔB.
- **Eficiência**: mantém compute e comunicação eficientes.

## 3-5. Datasets e resultados

- **Modelos**: ViT e MLP-Mixer.
- **Datasets**: large-scale visual datasets (CV).
- **Resultado**: consistentemente supera SOTA em settings FL.

## 7. Aplicação ao pipeline v3.2

- **Não é federated** — nossa tese é centralized — direção complementar.
- **Track J** (conditioning moderno) — comparação conceitual com
  LoRA-style adapters.
- **Cap 2** (Revisão): citação para fronteira federated × fairness em
  conditioning moderno.

## 8. Citar

- *"Bian, Wang, Zhang & Xu (ICCV 2025, University of Florida + Middle
  Tennessee State University) propõem LoRA-FAIR, um método de
  federated LoRA fine-tuning que corrige dois problemas técnicos
  específicos da combinação Federated Learning + LoRA — Server-Side
  Aggregation Bias e Client-Side Initialization Lag — via correction
  term no servidor que reconstrói o ideal global update. Apesar de
  operar em setting federated, a abordagem evidencia complexidade
  arquitetural inerente a adapters modernos, contexto para a escolha
  desta dissertação por FiLM-conditioning como mecanismo central
  (mais simples e interpretável que LoRA-based variants)."*

## 9-12.

PDF: `pdfs/bian_2025_lorafair.pdf`. Conexões: [[fairlora_2024]] (LoRA
fairness vision), [[fairness_lora_2024]] (On Fairness of LoRA),
[[zhao_2025_aimfair]] (AIM-Fair selective fine-tuning), [[perez_2018]]
(FiLM — mecanismo central da nossa tese, comparação conceitual).
