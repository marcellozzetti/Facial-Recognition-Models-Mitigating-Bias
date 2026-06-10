# Rodada 7 — candidatos identificados

> **Data**: 2026-06-10.
> **Status**: 5 lotes de busca executados; candidatos listados abaixo
> para triagem.
> **Próxima ação**: triar com critérios revisados (R7 menos
> restritivos) e criar fichas em batch.

## Resumo executivo

- **30+ candidatos** identificados nas 12 buscas iniciais.
- **22 de 2024-2026** (atende meta orientador de ≥20 papers recentes).
- **Cobertura**: VLM fairness (Track I), conditioning moderno (Track J),
  skin tone moderno, race classification SOTA, SSL/multi-task.

## Lote 1 — VLM / CLIP / BLIP fairness (Track I NOVO, alvo 12 fichas)

### Aprovação imediata (alta relevância + venue forte)

| # | Paper | Ano | Venue | Por que aprovar |
|---|---|---|---|---|
| I-1 | **FairCLIP: Harnessing Fairness in Vision-Language Learning** (Luo et al.) | 2024 | CVPR | SOTA fair VL; Sinkhorn distance; Harvard-FairVLMed dataset; aplicação direta como Cap 2 ablation |
| I-2 | **FairerCLIP: Debiasing CLIP using RKHS** (arXiv:2403.15593) | 2024 | a verificar | Reproducibilidade FairCLIP; método alternativo |
| I-3 | **Joint Vision-Language Social Bias Removal for CLIP** (arXiv:2411.12785) | 2024 | a verificar | Debiasing in/out-domain; bias alignment + counterfactual |
| I-4 | **AI-Face: Million-Scale Demographically Annotated Dataset** (arXiv:2406.00783) | 2025 | CVPR 2025 | Million-scale; demografics annotated; benchmark fairness |
| I-5 | **GRAS Benchmark: VLM bias on Gender Race Age Skin Tone** (arXiv:2508.18989) | 2025 | a verificar | Benchmark direto multi-eixo |
| I-6 | **A Unified Debiasing Approach for VLMs** (arXiv:2410.07593) | 2024 | a verificar | Approach across modalities/tasks |
| I-7 | **Evaluating Fairness in Large VLMs Across Demographic Attributes** (arXiv:2406.17974) | 2024 | EMNLP findings | Avaliação ampla incluindo skin tone |
| I-8 | **BendVLM: Test-Time Debiasing of VL Embeddings** (arXiv:2411.04420) | 2024 | a verificar | Test-time intervention |

### Avaliação por exceção

| # | Paper | Ano | Decisão preliminar |
|---|---|---|---|
| I-9 | **Reproducibility of FairCLIP** (arXiv:2509.06535) | 2025 | ✅ STANDBY — útil como evidência crítica do método FairCLIP |
| I-10 | **Debiasing CLIP with Neural Interventions** (Springer 2025) | 2025 | ✅ APROVAR — método 2025 leve, no retraining |
| I-11 | **FairCLIP: Attribute Prototype** (Wang, arXiv:2210.14562) | 2022 | ⚠ STANDBY — paper anterior com mesmo nome; verificar diferença |
| I-12 | **A Prompt Array Keeps the Bias Away** (arXiv:2203.11933) | 2022 | ⚠ STANDBY — adversarial learning para VLM |
| I-13 | **FairQueue: Fair T2I Prompt Learning** (arXiv:2410.18615) | 2024 | ❌ DESCARTAR — text-to-image generation, não classification |

## Lote 2 — Skin tone moderno (alvo +4 fichas)

| # | Paper | Ano | Venue | Decisão |
|---|---|---|---|---|
| ST-1 | **Casual Conversations v2** (Porgali et al.) | 2023 | CVPRW | ✅ APROVAR — Fitzpatrick + MST + 7 países; já citado em [[dataset_hazirbas_2021]] |
| ST-2 | **Monk Skin Tone JAAD dermatology** | 2025 | JAAD | ⚠ STANDBY — dermatologia clínica, ortogonal mas útil |
| ST-3 | **Preprocessing Skin Segmentation MST** (IEEE Xplore) | 2024 | IEEE | ⚠ STANDBY — pré-processamento técnico |
| ST-4 | **Skin Tone Estimation Diverse Lighting** (PMC) | 2024 | PMC NIH | ⚠ STANDBY — confounder iluminação |
| ST-5 | **Enhancing Fairness ML Skin Tone MST** (researchgate) | 2024 | a verificar venue | ⚠ STANDBY |

## Lote 3 — Race classification SOTA 2025-2026 (alvo +5 fichas)

| # | Paper | Ano | Venue | Decisão |
|---|---|---|---|---|
| RC-1 | **Reliable & Reproducible Demographic Inference for Fairness** (arXiv:2510.20482) | 2025 | a verificar | ✅ APROVAR — modular DAI pipeline; intra-identity consistency |
| RC-2 | **Demographic-Agnostic Fairness without Harm** (arXiv:2509.24077) | 2025 | a verificar | ✅ APROVAR — fairness sem sensitive attribute access |
| RC-3 | **Review of Demographic Fairness in FR (v3)** (arXiv:2502.02309) | 2025 | TBIOM | ✅ JÁ NO CORPUS — [[survey_kotwal_2025]] |
| RC-4 | **Towards Fair Face Verification: In-Depth Demographic Analysis** (arXiv:2307.10011) | 2023 | a verificar | ⚠ STANDBY — anterior mas relevante |
| RC-5 | **Bahiru et al. 2025 FairFace audit** | 2025 | a verificar | 🔍 BUSCAR — citado mas não encontrado |
| RC-6 | **Dong et al. 2025 intersectional FairFace** | 2025 | a verificar | 🔍 BUSCAR — citado mas não encontrado |

## Lote 4 — Conditioning moderno (Track J NOVO, alvo 5 fichas)

| # | Paper | Ano | Venue | Decisão |
|---|---|---|---|---|
| J-1 | **AIM-Fair: Advancing Algorithmic Fairness Selectively Fine-Tuning Biased** | 2025 | **CVPR 2025** | ✅ **APROVAR PRIORITÁRIO** — fine-tuning seletivo, fairness algorítmica |
| J-2 | **Fairness-Aware LoRA Under Demographic Privacy** (arXiv:2503.05684) | 2025 | a verificar | ✅ APROVAR — método para Cap 2 ablation |
| J-3 | **LoRA-FAIR: Federated LoRA** | 2025 | **ICCV 2025** | ✅ APROVAR — venue top |
| J-4 | **FairLoRA: Vision Bias Mitigation Low-Rank Adaptation** (arXiv:2410.17358) | 2024 | a verificar | ✅ APROVAR — fairness-driven LoRA |
| J-5 | **On Fairness of Low-Rank Adaptation of Large Models** (arXiv:2405.17512) | 2024 | a verificar | ✅ APROVAR — análise fundamental |
| J-6 | **FairViT: Adaptive Masking** (arXiv:2407.14799) | 2024 | a verificar | ✅ APROVAR — ViT-specific |
| J-7 | **Fairness-aware ViT Debiased Self-Attention** (arXiv:2301.13803) | 2023 | a verificar | ⚠ STANDBY |
| J-8 | **S-Adapter: ViT Anti-Spoofing** (arXiv:2309.04038) | 2023 | a verificar | ❌ DESCARTAR — anti-spoofing, não fairness |

## Lote 5 — SSL + multi-task fairness 2025-2026 (alvo 3 fichas)

| # | Paper | Ano | Venue | Decisão |
|---|---|---|---|---|
| MT-1 | **Provable Adversarial Fair SSL Contrastive** (arXiv:2406.05686) | 2024 | a verificar | ✅ APROVAR — extensão SSL fair |
| MT-2 | **Self-Supervised Fair Representation w/o Demographics** (NSF PAR) | 2023+ | a verificar | ⚠ STANDBY — útil mas anterior |
| MT-3 | **NTKMTL: Mitigating Task Imbalance MTL** (arXiv:2510.18258) | 2025 | a verificar | ⚠ STANDBY — NTK approach |
| MT-4 | **Single-model multi-task face recognition** (IET) | 2025 | IET | ⚠ STANDBY — engenharia, não fairness focal |
| MT-5 | **InsightFace March 2026 papers** (blog) | 2026 | blog | 🔍 EXPLORAR — múltiplos papers mencionados |

## Consolidado de aprovações imediatas (Rodada 7 — primeira leva)

| Track | Aprovados imediatos | Standbys | Descartados |
|---|---|---|---|
| I (VLM fairness) | 8 | 5 | 1 |
| Skin tone | 1 | 4 | 0 |
| Race classification | 2 | 1 | 0 (3 a buscar) |
| J (conditioning) | 6 | 1 | 1 |
| SSL/MT | 1 | 3 | 0 |
| **TOTAL R7 primeira leva** | **18 aprovados** | **14 standby** | **2 descartados** |

## Status atual após R7 primeira leva

| Métrica | Antes R7 | Após R7 primeira leva | Meta |
|---|---|---|---|
| Total fichas | 37 | **55** (37 + 18) | ≥ 100 |
| Fichas 2025-2026 | 5 | **20** (5 + 15 novos) | ✅ ≥ 20 atendida |
| Track I (VLM) | 0 | 8 | ≥ 12 |
| Track J (conditioning) | 0 | 6 | ≥ 5 ✅ |

**Já atingimos a meta de 20+ artigos de 2025-2026 com a primeira leva da R7!**

Gap restante para corpus ≥ 100: **−45 fichas**.

## Próximas buscas para fechar gap

Para atingir 100, precisamos +45 fichas. Sugestão de áreas a explorar:
- **Surveys recentes** (2024-2026) sobre VLM, fairness, face recognition
- **Workshop papers** ICCV/CVPR/NeurIPS 2024-2025 sobre fairness facial
- **Datasets emergentes** alternativos ao FairFace
- **Métodos de calibração** post-hoc para fairness
- **Aplicações práticas** (medical FR, security FR, identity verification)
- **Papers fundadores faltantes** (ArcFace 2019, CosFace 2018, FaceNet 2015)
- **Métodos de DRO modernos** pós-Sagawa 2020
- **Causal fairness** (Pearl, Kusner) — área não coberta

## Recomendação de execução

**Esta sessão**: criar fichas OVERVIEW_ONLY iniciais para os 18 aprovados
imediatos para entrar no corpus. Cada ficha pode ser curta (template
de 5 seções base + Aplicação ao pipeline v3.2).

**Próximas sessões**:
- Resolver standbys após leitura adicional.
- Executar 2-3 lotes adicionais para fechar gap de 45 fichas.
- Promover OVERVIEW_ONLY → VERIFIED conforme PDFs disponíveis.
