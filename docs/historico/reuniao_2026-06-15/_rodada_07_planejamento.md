# Rodada 7 — planejamento (pós-reunião 2026-06-08)

> **Aberta em 2026-06-10** após reunião com Prof. Quiles.
> **Target**: ≥ 70 novos artigos; corpus final ≥ 100; ≥ 20 artigos
> 2025-2026.
> **Critérios menos restritivos** que rodadas anteriores (autorizado
> pelo orientador).
> **Foco**: CLIP/BLIP/VLM fairness, skin tone moderno, race
> classification 2025-2026, Vision Transformers fairness.

## 1. Critérios de triagem revisados (mais permissivos)

| Critério | R1-R6 (anterior) | R7 (revisado) |
|---|---|---|
| Venue | Top conferences + journals peer-reviewed | + workshops, + preprints recentes (≤ 18 meses) |
| Citações | Threshold por ano (2024≥20, 2025≥10, etc.) | Threshold reduzido a 50%; 2026 sem threshold |
| Aplicabilidade | Direta ao pipeline v3.2 | + adjacente (VLM bias, FR bias mesmo sem skin tone) |
| Cobertura única | Caso especial | Mantido como critério forte |

Justificativa: orientador autorizou expansão para 100+ artigos
"mesmo que o filtro diminua mais". Mantemos rastreabilidade de
critérios.

## 2. Áreas-foco da Rodada 7

### 2.1 Vision-Language Models em fairness facial (CLIP / BLIP / SigLIP)

> **CRÍTICO**: orientador recomendou GLIP (provavelmente CLIP).
> Investigação confirma que **CLIP é o VLM canônico** em fairness
> facial; GLIP é detection, não usado em fairness.

Buscar:
- **FairCLIP** (Luo et al., CVPR 2024) — primeiro VL fairness médico
- Debiasing CLIP for face attribute fairness
- BLIP-2 face attribute fairness
- SigLIP (Google 2024) for fairness
- FaceScanPaliGemma já está no corpus (AlDahoul 2026)
- IndicFairFace (2026 preprint) — geographic bias VLM
- GRAS benchmark (2025) — gender, race, age, skin tone VLM bias

### 2.2 Skin tone moderno

- **STW dataset** (Pereira 2026) já no corpus
- **TrueSkin** (Lu 2025) — já em standby R6-9
- Casual Conversations v2 (Porgali 2023)
- Demographic Calibration VLM (2025)
- Skin tone clinical dermatology fairness (Adapting LLMs 2025)

### 2.3 Race classification SOTA 2025-2026

- Bahiru et al. 2025 (FairFace audit)
- Dong et al. 2025 (intersectional FairFace)
- AlDahoul 2026 já no corpus
- Buscar competidores pós-FaceScanPaliGemma

### 2.4 Mecanismos modernos de conditioning (alternativos ao FiLM)

- **CLIP-conditioning**: usar embeddings CLIP como contexto
- **BLIP-2 Q-Former**: conditioning via cross-attention
- **LoRA**: low-rank adaptation
- **Prompt tuning**: continuous prompts
- **Adapter modules**: insertion layers

Comparação contra FiLM (Perez 2018).

### 2.5 Self-supervised + multi-task fairness 2025-2026

- Continuação Ramachandran (R6-7)
- Aguirre & Dredze (R6-4) extensões
- Self-supervised face attribute fairness 2025

### 2.6 Auditoria FR moderna

- IndicFairFace 2026 (geographic VLM bias)
- Demographic Inference for Fairness 2025 (Bahiru, Dong)
- Continuous demographic labels (Neto 2025 já no corpus)

## 3. Buscas planejadas (queries)

Lote 1 — VLM fairness:
- "FairCLIP CVPR 2024 fairness face attribute"
- "CLIP debiasing race classification 2025"
- "BLIP face attribute fairness 2024 2025"
- "vision language model bias face 2025 SOTA"

Lote 2 — Skin tone:
- "Monk Skin Tone classifier deep learning 2025 2026"
- "TrueSkin skin tone benchmark 2025"
- "Casual Conversations V2 fairness audit"

Lote 3 — Race classification 2025-2026:
- "FairFace audit 2025 SOTA"
- "intersectional fairness FairFace race gender 2025"
- "FaceScanPaliGemma competitor face attribute 2026"

Lote 4 — Conditioning:
- "FiLM vs CLIP conditioning visual classification"
- "LoRA face attribute fine-tuning fairness"
- "prompt tuning face recognition 2025"

Lote 5 — SSL + multi-task:
- "self-supervised facial attribute fairness 2025"
- "multi-task fairness vision 2025"

## 4. Comparação CLIP/BLIP/FiLM (output para Cap metodológico)

### 4.1 CLIP (Radford et al. 2021, OpenAI)

- **Arquitetura**: dois encoders (text + image) treinados com
  contrastive loss.
- **Conditioning**: image embedding (512 ou 768 dim) usado como
  contexto.
- **Para fairness**: FairCLIP (Luo 2024) usa Sinkhorn distance entre
  distribuições demográficas.
- **Pros**: zero-shot, embeddings ricos, comunidade ampla.
- **Cons**: enviesado em treino (LAION desbalanceado), 14% misclass
  rate em Black faces.

### 4.2 BLIP-2 (Li et al. 2023, Salesforce)

- **Arquitetura**: Q-Former entre visual encoder (ViT) e LLM.
- **Conditioning**: cross-attention via Q-Former.
- **Para fairness**: Linear probes em BLIP-2 ViT-L/14 mostraram
  trade-off com fairness regularization.
- **Pros**: mais moderno, melhor entendimento visual-linguístico.
- **Cons**: pesado computacionalmente, menos literatura em fairness
  facial específica.

### 4.3 FiLM (Perez et al. 2018, AAAI)

- **Arquitetura**: camada conditional inserida em CNN. F' = γ⊙F + β.
- **Conditioning**: vetor de contexto via MLPs simples.
- **Para fairness**: NUNCA testado em fairness facial até nossa tese.
- **Pros**: parameter-efficient (~1% overhead), interpretável,
  paradigma estabelecido.
- **Cons**: mais antigo, sem evidência em fairness, pontos de
  inserção manuais.

### 4.4 Tabela comparativa

| Critério | FiLM | CLIP | BLIP-2 |
|---|---|---|---|
| Ano de origem | 2018 | 2021 | 2023 |
| Compute para conditioning | Baixo (~1%) | Médio (encoder pré-treinado) | Alto (Q-Former) |
| Parâmetros adicionais | ~380K | ~50M+ (encoder) | ~100M+ (Q-Former) |
| Evidência em fairness facial | Zero | FairCLIP (Luo 2024) | Limitada |
| Interpretabilidade | Alta | Média | Baixa |
| Aplicabilidade ao nosso pipeline | Direta | Requer adaptação | Requer pesquisa adicional |

### 4.5 Recomendação preliminar para a tese

**Adotar ABORDAGEM DUPLA**:

1. **FiLM como mecanismo PRINCIPAL** (originalidade da contribuição):
   - Primeira aplicação de FiLM a fairness racial (gap claro).
   - Mecanismo eficiente, interpretável.
   - Ancora a tese em paradigma estabelecido.

2. **CLIP-conditioning como ABLATION**:
   - Comparar contra alternativa moderna.
   - Defende-se contra revisor que questione "por que FiLM e não CLIP?".
   - Citar FairCLIP (Luo 2024) como precedente.

3. **BLIP-2 NÃO no escopo principal** — mencionar como future work.

## 5. Mapeamento das áreas para o cronograma de qualificação

| Semana | Frente | Output |
|---|---|---|
| 1 (10-16/jun) | Rodada 7 buscas + triagem inicial | 50+ candidatos identificados |
| 1-2 (10-23/jun) | Investigação CLIP/BLIP/FiLM | Tabela comparativa pronta para Cap metodológico |
| 2 (17-23/jun) | Cap 1 Introdução + RL parte 1 | ~20 páginas escritas |
| 3 (24-30/jun) | RL parte 2 + Metodologia | ~40 páginas no total |
| 4 (1-7/jul) | Bibliografia + figuras | Documento polido |
| 5 (8-14/jul) | Revisão final + submissão 15/jul | Documento entregue |

## 6. Quantitativo de literatura recente

Status atual do corpus (2026-06-10):

| Ano | Fichas no corpus | Meta orientador |
|---|---|---|
| 2026 | 3 (AlDahoul Nature, Pereira, Liu BNMR pode ser submetido em maio) | — |
| 2025 | 4-5 (Lafargue, Neto, Liu, Schumann é 2023...) | — |
| 2025+2026 combinados | ~7-8 | **≥ 20** |
| Total corpus | 37 | **≥ 100** |

**Gap**: precisamos de ~13 novos artigos 2025-2026 + ~60-65 artigos
adicionais para chegar a 100.

Estratégia: **Rodada 7 focada 70% em 2025-2026**.
