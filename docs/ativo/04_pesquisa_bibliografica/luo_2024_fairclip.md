---
name: luo-2024-fairclip
status_verificacao: VERIFIED
autores: [Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, Yi Fang, Mengyu Wang]
ano: 2024
titulo: "FairCLIP: Harnessing Fairness in Vision-Language Learning"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)"
tipo_publicacao: conference
arxiv_id: "2403.19949"
doi: null
url_primario: https://arxiv.org/abs/2403.19949
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: 16
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/luo_2024_fairclip.pdf), 16 páginas (8 main + 8 suplementar), lido em 2026-06-15.
---

# FairCLIP — Harnessing Fairness in Vision-Language Learning (Luo et al., CVPR 2024)

> **Primeiro fair vision-language medical dataset** (Harvard-FairVLMed)
> + abordagem baseada em **optimal transport (distância de Sinkhorn)**
> para alinhar distribuições demográficas no espaço de similaridade
> vision-text. Diretamente relevante como **baseline principal do
> estudo comparativo** de mecanismos de conditioning no Cap 2 da
> dissertação.

## 1. Resumo do problema atacado

Fairness em vision-language models (VLMs) é problema crítico em
healthcare — modelos influenciam diagnóstico e tratamento. Embora
fairness tenha sido investigado no domínio vision-only, fairness em
**medical VL models** permanece inexplorada por dois motivos:

1. **Escassez de datasets VL médicos com atributos demográficos**.
   Datasets existentes (CheXpert, MIMIC-CXR) usam labels noisy
   extraídos automaticamente de radiology reports.
2. **Datasets disponíveis foram criados para outras finalidades**, com
   poucos atributos demográficos.

Adicionalmente, **radiology reports** focam em observações diretas das
imagens, com pouca informação clínica adicional — limitando sua
utilidade para fairness studies.

## 2. Método

### 2.1 Dataset Harvard-FairVLMed

- **Primeiro fair vision-language medical dataset** publicamente
  disponível.
- **10.000 pacientes**, cada um com: 1 SLO fundus image + 1 clinical
  note + ground-truth glaucoma label.
- **Splits**: 7.000 train / 1.000 val / 2.000 test.
- **Atributos demográficos**: age (média 60.9 ± 16.2), gender, race,
  ethnicity, preferred language, marital status.
- **Distribuição racial**: White 76.9% | Black 14.9% | Asian 8.2%.
- **Gender**: Female 56.3% | Male 43.7%.
- **Ethnicity**: Non-Hispanic 90.6% | Hispanic 4.0% | unspecified 5.4%.
- **Language**: English 92.5% | Spanish 1.7% | Others 0.8% |
  unknown 5.0%.
- **Notas clínicas**: 11-332 palavras (média 147).
- **De-identification em 3 etapas**: Presidio (Microsoft) + regras +
  validação por 4 especialistas médicos.
- **Categorização binária**: non-glaucoma (VF mean deviation ≥ -1 dB)
  vs glaucoma (VF mean deviation < -3 dB).

### 2.2 FairCLIP — formulação

Dado modelo VL `f` com encoder visual `f_I` e text encoder `f_T`,
gerando features `z_I, z_T`. Para batch de N pares, matriz `M ∈ ℝ^{N×N}`
com `M_ij = z_Ii · z_Tj` (similaridade coseno).

CLIP padrão minimiza cross-entropy simétrico sobre M (contrastive loss).

**FairCLIP** adiciona objetivo de fairness:

```
min_f sum_α^A d(D_{(x_I,x_T,a)|f} - D_{(x_I,x_T,a)|a=α}|f)
```

onde `D_B|f` é distribuição empírica de `M_{i,i}` no batch e `D_{B_a}|f`
é a distribuição restrita ao grupo demográfico `a`.

### 2.3 Sinkhorn distance (escolha do `d`)

KL-divergence foi rejeitada por **não ser simétrica nem satisfazer
desigualdade triangular**. FairCLIP usa **Sinkhorn distance** (variante
da Wasserstein, baseada em Peyré & Cuturi 2019):

```
W_ε(D_B, D_B_a) = inf_{γ ∈ Γ(D_B, D_B_a)} {E[c(p,q)] + ε·H(γ | μ⊗ν)}
```

A Sinkhorn loss é **adicionada à loss original do CLIP** na fase de
pre-training.

### 2.4 Arquitetura e hiperparâmetros

- **Base models testados**: CLIP (ViT-B/16 e ViT-L/14) + BLIP-2.
- **CLIP fine-tuning**: Adam, lr=1e-5, β1=β2=0.1, weight decay=6e-5,
  batch_size=32, 10 epochs.
- **FairCLIP específico**:
  - `|B_a|` = 32 (samples por grupo no batch).
  - Sinkhorn loss weight = **1e-7**.
- **BLIP-2**: ViT-L/14 frozen + Q-former, AdamW (β1=0.9, β2=0.98,
  wd=0.05), cosine lr decay (max 1e-4 → warmup 1e-6 → min 1e-5).
- **Treinamento**: 50 épocas em 1× V100 GPU.
- **Augmentations**: random resized crop 224×224 + horizontal flip.
- **Clinical notes**: sumarizadas com GPT-4 / PMC-LLAMA / MED42 para
  caber no limite de 77 tokens do CLIP (prompt: "Summarize the key
  details, including the presence of glaucoma, from the clinical note
  within 180 characters").

## 3. Datasets e setup experimental

- **Treino e teste**: Harvard-FairVLMed (próprio paper).
- **3 seeds** reportados (com média ± desvio padrão em todas as
  tabelas).
- **Comparações**: CLIP natural pre-trained vs CLIP fine-tuned (CLIP-FT)
  vs BLIP-2 vs BLIP-2-FT vs FairCLIP.
- **2 modos de avaliação**:
  - **Linear probing**: classifier linear sobre features visuais.
  - **Zero-shot transfer**: classe escolhida por maior similaridade
    text-image embedding.

## 4. Métricas reportadas

- **DPD** (Demographic Parity Difference) ↓ — fairness.
- **DEOdds** (Difference in Equalized Odds) ↓ — fairness.
- **AUC** (Area Under ROC) ↑ — performance.
- **ES-AUC** (Equity-Scaled AUC) ↑ — performance + fairness combinados:
  ```
  ES-AUC = AUC / (1 + sum_a |AUC - AUC_a|)
  ```
- **Group-wise AUC** ↑ — performance por subgrupo demográfico.

## 5. Resultados principais (valores numéricos)

### 5.1 Tabela 2 — Linear probing, atributo Race (todos VL models)

| Modelo | DPD ↓ | DEOdds ↓ | AUC ↑ | ES-AUC ↑ | Asian | Black | White |
|---|---|---|---|---|---|---|---|
| CLIP (natural) | 5.30 ± 0.63 | 14.00 ± 1.01 | 77.27 ± 0.03 | **72.43 ± 0.29** | 79.74 | 73.60 | 77.82 |
| CLIP-FT (medical) | 4.01 ± 0.47 | 9.57 ± 0.83 | 80.27 ± 0.08 | 74.70 ± 0.33 | 82.19 | 75.67 | 81.20 |
| BLIP-2 (natural) | 9.44 ± 0.65 | 10.62 ± 0.22 | 73.81 ± 0.02 | 68.88 ± 0.04 | 76.28 | 69.55 | 74.22 |
| BLIP-2-FT (medical) | 8.30 ± 0.36 | 10.91 ± 0.32 | 80.10 ± 0.03 | 73.81 ± 0.10 | 82.09 | 74.43 | 80.97 |

**Achado**: pre-training médico melhora performance-fairness trade-off
em **3 dos 4 atributos** (exceto language).

### 5.2 Tabela 3 — Zero-shot CLIP vs FairCLIP, atributo Race

| Modelo | DPD ↓ | DEOdds ↓ | AUC ↑ | ES-AUC ↑ |
|---|---|---|---|---|
| CLIP (ViT-B/16) | 15.35 ± 6.50 | 15.11 ± 5.01 | 67.84 ± 0.90 | 61.67 ± 0.63 |
| **FairCLIP (ViT-B/16)** | **6.07 ± 2.44** | **10.50 ± 2.73** | **70.24 ± 1.26** | **65.50 ± 2.60** |
| CLIP (ViT-L/14) | 10.10 ± 9.44 | 10.79 ± 10.41 | 67.83 ± 2.92 | 63.53 ± 1.83 |
| FairCLIP (ViT-L/14) | 17.79 ± 4.86 | 18.30 ± 2.07 | 69.88 ± 2.00 | 66.54 ± 1.73 |

**Achado**: FairCLIP **reduz DPD em 60.5%** (15.35 → 6.07) na ViT-B/16,
mas **piora DPD em 76%** na ViT-L/14 (10.10 → 17.79). ES-AUC melhora em
ambos (+3.83 e +3.01 pontos).

### 5.3 Tabela 3 — Zero-shot, atributo Gender

| Modelo | DPD ↓ | DEOdds ↓ | ES-AUC ↑ | Female | Male |
|---|---|---|---|---|---|
| CLIP (ViT-B/16) | 4.34 | 9.95 | 63.21 | 64.62 | 71.96 |
| **FairCLIP (ViT-B/16)** | **0.84** | **2.97** | **65.39** | 66.81 | 73.50 |

**Achado**: redução de **80.6% no DPD** (4.34 → 0.84) e **70.2% no
DEOdds** em gender. Melhoria mais consistente.

### 5.4 Tabela 5 — End-to-end fine-tuning (suplementar)

Trade-offs persistem: FairCLIP às vezes piora DPD enquanto melhora
AUC e ES-AUC. Por exemplo, em Race (ViT-B/16): CLIP DPD 5.85 →
FairCLIP DPD 11.38, mas AUC sobe (81.19 → 81.70) e ES-AUC sobe
(75.07 → 76.85).

### 5.5 Subgrupos preferidos (achado central)

**Asian, Male, Non-Hispanic, Spanish** são consistentemente os
subgrupos com maior performance — **mesmo sendo grupos minoritários**
(exceto Non-Hispanic) — sugerindo viés derivado do pre-training, não
do imbalance dos dados.

### 5.6 Comparação com outras abordagens fair (Figura 5d)

Ordem de ES-AUC em race: **FairCLIP > CLIP w/ FSCL > CLIP w/ Adv > CLIP**.

## 6. Limitações declaradas pelos autores

O paper **não tem seção explícita de limitações**. A conclusão é apenas
recapitulativa. Algumas limitações estão implícitas:

- Domínio restrito (medical, especificamente glaucoma).
- Classification binária (glaucoma vs non-glaucoma).
- Dataset com 76.9% White — imbalance racial significativo.
- Sinkhorn loss weight (1e-7) parece muito específico, sem ablation
  rigorosa na main paper.

## 7. Limitações que identifiquei (leitura crítica)

- **3 seeds** é pouco para garantir significância estatística,
  especialmente quando desvios padrão são altos (e.g., 6.50, 9.44 em
  DPD da Tabela 3).
- **Trade-off inconsistente**: FairCLIP **piora DPD em Race (ViT-L/14)**
  e em **Ethnicity (ViT-B/16)**, demonstrando que Sinkhorn não garante
  melhoria universal.
- **Análise de sensibilidade do hiperparâmetro ε** (Figura 5b) está
  no suplementar e não é discutida na main paper.
- **Pre-training data leakage** possível: a Sinkhorn loss alinha
  distribuições durante pre-training; modelo pode aprender a **mascarar
  bias** sem realmente reduzi-lo.
- **Falta comparação com Hardt-style post-processing** ou
  re-calibração — apenas adversarial (Beutel 2017) e FSCL (Park 2022)
  são baseline.
- **Não testa em face recognition** — domínio totalmente diferente da
  nossa tese.
- Race é **3 classes** (Asian/Black/White) — bem menos granular que
  7-class FairFace usado em nossa tese.
- **Sinkhorn distance** entre **2 distribuições** apenas (overall vs
  subgroup); não há tratamento joint para múltiplos atributos
  protegidos simultaneamente.

## 8. Relação com nossa pesquisa

### 8.1 Centralidade para o estudo comparativo de mecanismos (Cap 2)

FairCLIP é **referência primária** para incluir CLIP-conditioning no
estudo comparativo. Especificamente:

1. **Mecanismo**: Sinkhorn loss penalizando disparidade entre
   distribuições demográficas no espaço de similaridade.
2. **Diferença em relação ao FiLM**: FairCLIP **não condiciona features
   via vetor auxiliar** — opera via **loss penalty** que requer
   identificação demográfica no batch.
3. **Comparabilidade**: para nossa Configuração D (cross-attention com
   CLIP), FairCLIP fornece um baseline de Sinkhorn-augmented CLIP.

### 8.2 Aplicabilidade direta vs adaptação necessária

| Aspecto | FairCLIP (Luo) | Nossa proposta |
|---|---|---|
| Tarefa | Binary glaucoma | 7-class race FairFace |
| Atributo | Race 3 classes | 7-class race / MST 10-classes |
| Imagem | SLO fundus | Face image |
| Texto | Clinical note | (não temos texto) |
| Mecanismo fairness | Sinkhorn loss | FiLM-conditioning (proposta) |
| Backbone | ViT-B/16 ou L/14 (CLIP) | ConvNeXt-T |

### 8.3 Implicação para nossa Hipótese H6

Pangelinan 2023 (pixel info explica disparidades) é uma das nossas
ameaças. FairCLIP **demonstra empiricamente** que subgrupos
minoritários (Asian, Male) podem **superar** subgrupos majoritários —
contrariando explicação puramente de pixel info / imbalance.
**Suporta H6**.

### 8.4 Risco identificado

**Pre-trained CLIP herda viés substancial** — paper mostra que CLIP
zero-shot tem ES-AUC 61.67 (race), bem pior que CLIP fine-tuned. Para
comparação justa em nossa ablation, **precisamos fine-tunar CLIP em
FairFace antes da comparação** — não usar zero-shot direto.

## 9. Pontos para citar / posicionar

- *"Luo et al. (2024), publicado no CVPR 2024, introduzem FairCLIP — a
  primeira abordagem baseada em transporte ótimo para fairness em
  vision-language models — junto com o dataset Harvard-FairVLMed,
  primeiro benchmark VL médico com atributos demográficos detalhados.
  A análise comparativa de CLIP e BLIP-2 documenta vieses substanciais
  em ambos os modelos, com Asian, Male, Non-Hispanic e Spanish sendo os
  subgrupos consistentemente preferidos cross-attribute."*

- *"A abordagem FairCLIP de Luo et al. (2024), que reduz a distância de
  Sinkhorn entre a distribuição overall e as distribuições por grupo
  demográfico, alcança redução de até 80.6% na Demographic Parity
  Difference (DPD) em zero-shot transfer (gender, ViT-B/16). No
  entanto, a abordagem apresenta inconsistência: piora a DPD em race
  com ViT-L/14 (10.10 → 17.79), demonstrando que Sinkhorn não garante
  melhoria universal de fairness e motivando ablations arquiteturais
  alternativas como a investigada na presente dissertação."*

- *"A presente dissertação compara o mecanismo de conditioning via
  FiLM proposto neste trabalho com o paradigma de Sinkhorn-loss
  augmentation de FairCLIP, em ablation arquitetural conduzida no
  Capítulo 2 sobre classificação racial 7-class do FairFace —
  endereçando a recomendação metodológica registrada em reunião de
  orientação (2026-06-08)."*

## 10. Arquivos relacionados

- PDF: `pdfs/luo_2024_fairclip.pdf` (1.3 MB, baixado 2026-06-15).
- Código: github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP.
- Dataset: ophai.hms.harvard.edu/datasets/harvard-fairvlmed10k.
- Entradas relacionadas: [[radford_2021]] (CLIP base), [[perez_2018]]
  (FiLM — alternativa arquitetural), [[park_2022]] (FSCL — comparado
  na Fig 5d), [[dehdashtian_2024_fairerclip]] (RKHS alternative),
  [[bendvlm_2024]] (test-time), [[hardt_2016]] (EO_h).

## 11. Trabalhos sugeridos pelos autores (Future Work)

A seção de conclusão **não lista future work explicitamente**. O paper
encerra apenas com agradecimentos. **Ausência interpretável como
limitação metodológica** — autores não sinalizam direções subsequentes.

Direções inferidas dos resultados:

- **Generalização para outros domínios médicos** além de glaucoma.
- **Múltiplos atributos protegidos simultâneos** (Sinkhorn é
  pairwise).
- **Análise de sensibilidade** mais rigorosa dos hiperparâmetros
  (Sinkhorn weight, ε).

## 12. Análise crítica do método

### (a) Rigor formal

- **Sinkhorn distance** tem fundamentação teórica sólida (Peyré &
  Cuturi 2019). Choice over KL é justificada (simetria, desigualdade
  triangular).
- **Função objetivo (1)** é matematicamente clara mas as distribuições
  D_{(x_I,x_T,a)|f} são intratáveis — uso de batch-empirical
  distributions é assumido como suficiente sem análise de variância.
- **ES-AUC** é definida formalmente, mas a expressão (`AUC/(1 + sum|AUC
  - AUC_a|)`) **não é monotônica em fairness ideal** — métricas
  combinadas podem mascarar trade-offs.

### (b) Reprodutibilidade

- ✅ Código público em GitHub (Harvard-Ophthalmology-AI-Lab).
- ✅ Dataset público.
- ✅ Hiperparâmetros principais declarados (Adam, lr=1e-5, batch=32,
  10 epochs).
- ✅ 3 seeds com média ± desvio.
- ⚠️ Sinkhorn loss weight 1e-7 é muito pequeno — sensibilidade no
  supplementary (Figura 5b) mas não discutida.
- ⚠️ Critério de seleção de checkpoint não declarado.
- ⚠️ Tempo de treinamento e consumo de GPU não reportados (apenas
  "single V100").

### (c) Aplicabilidade ao pipeline v3.2

- **Adaptação parcial**: Sinkhorn loss pode ser adicionada ao
  ConvNeXt-T + FairFace baseline.
- **Backbone CLIP**: usar OpenAI CLIP fine-tuned em FairFace como
  Configuração D do estudo comparativo do Cap 2.
- **Texto sintético**: precisamos gerar prompts texto-image pairs no
  estilo `"a photo of a [race] person"` para usar CLIP nativo.
- **Multi-classe**: 7-class FairFace requer generalização de Sinkhorn
  para multiple groups (paper testa 2-3 grupos por atributo).
- **Risco**: pre-training CLIP em natural images carrega viés WIT —
  fine-tuning em FairFace pode ser insuficiente para neutralizar.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Sinkhorn distance vs KL | ✅ Sim — propriedades de métrica |
| Sinkhorn weight 1e-7 | ❌ Não justificada — apenas Fig 5b suplementar |
| Batch_size |B_a| = 32 | ✅ Sim — após extensive tuning |
| 3 seeds | ⚠️ Suficiente para σ alto observado? |
| Pre-training em medical antes de FairCLIP | ✅ Sim — comparação domínios |
| Não testar face recognition | ⚠️ Limita generalização |
| ES-AUC como métrica principal | ⚠️ Combinação que mascara trade-offs |

### (e) Conexão com outros papers do corpus

- [[radford_2021]] **CLIP base**: FairCLIP estende diretamente.
- [[perez_2018]] **FiLM**: paradigma alternativo — FiLM modula
  features via contexto auxiliar; FairCLIP modula loss via Sinkhorn.
- [[madras_2018]] **LAFTR**: paradigma adversarial — FairCLIP usa
  optimal transport ao invés de min-max game.
- [[park_2022]] **FSCL**: contrastive learning fair — comparado em
  Figura 5d (FairCLIP > FSCL em ES-AUC).
- [[dehdashtian_2024_fairerclip]] **FairerCLIP**: mesma família
  (debiasing CLIP) com RKHS ao invés de Sinkhorn.
- [[bendvlm_2024]] **BendVLM**: debiasing test-time vs FairCLIP
  pre-training.
- [[hardt_2016]] **EO_h**: DEOdds usado por FairCLIP deriva diretamente.
- [[kleinberg_2017]] **Impossibility**: FairCLIP **não discute** as
  três condições impossíveis simultaneamente — apenas DPD e DEOdds.
- **Implicação para nossa Cap 2**: FairCLIP entra como baseline forte
  do estudo comparativo, ao lado de FiLM-MST (nossa proposta) e
  LoRA-FAIR (Bian 2025).
