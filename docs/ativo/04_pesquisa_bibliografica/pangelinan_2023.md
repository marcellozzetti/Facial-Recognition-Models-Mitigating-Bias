---
name: pangelinan-2023
status_verificacao: VERIFIED
autores: [Gabriella Pangelinan, K.S. Krishnapriya, Vitor Albiero, Grace Bezold, Kai Zhang, Kushal Vangara, Michael C. King, Kevin W. Bowyer]
ano: 2023
titulo: "Exploring Causes of Demographic Variations In Face Recognition Accuracy"
venue: "arXiv preprint (capítulo book-style, abr 2023) — Florida Institute of Technology + University of Notre Dame"
tipo_publicacao: preprint
arxiv_id: "2304.07175"
doi: null
url_primario: https://arxiv.org/abs/2304.07175
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: 34
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/pangelinan_2023.pdf), 34 páginas, lido em 2026-06-15.
---

# Exploring Causes of Demographic Variations In FR Accuracy (Pangelinan, Krishnapriya, Albiero, Bezold, Zhang, Vangara, King, Bowyer — 2023)

> **Revisão crítica em 4 experimentos** das causas comumente especuladas
> para a disparidade demográfica em face recognition. Conclui que
> **skin tone isolado, face geometry e balanced training NÃO explicam**
> o gender/race gap; **pixel information (face-pixel fraction) explica
> o FNMR gap** mas o FMR gap persiste. **Esta é a refutação central que
> reformulamos via Hipótese H6** — variance pixel info × skin tone como
> contribuição quantitativa da tese.

## 1. Resumo do problema atacado

Após a explosão de pesquisa sobre fairness em FR pós-Gender Shades
(2018) e o NIST FRVT Part 3 (2019), surgiram várias hipóteses sobre as
causas das disparidades observadas:

1. **Skin tone**: faces mais escuras causam pior FR?
2. **Face geometry**: diferenças de width/height entre gêneros causam o
   gender gap?
3. **Balanced training data**: balancear treino elimina o gap?
4. **Pixel information**: faces femininas têm menos pixels úteis
   (devido a hairstyle/morfologia)?

O paper conduz **4 experimentos** isolando cada fator e remove
confounders. **Achado central**: três das quatro hipóteses não se
sustentam isoladamente; apenas pixel information explica parcialmente
o FNMR gap.

## 2. Método

### 2.1 Baseline experimental (Seção 0.2)

Replicação do estudo de Albiero et al. (2021) [14]:
- **Matcher**: ArcFace [15] (public weights) treinado em MS1Mv2
- **Datasets curados**:
  - **MORPH 3** (Caucasian + African American, mugshot-style controlado)
  - **Asian-Celeb** (web-scraped, com curation para pose e duplicates)
- **Composição** (Tabela 0.1):

| Demographic | Female Imgs/Subjs | Male Imgs/Subjs |
|---|---|---|
| Caucasian | 10.941 / 2.798 | 35.276 / 8.835 |
| African American | 24.857 / 5.929 | 56.245 / 8.839 |
| Asian | 43.356 / 6.083 | 73.376 / 12.673 |

- **Achado replicado**: em **todos** os 3 grupos raciais, **female
  impostor** está deslocada para scores maiores (maior FMR) e
  **female genuine** está em scores menores (maior FNMR).

### 2.2 Experimento 1 — Skin tone isolado (Seção 0.3)

Estudo de Krishnapriya et al. [24, 28] em MORPH **African American Male
(AAM) cohort**:

- **Setup**: comparar skin tone distribution entre:
  - **HST** (High-Similarity Tail) — pares acima do 1-in-10k FMR threshold
  - **Center** — pares acima do 1-in-2 FMR threshold
- **Hipótese a testar**: se skin tone causa FMR, HST tem mais imagens
  com skin tone IV-VI
- **Anotação**: 3 raters independentes → escala FST 1-6
- **2 matchers**: ArcFace + VGGFace2
- **Inter-annotator agreement**: 93-96% de imagens com agreement de 2+ raters

Extensão (Seção 0.3.2):
- **Color correction** via 18% gray background do MORPH
- **6 manual raters** + **automated rating** via ITA (Individual Typology Angle)

### 2.3 Experimento 2 — Face geometry (Seção 0.4)

- **Dataset**: FRGC 3D+2D (Notre Dame), 1618 imagens (583 ♀, 480 ♂)
- **Landmarks**: ex (outer eye corner), n (nasion), gn (gnathion)
- **Medidas**:
  - ex-ex (face width): ♂ 89.4 mm (SD 3.6) vs ♀ 86.8 mm (SD 4.0)
  - n-gn (face height): ♂ 121.3 mm (SD 6.8) vs ♀ 111.8 mm (SD 5.2)
  - aspect ratio: ♂ mais elongados, ♀ mais arredondados
- **3 sub-experimentos**:
  1. Balancear face width entre ♀/♂
  2. Balancear face height SD via remoção iterativa de outliers
  3. Balancear aspect ratio (pareamento ♀-♂ com diff ≤ 0.025)

### 2.4 Experimento 3 — Balanced training data (Seção 0.5)

Estudo Albiero et al. [38, 40]:
- **Backbone**: ResNet-50
- **3 loss functions**: softmax, combined margin, triplet
- **Training datasets**: VGGFace2 e MS1Mv2
- **7 subsets por dataset** (Tabela 0.5):
  - Full / Balanced / F100 / M25F75 / M50F50 / M75F25 / M100
- **Testing**: MORPH Caucasian, MORPH African American, Notre Dame, AFD
- **Métrica**: TAR @ FAR=0.001%

### 2.5 Experimento 4 — Pixel information (Seção 0.6)

Estudo Albiero et al. [14]:
- **Segmentation**: BiSeNet → binary mask (face vs non-face pixels)
- **Face pixels** = skin + eyebrows + eyes + ears + nose + mouth
- **Não-face** = neck + hair
- **Análise**: distribution do face-pixel fraction por gênero
- **Information-equalized dataset**:
  1. Mascarar área fora do 10% level no female heatmap
  2. Pairing IoU-maximizado: para cada ♀, escolher o ♂ com maior IoU de face pixels
- **Comparação**: FMR/FNMR antes vs após equalização

## 3. Datasets e setup experimental

- **MORPH 3**: mugshot-style, controlled lighting, 18% gray background, frontal pose, neutral expression
- **Asian-Celeb**: web-scraped, curado
- **FRGC**: 3D+2D Notre Dame
- **MS1Mv2** e **VGGFace2**: training
- **Notre Dame** e **AFD**: testing
- **Sem cross-validation explícita** — single train/test split
- **Sem múltiplos seeds** reportados em Albiero [14, 40]

## 4. Métricas reportadas

- **FMR** (False Match Rate) — taxa de falso aceite
- **FNMR** (False Non-Match Rate) — taxa de falso rejeito
- **d-prime** — separação impostor/genuine distributions
- **TAR @ FAR=0.001%** — Tabela 0.6
- **Acc Male/Female/Avg** por subset de balancing (Tabela 0.7)
- **HST sample** count por skin tone (Figuras 7, 10)
- **Face pixel fraction** percentile por gênero (Figuras 22, 25)

## 5. Resultados principais (valores numéricos)

### 5.1 Skin tone NÃO é causa única do FMR

**Figura 7** (color-corrected):
- ArcFace AAM impostor — Skin tone V mais frequente em **center e HST**
- Distribuição praticamente idêntica entre center e HST
- **Resultado**: nenhum shift em direção a tons mais escuros nas HST

**Quote-chave** (Seção 0.3.3): *"darker skin tone, in and of itself,
does not cause increased FMR"*.

### 5.2 Face geometry NÃO explica gender gap

- Balancear face width (Fig 16): impostor/genuine **inalteradas**
- Balancear n-gn SD (Fig 17): female impostor **continua** com scores maiores
- Balancear aspect ratio (Fig 19): female distributions **continuam** mais próximas (pior d-prime: 10.94 vs male 11.71)

**Conclusão (Seção 0.4.3)**: *"no clear evidence that the observed facial differences contribute to the gender gap"*.

### 5.3 Balanced training data NÃO equaliza accuracy

**Tabela 0.6 — TAR@FAR=0.001%, MS1Mv2 + Combined Margin**:

| Dataset | Subset | ♂ Acc | ♀ Acc | Avg |
|---|---|---|---|---|
| MORPH C | Full | 99.81 | 99.39 | 99.60 |
| MORPH C | Balanced | 99.65 | 99.09 | 99.37 |
| MORPH AA | Full | 99.93 | 99.70 | 99.82 |
| MORPH AA | Balanced | 99.88 | 99.59 | 99.74 |
| Notre Dame | Full | 100 | 99.97 | 99.99 |
| Notre Dame | Balanced | 100 | 99.99 | 100 |

**Achado contraintuitivo**: Full (imbalanced) ≥ Balanced em quase
todos os casos.

**Tabela 0.7 — 75%-male training dá melhor average accuracy**:

| Test | F100 | M25F75 | M50F50 | M75F25 | M100 |
|---|---|---|---|---|---|
| MORPH C avg | 94.31 | 98.08 | **98.35** | 97.86 | 97.22 |
| MORPH AA avg | 96.57 | 99.14 | 99.08 | **99.18** | 98.85 |
| Notre Dame avg | 99.74 | 99.98 | 99.97 | **99.99** | 99.93 |

**M75F25** (75% male) frequentemente bate M50F50. **Refuta** a
hipótese "balancear training basta".

### 5.4 Pixel information EXPLICA o FNMR gap

**Face-pixel fraction (Figura 22)**:
- ♀: 25-45% (MORPH), 20-55% (Asian-Celeb)
- ♂: 45-70% (MORPH), 55-80% (Asian-Celeb)
- **Gap consistente** ~20-30 pp em todos os grupos raciais

**Após equalização (Figura 25)**:
- **female FNMR torna-se ≥ male FNMR** em MORPH AA e MORPH Caucasian
- **female FMR continua pior** em todos grupos
- Conclusão: *"the female genuine distribution is as good as (or better than!) the male genuine distribution"* quando pixels são equalizados

**FMR gap persiste** (Figura 26):
- Female clusters formam em distâncias menores que male
- Implica: *"images of two different females are more similar than images of two different males"*
- **FMR gap NÃO é explicado** por pixel info — fica em aberto

### 5.5 Síntese (Tabela 0.2 — meta-análise das 6 publicações comparadas)

| Reference | FMR | FNMR | Datasets |
|---|---|---|---|
| Grother 2017 | AA better | C better | 300K |
| Cook 2019 | C better | N/A | 6K |
| Howard 2019 | N/A | C better | 6K |
| Krishnapriya 2019 | AA better | C better | 45K (Yes, available) |
| Wang 2019 | N/A | C better | 20K |
| Grother 2019 | C better | AA better | Multiple |

**Inconsistência cross-paper**: AA vs C accuracy varia conforme dataset
e matcher.

## 6. Limitações declaradas pelos autores

Citações diretas (Seção 0.7 Discussion):

- *"there is no complete or definitive answer to why face recognition performance varies across race and gender groups"*
- FMR gap entre ♀/♂ **continua inexplicado** mesmo após pixel equalização
- *"the issue requires further study"*
- *"In practice, test data is generally not evaluated for 'fairness'"* — assumption tacitamente aceita pela maioria da literatura

## 7. Limitações que identifiquei (leitura crítica)

- **Skin tone testado apenas em AAM cohort** — não testa cross-race com
  skin tone isolado (e.g., Caucasian skin tone V vs African American skin
  tone V). **Generalização limitada**.
- **FST 6-class** (não MST 10-class) — escala **menos granular** que a
  nossa.
- **3 testing datasets** (MORPH, Notre Dame, Asian-Celeb) — **não inclui
  FairFace**. Generalização para in-the-wild questionable.
- **Verification task** (FMR/FNMR), não **classification 7-class** —
  diferente da nossa tarefa central.
- **Datasets controlled** (MORPH mugshot-style, FRGC studio) — **não
  in-the-wild**. Resultados podem não transferir.
- **Sem modelo causal formal** — apenas correlations e ablations
  descritivas.
- **Sem 3-way interaction** (skin tone × race × gender).
- **Sem controle de idade** — idade pode confundir todos os achados.
- **Single matcher analysis primário** (ArcFace) — VGGFace2 é apenas
  baseline secundário.
- **Sem replicação multi-seed** das figuras críticas (5.1, 7, 17, 25).

## 8. Relação com nossa pesquisa

### 8.1 Conflito direto com nossa narrativa inicial (RESOLVIDO via H6)

A versão preliminar da tese **implicava skin tone como causa direta** de
disparidade racial. Pangelinan refuta essa simplificação. **Resposta
defensiva mapeada**:

- **H5 reformulada em H5 + H6**:
  - **H5**: ConvNeXt-T + FiLM condicionado em MST reduz disparidade
    cross-race em 7-class FairFace
  - **H6**: parte da disparidade é explicada por **face-pixel info
    variance** (Pangelinan); FiLM pode compensar parcialmente
- **Contribuição quantitativa**: decomposição da variance entre skin
  tone (sinal) e pixel info (confounder) — transforma a refutação em
  **contribuição original**

### 8.2 Diferenças de tarefa que protegem nossa tese

| Aspecto | Pangelinan | Nossa tese |
|---|---|---|
| Tarefa | Verification (FMR/FNMR) | Classification 7-class |
| Atributo | Race binário (AA vs C) ou Gender | Race 7-class + MST 10-class |
| Dataset | MORPH (controlled) | FairFace (in-the-wild) |
| Backbone | ArcFace 512-d | ConvNeXt-T |
| Conditioning | Nenhum | FiLM via MST |

Nossa tese explora **diferentes** condições onde skin tone pode ser
sinal útil (classification) vs onde Pangelinan testou (verification).

### 8.3 Suporte forte para Cap 3 (Metodologia)

Pangelinan motiva inclusão explícita no nosso protocolo de:

1. **Color correction** quando aplicável (rejeitado em FairFace por ser
   in-the-wild, mas reconhecer limitação)
2. **Quality control de imagem** (occlusion, illumination)
3. **Cross-pixel-fraction analysis** como diagnóstico paralelo
4. **Reportar nossas disparidades 7-class versus gender gap** observado
   por Pangelinan

### 8.4 Refutação parcial à mitigação por balanceamento

Pangelinan mostra que **balanced training NÃO equaliza accuracy** — em
alguns casos 75%-male é melhor que 50/50. Isto **suporta nossa
contribuição C5** (LAFTR-style representation learning) — balanceamento
sozinho não basta; precisa-se de mitigation algorítmica.

### 8.5 Comparação direta de magnitude

Pangelinan reporta **gender gap em d-prime** ~1-2 pontos. Nossa
disparidade max/min em 7-class (FaceScanPaliGemma) é F1 90/60 = **1.5×
ratio**. Magnitudes comparáveis sugerem que nossa contribuição operará
em **mesma ordem de grandeza** que os achados de Pangelinan.

## 9. Pontos para citar / posicionar

- *"Pangelinan et al. (2023) conduzem revisão crítica em quatro
  experimentos isolados das causas comumente especuladas para a
  disparidade demográfica em face recognition, demonstrando que skin
  tone isolado, face geometry e balanceamento de dados de treino não
  explicam a disparidade observada. O estudo identifica pixel
  information como variável explicativa parcial do gender gap em FNMR,
  mas reporta que o gap em FMR persiste mesmo após equalização — uma
  lacuna que motiva a investigação proposta nesta dissertação."*

- *"Em contraponto à narrativa simplificada de que 'face recognition é
  menos acurada para tons de pele mais escuros', Pangelinan et al.
  (2023, arXiv:2304.07175) reportam, com base em estudos de
  Krishnapriya et al. (2019, 2022) sobre o cohort African American Male
  do MORPH, que skin tone em si não desloca significativamente a
  distribuição impostor — invalidando a hipótese de causalidade direta.
  Esta dissertação reformula a hipótese original em duas: H5 (skin
  tone via MST como sinal de condicionamento para classification 7-class
  em FairFace) e H6 (pixel-information variance como confounder a ser
  quantificado), transformando o ponto de tensão metodológica em
  contribuição quantitativa."*

- *"O achado de Albiero et al. (2020a, 2020b), revisado por Pangelinan
  et al. (2023), de que datasets de treino com 75% de imagens masculinas
  resultam em maior acurácia média que datasets perfeitamente
  balanceados sustenta a justificativa para mitigação algorítmica
  (representation learning, FSCL, Group DRO) além de pré-processamento
  por balanceamento de dados — direção adotada nesta dissertação."*

## 10. Arquivos relacionados

- PDF: `pdfs/pangelinan_2023.pdf` (3.9 MB, 34 páginas, baixado 2026-06-15).
- arXiv: arxiv.org/abs/2304.07175
- Entradas relacionadas: 
  - [[dataset_karkkainen_2021]] FairFace (citado como ref 35 mas não usado experimentalmente por Pangelinan)
  - [[buolamwini_2018]] Gender Shades (contexto histórico)
  - [[grother_2019]] NIST FRVT Part 3 (citado como motivação)
  - [[schumann_2023]] MST scale (citado como Doshi 2022)
  - [[neto_2025]] (mesma família de refutação: balanceamento não basta)
  - [[perez_2018]] FiLM — solução arquitetural que Pangelinan não testou
  - [[hardt_2016]] EO/EOD — métricas que Pangelinan não usa (foca FMR/FNMR)

## 11. Trabalhos sugeridos pelos autores (Future Work)

Seção 0.7 Discussion lista direções implícitas:

- **Investigar FMR gap inexplicado** após pixel equalização. ✅
  **Alinhado parcialmente com nossa tese** — exploramos como
  conditioning afeta classification (não verification)
- **Avaliar 'fairness do test data'** em si — não apenas attribuir
  bias ao modelo. ✅ Cap 3 deve declarar qualidade do FairFace test set
- **Cross-demographic study com mais matchers** (apenas ArcFace
  primário). ❌ Fora do escopo direto, mas Cap 2 ablation cobre
  parcialmente

## 12. Análise crítica do método

### (a) Rigor formal

- **Experimentos descritivos** sem modelo causal formal explícito (e.g.,
  DAG, causal inference framework)
- **Comparações via distribuições** (impostor/genuine) e correlations —
  rigoroso para refutar hipóteses simples, **insuficiente para
  estabelecer causalidade**
- Inter-annotator agreement reportado (93-96%) — bom rigor experimental
- Color correction step matematicamente justificada via 18% gray
  background

### (b) Reprodutibilidade

- ✅ **Datasets públicos** parcialmente (MORPH disponível com NDA;
  Asian-Celeb web-scraped reproducible)
- ❌ **Código não declarado** publicly available
- ⚠️ **Hiperparâmetros parciais** — ResNet-50 + combined margin
  declarado, mas lr, batch, scheduler **não estão na main paper**
- ❌ **Single seed** — sem desvios padrão, sem intervalos de confiança
- ⚠️ **Critério de checkpoint não declarado**

### (c) Aplicabilidade ao pipeline v3.2

- **Direta**: 4 hipóteses refutadas reframan nosso storytelling para Cap 1
- **Indireta**: BiSeNet segmentation aplicável ao FairFace para
  análise auxiliar de pixel info
- **H6 quantification**: nossa contribuição pode estimar **variance
  decomposition** (R² entre MST e pixel info no FairFace)

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| MORPH como controlled benchmark | ✅ Sim — único mugshot-style large-scale público |
| Asian-Celeb apesar de noisy | ⚠️ Reconhecido pelos próprios autores |
| 3 manual raters para skin tone | ✅ Sim — inter-rater agreement reportado |
| FST 6-class (não MST 10-class) | ⚠️ Histórico — MST de 2022, paper de 2023 |
| Single seed | ❌ Não justificada — limitação aceita |
| Apenas verification task | ⚠️ Reflete uso real de FR, mas limita generalização |
| BiSeNet sem fine-tuning | ⚠️ Generic segmentation pode subestimar pixels |

### (e) Conexão com outros papers do corpus

- [[buolamwini_2018]] **Gender Shades**: paper que motivou o campo;
  Pangelinan refuta a narrativa simplificada que dele decorreu
- [[grother_2019]] **NIST FRVT Part 3**: explicit motivation; NIST
  reportou disparidades mas declarou "did not analyze cause and effect"
- [[schumann_2023]] **MST**: citado como evidência de "growing desire
  for greater inclusivity"; **Pangelinan não usa MST**
- [[madras_2018]] **LAFTR**: paradigma alternativo (representation
  learning) — não testado por Pangelinan
- [[perez_2018]] **FiLM**: mecanismo que Pangelinan poderia ter testado
  mas não testou — **direção original da nossa tese**
- [[park_2022]] **FSCL**: fair contrastive — não comparado por
  Pangelinan
- [[neto_2025]] **discretização**: linha de raciocínio paralela
  refutando balanceamento como solução
- **Implicação para nossa Cap 1**: Pangelinan é a **referência canônica**
  para fundamentar "balanceamento não basta" e "skin tone não é causa
  isolada" → motivação da tese
