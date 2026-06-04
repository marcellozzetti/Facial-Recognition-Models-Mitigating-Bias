# Landscape — síntese transversal da literatura

> Síntese cruzada das **29 fichas** catalogadas em
> [`04_pesquisa_bibliografica/`](04_pesquisa_bibliografica/) (Rodadas
> 1, 2, 2.5, 3, 4, 5 e 2.6 da triagem editorial). Cobre fairness em
> biometria facial, com ênfase em **classificação racial**.
>
> **Atualizado em 2026-06-04** após Rodada 5 (+6 papers de mecanismos
> ML/redes neurais, Track G) e Rodada 2.6 (re-verificação SOTA).
>
> Este documento é a **base argumentativa** para `06_gap.md`
> (identificação de gaps) e `07_thesis_statement.md` (tese v3.2).

## 1. Sumário executivo

**O campo** de fairness em biometria facial é fragmentado em **4 tracks
paralelos** que raramente se cruzam, com **forte convergência sobre
o problema** ("balanceamento não basta") mas **fragmentação sobre a
solução** (cada subcomunidade usa métricas e datasets distintos).

**Posicionamento da nossa pesquisa:** opera no **track de race
classification multi-classe in-domain (FairFace 7-class)** — track
em que o ecossistema metodológico é o **menos maduro** entre os
quatro. Especificamente:

- Mitigações algorítmicas testadas: **2 de 8+ técnicas** identificadas
  (apenas VLM fine-tuning e fairness-aware pruning).
- Métrica de fairness consensual: **inexistente** (Q05).
- Cross-reference com skin tone: **inexistente** (Q10).
- Anotação validada por especialistas: **inexistente** (Q01).

Há, portanto, **densidade alta de gaps** combinada com **forte
endosso da literatura** para mais pesquisa neste track — situação
ideal para contribuição experimental original.

## 2. Topologia da literatura

### 2.1 Os sete tracks (atualizado após Rodadas 4 e 5)

```
                    ┌─ Track A: Race CLASSIFICATION ─┐
                    │  FairFace, UTKFace, LFWA+      │
                    │  ↓                              │
                    │  AlDahoul (SOTA), Lin (FairGR)  │
                    │  Karkkainen (origem)            │
                    │  ★ NOSSO TRACK                 │
                    └─────────────────────────────────┘
                                  │
                                  ↕ poucas conexões
                                  │
┌─ Track B: Race RECOGNITION ────┐
│  RFW, BFW, BUPT, NIST FRVT      │            ┌─ Track D: Mitigação ALGORÍTMICA ─┐
│  Wang, Robinson, Grother, Neto  │←── métodos │  FSCL (Park), FineFACE (Manzoor) │
│  verification 1:1, identification│            │  Group DRO (Sagawa), U-FaTE      │
└─────────────────────────────────┘            │  Bhaskaruni, FairGRAPE (Lin)     │
                                                │  testbeds: CelebA, Waterbirds    │
┌─ Track C: SKIN TONE fairness ───┐            │  ★ aplicáveis ao Track A         │
│  PPB, Casual Conv, MST-E         │            │     mas NUNCA aplicados          │
│  Buolamwini, Hazirbas, Schumann  │            └──────────────────────────────────┘
│  Lafargue (audit)                │
│  Fitzpatrick / Monk Skin Tone    │            ┌─ Track E: AUDITORIA / metodologia ┐
└──────────────────────────────────┘            │  DSAP, Mehrabi, Kotwal, Lafargue  │
                                                │  surveys + análises de bias       │
                                                └───────────────────────────────────┘
```

### 2.2 Conexões entre tracks

**Conexões fortes (vários papers em comum):**

- **A ↔ E**: Mehrabi e Kotwal cobrem Track A indiretamente; Lafargue
  e DSAP auditam datasets como FairFace.

**Conexões fracas:**

- **A ↔ B**: FairFace e RFW são frequentemente citados juntos como
  "alternativas balanceadas", mas modelos treinados em A raramente
  são testados em B (verification ≠ classification).
- **A ↔ D**: técnicas de mitigação foram desenvolvidas em D mas
  validadas em CelebA (gender), Waterbirds, MultiNLI — **não em
  FairFace race 7-class**.
- **A ↔ C**: nenhuma. **GAP CENTRAL (Q10) é exatamente conectar
  esses dois tracks.**

**Conexões inexistentes:**

- **C ↔ D**: técnicas de mitigação treinadas com skin tone como
  protected attribute são raras (FineFACE faz parcialmente em CelebA).

### 2.3 Tracks adicionais (Rodadas 4 e 5)

**Track F — Fundamentação científica de raça e tom de pele** (Rodada 4):

- [[fuentes_2019]] — AAPA Statement on Race and Racism (Am J Phys Anthropol)
- [[lewontin_1972]] — Apportionment of Human Diversity (Springer, Test-of-Time)
- [[fitzpatrick_1988]] — Validity and Practicality of Sun-Reactive Skin Types I-VI (JAMA Network)
- [[massey_martin_2003]] — NIS Skin Color Scale (Princeton)

**Função**: fundamenta teoricamente que race é **construto social,
não biológico**. Sustenta Q11 e a limitação reconhecida da tese
v3.2 §6.3. Conecta diretamente aos achados de Lewontin (85.4% da
variação genética é intra-populacional).

**Track G — Mecanismos ML / Redes Neurais** (Rodada 5):

- [[hardt_2016]] — Equality of Opportunity (NeurIPS 2016)
- [[perez_2018]] — FiLM Visual Reasoning (AAAI 2018)
- [[zemel_2013]] — Learning Fair Representations (ICML 2013, Test-of-Time)
- [[madras_2018]] — LAFTR Adversarially Fair Transferable (ICML 2018)
- [[zhang_2018]] — Mitigating Unwanted Biases with Adversarial Learning (AAAI/ACM AIES)
- [[kleinberg_2017]] — Inherent Trade-Offs in Fair Determination (ITCS)

**Função**: fornece **mecanismos formais** para o pipeline v3.2 do
orientador. Especificamente:

- **FiLM ([[perez_2018]])**: mecanismo de condicionamento neural —
  usar saída do MST classifier (10 logits) como contexto para
  modular features do race classifier. **Diretamente operacional**.
- **LAFTR ([[madras_2018]])**: prova de **fair transferência** —
  representação treinada fair em uma tarefa permanece fair em
  tarefas downstream. Sustenta extensão a face recognition (Cap 3 v3.2).
- **Hardt 2016 ([[hardt_2016]])**: paper-fonte das métricas EO_h/EOD
  usadas pelos baselines (Park, Sagawa, Manzoor).
- **Zemel 2013 ([[zemel_2013]])**: paradigma fundador (Test-of-Time
  ICML 2023). Posiciona historicamente o pipeline v3.2 na linhagem
  de Fair Representation Learning.
- **Zhang 2018 ([[zhang_2018]])**: baseline adversarial alternativo
  para Cap 2.
- **Kleinberg 2017 ([[kleinberg_2017]])**: teorema da impossibilidade —
  justifica triangulação DR + worst-class F1 + CV (C4 da tese v3.2).

## 3. Cross-reference matrix

### 3.1 O que cada paper testa (datasets × tarefa × técnica)

Codificação: ✓ = paper testa; ⚠ = paper menciona/discute mas não
testa; ○ = paper não toca.

| Ficha | FairFace | RFW | BFW | UTKFace | CelebA | LFWA+ | PPB | MST-E | Outros | Tarefa-alvo |
|---|---|---|---|---|---|---|---|---|---|---|
| [karkkainen_2021](04_pesquisa_bibliografica/dataset_karkkainen_2021.md) | ✓ (origem) | ⚠ | ○ | ✓ (cross) | ⚠ | ✓ (cross) | ⚠ | ○ | Twitter/Media/Protest | race classification 7-class |
| [aldahoul_2024](04_pesquisa_bibliografica/aldahoul_2024.md) | ✓ | ○ | ○ | ✓ (cross) | ○ | ○ | ○ | ○ | DiverseFaces | race + gender + age + emotion |
| [lin_2022](04_pesquisa_bibliografica/lin_2022.md) | ✓ | ○ | ○ | ✓ | ✓ | ○ | ○ | ○ | ImageNet (person) | race + gender + pruning |
| [manzoor_2024](04_pesquisa_bibliografica/manzoor_2024.md) | ✓ (gender) | ○ | ○ | ✓ | ✓ | ✓ | ○ | ○ | — | gender + 13 attrs |
| [dehdashtian_2024](04_pesquisa_bibliografica/dehdashtian_2024.md) | ✓ (sex) | ○ | ○ | ○ | ✓ | ○ | ○ | ○ | Folktable | sex + attribute |
| [lafargue_2025](04_pesquisa_bibliografica/lafargue_2025.md) | ⚠ (aux model) | ○ | ○ | ○ | ✓ | ○ | ○ | ○ | Generated Photos | dataset audit |
| [park_2022](04_pesquisa_bibliografica/park_2022.md) | ○ | ○ | ○ | ✓ (binarizado) | ✓ | ○ | ○ | ○ | Dogs&Cats | gender + attr |
| [sagawa_2020](04_pesquisa_bibliografica/sagawa_2020.md) | ○ | ○ | ○ | ○ | ✓ (hair color) | ○ | ○ | ○ | Waterbirds, MultiNLI | various |
| [bhaskaruni_2019](04_pesquisa_bibliografica/bhaskaruni_2019.md) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | Credit, Crime tabular | fair LR |
| [wang_2019](04_pesquisa_bibliografica/dataset_wang_2019.md) | ○ | ✓ (origem) | ○ | ○ | ○ | ○ | ○ | ○ | GBU, IJB-A | face verification |
| [robinson_2020](04_pesquisa_bibliografica/dataset_robinson_2020.md) | ○ | ○ | ✓ (origem) | ○ | ○ | ○ | ○ | ○ | — | face verification |
| [neto_2025](04_pesquisa_bibliografica/neto_2025.md) | ○ | ✓ | ✓ | ○ | ○ | ○ | ○ | ○ | (treino em VGGFace2) | face verification |
| [grother_2019](04_pesquisa_bibliografica/grother_2019.md) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | Government (18.27M) | recognition 1:1, 1:N |
| [buolamwini_2018](04_pesquisa_bibliografica/buolamwini_2018.md) | ○ | ○ | ○ | ○ | ○ | ○ | ✓ (origem) | ○ | IJB-A, Adience (comparison) | gender (audit) |
| [hazirbas_2021](04_pesquisa_bibliografica/dataset_hazirbas_2021.md) | ⚠ (compare) | ○ | ○ | ○ | ○ | ○ | ⚠ | ○ | DFDC, Casual Conv | DFDC + age/gender |
| [schumann_2023](04_pesquisa_bibliografica/schumann_2023.md) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ✓ (origem) | — | MST annotation consensus |
| [dominguez_2024](04_pesquisa_bibliografica/dominguez_2024.md) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | 20 FER datasets | dataset audit (FER) |
| [mehrabi_2021](04_pesquisa_bibliografica/survey_mehrabi_2021.md) | — | — | — | — | — | — | — | — | — | survey |
| [kotwal_2025](04_pesquisa_bibliografica/survey_kotwal_2025.md) | — | — | — | — | — | — | — | — | — | survey |

**Padrões visíveis:**

1. **FairFace é multiuso** mas raramente como race 7-class target —
   AlDahoul (race), Lin (race), Manzoor (gender), Dehdashtian (sex),
   Lafargue (aux).
2. **CelebA é DOMINANTE em Track D** (mitigação) — 5/19 fichas.
3. **MST-E só aparece no Schumann** — não há ainda dataset com MST
   anotações em escala grande.
4. **PPB só no Buolamwini** — Fitzpatrick não foi adotado pela
   literatura subsequente em race classification (CelebA virou padrão
   de Track D).

### 3.2 Métricas reportadas

| Ficha | Accuracy | F1 | EOD/EO_h | DP | Max-Min ratio | DR | Worst-class | DoB / std | CV | DSAP (DSR/DSE/DSS) | FMR / FNMR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| karkkainen_2021 | ✓ | ○ | ⚠ (eq.1) | ○ | ○ | ✓ (ε log-ratio) | ○ | ✓ | ○ | ○ | ○ |
| aldahoul_2024 | ✓ | ✓ | ○ | ○ | ⚠ (per-class) | ○ | ○ | ○ | ○ | ○ | ○ |
| lin_2022 | ✓ | ○ | ○ | ○ | ○ | ○ | ✓ | ✓ (ρA) | ○ | ○ | ○ |
| manzoor_2024 | ✓ | ○ | ○ | ⚠ | ✓ | ✓ | ⚠ | ✓ (DoB) | ○ | ○ | ○ |
| dehdashtian_2024 | ✓ | ○ | ✓ | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| park_2022 | ✓ | ○ | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| sagawa_2020 | ✓ | ○ | ○ | ○ | ○ | ○ | ✓✓ | ○ | ○ | ○ | ○ |
| bhaskaruni_2019 | ⚠ (error) | ○ | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| wang_2019 | ✓ | ○ | ○ | ○ | ⚠ | ○ | ○ | ✓ | ○ | ○ | ○ |
| robinson_2020 | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ✓ (FNR/FPR/EER) |
| grother_2019 | ○ | ○ | ○ | ○ | ⚠ (10-100×) | ○ | ○ | ○ | ○ | ○ | ✓✓ (centro do paper) |
| lafargue_2025 | ✓ | ○ | ✓ | ✓ (DI) | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| dominguez_2024 | ○ | ○ | ○ | ⚠ | ○ | ○ | ○ | ○ | ○ | ✓✓ | ○ |

**Padrão crítico:** **não há 2 papers que usem a MESMA combinação
de métricas**. Cada paper inventa adaptação ad-hoc para multi-classe.
Confirma Q05 ❌→🔬.

## 4. Achados convergentes

### 4.1 "Balanceamento de dataset não é suficiente para fairness"

**Endosso múltiplo (8+ fichas):**

- Karkkainen 2021: FairFace balanceado → Latino F1=.247 em external set.
- Wang 2019 RFW: race-balanced training → African 2× error vs
  Caucasian.
- Grother 2019 NISTIR: 189 algoritmos → diferenciais 10-100× FPR
  entre raças.
- Lin 2022: balanceado → Hispanic 59.6% baseline.
- AlDahoul 2024: balanceado → Latinx 60% F1 mesmo SOTA.
- Kotwal & Marcel 2025: survey cita 8+ referências independentes.
- Sagawa 2020: ERM com balanceamento padrão → worst-group 41.1%.
- Klare et al. (citado em Kotwal): pré-deep learning também observou.

**Implicação:** mitigação algorítmica é necessária além do dataset.
**Direta justificativa para Track D ser aplicado a Track A.**

### 4.2 "Ensembles naive não melhoram fairness"

**Endosso convergente:**

- Bhaskaruni 2019: bagging PIORA SD vs LR baseline em Community
  Crime.
- Sagawa 2020: naive Group DRO ≈ ERM em overparameterized — só
  funciona com strong regularization.
- Park 2022: SupCon **aumenta** EO sobre Cross-Entropy.

**Implicação:** se aplicarmos deep ensemble + temperature scaling +
group reweighting (H3 da Q04), precisamos **ser explícitos sobre
ponderação demográfica** — não esperar fairness "emergir" do
ensemble.

### 4.3 "Latinx/Hispanic é a classe mais difícil"

**Endosso quádruplo:**

- Karkkainen 2021 Tabela 6: Latino F1=.247 (external Twitter/Media/
  Protest set).
- Lin 2022 Tabela 2: Hispanic 59.6% accuracy baseline ResNet-34.
- AlDahoul 2024 Tabela 16: Latinx/Hispanic 60% F1 (FaceScanPaliGemma
  SOTA).
- Buolamwini 2018: argumenta que categorização Latino é
  metodologicamente problemática (não testa diretamente).

**Hipótese causal divergente:** ainda não há consenso sobre **POR
QUÊ** Latinx é difícil:

- (a) Anotação ruidosa (frente Q01 🔬).
- (b) Sobreposição fenotípica com White/Middle East (frente Q10 🔬).
- (c) Limitação do modelo (não testado por nenhum paper).

### 4.4 "Race é socialmente construída"

**Reconhecimento múltiplo:**

- Karkkainen 2021 §3.1: "Race is not a discrete concept".
- Buolamwini 2018: escolhem Fitzpatrick por race ser "unstable".
- Hazirbas 2021: argumento explícito contra race labeling.
- Neto 2025: questiona discretização per se.
- Schumann 2023: mostra subjetividade entre anotadores regionais.

**Implicação:** a escolha de 7-class FairFace é **pragmática**, não
absoluta. Q09 ❌→🔬.

### 4.5 "Cross-dataset evaluation em multi-classe race é estruturalmente
limitado"

**Endosso:**

- Karkkainen 2021 Tabela 4 rodapé: "Only 4 races were used to make it
  comparable to UTKFace".
- AlDahoul 2024: testa 6-class (East+SE merged) com 81.1% vs
  7-class 75.7%.
- RFW (4) e BFW (4) incompatíveis com FairFace (7).

**Implicação:** **avaliação 7-class fica in-domain** (FairFace
train→val). Comparações cross-dataset são forçosamente em 4
classes.

## 5. Achados divergentes

### 5.1 "Skin tone (Fitzpatrick) vs race"

| Posição | Defendido por |
|---|---|
| **Skin tone é melhor** (mais estável, biológico, mensurável) | Buolamwini 2018, Hazirbas 2021, Schumann 2023, Lafargue 2025 |
| **Race é melhor** (cultural, aplicabilidade social) | Karkkainen 2021 (explicitamente argumenta) |
| **Ambos têm limites; usar contínuo** | Neto 2025 |

Nossa pesquisa: usa **race 7-class por escolha pragmática** (dataset),
mas **proposta Q10** explicitamente conecta os dois lados — primeira
contribuição que **não escolhe** mas **cruza**.

### 5.2 "Balanceamento ideal: equidistribuído ou skewed?"

- **Equidistribuído** (intuitivo): FairFace, RFW, BFW design.
- **Skewed pode ser melhor:** Gwilliam et al. (citado em Kotwal 2025):
  distribuição enviesada para African reduziu disparities **mais
  efetivamente** que balanced.
- **Continuous balance:** Neto 2025: skewed em contínuo > balanced em
  discreto.

Implicação: **definição de "balanceado" é em si um problema aberto**.

### 5.3 "Threshold global vs subgroup-specific"

- **Global** (padrão): Karkkainen, AlDahoul, Manzoor, Lin.
- **Subgroup-specific**: Robinson 2020 (BFW paper) — **demonstra
  ganho simultâneo** de accuracy e fairness.
- **Cohort-specific models** (Klare em Kotwal): ainda mais radical.

Nossa pesquisa: usa **global** (classification softmax não tem
threshold único como verification).

### 5.4 "Multi-task vs single-task"

- **Single-task**: AlDahoul (fine-tunes separados para race/gender/
  age/emotion), Karkkainen, Lin, FairGRAPE.
- **Multi-task**: Manzoor (13 attrs simultâneos), Hazirbas (4-axis),
  Dominguez 2024 (DSAP multi-axis).

Nossa pesquisa: single-task race classification.

## 6. Trabalhos sugeridos pelos autores — síntese agregada

Compilação das **Seções 11** de cada ficha (trabalhos sugeridos pelos
autores). Direções por **número de papers que endossam**:

### 6.1 Direções com endosso **forte** (≥4 papers)

| Direção sugerida | Papers convergindo | Frente 🔬 nossa | Status na literatura |
|---|---|---|---|
| **Mitigação algorítmica em race classification 7-class** | Park, Sagawa, Dehdashtian, Bhaskaruni, Manzoor, AlDahoul, Lin (7) | **Q04** | ★ Endosso máximo, **execução zero** |
| **Métricas multi-classe formalmente robustas** | Park, Manzoor, Lin, Sagawa, Mehrabi, Lafargue, Dehdashtian (7) | **Q05** | Endosso forte; cada paper inventa adapta cão diferente |
| **Backbones modernos** (ViT, ConvNeXt) em race classification | Manzoor, Lin, Sagawa, Park, Wang (5) | **Q06** | Endosso moderado; raros papers usam |
| **Self-identification / annotation reliability** | Hazirbas, Schumann, Kotwal, Lafargue, Buolamwini (5) | **Q01** | Endosso forte; restrito ao Track C |
| **Investigar dificuldade Latinx/Hispanic** | AlDahoul, Lin, Karkkainen, Buolamwini (4) | Q01 + Q10 | Reconhecido empíricamente, não diagnosticado |

### 6.2 Direções com endosso **moderado** (2–3 papers)

| Direção | Papers | Frente 🔬 |
|---|---|---|
| Combinação de técnicas (FSCL + DRO + ensemble) | Park, Manzoor, Sagawa (3) | Q04 (H3) |
| Continuous demographic labels | Neto, Hazirbas (2) | Q09 |
| MST scale em fairness research | Schumann, Lafargue (2) | Q10 |
| Cross-dataset / generalization | Karkkainen, Wang, AlDahoul (3) | aux |
| Fairness sem demographic labels (privacy-preserving) | Sagawa, Mehrabi (2) | aux paralelo |

### 6.3 Direções **singulares** (1 paper) — possíveis frentes originais

| Direção | Paper | Nossa avaliação |
|---|---|---|
| Subgroup-specific thresholds | Robinson 2020 | Aplicabilidade limitada (verification ≠ classification) |
| Cohort-specific models | Klare em Kotwal | Direção radical; fora do nosso escopo |
| Causal fairness | Mehrabi | Direção emergente; fora do escopo experimental |
| Fairness em generative models | Mehrabi, Lafargue | Fora do escopo |
| Standards de reporting (ISO/IEC 19795-10) | Kotwal | Tangencial |

### 6.4 Direções **ausentes da literatura** (mas possíveis)

| Direção | Por que ausente | Nossa avaliação |
|---|---|---|
| **Cross-reference skin tone × race FairFace** | Tracks A e C não se cruzam; Draelos 2025 é parcial e dermatológico | **🔬 Q10 — frente CENTRAL ORIGINAL** |
| Re-anotação multi-anotador independente FairFace Latinx | Custoso; nenhum grupo academic-funded fez | 🔬 Q01 aux |
| Decomposição arquitetura/método/dados do ceiling 72% | Cada paper varia um eixo, não controla outros | 🔬 Q06 aux |
| Test set público separado FairFace | FairFace só libera train+val | Resolvido por convenção (Q03), não pesquisa |

## 7. Mapeamento literatura → frentes 🔬 da dissertação

**5 frentes consolidadas:**

| Frente | Pergunta | Endosso da literatura | Originalidade |
|---|---|---|---|
| **Q04** — Mitigação algorítmica em race 7-class FairFace | "Aplicar FSCL+/DRO/ensemble em FairFace race 7-class" | **7 papers convergindo** | **Execução original** sobre direção sugerida |
| **Q10** — Matriz skin tone × race | "Cross-reference Fitzpatrick/MST × FairFace 7-race" | **0 papers fazem** | **TOTALMENTE ORIGINAL** |
| **Q05** — Métrica fairness multi-classe | "Triangulação DR + worst-class + CV" | 7 papers reconhecem problema | Solução metodológica original |
| **Q06** — Decomposição do ceiling 72% | "ConvNeXt-T isolado vs ResNet-34" | 5 papers sugerem backbones novos | Diagnóstico controlado original |
| **Q01** — Confiabilidade anotação Latinx | "κ de Fleiss por classe" | 5 papers reconhecem | Aplicação ao FairFace original |

**Combinação chave:** Q04 (forte endosso, execução vazia) + Q10
(originalidade total) + Q05 (contribuição metodológica) formam um
**conjunto coerente** para a dissertação.

## 8. Síntese final

### 8.1 O que a literatura **estabelece com certeza**

1. **Dataset balanceado é necessário mas não suficiente** para
   fairness racial.
2. **Latinx/Hispanic é a classe mais difícil** em FairFace 7-class
   (replicado em ≥4 estudos independentes).
3. **Race é construto social**, não biológico — qualquer taxonomia
   discreta é aproximação.
4. **Naive ensembles e naive DRO não funcionam** — fairness exige
   intervenção explícita.
5. **Métricas multi-classe** são fragmentadas, sem padrão consensual.
6. **Track de race classification é metodologicamente menos maduro**
   que Track de face recognition.

### 8.2 O que a literatura **deixa em aberto**

1. **Que técnica de mitigação algorítmica funciona melhor para race
   classification 7-class FairFace?** (Q04 — 0 estudos)
2. **Por que Latinx é estruturalmente difícil?** Annotation, fenótipo
   ou modelo? (Q10 — 0 estudos)
3. **Qual métrica de fairness reportar em multi-classe?** (Q05 —
   fragmentação)
4. **Onde está o ceiling real do FairFace race 7-class?** (Q06 —
   decomposição não feita)
5. **As anotações MTurk de Latinx têm κ comparável às outras
   classes?** (Q01 — não auditado)

### 8.3 O que esta dissertação pode contribuir

**Contribuição primária — Q10 + Q04 combinadas:**

> Construir a **primeira matriz Fitzpatrick/MST × FairFace 7-race**
> publicada, **e simultaneamente** testar **mitigações algorítmicas
> de Track D** (FSCL+, Group DRO, ensemble + reweighting) sobre o
> FairFace race 7-class — gerando o **primeiro mapeamento empírico**
> de qual mitigação corrige o componente de erro modelo vs qual
> componente é **irredutível por sobreposição fenotípica
> estrutural** (informado pela matriz Q10).

**Contribuição secundária — Q05:**

> Propor e defender a **triangulação DR + worst-class F1 + CV** como
> métrica de fairness para classificação multi-classe — convertendo
> a fragmentação em diretriz prática.

**Contribuição auxiliar — Q06:**

> Decomposição experimental controlada do ceiling 72%: variação
> isolada de backbone (ResNet-34 → ConvNeXt-T), seed (single →
> 3-seed casado), HPO (default → modesto) — quantificando quanto do
> "limite" é cada componente.

---

**Próximo arquivo:** [`06_gap.md`](06_gap.md) — ranqueamento das 5
frentes 🔬 por viabilidade, originalidade e impacto; decisão final
sobre escopo experimental da dissertação.

## 9. Atualização v3.2 (2026-06-04)

Após reunião com orientador e Rodada 5 (+6 papers Track G), a tese
foi reformulada de **diagnóstica** (v3.1) para **prescritiva** (v3.2)
com integração de Q04 + Q10 em pipeline único, ancorado em FiLM
([[perez_2018]]) e LAFTR ([[madras_2018]]):

**Pipeline integrado v3.2:**

```
Classifier MST (Schumann 2023)
    ↓
Auditoria matriz P(MST | race) sobre FairFace val
    ↓
Race classifier ConvNeXt-T + FiLM(MST)  ← NOVO: condicionamento explícito
    ↓
Comparação fairness com vs sem MST
    ↓
Extensão a face recognition (RFW ou BFW)  ← NOVO: linha do orientador
    ↓
Validação fair transferência (LAFTR-style)
```

**Implicações para o landscape:**

- **Conexão A ↔ C** deixa de ser gap teórico e vira **execução
  empírica** (Cap 1 + Cap 2 v3.2).
- **Conexão A ↔ B** (race classification ↔ face recognition) torna-se
  **central** via extensão Cap 3.
- **Track G** fornece **toolkit formal** que não existia em v3.1.

Detalhamento da tese v3.2 em [`07_thesis_statement.md`](07_thesis_statement.md).
Plano experimental detalhado em [`06_gap.md`](06_gap.md) §4.
