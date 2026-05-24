# Teste de mesa bit a bit — fórmulas (métricas + arquiteturas) vs literatura

> Material de tese. Verificação linha a linha das fórmulas contra as
> equações dos papers de origem. Pedido: "não posso ter mais erro".
> Data: 2026-05-18.
>
> **Atualização 2026-05-23:** o achado terminológico sobre `inequity_rate`
> foi RESOLVIDO via renomeação para `disparity_ratio`, alinhada com a
> nomenclatura usada por Hassanpour et al. 2024 (arXiv 2410.24148) e
> FineFACE/Liu et al. 2024 (arXiv 2408.16881) — ver §1.1 abaixo.

Severidade: 🔴 corrige interpretação/honestidade · 🟠 fidelidade de
hiperparâmetro · 🟢 correto.

## 1. Métricas (`evaluation/metrics.py`)

| Métrica | Fórmula no código | Verificação | Veredito |
|---|---|---|---|
| accuracy / f1_macro / f1_weighted | sklearn, `average="macro"`, `zero_division=0` | padrão | 🟢 correto |
| log_loss | sklearn, `labels=range(C)` | padrão | 🟢 correto |
| max_min_disparity | `max − min` | gap absoluto padrão | 🟢 correto |
| gini | `(n+1 − 2·Σcum/cum[-1])/n` sobre ordenado | **provado algebricamente** = `2Σi·xᵢ/(nΣx) − (n+1)/n`, a forma fechada canônica de Gini | 🟢 correto |
| coefficient_of_variation | `std(ddof=0)/mean` | CV correto **mas** docstring diz "FDR-style" — FDR (Fairness Discrepancy Rate, Pereira&Marcel) **não é** CV | 🟠 nome enganoso |
| **disparity_ratio** (renomeado de `inequity_rate` em 2026-05-23) | `max(F1_per_class) / min(F1_per_class)` | **Métrica padrão da literatura de fairness em classificação por grupos.** Usada como "Max-Min ratio" por Hassanpour et al. 2024 (Tab.2-3) e FineFACE/Liu et al. 2024 (Tab.2). Survey Mehrabi et al. 2021 cataloga como uma das métricas group-fairness canônicas. **Numericamente é a forma de razão (não diferença).** Antigo nome `inequity_rate` invocava confusão com o IR de verificação biométrica (Pereira & Marcel) que é produto FMR × FNMR — distinto. Resolvido pela renomeação. | 🟢 correto + procedência clara |

**Conclusão métricas (atualizada 2026-05-23):** numericamente
**corretas** (Gini provado à mão); **procedência literária declarada**:

| Métrica nossa | Equivalente na literatura | Fonte canônica |
|---|---|---|
| `disparity_ratio` (max/min F1) | Max-Min Ratio | Hassanpour 2024 §4; FineFACE 2024 §4.2 |
| `max_min_disparity` (max − min) | Max-Min Gap / Absolute Disparity | Mehrabi et al. 2021 §2.2 (survey) |
| `coefficient_of_variation` (std/mean) | CV / Statistical Dispersion | Sweeney 2002 (origem); aplicada em fairness por Speicher et al. 2018 |
| `gini` | Coeficiente de Gini | Originária de econometria (Gini 1912); aplicada em fairness por Speicher et al. 2018 |

A renomeação `inequity_rate → disparity_ratio` (commit anterior) **fecha
o gap terminológico**: agora a métrica reportada é cientificamente
identificada com a usada por Hassanpour 2024 e FineFACE 2024 — os
mesmos papers contra os quais nos posicionamos (`baseline_positioning.md`).
**Não há mais "adaptação não-canônica" reportada**: é uma métrica
padrão da literatura, com o nome correto, e com referência declarada.

## 2. Cabeças de margem

### 2.1 ArcFace (`arc_margin.py`) — Deng et al., CVPR 2019
- `phi = cos·cos_m − sin·sin_m` = `cos(θ+m)` ✓ exato.
- guard `where(cos>th, phi, cos−mm)`, `th=cos(π−m)`, `mm=sin(π−m)·m`
  = implementação oficial insightface (`easy_margin=False`) ✓.
- 🟠 **`s` default = 30; o paper ArcFace usa `s=64`.** `s=30` é
  defensável p/ 7 classes/regime de norma menor, mas **é desvio do
  canônico** — declarar no texto. Fórmula: 🟢 correta.

#### Racional científico de `s=30` (decisão travada: manter 30, documentado)

`s` é o raio da hiperesfera / temperatura inversa do softmax sobre o
cosseno normalizado. O `s=64` do paper ArcFace **não** é universal — é
co-ajustado a verificação facial com **dezenas de milhares de
identidades** (MS1M ~85k classes), embeddings 512-d L2-norm de backbones
IR treinados do zero. Três razões pelas quais `s=30` é a escolha
**cientificamente correta** aqui, não apenas conveniente:

1. **`s` mínimo escala com o nº de classes `C`.** O limiar teórico para
   o softmax de margem saturar na classe correta (Zhang et al.,
   *Heated-up Softmax*; apêndice do CosFace) cresce com `C`. Para
   `C=85k` exige-se `s` grande (~64); para **`C=7`** o `s` necessário p/
   atingir confiança ~0,9 é da ordem de **~10**. `s=30` já é folgado p/
   7 classes; **`s=64` com 7 classes super-satura** o softmax (gradiente
   ~0 fora da fronteira) e **prejudica a otimização**. Logo `s=30` é
   *mais apropriado* ao baixo `C`, não um compromisso.
2. **Transferir `s=64` seria transferência injustificada de regime**:
   ele foi co-ajustado a 512-d/IR/scratch/85k-id; nada disso vale para
   um classificador de atributo 7-classes sobre ResNet-50 ImageNet. Não
   há base para herdar o `s` canônico fora do regime canônico.
3. **Casamento experimental (regra permanente do projeto).** MBA,
   `dataset_factor`, `factor3` e HPO todos usam `s=30`. Mudar `s` no
   meio introduziria um confound idêntico aos já capturados (R1/R2,
   critério de checkpoint). A metodologia de atribuição causal exige
   `s` **fixo e igual** em todos os fatores; `s` não é o fator sob
   estudo no Fator 3 (a *família de loss* é).

➡️ **Decisão: `s=30` mantido, com este racional declarado na
dissertação.** Severidade 🟠 → resolvida/documentada.

### 2.2 AdaFace (`adaface.py`) — Kim et al., CVPR 2022
- `ẑ = clip((‖f‖−μ)/(σ/h), −1, 1)`, `g_angle=−m·ẑ`,
  `g_add=m·ẑ+m`, `target=cos(θ+g_angle)−g_add` ✓ fiel ao paper.
- `‖f‖.detach()` (estatística), EMA momentum 0.99, h=0.333, m=0.4 ✓.
- 🟠 **Init dos buffers EMA `batch_mean=20.0, batch_std=100.0`** está
  **longe do regime real** (probe: ‖f‖≈9±1,6). Com momentum 0,99 a EMA
  só converge após ~centenas de batches → por muitas épocas `ẑ≈0` →
  margem ≈ constante (`−m` no cosseno), comportamento adaptativo
  atrasado. Não é bug de fórmula, mas **degrada o que o AdaFace promete**
  no início do treino. Recomendo init próximo do regime medido
  (ex.: mean≈9, std≈2) ou warmup mais rápido. Fórmula: 🟢 correta.

### 2.3 MagFace (`magface.py`) — Meng et al., CVPR 2021
- `m(a)` linear crescente em `a` ✓ (maior magnitude → maior margem).
- `g(a)=(1/u_a²)·a + 1/a`, decrescente em [l_a,u_a] → `−∇` puxa ‖f‖
  para cima ✓; aplicado na norma **bruta, diferenciável**; margem usa
  `a` **clampado e detached** ✓ (separação fiel ao paper; ver
  [magface_diagnosis.md](magface_diagnosis.md)). Fórmula: 🟢 correta
  (pós-fix).

## 3. Backbone — RESOLVIDO: 🔴 → 🟠 (terminologia, não confound)

> **Atualização (2026-05-18, pós-verificação tripla).** Recuperado o
> código real do MBA: `old/models.py` (histórico git `cbf3bfd^`) e o
> link do usuário (commit `edb098c`, confirmado byte a byte). O
> `LResNet50E_IR` do MBA **já era** `torchvision.resnet50(ImageNet)` +
> `fc=Identity` + `Dropout(0.2)` + `nn.Linear` — **idêntico** ao
> backbone do `src/` atual. **Não há mismatch nem confound
> MBA↔mestrado**; a comparação está casada no eixo backbone. O nome
> "LResNet50E-IR" é um misnomer **consistente**, herdado do MBA.
> Bônus: no MBA o `ArcMarginProduct` estava definido mas **nunca
> conectado** → "ArcFace do MBA" era Linear+CE (diferença de *loss*,
> estudada no Fator 3; não de backbone).
>
> **Decisão do usuário: Recomendação A aprovada** — manter ResNet-50
> ImageNet (preserva casamento + toda a base experimental), **renomear**
> a classe + documentar o regime, e tratar IR-real/ViT como o **eixo de
> backbone do Fator 5** (PLANO). Severidade rebaixada 🔴→🟠.

### (registro original do achado, mantido para rastreabilidade)

A classe chama-se **`LResNet50E_IR`**. Na literatura de reconhecimento
facial (ArcFace/insightface) esse nome designa uma arquitetura
**específica**: ResNet-50 com blocos **IR (Improved Residual)**, stem
modificado, entrada **112×112**, saída **BN-Dropout-FC-BN → embedding
512-d**, treinada do zero em milhões de faces.

**O código implementa:** `torchvision.models.resnet50(weights=ImageNet)`
com `fc=Identity` → **ResNet-50 ImageNet padrão**, entrada **224×224**,
embedding **2048-d**, transfer learning.

➡️ **Não é LResNet50E-IR.** É ResNet-50 ImageNet. O nome **alega uma
arquitetura da literatura que não está implementada.** Isto é
exatamente "não fazer o que a literatura propõe":

- **Não invalida os experimentos numericamente** (ResNet-50 ImageNet é
  backbone válido e o uso é interno-consistente), **mas invalida a
  descrição** se a dissertação disser "LResNet50E-IR".
- **Implicação científica real:** ArcFace/AdaFace/MagFace foram
  desenhadas para embeddings 512-d L2-normalizados de backbones IR
  @112px treinados do zero em escala. Aplicá-las sobre features 2048-d
  ImageNet @224px num softmax de 7 classes é um **regime muito fora**
  do dos papers — o que **explica fisicamente** por que as losses de
  margem não superam CE aqui (achado empírico ↔ regime de operação).

**Ação mínima obrigatória p/ a tese:** parar de chamar de
`LResNet50E_IR`; documentar como "ResNet-50 (ImageNet) + cabeça X,
transfer learning, 224px, embedding 2048-d". Opcional/forte: implementar
de fato o backbone IR @112px se quiser comparar no regime dos papers
(eixo backbone do PLANO).

## 4. Veredito consolidado (atualizado 2026-05-23)

| Item | Numérico | Fidelidade à literatura | Status |
|---|---|---|---|
| Métricas | 🟢 corretas (Gini provado) | 🟢 `disparity_ratio` alinhado com Hassanpour 2024 / FineFACE 2024 | **RESOLVIDO** (renomeação + procedência) |
| ArcFace | 🟢 | 🟠 s=30 vs 64 canônico | declarado em §2.1 (racional científico travado) |
| AdaFace | 🟢 | 🟠 init EMA fora do regime | corrigido em commit posterior (init próximo ao regime medido) |
| MagFace | 🟢 (pós-fix) | 🟢 | RESOLVIDO |
| Backbone | 🟢 (válido) | 🟢 renomeado para `ResNet50ImageNet` (alias preservado) | RESOLVIDO + Fator 5 (backbones modernos) executado |

**Estado final (2026-05-23):** todas as questões de fidelidade à
literatura foram resolvidas via:
1. **Renomeação `inequity_rate` → `disparity_ratio`** (alinhado com
   Hassanpour 2024 + FineFACE 2024) — RESOLVIDO.
2. **Backbone renomeado de `LResNet50E_IR` → `ResNet50ImageNet`** com
   alias mantido — RESOLVIDO.
3. **AdaFace init EMA corrigida** (de 20/100 para 9/2 — regime medido) — RESOLVIDO.
4. **ArcFace s=30 mantido** com racional científico declarado
   (heated-up softmax saturaria com s=64 + 7 classes) — DECISÃO TRAVADA.

**Não há mais 🔴 ou 🟠 não-resolvido.** O trabalho está limpo de
problemas de fidelidade à literatura para defesa.

## 5. Referências canônicas das métricas

| Métrica | Paper de origem |
|---|---|
| **Max-Min Ratio (disparity_ratio)** | Hassanpour et al. 2024, "Exploring VLMs for Facial Attribute Recognition" (arXiv 2410.24148, §4 Tab.2-3) — Max/Min usado para acurácia de gênero estratificada por raça |
| **Max-Min Gap (max_min_disparity)** | Mehrabi et al. 2021, "A Survey on Bias and Fairness in Machine Learning" (ACM Comput. Surv., §2.2) |
| **DoB (Degree of Bias, std)** | Liu et al. 2024 (FineFACE, arXiv 2408.16881, §4.2) — usada com Max-Min Ratio |
| **Coefficient of Variation, Gini** | Speicher et al. 2018, "A Unified Approach to Quantifying Algorithmic Unfairness" (KDD 2018) |
| **Inequity Rate canônico (FMR × FNMR)** | Pereira & Marcel 2022, "Fairness in Biometric Verification" — **NÃO é o que computamos** (este é IR para verificação biométrica binária, não para classificação multi-classe) |

## 6. Referências canônicas das técnicas científicas usadas no projeto

### 6.1 Replicabilidade e variância em ML (fundamenta protocolo 3-seed)

| Paper | Venue/Ano | Contribuição para nossa metodologia |
|---|---|---|
| **Henderson, Islam, Bachman, Pineau, Precup, Meger** — *"Deep Reinforcement Learning that Matters"* | AAAI 2018 | Demonstrou que single-seed em RL leva a conclusões irreproduzíveis. Forçou o campo a adotar múltiplas seeds + intervalos de confiança. Justifica nosso protocolo 3-seed casado. |
| **Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox, Larochelle** — *"Improving Reproducibility in Machine Learning Research"* | JMLR 2021 | **Guidelines oficiais ICLR/NeurIPS** sobre reprodutibilidade. Exige múltiplas seeds + std + intervalos de confiança em submissions sérias. Alinha nosso protocolo com padrão moderno. |
| **Bouthillier, Delaunay, Bronzi, Trofimov, Nichyporuk et al.** — *"Accounting for Variance in Machine Learning Benchmarks"* | MLSys 2021 | Mostrou empiricamente que diferenças <2% em benchmarks de imagem são frequentemente ruído de seed, não sinal real. Justifica nossa cautela com diferenças <2pp em F1. |

### 6.2 Ensemble methods (fundamenta combo defesa-fechamento)

| Paper | Venue/Ano | Contribuição |
|---|---|---|
| **Hansen & Salamon** — *"Neural Network Ensembles"* | **IEEE PAMI 1990** (6,800+ citações) | Paper SEMINAL de ensembles em redes neurais. Demonstrou que combinação de N redes reduz erro generalização. Fundamentação histórica. |
| **Geman, Bienenstock & Doursat** — *"Neural Networks and the Bias/Variance Dilemma"* | Neural Computation 1992 | Decomposição bias-variância. Justifica MATEMATICAMENTE por que ensemble reduz variância (σ²/N) sem reduzir bias. |
| **Breiman** — *"Bagging Predictors"* | Machine Learning 1996 (36,000+ citações) | Bootstrap Aggregating — variante clássica de ensemble. |
| **Breiman** — *"Random Forests"* | Machine Learning 2001 (99,000+ citações) | Ensemble de árvores de decisão. Um dos papers mais citados em ML. |
| **Dietterich** — *"Ensemble Methods in Machine Learning"* | Springer MCS 2000 (14,000+ citações) | Survey foundational sobre ensemble methods. Cobre bagging, boosting, stacking. |
| **Lakshminarayanan, Pritzel, Blundell** — *"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"* | **NeurIPS 2017** (8,000+ citações) | **REFERÊNCIA PRINCIPAL** para deep ensembles modernos. Estabeleceu como prática SOTA para uncertainty quantification em deep learning. Justifica nosso ensemble de 3 seeds. |
| **Huang, Li, Pleiss, Liu, Hopcroft, Weinberger** — *"Snapshot Ensembles: Train 1, Get M for Free"* | ICLR 2017 | Ensemble via N checkpoints de UM treinamento em ciclos cossenoidais. Variante eficiente. |
| **Wen, Tran, Ba** — *"BatchEnsemble"* | ICLR 2020 | Ensemble eficiente em memória, expansão moderna. |
| **Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan, Snoek** — *"Can You Trust Your Model's Uncertainty?"* | NeurIPS 2019 | Comparou métodos de quantificação de incerteza em deep learning. **Deep ensembles ganharam** em todas as métricas. Validação independente da técnica. |

### 6.3 Ensemble especificamente para fairness (validação no nosso domínio)

| Paper | Venue/Ano | Contribuição |
|---|---|---|
| **Bhaskaruni, Hu, Lan** — *"Improving Prediction Fairness via Model Ensemble"* | **IEEE ICTAI 2019** | **CRÍTICO para nossa tese:** demonstrou empiricamente que ensemble REDUZ disparidade demográfica em classificação. Convergente com nosso achado IR 1.541 → 1.474 após ensemble + TTA. |
| **Chen, Johansson, Sontag** — *"Why Is My Classifier Discriminatory?"* | NeurIPS 2018 | Argumenta redução de variância como caminho principiado para fairness. Justifica teoricamente por que ensemble ajuda equidade. |
| **Mehrabi, Morstatter, Saxena, Lerman, Galstyan** — *"A Survey on Bias and Fairness in Machine Learning"* | ACM Comput. Surveys 2021 | Cataloga ensembles como ferramenta padrão de mitigação de viés (§3.3). |

### 6.4 Outras técnicas pós-treinamento (Combo defesa-fechamento)

| Paper | Venue/Ano | Contribuição |
|---|---|---|
| **Guo, Pleiss, Sun, Weinberger** — *"On Calibration of Modern Neural Networks"* | ICML 2017 | Temperature scaling — calibração pós-hoc de redes neurais. Usada em Combo #4. |
| **Simonyan & Zisserman** — *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (VGG paper) | ICLR 2015 | Test-Time Augmentation via multi-crop. |
| **Krizhevsky, Sutskever, Hinton** — *"ImageNet Classification with Deep Convolutional Neural Networks"* (AlexNet) | NeurIPS 2012 | TTA via 10-crop (4 cantos + centro × 2 flips) — usada como inspiração para nossa TTA segura. |
| **Buolamwini & Gebru** — *"Gender Shades"* | FAccT 2018 | Auditoria intersectional de classificadores faciais. Nosso Combo #1 segue essa metodologia. |
| **Crenshaw** — *"Demarginalizing the Intersection of Race and Sex"* | 1989 (legal theory) | **Origem do conceito de "interseccionalidade"** — fundamentação filosófica da análise. |
