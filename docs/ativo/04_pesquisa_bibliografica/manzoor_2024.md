---
name: manzoor-2024
status_verificacao: VERIFIED
autores: [Ayesha Manzoor, Ajita Rattani]
ano: 2024
titulo: "FineFACE: Fair Facial Attribute Classification Leveraging Fine-grained Features"
venue: "International Conference on Pattern Recognition (ICPR), Springer LNCS"
tipo_publicacao: conference
arxiv_id: "2408.16881"
doi: null
url_primario: https://arxiv.org/abs/2408.16881
citacoes_google_scholar: null
citacoes_semantic_scholar: 2
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 35
lente_disrupcao: metodologica
fonte_leitura: PDF integral extraído via pypdf (pdfs/manzoor_2024_fineface.pdf)
---

# FineFACE (Manzoor & Rattani, 2024)

> **Nota sobre confusão anterior:** este paper foi inicialmente
> descrito na documentação como "classificação de raça em 13
> atributos" (sob atribuição errada "Liu et al. 2024"). A leitura
> integral revela que o **alvo de classificação é gênero** (e 13
> atributos faciais sem gênero), com **raça como atributo protegido**
> para avaliação de fairness, não como alvo. O título do paper —
> "Fair *Facial Attribute* Classification" — torna ambiguidade fácil.
> Esta ficha esclarece definitivamente.

## 1. Resumo do problema atacado

Classificadores de atributos faciais (gender, smiling, high cheekbones,
etc.) sofrem disparidade entre grupos demográficos (gender×race). As
técnicas existentes de mitigação de viés (adversarial debiasing,
GAN-oversampling, network pruning, regularização) tipicamente:

- (a) **exigem anotações demográficas** durante o treinamento;
- (b) **sacrificam acurácia** para ganhar fairness — Pareto-inefficient.

O paper propõe **FineFACE**, reformulando o problema como
**fine-grained classification** com **cross-layer mutual attention
learning**. Não exige anotações demográficas no treinamento; alcança
**Pareto-efficient** (melhora fairness E accuracy simultaneamente).

## 2. Método

### 2.1 Cross-layer mutual attention learning

- Backbone: **ResNet50** (5 stages convolucionais).
- **3 experts** construídos progressivamente:
  - e1: stages 1-3 (shallow, captura edges, contours, textura)
  - e2: stages 1-4 (mid-level)
  - e3: stages 1-5 (deep, captura semantic features)
- Cada expert produz:
  - **Categorical prediction** via classifier FC.
  - **Attention region** via CAM (Class Activation Map) → upsample
    → min-max normalize → mask threshold (0.5).
- **Mutual learning:** o crop da attention region de um expert é
  usado como **data augmentation** para os outros experts. Shallow
  experts aprendem com regiões propostas por deep experts (semantic);
  deep experts aprendem com regiões propostas por shallow experts
  (low-level cues).

### 2.2 Multi-step training (por iteração)

- **N + 2 steps por iteração:**
  - Steps 1..N: treinar cada expert sequencialmente (deepest →
    shallowest), com augmentation via attention regions dos outros.
  - Step N+1: treinar concatenação de experts com attention region
    global Aoval.
  - Step N+2: treinar concatenação com raw input.
- **Loss:** cross-entropy padrão. Sem regularização explícita de
  fairness — a fairness emerge do **viés-variância
  decomposition** (reduzir variância intra-grupo via fine-grained
  features).

### 2.3 Inference

- 2 × (N+1) prediction scores por imagem (raw input + overall
  attention region, cada um produz N+1 scores).
- Average dos 2×(N+1) = score final.

### 2.4 Hiperparâmetros declarados (raridade na literatura — usar!)

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | SGD |
| Momentum | 0.9 |
| Weight decay | 5×10⁻⁴ |
| Mini-batch size | 16 |
| Learning rate | 0.002, cosine annealing |
| Image size | 448×448 |
| Mask threshold t | 0.5 |
| Épocas | early stopping |

## 3. Datasets e setup experimental

- **Tarefa 1 — gender classification** (alvo) com race+gender
  protected attributes:
  - Train: **FairFace**.
  - Evaluation: FairFace, UTKFace, LFWA+, CelebA.
- **Tarefa 2 — 13 gender-independent attributes** sobre CelebA:
  - Atributos: "bags under eyes", "bangs", "black hair", "blond
    hair", "brown hair", "chubby", "eyeglasses", "gray hair", "high
    cheekbones", "mouth slightly open", "narrow eyes", "smiling",
    "wearing hat".
  - Protected attribute: **gender**.
  - Métricas reportadas como média sobre os 13.

**Datasets details (Tabela 1 do paper):**
- FairFace: 108K, 7 grupos raciais.
- UTKFace: 20K, 5 grupos raciais (Others mescla Hispanic, Latinx,
  Middle Eastern).
- LFWA+: 13K, 5 grupos (Undefined inclui).
- CelebA: 202K, sem race annotations.

## 4. Métricas reportadas

- **Accuracy overall + por subgrupo demográfico**.
- **Max-Min ratio:** acc_max / acc_min — análoga à nossa DR.
- **Degree of Bias (DoB):** standard deviation da accuracy entre
  subgrupos.
- **TPR (True Positive Rate)** + **DEO (Difference of Equal
  Opportunity)** + **DEOdds** para os 13 atributos.

## 5. Resultados principais (valores numéricos)

### 5.1 Gender classification intra-dataset (FairFace, Tabela 2)

Accuracy (%) por subgrupo race × gender (M/F):

| | White M | White F | Black M | Black F | EA M | EA F | SE M | SE F | Latino M | Latino F | Ind M | Ind F | ME M | ME F | **Max/Min** | **Overall** | **DoB** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline ResNet50** | 96.5 | 89.9 | 94.4 | 82.4 | 97.2 | 88.9 | 94.4 | 91.5 | 95.6 | 92.2 | 98.1 | 93.3 | 97.8 | 92.4 | **1.18** | 93.2 | **4.2** |
| **FineFACE** | 97.1 | 97 | 97.2 | 96.2 | 97 | 96.2 | 96.2 | 97 | 96 | 96.3 | 96.6 | 95.9 | 96.1 | 95.1 | **1.02** | **96.4** | **0.6** |

**Achados:**
- **Pareto-efficient:** overall +3.2 pp, DoB cai 86% (4.2 → 0.6),
  Max/Min ratio cai 13% (1.18 → 1.02).
- Baseline mostra gap de **12 pp** entre Black M (94.4) e Black F
  (82.4). FineFACE reduz para ~1 pp.
- Black female é o subgrupo mais afetado pelo viés do baseline —
  consistente com Gender Shades.

### 5.2 Cross-dataset evaluation (Tabela 3, UTKFace + LFWA+)

| Dataset | Modelo | Max/Min | Overall | DoB |
|---|---|---|---|---|
| UTKFace | Baseline | 1.45 | 81.9 | 11.1 |
| UTKFace | FineFACE | 1.19 | 88.5 | 5.0 |
| LFWA+ | Baseline | 1.26 | 95.3 | 7.7 |
| LFWA+ | FineFACE | 1.05 | 98.6 | 1.9 |

Eficácia preservada cross-dataset: DoB reduzido 55-77%.

### 5.3 CelebA (sem race annotations, só gender)

| Modelo | Male | Female | Max/Min | Overall | DoB |
|---|---|---|---|---|---|
| Baseline | 90.6 | 94.9 | 1.05 | 93.2 | 2.2 |
| FineFACE | 96.4 | 99.0 | 1.03 | 98.0 | 1.3 |

### 5.4 Comparação com SOTA (não tabelada no excerto lido)

Abstract afirma: *"improves accuracy by 1.32% to 1.74% and fairness
by 67% to 83.6%, over the SOTA bias mitigation techniques."* SOTAs
listados: g-SMOTE, fair-mixup, adversarial debiasing (Lagrangian), etc.

## 6. Limitações declaradas pelos autores

- Não exige demographic annotations no **treino**, mas exige no
  **eval** para reportar disparidades.
- ResNet50 backbone — não testado em ConvNeXt, ViT, etc.
- Imagens 448×448 — alto custo computacional.
- 13 gender-independent attributes apenas em CelebA (não FairFace).

## 7. Limitações que identifiquei (leitura crítica)

- **CONFUSÃO TEXTUAL CRÍTICA:** o título "Fair Facial Attribute
  Classification" + o termo "race" em contexto de protected attribute
  é facilmente confundível com classificação de raça. **FineFACE
  classifica gender, não race.** Esta confusão custou tempo no
  projeto anteriormente. Documentação explícita aqui é a mitigação.
- **Single seed.** Como AlDahoul (2024), o paper roda em uma seed e
  não reporta intervalos de confiança. Diferenças de 1-2 pp podem
  estar dentro do ruído.
- **Critério de seleção de checkpoint:** "early stopping mechanism"
  mencionado, mas critério (val loss? val acc? val DoB?) não
  declarado.
- **Max-Min ratio aplicado sobre 14 subgrupos** (7 race × 2 gender) —
  ratio é dominado pelos extremos. Subgrupos com poucos exemplos
  têm alta variância natural; max/min pode ser artefato dessa
  variância, não viés sistemático. **CV ou Gini seriam mais
  robustos.**
- **Image size 448×448 vs 224×224 padrão.** Custo computacional 4×
  maior. Não há ablation de tamanho de imagem para isolar contribuição.
- **3 experts é hiperparâmetro fixo** — não há ablation de N
  experts.
- **Mutual learning é multi-step ranzinza:** 5 passos por iteração
  (3 experts + concat + raw) torna o procedimento difícil de
  reproduzir exatamente. **Github code é necessário para qualquer
  replicação séria.**
- **Comparação contra SOTA em fairness não é matched protocol:**
  diferentes papers usam diferentes splits, augmentations,
  hiperparâmetros. Os 1.32-1.74 pp e 67-83.6% claim devem ser lidos
  com cuidado.
- **FairFace train set não é perfeitamente balanceado** (já notado
  em [[aldahoul_2024]]); FineFACE herda isso sem comentar.

## 8. Relação com nossa pesquisa

**Centralidade:** FineFACE é **paradigmaticamente próximo da nossa
pesquisa** (fairness via decisões algorítmicas, não via balancing de
dados), mas **diverge no alvo da classificação** (gender vs race).
Útil como **referência metodológica** para a métrica Max-Min e DoB,
e como **demonstração de existência** de soluções Pareto-efficient.

**Pontos de ancoragem:**

1. **Max-Min ratio é precedente direto da nossa razão de disparidade
   (DR).** Manzoor & Rattani usam max(acc_subgrupo) / min(acc_subgrupo).
   Nossa DR usa max(F1_classe) / min(F1_classe). A semântica é
   idêntica: "razão entre melhor e pior grupo".
2. **Degree of Bias (DoB) = standard deviation** das acurácias entre
   subgrupos. **Equivalente ao nosso `std` por seed × classe**.
   Conceito compartilhado.
3. **Pareto-efficient é alcançável.** O paper estabelece prova-de-
   conceito de que técnicas algorítmicas **podem** melhorar fairness
   sem sacrificar accuracy. Justifica a busca da nossa pesquisa
   por soluções equivalentes para classificação de raça (não gender).
4. **Hiperparâmetros declarados (mini-batch 16, lr 0.002 com cosine,
   weight decay 5e-4, image 448×448, SGD momentum 0.9, 5 stages
   ResNet50, 3 experts, t=0.5).** Permite **referência comparativa**
   para protocolos da nossa pesquisa.
5. **Cross-layer attention** é uma ideia transponível para o nosso
   problema de **race classification 7-class**. Se nosso trabalho
   futuro investigar fine-grained features para classes raciais
   próximas (East vs Southeast Asian), FineFACE é o template
   metodológico.
6. **Reclassificação como armadilha textual:** documentar este caso
   na dissertação como **lição metodológica** sobre por que síntese
   de literatura via abstract é insuficiente — o título sugere
   uma coisa, o conteúdo entrega outra.

## 9. Pontos para citar / posicionar

- *"Manzoor e Rattani (2024), em paper publicado na ICPR 2024,
  propõem o FineFACE: uma reformulação da classificação de atributos
  faciais como problema de fine-grained classification via cross-layer
  mutual attention learning. O método melhora simultaneamente a
  acurácia média e a paridade entre subgrupos demográficos, atingindo
  uma operação Pareto-efficient antes inalcançada pelas técnicas
  baseadas em adversarial debiasing ou oversampling generativo."*
- *"É importante registrar que o paper FineFACE (Manzoor & Rattani,
  2024) trata da classificação de gênero — e de 13 atributos faciais
  independentes de gênero — não de classificação racial. Raça aparece
  como atributo protegido para avaliação de paridade entre subgrupos
  gender×race, e não como variável alvo. Esta distinção é crítica para
  posicionamento adequado na literatura."*
- *"O conceito de razão Max-Min de acurácia entre subgrupos
  demográficos, central em FineFACE (Manzoor & Rattani, 2024), é
  metodologicamente equivalente à razão de disparidade adotada na
  presente dissertação, calculada como max(F1_classe)/min(F1_classe).
  Ambas medem a 'pior pior-experiência' entre grupos."*

## 10. Arquivos relacionados

- PDF local: `pdfs/manzoor_2024_fineface.pdf` (gitignored).
- Texto extraído: `pdfs/manzoor_2024_fineface.txt` (gitignored).
- Código: https://github.com/VCBSL-Fairness/FineFACE.
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace —
  usado para gender training),
  [[buolamwini_2018]] (Gender Shades — fonte conceitual da Max-Min
  ratio em gender classification),
  [[aldahoul_2024]] (paper para classificação de raça propriamente
  dita).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S3.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Conclusão (ICPR 2024 papers são curtos; future work
geralmente inferido):

- **Aplicar FineFACE a outras tarefas faciais além de gender e 13
  attributes** — race classification 7-class é candidato natural.
  ✅ **Alinhada com Q04** — diretamente.
- **Testar backbones mais modernos** (ViT, ConvNeXt) — paper só usa
  ResNet50. ✅ **Alinhada com Q06**.
- **Ablation de número de experts** (atualmente N=3) — efeito de
  N=5 ou N=10 não testado. ⚠ Detalhe metodológico.
- **Reduzir image size de 448×448** — custo computacional 4× de
  224×224. ⚠ Ortogonal à nossa pesquisa.
- **Aplicar a image domains além de face** (autores citam): outras
  fine-grained classification (birds, cars, plants). ❌ Fora do
  escopo.
- **Sem mencionar uso de demographic annotations no treino**, mas
  exigem no eval — possível direção: fairness sem demographic labels.
  ⚠ Direção paralela emergente (SensitiveNets).
