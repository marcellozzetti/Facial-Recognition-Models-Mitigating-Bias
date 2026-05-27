# Perguntas de pesquisa — interrogação ativa do corpus

> Registro central de perguntas de pesquisa e suas respostas
> agregadas das fichas do diretório `04_pesquisa_bibliografica/`.
> Metodologia detalhada em `README.md` §9.
>
> **Status flags:**
> - ✅ ANSWERED — corpus oferece resposta consolidada, cross-validated.
> - ⚠ PARTIAL — resposta existe mas cobertura limitada.
> - ❌ OPEN — nenhuma ficha aborda; precisa snowballing adicional.
> - 🔬 NEW RESEARCH FRONT — gera direção experimental para a dissertação.
>
> **Corpus consultado (2026-05-25):** 14 fichas — Rodada 1 (9 papers) +
> Rodada 2 (5 papers) + Rodada 2.5 (verificação SOTA). Lista completa
> em `00_referencias.md`.

## Índice de perguntas

| # | Pergunta | Status |
|---|---|---|
| Q01 | Existem melhores datasets para corrigir fairness em biometria facial de raças? | ⚠ PARTIAL → 🔬 |
| Q02 | Qual o dataset mais utilizado nos estudos de fairness em biometria facial de raças? | ✅ ANSWERED |
| Q03 | Qual é o split oficial do FairFace e por que diferentes papers usam splits distintos? | ✅ ANSWERED |
| Q04 | Quais técnicas de mitigação algorítmica já foram testadas em FairFace race 7-class? | ⚠ PARTIAL → 🔬 |
| Q05 | Existe consenso na literatura sobre métrica de fairness para classificação multi-classe (vs binária)? | ❌ OPEN → 🔬 |
| Q06 | O baseline ResNet-34 = 72% acurácia é teto da arquitetura ou da metodologia? | ⚠ PARTIAL → 🔬 |

---

## Q01 — Existem melhores datasets para corrigir fairness em biometria facial de raças?

- **Status:** ⚠ PARTIAL → 🔬 NEW RESEARCH FRONT
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[dataset_wang_2019]], [[buolamwini_2018]], [[aldahoul_2024]], [[manzoor_2024]], [[lin_2022]], [[park_2022]], [[dehdashtian_2024]], [[lafargue_2025]], [[dominguez_2024]], [[grother_2019]]

### Evidências coletadas

**Datasets de biometria facial com anotação racial explícita
identificados nas fichas:**

| Dataset | Tamanho | Categorias raciais | Anotação | Origem | Limitações | Ficha de origem |
|---|---|---|---|---|---|---|
| **FairFace** | 108 501 imgs | **7** (White, Black, Indian, East Asian, Southeast Asian, Middle East, Latino) | MTurk 3 anotadores + refinamento por modelo | Flickr (YFCC-100M) | sem inter-annotator agreement; train não perfeitamente balanceado | [[dataset_karkkainen_2021]] |
| **RFW** | ~40K imgs / 12K IDs | **4** (Caucasian, Asian, Indian, African) | FreeBase nationality + Face++ + manual review | MS-Celeb-1M | racially coarse; race assignment circular (Face++ tem viés) | [[dataset_wang_2019]] |
| **UTKFace** | 20-24K imgs | **5** (White, Black, Asian, Indian, Others — mescla Hispanic, Latinx, Middle Eastern) | (variadas, não centralizada) | MORPH, CACD, web | "Others" agrega múltiplas raças; "Asian" mescla East+SE | [[lin_2022]], [[aldahoul_2024]] |
| **LFWA+** | 13K imgs | **5** (incluindo "Undefined") | derivada de LFW | LFW + newspapers | celebrity-biased | [[lin_2022]] |
| **CelebA** | 202K imgs | **0** (sem race annotations) | só 40 binary attrs | celebridade | irrelevante para race fairness | [[park_2022]], [[manzoor_2024]] |
| **PPB** (Pilot Parliaments) | 1 270 indivíduos | **0** raça; usa Fitzpatrick (6 skin types) | dermatologista (gold) + 3 anotadores | parlamentares oficiais | constrained (poses oficiais); skin type ≠ raça | [[buolamwini_2018]] |
| **Generated Photos / GAN** | 10 000 imgs sintéticas | **4** (Caucasian, Asian, Hispanic/Latino, Black) | balanceado por design | GAN | sintético, não in-the-wild | [[lafargue_2025]] |
| **IJB-A** | 500 sujeitos | (0 raça explícita; geographically diverse) | manual | gov source | small scale | [[buolamwini_2018]] |
| **NIST datasets (FRVT)** | 18.27M imgs / 8.49M pessoas | (country-of-birth como proxy para raça em 3 dos 4 datasets; race direta apenas em domestic mugshots) | governmental records | mugshots, visa, immigration, border crossing | fechado ao público; legalmente restrito | [[grother_2019]] |
| **DiverseFaces** | 1 790 imgs multi-pessoa | 4 (derivada de UTKFace) | herdada do UTKFace | UTKFace composite | sintético via composição; multi-pessoa por imagem | [[aldahoul_2024]] |

**Datasets faciais frequentemente usados em estudos de fairness
porém SEM anotação racial:**

- AffectNet (~400K, emotion); CelebA (200K, attributes); MS-Celeb-1M
  (10M+, identity); VGGFace2 (3.3M, identity); CASIA-WebFace
  (~500K, identity). Estes são usados como **datasets de treinamento
  upstream** ou para outras tarefas, mas não permitem auditoria
  racial direta.

### Resposta

**Não existe um dataset "absolutamente melhor" para fairness em
biometria facial de raças** — cada um faz trade-offs distintos.
Hierarquia por critério:

- **Granularidade racial máxima (7 classes):** **FairFace** é o
  único.
- **Volume × balanceamento:** **FairFace** é melhor (108K balanceado
  por design vs UTKFace 20K).
- **Race verification (1:1) com balanceamento:** **RFW** é o único.
- **Tom de pele em vez de raça:** **PPB** (Fitzpatrick, gold-standard
  dermatologista).
- **Escala industrial:** **NIST FRVT datasets** (~18M imagens), mas
  fechados.
- **Geração sintética balanceada:** **Generated Photos**, com
  trade-off de não ser in-the-wild.

**Para a tarefa específica desta dissertação — classificação racial
em 7 categorias com avaliação in-domain — FairFace é o único dataset
viável** (sem alternativa equivalente em 7-class).

### Lacunas identificadas / 🔬 Nova frente de pesquisa

Nenhum dataset cobre simultaneamente:

1. **≥ 7 categorias raciais** (granularidade FairFace).
2. **Escala MS-Celeb-1M** (milhões de imagens com identidade).
3. **Anotação validada por especialistas** (dermatologistas,
   antropólogos físicos, ou cross-validation por anotadores
   independentes — único caso é o PPB de Buolamwini com
   dermatologista, mas usa Fitzpatrick, não raça).
4. **Inter-annotator agreement reportado por categoria racial** —
   nenhum dataset publica κ de Cohen ou similar **por classe**.
   Especialmente faltante para Latinx/Hispanic, classe
   persistentemente difícil ([[lin_2022]] 59.6% acc; [[aldahoul_2024]] 60% F1;
   [[dataset_karkkainen_2021]] external 0.247 acc).
5. **Anotação por self-identification** (gold standard em
   demografia) — todos os datasets atuais usam atribuição por
   terceiros (MTurk, APIs).

🔬 **Nova frente de pesquisa identificada:**
**"A confiabilidade da anotação racial em FairFace, especialmente
para Latinx/Hispanic, é um possível confundidor estrutural dos
achados de disparidade em classificação 7-class."**

Hipótese: parte da disparidade Latinx (60% F1 vs Black 90% F1) pode
ser **artefato de anotação inconsistente** entre MTurk workers, não
dificuldade intrínseca da classe. Auditoria possível:

- Subsample do FairFace val/test set Latinx.
- Re-anotação por anotadores independentes (3+).
- Mensurar inter-annotator agreement (κ de Fleiss) **por classe**.
- Se κ_Latinx << κ_Black/White, viés de anotação confirmado como
  contribuinte.

**Impacto na dissertação:** esta frente abre questionamento
metodológico **sobre a própria métrica de disparidade**. Se κ
desigual entre classes, a DR=max/min está medindo **ruído de
anotação + viés do modelo confundidos**. Documentar isso é
contribuição metodológica relevante mesmo sem rodar a re-anotação.

**Encaminhamento:** registrar em `06_gap.md` como "Gap-aux 3 —
confiabilidade de anotação racial". Decidir prioridade após
finalização do `07_thesis_statement.md`.

---

## Q02 — Qual o dataset mais utilizado nos estudos de fairness em biometria facial de raças?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** todas as 14 fichas do corpus.

### Evidências coletadas

**Contagem de uso (treino + avaliação) dos datasets de race fairness
nas 14 fichas catalogadas:**

| Dataset | Usado em (treino ou avaliação) | Quantidade | Papéis | Fichas |
|---|---|---|---|---|
| **FairFace** | Karkkainen 2021 (origem), AlDahoul 2024 (treino+eval), Manzoor 2024 (treino+eval gender), Dehdashtian 2024 (treino+eval sex), Lin 2022 (treino+eval race+gender), Lafargue 2025 (modelo auxiliar) | **6 / 14** | dataset central de race classification + auditoria | [[dataset_karkkainen_2021]] [[aldahoul_2024]] [[manzoor_2024]] [[dehdashtian_2024]] [[lin_2022]] [[lafargue_2025]] |
| **UTKFace** | Karkkainen 2021 (cross-dataset), AlDahoul 2024 (cross-dataset eval), Park 2022 (treino+eval gender com race binarizada), Manzoor 2024 (cross-dataset eval), Lin 2022 (cross-dataset eval) | **5 / 14** | cross-dataset secundário | [[dataset_karkkainen_2021]] [[aldahoul_2024]] [[park_2022]] [[manzoor_2024]] [[lin_2022]] |
| **CelebA** | Park 2022 (treino+eval attribute), Manzoor 2024 (treino+eval attribute), Dehdashtian 2024 (treino+eval attribute), Lafargue 2025 (eval), Sagawa 2020 (treino+eval hair color) | **5 / 14** | **MAS sem race annotations** — usado para gender/attribute fairness apenas | [[park_2022]] [[manzoor_2024]] [[dehdashtian_2024]] [[lafargue_2025]] [[sagawa_2020]] |
| **LFWA+** | Karkkainen 2021 (cross-dataset), Manzoor 2024 (cross-dataset), Lin 2022 (eval) | **3 / 14** | benchmark histórico | [[dataset_karkkainen_2021]] [[manzoor_2024]] [[lin_2022]] |
| **PPB** | Buolamwini 2018 (origem) | **1 / 14** | usa Fitzpatrick, não raça | [[buolamwini_2018]] |
| **RFW** | Wang 2019 (origem) | **1 / 14** | verification, não classification | [[dataset_wang_2019]] |
| **AffectNet** | AlDahoul 2024, Dominguez-Catena 2024 | 2 / 14 | emoção, não race | [[aldahoul_2024]] [[dominguez_2024]] |
| **NIST FRVT datasets** | Grother 2019 (origem) | 1 / 14 | proprietary | [[grother_2019]] |
| **Waterbirds, MultiNLI** | Sagawa 2020 | 1 / 14 | não-facial | [[sagawa_2020]] |
| **20+ FER datasets** | Dominguez-Catena 2024 | 1 / 14 | expressão facial | [[dominguez_2024]] |
| **Generated Photos** | Lafargue 2025 | 1 / 14 | sintético | [[lafargue_2025]] |

### Resposta

**FairFace é, comprovadamente, o dataset mais utilizado nos estudos
de fairness em biometria facial **com foco em raças**, com 6 ocorrências
em um corpus de 14 papers (≈43%).**

Detalhamento:

1. **FairFace (6/14):** posição dominante. Origem em [[dataset_karkkainen_2021]] (WACV
   2021) + 5 papers que treinam ou avaliam nele. Único com 7 classes
   raciais.

2. **UTKFace (5/14):** segundo mais usado, frequentemente como **cross-
   dataset validation** sobre modelos treinados em FairFace. Granularidade
   menor (4-5 raças).

3. **CelebA (5/14):** popular para fairness em geral, **mas sem
   anotação racial.** É usado em estudos de fairness em **gender**
   (Park, Manzoor, Sagawa) ou **atributos faciais** (Dehdashtian). Para
   raça especificamente, **CelebA é irrelevante.**

4. **LFWA+ (3/14):** benchmark histórico mais antigo, racialmente coarse.

5. **RFW (1/14):** ICCV 2019, mas para **verification 1:1**, não
   classification.

6. **PPB (1/14):** Gender Shades, mas usa **Fitzpatrick**, não raça.

### Implicações para a dissertação

1. **Validação da escolha do FairFace** na presente dissertação — está
   alinhada com o **uso dominante na literatura específica de race
   fairness facial**.
2. **Cross-dataset evaluation em UTKFace** seria adição valiosa para
   generalização do nosso trabalho (precedente em 5 papers do corpus).
3. **Não citar CelebA como base para race fairness:** CelebA é para
   gender/attribute fairness. Confundir os dois é erro comum na
   literatura — evitar.
4. **Posicionamento metodológico:** quando dizemos "treinamos sobre
   FairFace 7-class", estamos no **mesmo pavimento metodológico** que
   AlDahoul, Manzoor, Dehdashtian, Lin, Lafargue, Karkkainen — todos
   os trabalhos relevantes.

### Lacunas

Nenhuma — pergunta fechada. Possíveis sub-perguntas decorrentes:

- **Q-related:** *"Qual é o splits oficial do FairFace e por que
  Lin (2022) usa 80/10/10 diferente do padrão?"* → registrar como
  Q03 se relevante para metodologia experimental.

---

## Q03 — Qual é o split oficial do FairFace e por que diferentes papers usam splits distintos?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[aldahoul_2024]], [[lin_2022]], [[manzoor_2024]]
- **Fonte primária adicional:** https://github.com/joojs/fairface (README oficial — consultado 2026-05-25)

### Evidências coletadas

**Do FairFace GitHub oficial (joojs/fairface):**

1. **Apenas dois splits são publicados oficialmente: train + validation.** Não há test set público separado.
2. **Duas versões da imagem (padding):**
   - `padding=0.25`: usado nos experimentos principais do paper.
   - `padding=1.25`: usado nos experimentos de auditoria de APIs comerciais.
3. Labels em arquivos CSV (training + validation).

**Do paper original** ([[dataset_karkkainen_2021]]):
- Total: **108 501** imagens.
- Splits específicos **NÃO declarados textualmente no paper** — apenas
  citados como "released via GitHub".

**Da literatura derivada:**

| Paper | Split usado | Train | Val | Test | Observação |
|---|---|---|---|---|---|
| AlDahoul 2024 ([[aldahoul_2024]]) | **Oficial** | 86 744 | — | 10 954 (chamado "test" no paper, é o val oficial) | Total = 97 698 (≠ 108 501 do paper original) |
| FineFACE / Manzoor 2024 ([[manzoor_2024]]) | Implícito (GitHub) | — | — | — | usa "FairFace" como label; números não declarados |
| FairGRAPE / Lin 2022 ([[lin_2022]]) | **Próprio 80/10/10** sobre 97 698 imagens | ~78 158 | ~9 770 | ~9 770 | **Não usa split oficial**; randomiza sobre o total que conseguiu baixar |

### Resposta

**O split oficial do FairFace é apenas train + validation:**

- **Train: 86 744 imagens**
- **Validation: 10 954 imagens**
- **Total publicado**: ~97 698 (conforme contagem usada por
  AlDahoul e Lin).
- **Test set público separado: NÃO EXISTE.** Papers que reportam
  "test accuracy" usam o validation set oficial **como test set**
  (prática comum em deep learning quando não há test split).

**O discrepância 97 698 vs 108 501** (número no abstract do paper
original) é provavelmente devido a:
- Múltiplas faces por imagem source contadas separadamente.
- Imagens descartadas no processo de release (qualidade, licenciamento).
- O número 108 501 inclui faces descartadas pós-coleta.
- Versões diferentes ao longo do tempo no GitHub.

**Por que papers usam splits diferentes:**

1. **AlDahoul (e maioria moderna): usa o split oficial.** Train→Val,
   tratando val como test set. Esta é a prática **canônica**.
2. **Lin (FairGRAPE): randomiza 80/10/10 próprio.** Justificativa
   implícita (provavelmente): paper precisava de test set separado
   do val (para tunar pruning hyperparams sem contaminar avaliação
   final), então split foi refeito.
3. **Manzoor (FineFACE) e outros: não declaram explicitamente.** Provavelmente
   usam split oficial mas falta documentação textual.

### Implicações para nossa dissertação

1. **Adotar split oficial: train 86 744 + val 10 954.** É a prática
   dominante e reprodutível.
2. **Reportar val accuracy como "test accuracy"** seguindo convenção
   de AlDahoul. Justificativa: não há test separado, e val é o que
   está públicamente disponível.
3. **Documentar a ambiguidade na metodologia:** o número 108 501 vs
   97 698 deve ser explicitado para evitar confusão com leitores.
4. **Comparações cross-paper com Lin (FairGRAPE)** devem ressalvar o
   split divergente — números do Lin podem **não ser estritamente
   comparáveis** ao AlDahoul / nossa pesquisa.

### Lacunas

- **Por que o paper original do FairFace não declara textualmente o
  split** é inexplicado.
- **Há test set "secreto" usado pelos autores** para reportar números
  out-of-domain (Twitter, Media, Protest)? Sim — esses são datasets
  externos (re-anotados pelos próprios autores) e não fazem parte do
  FairFace propriamente dito. Documentado em [[dataset_karkkainen_2021]] §3.3.

---

## Q04 — Quais técnicas de mitigação algorítmica já foram testadas em FairFace race 7-class?

- **Status:** ⚠ PARTIAL → 🔬 NEW RESEARCH FRONT
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** todas as 14 do corpus.

### Evidências coletadas

**Estrita interpretação ("testado em FairFace, target = race, 7 classes
in-domain"):**

| Ficha | Técnica | Testado em FairFace race 7-class? |
|---|---|---|
| [[aldahoul_2024]] | Fine-tuning de VLM (FaceScanPaliGemma sobre PaliGemma 3B) | ✅ SIM — 75.7% acc / 75% F1 |
| [[lin_2022]] (FairGRAPE) | Pruning fairness-aware via group-importance preservation | ✅ SIM — mas em contexto de compressão (99% sparsity); accuracy 66.8% (degradação) |
| [[dataset_karkkainen_2021]] (FairFace original) | Treino baseline ResNet-34 com cross-entropy padrão (sem mitigação algorítmica) | ⚠ (é o baseline, não é mitigação) |
| [[park_2022]] (FSCL) | Fair Supervised Contrastive Loss | ❌ NÃO — testa em CelebA e UTKFace (ethnicity binária) |
| [[manzoor_2024]] (FineFACE) | Cross-layer mutual attention fine-grained | ❌ NÃO — testa em FairFace para **gender**, não race |
| [[dehdashtian_2024]] (U-FaTE) | Utility-Fairness Trade-Off estimator | ❌ NÃO — testa em FairFace para **sex** (binário), não race |
| [[sagawa_2020]] (Group DRO) | Distributionally Robust Optimization + strong reg | ❌ NÃO — testa em Waterbirds, CelebA hair color, MultiNLI |
| [[bhaskaruni_2019]] | Ensemble AdaBoost-fair | ❌ NÃO — testa em tabular (Credit, Crime) |
| [[buolamwini_2018]] (Gender Shades) | Auditoria (não mitigação) | ❌ NÃO — auditoria de gender em PPB |
| [[dataset_wang_2019]] (RFW + IMAN) | IMAN domain adaptation | ❌ NÃO — testa em RFW (verification, não classification) |
| [[grother_2019]] (NIST) | Auditoria industrial (não mitigação) | ❌ NÃO — recognition (1:1, 1:N), não classification |
| [[dominguez_2024]] (DSAP) | Auditoria de datasets (não mitigação) | ❌ NÃO — auditoria, sem treinamento |
| [[lafargue_2025]] | Auditoria com testes estatísticos uncertainty-aware (não mitigação) | ❌ NÃO — auditoria |
| [[survey_mehrabi_2021]] | Survey (não método) | ❌ N/A |

### Resposta

**Apenas DUAS técnicas de mitigação algorítmica foram explicitamente
testadas em FairFace race 7-class in-domain no corpus catalogado:**

1. **VLM fine-tuning** (AlDahoul 2024/2026, FaceScanPaliGemma): 75.7%
   accuracy / 75% F1 — o SOTA.
2. **Fairness-aware pruning** (Lin 2022, FairGRAPE): mais relevante
   como mitigação de degradação por compressão; accuracy cai para 66.8%
   em 99% sparsity (vs 72% baseline), mas reduz desvio entre classes.

**Técnicas NÃO testadas em FairFace race 7-class — mas amplamente
testadas em tarefas adjacentes:**

| Técnica | Onde foi testada | Por que NÃO em FairFace race 7-class |
|---|---|---|
| Fair contrastive learning (FSCL+) | CelebA gender, UTKFace ethnicity-binary | autores não exploraram race 7-class |
| Fine-grained mutual attention (FineFACE) | FairFace gender, CelebA attributes | autores focaram em binary gender |
| Group DRO + strong regularization | CelebA hair-color, Waterbirds | tarefa binary, não 7-class |
| Adversarial debiasing (GRL, LNL) | CelebA, UTKFace | binary tasks |
| Fair distillation (MFD) | CelebA | binary tasks |
| Ensemble AdaBoost-fair | Credit, Crime tabular | não testou em facial |
| Disparate Impact removal | tabular | não facial |
| Temperature scaling / calibration | classification geral | aplicável mas não testado especificamente para race fairness |
| Counterfactual fairness | NLP, decision systems | não facial |

### 🔬 Nova frente de pesquisa identificada

**ESTE É O GAP CENTRAL DA DISSERTAÇÃO.** A literatura de mitigação
algorítmica de fairness facial **migrou seu testbed para binary tasks**
(gender, attributes binários) **sem nunca validar as técnicas em race
classification multi-classe** — apesar de FairFace estar disponível
desde 2019 e ser amplamente reconhecido como o benchmark dominante
(Q02).

**Hipóteses experimentais geradas:**

1. **H1 — Transferência direta:** FSCL+ (Park 2022), originalmente
   testado em CelebA gender, **pode ser aplicado a FairFace race 7-class
   com adaptação multi-classe**, com hipótese de redução de DR
   max/min em pelo menos 30% sobre baseline ResNet-34.
2. **H2 — Group DRO + strong reg:** Sagawa 2020 mostra ganho de 45 pp
   em worst-group para binary task. **Aplicar a FairFace race 7-class**
   pode produzir ganho equivalente em worst-class accuracy
   (especialmente Latinx).
3. **H3 — Combinação:** **deep ensemble (3 seeds) + temperature
   scaling + group reweighting** pode superar SOTA (75.7%)
   marginalmente em accuracy mas substancialmente em DR.
4. **H4 — ConvNeXt-T como backbone:** AlDahoul usa PaliGemma 3B (3 bi
   params). **ConvNeXt-T (28M params) com mitigação algorítmica**
   pode ser **Pareto-eficiente** vs PaliGemma fine-tuned — menor
   custo computacional, accuracy competitiva.

**Encaminhamento:** consolidar em `06_gap.md` como "Gap-principal —
mitigação algorítmica não testada em FairFace race 7-class". Cada
hipótese H1-H4 vira potencial fator experimental de Fase 5.

---

## Q05 — Existe consenso na literatura sobre métrica de fairness para classificação multi-classe (vs binária)?

- **Status:** ❌ OPEN → 🔬 NEW RESEARCH FRONT
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[survey_mehrabi_2021]], [[buolamwini_2018]], [[dehdashtian_2024]], [[manzoor_2024]], [[lin_2022]], [[aldahoul_2024]], [[park_2022]], [[sagawa_2020]], [[bhaskaruni_2019]], [[dominguez_2024]], [[lafargue_2025]]

### Evidências coletadas

**Métricas formuladas para o caso BINÁRIO (Y ∈ {0,1}, S ∈ {0,1}):**

| Métrica | Definição | Origem | Usada em |
|---|---|---|---|
| Demographic Parity (DP) | P(Ŷ=1\|S=0) = P(Ŷ=1\|S=1) | Dwork et al. 2012 | [[survey_mehrabi_2021]] |
| Equal Opportunity (EO_h) | P(Ŷ=1\|Y=1, S=0) = P(Ŷ=1\|Y=1, S=1) — equal TPR | Hardt et al. 2016 | [[survey_mehrabi_2021]] |
| Equalized Odds (EOD) | EO_h + equal FPR | Hardt et al. 2016 | [[sagawa_2020]] [[dehdashtian_2024]] [[park_2022]] |
| Treatment Equality | equal FN/FP ratio | Berk et al. | [[survey_mehrabi_2021]] |
| Test Fairness | equal P(Y=1\|S=s, R=r) por score | Chouldechova | [[survey_mehrabi_2021]] |
| Disparate Impact (DI) | min(P_a, P_b) / max(P_a, P_b) ≥ 0.8 | EEOC 80%-rule | [[lafargue_2025]] [[survey_mehrabi_2021]] |
| Statistical Disparity (SD) | \|P(Ŷ=1\|S=0) − P(Ŷ=1\|S=1)\| | classical | [[bhaskaruni_2019]] |
| Counterfactual Fairness | individual-level, via causal graph | Kusner et al. | [[survey_mehrabi_2021]] |

**Métricas ad-hoc para multi-classe — sem consenso formal:**

| Métrica multi-classe | Definição | Origem corpus |
|---|---|---|
| **Max-Min Ratio (DR)** | max_g(acc_g) / min_g(acc_g) | [[manzoor_2024]] (Tabela 2), nossa pesquisa |
| **Degree of Bias (DoB)** | std accuracy entre grupos | [[manzoor_2024]] |
| **ρ(A)** = std accuracy | mesmo que DoB | [[lin_2022]] |
| **ρ(∆)** = std degradação accuracy | std da queda accuracy entre grupos | [[lin_2022]] |
| **Max accuracy disparity ε** | max log(P_j / P_k) — log-ratio | [[dataset_karkkainen_2021]] |
| **Worst-group accuracy** | min_g(acc_g) | [[sagawa_2020]] |
| **Equalized Odds Difference (EOD)** generalizado | soma sobre pares de grupos × classes | [[park_2022]] (eq.6), [[sagawa_2020]] |
| **Cramer's V** | indicador stereotypical via χ² | [[dominguez_2024]] |
| **DSR / DSE / DSS** | similaridade demográfica em [0,1] | [[dominguez_2024]] |
| Acurácia per-classe (raw) | acc_g por classe, sem agregação | [[aldahoul_2024]] [[buolamwini_2018]] |

### Resposta

**NÃO há consenso na literatura sobre métrica de fairness para
classificação multi-classe.** Especificamente:

1. **Definições canônicas (DP, EO, EOD)** são **formuladas em binário**
   (Y, S ∈ {0,1}). Extensão para multi-classe é **ad-hoc**: cada
   paper escolhe sua adaptação.

2. **Cada paper do corpus que opera em multi-classe usa métrica
   diferente:**
   - AlDahoul: reporta F1 macro overall + F1 per class (sem métrica
     agregada de disparidade).
   - Lin (FairGRAPE): ρ(A) std accuracy entre grupos.
   - Manzoor (FineFACE): Max-Min ratio + DoB.
   - Park (FSCL): EOD generalizada (somatório sobre pares).
   - Sagawa: worst-group accuracy.
   - Dominguez (DSAP): DSR/DSE/DSS via Renkonen similarity.

3. **Mehrabi survey** lista 10 definições mas **não prescreve para
   multi-classe**.

4. **Impossibilidade Kleinberg-Mullainathan-Raghavan (2017):**
   demonstra que **calibration e balance for positive/negative class
   não podem ser satisfeitas simultaneamente** — corolário: não
   existe métrica "ótima" universal.

5. **Comparações cross-paper são problemáticas:** quando AlDahoul
   reporta "F1 macro = 75%" e Lin reporta "ρ(A) = 13.4", os dois
   valores **não são comparáveis** sem mediação por uma terceira
   métrica.

### 🔬 Nova frente de pesquisa identificada

A **ausência de consenso** sugere oportunidade metodológica:

**Hipótese:** reportar **três métricas complementares** em vez de
uma única para classificação racial multi-classe oferece **visão
mais robusta** que qualquer escolha singular:

1. **Disparity Ratio (DR) = max/min de F1 por classe** — captura
   extremo (paridade fraca).
2. **Worst-class F1 absoluta** — captura piso operacional.
3. **Coeficiente de variação (CV) = std/mean de F1 por classe** —
   captura dispersão normalizada.

**Triangulação:** se DR baixo, CV baixo, e worst-class F1 alto
**simultaneamente**, há fairness robusta. Se discrepância entre as
três (e.g., DR baixo mas worst-class F1 alto), há **distribuição
não-extrema** mas piso aceitável — interpretação contextual.

**Encaminhamento:** registrar como direção metodológica em
`06_gap.md`. Esta tripla pode tornar-se **contribuição metodológica
da dissertação**: "**ausência de métrica única → triangulação
explícita**". Justifica o uso de DR + worst-class + CV simultaneamente
nos nossos experimentos.

---

## Q06 — O baseline ResNet-34 = 72% acurácia é teto da arquitetura ou da metodologia?

- **Status:** ⚠ PARTIAL → 🔬 NEW RESEARCH FRONT
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[aldahoul_2024]], [[lin_2022]], [[dehdashtian_2024]]

### Evidências coletadas

**Números reportados para FairFace race 7-class in-domain:**

| Modelo | Parâmetros | Accuracy | F1 | Fonte |
|---|---|---|---|---|
| **ResNet-34 baseline** (FairFace original) | ~21M | **72.0%** | 72% | [[aldahoul_2024]] Tabela 10 + [[lin_2022]] Tabela 2 (confirmação cruzada) |
| VGGFace-ResNet-50 + SVM | ~25M + SVM | 72.6% | 72% | [[aldahoul_2024]] Tabela 10 |
| FaceNet + SVM | ~7M + SVM | 68.9% | 68% | [[aldahoul_2024]] Tabela 10 |
| CLIP zero-shot | ~150M-400M | 64.2% | 65% | [[aldahoul_2024]] Tabela 10 |
| GPT-4o zero-shot | ~trillion (proprietary) | 68% | 65% | [[aldahoul_2024]] Tabela 10 |
| **FaceScanPaliGemma (fine-tuned)** | ~3B | **75.7%** | 75% | [[aldahoul_2024]] Tabela 10 — SOTA |

**Insights cruciais:**

1. **Arquiteturas grandes (GPT-4o, CLIP) sem fine-tuning falham**:
   abaixo da ResNet-34 baseline! Sugere que o **gap entre arquitetura
   moderna e antiga é menor do que parece** em zero-shot.

2. **Fine-tuning de PaliGemma 3B atinge 75.7%** — apenas **+3.7 pp**
   sobre ResNet-34 (21M) e **+3.1 pp** sobre VGGFace-ResNet-50 + SVM.
   Para um aumento de **~150× em parâmetros**, o ganho é modesto.

3. **Per-class F1 do SOTA**:
   - Black: 90% | White: 80% | Indian: 79% | East Asian: 78%
   - Middle Eastern: 73% | Southeast Asian: 67% | **Latinx/Hispanic: 60%**
   - **Gap White-Latinx = 20 pp** mesmo no SOTA.

4. **Per-class no baseline** (Lin 2022 ResNet-34): Latinx 59.6%, idêntico
   ao SOTA Latinx 60%. **O bottleneck Latinx persiste através de 150×
   aumento de parâmetros.**

5. **Backbones não testados em FairFace race 7-class:**
   - **ConvNeXt-T** (28M): outperformer ResNet-50 em ImageNet,
     LayerNorm.
   - **ViT-B/16** (86M): atinge 81-85% ImageNet top-1.
   - **DINOv2** (~300M-1B): SOTA self-supervised, alta performance em
     downstream.
   - **EfficientNet-B7** (66M): mencionado em [[aldahoul_2024]] Seção
     2.2 como "tuned" mas resultado específico não reportado.

### Resposta

**O baseline 72% acc é um teto MISTO — provavelmente dominado por
limites METODOLÓGICOS e DE DADOS, não arquiteturais.**

Argumentação:

**A favor de teto arquitetural (ResNet-34 antiga):**
- ResNet-34 é de 2016; ConvNeXt/ViT são 5+ anos mais novos.
- Em ImageNet, ResNet-50 (76% top-1) << ConvNeXt-T (82.1%).
- Sugere espaço para +5-8 pp via backbone.

**A favor de teto metodológico/de dados (mais forte):**
1. **PaliGemma 3B (150× ResNet-34)** atinge só +3.7 pp — diminishing
   returns extremos.
2. **Latinx F1 ≈ 60% no baseline E no SOTA** — não move com
   arquitetura. Indica **limite de dados/anotação**, não capacidade
   do modelo.
3. **Train não perfeitamente balanceado** (White 19% vs Middle East
   11%) — class imbalance contribuiu.
4. **Single-seed reporting** mascara variância — talvez o "72%" tem
   IC de ±2 pp, dentro do qual ConvNeXt-T poderia estar.
5. **Anotação MTurk** sem κ por classe (ver Q01) — provavelmente
   adiciona ruído de ~5-10 pp ao limite teórico.

### Resposta sintética

**Provável decomposição do gap:**

| Componente | Contribuição estimada para "ceiling" |
|---|---|
| Limite arquitetural (ResNet-34 vs ConvNeXt-T) | **+2 a +5 pp** possível |
| Limite metodológico (single-seed, sem HPO, sem ensemble) | **+1 a +3 pp** possível |
| Limite de dados (class imbalance, MTurk noise) | **−3 a −5 pp** do teto teórico inviável corrigir sem re-anotação |
| Limite Latinx/Hispanic (anotação ruidosa?) | **−2 a −4 pp** específico |

**Hipótese sintética:** com ConvNeXt-T + multi-seed casado + HPO
modesto, accuracy esperada **~76-78% F1 macro** (próximo ou acima do
PaliGemma 3B). Latinx F1 **permanece ≈ 60%** sem mitigação
algorítmica específica.

### 🔬 Nova frente de pesquisa identificada

**Decomposição experimental do "ceiling" via fatores:**

| Fator | Investigação |
|---|---|
| Backbone | Comparar ResNet-34 vs ConvNeXt-T vs ViT-B mantendo todo o resto |
| Seed casado vs single | 3 seeds + média ± std vs single-run |
| HPO | Padrão (lr=0.0001 ADAM) vs varredura limitada |
| Class reweighting | Sem vs proporcional à frequência inversa |
| Augmentation | Padrão vs forte (MixUp, RandAugment) |

**Hipótese H5 (gap):** **com mesmo pipeline experimental rigoroso
mas variando apenas backbone (ResNet-34 → ConvNeXt-T), ganho de
accuracy esperado é +2-3 pp; Latinx F1 não muda. O ceiling principal
é metodológico-de-dados, não arquitetural.**

**Encaminhamento:** essa investigação é **parcialmente cobertura
direta da nossa pesquisa experimental** (já temos resultados de
backbone comparison). Documentar em `06_gap.md` como "Gap-aux 2 —
decomposição do ceiling".

---

## Convenções de manutenção

- **Adicionar nova pergunta:** copiar template do §9.1 do
  `README.md`. Numerar sequencialmente Q01, Q02, Q03...
- **Atualizar resposta:** revisar "Última atualização" sempre que
  nova evidência alterar status ou síntese.
- **Cross-reference em fichas:** opcional. Se uma ficha responde Q<NN>
  centralmente, adicionar nota na Seção 10 da ficha:
  `Responde Q<NN> em _perguntas.md`.
- **Frentes 🔬:** consolidar em `06_gap.md` durante Fase 4. Não duplicar
  conteúdo — referenciar `_perguntas.md`.
