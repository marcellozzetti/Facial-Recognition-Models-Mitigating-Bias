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
| Q07 | Existe pesquisa de fairness em biometria facial **sem usar FairFace**? | ✅ ANSWERED |
| Q08 | Por que os estudos tentam sempre fazer merge de classes raciais? | ✅ ANSWERED |
| Q09 | 7 classes é realmente a taxonomia correta para fairness racial facial? | ❌ OPEN → 🔬 |
| Q10 | Existe matriz associativa Fitzpatrick/MST × FairFace 7-race? | ❌ OPEN → 🔬 (NEW RESEARCH FRONT — CANDIDATA PRINCIPAL) |
| Q11 | As 7 categorias raciais do FairFace têm fundamento biológico ou são socio-políticas? | ✅ ANSWERED (FUNDAMENTAL) |
| Q12 | Como antropologia forense moderna trata "raça"? Qual o consenso? | ✅ ANSWERED |
| Q13 | Origem e propósito da Fitzpatrick scale — para que foi criada cientificamente? | ✅ ANSWERED |
| Q14 | Quantos tons de pele existem cientificamente (medicina/biometria)? | ✅ ANSWERED |

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

## Q07 — Existe pesquisa de fairness em biometria facial sem usar FairFace?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25 (após Rodada 3 — broadening intencional)
- **Fichas consultadas:** todas as 19 do corpus, com ênfase em [[dataset_wang_2019]], [[dataset_robinson_2020]], [[buolamwini_2018]], [[dataset_hazirbas_2021]], [[schumann_2023]], [[grother_2019]], [[neto_2025]], [[survey_kotwal_2025]]

### Evidências coletadas

**SIM, há um ecossistema substancial de pesquisa de fairness facial
sem FairFace.** Mapeamento por dimensão demográfica:

**Track 1 — Face Recognition (verification 1:1 / identification 1:N) com balanceamento racial:**

| Dataset | Imagens / IDs | Categorias | Origem | Ficha |
|---|---|---|---|---|
| **RFW** | ~40K / 12K | 4 (Caucasian, Asian, Indian, African) | ICCV 2019 | [[dataset_wang_2019]] |
| **BFW** | 20K / 800 | 4×2 (race×gender) | CVPRW 2020 | [[dataset_robinson_2020]] |
| **BUPT-Balancedface** | ~1.3M / 28K | 4 ethnicities | Wang et al. (citado em Neto, [[neto_2025]]) | (mencionado, sem ficha dedicada) |
| **DemogPairs** | curado | 3 (Asian, Black, White) | Hupont & Tena (citado em Robinson) | (mencionado) |
| **NIST FRVT datasets** | 18.27M / 8.49M | country-of-birth proxy | governmental | [[grother_2019]] |

**Track 2 — Skin tone (Fitzpatrick / MST) como alternativa a race:**

| Dataset | Sujeitos / Imagens | Escala | Ficha |
|---|---|---|---|
| **PPB** | 1 270 | Fitzpatrick (anotador dermatologista) | [[buolamwini_2018]] |
| **Casual Conversations v1** | 3 011 / 45K vídeos | Fitzpatrick + self-reported age/gender | [[dataset_hazirbas_2021]] |
| **Casual Conversations v2** | 5 567 (7 países) | (extensão) | (mencionado em [[schumann_2023]]) |
| **MST-E** | 1 515 + 31 vídeos | Monk Skin Tone 10-pt | [[schumann_2023]] |
| **Generated Photos** | 10 000 sintéticas | Fitzpatrick | [[lafargue_2025]] |

**Track 3 — Fairness em outras tarefas faciais (gender, attribute, expression):**

| Dataset | Tarefa | Ficha |
|---|---|---|
| CelebA | gender / 40 attributes | [[park_2022]], [[manzoor_2024]], [[sagawa_2020]] |
| UTKFace | gender / ethnicity 5-class | múltiplas |
| AffectNet | emotion (8 classes) | [[aldahoul_2024]], [[dominguez_2024]] |
| 20 FER datasets | facial expression | [[dominguez_2024]] |

**Track 4 — Fairness fora de visão facial (paralelos teóricos):**

| Dataset | Domínio | Ficha |
|---|---|---|
| Waterbirds | bird classification + spurious bg | [[sagawa_2020]] |
| MultiNLI | language inference | [[sagawa_2020]] |
| Community Crime, Credit Default | tabular | [[bhaskaruni_2019]] |

### Resposta

**SIM, existe pesquisa de fairness em biometria facial sem FairFace e
em escala substancial.** Distribuição aproximada por tarefa:

- **Para race classification (nosso problema específico):** FairFace
  domina (Q02), com **UTKFace** como cross-dataset.
- **Para race em face recognition (verification/identification):**
  ecossistema próprio com **RFW, BFW, BUPT-Balancedface, NIST FRVT,
  DemogPairs**. **FairFace tem papel auxiliar nessa subliteratura,
  não central.**
- **Para skin tone fairness:** **track paralelo** com PPB, Casual
  Conversations, MST-E — **não usa race labels**, opta por
  Fitzpatrick ou MST.
- **Para gender/attribute fairness facial:** CelebA é dominante.

### Implicação para nossa pesquisa

1. **Posicionamento da dissertação:** estamos no **track de race
   classification** especificamente. FairFace é justificadamente o
   dataset principal nesse track.
2. **Triangulação possível:** podemos citar evidência de RFW, BFW,
   NIST FRVT para mostrar **convergência de findings** ("balanceamento
   não basta" aparece em 4+ datasets independentes).
3. **Track skin-tone (Casual Conversations, MST-E) é parente próximo:**
   alimenta Q10 🔬 (matriz skin tone × race). Pode-se argumentar que
   nossa dissertação está **conectando dois tracks paralelos**.

### Lacunas residuais

- **BUPT-Balancedface** e **DemogPairs** não têm fichas dedicadas
  ainda. Decidir em `06_gap.md` se relevantes ou citáveis sem ficha
  individual.
- **Casual Conversations v2 (Porgali et al. 2023)** não tem ficha
  separada — pode justificar Rodada 4 mais tarde.

---

## Q08 — Por que os estudos tentam sempre fazer merge de classes raciais?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[aldahoul_2024]], [[lin_2022]], [[dataset_wang_2019]], [[dataset_robinson_2020]], [[neto_2025]], [[buolamwini_2018]]

### Evidências coletadas

**Padrão observado:** estudos com taxonomia 7-classe (FairFace) e
4-classe (RFW, BFW) raramente convergem; mais comum é estudo cobrir
**4 ou 5 classes** com **fusões pragmáticas**.

**Exemplos no corpus:**

| Paper | Taxonomia usada | Classes mescladas |
|---|---|---|
| FairFace original | 7 train, **4 em cross-dataset Table 4** | "Only 4 races (W, B, Asian, Indian)" para comparar com UTKFace [[dataset_karkkainen_2021]] |
| AlDahoul 2024/2026 | testa **ambos** 6 e 7 | 6-class merge East+SE Asian → "Asian"; resultado: 81.1% (6) vs 75.7% (7) acc [[aldahoul_2024]] |
| FairGRAPE (Lin 2022) | 7 mantido | uso pleno do FairFace [[lin_2022]] |
| RFW | **4** (W, A, I, Afr) | sem distinção East/SE Asian [[dataset_wang_2019]] |
| BFW | **4×2** (race×gender) | sem Middle East ou Latinx [[dataset_robinson_2020]] |
| UTKFace | **5** (W, B, A, I, "Others") | "Others" mescla Latinx + Middle East + Hispanic |
| Neto 2025 | **4** (W, A, I, Afr) | aceita taxonomia do RFW/BFW [[neto_2025]] |
| Gender Shades | **0 race** (Fitzpatrick 6) | descarta race por instabilidade [[buolamwini_2018]] |

### Razões identificadas (consolidadas)

**1. Cross-dataset comparability** (mais frequente)

FairFace tem 7; UTKFace tem 5; RFW tem 4; LFWA+ tem 5 (com
"Undefined"). Para comparação cross-dataset, **só o menor denominador
comum** funciona — tipicamente 4 (W, B, Asian, Indian). O paper do
FairFace explicita isso em Tabela 4: *"only 4 races were used to make
it comparable to UTKFace"*.

**2. Annotation reliability** (East Asian vs Southeast Asian)

[[schumann_2023]] documenta que anotadores de diferentes regiões
divergem sistematicamente. Distinguir East Asian de Southeast Asian
exige conhecimento etnográfico que MTurk workers (especialmente nos
EUA) tipicamente não têm. **Fusão para "Asian" resolve o
disagreement.**

**3. Class imbalance** mesmo em datasets "balanceados"

FairFace train: White=19%, Latinx=15%, Middle East=11% (lido em
[[aldahoul_2024]]). Classes menores têm sinal mais ruidoso; fusão
aumenta o número de exemplos por classe efetiva.

**4. Ambiguidade conceitual estrutural**

[[neto_2025]] argumenta que a fronteira East Asian / Southeast Asian
é fundamentalmente **borrada**: traços fenotípicos têm continuidade
no sudeste asiático. Não é falha de anotação — é a categoria que
é fuzzy. Análogo: "Hispanic" como ethnia × race.

**5. Conveniência metodológica**

Métricas multi-classe são ad-hoc (Q05). Reduzir para 4 simplifica
formulação (e.g., Equalized Odds binarizada por One-vs-Rest).

**6. AlDahoul demonstra trade-off explícito**

Em 7-class: 75.7% acc, F1 = 75%.
Em 6-class (East+SE merged): 81.1% acc.
**Mergeando ganha-se 5.4 pp.** **Trade-off:** perde-se granularidade
auditável vs ganha-se accuracy reportável. Papers frequentemente
optam pelo accuracy maior.

### Resposta sintética

**Os estudos fazem merge por uma combinação de fatores estruturais:**

1. **Cross-dataset comparability** (sobreposição mínima de taxonomias).
2. **Annotation reliability** (categorias fuzzy entre humanos).
3. **Class imbalance** mesmo em "balanced" datasets.
4. **Ambiguidade conceitual** intrínseca de algumas categorias
   (East/SE Asian, Latinx/Hispanic).
5. **Métricas multi-classe não-padronizadas** (Q05).
6. **Trade-off accuracy×granularidade** — accuracy reportável é
   prioridade competitiva.

### Implicação para nossa pesquisa

1. **Justifica reportar AMBOS 7-class E 6-class** seguindo AlDahoul.
2. **Reconhecer fronteira fuzzy** entre East/SE Asian e
   Latinx/Hispanic na discussão metodológica.
3. **Não comparar diretamente nossos números contra RFW/BFW** (taxonomia
   diferente; comparação é falsa).

---

## Q09 — 7 classes é realmente a taxonomia correta para fairness racial facial?

- **Status:** ❌ OPEN → 🔬 NEW RESEARCH FRONT
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[neto_2025]], [[buolamwini_2018]], [[dataset_hazirbas_2021]], [[schumann_2023]], [[survey_kotwal_2025]], [[survey_mehrabi_2021]]

### Evidências coletadas — argumentos POR e CONTRA

**Argumentos POR 7 classes (FairFace):**

- Inclusão explícita de Middle Eastern e Latinx (ausentes em RFW/BFW).
- Distinção East × Southeast Asian — relevante para
  representatividade global.
- Granularidade > 4 → melhor para auditoria interseccional.
- Aceitação pela literatura (Q02 — dataset mais usado).

**Argumentos CONTRA 7 classes (e contra discretização em geral):**

1. **Race é socialmente construída** (não biológica):
   - US Census 2010: 6 categorias + Hispanic como **ethnia separada**.
   - IBGE (Brasil): 5 categorias (Branca, Preta, Parda, Amarela,
     Indígena).
   - Sistemas latino-americanos: contínuo de mestiçagem.
   - **Qualquer número discreto é arbitrário** ([[neto_2025]] formaliza
     este argumento).

2. **Latinx/Hispanic é etnia, não raça** (consenso US Census).
   FairFace trata como raça por "facial appearance" —
   [[dataset_karkkainen_2021]] §3.1. Esta escolha **introduz
   confusão** (caso mais difícil — F1=60% em todos os modelos).

3. **Buolamwini & Gebru (2018)** argumentam explicitamente contra
   race labels:
   > *"Race and ethnic labels are unstable... we decided to use skin
   > type as a more visually precise label."*

4. **Hazirbas et al. (2021)** [[dataset_hazirbas_2021]] reforça:
   > *"There may be no difference in facial appearance of
   > African-American and African people, although they may be
   > referred to with two distinct racial categories."*

5. **Neto et al. (2025)** [[neto_2025]] demonstra **empiricamente**
   que modelos treinados em datasets balanceados no **espaço
   contínuo** superam discreto-balanceados — mesmo com 50% menos
   identidades.

6. **Schumann et al. (2023)** [[schumann_2023]] mostra que anotadores
   de diferentes regiões divergem **sistematicamente** sobre skin
   tone (MST). Por extensão, divergem **mais ainda** sobre race
   (que é mais subjetiva).

7. **Kotwal & Marcel (2025)** [[survey_kotwal_2025]] sintetiza:
   *"Wang et al. observed that even race-balanced datasets failed to
   eliminate accuracy differentials, hypothesizing that certain
   ethnicities are inherently more challenging to recognize."*
   **Hipótese alternativa nossa:** não é "inherently more challenging";
   é **rótulo mal definido**.

### Resposta

**NÃO há resposta definitiva** e nem deveria haver — a pergunta
revela uma tensão estrutural da literatura:

- **7 classes (FairFace) é o melhor compromisso prático** disponível
  para race classification — granular, inclusivo, balanceado em
  data, amplamente aceito.
- **MAS 7 classes não é "correto"** em sentido absoluto:
  - Race é construto social, não biológico.
  - Latinx como categoria racial é metodologicamente problemática.
  - Categorias fuzzy (East/SE Asian) refletem realidade demográfica.
  - Skin tone (MST) seria mais defensável biologicamente.
  - Rótulos contínuos seriam mais defensáveis estatisticamente.

### 🔬 Frente nova identificada

**A pergunta "7 classes é correto?" abre 3 sub-frentes:**

1. **Continuação discreta:** seguir literatura dominante. Pragmática.
2. **Transição skin tone:** abandonar race, usar MST (alinhado com
   Schumann 2023, Hazirbas 2021).
3. **Transição contínua:** rótulos contínuos (alinhado com Neto
   2025).

**Para a dissertação:** propostamos um **caminho híbrido — Q10**:
**manter 7 classes (FairFace) E adicionar Fitzpatrick/MST como
dimensão paralela**, construindo a matriz de associação. Isto
preserva comparabilidade com literatura existente E adiciona
contribuição metodológica conectando dois tracks paralelos.

---

## Q10 — Existe matriz associativa Fitzpatrick/MST × FairFace 7-race?

- **Status:** ❌ OPEN → 🔬 NEW RESEARCH FRONT (**CANDIDATA PRINCIPAL** para contribuição experimental da dissertação)
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[dataset_karkkainen_2021]], [[dataset_hazirbas_2021]], [[schumann_2023]], [[buolamwini_2018]], [[lafargue_2025]], [[neto_2025]]
- **Pesquisa externa:** Draelos, Kesty & Kesty (J. Cosmetic Dermatology, 2025), Schumann et al. NeurIPS 2023 (MST-E).

### Evidências coletadas

**O que existe (precedentes parciais):**

| Trabalho | O que faz | Por que NÃO responde Q10 |
|---|---|---|
| **FairFace original Figure 1** ([[dataset_karkkainen_2021]]) | ITA distribuição por raça | Apenas plot ilustrativo; sem matriz tabular; sem validação manual |
| **Draelos et al. 2025** (J Cosmetic Dermatology) | Aplica Fitzpatrick a subset FairFace (parte de "SkinAnalysis 3 662 imagens") | NÃO cross-referencia com race labels; foco dermatológico (pigmentação, vermelhidão, rugas); **dados não públicos** ("privacy/ethical restrictions") |
| **Casual Conversations** ([[dataset_hazirbas_2021]]) | Fitzpatrick anotado | **Não tem race labels** (design) |
| **MST-E** ([[schumann_2023]]) | MST anotado | **Não usa FairFace** |
| **Lafargue 2025** ([[lafargue_2025]]) | Fitzpatrick + ITA + skin segmentation | **Usa Generated Photos + CelebA, não FairFace** |
| **PPB** ([[buolamwini_2018]]) | Fitzpatrick por dermatologista | 1 270 sujeitos; não FairFace |

### Resposta

**NÃO existe trabalho público que construa a matriz P(Fitzpatrick | race) ou P(MST | race) sobre o dataset FairFace.**

A peça mais próxima é Draelos et al. 2025, mas é (i) dermatológica,
(ii) subset não-público, (iii) sem cross-reference com race labels.

### 🔬 Desenho experimental proposto

**Justificativa:** se construirmos a matriz e ela revelar (como
hipótese) que **Latinx span ampla faixa de skin tone com sobreposição
com White, Middle Eastern e Indian**, podemos decompor o "erro
intrínseco" da classe Latinx em duas componentes:

- **Componente A — Erro fenotípico estrutural:** ambiguidade visual
  com classes vizinhas (não-redutível por modelo).
- **Componente B — Erro de modelo:** falha de aprender features
  discriminativas (redutível por mitigação algorítmica).

**Fase 1 — Anotação automatizada (≥10 954 imagens, FairFace val set):**

- Aplicar modelo MST automatizado disponível (Schumann et al. 2023
  open-source da Google ou model auxiliar treinado em MST-E).
- Saída: para cada imagem `i ∈ FairFace_val`, vetor de logits sobre
  10 categorias MST → MST_pred(i).
- Custo: cpu/gpu hours; sem custo humano direto.

**Fase 2 — Validação manual (subset 500–700 imagens estratificadas):**

- Amostragem estratificada: 100 imagens por raça (7 × 100 = 700) ou
  proporcional ao train (e.g., White 100, Latinx 100, ME 70, etc.).
- **3 anotadores regionalmente diversos** (recomendação Schumann
  2023): e.g., um do Brasil, um dos EUA, um da Índia ou Oriente
  Médio.
- **Material de treinamento:** MST-E dataset.
- **Voto majoritário + reportar inter-rater agreement (κ de Fleiss)
  por classe racial.**
- Comparar MST manual vs MST_pred automatizado → medir confiabilidade
  do classifier automatizado.

**Fase 3 — Construção da matriz:**

Tabela P(MST_k | race_j), k ∈ {1..10}, j ∈ {7 raças FairFace}:

- Linhas: 7 raças.
- Colunas: 10 MST.
- Células: frequência condicional.
- Identificar **overlapping zones** (e.g., MST 4-6 onde White,
  Latinx, Middle East coexistem).

**Fase 4 — Análise diagnóstica:**

- Pegar confusion matrix do nosso modelo ResNet/ConvNeXt sobre
  FairFace val 7-class race.
- Cross-reference: das misclassificações Latinx→White e Latinx→ME,
  qual % está em MST overlapping zone vs MST non-overlapping?
- Hipótese: se >70% das misclassificações estão em overlapping zone,
  conclui-se que **erro Latinx é majoritariamente fenotípico
  estrutural** (Componente A), não falha de modelo.

**Fase 5 — Publicação:**

- Matriz pública (open dataset CSV).
- Análise como capítulo da dissertação OU paper standalone
  (workshop CVPRW Fair AI, FAccT, ou TBIOM extended).
- Posição: **conecta tracks paralelos** (race classification ←→ skin
  tone fairness) que a literatura tratou separadamente.

### Estimativa de custo e cronograma

| Fase | Trabalho | Tempo estimado |
|---|---|---|
| Fase 1 (automatizado) | Implementar pipeline MST classifier + rodar sobre val set | 2-3 semanas |
| Fase 2 (manual) | Recrutar anotadores; protocolo + treinamento; 500-700 imagens × 3 anotadores | 4-6 semanas (gargalo principal) |
| Fase 3 (matriz) | Análise estatística | 1 semana |
| Fase 4 (diagnóstico) | Cross-reference com confusion matrix do modelo | 1-2 semanas |
| Fase 5 (escrita) | Capítulo de dissertação | 3-4 semanas |
| **Total estimado** | | **3 meses** |

### Risco e mitigação

- **Risco:** anotadores manuais difíceis de recrutar com diversidade
  regional.
- **Mitigação A:** começar com Fase 1 + Fase 3-4 apenas com MST
  automatizado (proof of concept).
- **Mitigação B:** Fase 2 pode ser feita com **5 anotadores via
  Prolific** (mais barato que MTurk, melhor controle de demografia).

### Encaminhamento

**Esta frente Q10 🔬 deve ser registrada em `06_gap.md` como CANDIDATA
PRINCIPAL** para a contribuição experimental original da dissertação.
Discutir prioridade vs alternativas (mitigação algorítmica testando
H1-H6 das Q04 e Q06) durante elaboração de `07_thesis_statement.md`.

---

## Q11 — As 7 categorias raciais do FairFace têm fundamento biológico ou são socio-políticas?

- **Status:** ✅ ANSWERED (FUNDAMENTAL para a tese)
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[fuentes_2019]], [[lewontin_1972]], [[dataset_karkkainen_2021]], [[neto_2025]], [[buolamwini_2018]]

### Evidências coletadas

**Da American Association of Biological Anthropologists (AAPA/AABA),
statement oficial 2019** ([[fuentes_2019]]):

> *"Race does not provide an accurate representation of human
> biological variation. It was never accurate in the past, and it
> remains inaccurate when referencing contemporary human populations."*

> *"Humans are not divided biologically into distinct continental
> types or racial genetic clusters."*

> *"The Western concept of race must be understood as a
> classification system that emerged from, and in support of,
> European colonialism."*

**Da genética populacional clássica** ([[lewontin_1972]]):

- **85.4%** da variação genética humana é **dentro** de qualquer
  população local.
- Apenas **6.3%** é **entre** categorias raciais convencionais.
- Confirmado por estudos genômicos modernos (Rosenberg et al. 2002,
  Bergstrom et al. 2020).

**Da literatura facial de fairness:**

- Karkkainen & Joo (2021) **declaram explicitamente** em §3.1:
  *"Race is not a discrete concept and needs to be clearly defined
  before data collection."*
- Buolamwini & Gebru (2018): *"Race and ethnic labels are
  unstable... we decided to use skin type as a more visually precise
  label."*
- Hazirbas et al. (2021): *"Labeling the ethnicity of subjects could
  lead to inaccuracies."*
- Neto et al. (2025): demonstra empiricamente que **continuous**
  > **discrete** balanceamento para fairness.

### Resposta

**As 7 categorias raciais do FairFace (White, Black, Indian, East
Asian, Southeast Asian, Middle Eastern, Latino) NÃO têm fundamento
biológico em sentido científico estrito.** São **construto socio-
político**, derivadas em parte do US Census Bureau e ajustadas pelos
autores do FairFace para incluir grupos previamente sub-representados.

**Argumentos hierarquizados:**

1. **Genético** (Lewontin 1972 + replicações modernas): a variação
   genética humana **não se particiona** majoritariamente entre
   categorias raciais — só 6% inter-grupos, 85% intra-grupos.
2. **Antropológico** (AABA 2019): consenso institucional de que
   raça é **construto colonial**, não biológico.
3. **Histórico**: categorias raciais variam por país (US Census 6
   + Hispanic ethnia separada; IBGE Brasil 5; FairFace 7) — não há
   "verdadeiro número de raças".
4. **Operacional**: mesmo os criadores do FairFace reconhecem
   arbitrariedade da escolha (descartaram Hawaiian/Pacific Islanders,
   Native Americans por amostragem insuficiente — escolha **prática**,
   não biológica).

### Implicação para a dissertação

**A tese v3 é fundamentada teoricamente:** o "ceiling Latinx 60% F1"
**não é falha do dataset ou do modelo**; é **consequência matemática
direta** de impor categorização discreta sobre fenótipo contínuo. Q10
(matriz MST × race) quantifica essa imposição.

**Citação canônica obrigatória:** Fuentes et al. (2019) + Lewontin
(1972) em introdução teórica da dissertação.

---

## Q12 — Como antropologia forense moderna trata "raça"? Qual o consenso?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[fuentes_2019]], [[lewontin_1972]]
- **Pesquisa externa:** Sparks & Jantz 2002 (PNAS, "Boas revisited"),
  Ousley & Jantz forensic literature.

### Evidências coletadas

**Posição moderna da antropologia forense (consolidada por AABA
2019 e literatura post-2000):**

1. **Termo "race" foi substituído por "population affinity" ou
   "biogeographical ancestry"** em literatura técnica forense.
2. **Cranial measurements estimam probabilísticamente** a origem
   geográfica de um esqueleto desconhecido — **não classificam em
   raças discretas**.
3. **Sparks & Jantz (2002, PNAS):** mostra que dimensões cranianas
   têm **alta estabilidade genética** (refutando Boas 1912 sobre
   plasticidade massiva), MAS essa estabilidade permite **identificar
   ancestralidade geográfica**, não validar taxonomia racial.
4. **Edwards (2003), "Lewontin's Fallacy":** com **suficientes
   loci/features**, classificação populacional é estatisticamente
   possível — mas isso NÃO equivale a justificação biológica de
   categorias raciais discretas (Lewontin permanece correto sobre
   a estrutura central da variação).

### Resposta

**A antropologia forense moderna abandonou "race" como categoria
biológica e adotou "population affinity" / "biogeographical
ancestry"** — termos que reconhecem:

- Existência de **clusters fenotípicos correlacionados com origem
  geográfica** (medíveis em ossos, dentes, e fenótipos visíveis).
- **Continuidade da variação** entre regiões — clusters têm bordas
  borradas.
- **Probabilidade estatística**, não certeza taxonômica, na atribuição
  individual.

**Distinção conceitual crítica:**

- "Population affinity" / "ancestry": **válido cientificamente**, é o
  que sistemas faciais MEDEM (clusters de feature space).
- "Race": **construto social**, é o que sistemas faciais ROTULAM.

Sistemas de classificação facial racial **conflagam** os dois —
medem ancestralidade probabilística e rotulam como categoria racial
discreta.

### Implicação para a dissertação

Nossa tese v3 pode ser **reformulada com mais precisão** usando
esta distinção:

> O ceiling em FairFace race 7-class reflete o **gap epistemológico**
> entre **ancestry probabilística** (o que modelos medem) e **categoria
> racial discreta** (o que FairFace rotula). Latinx é caso paradigmático:
> ancestry continua entre múltiplos clusters geográficos (Indígena
> americano + Europeu + Africano), forçada em rótulo único.

---

## Q13 — Origem e propósito da Fitzpatrick scale — para que foi criada cientificamente?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[fitzpatrick_1988]], [[buolamwini_2018]],
  [[dataset_hazirbas_2021]], [[lafargue_2025]], [[schumann_2023]]

### Evidências coletadas

**Da ficha [[fitzpatrick_1988]]:**

- **Ano**: conceito introduzido em **1975** por Thomas B. Fitzpatrick
  (dermatologista, Harvard Medical School).
- **Propósito original**: **prescrever dose inicial de UVA segura**
  em fotoquimioterapia oral com metoxsaleno (PUVA) para psoríase.
- **Variável proxy**: histórico de queimadura solar + capacidade de
  bronzear (não cor de pele direta).
- **Validade revisada**: paper de 1988 (*Arch Dermatol*) ampliou para
  6 tipos (originalmente 4) — admitindo que escala original era
  **inadequada para skin of color**.

**Da prática clínica (literatura moderna em dermatologia):**

- Hall et al. (2022): "Fitzpatrick scale gives a false sense of
  security for skin cancer risk in skin types IV-VI" — escala
  subestima risco em pele escura.
- **1/3 dos dermatologistas confunde Fitzpatrick com raça/etnia**
  — erro categorial documentado.

### Resposta

**A escala Fitzpatrick foi criada para um propósito DERMATOLÓGICO
ESPECÍFICO — dosagem segura de UV em fotoquimioterapia — e NÃO
para classificação racial.**

Confusão posterior na literatura de ML fairness (e em parte da
dermatologia) **estende uso indevido** da escala:

- Buolamwini & Gebru (2018) — **escolheram Fitzpatrick justamente por
  ser mais "estável" que raça**, mas reconhecem limitações da escala.
- Hazirbas et al. (2021) — **adotam Fitzpatrick** em Casual
  Conversations, mas defendem a escolha por dificuldade de race
  labeling (não por adequação biométrica).
- Lafargue et al. (2025) — usam Fitzpatrick + ITA; reconhecem MST
  como sucessor mas adotam Fitzpatrick por simplicidade.
- Schumann et al. (2023) — argumentam que Fitzpatrick é
  **caucasiano-cêntrica** (3/6 tipos cobrem "perceived White") e
  propõem MST 10-point como alternativa.

### Implicação para a dissertação

1. **Citar Fitzpatrick 1988 explicitamente** ao introduzir a escala —
  reconhecer propósito original.
2. **Justificar adoção de MST** em vez de Fitzpatrick para Q10:
   MST foi explicitamente desenhada para fairness research.
3. **Discussão metodológica honesta:** a escala Fitzpatrick **funciona
   como proxy** mas não é instrumento purpose-built para fairness.

---

## Q14 — Quantos tons de pele existem cientificamente (medicina/biometria)?

- **Status:** ✅ ANSWERED
- **Data:** 2026-05-25
- **Última atualização:** 2026-05-25
- **Fichas consultadas:** [[fitzpatrick_1988]], [[massey_martin_2003]],
  [[schumann_2023]], [[lafargue_2025]]

### Evidências coletadas

**Inventário de escalas de tom de pele em uso científico:**

| Escala | Ano | Categorias | Origem disciplinar | Propósito | Status atual |
|---|---|---|---|---|---|
| **Felix von Luschan** | 1897 | 36 | Antropologia (eugenista) | Classificação racial colonial | **Descontinuada** |
| **Fitzpatrick** | 1975/1988 | 6 (I-VI) | Dermatologia | Dosimetria PUVA | **Dominante em derma**; criticada por viés |
| **NIS / Massey-Martin** | 2003 | 11 (0-10) | Sociologia (NIS, GSS, ANES) | Estudos de colorism | **Estabelecida em ciência social** |
| **ITA (Individual Typology Angle)** | 1991 (Chardon) | **Contínuo** (graus) | Dermatologia / biometria | Medição objetiva via L*a*b* colorimetria | **Padrão biométrico** em pesquisa |
| **Monk Skin Tone (MST)** | 2023 | 10 | Sociologia + tech (Monk + Google) | Fairness em ML | **Padrão emergente** pós-2023 |

### Resposta

**Não há "número cientificamente correto" de tons de pele.** Cada
escala é **instrumento** para um propósito específico, com tradeoffs
diferentes:

- **Mais granular = mais inclusivo**, mas **menor inter-rater
  agreement** (Schumann 2023 documenta cognitive load em escalas
  finas).
- **Menos granular = mais confiável**, mas **perde nuance**.
- **Contínua (ITA)** = objetiva mas exige espectrofotometria, não
  anotação visual.

**Hierarquia de adequação por propósito:**

| Propósito | Escala recomendada |
|---|---|
| Dosimetria UV / dermatologia clínica | Fitzpatrick (origem) |
| Estudos sociológicos de colorism | Massey-Martin / NIS (11 pts) |
| Fairness em ML / visão computacional | **MST (Monk, 2023)** — desenhada para isso |
| Medição biométrica objetiva | ITA contínuo |

**Para nossa pesquisa (Q10):** **MST 10-point é a escolha correta**:

- Desenvolvida explicitamente para fairness research.
- Mais granular e menos caucasiano-cêntrica que Fitzpatrick.
- Estabelecida via partner Google + research peer-reviewed (Schumann
  NeurIPS 2023).
- API publicamente disponível ([skintone.google](https://skintone.google)).

### Implicação para a dissertação

**Adotar MST em Q10** com justificativa explícita ancorada em Q14.
Reportar também ITA contínuo onde possível (proxy biométrico
objetivo).

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
