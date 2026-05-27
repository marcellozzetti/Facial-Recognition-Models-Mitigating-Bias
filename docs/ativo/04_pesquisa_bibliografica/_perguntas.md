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
