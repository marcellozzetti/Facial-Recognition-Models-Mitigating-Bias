---
name: grother-2019
status_verificacao: VERIFIED
autores: [Patrick J. Grother, Mei L. Ngan, Kayee K. Hanaoka]
ano: 2019
titulo: "Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects"
venue: "NIST Interagency Report 8280 (NISTIR 8280)"
tipo_publicacao: technical_report
arxiv_id: null
doi: "10.6028/NIST.IR.8280"
url_primario: https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~50
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/grother_2019_nistir8280.pdf), 81 páginas + anexos não lidos
---

# NISTIR 8280: FRVT Part 3 — Demographic Effects (Grother, Ngan & Hanaoka, 2019)

## 1. Resumo do problema atacado

Em 2018, após Gender Shades, asserções públicas sobre viés demográfico
em reconhecimento facial eram baseadas em estudos de **classificação**
(gender, race), não em **reconhecimento** propriamente dito (1:1
verificação + 1:N identificação). NIST observa essa lacuna: *"Much of
the discussion of face recognition bias in recent years cites two
studies showing poor accuracy of face gender classification algorithms
on black women. Those studies did not evaluate face recognition
algorithms, yet the results have been widely cited to indict their
accuracy."*

O relatório quantifica diferenciais demográficos em **189 algoritmos
comerciais e acadêmicos** sobre **18.27 milhões de imagens de 8.49
milhões de pessoas**, em 4 datasets governamentais americanos.
Constitui a **maior auditoria industrial pública** de viés em FR já
publicada.

## 2. Método

### 2.1 Algoritmos avaliados

- **189 algoritmos** de **99 developers** (corporativos + universidades).
- Submetidos ao FRVT entre 2018–2019 como protótipos (não-products).
- Performance individual documentada nos FRVT 1:1 e 1:N reports
  (NISTIR 8271, 8238).

### 2.2 Datasets

| Dataset | Tipo | Volume aprox. | Race metadata? |
|---|---|---|---|
| Domestic mugshots | Booking photos (EUA) | milhões | ✅ explícito |
| Application photos | Immigration benefits applicants (global) | milhões | apenas country-of-birth |
| Visa photos | Visa applicants | milhões | apenas country-of-birth |
| Border crossing photos | Travelers entering EUA | milhões | apenas country-of-birth |

**Substituição de "raça" por country-of-birth:** para os 3 datasets
sem race metadata, NIST restringe análise a **24 países em 7 regiões
globais** com baixa imigração de longa distância, usando
country-of-birth como proxy. **Esta é uma decisão metodológica
importante e diferente da literatura acadêmica (que usa rótulos raciais
diretos como FairFace).**

### 2.3 Tarefas e métricas

- **1:1 Verification:** dois fotos comparadas → "mesma pessoa?". Erros:
  - **FMR** (False Match Rate) = false positive.
  - **FNMR** (False Non-Match Rate) = false negative.
- **1:N Identification:** probe vs gallery → top-N candidatos. Erros:
  - **FPIR** (False Positive Identification Rate).
  - **FNIR** (False Negative Identification Rate).

### 2.4 Decisão metodológica crítica: **threshold fixo**

> *"Most academic studies ignore this point... by reporting false
> negative rates at fixed false positive rates rather than at fixed
> thresholds, thereby hiding excursions in false positive rates and
> misstating false negative rates."*

Sistemas operacionais usam **threshold fixo**, não threshold ajustado
por demografia. NIST reporta FMR e FNMR ambos a threshold fixo,
revelando demographic differentials que estudos acadêmicos normalmente
escondem.

## 3. Datasets e setup experimental

Ver §2.2. Detalhamento adicional:

- Domestic mugshots e Application/Visa photos: alta conformidade com
  padrões de captura (ISO/IEC 19794-5, ICAO 9303).
- Border crossing photos: baixa qualidade (constraints operacionais de
  duração e ambiente).

## 4. Métricas reportadas

- **FMR, FNMR** (1:1) e **FPIR, FNIR** (1:N).
- **Demographic differential** = razão entre métrica em grupo A e
  grupo B, a threshold fixo.

## 5. Resultados principais (valores numéricos)

### 5.1 Achados-bandeira

**False positive differentials são MUITO maiores que false negative
differentials.**

| Tipo de erro | Faixa típica de diferencial |
|---|---|
| False positive (FMR/FPIR) | **10× a 100×** entre demografias |
| False negative (FNMR/FNIR) | usualmente **< 3×** |

### 5.2 Demografia × erro (Application photos, alta qualidade)

| Grupo | False positive |
|---|---|
| West African + East African | mais alto |
| East Asian | mais alto (mas: ver achado in-group abaixo) |
| Eastern European | mais baixo |

**Achado in-group (importante):** algoritmos desenvolvidos na **China**
mostram **FPR baixa em East Asians** — efeito reverso do padrão geral.
Sugere influência da **composição racial do dataset de treino** do
desenvolvedor.

### 5.3 Domestic mugshots — race breakdown

| Grupo | False positive rate (relativo) |
|---|---|
| American Indians | mais alto |
| African American | elevado |
| Asian | elevado |
| White | mais baixo |

(Ordem relativa varia por algoritmo e sexo.)

### 5.4 Sexo

- **Mulheres > Homens** em false positives (consistente entre
  algoritmos e datasets).
- Magnitude **menor que a de raça**.

### 5.5 Idade

- **Elevado** em **crianças** e **idosos**; **baixo** em adultos de
  meia-idade.

### 5.6 False negatives (com baixa qualidade — border crossing)

- Higher em pessoas nascidas em África e Caribe (effect amplified em
  older individuals).
- Forte interação com **qualidade da imagem** — não puro efeito
  demográfico.

### 5.7 1:N vs 1:1

- Differentials presentes em 1:1 **usualmente persistem** em 1:N.
- Algumas algoritmos altamente acurados **eliminam differential em
  false positive** para 1:N — não é universal.

## 6. Limitações declaradas pelos autores

- **Country-of-birth como proxy de raça** é imperfeito; reconhecido
  no relatório.
- **Protótipos**, não produtos comerciais maduros — performance dos
  produtos vendidos pode ser melhor (ou pior).
- Datasets governamentais americanos podem **não generalizar** para
  outras populações operacionais.
- **Interaction com human-in-the-loop** (investigations,
  adjudication) está fora de escopo.
- O relatório não testa **classificação** (race, gender, age) — apenas
  reconhecimento. (Esta é uma fronteira deliberada, não uma omissão.)

## 7. Limitações que identifiquei (leitura crítica)

- **Não cobre face attribute classification.** O paper distingue
  explicitamente recognition (1:1, 1:N) de classification (gender,
  race, age). **Os achados não se transferem automaticamente para
  classification.** Para a tarefa da nossa pesquisa (race
  classification 7-class no FairFace), o NISTIR 8280 é evidência
  industry-wide de viés em **um problema relacionado mas diferente**.
- **Threshold fixo** é correto para sistemas operacionais, mas
  algoritmos modernos (ArcFace, CosFace, DINOv2-based) calibram via
  margin loss — a abstração "threshold fixo" pode ser menos cabível
  hoje que em 2019.
- **Race metadata só em mugshots.** Os 3 outros datasets usam
  country-of-birth como proxy, o que dilui o sinal racial e
  introduz confounders socioeconômicos (imigração legal vs ilegal,
  etc.).
- **Sem reprodução pública**: dados governamentais não-públicos por
  motivos legais; impossível replicar.
- **Lista os 189 algoritmos no anexo** (1200+ páginas adicionais),
  mas comparação entre 2019 e 2026 é difícil — muitos developers
  evoluíram seus algoritmos sem documentação versionada.
- **Falta de informação sobre treino:** datasets de treino dos 189
  algoritmos não são revelados (segredo comercial). Isso impede
  análise causal: differentials são por **dataset** ou por
  **arquitetura** ou ambos?

## 8. Relação com nossa pesquisa

**Centralidade:** NISTIR 8280 é a **referência industry-wide de
escala única** sobre viés demográfico em sistemas faciais.

**Pontos de ancoragem:**

1. **Diferenciação tarefa-essencial:** o paper distingue claramente
   **classification ≠ recognition**. Nossa pesquisa opera em
   classification (race 7-class), não recognition. Isto deve ser
   explicitado: **NISTIR 8280 é evidência indireta**, importante por
   estabelecer que viés demográfico é um problema industry-wide, mas
   o tipo de erro (FMR vs accuracy/F1) e a estatística (10-100x vs
   ~1.5-3x de razão de disparidade) **não são diretamente
   comparáveis**.
2. **Magnitude de differential racial:** 10x-100x FPR entre raças
   estabelece **limite superior** do quão extremo o viés pode ser.
   Nossa razão de disparidade em F1 (na ordem de ~1.1-1.5) é uma
   medida mais branda, mas no mesmo eixo conceitual.
3. **Decisão "threshold fixo":** justifica nossa decisão metodológica
   de reportar accuracy/F1 em **best epoch único por seed** (não
   threshold tuning por classe) — operacionalmente honesto.
4. **In-group effect (algoritmos chineses em East Asians):**
   evidência forte de que **composição racial do dataset de treino
   importa**. Justifica nossa escolha do FairFace (balanceado por
   raça) sobre ImageNet (enviesado para White).
5. **Recomendação NIST:** *"It is necessary to report both false
   negative and false positive rates for each demographic group at
   that threshold."* Adaptado para nossa tarefa: reportar
   accuracy/F1/recall/precision **por classe**, não apenas média —
   prática já estabelecida na dissertação.

## 9. Pontos para citar / posicionar

- *"O relatório NISTIR 8280 (Grother, Ngan & Hanaoka, 2019) constitui
  a maior auditoria pública de viés demográfico em reconhecimento
  facial, abrangendo 189 algoritmos de 99 desenvolvedores sobre 18.27
  milhões de imagens. Estabelece, com escala única, que diferenciais
  de falso positivo entre grupos raciais variam tipicamente em uma a
  duas ordens de magnitude (10× a 100×) em sistemas comerciais
  contemporâneos."*
- *"É importante distinguir, conforme enfatizado por Grother et al.
  (2019), face recognition (verificação 1:1, identificação 1:N) de
  face classification (atributos como raça, idade, gênero). Os dois
  problemas têm taxonomia de erro diferente (FMR/FNMR vs accuracy/F1)
  e diagnósticos de viés não-intercambiáveis. A presente dissertação
  opera no segundo problema."*
- *"Grother et al. (2019) documentam um efeito in-group: algoritmos
  desenvolvidos na China apresentam taxa de falso positivo baixa em
  East Asians, invertendo o padrão geral. Esta observação reforça a
  hipótese de que a composição demográfica do dataset de treinamento
  determina parte substancial do viés observado, motivando o uso de
  datasets balanceados como FairFace (Kärkkäinen & Joo, 2021) para
  trabalhos sobre classificação de raça."*

## 10. Arquivos relacionados

- PDF local: `pdfs/grother_2019_nistir8280.pdf` (gitignored).
- Texto extraído: `pdfs/grother_2019_nistir8280.txt` (gitignored).
- Entradas relacionadas: [[buolamwini_2018]] (Gender Shades — fonte
  citada pelo próprio NIST como motivação),
  [[dataset_wang_2019]] (RFW — análogo acadêmico, em verification),
  [[dataset_karkkainen_2021]] (FairFace — domínio de classification
  como complemento).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S9.
- Anexos NIST com 1200+ páginas de breakdowns por algoritmo: ver
  doi.org/10.6028/NIST.IR.8280.
