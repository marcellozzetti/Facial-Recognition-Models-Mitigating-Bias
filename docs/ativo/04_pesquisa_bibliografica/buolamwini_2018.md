---
name: buolamwini-2018
status_verificacao: VERIFIED
autores: [Joy Buolamwini, Timnit Gebru]
ano: 2018
titulo: "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification"
venue: "Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAT*, agora FAccT), PMLR vol 81, pp. 77-91"
tipo_publicacao: conference
arxiv_id: null
doi: null
url_primario: https://proceedings.mlr.press/v81/buolamwini18a.html
citacoes_google_scholar: null
citacoes_semantic_scholar: 4933
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~50
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/buolamwini_2018_gendershades.pdf)
---

# Gender Shades (Buolamwini & Gebru, 2018)

## 1. Resumo do problema atacado

Sistemas comerciais de classificação de gênero (Microsoft, IBM, Face++)
são apresentados ao mercado sem qualquer **breakdown demográfico** de
acurácia. Benchmarks vigentes (IJB-A, Adience) são desbalanceados em
tom de pele (79.6% e 86.2% de "lighter-skinned" respectivamente),
impedindo avaliação interseccional. O paper introduz a **primeira
auditoria pública de viés algorítmico em visão computacional comercial**,
combinando: (i) novo dataset balanceado por tom de pele × gênero; (ii)
métrica de erro por subgrupo interseccional (4 células: darker female,
darker male, lighter female, lighter male).

## 2. Método

### 2.1 Construção do Pilot Parliaments Benchmark (PPB)

- **Fontes:** fotos oficiais de parlamentares de 6 países, escolhidos
  por dois critérios cruzados:
  - **Gênero**: ranking de paridade da Inter-Parliamentary Union
    (Rwanda lidera; Nordics no top 10).
  - **Tom de pele**: 3 países africanos (Rwanda, Senegal, África do
    Sul) + 3 nórdicos (Islândia, Finlândia, Suécia).
- **Total:** 1 270 indivíduos únicos. Fotos oficiais → pose constante,
  iluminação controlada, expressão neutra/sorriso. **Não é
  in-the-wild.**

### 2.2 Anotação de tom de pele — Fitzpatrick Skin Type

- Escala dermatológica padrão I (mais claro) a VI (mais escuro).
- Para PPB: **dermatologista certificado** forneceu rótulos definitivos
  (gold standard); 3 anotadores (incluindo as autoras) deram rótulos
  preliminares.
- **Agregação binária para análise:** lighter = {I, II, III}; darker =
  {IV, V, VI}. Justificativa: tolerar off-by-one no rótulo.

### 2.3 Decisão metodológica fundamental: por que tom de pele, não raça

> *"Race and ethnic labels are unstable... we decided to use skin type
> as a more visually precise label to measure dataset diversity."*

Argumento explícito do paper:
- Variação fenotípica dentro de uma categoria racial é alta.
- Categorias raciais mudam entre países e ao longo do tempo.
- Câmeras default são calibradas para subexpor pele clara (Roth 2009).
- Tom de pele é **medível** com escala científica (Fitzpatrick).

### 2.4 Classificadores auditados

- **Microsoft Cognitive Services Face API**
- **IBM Watson Visual Recognition**
- **Face++** (chinesa, escolhida para testar hipótese de "in-group
  bias" — Phillips et al. 2011 sobre Western vs Asian face recognition)
- Avaliados via API pública, abril-maio 2017.
- **Nenhuma das empresas reporta métricas de performance demográfica
  na documentação**, conforme registro do paper.

### 2.5 Métricas de avaliação

- **TPR (True Positive Rate)** por subgrupo.
- **Error rate = 1 − TPR**.
- **PPV (Positive Predictive Value)** = precision.
- **FPR (False Positive Rate)**.
- Reportados por 9 cortes: All, F, M, Darker, Lighter, DF, DM, LF, LM.

## 3. Datasets e setup experimental

- **PPB:** 1 270 indivíduos = 21.3% DF + 25.0% DM + 23.3% LF + 30.3% LM.
- **Benchmarks comparados (apenas para mostrar desbalanceamento):**
  - **IJB-A** (NIST 2015): 500 sujeitos, 79.6% lighter.
  - **Adience** (Levi & Hassner 2014): 2 194 sujeitos, 86.2% lighter.
- South Africa isolada como subgrupo (n=437, 79.2% darker) para
  controlar potencial confundidor de qualidade de imagem.

## 4. Métricas reportadas

Por subgrupo: TPR, error rate, PPV, FPR. Total de 4 métricas × 9
subgrupos × 3 classificadores = 108 valores numéricos individuais.

## 5. Resultados principais (valores numéricos)

### 5.1 Tabela 4 — Performance interseccional sobre PPB integral

Error rate (%) por classificador × subgrupo:

| Classificador | All | F | M | Darker | Lighter | DF | DM | LF | LM |
|---|---|---|---|---|---|---|---|---|---|
| **Microsoft** | 6.3 | 10.7 | 2.6 | 12.9 | 0.7 | **20.8** | 6.0 | 1.7 | **0.0** |
| **Face++** | 10.0 | 21.3 | 0.7 | 16.5 | 4.7 | **34.5** | 0.7 | 9.8 | 0.8 |
| **IBM** | 12.1 | 20.3 | 5.6 | 22.4 | 3.2 | **34.7** | 12.0 | 7.1 | 0.3 |

**Achado-bandeira:** **gap entre lighter male e darker female =
34.7 pp** (IBM) — a maior disparidade já documentada em sistemas
comerciais de visão até 2018.

### 5.2 Concentração de erro em darker females

- Darker females são 21.3% do PPB **mas concentram 61.0%-72.4% dos
  erros totais** dos classificadores.
- Lighter males são 30.3% do PPB **mas contribuem com 0.0%-2.4% dos
  erros**.

### 5.3 Tabela 5 — South Africa isolada

Quando filtrado por país (South Africa, n=437):

| Classificador | DF Error | DM Error | LF Error | LM Error |
|---|---|---|---|---|
| Microsoft | 23.8 | 0.0 | 0.0 | 0.0 |
| Face++ | 36.0 | 0.5 | 7.4 | 0.0 |
| IBM | 33.1 | 5.7 | 0.0 | 1.6 |

**Implicação:** o gap **não desaparece** ao controlar país (qualidade
de imagem, condições de captura). O viés é **demográfico-fenotípico**,
não de qualidade.

### 5.4 Hipótese causal proposta pelas autoras

> *"Darker skin may be highly correlated with facial geometries or
> gender display norms that were less represented in the training
> data of the evaluated classifiers."*

As autoras não atribuem o erro a "pele escura" isoladamente, mas a
**representação insuficiente de combinações fenotípicas no treino**.

## 6. Limitações declaradas pelos autores

- **Gênero binário:** o paper reconhece explicitamente que classificar
  gênero como binário "does not adequately capture the complexities
  of gender or address transgender identities". Use-se "male/female"
  por restrição imposta pelas APIs auditadas, não por concordância
  conceitual.
- **PPB é constrained:** fotos oficiais de parlamentares têm pose
  constante, iluminação controlada, expressão neutra → resultado
  pode subestimar erros in-the-wild.
- **Escala Fitzpatrick é coarse e enviesada para pele clara:** 3 das
  6 categorias cobrem o espectro "perceived as White", deixando o
  "sepia spectrum" pouco resolvido.
- **Single image per subject:** apenas uma imagem por parlamentar foi
  rotulada — sem variabilidade intra-individual.
- **Auditoria de caixa-preta:** sem acesso aos dados de treino dos
  3 classificadores; não conseguem provar a causa raiz do viés,
  apenas medi-lo.

## 7. Limitações que identifiquei (leitura crítica)

- **Tom de pele ≠ raça** (escolha consciente das autoras), o que torna
  resultados não diretamente transponíveis para tarefas que usam
  taxonomia racial (caso da nossa pesquisa com FairFace 7-class). É
  preciso re-validar empiricamente a propagação desses achados quando
  a variável de auditoria é raça e não Fitzpatrick.
- **Escalonamento estatístico:** 1 270 sujeitos × 4 células
  interseccionais → ~318 sujeitos por célula. Para classificadores
  com error rate de ~1%, intervalos de confiança são largos. O paper
  não reporta CIs.
- **Single image per subject** elimina variância intra-individual mas
  também impede análise de robustez (foto ruim de bom modelo vs foto
  boa de mau modelo).
- **Os classificadores auditados eram caixas-pretas comerciais.**
  Empresas refizeram seus produtos pouco depois da publicação — os
  números de 2017 não são reproduzíveis em 2026. O paper é
  historicamente importante como **prova de viés**, não como referência
  numérica viva.
- **Generalização para outras tarefas:** o paper auditou gênero. Ele
  não testa **raça/idade/emoção**, embora a literatura subsequente
  tenha estendido a metodologia.
- **Ausência de discussão sobre "lighter ≠ White"** — os clusters
  fenotípicos não correspondem 1:1 a categorias raciais. Para uma
  audiência iniciante o paper pode propagar a equivalência errônea.

## 8. Relação com nossa pesquisa

**Centralidade narrativa:** este é o **marco fundador** do campo de
fairness em visão computacional comercial. Toda dissertação séria sobre
viés demográfico facial cita este paper.

**Pontos de ancoragem:**

1. **Justificativa ética/social do trabalho:** o gap 34.7 pp documentado
   por Gender Shades **estabeleceu o campo**. Nossa pesquisa atua sobre
   o mesmo problema (disparidade demográfica em sistemas faciais),
   estendendo de gênero × tom de pele para **raça × classificação
   facial em 7 classes**.
2. **Razão de disparidade como métrica herdeira:** nosso DR = max/min
   de F1 por classe é descendente direto do "max error rate disparity"
   de Gender Shades. Ambos quantificam a **pior pior-experiência**.
3. **Lente paradigmática:** Gender Shades introduz a noção de
   **avaliação interseccional**. Nossa pesquisa em 7 classes raciais
   (sem corte transversal por gênero) é **uni-dimensional na demografia**
   — uma limitação a reconhecer. Análises pós-treinamento interseccionais
   (raça × idade, raça × gênero) ficam como **trabalho futuro
   explicitamente fundamentado** em Buolamwini & Gebru.
4. **Decisão tom-de-pele vs raça:** o paper escolhe Fitzpatrick **por
   estabilidade**; o FairFace escolhe raça **por aplicabilidade em
   ciências sociais**. As duas escolhas são complementares; nossa
   pesquisa opera no eixo raça (FairFace) e pode comentar a limitação
   conceitual à luz de Gender Shades.
5. **Padrão de reporte:** Gender Shades reporta 4 métricas × 9
   subgrupos. Nossa pesquisa reporta accuracy, F1, recall, DR por
   classe + interseccional pós-treino — o nível de granularidade é
   compatível, ainda que com nomenclatura diferente.

## 9. Pontos para citar / posicionar

- *"A auditoria seminal de Buolamwini e Gebru (2018) sobre três
  sistemas comerciais demonstrou disparidades de até 34.4 pontos
  percentuais na taxa de erro entre faces de homens claros e mulheres
  escuras, estabelecendo o campo de auditoria interseccional em visão
  computacional."*
- *"Embora Buolamwini e Gebru (2018) tenham operado sobre tom de pele
  (Fitzpatrick), pela maior estabilidade conceitual do rótulo, a
  literatura subsequente (Kärkkäinen & Joo, 2021) optou por categorias
  raciais para preservar aplicabilidade em domínios de ciências
  sociais. Esta dissertação adota a segunda opção pela ancoragem no
  dataset FairFace."*
- *"A descoberta de que darker females, com 21.3% da composição do
  benchmark, concentram entre 61% e 72% dos erros dos classificadores
  comerciais (Buolamwini & Gebru, 2018, Tabela 4) ilustra que viés
  demográfico não se manifesta como deslocamento médio, mas como
  concentração de erro em subgrupos minoritários — o que motiva
  métricas de disparidade (DR, CV, Gini) em vez de accuracy global."*

## 10. Arquivos relacionados

- PDF local: `pdfs/buolamwini_2018_gendershades.pdf` (gitignored).
- Texto extraído: `pdfs/buolamwini_2018_gendershades.txt` (gitignored).
- Entradas relacionadas: [[dataset_karkkainen_2021]] (resposta direta
  ao gap identificado por Gender Shades, mas mudando para raça),
  [[grother_2019]] (auditoria NIST industry-wide subsequente),
  [[dehdashtian_2024]] (trade-off teorizado posteriormente).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 3.2, linha F10.
