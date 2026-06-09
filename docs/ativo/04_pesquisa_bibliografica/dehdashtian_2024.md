---
name: dehdashtian-2024
status_verificacao: VERIFIED
autores: [Sepehr Dehdashtian, Bashir Sadeghi, Vishnu Naresh Boddeti]
ano: 2024
titulo: "Utility-Fairness Trade-Offs and How to Find Them"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
tipo_publicacao: conference
arxiv_id: "2404.09454"
doi: null
url_primario: https://arxiv.org/abs/2404.09454
citacoes_google_scholar: null
citacoes_semantic_scholar: 23
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 40
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/dehdashtian_2024_ufate.pdf)
---

# U-FaTE: Utility-Fairness Trade-Offs (Dehdashtian, Sadeghi & Boddeti, 2024)

## 1. Resumo do problema atacado

Trabalhos prévios em fairness algorítmico tratam o trade-off
utility ↔ fairness como **fato empírico observado** (existem bounds
teóricos para casos restritos: binário, paridade demográfica), mas
**não há método numérico geral para estimá-lo** sobre dados reais.
Consequência prática: quando uma técnica de mitigação produz uma curva
(accuracy, EOD), não se sabe se ela está **próxima do ótimo possível**
ou **muito longe**. O paper propõe duas trade-offs **intrínsecas aos
dados** (DST e LST) e um método (U-FaTE) para estimá-las
empiricamente, transformando o plano accuracy × unfairness em um
mapa com três regiões: **possível**, **possível com dados extras**,
**impossível**.

## 2. Método

### 2.1 Duas trade-offs formais

- **DST (Data-Space Trade-Off):**
  f_λ^DST = argmin_{f ∈ H_X} { (1-λ) E[L_Y(g_Y(f(X)), Y)] + λ Dep(f(X), S | Y=y) }
  Encoder f mapeia X→Z; classifier g prediz Ŷ. Limite achievable
  **com os dados disponíveis**.

- **LST (Label-Space Trade-Off):**
  Limite teórico achievable por um **oracle** que tem acesso direto a
  (Y, S) sem precisar inferir de X. Dominante (≥) sobre DST.

- O **gap LST − DST** mede a **informação ausente em X** para a
  tarefa. Se positivo, indica que mais dados ou melhores features
  podem fechar parte do gap.

### 2.2 Plano accuracy × unfairness em 3 regiões

1. **Abaixo do DST**: possível com algoritmos sobre os dados atuais.
2. **Entre DST e LST**: possível **com dados extras**.
3. **Acima do LST**: impossível para qualquer algoritmo (limite
   informação-teórico).

### 2.3 U-FaTE — método numérico

- Composição encoder + fair-classifier.
- Mede **dependência estatística** entre Z e S (não só correlação
  linear — usa medida universal que captura todas as dependências
  não-lineares).
- **Solver em closed-form** para o último layer (inspirado em K-TOpt
  de Sadeghi et al. 2022).
- Sweep sobre λ ∈ [0, 1) gera a curva trade-off inteira.
- Generaliza para 3 noções de fairness:
  - **DPV** — Demographic Parity Violation (independência Ŷ ⊥ S).
  - **EOD** — Equalized Opportunity Difference (separação condicional
    em Y=1).
  - **EOOD** — Equality of Odds Difference (separação para todos y).

### 2.4 Baselines comparados

- **ARL** (Adversarial Representation Learning) — iterativo.
- **FairHSIC** — não-iterativo, dependência completa.
- **OptNet-ARL** — não-iterativo.
- **K-TOpt** — closed-form, antecessor direto.
- **SARL, LEACE, MaxEnt-ARL, FairerCLIP** — citados na related work.

## 3. Datasets e setup experimental

- **CelebA**: tarefa = predizer "high cheekbones"; S = gender & age.
- **FairFace**: tarefa = predizer **sex (binário)**; S = race.
- **Folktable**: tabular, S = race.

**Avaliação em larga escala:**

- **100+ modelos zero-shot** do OpenCLIP (DataComp, LAION2B,
  MetaCLIP, CommonPool, OpenAI WIT, COCO, LAION400M, CC12M, YFCC).
- **900+ modelos supervisionados** do timm (PyTorch Image Models),
  com pre-training em Merged38M, ImageNet22K, Merged30M, OpenAI WIT,
  ImageNet, LAION, etc.

## 4. Métricas reportadas

- **Accuracy** (utility).
- **EOD, EOOD, DPV** (unfairness — 3 noções).
- **DistDST** e **DistLST** — distância euclidiana do ponto (accuracy,
  unfairness) do método até a curva estimada DST/LST.

## 5. Resultados principais (valores numéricos)

### 5.1 Tabela 1 — FairFace (target: sex; sensitive: race)

| Método | Accuracy | Unfairness | DistDST | DistLST |
|---|---|---|---|---|
| ARL | 93.39 | 1.34 (EOD) | 0.448 | 0.559 |
| FairHSIC | 91.02 | 1.33 (EOD) | 0.445 | 0.557 |
| OptNet-ARL | 92.94 | 1.70 (EOD) | 0.598 | 0.709 |
| **U-FaTE-DST** | **96.17** | **0.263** | — | 0.133 |
| **U-FaTE-LST** | **100.0** | **0.0** | — | — |

**Achado-bandeira para sex em FairFace**: *"On FairFace, observe that
trade-offs do not exist since, on this task, it is possible to mitigate
unfairness without sacrificing accuracy."* Isto significa que, **para
predizer sex em FairFace**, não há tensão acurácia ↔ fairness:
existe representação que satisfaz ambos. O trade-off é **task-dependent**.

### 5.2 Para EOD/EOOD: LST ≈ 100% accuracy

Como EOD e EOOD condicionam em Y, um classificador 100% acurato
(Ŷ = Y) tem EOD = EOOD = 0 automaticamente. Logo o LST é plano em
acc=100. Conclusão dos autores: **EOD/EOOD são métricas mais
pragmáticas que DPV**, porque não forçam sacrifício de acurácia para
fairness perfeito (independente confirmação de Hardt et al. 2016).

### 5.3 CelebA — trade-off real

Para "high cheekbones" em CelebA, o gap LST-DST é ~20% accuracy em
baixa fairness, ~40% em alta fairness para EOD/EOOD. Mostra trade-off
substantivo, ao contrário do FairFace.

### 5.4 Modelos pré-treinados pré-existentes

- A maioria dos modelos zero-shot e supervisionados está **longe** do
  DST estimado — i.e., há espaço de melhoria substancial **mesmo com
  os mesmos dados**.
- **Alguns modelos supervisionados em FairFace surpassam o DST** e
  entram na região "possível com dados extras". Modelos
  pré-treinados em datasets grandes (OpenAI WIT, ImageNet22K,
  LAION2B) atingem isso, sugerindo que **escala de dados pré-treino
  importa para fairness**.

## 6. Limitações declaradas pelos autores

- U-FaTE estima trade-offs **a partir de amostra finita**; estimativas
  têm variância. Apenas a média (linha) e variância (sombra) são
  reportadas, sem CIs formais.
- A medida de dependência usada é "universal" mas a escolha exata é
  hiperparametrizável; resultados podem mudar com outras escolhas.
- Apenas três datasets testados (CelebA, FairFace, Folktable).
- Generalização para fairness individuais (vs grupos) não tratada.

## 7. Limitações que identifiquei (leitura crítica)

- **A tarefa em FairFace é predizer sex, não race em 7 classes.** O
  paper trata FairFace como benchmark de gender-prediction com race
  como atributo sensível. **Não testa classificação racial em 7
  classes**, que é o foco da nossa dissertação. O achado "no
  trade-off on FairFace" é específico para sex-prediction.
- **Definições de fairness em raça binária:** o paper formula DP/EOD
  para Ŷ binário e S binário. Para raça em 7 classes (nosso caso),
  as extensões existem (group-wise EOD, demographic parity macro)
  mas o paper não as discute. Reaproveitamento direto exige
  adaptação.
- **U-FaTE assume access ao S** (atributo sensível) durante o
  treinamento. Em cenários reais, S pode não estar disponível
  (legalmente restrito) — o paper não trata fairness without
  demographics.
- **Avaliação em 900+ modelos supervisionados** é impressionante, mas
  ofusca o fato de que cada ponto é apenas accuracy × fairness em
  um setup específico; não há análise causal de **por que** alguns
  pre-trained datasets produzem modelos mais fair.
- **"Possible with Extra Data" region é ilustrativa, não
  prescritiva:** o paper observa que alguns modelos surpassam DST mas
  não quantifica **quantos dados extras** levariam a quanto
  improvement.

## 8. Relação com nossa pesquisa

**Centralidade teórica:** este é o **esqueleto matemático** que
fundamenta a noção de "trade-off acurácia × disparidade" usada nesta
dissertação. Antes de U-FaTE, falar de trade-off era observação
empírica; depois, há um construto formal (DST/LST) e um método de
estimação.

**Pontos de ancoragem:**

1. **Justificativa para reportar accuracy E disparidade simultaneamente
   em todos os experimentos** — não basta otimizar uma. A
   apresentação tabular padrão de cada experimento da dissertação
   (linha por seed + média + DR) é uma instância prática do plano
   accuracy × fairness.
2. **Limitação metodológica registrada:** nossa pesquisa **não estima
   DST/LST** para FairFace 7-class porque (a) U-FaTE foi testado em
   binário; (b) extensão para 7 classes exige adaptação. Esta é uma
   pendência metodológica importante para trabalhos futuros.
3. **Calibração de expectativa:** se U-FaTE encontra que trade-off
   para **sex em FairFace** é nulo, isso sugere que **raça em
   FairFace** pode também ter trade-off menor que o esperado
   intuitivamente — i.e., nossa pesquisa não deveria assumir trade-off
   forte a priori sem evidência.
4. **Pre-trained representation matters:** o paper mostra que ImageNet22K
   e OpenAI WIT produzem representations mais próximas do DST. Isso
   sustenta nossa escolha de **backbone pré-treinado em ImageNet
   (ConvNeXt-T)** vs treinar from-scratch.
5. **DPV ≠ EOD pragmaticamente:** nosso DR (max/min de F1 por classe)
   é mais próximo de um EOD-like (separação condicional em Y=classe)
   do que de DPV. Conforme o paper, isto é a métrica **mais
   pragmática** das três.

## 9. Pontos para citar / posicionar

- *"Dehdashtian, Sadeghi e Boddeti (2024) formalizam a tensão
  utility–fairness em duas trade-offs intrínsecas aos dados: a
  Data-Space Trade-Off (DST), achievable por algoritmos sobre os
  dados disponíveis, e a Label-Space Trade-Off (LST), achievable por
  um oracle com acesso direto a (Y, S). O gap entre as duas mede a
  informação ausente em X."*
- *"Para o caso de classificação binária de sex sobre o dataset
  FairFace, Dehdashtian et al. (2024, Tabela 1) demonstram
  empiricamente que o trade-off acurácia-fairness é desprezível: o
  estimador U-FaTE-DST atinge 96.17% de acurácia com unfairness
  EOD = 0.263%. Esta observação sugere que, ao menos para algumas
  tarefas faciais com FairFace, mitigação de viés não impõe penalidade
  forte sobre utilidade."*
- *"Embora U-FaTE (Dehdashtian et al., 2024) seja formulado para
  classificação binária e fairness em duas categorias demográficas,
  sua extensão para classificação racial em 7 categorias —
  configuração da presente dissertação — permanece como pendência
  metodológica."*

## 10. Arquivos relacionados

- PDF local: `pdfs/dehdashtian_2024_ufate.pdf` (gitignored).
- Texto extraído: `pdfs/dehdashtian_2024_ufate.txt` (gitignored).
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace),
  [[buolamwini_2018]] (motivação empírica do trade-off),
  [[dominguez_2024]] (DSAP — auditoria de datasets como input para
  estimativa de trade-offs).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S4.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section 7 (Concluding Remarks):

- **Estender U-FaTE para classificação multi-classe** — formulação
  atual é Y, S binários. Para race 7-class, generalização é não-
  trivial. ✅ **Alinhada com Q04 e Q05** — direção crítica.
- **Estimar trade-offs com fairness individual** — paper só cobre
  group fairness. ❌ Direção paralela.
- **Aplicar a tarefas faciais multi-classe complexas** — autores
  citam emoção, atributos múltiplos. ✅ **Alinhada com Q04** —
  race classification é candidato direto.
- **Investigar por que alguns modelos supervisionados ultrapassam
  DST** — região "Possible with Extra Data" precisa caracterização.
  ✅ **Alinhada com Q06** (decomposição do ceiling).
- **Combinar U-FaTE com técnicas algorítmicas explícitas**
  (FSCL, Group DRO) — paper só avalia representations existentes.
  ✅ **Alinhada com Q04**.
- **Aplicar a FairFace race 7-class** — autores testam sex em
  FairFace mas não race. Extensão direta. ✅ **GAP CENTRAL — Q04**.

## 12. Análise crítica do método

### (a) Rigor formal

- **DST e LST formalmente definidos** como minimização sob restrição
  de dependência estatística — definições matemáticas claras.
- **Solver em closed-form** para o último layer (inspirado em K-TOpt
  de Sadeghi 2022) — derivação reproduzível.
- **Limitação**: a métrica de dependência usada é "universal" mas
  hiperparametrizável; sensibilidade a essa escolha não totalmente
  caracterizada.

### (b) Reprodutibilidade

- ✅ Avaliação massiva: 100+ modelos zero-shot do OpenCLIP + 900+
  supervisionados do timm — escala única que reforça a generalidade
  dos achados.
- ⚠ Hiperparâmetros do sweep λ ∈ [0, 1) não detalhados em todos os
  experimentos.
- ⚠ Sem multi-seed reportado para cada ponto da curva — variância
  ilustrada como sombra mas sem IC formal.

### (c) Aplicabilidade ao pipeline v3.2

- **Esqueleto teórico para nossa pesquisa**: justifica reportar
  accuracy E disparidade simultaneamente (não basta otimizar uma).
- **Limitação metodológica registrada**: U-FaTE é binário; race
  7-class exigiria adaptação não trivial.
- **Achado "no trade-off em sex/FairFace"** sugere que mitigação de
  viés em FairFace pode não impor penalidade forte sobre utility —
  alinha com nossa expectativa para H1.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Duas trade-offs (DST + LST) em vez de uma | ✅ Justificada — gap LST-DST mede informação ausente |
| Closed-form solver (não SGD adversarial) | ✅ Justificada — estabilidade superior |
| Avaliação em 1000+ modelos pré-treinados | ✅ Justificada — generalidade do framework |
| Apenas Y, S binários | ⚠ Limitação reconhecida, não justificada como escolha |
| Single seed por ponto | ❌ Assumida — sem justificativa de custo |

### (e) Conexão com R5/R6

- [[hardt_2016]]: U-FaTE generaliza para DPV/EOD/EOOD — três noções
  conectam-se diretamente ao framework de Hardt.
- [[zemel_2013]]: U-FaTE é **descendente** do Fair Representation
  Learning — formaliza o trade-off que Zemel apenas observou.
- [[madras_2018]] LAFTR: U-FaTE substitui adversarial training por
  closed-form solver — estabilidade superior.
- [[kleinberg_2017]]: U-FaTE quantifica o trade-off que Kleinberg
  prova matematicamente — instância empírica do teorema.
- **Implicação para v3.2**: nossa pesquisa **não estima DST/LST** para
  FairFace 7-class, mas o framework U-FaTE é referência conceitual
  para "qual é o limite achievable?".
