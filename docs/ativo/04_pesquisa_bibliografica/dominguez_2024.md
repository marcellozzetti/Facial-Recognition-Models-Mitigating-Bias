---
name: dominguez-2024
status_verificacao: VERIFIED
autores: [Iris Dominguez-Catena, Daniel Paternain, Mikel Galar]
ano: 2024
titulo: "DSAP: Analyzing Bias Through Demographic Comparison of Datasets"
venue: "Information Fusion (Elsevier)"
tipo_publicacao: journal
arxiv_id: "2312.14626"
doi: null
url_primario: https://arxiv.org/abs/2312.14626
citacoes_google_scholar: null
citacoes_semantic_scholar: 7
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~85
lente_disrupcao: metodologica
fonte_leitura: PDF integral extraído via pypdf (pdfs/dominguez_2024_dsap.pdf)
---

# DSAP: Demographic Similarity from Auxiliary Profiles (Dominguez-Catena, Paternain & Galar, 2024)

## 1. Resumo do problema atacado

Para avaliar viés demográfico em datasets, a literatura usa medidas
específicas e incomparáveis: **ENS** (Effective Number of Species) para
representational bias, **SEI** (Shannon Evenness Index) para evenness,
**Cramer's V** para stereotypical bias. Cada uma tem range próprio,
interpretação distinta, e **não permite comparar magnitude entre tipos
de viés ou entre datasets**. Além disso, todas exigem **rótulos
demográficos explícitos**, ausentes na maioria dos datasets de
imagem/texto. O paper propõe DSAP, uma metodologia unificada em **2
passos** que (a) infere perfil demográfico via modelo auxiliar (sem
exigir rótulos), e (b) mede similaridade entre perfis usando o
**Renkonen similarity index**, importado da ecologia (β-diversity) e
arqueologia (Brainerd-Robinson similarity).

## 2. Método

### 2.1 Passo 1 — Demographic profile via auxiliary model

- Para cada amostra do dataset, um **modelo auxiliar de classificação
  demográfica** (race classifier, age classifier, gender classifier)
  estima o atributo demográfico.
- A distribuição empírica resultante é o **demographic profile** do
  dataset, decomponível em **eixos** (age, gender, race) e
  **combinação** (interseção de eixos).
- **Vantagem chave:** não exige rótulos demográficos no dataset
  original — funciona em datasets sem anotação de raça/gênero/idade.

### 2.2 Passo 2 — Similaridade Renkonen

Para dois perfis p, q sobre o mesmo eixo demográfico G:

DS(p, q) = Σ_{g ∈ G} min(p_g, q_g)  ∈ [0, 1]

- DS = 1 quando p e q são idênticos.
- DS = 0 quando p e q não compartilham nenhum grupo.
- **Matematicamente equivalente** a Sørensen-Dice e Brainerd-Robinson
  (escalado a [0,1]).
- Inspirado em ecologia (β-diversity) e arqueologia (cronologia de
  sítios via composição de artefatos).

### 2.3 Três aplicações de DSAP

1. **Pairwise dataset comparison (DS):** compara dois datasets
   diretamente.
2. **Bias measurement** (comparação contra dataset ideal):
   - **DSR** (Demographic Similarity Representational) — compara
     contra balanced ideal → mede representational bias.
   - **DSE** (Demographic Similarity Evenness) — variante focada em
     uniformidade.
   - **DSS** (Demographic Similarity Stereotypical) — compara contra
     ideal sem correlação atributo-classe → mede stereotypical bias.
3. **Demographic dataset shift:** compara train vs test/deployment.

### 2.4 Vantagens declaradas

- **Range unificado [0, 1]** entre todos os tipos de viés (vs ENS
  com range variável, Cramer's V em [0, 1] mas com semântica
  diferente).
- **Comparabilidade entre datasets** independentemente do tamanho.
- **Suporte a target populations não-balanceadas** (e.g., age
  distribution mundial 2021) — primeiro framework que aceita ideal
  ≠ balanced uniform.

## 3. Datasets e setup experimental

- **20 datasets de Facial Expression Recognition (FER):**
  - Laboratory: ADFES, Oulu-CASIA, GEMEP, MUG, CK, CK+, KDEF,
    WSEFEP, iSAFE, JAFFE, LIRIS-CSE.
  - In-the-Wild Internet (ITW-I): NHFIER, AFFECTNET, FER2013,
    FER+, RAF-DB, EXPW, MMAFEDB.
  - In-the-Wild Movie (ITW-M): SFEW, CAER-S.
- **Tarefa:** Facial Expression Recognition (6 emoções básicas de
  Ekman).
- **Modelo auxiliar:** classificadores externos para age, gender,
  race (não especificado no excerto qual exatamente; código
  github.com/irisdominguez/DSAP).

## 4. Métricas reportadas

- **DS** (Demographic Similarity) — pairwise comparison.
- **DSR, DSE, DSS** — bias measurement variants.
- **DSE/DSR corrected** — para target population não-uniforme (e.g.,
  age corrected pela distribuição mundial 2021).
- Comparações **R²** contra métricas clássicas (ENS, SEI, Cramer's V).

## 5. Resultados principais (valores numéricos)

### 5.1 Correlação com métricas clássicas

| DSAP measure | Métrica clássica | R² (combinação de eixos) |
|---|---|---|
| **DSR** vs **ENS** (representational) | | 0.93–0.98 (age: 0.97; gender: 0.93; race: 0.98; combo: 0.97) |
| **DSE** vs **SEI** (evenness) | | 0.82–0.94 (age: 0.82; gender: 0.94; race: 0.88; combo: 0.86) |
| **DSS** vs **Cramer's V** (stereotypical) | | -0.80 a -0.97 (negativo porque V mede viés direto, DSS similaridade ao ideal) |

**Conclusão dos autores:** DSAP captura o mesmo sinal das métricas
clássicas em alta fidelidade (R² > 0.82 em todos os eixos), com **vantagem
de range unificado e interpretabilidade comparável**.

### 5.2 Clustering em FER (eixo combinado age × gender × race)

DSAP identificou **10 clusters distintos** entre os 20 datasets de FER:

- Cluster D agrupa **datasets ITW-I** (AFFECTNET, FER2013, FER+,
  RAF-DB, EXPW, MMAFEDB, NHFIER) com **alta similaridade entre si**
  (~0.84-0.95) — convergência fenotípica de "internet faces".
- Clusters A, B, C agrupam datasets com **populações únicas não-
  representadas em outros** (e.g., iSAFE só com indianos, JAFFE só
  com japoneses, Oulu-CASIA com mix branco+east asian).

### 5.3 Stereotypical < Representational em FER

Achado quantitativo do paper: **DSS global médio = 0.91 ± 0.076**
(viés stereotypical baixo); **DSR global médio = 0.46 ± 0.3** (viés
representational alto). Confirma intuição de literatura prévia
(Dominguez-Catena et al. 2022) de que **representação é o problema
maior** em datasets FER, não correlação atributo-classe.

## 6. Limitações declaradas pelos autores

- **Dependência do modelo auxiliar:** se o classificador de raça/idade/
  gênero usado para profiling tem viés, o profile herda esse viés.
  O paper reconhece esse risco mas não quantifica.
- **Tarefa específica:** experimentos limitados a FER. Generalização
  para outras tarefas é deixada como trabalho futuro.
- **Apresentação/visualização:** não dispõe de demographic
  relatedness scores (ecologia avançada usa eles para β-diversity
  filogenética; em demographic isso "is nearly impossible").
- **Stereotypical bias** medido apenas sob assumption de classes
  discretas; tarefas com target contínuo não cobertas.

## 7. Limitações que identifiquei (leitura crítica)

- **Circularidade do auxiliary model:** o paper usa modelos externos
  para estimar atributos demográficos. Mas estes modelos foram
  treinados em **datasets já potencialmente viesados** (e.g.,
  FairFace, RFW). O perfil resultante é uma **projeção do viés do
  modelo auxiliar sobre o dataset target**, não o viés "verdadeiro".
  O paper trata isso como aproximação aceitável, mas a validação
  rigorosa exigiria ground truth manual em subset.
- **Renkonen similarity é insensível a grupos ausentes:** se dataset
  A tem composição (50% White, 50% Black, 0% Asian) e dataset B tem
  (50% White, 50% Asian, 0% Black), DS(A,B) = 0.5 mesmo eles sendo
  conceitualmente muito diferentes. O paper menciona isso mas não
  oferece variante que penalize ausência.
- **Tarefa testada (FER) não é race classification.** A aplicabilidade
  para classification de raça (nosso caso) seria interessante mas
  exigiria adaptação metodológica.
- **Auxiliary models reportados como FairFace classifier** (provável,
  inferindo do GitHub). Significa que **avaliar bias do FairFace
  usando classifier treinado no FairFace** é circular. Útil para
  comparar entre datasets, não para auditar o próprio FairFace.
- **Não testa interações com dataset shift** sob shift adversarial —
  só observa shift natural entre train/test do mesmo dataset.

## 8. Relação com nossa pesquisa

**Centralidade:** DSAP é uma **ferramenta de auditoria** — não uma
técnica de mitigação. Para a dissertação, é útil em três frentes:

1. **Auditoria do nosso pipeline de dataset:** poderíamos aplicar
   DSAP comparando o split train/val/test do FairFace contra
   distribuição mundial real (via correção de target population),
   para reportar **shift entre nosso experimento e produção
   real-world**. Esta é uma adição importante ao posicionamento.
2. **Linguagem unificada para discutir viés:** as 3 categorias
   (representational, evenness, stereotypical) e a notação [0,1] são
   convenientes para apresentar achados experimentais. Nossa razão
   de disparidade é um stereotypical-like indicator (correlação
   raça↔erro), mas a categoria representational descreve **viés do
   dataset**, não **viés do modelo treinado**.
3. **Análise comparativa do FairFace contra distribuição mundial:**
   é defensível argumentar que mesmo o FairFace balanceado por raça
   não é "representativo" do mundo (distribuição racial não-uniforme
   global). DSAP fornece o mecanismo para essa comparação.

**Limitação para reuso direto:**

- DSAP precisa de **modelo auxiliar de classificação racial**. O
  candidato natural é... o próprio modelo que estamos treinando no
  FairFace. Circularidade.
- Para a dissertação, faz mais sentido **citar DSAP como referência
  metodológica** do que aplicá-lo empiricamente.

## 9. Pontos para citar / posicionar

- *"Dominguez-Catena, Paternain e Galar (2024) propõem o framework
  DSAP (Demographic Similarity from Auxiliary Profiles), publicado
  no periódico Information Fusion, que unifica em uma única métrica
  com range [0,1] três tipos previamente incompatíveis de viés em
  datasets: representational, evenness e stereotypical."*
- *"O DSAP permite comparar a composição demográfica de um dataset
  com uma população-alvo arbitrária (não necessariamente balanceada),
  abordando uma limitação dos frameworks anteriores que assumiam
  ideal uniforme (Dominguez-Catena et al., 2024)."*
- *"Em análise sobre 20 datasets de reconhecimento de expressão
  facial, Dominguez-Catena et al. (2024) reportam que o viés
  representational predomina sobre o stereotypical (DSR médio = 0.46
  vs DSS médio = 0.91 em escala [0,1] onde 1 = ideal), confirmando
  que o problema central de datasets faciais está na sub-representação
  de grupos demográficos e não na correlação atributo-classe."*

## 10. Arquivos relacionados

- PDF local: `pdfs/dominguez_2024_dsap.pdf` (gitignored).
- Texto extraído: `pdfs/dominguez_2024_dsap.txt` (gitignored).
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace —
  candidato a auditoria via DSAP), [[grother_2019]] (NIST FRVT —
  auditoria de outro estilo, sobre algoritmos).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S6.
- Código DSAP: https://github.com/irisdominguez/DSAP.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section VI (Conclusions):

- **Aplicar DSAP a outras tarefas além de FER** — paper só testa
  emoção. Race, gender, age classification são candidatos. ✅ **Alinhada
  com Q10** — auditar FairFace via DSAP é exatamente o que
  precisaríamos.
- **Investigar dependência do auxiliary model** — autores reconhecem
  circularidade. ✅ **Alinhada com Q01** (anotação não confiável).
- **Incorporar continuous data** — DSAP atualmente assume classes
  discretas. ✅ **Alinhada com Q09** (Neto 2025).
- **Demographic relatedness scores** — autores dizem "nearly
  impossible" mas reconhecem que ecologia avançada usa relatedness
  filogenético. Inspiração possível: hierarquia étnica do US Census
  (Latinx como sub-grupo de White ou separado). ⚠ Tangencial.
- **Análise temporal** (dataset drift) — DSAP suporta mas não foi
  validado longitudinalmente. ❌ Fora do nosso escopo (single-
  snapshot).
- **Combinar DSAP com bias mitigation techniques** — auditoria sem
  ação é incompleta. ✅ **Alinhada com Q04**.
