# Revisão Sistemática do Estado da Arte (SOTA) — Fairness em Reconhecimento Facial

**Fase 0 do [PLANO_TRABALHO.md](PLANO_TRABALHO.md) — gate G0 (destrava o objetivo definitivo)**
**Data de início:** 2026-05-14
**Status:** Primeira passada concluída (web-based). Passada completa pendente do corpus de ~900 referências (Diretriz 8).
**Autor:** Marcello Ozzetti

> Este documento segue uma metodologia de **revisão sistemática de
> literatura** (inspirada em PRISMA 2020 e nas diretrizes de Kitchenham
> para Systematic Literature Review em engenharia de software/IA). A
> execução é **iterativa**: a primeira passada (aqui documentada) usa
> busca web estruturada; a passada definitiva incorpora a pesquisa
> semântica sobre as ~900 referências do FairFace (Diretriz 8 do
> orientador) quando o corpus estiver disponível. **O protocolo é fixo;
> a execução é incremental.**

---

## 1. Protocolo da revisão (metodologia)

### 1.1 Objetivo da revisão

Mapear o estado da arte para **travar o objetivo da dissertação** como um
*delta publicável* — i.e., identificar o que já existe na literatura
2021–2026 e onde está a lacuna que o trabalho preenche.

### 1.2 Perguntas de pesquisa (Research Questions)

Derivadas diretamente das 4 orientações do aluno:

| RQ | Pergunta | Orientação de origem |
|---|---|---|
| **RQ1** | Qual o SOTA de *parâmetros e desempenho* em publicações que usam/citam o FairFace para classificação de atributo (raça)? | "SOTA dos parâmetros sobre publicações que citam FairFace" |
| **RQ2** | Qual o SOTA de *arquitetura de backbone* atual para reconhecimento/atributo facial? | "SOTA para a arquitetura backbone atual" |
| **RQ3** | Qual o SOTA de *benchmark de fairness* (independente de arquitetura)? | "SOTA benchmark atual para fairness" |
| **RQ4** | Quais *lacunas* a literatura identifica que possam fundamentar o objetivo de um mestrado? | "Demais análises p/ elaborar objetivo" |

### 1.3 Fontes consultadas (search sources)

- **Bases primárias:** arXiv, CVF Open Access (CVPR/ICCV/ECCV/WACV),
  IEEE Xplore, ACM DL, Springer Link.
- **Venues-alvo de fairness:** WACV Workshops (DVPBA, Fair CV), IJCB,
  ACM FAccT, IEEE TBIOM, TPAMI, NIST FRVT/FRTE reports.
- **Agregadores:** Papers With Code, Emergent Mind, Semantic Scholar.
- **Período:** 2019 (FairFace seminal) → 2026 (estado atual), com foco
  em **2024–2026** para o SOTA recente.

### 1.4 Strings de busca (executadas nesta passada)

1. `FairFace dataset race classification state of the art accuracy 2024 2025 benchmark`
2. `FairFace 7 race classification accuracy ResNet ViT benchmark in-domain test set 2024`
3. `face recognition demographic fairness benchmark RFW BUPT-Balancedface SOTA 2024 2025 inequity rate`
4. `multi-objective hyperparameter optimization fairness accuracy Pareto neural network Optuna 2024 2025`
5. `state of the art backbone face recognition 2024 2025 ViT ConvNeXt AdaFace transformer ArcFace benchmark`

### 1.5 Critérios de inclusão / exclusão

**Inclusão:** (i) trabalho peer-reviewed ou preprint arXiv com ≥ 1
citação ou de venue reconhecida; (ii) trata de reconhecimento facial OU
classificação de atributo facial; (iii) reporta métrica de desempenho OU
de fairness; (iv) 2019–2026.

**Exclusão:** (i) sem avaliação demográfica nem de atributo; (ii)
domínio não-facial; (iii) duplicata/versão anterior do mesmo trabalho
(mantém-se a versão mais recente).

### 1.6 Processo de triagem (PRISMA-style, primeira passada)

```
Identificação:  ~45 resultados retornados pelas 5 strings
      ↓ (remoção de duplicatas e fora de escopo)
Triagem:        ~20 títulos/abstracts relevantes
      ↓ (leitura de abstract + tabelas-chave)
Elegibilidade:  ~9 trabalhos núcleo extraídos em detalhe
      ↓
Incluídos:      6 trabalhos-âncora (ver §6 Referências)
```

> **Limitação declarada:** esta primeira passada é *web-based* e portanto
> não-exaustiva. A passada definitiva (Diretriz 8) roda
> `scripts/semantic_search_corpus.py` sobre as ~900 referências do
> FairFace, com embeddings semânticos + clustering por tópico, e
> atualiza as sínteses abaixo. As conclusões aqui são **preliminares mas
> suficientes para o gate G0**.

---

## 2. RQ1 — SOTA de parâmetros em publicações FairFace

### 2.1 Baseline seminal (Kärkkäinen & Joo, WACV 2021)

| Parâmetro | Valor reportado |
|---|---|
| Backbone | **ResNet-34** |
| Raça (4 grupos, média cross-dataset) | **81.5%** |
| Raça (7 grupos, in-domain) | Não reportado como número único (Tabela 6 quebra por subset) |
| Gênero (média) | **95.7%** |
| Idade (média) | **53.6%** |
| Raça vs UTKFace / LFWA+ | 81.5% vs 67.4% / 68.4% |

**Achado-chave para o posicionamento:** o paper seminal usa **ResNet-34**.
O nosso pipeline (MBA → mestrado) usa **ResNet-50** — já é um upgrade de
backbone sobre a referência canônica.

### 2.2 Natureza do "SOTA" em FairFace (nuance metodológica importante)

O FairFace **não é um leaderboard de classificação com um número SOTA
único** como ImageNet ou IJB-C. É usado predominantemente como:

1. **Conjunto de treino** para modelos de atributo demográfico que depois
   são usados como *auditores* de outros sistemas;
2. **Conjunto de auditoria** para medir fairness de modelos terceiros.

Implicação: **comparar o nosso F1≈0.69 em 7-race contra "o SOTA do
FairFace" é uma comparação mal-posta** — não existe esse SOTA único.
O posicionamento correto é: nosso baseline ResNet-50+Linear vs nossas
intervenções (limpeza + topologia), com FairFace como ambiente
controlado de auditoria. Isto **reforça** o desenho experimental atual
(R1/R2/decomposição) em vez de enfraquecê-lo.

---

## 3. RQ2 — SOTA de backbone (reconhecimento facial)

### 3.1 Panorama 2024–2026

| Família | Exemplos SOTA | Benchmark de referência |
|---|---|---|
| ViT (Vision Transformer) | **LVFace** (ViT-S/B/L, Glint360K), TransFace, KP-RPE (ViT-B) | IJB-C ~96.5–97% (ViT-S/B); TAR ~98% @0.01% FAR |
| CNN profunda | ArcFace R50/R100, ResNet-200, PFC (R200) | IJB-C ~98% TAR @0.01% FAR |
| ConvNeXt | ConvNeXt-T + ECA attention | 99.76% em masked-face real-world |
| Loss adaptativa | **AdaFace**, MagFace, TopoFR | Melhor *robust accuracy* sob variação de aparência (YTF) |

### 3.2 Nuance crítica: verificação ≠ classificação de atributo

**Quase todo o SOTA de backbone acima é para face *verification* (1:1
matching, IJB-C/IJB-B/LFW), não classificação de atributo de raça.** São
tarefas diferentes:

- *Verification:* "estas duas fotos são a mesma pessoa?" — métrica TAR@FAR.
- *Attribute classification (nosso caso):* "qual a raça desta pessoa?" —
  métrica F1 macro / acurácia.

Portanto, o "SOTA de backbone" relevante para a tese é: **ViT-B/16 e
ConvNeXt-T como alternativas modernas à ResNet-50** para o eixo
experimental de backbones (Semanas 9–10 do PLANO). LVFace/KP-RPE/AdaFace
informam o eixo de *losses* (Semanas 7–8), não o de classificação direta.

---

## 4. RQ3 — SOTA de benchmark de fairness (independente de arquitetura)

Fonte-âncora: **"Review of Demographic Fairness in Face Recognition"**
(arXiv 2502.02309v3, fev/2025) — survey recente e abrangente.

### 4.1 Datasets-padrão de fairness

| Dataset | Escala | Tarefa | Uso |
|---|---|---|---|
| RFW (Racial Faces in-the-Wild) | 24k pares, 4 raças | Verificação | Benchmark racial canônico |
| BUPT-Balancedface | 1.3M imgs, 7k/raça | Treino balanceado | Treino fair |
| BFW (Balanced Faces in the Wild) | 20k imgs | Verificação | Gênero+etnia |
| DemogPairs | 10.8k imgs, 58.3M pares | Verificação | Folds demográficos |
| MORPH-II/III | 55k–127k | Atributo | Raça/gênero/idade controlado |
| **FairFace** | **108k** | **Atributo** | **Treino auditor + auditoria** |

### 4.2 Métricas-padrão de fairness

O survey cataloga (Tabela IV): **Inequity Rate (IR)**, **FDR** (Fairness
Discrepancy Rate), **GARBE** (Gini-based), **d-prime**, **SFI/CFI**,
**DFI** (KL-divergence), **CEI** (Comprehensive Equity Index), **MAPE**.

→ **O projeto já implementa IR, max-min disparity, Gini, FDR, CEI** — ou
seja, estamos **metricamente alinhados com o SOTA**. Lacuna nossa: não
reportamos d-prime nem DFI (fácil de adicionar se a banca pedir).

### 4.3 Consenso central da literatura (validação da premissa da tese)

> "Mere balance in identities or number of images is insufficient to
> address disparities" — balancear dados sozinho **não elimina** o gap
> demográfico (survey 2025; NIST FRVT).

Isto **valida empiricamente a premissa herdada do MBA** (undersampling
não basta) — não é uma hipótese isolada nossa, é consenso de survey 2025.

---

## 5. RQ4 — Lacunas identificadas → fundamentação do objetivo

O survey 2025 (arXiv 2502.02309v3) enumera explicitamente lacunas
abertas. Cruzando com o que o nosso trabalho já faz:

| Lacuna identificada na literatura | O nosso trabalho endereça? |
|---|---|
| "Need for **careful experimental control to isolate algorithmic vs data factors**" | ✅ **Diretamente** — desenho R1/R2 + decomposição (limpeza vs topologia) |
| "**Causal attribution** remains complex, requires rigorous experimental isolation" | ✅ Mesma decomposição controlada (mesma seed, mesmo split, 1 fator por vez) |
| "**Metric aggregation**: combining indicators into a single scalar introduces limitations" | ✅ **Abordagem Pareto multi-objetivo evita escalarização** — exatamente a lacuna |
| "Score-level fairness: fewer attempts" | ❌ Fora do escopo (trabalho futuro) |
| "1:N identification under-explored" | ❌ Fora do escopo (fazemos classificação) |
| "Intersectionality (raça+gênero+idade) poorly understood" | ❌ Fora do escopo desta fase (trabalho futuro) |

### 5.0 Check de novidade — análise de sobreposição (passada de citações ao FairFace)

Busca direcionada testando se as 3 contribuições candidatas já existem.
**Dois trabalhos próximos foram encontrados e analisados em detalhe:**

#### Trabalho A — "Rethinking Bias Mitigation: Fair Architectures Make for Fair Face Recognition" (NeurIPS 2023, arXiv:2210.09943)

| Dimensão | Trabalho A | Nosso trabalho | Sobreposição? |
|---|---|---|---|
| Tarefa | Classificação de atributo facial | Classificação de raça (7 classes) | Parcial |
| Dataset | **CelebA + VGGFace2** | **FairFace** | Não |
| O que busca | **NAS de arquitetura completa** | **HPO só do head** (backbone fixo) | Não |
| Multi-objetivo | Sim (acc × fairness) | Sim (F1 × IR) | **Sim** |
| Decomposição dataset vs arquitetura | **Não** | **Sim** | Não |
| Critério best-epoch Pareto-aware | **Não** | **Sim** | Não |

**Veredito:** o conceito "busca multi-objetivo de arquitetura para
fairness em classificação de atributo facial" **já existe** (NeurIPS
2023). Isto **derruba o uso de HPO multi-objetivo como contribuição
principal isolada.** Diferenciais reais que permanecem: dataset (FairFace
raça vs CelebA/VGGFace2), escopo (head-only com backbone fixo —
mais controlado/interpretável e relevante para deploy, pois não
re-treina backbone), e os dois diferenciais abaixo que A **não tem**.

#### Trabalho B — "FineFACE: Fair Facial Attribute Classification Leveraging Fine-grained Features" (2024, arXiv:2408.16881)

| Dimensão | Trabalho B | Nosso trabalho |
|---|---|---|
| Dataset | FairFace + CelebA + UTKFace + LFWA+ | FairFace |
| Backbone | ResNet50 | ResNet50 |
| Método | Multi-expert por camada (fine-grained) | HPO de topologia de head |
| HPO / Pareto | **Não** (hparams fixos) | Sim |
| Limpeza de dataset | **Não** | Sim (multi-face) |
| Variação de topologia de head | **Não** | Sim |

**Veredito:** paradigma de mitigação **ortogonal** ao nosso. Sem
sobreposição metodológica. Útil como **referência de SOTA de fairness
improvement** (reporta "67%–83.6% de melhoria de fairness sobre SOTA"
em classificação de atributo) — vira ponto de comparação.

#### Conclusão do check de novidade

- ❌ **"HPO multi-objetivo para fairness facial" NÃO é novidade** —
  Trabalho A (NeurIPS 2023) já fez algo equivalente em conceito.
- ✅ **Diferenciais defensáveis (nenhum dos dois trabalhos tem):**
  1. **Decomposição experimental controlada** dataset-cleaning vs
     head-topology (Trabalho A faz NAS completo, não isola fatores;
     survey 2025 lista isso como lacuna aberta).
  2. **Critério Pareto-aware best-epoch** em HPO multi-objetivo de
     fairness (não encontrado em nenhum trabalho — micro-contribuição
     metodológica).
  3. **Limpeza por integridade de rotulação (multi-face MTCNN) no
     FairFace** + seu efeito **recipe-dependent** (CE vs ArcFace).
  4. **Head-only com backbone congelado** — desenho mais controlado e
     interpretável que NAS completo (Trabalho A), e deployment-relevant.

> **Implicação direta para o objetivo:** o headline "usamos HPO
> multi-objetivo para fairness" **não sustenta uma tese sozinho** — já
> foi feito. A contribuição **tem que ser ancorada nos diferenciais 1 e
> 2** (decomposição controlada + critério Pareto-aware), com o HPO como
> veículo, não como novidade. Isto reforça (não enfraquece) a §5.2.

> **Limitação do check:** passada web-based, não-exaustiva. A passada
> definitiva (semantic search sobre ~900 refs + traversal de ACM
> FAccT/AutoML/IJCB) deve confirmar especificamente se o **critério
> Pareto-aware best-epoch** é mesmo inédito — é o ativo de maior valor e
> precisa de verificação rigorosa antes de submissão de paper.

#### Cross-check: auditoria semântica (2026-05-15) — CONFIRMA §5.0

A passada definitiva semântica foi executada (corpus de 555 docs: 76
referências do FairFace + 479 papers que o citam, via OpenAlex;
`scripts/semantic_search_corpus.py`). Resultado completo em
[literature_semantic_audit.md](literature_semantic_audit.md) §5
(interpretação humana). Veredito consolidado:

| Contribuição | Q (sim. máx.) | Veredito |
|---|---|---|
| HPO multi-obj. de arquitetura p/ fairness | Q11=0.873 | ❌ não é novidade (FairGRAPE 2022, FineFACE 2024, NeurIPS 2023) |
| **Critério Pareto-aware best-epoch** | Q12=0.795 | ✅ **novidade mais forte — 0 método igual em 555 docs** |
| **Decomposição controlada cleaning×topologia** | Q13=0.870 | ✅ defensável (adjacente: DSAP 2024; sem overlap de método) |
| Efeito recipe-dependent da limpeza | Q14=0.834 | ✅ baixo risco (adjacente: "Fairness is in the Details" 2025) |

A auditoria semântica **confirma independentemente** o §5.0: o conceito
"busca multi-objetivo de arquitetura justa" já existe; o **delta real é
o critério Pareto-aware (Q12) + a decomposição controlada (Q13)**.

### 5.1 Posicionamento do delta publicável

Três candidatos a delta, ordenados por probabilidade de novidade
(revisados após o check de novidade §5.0):

1. **Decomposição empírica controlada da contribuição
   dataset-cleaning vs head-topology para disparidade demográfica** —
   responde diretamente à lacuna "isolate algorithmic vs data factors"
   (survey 2025). O Trabalho A (NeurIPS 2023) faz NAS completo e **não
   isola fatores** — então essa decomposição controlada é defensável
   como novidade. **Promovido a candidato principal** após §5.0.

2. **Critério Pareto-aware best-epoch para HPO multi-objetivo de
   fairness** — não documentado em A nem B nem no survey 2025.
   Micro-contribuição metodológica. Alinha com a lacuna "metric
   aggregation/scalarization". **Maior valor potencial, mas precisa de
   verificação rigorosa na passada definitiva** (pode existir em venue
   de AutoML não coberto por esta passada).

3. **Efeito recipe-dependent da limpeza por multi-face em margin-based
   vs softmax-based losses** — achado empírico específico (R2: ArcFace
   +5.8pp F1 vs CE +0.3pp). Não encontrado em A nem B. Novidade de
   menor escopo mas sólida e de baixo risco.

4. ~~"HPO multi-objetivo de arquitetura para fairness facial"~~ —
   **descartado como novidade** após §5.0 (Trabalho A, NeurIPS 2023, já
   cobre o conceito). Permanece como **veículo metodológico**, não como
   contribuição.

→ **Recomendação para o gate G0:** o objetivo definitivo deve ancorar no
**candidato 1 + candidato 2 combinados** — "metodologia de busca
Pareto-aware multi-objetivo que, aplicada a uma decomposição controlada
dataset/arquitetura, isola e quantifica as contribuições para
disparidade demográfica em classificação facial". Os candidatos 3 e os
eixos contrastivo/loss/backbone (PLANO Semanas 5–10) **expandem a
validade externa** do achado.

### 5.2 Objetivo definitivo proposto (para validar com orientador no gate G0)

> "Propor e validar uma metodologia de otimização multi-objetivo
> Pareto-aware para mitigação de viés demográfico em classificação
> facial, aplicando-a a uma decomposição experimental controlada que
> isola e quantifica as contribuições relativas de (i) integridade de
> rotulação do dataset e (ii) topologia do classificador, e
> estabelecendo a generalidade do achado sobre múltiplas famílias de
> loss, paradigmas de aprendizado (contrastivo) e backbones."

Diferença vs objetivo provisório do PLANO: **a metodologia
Pareto-aware vira o núcleo da contribuição** (não apenas um meio),
porque é o que tem maior probabilidade de ser delta publicável segundo
esta revisão.

### 5.3 GATE G0 RESOLVIDO (2026-05-15) — objetivo definitivo travado

Decisão tomada: **Linha A — Atribuição Causal de Viés**. O objetivo §5.2
foi refinado para o framing de *atribuição* (não *mitigação*),
explicitamente diferenciado do SOTA saturado de "método X melhora
métrica Y". Texto definitivo travado em
[PLANO_TRABALHO.md](PLANO_TRABALHO.md) §1 ("Objetivo DEFINITIVO").
Resumo do delta defendido na qualificação:

> Não responder *se* a IA pode ser mais justa (saturado), mas **onde
> intervir** — quantificando a contribuição marginal causal de cada
> fator (dataset, topologia, loss, contrastivo, backbone) via
> decomposição controlada + Pareto-aware. Linha B (critério
> Pareto-aware isolado) e Linha C (viés de cena correlacionado a raça)
> tornam-se papers/capítulos derivados.

---

## 6. Referências-âncora desta passada

1. Kärkkäinen & Joo. *FairFace: Face Attribute Dataset for Balanced
   Race, Gender, and Age.* WACV 2021. arXiv:1908.04913.
2. *Review of Demographic Fairness in Face Recognition.* arXiv:2502.02309v3,
   2025 — survey âncora para RQ3/RQ4.
3. Kotwal & Marcel. *Mitigating Demographic Bias in Face Recognition via
   Regularized Score Calibration.* WACVW 2024 — baseline de score-level.
4. *Fairer Analysis and Demographically Balanced Face Generation for
   Fairer Face Verification.* arXiv:2412.03349v2.
5. LVFace. *Progressive Cluster Optimization for Large Vision Models in
   Face Recognition.* arXiv:2501.13420, 2025 — SOTA backbone ViT.
6. *Multi-Objective Hyperparameter Optimization in ML — An Overview.*
   ACM TELO, dl.acm.org/doi/10.1145/3610536 — fundamentação metodológica.

---

## 7. Próximos passos da revisão (passada definitiva)

1. **Diretriz 8:** rodar `scripts/semantic_search_corpus.py` sobre as
   ~900 referências do FairFace (quando o corpus chegar) — embeddings +
   clustering por tópico, extração estruturada.
2. Preencher tabela de extração de dados (`docs/sota_extraction.csv`)
   com: referência, backbone, dataset, métrica, valor, tipo de mitigação.
3. Confirmar/refutar o candidato 1 (Pareto-aware best-epoch é mesmo
   inédito?) com busca direcionada em ACM FAccT + AutoML venues.
4. Levar a §5.2 (objetivo definitivo proposto) para o **gate G0 com o
   orientador** (fim da Semana 4 do PLANO).

---

## 8. Conclusão preliminar (suficiente para o gate G0)

- O FairFace **não tem SOTA-número único** — comparação correta é
  intra-trabalho (baseline vs intervenções), o que **valida** o desenho
  R1/R2.
- Backbone SOTA moderno é **ViT/ConvNeXt** (mas para *verificação*); para
  o nosso eixo de classificação, são as alternativas naturais à ResNet-50
  no eixo experimental de backbones.
- Métricas do projeto **já alinhadas com o SOTA** (IR/FDR/Gini/CEI).
- A premissa "balancear não basta" é **consenso de survey 2025**, não
  hipótese isolada.
- **Lacuna mais promissora para delta publicável:** critério
  Pareto-aware multi-objetivo + decomposição experimental controlada —
  ambos explicitamente listados como problemas abertos no survey 2025.

→ **Recomendação:** adotar o objetivo da §5.2 no gate G0, mantendo
contrastivo/loss/backbone como eixos de validade externa (PLANO).
