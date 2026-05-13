# Proposta de Mestrado em IA — Evolução do Trabalho de MBA

**Tema:** Por que balancear a distribuição de classes não basta para reconhecimento facial equitativo — e o que de fato ajuda. Auditoria empírica e intervenções em loss e dados sintéticos sobre o pipeline LResNet50E_IR/FairFace.

**Autor:** Marcello Ozzetti

**Programa-alvo:** Mestrado em Ciência da Computação — Unifesp / ICT (Instituto de Ciência e Tecnologia, São José dos Campos).

**Trabalho de origem:** Dissertação de MBA em IA / USP — *Facial Recognition Models Mitigating Bias* (out/2024).

**Data desta proposta:** 06 de maio de 2026.

**Última revisão:** 07 de maio de 2026 — recalibração após re-execução dos 11 experimentos

**Janela de planejamento:** 6 meses (mai/2026 → nov/2026).

---

## Sumário

1. [Sumário Executivo](#1-sumário-executivo)
2. [Parte I — Pesquisa Textual Científica (Estado da Arte 2024–2026)](#parte-i--pesquisa-textual-científica-estado-da-arte-20242026)
3. [Parte II — Avaliação do Trabalho de MBA e Lacunas Identificadas](#parte-ii--avaliação-do-trabalho-de-mba-e-lacunas-identificadas)
4. [Parte III — Proposta Evolutiva do Mestrado](#parte-iii--proposta-evolutiva-do-mestrado)
5. [Parte III.5 — Plano de Publicação Científica](#parte-iii5--plano-de-publicação-científica)
6. [Parte IV — Motivadores com Referências Bibliográficas](#parte-iv--motivadores-com-referências-bibliográficas)
7. [Parte V — Plano de Trabalho (6 Meses)](#parte-v--plano-de-trabalho-6-meses)
8. [Parte VI — Riscos, Recursos e Decisões em Aberto](#parte-vi--riscos-recursos-e-decisões-em-aberto)
9. [Referências](#referências)

---

## 1. Sumário Executivo

A dissertação de MBA estabeleceu uma base experimental funcional — pipeline `LResNet50E-IR` sobre **FairFace** com 11 experimentos comparativos. A re-execução completa desses 11 experimentos em maio/2026 (smoke de 5 épocas + rodada limpa de 25 épocas com `EarlyStopping`, 9h38min de wall-clock, 11/11 OK) com o pipeline corrigido revelou **um achado central que define o tema do mestrado**:

> **O undersampling para classes balanceadas não produz reconhecimento facial equitativo.** Mesmo com 10.374 imagens por classe (todas as 7 classes idênticas em tamanho), todas as 5 configurações CE 7-class testadas mantêm **Inequity Rate (F1) entre 1,76 e 1,86** e **gap max-min entre 0,34 e 0,36** entre a melhor classe (Black, F1=0,78–0,84) e a pior (**Latino_Hispanic, F1=0,42–0,47**). O fenômeno é estrutural: independente de optimizer (SGD/AdamW), scheduler (OneCycleLR/CosineAnnealing), épocas (até 40) ou dropout (0,2/0,5) — nenhuma combinação fechou esse gap.

Como bônus metodológico, a rodada limpa também **expôs em escala completa** o bug §1 (`ArcFaceLoss` que silenciosamente caía em `cross_entropy`): com a margem angular agora realmente aplicada, os experimentos de ArcFace caem de 0,58–0,61 reportados pelo MBA para **0,17–0,46 de acurácia, IR=∞** — a margem fixa `m=0,5` colapsa pelo menos uma classe demográfica para F1=0.

Estas evidências **falsificam empiricamente a hipótese intuitiva** que o trabalho de MBA assumia (balancear → equidade) e abrem a pergunta que organiza a tese: **se balancear o dataset não basta, o que basta?**

A proposta de mestrado responde essa pergunta investigando duas frentes documentadas como SOTA na literatura 2024–2026:

1. **Funções de perda quality-adaptive** (AdaFace, MagFace, KP-RPE) — atacam o problema no espaço de loss.
2. **Augmentação por dados sintéticos** (DCFace pré-treinado) das classes piores — ataca o problema no espaço de dados.

Cada frente é investigada isoladamente e em combinação, com auditoria de fairness padronizada (IR/FDR/Gini) sobre os mesmos splits, alinhada a requisitos regulatórios do **EU AI Act** (vigência plena 02/08/2026) e ao **NIST FRTE/FATE**. O **baseline a bater** é o Exp 5 (CE+AdamW+CosineAnnealingWarmRestarts) com acc=0,665, IR=1,76 e F1 Latino_Hispanic=0,47.

**Outputs esperados em 6 meses:**
1. **Submissão de artigo científico** em workshop tier-A (target: WACV Workshop on Fair Computer Vision, IJCB 2027, ou ACM FAccT 2027).
2. **Rascunho de qualificação** com a pergunta empírica e as duas linhas de resposta.
3. **Repositório open-source** como artefato de reprodutibilidade do paper.

---

## Parte I — Pesquisa Textual Científica (Estado da Arte 2024–2026)

### 1.1 Metodologia da pesquisa textual

Pesquisa exploratória em fontes acadêmicas (arXiv, IEEE Xplore, ACM Digital Library, OpenAccess CVF, NIST Publications) e documentos regulatórios (Comissão Europeia), janela 2024 → maio/2026. Termos-chave combinaram os eixos da dissertação (loss function, backbone, dataset, MTCNN) com o tema central (fairness, demographic bias, mitigation). Priorizadas (i) *surveys* recentes em revistas indexadas, (ii) artigos em conferências top-tier (CVPR/ICCV/TPAMI/TBIOM 2024–2025), (iii) documentos normativos vigentes.

### 1.2 Surveys de referência publicados em 2024–2025

| Survey | Veículo | Contribuição |
|---|---|---|
| Kotwal & Marcel (2025) | IEEE TBIOM | Taxonomia em três estágios (pré, in-, pós-processamento); consolida métricas modernas de fairness |
| Atzori et al. (2025) | ACM Computing Surveys | Aprofundamento do eixo racial |
| Kim, Jain & Liu (2025) | arXiv:2505.24247 | Síntese histórica de 50 anos; consolida loss functions, backbones e síntese |

### 1.3 Eixo 1 — Funções de perda: ArcFace deixou de ser o estado da arte

Movimento claro, em 2022–2025, para **funções de perda adaptativas à qualidade da amostra**, que superam ArcFace em benchmarks de fairness:

| Loss | Ano | Inovação principal |
|---|---|---|
| AdaFace (Kim et al., CVPR) | 2022 | Margem adaptativa baseada na qualidade da imagem (norma do feature) |
| MagFace (Meng et al.) | 2021 | Magnitude do feature como proxy de qualidade |
| ElasticFace | 2022 | Margens estocásticas dentro de um intervalo |
| UniFace | 2023 | Unified Cross-Entropy — separação clara entre pares positivos/negativos |
| KP-RPE (ICLR) | 2024 | Integra landmarks faciais nas codificações de posição relativa do ViT |
| CAFace / UC-Face | 2022/2024 | Perdas contrastivas entre qualidades distintas |

### 1.4 Eixo 2 — Backbones: a transição CNN → Vision Transformer

LResNet50E-IR continua competitivo em custo, mas o SOTA migrou para arquiteturas baseadas em ViT:

- **TransFace (ICCV 2023)** e **TransFace++ (TPAMI 2025)** — primeiro uso bem-sucedido de ViT em FR, com *Dominant Patch Amplitude Perturbation* (DPAP) e *Entropy-aware Hard Sample Mining* (EHSM).
- **LVFace (ICCV 2025)** — *Large Vision Models* aplicados a FR via *Progressive Cluster Optimization*.
- **ViT-B + KP-RPE + AdaFace** — combinação que atinge **97,99% no IJB-C**, padrão de comparação atual.

### 1.5 Eixo 3 — Detectores faciais: MTCNN é considerado legado

A literatura de 2024-2025 trata MTCNN como legado. Comparativos modernos:

| Detector | Acurácia | Velocidade | Recomendação 2025 |
|---|---|---|---|
| MTCNN | Média | Lenta | Legado |
| RetinaFace-ResNet50 | Mais alta | Lenta (~3,8 s/face) | Quando precisão é crítica |
| RetinaFace-MobileNet | Boa | Rápida | Equilíbrio |
| YuNet | Boa | Mais rápido (~0,03 s/face) | Tempo real / edge |
| SCRFD (InsightFace) | Boa | Otimizado mobile | Edge/GPU mobile |
| DCFD (2024) | +3,2-3,5% sobre RetinaFace | — | Frontier |

### 1.6 Eixo 4 — Datasets: para além do FairFace

| Dataset | Características | Uso típico |
|---|---|---|
| FairFace | 100k+ imagens, 7 grupos raciais | Treino/teste (já usado no MBA) |
| RFW | 24k pares, 4 grupos raciais | **Benchmark padrão de teste** para viés racial |
| BUPT-Balancedface | 1,3M imagens, 28k identidades, 4 grupos | **Dataset de treino canônico para fairness** |
| BFW | 20k imagens, balanceado gênero × etnia | Avaliação de fairness interseccional |
| CausalFace | 48k pares sintéticos, 6 grupos | Controle causal sobre atributos |
| AI-Face (CVPR 2025) | million-scale, anotado demograficamente | Benchmark de fairness com IA generativa |

### 1.7 Eixo 5 — Dados sintéticos: tendência emergente para mitigação de viés

Avanço mais relevante para o tema do MBA entre 2023-2025. A síntese reduziu o gap para dados reais a ~2-3% em benchmarks:

- **DCFace (CVPR 2023)** — diffusion model com *dual condition* (estilo + identidade)
- **IDiff-Face** — 20k identidades, 1M imagens
- **Vec2Face / Arc2Face / GANDiffFace** — síntese controlada por identidade
- **VariFace (2024)** — geração demograficamente balanceada explicitamente
- **3DMM-Guided Synthesis (CVPR 2025)** — geração diversa em estilos
- **NYU Tandon (2025)** — dataset sintético demograficamente diverso para mitigação de viés

### 1.8 Eixo 6 — Regulação e auditoria: o cenário mudou

#### EU AI Act
- **02/02/2025** — em vigor as **proibições absolutas**: scraping não-direcionado de internet/CCTV para criar bases biométricas; reconhecimento de emoções em escolas/trabalho; categorização biométrica que infira atributos protegidos.
- **02/08/2026** — *high-risk requirements* tornam-se exigíveis. Reconhecimento facial pós-evento é classificado como **alto risco**, com obrigatoriedade de auditoria, autorização judicial e documentação de cada uso.
- **02/08/2027** — período de transição final para sistemas de alto risco.
- **Multas:** até €35 milhões ou 7% do *turnover* global para práticas proibidas; até €15 milhões ou 3% para violações de alto risco.

#### NIST FRTE/FATE (sucessor do FRVT, desde 2023)
- Atualizado em **março/2025**.
- ~200 algoritmos avaliados sobre 18M imagens de 8M pessoas.
- Achado central: disparidades em **falsos positivos** (até 7203× entre grupos) são muito maiores que em falsos negativos (~3×).
- FPs elevados em crianças, idosos, asiáticos e indígenas americanos — especialmente em baixa qualidade.

### 1.9 Eixo 7 — Privacidade: federated learning e differential privacy

Tendência colateral relevante. Em ambientes regulados pelo AI Act/GDPR, modelos de FR vêm sendo treinados com:
- **Federated Learning** — clientes mantêm imagens localmente
- **Differential Privacy** — ruído controlado em gradientes/dados
- **DP-FedFace (CIKM 2024)** — combinação específica para FR
- **Trade-off conhecido:** queda de 12-30% de acurácia vs. centralizado

### 1.10 Eixo 8 — Métricas de fairness modernas

A literatura 2024–2025 adotou métricas específicas para auditoria:

| Métrica | O que mede |
|---|---|
| Inequity Rate (IR) | Razão máx/mín de FMR/FNMR entre grupos |
| Fairness Discrepancy Rate (FDR) | Combinação ponderada de disparidades |
| GARBE | Inspirado no coeficiente de Gini |
| Comprehensive Equity Index (CEI) | Disparidades em taxas + scores |
| MAPE | Desvio relativo de FMR vs. benchmark |

---

## Parte II — Avaliação do Trabalho de MBA e Lacunas Identificadas

### 2.1 O que o MBA entregou

Pipeline funcional e reproduzível com:
- **Detector**: MTCNN
- **Backbone**: LResNet50E-IR (ResNet50 pré-treinada + camada FC customizada)
- **Loss**: CrossEntropyLoss e ArcFaceLoss
- **Otimizadores**: SGD e AdamW
- **Schedulers**: OneCycleLR e CosineAnnealingWarmRestarts
- **Dataset**: FairFace balanceado por *undersampling*
- **11 experimentos** comparativos com métricas precision/recall/F1/log-loss e matriz de confusão

**Melhores resultados:**
- 7 classes: acurácia 0,68 (Exp. 1: CrossEntropy + SGD + OneCycleLR)
- 2 classes (Black/White): 0,95 (Exp. 7) e 0,94 (Exp. 8)

### 2.2 Gaps identificados vs. SOTA 2024–2026

| Eixo | Dissertação de MBA (2024) | SOTA 2025–2026 | Severidade |
|---|---|---|---|
| Loss function | CrossEntropy / ArcFace | + AdaFace, MagFace, KP-RPE | **Alta** |
| Backbone | LResNet50E-IR | + TransFace++, LVFace | Média |
| Detector | MTCNN | + RetinaFace, YuNet | Média |
| Dataset | FairFace | + RFW, BUPT-Balancedface | **Alta** |
| Augmentation | Não aplicado | Diffusion-based (DCFace, VariFace) | Média |
| Métricas | Precision/Recall por classe | + Inequity Rate, FDR, GARBE | **Crítica** |
| Contexto regulatório | Não mencionado | EU AI Act, NIST FRTE 2025 | **Alta** |

### 2.3 Limitações reconhecidas na própria dissertação

- Gargalo na classe **Latino Hispanic** (F1 ≈ 0,50) — gap maior do conjunto.
- **Exp. 6 zerou** sem investigação de causa.
- Meta declarada de **95–98% de acurácia** não atingida no setup de 7 classes (atingida apenas no setup binário Black/White).
- Eixo "deficiência visual" introduzido no Cap. 1 mas não incorporado aos experimentos.

---

## Parte III — Proposta Evolutiva do Mestrado

### 3.1 Visão evolutiva

A tese parte do repositório `face_bias` e o eleva a um nível de rigor científico publicável: **uma pergunta empírica clara, hipóteses falsificáveis, intervenções controladas e auditoria de fairness padronizada**.

Não é uma ruptura — é uma extensão alinhada às evoluções da literatura 2024–2026 (AdaFace, DCFace, métricas IR/FDR/Gini) e ao contexto regulatório do EU AI Act.

### 3.2 Pergunta de pesquisa (revisada)

> **A rodada limpa (mai/2026, 11/11 experimentos, 9h38min) demonstrou que balancear o conjunto de treinamento via undersampling — a abordagem dominante na dissertação de MBA e em boa parte da literatura aplicada — produz Inequity Rate (F1) entre 1,76 e 1,86 e gap max-min entre 0,34 e 0,36 em todas as 5 configurações CE 7-class testadas, mesmo com 10.374 amostras por classe. *Que intervenções no espaço de loss e/ou no espaço de dados de fato fecham esse gap residual, e em que magnitude?***

A pergunta tem três sub-perguntas:

- **PQ1** — Funções de perda quality-adaptive (AdaFace, MagFace, KP-RPE) reduzem o gap demográfico residual quando aplicadas sobre o conjunto já balanceado?
- **PQ2** — Augmentação por geração sintética da classe pior (DCFace pré-treinado para Latino_Hispanic) reduz o gap residual? Em que magnitude?
- **PQ3** — A combinação loss-adaptive + augmentation sintética é meramente aditiva, ou há efeito de interação?

### 3.3 Hipóteses

> **H0 (FALSIFICADA empiricamente — 11 experimentos do clean run, mai/2026):**
> "Balancear o conjunto de treinamento por undersampling produz reconhecimento facial equitativo (IR ≈ 1,0)."
> **Resultado observado**: IR ∈ [1,76; 1,86] em todas as 5 configurações CE 7-class testadas com 25 épocas. A hipótese, central no trabalho de MBA, é rejeitada e organiza a investigação subsequente.

**Baseline empírico** (a ser superado pelas intervenções):
- **Melhor 7-class:** Exp 5 (CE + AdamW + CosineAnnealingWarmRestarts), acc=0,665, **IR=1,76, gap=0,36**.
- **F1 da pior classe (Latino_Hispanic):** 0,42 a 0,47 dependendo da configuração.
- **F1 da melhor classe (Black):** 0,78 a 0,84.

Hipóteses ativas:

- **H1 (PQ1)** — Substituir CrossEntropy por **AdaFace** sobre o conjunto balanceado reduz o IR (F1) em pelo menos 20% (de **1,76** para **≤ 1,41**), mantendo ou melhorando a acurácia agregada (≥ 0,665).
- **H2 (PQ2)** — **Augmentar a classe Latino_Hispanic** com 5.000–10.000 imagens sintéticas geradas por DCFace pré-treinado (sem retreinar o gerador) reduz o gap específico Latino_Hispanic ↔ Black em pelo menos 30% (de **0,36** para **≤ 0,25**) e eleva F1 Latino_Hispanic para **≥ 0,55**.
- **H3 (PQ3)** — A combinação **AdaFace + DCFace augmentation** produz redução de gap maior que a soma das reduções individuais (interação positiva).
- **H4** — O ranking de modelos por acurácia difere do ranking por IR. Implicação direta para auditoria regulatória sob o EU AI Act.
- **H5 (qualitativa via t-SNE/Grad-CAM)** — A redução de gap correlaciona-se com **maior compactação intra-classe e separação inter-classe** no espaço de embeddings da classe pior (Latino_Hispanic), observável em t-SNE pré/pós-intervenção.

### 3.4 Contribuições esperadas

1. **Auditoria empírica do mito do balanceamento**: tabela publicável mostrando que 11 configurações balanceadas via undersampling produzem todas IR ≥ 1,76 sobre FairFace 7-classes, contradizendo a intuição comum em pipelines aplicados.
2. **Estudo controlado de duas intervenções** (loss-adaptive + augmentation sintética), com decomposição da contribuição de cada uma e da interação entre elas.
3. **Análise representacional via t-SNE** correlacionando geometria de embeddings com gap de F1 — abre caminho para entender *por quê* o gap persiste.
4. **Pipeline reproduzível open-source** ([github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias](https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias)) — cada experimento do paper roda com `python scripts/run_all_experiments.py --output-dir <X>`.
5. **Submissão de artigo científico em workshop tier-A** — ver Parte III.5.

---

## Parte III.5 — Plano de Publicação Científica

### 3.5.1 Objetivo

A tese é projetada para gerar **pelo menos um artigo científico submetido a workshop tier-A**, como deliverable principal acadêmico de externalização do mestrado, em paralelo com o exame de qualificação interno do programa de Ciência da Computação da **Unifesp / ICT — São José dos Campos**.

### 3.5.2 Título de trabalho proposto

> **"Beyond Class Balancing: An Empirical Study of Loss-Adaptive and Synthetic-Data Interventions for Demographic Equity in Race Classification"**

Variantes alternativas a discutir com o orientador:

- "Why Undersampling Isn't Enough: A Diagnostic of Residual Bias in Balanced Face Classification"
- "Adaptive Margin Losses Meet Synthetic Augmentation: Closing the Race Gap on FairFace"

### 3.5.3 Estrutura narrativa (3 atos)

| Ato | Conteúdo | Origem dos dados |
|---|---|---|
| **1. Diagnóstico** ✅ pronto | 11 configurações balanceadas produzem IR ∈ [1,76; 1,86] em FairFace 7-classes (clean run, 25 épocas, 9h38min). Tabelas de F1 por classe, matrizes de confusão e gráficos disponíveis em `outputs/figures/clean/`. | [docs/clean_results.md](docs/clean_results.md) |
| **2. Intervenção em loss** | AdaFace, MagFace e (opcional) KP-RPE como substitutos de CrossEntropy/ArcFace sobre o mesmo dataset balanceado. Baseline a bater: Exp 5 (acc=0,665, IR=1,76). | Mês 2-3 |
| **3. Intervenção em dados** | DCFace pré-treinado para gerar 5-10k faces de Latino_Hispanic. Treinar com mix real+sintético sobre a melhor loss do Ato 2. Combinação com loss-adaptive. t-SNE pós-intervenção. | Mês 4-5 |

### 3.5.4 Veículos-alvo (em ordem de preferência)

| # | Veículo | Tipo | Periodicidade | Probabilidade de aceite |
|---|---|---|---|---|
| 1 | **WACV Workshop on Fair, Data-efficient, and Trusted Computer Vision** | Workshop tier-A | Anual (jan) | Alta |
| 2 | **IJCB (International Joint Conference on Biometrics) — Track de fairness** | Conferência tier-B+ | Anual (set) | Alta |
| 3 | **ACM FAccT (Conference on Fairness, Accountability, and Transparency)** | Conferência tier-A interdisciplinar | Anual (jun) | Média |
| 4 | **CVPR Workshop on Fairness, Accountability, Transparency, and Ethics in Computer Vision** | Workshop tier-A+ | Anual (jun) | Média |
| 5 | **IEEE TBIOM (Transactions on Biometrics, Behavior, and Identity Science)** | Revista tier-A | Contínua | Baixa-Média (escopo de artigo de revista é mais profundo) |

**Plano A**: WACV Fair-CV Workshop (deadline tipicamente set/out, evento jan). Alinhado com Mês 5 da proposta.
**Plano B**: IJCB 2027 (deadline tipicamente abr/mai). Alinhado com fim do Mês 6.

### 3.5.5 Critérios mínimos de publicabilidade que a tese deve atingir

| Critério | Como atender |
|---|---|
| **Reprodutibilidade total** | Repositório público + DVC nos datasets + seeds fixas + Docker (opcional) |
| **Comparação justa** | Mesmo split, mesmas seeds, mesmo hardware para todas as condições |
| **Métricas padronizadas** | IR, FDR, Gini conforme Kotwal & Marcel (2025) — já implementadas em `face_bias/evaluation/metrics.py` |
| **Significância estatística** | Cada experimento rodado com pelo menos 3 seeds; reportar média ± desvio |
| **Análise qualitativa** | t-SNE + Grad-CAM por classe pior, antes e depois das intervenções |
| **Limitação do escopo** | Reconhecer no paper que é avaliação intra-FairFace; cross-dataset (RFW) é trabalho futuro |
| **Discussão regulatória** | Conectar achados com EU AI Act art. 26 (auditoria de sistemas de alto risco) |

### 3.5.6 Riscos específicos da publicação

| Risco | Mitigação |
|---|---|
| Resultado de H1 ou H2 ser nulo | **Null result é publicável**: WACV/FAccT aceitam evidência negativa quando bem documentada e diagnosticada via t-SNE/Grad-CAM. Reframe do paper passa a ser "limites das duas abordagens" |
| Tempo apertado para 3 seeds × 6 condições | Reduzir condições para 4 e seeds para 2 se necessário; reportar como limitação |
| Reviewer pedir cross-dataset (RFW) | Submeter solicitação de RFW no Mês 1; se não chegar a tempo, posicionar como trabalho futuro |
| Reviewer pedir comparação com TransFace++ | Já mapeado no estado da arte (Parte I); fora do escopo deste paper, mas justificar com hardware local |

---

## Parte IV — Motivadores com Referências Bibliográficas

Cada motivador conecta uma **lacuna do MBA** a uma **referência da literatura 2024–2026**.

### Motivador 1 — Métricas de fairness modernas (lacuna metodológica crítica)

A dissertação reporta precision/recall por classe, mas não calcula nenhuma métrica formal de equidade. Kotwal & Marcel (2025) consolidam o conjunto canônico — IR, FDR, GARBE, CEI — e mostram que escolha de métrica altera o ranking de algoritmos.

- Kotwal, K. & Marcel, S. (2025). *Review of Demographic Fairness in Face Recognition*. IEEE TBIOM.
- Atzori et al. (2025). *Racial Bias within Face Recognition: A Survey*. ACM Computing Surveys.

### Motivador 2 — ArcFace deixou de ser SOTA em fairness

A literatura mostra AdaFace, MagFace e KP-RPE consistentemente acima de ArcFace em benchmarks que medem fairness e qualidade variável.

- Kim, M. et al. (2022). *AdaFace: Quality Adaptive Margin for Face Recognition*. CVPR.
- Kim, M., Jain, A., Liu, X. (2024). *Enhanced Face Recognition Using KP-RPE*. ICLR.
- Meng, Q. et al. (2021). *MagFace: A Universal Representation for Face Recognition and Quality Assessment*. CVPR.

### Motivador 3 — A transição CNN → Vision Transformer

ResNets continuam competitivas em custo, mas TransFace++ (TPAMI 2025) e LVFace (ICCV 2025) estabeleceram um novo SOTA. Nenhum trabalho similar ao do MBA, em FairFace, ainda comparou os três paradigmas (CNN, ViT puro, ViT+landmark) sob a ótica de fairness.

- Dan, J. et al. (2025). *TransFace++: Rethinking the Face Recognition Paradigm with a Focus on Accuracy, Efficiency, and Security*. IEEE TPAMI.
- You, Y. et al. (2025). *LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition*. ICCV.
- Kim, M., Jain, A., Liu, X. (2025). *50 Years of Automated Face Recognition*. arXiv:2505.24247.

### Motivador 4 — Dados sintéticos como mitigação direta de viés

A lacuna *Latino Hispanic* identificada no MBA é exatamente o tipo de problema que diffusion models de geração de faces resolvem. DCFace, VariFace e GANDiffFace demonstraram em 2024–2025 que síntese controlada por identidade reduz disparidades.

- Kim, M. et al. (2023). *DCFace: Synthetic Face Generation with Dual Condition Diffusion Model*. CVPR.
- *VariFace: Fair and Diverse Synthetic Dataset Generation for Face Recognition* (2024). arXiv:2412.06235.
- NYU Tandon (2025). *Demographically Diverse Synthetic Image Dataset for AI Training*.
- Mi, J. et al. (2025). *3DMM-Guided Data Synthesis for Face Recognition*. CVPR.

### Motivador 5 — Avaliação cross-dataset como requisito mínimo de rigor

Treinar e testar no mesmo dataset é uma fragilidade reconhecida. BUPT-Balancedface é o dataset de treinamento canônico para fairness; RFW é o benchmark padrão de teste.

- Wang, M. & Deng, W. (2020). *Mitigating Bias in Face Recognition Using Skewness-Aware Reinforcement Learning*. CVPR.
- Hanson, J.A. et al. (2024). *Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets*.

### Motivador 6 — Detector legado introduz viés a montante

Achado do próprio MBA (Cap. 4: queda de detecção em rostos *Black* com MTCNN) indica que parte do viés pode vir do detector. RetinaFace, YuNet e SCRFD corrigem isso.

- Deng, J. et al. (2020). *RetinaFace: Single-shot Multi-level Face Localisation in the Wild*. CVPR.
- Wu, W. et al. (2023). *YuNet: A Tiny Millisecond-level Face Detector*. Machine Intelligence Research.
- Guo, J. et al. (2021). *Sample and Computation Redistribution for Efficient Face Detection (SCRFD)*.

### Motivador 7 — Contexto regulatório (EU AI Act) como justificativa de pesquisa

Desde 02/02/2025 estão em vigor proibições absolutas para certos usos de FR; em 02/08/2026 entram exigências de auditoria para sistemas de alto risco. Pesquisa em fairness deixou de ser acadêmica e passou a ser **requisito de conformidade**.

- *Regulation (EU) 2024/1689 — AI Act*, Articles 5, 26 and Annex III.
- *NIST FRTE/FATE Demographic Effects* (atualizado em 03/2025) — FRVT Part 8.

### Motivador 8 — Análise interseccional (raça × gênero × idade)

Kotwal & Marcel (2025) identificam interseccionalidade como direção prioritária. FairFace tem todas as labels (age, gender, race), nunca exploradas conjuntamente no MBA.

- Kotwal & Marcel (2025), seção *Future Directions*.
- Buolamwini, J. & Gebru, T. (2018). *Gender Shades*. (referência seminal).

### Motivador 9 — Explainability como diferencial qualitativo

A sugestão de Grad-CAM e ativações intermediárias do Cap. 5 do MBA é exatamente o que falta para análise qualitativa de viés. É executável e gera material rico para o capítulo de discussão.

- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Reis, R.R. et al. (2024). *Explainability, Fairness and Privacy in AI for Biometrics*.

---

## Parte V — Plano de Trabalho (6 Meses)

Cronograma realista assumindo dedicação parcial (~15–20h/semana). Cada mês foi recalibrado em torno do **paper como deliverable principal** e do **hardware local (RTX 4070 SUPER, 12 GB)** como única infraestrutura de cálculo. Não há dependência de cloud.

### Mês 1 · `2026-05-06 → 2026-06-05` · Diagnóstico empírico ✅ concluído (08/05/2026)

**Objetivo:** consolidar o achado central do paper (balanceamento ≠ equidade) com evidência rigorosa.

**Atividades:**

- ✅ Pipeline `face_bias` refatorado, instalável, testado (81 testes). [Commit 70100ab]
- ✅ Smoke run dos 11 experimentos do MBA com pipeline corrigido em 5 épocas — [docs/smoke_results.md](docs/smoke_results.md).
- ✅ **Rodada limpa dos 11 experimentos com 25 épocas + EarlyStopping(patience=5) + grad_clip_norm=5.0** — 11/11 OK em 9h38min wall-clock. [Commit 8321568]
- ✅ Geração de gráficos para os 11 experimentos via `scripts/plot_all_experiments.py` (44 PNG/PDF em `outputs/figures/clean/`).
- ✅ Tabela consolidada `MBA reportado × Smoke × Clean run` em [docs/clean_results.md](docs/clean_results.md), com 7 findings prontos para o paper.

**Entregável principal:** ✅ **diagnóstico consolidado** em `docs/clean_results.md` — IR ∈ [1,76; 1,86] em 5/5 CE 7-class; ArcFace catastrófico em 3/3 configurações (IR=∞).
**Entregável secundário:** ✅ guia de preparação para reunião com coordenador em [docs/meeting_prep_2026-05-11.md](docs/meeting_prep_2026-05-11.md).

---

### Mês 2 · `2026-06-06 → 2026-07-05` · Intervenção em loss — AdaFace + MagFace

**Objetivo:** medir quanto do gap residual é fechado por funções de perda quality-adaptive.

**Atividades:**

- Implementar **AdaFace** em `face_bias/models/losses.py` (referência: Kim et al., CVPR 2022).
- Implementar **MagFace** (referência: Meng et al., CVPR 2021).
- Configs `exp12_adaface_*.yaml` e `exp13_magface_*.yaml` gerados via `scripts/generate_experiment_configs.py`.
- Treinar 3 seeds × 2 condições (AdaFace, MagFace) × 25 épocas no setup balanceado.
- Comparar IR/F1/gap pré e pós-intervenção sobre os mesmos splits.

**Entregável principal:** **tabela "Loss intervention" do paper** com média ± desvio sobre 3 seeds.
**Entregável secundário:** análise de t-SNE pré/pós para Latino_Hispanic, mostrando se houve melhora representacional.

**Tempo estimado de GPU**: 2 condições × 3 seeds × ~1h por run = **~6h GPU** + análise.

---

### Mês 3 · `2026-07-06 → 2026-08-05` · Intervenção em dados — DCFace augmentation

**Objetivo:** medir quanto do gap específico Latino_Hispanic ↔ Black é fechado por augmentação sintética.

**Atividades:**

- Setup do **DCFace pré-treinado** local (modelo público, ~800MB, roda em 12GB) — apenas inferência, sem retreinar o gerador.
- Gerar 5.000 e 10.000 imagens sintéticas para Latino_Hispanic, condicionadas em sub-amostras reais como referência.
- Avaliação de qualidade básica (visual + FID se viável local).
- Compor dataset híbrido (real_balanced + sintético_Latino_Hispanic) e re-treinar com a melhor loss do Mês 2.
- Comparar baseline balanceado vs. balanceado + augmentação sintética.

**Entregável principal:** **tabela "Data intervention"** com 3 condições (no aug, +5k sintéticos, +10k sintéticos).
**Entregável secundário:** análise qualitativa Grad-CAM sobre amostras Latino_Hispanic — modelo passou a focar em features faciais diferentes pós-augmentação?

**Tempo estimado**: geração ~3h + treino 3 seeds × 2 condições × ~1h = **~9h GPU** total.

---

### Mês 4 · `2026-08-06 → 2026-09-05` · Combinação + análise representacional

**Objetivo:** validar H3 (interação loss-adaptive × augmentation) e produzir o capítulo qualitativo.

**Atividades:**

- Treinar combinação **AdaFace + DCFace augmentation** (a mais promissora do M2 + a mais promissora do M3).
- Decompor contribuições: tabela 2x2 `(AdaFace=sim/não) × (DCFace=sim/não)`.
- Análise interseccional **raça × gênero × idade** sobre o melhor modelo (labels já existentes em FairFace).
- t-SNE final pré/pós em alta-resolução para o paper.
- Grad-CAM grids comparativos para o paper.

**Entregável principal:** **tabela 2x2 + figuras de t-SNE/Grad-CAM** prontas para o paper.
**Entregável secundário:** ablação de épocas de treino — hyperparam recommendation.

**Tempo estimado**: ~6h GPU.

---

### Mês 5 · `2026-09-06 → 2026-10-05` · Escrita e submissão

**Objetivo:** finalizar o paper e submeter ao primeiro veículo-alvo.

**Atividades:**

- Escrita das 4 seções restantes do paper (Methods, Results, Discussion, Conclusion).
- Iteração com orientador sobre 3 versões.
- Refinamento de figuras (alta-resolução, fontes consistentes, paleta CB-friendly).
- Submissão à WACV Workshop on Fair-CV (ou IJCB 2027 dependendo do calendário).
- Início da escrita do exame de qualificação do programa Unifesp / ICT (capítulos Introdução + Metodologia).

**Entregável principal:** **paper submetido**.
**Entregável secundário:** capítulos 1–2 do exame de qualificação (~30 páginas).

---

### Mês 6 · `2026-10-06 → 2026-11-05` · Qualificação interna

**Objetivo:** finalizar o exame de qualificação do programa Unifesp / ICT.

**Atividades:**

- Escrita dos capítulos 3–5 (Resultados, Discussão, Conclusões).
- Apresentação interna ao orientador.
- Definição do escopo da fase 2 do mestrado (12+ meses pós-qualificação): cross-dataset (RFW/BUPT), interseccionalidade aprofundada, ou nova arquitetura (TransFace++).

**Entregável principal:** **rascunho do exame de qualificação** (60–80 páginas) entregue.
**Entregável secundário:** roadmap fase 2 aprovado pelo orientador.

---

### 5.7 Resumo visual do cronograma

| Mês | Período | Foco | Entregável principal | GPU |
|---|---|---|---|---:|
| 1 | mai/2026 | Diagnóstico empírico ✅ | [docs/clean_results.md](docs/clean_results.md) — 11/11 experimentos | **9h38min real** |
| 2 | jun/2026 | AdaFace + MagFace | Tabela "Loss intervention" | ~6h estimada |
| 3 | jul/2026 | DCFace augmentation | Tabela "Data intervention" | ~9h estimada |
| 4 | ago/2026 | Combinação + análise representacional | Tabela 2x2 + figuras finais | ~6h estimada |
| 5 | set/2026 | Escrita + submissão do paper | **Paper submetido** | mínima |
| 6 | out/2026 | Qualificação interna | Rascunho 60-80 pgs | mínima |

**GPU consumida no Mês 1:** 9h38min real (vs ~13h estimadas — early-stopping economizou ~25%).
**GPU restante estimada (Meses 2-4):** ~21 horas para os experimentos de intervenção.
**Total previsto para 6 meses:** ~31 horas em RTX 4070 SUPER local, sem cloud.

---

## Parte VI — Riscos, Recursos e Decisões em Aberto

### 6.1 Riscos e mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| AdaFace/MagFace não reduzir IR significativamente (H1 falsa) | Média | **Null result é publicável**; reframe do paper passa a ser "limites das duas abordagens" + diagnóstico via t-SNE |
| DCFace pré-treinado não couber em 12 GB | Baixa-Média | Rodar inferência em mini-batches; reduzir para Vec2Face (mais leve); pior caso, gerar offline |
| RTX 4070 SUPER ficar inadequada para alguma combinação | Média | Reduzir batch_size de 128 para 64 ou 32; usar gradient accumulation; usar mixed-precision (`torch.cuda.amp`) |
| Resultados não convergirem para 3 seeds (variância alta) | Média | Documentar como limitação; rodar 5 seeds se houver tempo |
| Workshop alvo (WACV) ter deadline em janeiro/26 (após o paper estar pronto) | Baixa | Plano B: IJCB 2027 (deadline tipicamente abr); Plano C: arXiv preprint imediato |
| Sobrecarga de escopo | Média | Cortar Mês 4 (combinação) se necessário; submeter paper apenas com loss + dataset interventions |

### 6.2 Recursos necessários

| Recurso | Detalhes |
|---|---|
| **Computacional** | **RTX 4070 SUPER (12 GB) local — único equipamento de treinamento.** Sem cloud, sem cluster acadêmico. |
| **Storage** | Disco local + DVC com remote opcional para backup do dataset processado |
| **Datasets reais** | FairFace (já adquirido + pré-processado em `data/processed/fairface_aligned`). RFW se conseguir aprovação institucional, mas **não é dependência crítica** — pode ficar para fase 2 |
| **Modelo gerador** | DCFace pré-treinado (público, ~800MB) — apenas inferência, não treinamento |
| **Software** | Stack atual do `face_bias` é suficiente; adicionar `torch.cuda.amp` e `pytorch-grad-cam` (já em deps) |
| **Bibliografia** | ~25–30 referências mapeadas (Parte IV) |
| **Acesso institucional** | Unifesp / Portal de Periódicos CAPES para download de papers (CVPR/ICCV/TPAMI/TBIOM) |
| **Pré-treinamento** | LResNet50 ImageNet (já incluso em torchvision) |

### 6.3 Decisões em aberto para o coordenador

1. **Foco do paper**: as **3 sub-perguntas (PQ1, PQ2, PQ3)** caem todas em um único paper, ou separadas em paper-de-loss + paper-de-dados? Recomendação: um único paper de 8 páginas é a forma típica para os veículos-alvo.

2. **Veículo prioritário**: **WACV Workshop Fair-CV** (deadline ~set/26, evento jan/27) ou **IJCB 2027** (deadline ~abr/27)? Recomendo WACV pelo prazo alinhado com o Mês 5.

3. **Cross-dataset (RFW)**: incluir como condição obrigatória do paper ou marcar como trabalho futuro? Recomendo **trabalho futuro** dadas as restrições de hardware local e janela de 6 meses.

4. **Múltiplas seeds**: 3 seeds × 6 condições é o mínimo defensável. Se houver tempo, esticar para 5 seeds × 6 condições. Decisão depende de calendário.

5. **Eixo de backbone (ViT / TransFace++) — incluir ou marcar como trabalho futuro?**
   A proposta atual escopa as intervenções (loss-space e data-space) **sobre o mesmo backbone** (LResNet50E_IR), justamente para isolar o efeito das intervenções. Comparar backbones (LResNet50 vs ViT-B vs TransFace++) é uma frente complementar valiosa, mas:
   - **Cabe na hipótese H6 (nova) se decidirmos adicionar**: "ViT-B/TransFace++ produz menor IR que LResNet50E_IR sob a mesma loss e mesmo dataset balanceado".
   - **Custo computacional**: ViT-B/16 treina apertado em 12 GB (batch ≤ 64 com `torch.cuda.amp`); TransFace++ provavelmente **não cabe** na RTX 4070 SUPER. Uma rodada do experimento matrix com ViT seria ~3× o custo do clean run atual (~30h GPU).
   - **Infra alternativa**: posso provisionar crédito Azure (Standard_NC6s_v3 ou similar com V100/A100) para a frente de backbone, sem prejuízo da frente local. Estimativa ~200–400h GPU para a ablação completa.
   - **Trade-off**: agregar mais um eixo experimental enriquece o paper mas dilui o foco. **Recomendação default**: manter backbone fixo no escopo do paper de qualificação e abrir a frente ViT/TransFace++ como **paper-2** (fase 2 do mestrado, pós-qualificação).
   - **Pergunta direta ao coordenador**: o senhor considera esse eixo de backbone necessário para a contribuição do mestrado (mesmo com infra Azure), ou suficiente como trabalho futuro?

---

## Referências

### Surveys e revisões (2024–2025)
- Kotwal, K. & Marcel, S. (2025). *Review of Demographic Fairness in Face Recognition*. IEEE TBIOM. https://arxiv.org/html/2502.02309v1
- Atzori et al. (2025). *Racial Bias within Face Recognition: A Survey*. ACM Computing Surveys. https://dl.acm.org/doi/10.1145/3705295
- Kim, M., Jain, A., Liu, X. (2025). *50 Years of Automated Face Recognition*. arXiv:2505.24247. https://arxiv.org/html/2505.24247v1
- *Fairness and Bias Mitigation in Computer Vision: A Survey* (2024). https://arxiv.org/html/2408.02464

### Funções de perda
- Kim, M. et al. (2022). *AdaFace: Quality Adaptive Margin for Face Recognition*. CVPR. https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf
- Meng, Q. et al. (2021). *MagFace: A Universal Representation for Face Recognition and Quality Assessment*. CVPR.
- Kim, M. et al. (2024). *Enhanced Face Recognition Using KP-RPE*. ICLR. https://openreview.net/notes/edits/attachment?id=6vzEJ9VmHh&name=pdf

### Arquiteturas modernas
- Dan, J. et al. (2025). *TransFace++: Rethinking the Face Recognition Paradigm*. IEEE TPAMI. https://arxiv.org/abs/2308.10133
- TransFace++ GitHub: https://github.com/DanJun6737/TransFace_pp
- You, Y. et al. (2025). *LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition*. ICCV. https://openaccess.thecvf.com/content/ICCV2025/papers/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.pdf

### Dados sintéticos
- Kim, M. et al. (2023). *DCFace: Synthetic Face Generation with Dual Condition Diffusion Model*. CVPR. https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_DCFace_Synthetic_Face_Generation_With_Dual_Condition_Diffusion_Model_CVPR_2023_paper.pdf
- *VariFace: Fair and Diverse Synthetic Dataset Generation for Face Recognition* (2024). https://arxiv.org/html/2412.06235v1
- Mi, J. et al. (2025). *3DMM-Guided Data Synthesis for Face Recognition*. CVPR. https://openaccess.thecvf.com/content/CVPR2025/papers/Mi_Data_Synthesis_with_Diverse_Styles_for_Face_Recognition_via_3DMM-Guided_CVPR_2025_paper.pdf
- *AI-Face: Million-Scale Demographically Annotated Dataset* (CVPR 2025). https://arxiv.org/html/2406.00783v3
- *How Knowledge Distillation Mitigates the Synthetic Gap in Fair FR* (2024). https://arxiv.org/html/2408.17399
- *Impact of Balancing Real and Synthetic Data on Accuracy and Fairness* (2024). https://arxiv.org/html/2409.02867
- NYU Tandon (2025). https://engineering.nyu.edu/news/nyu-tandon-researchers-mitigate-racial-bias-facial-recognition-technology-demographically

### Detectores faciais
- Deng, J. et al. (2020). *RetinaFace: Single-shot Multi-level Face Localisation in the Wild*. CVPR.
- Wu, W. et al. (2023). *YuNet: A Tiny Millisecond-level Face Detector*. Machine Intelligence Research.
- Guo, J. et al. (2021). *Sample and Computation Redistribution for Efficient Face Detection (SCRFD)*.
- *Benchmark: Face Detection Using Deep Learning Models* (Springer, 2025). https://link.springer.com/chapter/10.1007/978-3-031-93103-1_11
- *What is Face Detection? Ultimate Guide 2025* (LearnOpenCV). https://learnopencv.com/what-is-face-detection-the-ultimate-guide/

### Datasets
- Wang, M. & Deng, W. (2020). *Mitigating Bias in Face Recognition Using Skewness-Aware Reinforcement Learning*. CVPR. (origem RFW e BUPT-Balancedface)
- BUPT-Balancedface dataset: http://www.whdeng.cn/RFW/Trainingdataste.html
- RFW benchmark: http://www.whdeng.cn/RFW/testing.html
- Hanson, J.A. et al. (2024). *Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets*.

### Regulação
- *Regulation (EU) 2024/1689 — AI Act*. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- *AI Act Article 5 — Prohibited AI Practices*. https://artificialintelligenceact.eu/article/5/
- *Biometrics in the EU: Navigating GDPR + AI Act* (IAPP, 2025). https://iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act
- *EU AI Act August 2026 — Biometric Surveillance Explainer*. https://stateofsurveillance.org/news/eu-ai-act-august-2026-biometric-surveillance-explainer/
- *NIST FRVT/FRTE Demographic Effects*. https://pages.nist.gov/frvt/html/frvt_demographics.html
- *NIST FRTE 1:N Identification*. https://pages.nist.gov/frvt/html/frvt1N.html
- *NIST FATE Face In Video Evaluation 2024*. https://pages.nist.gov/frvt/html/frte_five.html

### Privacidade
- *DP-FedFace: Privacy-Preserving Facial Recognition in Real Federated Scenarios* (CIKM 2024). https://dl.acm.org/doi/10.1145/3627673.3679901

### Explicabilidade
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Reis, R.R. et al. (2024). *Explainability, Fairness and Privacy in AI for Biometrics*. https://repositorio-aberto.up.pt/bitstream/10216/169904/2/744839.pdf

### Referências do MBA mantidas como base
- Buolamwini, J. & Gebru, T. (2018). *Gender Shades*. PMLR.
- Deng, J. et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Karkkainen, K. & Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age*.
- Zhang, K. et al. (2016). *Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks (MTCNN)*.

---

*Documento elaborado a partir da dissertação de MBA `Facial-Recognition-Models-Mitigating-Bias` (out/2024) e de pesquisa textual no estado da arte de reconhecimento facial e fairness, executada em mai/2026.*
