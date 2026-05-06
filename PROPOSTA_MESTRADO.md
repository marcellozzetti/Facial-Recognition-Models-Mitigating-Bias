# Proposta de Mestrado em IA — Evolução do Trabalho de MBA

**Tema:** Mitigação de viés demográfico em reconhecimento facial: avaliação sistemática de funções de perda quality-adaptive, dados sintéticos balanceados e métricas de equidade.

**Autor:** Marcello Ozzetti
**Trabalho de origem:** Dissertação de MBA em IA / USP — *Facial Recognition Models Mitigating Bias*
**Data desta proposta:** 06 de maio de 2026
**Janela de planejamento:** 6 meses (mai/2026 → nov/2026)

---

## Sumário

1. [Sumário Executivo](#1-sumário-executivo)
2. [Parte I — Pesquisa Textual Científica (Estado da Arte 2024–2026)](#parte-i--pesquisa-textual-científica-estado-da-arte-20242026)
3. [Parte II — Avaliação do Trabalho de MBA e Lacunas Identificadas](#parte-ii--avaliação-do-trabalho-de-mba-e-lacunas-identificadas)
4. [Parte III — Proposta Evolutiva do Mestrado](#parte-iii--proposta-evolutiva-do-mestrado)
5. [Parte IV — Motivadores com Referências Bibliográficas](#parte-iv--motivadores-com-referências-bibliográficas)
6. [Parte V — Plano de Trabalho (6 Meses)](#parte-v--plano-de-trabalho-6-meses)
7. [Parte VI — Riscos, Recursos e Decisões em Aberto](#parte-vi--riscos-recursos-e-decisões-em-aberto)
8. [Referências](#referências)

---

## 1. Sumário Executivo

A dissertação de MBA estabeleceu uma base experimental funcional — pipeline `LResNet50E-IR + ArcFace/CrossEntropy` sobre o dataset **FairFace**, com 11 experimentos comparativos — e identificou três limitações que motivam a continuidade em nível de mestrado:

- **Gargalo persistente** de desempenho na classe *Latino Hispanic* (F1 ≈ 0,50);
- **Ausência de métricas formais de equidade** (apenas precision/recall por classe);
- **Avaliação restrita** a um único dataset, uma única arquitetura e um único detector facial (MTCNN).

A proposta de mestrado **não recomeça do zero**: estende o repositório existente para incorporar as evoluções da literatura 2024–2026 (AdaFace, TransFace++, dados sintéticos via DCFace/VariFace, métricas IR/FDR/CEI, datasets RFW/BUPT-Balancedface) e produz uma contribuição original auditável, alinhada ao novo contexto regulatório do **EU AI Act** (vigência plena em ago/2026) e às atualizações do **NIST FRTE/FATE**.

**Output esperado em 6 meses:** rascunho do exame de qualificação (60–80 páginas) e submissão de artigo em workshop *tier-A*.

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

A tese de mestrado **parte do repositório existente** (`pipelines/`, `models/`, `utils/`) e o eleva a três níveis de rigor não cobertos no MBA: **multi-dataset**, **multi-arquitetura** e **fairness-first**.

Não é uma ruptura — é uma extensão sistemática alinhada às evoluções de 2024–2026 e ao novo contexto regulatório.

### 3.2 Pergunta de pesquisa

> **Em que medida a combinação de funções de perda quality-adaptive (AdaFace), aumentação por dados sintéticos demograficamente balanceados (DCFace/VariFace) e arquiteturas Vision-Transformer (TransFace++) reduz o gap de equidade demográfica em reconhecimento facial, medido por métricas modernas de fairness (Inequity Rate, FDR, CEI), em comparação ao baseline LResNet50E-IR + ArcFace?**

### 3.3 Hipóteses

- **H1** — Substituir ArcFace por AdaFace reduz o gap de F1 entre o pior e o melhor grupo demográfico em pelo menos 30%, sem perda de acurácia agregada.
- **H2** — Aumentar a classe minoritária com dados sintéticos via DCFace/VariFace reduz o gap específico de *Latino Hispanic*, com efeito superior a *undersampling*.
- **H3** — A migração de ResNet50 para TransFace++/ViT-B melhora robustez à variação de pose e iluminação, especialmente em grupos sub-representados.
- **H4** — O ranking de modelos por acurácia agregada difere significativamente do ranking por Inequity Rate, evidenciando a inadequação de métricas tradicionais para auditoria regulatória sob o EU AI Act.

### 3.4 Contribuições esperadas

1. **Benchmark cross-dataset reproduzível** (FairFace + RFW + BUPT-Balancedface) com fairness metrics padronizadas.
2. **Estudo controlado** do impacto isolado de cada variável (loss × backbone × synthetic augmentation) sobre fairness.
3. **Pipeline open-source** com Grad-CAM integrado para análise interpretável de viés por subgrupo.
4. **Análise interseccional** raça × gênero × idade, aproveitando labels existentes do FairFace.
5. **Submissão de artigo** em conferência/revista *tier-A* (target: IEEE TBIOM, IJCB, ou WACV/Workshop).

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

Cronograma realista assumindo dedicação parcial (~15–20h/semana). Cada mês tem **um entregável principal e um secundário**.

### Mês 1 · `2026-05-06 → 2026-06-05` · Consolidação e reprodutibilidade

**Objetivo:** estabelecer baseline reprodutível e formalizar a proposta acadêmica.

**Atividades:**
- Refatorar `pipelines/training_pipeline.py` e `models/` para reprodutibilidade total (seeds fixas, logging estruturado, MLflow ou Weights & Biases).
- Adquirir e integrar **RFW** e **BUPT-Balancedface** ao pipeline.
- Implementar `utils/fairness_metrics.py` com **IR, FDR, GARBE, CEI**.
- Recalcular fairness sobre as matrizes de confusão dos 11 experimentos do MBA (não exige re-treino).

**Entregável principal:** **proposta formal de mestrado** (15–20 páginas) — pergunta, hipóteses, métodos, cronograma, bibliografia anotada.
**Entregável secundário:** tabela comparativa de fairness sobre o MBA, evidenciando o gap real.

---

### Mês 2 · `2026-06-06 → 2026-07-05` · Modernização do pipeline base

**Objetivo:** substituir componentes legados sem trocar arquitetura.

**Atividades:**
- Substituir **MTCNN por RetinaFace** (preferencialmente) ou YuNet (se houver restrição computacional).
- Implementar **AdaFace** e **MagFace** em `models/losses.py`.
- Re-rodar Exp. 1, 3, 7 e 9 do MBA com (i) detector novo, (ii) AdaFace, (iii) MagFace.

**Entregável principal:** tabela comparativa **baseline (MBA) vs. modernized (M2)** com fairness metrics.
**Entregável secundário:** análise de quanto do gap em *Latino Hispanic* é atribuível ao detector vs. à loss.

---

### Mês 3 · `2026-07-06 → 2026-08-05` · Síntese de dados balanceados

**Objetivo:** gerar e validar augmentação sintética para classes minoritárias.

**Atividades:**
- Setup do **DCFace** pré-treinado e geração controlada de imagens para *Latino Hispanic*, *Middle Eastern* e *Southeast Asian*.
- Avaliação de qualidade (FID, identidade preservada).
- Composição de novo dataset híbrido (real + sintético) e re-treino dos melhores modelos do M2.
- Comparação direta vs. *undersampling* (técnica do MBA).

**Entregável principal:** **estudo controlado de síntese** mostrando ganho em fairness por classe alvo.
**Entregável secundário:** dataset híbrido versionado (DVC ou similar).

---

### Mês 4 · `2026-08-06 → 2026-09-05` · Vision Transformers

**Objetivo:** avaliar substituição de backbone.

**Atividades:**
- Integrar **ViT-B/16** pré-treinado e **TransFace++** (se houver pesos públicos; caso contrário, ViT-B + DPAP/EHSM reimplementados).
- Treinar com a melhor combinação loss/dataset do M3.
- Comparar **LResNet50E-IR vs. ViT-B vs. TransFace++** sob fairness metrics.

**Entregável principal:** **estudo de ablação backbone**.
**Entregável secundário:** rascunho da seção *Resultados* do mestrado.

---

### Mês 5 · `2026-09-06 → 2026-10-05` · Análise interseccional e explicabilidade

**Objetivo:** aprofundamento qualitativo.

**Atividades:**
- Análise interseccional **raça × gênero × idade** (FairFace tem todas as labels — pivot tables, heatmaps).
- Implementar **Grad-CAM** e gerar mapas para o melhor e pior modelo, por subgrupo.
- Identificar **failure modes** sistemáticos (ex.: classe Black + faixa idosa, mulher Latino Hispanic).

**Entregável principal:** **capítulo de discussão qualitativa** com Grad-CAM e análise interseccional.
**Entregável secundário:** submissão de **artigo curto** para workshop (target: workshop de fairness em CVPR/ECCV/IJCB 2027).

---

### Mês 6 · `2026-10-06 → 2026-11-05` · Escrita e fechamento da fase 1

**Objetivo:** consolidar a fase experimental e estruturar o restante do mestrado.

**Atividades:**
- Escrita dos capítulos 1–4 do mestrado (introdução, fundamentação, metodologia, resultados parciais).
- Reunião de qualificação com orientador.
- Definição do escopo da fase 2 (próximos 6–12 meses): federated learning + DP, aprofundamento em interseccionalidade, ou avaliação cross-cultural.

**Entregável principal:** **rascunho do exame de qualificação** (60–80 páginas).
**Entregável secundário:** roadmap fase 2 aprovado pelo orientador.

---

### 5.7 Resumo visual do cronograma

| Mês | Período | Foco | Entregável principal |
|---|---|---|---|
| 1 | mai/2026 | Reprodutibilidade + métricas | Proposta formal + fairness retroativo do MBA |
| 2 | jun/2026 | Detector novo + losses modernas | Comparativo baseline vs. modernized |
| 3 | jul/2026 | Dados sintéticos | Estudo controlado de síntese |
| 4 | ago/2026 | Vision Transformers | Ablação backbone |
| 5 | set/2026 | Interseccionalidade + Grad-CAM | Discussão qualitativa + submissão artigo |
| 6 | out/2026 | Escrita | Rascunho de qualificação |

---

## Parte VI — Riscos, Recursos e Decisões em Aberto

### 6.1 Riscos e mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| RFW exigir aplicação formal e demorar | Alta | Submeter solicitação no Mês 1; ter BUPT-Balancedface como plano B |
| DCFace/VariFace exigirem GPU além da V100 do MBA | Média | Solicitar crédito Azure acadêmico; usar amostragem reduzida; alternativa Vec2Face (mais leve) |
| TransFace++ não ter pesos pré-treinados públicos | Média | Cair para ViT-B padrão + perdas modernas; documentar como limitação |
| Resultados não confirmarem hipóteses | Baixa-Média | Falsificação tem valor científico; reportar honestamente fortalece a tese |
| Sobrecarga de escopo (6 meses, 4 dimensões) | Alta | Priorizar em ordem: fairness metrics → AdaFace → synthetic data → ViT. Cortar ViT se necessário |
| EU AI Act mudar interpretação durante a pesquisa | Baixa | Acompanhar atualizações da Comissão Europeia; usar versão estável como referência |

### 6.2 Recursos necessários

- **Computacional:** Azure Standard_NC6s_v3 (V100) já validado no MBA; estimar 200–400h adicionais. Considerar upgrade para A100 no M4 (TransFace++).
- **Datasets:** FairFace (já tem), RFW (solicitar), BUPT-Balancedface (público), DCFace pré-treinado (público).
- **Software novo:** MLflow ou Weights & Biases (logging), DVC (versionamento de dados), `insightface` (detectores modernos), `pytorch-grad-cam`.
- **Bibliografia:** ~25–30 referências novas (já mapeadas).
- **Acesso institucional:** ICCV/CVPR/TPAMI via USP.

### 6.3 Decisões em aberto para o coordenador

1. **Foco do mestrado:** manter os **3 eixos (loss + backbone + synthetic)** com profundidade média, ou **focar em apenas 1** com profundidade alta? Mestrado bem feito em 1 eixo > superficial em 3.
2. **Output acadêmico:** mirar **artigo + dissertação** ou **dissertação apenas**? Isso muda o cronograma do Mês 5.
3. **Programa/orientação:** definição de programa, linha de pesquisa e orientador adequados ao tema (preferência por viés algorítmico / arquiteturas / aplicações).

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
