# Apresentação ao orientador — Atividades das últimas 2 semanas (v2)

> Conteúdo destinado à reunião com Prof. Marcos Quiles (orientador,
> coordenador da matéria de Redes Neurais Unifesp/ICT). Período coberto:
> 2026-05-10 a 2026-05-24 (14 dias). Estrutura revisada 2026-05-24:
> adicionados slides de decisões estratégicas, metodologia, métricas e
> resumo dos 5 fatores; removido slide de custos. **Total: 17 slides.**
> Terminologia alinhada com Haykin (*Redes Neurais: Princípios e
> Práticas*, 2ª ed.).

---

## SLIDE 1 — Título e contexto

**Atribuição Causal e Pareamento Metodológico em Classificação de Raça 7-classes no FairFace**

Reunião de progresso — atividades 10-24 de maio de 2026

Autor: Marcello Ozzetti
Orientador: Prof. Marcos Quiles
Programa: Mestrado em Ciência da Computação — Unifesp/ICT

---

## SLIDE 2 — Decisões estratégicas do projeto (sumário)

**Decisões fundamentais tomadas e mantidas:**

| Dimensão | Decisão | Justificativa |
|---|---|---|
| **Conjunto de dados** | Manter FairFace (97k brutas / 72k clean) | Único dataset facial explicitamente balanceado por raça (Kärkkäinen 2021); padrão na literatura de fairness |
| **Asset do MBA** | Aproveitar pipeline + ganho de +1.35pp F1 do cleaning | Reuso de investimento — Cap. 4 MBA-USP estabeleceu baseline reproduzível |
| **Tarefa de classificação** | Raça em 7 classes, no-domain | Tarefa sem benchmark canônico publicado — espaço fértil para atribuição |
| **Arquitetura padrão** | ResNet-50 ImageNet pré-treinada | Equivalente arquitetural a AlDahoul et al. 2024 ResNet-34 baseline |
| **Recipe de treino** | AdamW lr=1e-3 + 224×224 + 25 épocas + cosine warm restart | Modernização incremental da recipe FairFace original |
| **Protocolo experimental** | 3 sementes aleatórias casadas (42, 1, 2) | Alinhado com Pineau JMLR 2021 (replicabilidade ML) |
| **Critério de seleção** | val_f1_macro (Pareto-aware) | Corrige viés sistêmico de min(val_loss) p/ cabeças com margem angular |
| **Posicionamento estratégico** | NÃO competir SOTA absoluta — contribuir em atribuição | Tarefa sem ecossistema de mitigação saturado |

➡️ **Estas decisões formam a "régua de escopo" do projeto** — declaradas
em `docs/THESIS_STATEMENT.md §11` e aplicadas a todo experimento que
entra/não entra na dissertação.

---

## SLIDE 3 — Metodologia adotada (práticas científicas aplicadas)

**Práticas aplicadas consistentemente em TODOS os experimentos:**

### Rigor experimental
- ✅ **3 sementes aleatórias casadas** (42, 1, 2) — todos os experimentos
- ✅ **Comparação casada com regra do 1σ** entre médias dos seeds
- ✅ **Reporte de média ± desvio-padrão** (incerteza quantificada)
- ✅ **Determinismo CUDA habilitado** (`deterministic=True`)

### Critério de seleção de modelo
- ✅ **val_f1_macro** como critério primário (Pareto-aware, ver Linha B)
- ❌ NÃO usar `min(val_loss)` — anti-correlacionado com F1 p/ margin heads

### Auditoria contínua
- ✅ **Auditoria de fórmulas vs literatura** (`docs/formula_desk_check.md`)
- ✅ **Procedência declarada** para cada métrica (AlDahoul et al., FineFACE, Mehrabi)
- ✅ **Documentação reproduzível** por experimento (configs YAML + scripts)

### Replicabilidade (alinhada com guidelines modernas)
- **Pineau et al. JMLR 2021** — múltiplas seeds + intervalos confiança
- **Henderson et al. AAAI 2018** — single-seed é insuficiente em DL
- **Bouthillier et al. MLSys 2021** — variância em benchmarks de imagem

---

## SLIDE 4 — Métricas adotadas

### Métricas primárias reportadas

| Métrica | Fórmula | O que mede |
|---|---|---|
| **F1 macro** | (1/C) Σ F1_c | Acurácia média balanceada entre as 7 classes (não pondera por suporte) |
| **Razão de disparidade ↓** | max(F1_c) / min(F1_c) | Equidade entre grupos demográficos (1.0 = perfeito; cresce com viés) |
| **Acurácia** | corretos / total | Métrica agregada não-balanceada (apenas referência) |

### Procedência das métricas (canônica na literatura)

| Métrica nossa | Equivalente literatura | Fonte |
|---|---|---|
| `disparity_ratio` | Max-Min Ratio | **AlDahoul et al. 2024**, **FineFACE 2024** |
| `max_min_disparity` | Max-Min Gap | Mehrabi et al. ACM CSur 2021 |
| `coefficient_of_variation` | CV / dispersão estatística | Speicher KDD 2018 |
| `gini` | Coeficiente de Gini | Speicher KDD 2018; origem econométrica |

### Decisão de nomenclatura

`inequity_rate → disparity_ratio` (renomeação em 2026-05-23) — alinhado
com AlDahoul et al. 2024 e FineFACE 2024, evitando confusão com o IR de
verificação biométrica de Pereira & Marcel (que é produto FMR × FNMR).

---

## SLIDE 5 — Os 5 Fatores executados (Linha A)

Cinco dimensões algorítmicas testadas sob protocolo casado 3-seed:

| Fator | Variável manipulada | Δ F1 vs controle | Δ Razão Disparidade | Veredito |
|---|---|---|---|---|
| **F1 — Conjunto de dados** | Limpeza multi-face (72k vs 97k) | +1.35pp (CE) | ✗ não-significativo | parcial (só acurácia) |
| **F2 — Topologia da camada de saída** | MLP vs Linear (HPO Optuna) | ~0 | ✓ −0.11 (≫1σ) | modesta (só equidade) |
| **F3 — Família de função de custo** | CE, ArcFace, AdaFace, MagFace | ✗ nulo (CE ≈ Ada ≈ Mag) | ✗ nulo | nulo (com correção Pareto-aware!) |
| **F4 — Paradigma de aprendizado** | CE+linear vs CE+SupCon (one-stage) | ✗ nulo | ✗ nulo | nulo (confirma predição FSCL) |
| **F5 — Rede dorsal pré-treinada** | RN-50, RN-34, ViT-B/16, **ConvNeXt-T** | ✅ **+2.3pp (~7σ)** | ✅ **−0.13 (~3σ)** | **forte — alavanca real** |

**Conclusão da Linha A:** das 5 dimensões testadas, **apenas a rede
dorsal moderna LayerNorm-based (ConvNeXt-T) é alavanca robusta de
acurácia + equidade simultâneas**. Loss, paradigma e topologia são
nulls atribuição-grade ou parciais. Cleaning paga em acurácia sem
mover equidade. Achado verificado e replicado em 3 protocolos.

---

## SLIDE 6 — Estado do projeto há 2 semanas (10/05/2026)

**O que tínhamos em 10/05:**
- 5 Fatores rodados (Linha A) — ConvNeXt-T identificado como alavanca
- Critério Pareto-aware documentado (Linha B)
- **Sem posicionamento absoluto** vs literatura

**Dúvida central:** *"Nosso F1 ≈ 0.69 é bom ou ruim? Comparado a quê?"*

**Plano para 2 semanas:**
1. Mapear SOTA real (pesquisa textual)
2. Rodar experimentos adicionais de posicionamento
3. Testar robustez do achado central
4. Auditar limitadores potenciais no nosso código
5. Aplicar análises pós-treinamento padrão (defesa-fechamento)

---

## SLIDE 7 — Atividade 1: Pesquisa textual SOTA (10–15/05)

**Mapeamento de 7 artigos relevantes** (resumo em `docs/sota_papers_summary.md`):

| Paper | Faz race 7-class? |
|---|---|
| FairFace (Kärkkäinen 2021) — paper-pai | ❌ (4-class merged) |
| **FineFACE (Liu 2024)** — citado como SOTA-fairness | ❌ **descoberta: classifica GÊNERO** |
| U-FaTE (CVPR 2024) | ❌ (multi-task) |
| FairGRAPE (ECCV 2022) | ❌ (compressão) |
| DSAP (Inf. Fusion 2024) | ❌ (auditoria de dataset) |
| Fairness-is-in-Details (ECML 2025) | ❌ (auditoria de dataset) |
| **AlDahoul et al. 2024 (arXiv 2410.24148)** | ✅ **ÚNICA referência publicada** |

**Achado central:** **AlDahoul et al. 2024 é a SOTA-CNN real** (ResNet-34 = 0.720
acurácia). FaceScanPaliGemma do mesmo paper = 0.757 mas é VLM de 3
bilhões de parâmetros — não comparável arquiteturalmente.

---

## SLIDE 8 — Atividade 2: Experimentos Adicionais de Posicionamento (15–22/05)

| Nome descritivo | O que isola | Resultado (3-seed) |
|---|---|---|
| **Exp-FairFace** | Recipe paper-pai (RN-34 + Adam) | F1 = 0.676 ± 0.006 |
| **Exp-FineFACE** | Recipe FineFACE sem multi-expert | F1 = 0.663 ± 0.007 |
| **Exp-DadosBrutos** | Sem nosso pré-processamento | F1 = 0.695 ± 0.006 |
| **Exp-ProtocoloSOTA** | Reprodução integral do protocolo AlDahoul et al. | F1 = 0.703 ± 0.005, IR = 1.541 |

**Observações:**
- Nosso recipe AdamW@224 é localmente ótimo (Exp-FairFace e Exp-FineFACE entregam F1 abaixo)
- Pré-processamento custa F1 e IR (Exp-DadosBrutos melhora)
- Sob protocolo AlDahoul et al. idêntico: −0.85pp single-vs-single (1.7σ da variância)

---

## SLIDE 9 — Atividade 3: Análise de Robustez (22/05)

**Pergunta:** *"A alavanca ConvNeXt-T persiste sem subamostragem por raça?"*

**Rob-SemSubamostragem** — 6 runs (controle + ConvNeXt × 3 seeds), `balance: none`

**Achado dual:**

1. **Subamostragem é estatisticamente NEUTRA** (Outcome B):
   - Acurácia, F1 e IR essencialmente idênticos com vs sem subamostragem
   - Refuta empiricamente prática-padrão da literatura

2. **Alavanca ConvNeXt-T PERSISTE em 3 protocolos:**
   - Protocolo casado original: +2.3pp F1 (~7σ)
   - Sem subamostragem: +1.4pp F1 (~1.5σ — atenuada)
   - Protocolo AlDahoul et al.: +2.4pp F1 (~3.5σ)
   - **Robusta em 3 protocolos distintos**

---

## SLIDE 10 — Atividade 4: Auditoria Empírica de Hiperparâmetros (22–23/05)

| Auditoria | Variável testada | Resultado | Veredito |
|---|---|---|---|
| **Aud-Paciência** | patience: 5 → 15 | bit-a-bit idêntico | ❌ não é limitador |
| **Aud-Dropout** | dropout: 0.2 → 0.0 | F1 −0.003, IR +0.058 | ❌ não é limitador |
| **Aud-Augmentation** | TrivialAugmentWide | F1 +0.9pp, mas **desloca viés** | ⚠️ rejeitado per-class |
| **Aud-Oversampling** | balance: none → oversample | F1 −2pp, **Latino −12.5pp** | ❌ catastrófico |

**Achado tese-relevante:** **nenhuma técnica comum "corrige" o gap. A
única intervenção robusta verificada é arquitetural (ConvNeXt-T).**

---

## SLIDE 11 — Atividade 5: Análises Pós-Treinamento (23/05)

**4 técnicas científicas estabelecidas, aplicadas SEM retreinamento:**

| Análise | Referência canônica | O que adiciona |
|---|---|---|
| **PT-Interseccional** | Buolamwini & Gebru, FAccT 2018 | Mede disparidade race × gênero × idade |
| **PT-Ensemble** | Lakshminarayanan et al., NeurIPS 2017 (8000+ cit) | Reduz variância via deep ensemble |
| **PT-TTA** | Krizhevsky 2012 (10-crop) | Test-Time Augmentation com transformações seguras |
| **PT-Calibração** | Guo et al., ICML 2017 | Temperature scaling + threshold per-class |

**Resultados principais:**
- **PT-Interseccional**: IR cresce 1.541 (raça) → 3.241 (raça × gênero × idade). Pior subgrupo: Middle Eastern × Female × 3-9 (28.6%)
- **PT-Ensemble**: estimativa agregada = 0.7299 acc / 0.7285 F1 / 1.501 IR
- **PT-TTA**: combinado com Ensemble entrega IR = **1.474 (melhor equidade do projeto)**
- **PT-Calibração**: T=0.95 (ensemble já bem-calibrado)

---

## SLIDE 12 — Achado Central #1: Alavanca ConvNeXt-T

**A única intervenção arquitetural que move acurácia E equidade
simultaneamente, robusta em 3 protocolos:**

| Protocolo | Δ F1 (ConvNeXt vs Controle) | Δ Razão de Disparidade |
|---|---|---|
| Casado original (5 Fatores) | **+2.3pp (~7σ)** | **−0.128 (~3σ)** |
| Sem subamostragem (Rob) | +1.4pp (~1.5σ) | −0.065 (~0.7σ) |
| Protocolo AlDahoul et al. (Exp-ProtocoloSOTA) | **+2.4pp (~3.5σ)** | **−0.087 (~1.8σ)** |

**Hipóteses mecanísticas a discutir** (Fator 5):
- H1: LayerNorm > BatchNorm em lotes demograficamente assimétricos
- H2: Kernel depthwise 7×7 captura features de escala intermediária
- H3: Recipe de pré-treinamento moderno (AdamW + augs + EMA)

---

## SLIDE 13 — Achado Central #2: Critério Pareto-aware (Linha B)

**Contribuição metodológica replicável:**

Descoberta: o critério padrão `min(val_loss)` (default em PyTorch
Lightning, Keras) é **anti-correlacionado com F1 macro** para cabeças
com margem angular (ArcFace, AdaFace, MagFace).

**Magnitude do viés sistemático sob protocolo casado:**

| Função de custo | Δ F1 sob critério correto vs ingênuo |
|---|---|
| Cross-Entropy | +0.021 |
| **AdaFace** | **+0.146** (12% de melhoria relativa) |
| MagFace | +0.087 |
| ArcFace | +0.034 |

**Implicação:** comparações entre famílias de função de custo podem ser
**invertidas** se o critério ingênuo for usado.

---

## SLIDE 14 — Posicionamento final vs SOTA-CNN (versão ética)

| Configuração | Acurácia | F1 | IR ↓ | Comparação |
|---|---|---|---|---|
| AlDahoul ResNet-34 baseline (single-run) | **0.720** | — | — | referência |
| **Nosso ConvNeXt-T single seed (s42)** | **0.7115** | 0.7083 | 1.496 | **−0.85pp (1.7σ)** — simétrica |
| Nosso ConvNeXt-T média 3 seeds | 0.7060 ± 0.005 | 0.7034 ± 0.005 | 1.541 ± 0.044 | −1.4pp (com variância) |
| Nosso ConvNeXt-T deep ensemble | 0.7299 | 0.7285 | 1.501 | agregação (Lakshminarayanan 2017) |
| Nosso ConvNeXt-T + TTA | 0.7298 | 0.7286 | **1.474** | melhor equidade do projeto |
| Nosso ConvNeXt-T + Ensemble + Calibração | 0.7304 | 0.7292 | 1.501 | melhor confiabilidade |

**Mensagem ética para a banca:**

> *"Sob comparação simétrica single-vs-single (consistente com a
> metodologia de single-run reportada por AlDahoul et al. 2024), nosso sistema
> fica a −0.85pp do baseline, dentro de 1.7σ da variância natural
> medida em nosso protocolo 3-seed. As estimativas agregadas (ensemble,
> TTA, calibração) refletem MAIOR CONFIABILIDADE da estimativa via
> técnicas científicas estabelecidas — não reivindicação de superação
> direta."*

---

## SLIDE 15 — Pontos de discussão com o orientador (5)

### Discussão 1 — Posicionamento vs SOTA: estamos no caminho certo?
Comparação simétrica nos dá −0.85pp (dentro da variância). Ensemble/TTA
elevam confiabilidade. Narrativa Linha A + Linha B é defensável?

### Discussão 2 — Achado de FineFACE não classificar raça
Destaque na introdução (1 parágrafo) ou discretamente no related work?

### Discussão 3 — Análise interseccional revela "Middle Eastern Female 3-9" como pior subgrupo
Vai para Resultados ou Discussão? Recomendamos mitigação específica?

### Discussão 4 — Roadmap pós-qualificação (Caminho 1)
Group DRO (Sagawa 2020) + DINOv2 (Oquab 2023) entre qualificação e
defesa final. Faz sentido?

### Discussão 5 — Auditoria empírica de código no corpo da tese
Capítulo de Resultados (como achado) ou Apêndice (como rigor metodológico)?

---

## SLIDE 16 — Próximos passos (junho 2026 — REDAÇÃO)

| Cap. | Conteúdo | Prazo sugerido |
|---|---|---|
| Capítulo de Metodologia | Protocolo, métricas, 5 fatores, anchors | 1-7/06 |
| Capítulo de Resultados | Tabelas consolidadas, narrativa Linha A + B | 8-14/06 |
| Capítulo de Discussão | Interpretar achados, conectar com literatura | 15-21/06 |
| Introdução + Conclusão | Por último, quando souber o que está dizendo | 22-30/06 |

**Régua de decisão (THESIS_STATEMENT §11):**

> *"Cada experimento, leitura ou análise daqui pra frente: avança uma
> das 3 contribuições declaradas (atribuição causal, Pareto-aware,
> auditoria de confundidores), ou fecha uma limitação declarada SEM
> abrir nova? Se não, é fora de escopo."*

**Defesa de qualificação:** ago/2026 (Art. 49 Unifesp/ICT)
**Defesa final:** data-limite fev/2028 (Art. 30)

---

## SLIDE 17 — Encerramento

**Estado da dissertação após 2 semanas intensas:**

- ✅ 5 Fatores + 4 Experimentos Adicionais + 1 Análise de Robustez + 4 Auditorias + 4 Análises pós-treino = **18 análises** em 5 fases
- ✅ Posicionamento absoluto: −0.85pp vs SOTA-CNN (dentro da variância)
- ✅ Auditoria empírica refutando suspeitos no nosso código
- ✅ Análise interseccional + ensemble + TTA + calibração documentadas
- ✅ Roadmap pós-qualificação preparado (Group DRO + DINOv2)
- ✅ Bateria experimental ENCERRADA

**Próximo bloco único:** REDAÇÃO dos capítulos da dissertação.

**Pergunta direta ao orientador:**

> *"O pacote está pronto para qualificação em agosto? Algum ajuste de
> escopo, ênfase ou priorização que o senhor recomenda antes de eu
> começar a redação?"*

---

## Apêndice — Glossário de nomes para apresentação ao vivo

Se o orientador perguntar sobre os códigos usados em docs antigos:

| Código antigo | Nome descritivo (slide) |
|---|---|
| 🅐.1 | Exp-FairFace |
| 🅐.2 | Exp-FineFACE |
| 🅓 | Exp-DadosBrutos |
| 🅔 | Exp-ProtocoloSOTA |
| 🅑 | Rob-SemSubamostragem |
| Teste A | Aud-Paciência |
| Teste B | Aud-Augmentation |
| Teste C1 | Aud-Dropout |
| Teste D | Aud-Oversampling |
| Combo #1 | PT-Interseccional |
| Combo #2 | PT-Ensemble |
| Combo #3 | PT-TTA |
| Combo #4 | PT-Calibração |
