# Apresentação ao orientador — Atividades das últimas 2 semanas

> Conteúdo destinado à reunião com Prof. Marcos Quiles (orientador,
> coordenador da matéria de Redes Neurais Unifesp/ICT). Período coberto:
> 2026-05-10 a 2026-05-24 (14 dias). Estrutura: SOTA encontrado +
> Fatores e Experimentos Realizados + Achados + Pontos de Discussão +
> Próximos Passos. Terminologia alinhada com Haykin (*Redes Neurais:
> Princípios e Práticas*, 2ª ed). Data: 2026-05-24.

---

## SLIDE 1 — Título e contexto

**Atribuição Causal e Pareamento Metodológico em Classificação de Raça 7-classes no FairFace**

Reunião de progresso — atividades 10-24 de maio de 2026

Autor: Marcello Ozzetti
Orientador: Prof. Marcos Quiles
Programa: Mestrado em Ciência da Computação — Unifesp/ICT

---

## SLIDE 2 — Estado do projeto antes (2 semanas atrás)

**O que tínhamos em 10/05:**
- 5 Fatores de atribuição causal rodados (Linha A)
- ConvNeXt-T identificado como única alavanca arquitetural (Fator 5: F1
  +2.3pp ~7σ; IR −0.13 ~3σ)
- Critério Pareto-aware best-epoch (Linha B) documentado
- Sem posicionamento absoluto vs literatura — não sabíamos onde estávamos

**Dúvida central:** *"Nosso F1≈0.69 é bom ou ruim? Comparado a quê?"*

---

## SLIDE 3 — Atividade 1: Pesquisa textual SOTA (10-15 maio)

**Mapeamento da literatura para race 7-class no FairFace:**

Auditoria sobre 7 artigos relevantes (resumo em `docs/sota_papers_summary.md`):

| Paper | Faz race 7-class? |
|---|---|
| FairFace (Kärkkäinen 2021) — paper-pai do dataset | ❌ (4-class merged) |
| **FineFACE (Liu 2024)** — citado como SOTA-fairness | ❌ (**descoberta nossa: classifica GÊNERO**) |
| U-FaTE (CVPR 2024) | ❌ (multi-task) |
| FairGRAPE (ECCV 2022) | ❌ (compressão) |
| DSAP (Inf. Fusion 2024) | ❌ (auditoria de dataset) |
| Fairness-is-in-Details (ECML 2025) | ❌ (auditoria de dataset) |
| **Hassanpour 2024 (arXiv 2410.24148)** | ✅ **ÚNICA referência para race 7-class** |

**Achado central da pesquisa textual:** **Hassanpour 2024 é a SOTA-CNN
real** (ResNet-34 baseline = 0.720 acurácia). FaceScanPaliGemma do mesmo
paper = 0.757 acurácia, mas é VLM de 3 bilhões de parâmetros — não
comparável arquiteturalmente.

---

## SLIDE 4 — Atividade 2: Experimentos adicionais de posicionamento (15-22 maio)

Rodamos **4 experimentos adicionais** para situar nosso F1 vs literatura:

| Nome descritivo | O que isola | Resultado (3-seed) |
|---|---|---|
| **Exp-FairFace** | Recipe do paper-pai (RN-34 + Adam) | F1 = 0.676 ± 0.006 |
| **Exp-FineFACE** | Recipe FineFACE sem multi-expert | F1 = 0.663 ± 0.007 |
| **Exp-DadosBrutos** | Sem nosso pré-processamento (MTCNN + cleaning) | F1 = 0.695 ± 0.006 |
| **Exp-ProtocoloSOTA** | **Reprodução integral do protocolo Hassanpour** | F1 = 0.703 ± 0.005, IR = 1.541 |

**Observações importantes:**
- Nosso recipe AdamW@224 é localmente ótimo (Exp-FairFace e Exp-FineFACE
  entregam F1 abaixo do nosso controle)
- Sem MTCNN/cleaning (Exp-DadosBrutos), F1 sobe ligeiramente — **nosso
  pré-processamento custa F1 e IR** (achado emergente)
- Sob protocolo Hassanpour idêntico, ficamos a **−0.85pp single-vs-single**
  (1.7σ da variância natural)

---

## SLIDE 5 — Atividade 3: Análise de robustez (22 maio)

**Pergunta:** *"A alavanca ConvNeXt-T persiste sem subamostragem por raça?"*

**Rob-SemSubamostragem** — 6 runs (controle + ConvNeXt × 3 seeds), `balance: none`

**Achado dual:**

1. **Subamostragem é estatisticamente neutra** (Outcome B):
   - Acurácia, F1 e IR essencialmente idênticos com vs sem subamostragem
   - Refuta empiricamente prática-padrão da literatura ("subamostrar por
     classe protegida melhora equidade")

2. **Alavanca ConvNeXt-T persiste**:
   - Sob protocolo casado original: +2.3pp F1 (~7σ)
   - Sem subamostragem: +1.4pp F1 (~1.5σ — atenuada)
   - Sob protocolo Hassanpour: +2.4pp F1 (~3.5σ)
   - **Robusta em 3 protocolos distintos**

---

## SLIDE 6 — Atividade 4: Auditoria empírica de código (22-23 maio)

**Pergunta:** *"Há limitadores no nosso código mascarando o desempenho real?"*

Testamos 4 hipóteses de hiperparâmetros sub-otimizados:

| Auditoria | Variável testada | Resultado | Veredito |
|---|---|---|---|
| **Aud-Paciência** | early_stopping_patience: 5 → 15 | bit-a-bit idêntico | ❌ não é limitador |
| **Aud-Dropout** | model.dropout: 0.2 → 0.0 | F1 −0.003, IR +0.058 | ❌ não é limitador (descoberta: dropout HELP a equidade) |
| **Aud-Augmentation** | TrivialAugmentWide | F1 +0.9pp, mas **desloca viés** | ⚠️ rejeitado por análise per-class |
| **Aud-Oversampling** | balance: none → oversample | F1 −2pp, **Latino_Hispanic −12.5pp** | ❌ catastrófico |

**Achado tese-relevante:** **nenhuma das técnicas comuns "corrige" o
gap residual. A única intervenção robusta verificada é arquitetural
(ConvNeXt-T como rede dorsal).**

---

## SLIDE 7 — Atividade 5: Análises pós-treinamento (23 maio)

**4 técnicas científicas estabelecidas, aplicadas SEM retreinamento:**

| Análise | Referência canônica | O que adiciona |
|---|---|---|
| **PT-Interseccional** | Buolamwini & Gebru, FAccT 2018 | Mede disparidade race × gênero × idade |
| **PT-Ensemble** | Lakshminarayanan et al., NeurIPS 2017 (8000+ citações) | Reduz variância via deep ensemble de 3 seeds |
| **PT-TTA** | Krizhevsky 2012 (AlexNet 10-crop) | Test-Time Augmentation com transformações seguras |
| **PT-Calibração** | Guo et al., ICML 2017 | Temperature scaling + threshold per-class |

**Resultados:**
- **PT-Interseccional**: IR cresce de 1.541 (raça) para 3.241 (raça × gênero × idade). Pior subgrupo: Middle Eastern × Female × 3-9 (28.6%)
- **PT-Ensemble**: estimativa agregada = 0.7299 acc / 0.7285 F1 / 1.501 IR
- **PT-TTA**: combinado com Ensemble entrega IR = **1.474 (melhor equidade do projeto)**
- **PT-Calibração**: T=0.95 (ensemble já bem-calibrado); threshold ajuste marginal

---

## SLIDE 8 — Achado central #1: Alavanca ConvNeXt-T

**A única intervenção arquitetural que move acurácia E equidade
simultaneamente, robusta em 3 protocolos:**

| Protocolo | Δ F1 (ConvNeXt vs Controle) | Δ Razão de Disparidade |
|---|---|---|
| Casado original (5 Fatores) | **+2.3pp (~7σ)** | **−0.128 (~3σ)** |
| Sem subamostragem (Rob) | +1.4pp (~1.5σ) | −0.065 (~0.7σ) |
| Protocolo Hassanpour (Exp-ProtocoloSOTA) | **+2.4pp (~3.5σ)** | **−0.087 (~1.8σ)** |

**Hipóteses mecanísticas a discutir** (Fator 5, Seção 4):
- H1: LayerNorm > BatchNorm em lotes demograficamente assimétricos
- H2: Kernel depthwise 7×7 captura features de escala intermediária
- H3: Recipe de pré-treinamento moderno (AdamW + augs + EMA)

---

## SLIDE 9 — Achado central #2: Critério Pareto-aware best-epoch

**Contribuição metodológica replicável (Linha B):**

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

**Implicação:** comparações entre famílias de função de custo podem
ser **invertidas** se o critério ingênuo for usado. Achado replicável
em outros pipelines de fine-tuning multi-loss.

---

## SLIDE 10 — Posicionamento final vs SOTA-CNN (ético)

| Configuração | Acurácia | F1 | IR ↓ | Comparação |
|---|---|---|---|---|
| Hassanpour ResNet-34 baseline (single-run) | **0.720** | — | — | referência |
| **Nosso ConvNeXt-T single seed (s42)** | **0.7115** | 0.7083 | 1.496 | **−0.85pp (1.7σ — simétrica)** |
| Nosso ConvNeXt-T média 3 seeds | 0.7060 ± 0.005 | 0.7034 ± 0.005 | 1.541 ± 0.044 | −1.4pp (com variância) |
| Nosso ConvNeXt-T deep ensemble | 0.7299 | 0.7285 | 1.501 | agregação (Lakshminarayanan 2017) |
| Nosso ConvNeXt-T + TTA | 0.7298 | 0.7286 | **1.474** | melhor equidade do projeto |
| Nosso ConvNeXt-T + Ensemble + Calibração | 0.7304 | 0.7292 | 1.501 | melhor confiabilidade |

**Mensagem ética para a banca:**

> *"Sob comparação simétrica single-vs-single (consistente com a
> metodologia de single-run reportada por Hassanpour 2024), nosso
> sistema fica a −0.85pp do baseline, dentro de 1.7σ da variância
> natural medida em nosso protocolo 3-seed. As estimativas agregadas
> (ensemble, TTA, calibração) refletem MAIOR CONFIABILIDADE da estimativa
> via técnicas científicas estabelecidas — não reivindicação de
> superação direta."*

---

## SLIDE 11 — Pontos de discussão com o orientador (5)

### Discussão 1 — Posicionamento vs SOTA: estamos no caminho certo?
Comparação simétrica nos dá −0.85pp (dentro da variância). Ensemble/TTA
elevam confiabilidade. A narrativa de "contribuição em atribuição
causal (Linha A) + critério metodológico (Linha B)" é defensável?

### Discussão 2 — Achado de FineFACE não classificar raça
Esse achado da auditoria textual elimina uma comparação fantasma comum
na literatura. Vale dar destaque na introdução (1 parágrafo) ou
mencionar discretamente no related work?

### Discussão 3 — Análise interseccional revela "Middle Eastern Female 3-9" como pior subgrupo
IR cresce de 1.5 para 3.2 quando crossado com gênero × idade. Como
posicionar isso? Vai para Resultados ou Discussão? Recomendamos
mitigação específica nas conclusões?

### Discussão 4 — Roadmap pós-qualificação (Caminho 1)
Identificamos 2 técnicas modernas candidatas para entre qualificação
e defesa final: **Group DRO** (Sagawa ICLR 2020, loss fairness-aware)
e **DINOv2 backbone** (Oquab 2023, self-supervised features). Doc
`disruption_roadmap.md`. Faz sentido?

### Discussão 5 — Auditoria empírica de código no corpo da tese
Os 4 testes refutando suspeitos no nosso próprio código (Aud-Paciência,
Aud-Dropout, Aud-Augmentation, Aud-Oversampling) são raros em
mestrado de CS. Vão para o capítulo de Resultados (como achado) ou
para Apêndice (como rigor metodológico)?

---

## SLIDE 12 — Próximos passos (junho 2026)

### Fase atual: REDAÇÃO (escopo controlado)

| Cap. | Conteúdo | Prazo sugerido |
|---|---|---|
| Capítulo de Metodologia | Protocolo, métricas, 5 fatores, anchors | 1-7/06 |
| Capítulo de Resultados | Tabelas consolidadas, narrativa Linha A + B | 8-14/06 |
| Capítulo de Discussão | Interpretar achados, conectar com literatura, limitações | 15-21/06 |
| Introdução + Conclusão | Por último, quando souber o que está dizendo | 22-30/06 |

### Régua de decisão (THESIS_STATEMENT §11):

> *"Cada experimento, leitura ou análise daqui pra frente: avança uma
> das 3 contribuições declaradas (atribuição causal, Pareto-aware,
> auditoria de confundidores), ou fecha uma limitação declarada SEM
> abrir nova? Se não, é fora de escopo."*

### Defesa de qualificação: ago/2026 (Art. 49 Unifesp/ICT)

---

## SLIDE 13 — Recursos (custo de tudo isso)

**Custo computacional acumulado nas últimas 2 semanas:**

| Item | Tempo GPU | Energia | Custo BR |
|---|---|---|---|
| 5 Fatores definitivos | ~25h | ~12.5 kWh | ~R$ 12 |
| 4 Experimentos Adicionais | ~25h | ~12.5 kWh | ~R$ 12 |
| Rob-SemSubamostragem | ~10h | ~5 kWh | ~R$ 5 |
| Auditoria empírica (4 testes) | ~7h | ~3.5 kWh | ~R$ 3.5 |
| Análises Pós-Treinamento | ~1h (só inferência) | ~0.5 kWh | ~R$ 0.5 |
| **TOTAL 2 semanas** | **~68h** | **~34 kWh** | **~R$ 33** |
| CO₂ equivalente | — | — | **~1.3 kg** (matriz BR) |

**Hardware:** RTX 4070 SUPER 12GB Windows + nobreak TSShara 1000VA 220V.

**Documentação produzida no período:**
- 18+ documentos em `docs/` (15+ novos, 3 atualizados)
- Terminologia alinhada com Haykin
- Padronização de nomenclatura (`nomenclatura_experimentos.md`)
- Roadmap de disrupção pós-qualificação (`disruption_roadmap.md`)
- Sumário de SOTA (`sota_papers_summary.md`)

---

## SLIDE 14 — Encerramento

**Estado da dissertação após 2 semanas intensas:**

- ✅ 5 Fatores + 4 Experimentos Adicionais + 1 Análise de Robustez + 4 Auditorias + 4 Análises pós-treino = **18 análises**
- ✅ Posicionamento absoluto: −0.85pp vs SOTA-CNN (dentro da variância)
- ✅ Auditoria empírica refutando suspeitos no nosso código
- ✅ Análise interseccional + ensemble + TTA + calibração documentadas
- ✅ Roadmap pós-qualificação preparado (Group DRO + DINOv2)
- ✅ Bateria experimental ENCERRADA

**Próximo bloco único:** REDAÇÃO dos capítulos da dissertação.

**Pergunta direta ao orientador:**
*"O pacote está pronto para qualificação em agosto? Algum ajuste de
escopo, ênfase ou priorização que o senhor recomenda antes de eu
começar a redação?"*

---

## Apêndice — Glossário de nomes para a apresentação ao vivo

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
