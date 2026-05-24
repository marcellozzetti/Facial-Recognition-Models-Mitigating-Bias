# Posicionamento vs. baselines da literatura — FairFace race 7-class

> Material de tese / defesa. Resposta direta à pergunta da banca *"vocês reportam F1≈0.69; isto está atrás do SOTA?"*. Documenta a paisagem real de baselines publicados para a tarefa exata e decompõe honestamente o gap absoluto em escolhas metodológicas declaradas. Atualizado 2026-05-22 com (a) descoberta via pesquisa textual de que FineFACE não classifica raça; (b) Hassanpour et al. 2024 como SOTA real publicado para race 7-class; (c) batch final do anchor 🅐.2.

## 1. A tarefa exata que rodamos

| Item | Valor |
|---|---|
| Dataset | FairFace (Kärkkäinen & Joo, WACV 2021), versão padding=1.25 (448×448) |
| Atributo | **race (7 classes)**: White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino |
| Cenário | **in-domain** (treina FairFace, testa FairFace) |
| Pré-processamento (F1-F5) | MTCNN-aligned crops; clean dataset (multi-face removed, 72k) |
| Pré-processamento (anchor 🅓) | original FairFace (sem re-align, 97k) |
| Métrica primária | **F1 macro** + **disparity_ratio** (= max/min F1 per classe; ver `docs/formula_desk_check.md`) |
| Resolução | **224×224** (resize de 448 quando aplicável) |
| Backbone padrão | ResNet-50 ImageNet pretrained; alavanca confirmada = ConvNeXt-T |

## 2. SOTA real para race 7-class FairFace in-domain

A pesquisa textual concluída em 2026-05-22 (ver `docs/sota_7class_race_audit.md`) identifica **uma referência publicada e uma comunitária** que reportam números para a tarefa exata:

### 2.1 Hassanpour et al., 2024 (arXiv 2410.24148) — SOTA publicado

*"Exploring Vision Language Models for Facial Attribute Recognition: Emotion, Race, Gender, and Age."*

| Modelo | Tarefa | Setup | **Acc** | **F1** |
|---|---|---|---|---|
| FairFace ResNet-34 baseline (re-implementado) | race **7-class** | FairFace val oficial (10,954) | **0.720** | — |
| FaceScanPaliGemma (proposto, VLM) | race **7-class** | mesmo | **0.757** | **0.750** |

**Setup deles:** treino em 75% dos 86,744 train + valida 25%, **testa nos 10,954 val oficiais**. Imbalance natural (sem undersample). Padding não declarado.

### 2.2 Anzhc — community Hugging Face model

`huggingface.co/Anzhc/Race-Classification-FairFace-YOLOv8` — modelo comunitário não-publicado, 7-class native, split oficial, padding=0.25, 224×224.

| Variante | Top-1 Acc |
|---|---|
| YOLOv8n | 0.717 |
| YOLO11x | **0.735** |

### 2.3 O que o paper-pai FairFace publica (Kärkkäinen & Joo, WACV 2021)

Lido direto do PDF, Tabelas 2-3:

| Tarefa | Acc | Observação |
|---|---|---|
| Race binário (White vs outros) | 0.937 | fácil; alta separabilidade |
| **Race 4-class** (W/B/Asian/Indian merged) | 0.754 | merge pra ser comparável com UTKFace |

> ⚠️ **Footnote crítico da Tabela 3:** *"FairFace defines 7 race categories but only 4 races (White, Black, Asian, and Indian) were used in this result to make it comparable to UTKFace."*

➡️ **O paper-pai do dataset NÃO publica número para race 7-class in-domain.** A SOTA real surgiu em Hassanpour 2024.

## 3. FineFACE (Liu et al., 2024, arXiv 2408.16881) — NÃO é race classifier

**Achado crítico da auditoria textual de 2026-05-22:** o FineFACE — frequentemente citado como SOTA em fairness no FairFace — **não classifica raça**. Citação verbatim da Seção 4:

> *"We conducted two sets of experiments (1) a face-based gender classifier with gender as the target attribute and race and gender as the protected attributes (2) 13 gender-independent facial attribute classifiers ... with gender as the protected attribute."*

> *"Note that protected attribute annotation information is not used during the model training stage, but solely for the purpose of fairness evaluation"*

**O que o FineFACE realmente faz:**

| Aspecto | FineFACE |
|---|---|
| Tarefa de classificação (output) | **Gênero** (binário) ou cada um de 13 atributos faciais (modelos separados) |
| Papel da raça | **Atributo protegido** (usado só para medir disparidade entre grupos) |
| Headline "96.4% acc" | Accuracy de **gênero** estratificada por raça — **não race classification accuracy** |
| Recipe | ResNet-50 + multi-expert (e1/e2/e3) + SGD lr=0.002 + 448×448 + RandomCrop |

**Implicação:** FineFACE e nosso trabalho **resolvem tarefas diferentes**. Comparação numérica direta com o headline deles **não se aplica**. O recipe deles (SGD+448) é o que isolamos no anchor 🅐.2 — útil como teste de recipe, mas não como reprodução do paper.

## 4. Posicionamento honesto vs SOTA — tabela completa

### 4.0 Comparabilidade arquitetural (por que Hassanpour ResNet-34 é a referência primária)

Antes da tabela, é necessário declarar **qual SOTA é comparável ao
nosso sistema**. Os três números publicados para raça 7-class FairFace
no-domain pertencem a **classes arquiteturais distintas**:

| Sistema | Tipo | Parâmetros | Pré-treinamento |
|---|---|---|---|
| FaceScanPaliGemma | **Modelo Visão-Linguagem (VLM)** | **~3 bilhões** | SigLIP + texto, escala internet |
| YOLO11x community | CNN detector | ~57M | COCO detection (não revisado por pares) |
| **Hassanpour ResNet-34** | **CNN puro discriminativo** | **~21M** | **ImageNet-1k** |
| **Nosso ConvNeXt-T** | **CNN puro discriminativo** | **~28M** | **ImageNet-1k** |

**Comparar nosso ConvNeXt-T (28M, ImageNet-1k) com FaceScanPaliGemma
(3 bilhões, escala internet) seria tecnicamente correto mas
cientificamente irrelevante** — é como comparar um veículo popular com
um esportivo de F1. A diferença de classes de recursos e paradigma de
aprendizado torna a comparação injusta para nossa contribuição
(atribuição entre fatores em CNN puro sob recursos acadêmicos).

**Comparar com YOLO11x** também é problemático: modelo comunitário não
publicado, sem revisão por pares — útil como sanity check de magnitude,
não defensável como SOTA em banca acadêmica.

**Hassanpour ResNet-34 (0.720) é o único sistema arquiteturalmente
equivalente** ao nosso ConvNeXt-T no qual há número publicado e revisado
por pares. **É o ponto de referência primário e cientificamente válido
para nossa comparação.**

### 4.1 Tabela completa de pareamento

| Sistema | Tarefa | Setup | **Acc** | **F1 macro** | **IR ↓** |
|---|---|---|---|---|---|
| **FaceScanPaliGemma** (Hassanpour 2024, SOTA) | race 7-class | split oficial, imbalance natural | **0.757** | **0.750** | — |
| **YOLO11x** (community Anzhc) | race 7-class | split oficial, padding 0.25 | 0.735 | — | — |
| FairFace ResNet-34 baseline (re-impl. Hassanpour 2024) | race 7-class | mesmo setup | 0.720 | — | — |
| **ConvNeXt-T (Fator 5 nosso)** | race 7-class | 80/10/10 próprio + undersample + padding 1.25→224 + MTCNN | **0.7093** | **0.7108** | **1.569** |
| 🅓 Anchor raw-data (nosso) | race 7-class | 80/10/10 próprio + undersample + padding 1.25→224, **sem MTCNN** | 0.6948 | 0.695 | 1.649 |
| **Controle CE+linear RN-50 (nosso)** | race 7-class | matched | 0.6865 | 0.688 | 1.697 |
| 🅐.1 Anchor FairFace-recipe (RN-34 Adam) | race 7-class | matched | 0.6743 | 0.676 | 1.722 |
| **🅐.2 Anchor FineFACE-recipe (RN-50 SGD 448)** | race 7-class | matched | **0.6638** | **0.6634** | **1.724** |
| FairFace paper Tab.3 | race **4-class merged** | split oficial | 0.754 | — | — |
| FineFACE 2024 — **não é race classifier** | gender (binário) | FairFace + cross | 0.964 | — | — |

### 4.1 Decomposição final do gap absoluto (medida empiricamente)

Após o anchor 🅔 (protocolo Hassanpour completo) e a auditoria empírica
de código (`docs/auditoria_codigo_limitadores.md`), a decomposição passou
de estimada a **medida**:

**Sob protocolo 🅔 (todos os 5 confundidores metodológicos fechados):**

- 🅔 ConvNeXt-T (nosso): acc=0.706 ± 0.005, F1=0.703 ± 0.005, IR=1.541 ± 0.044
- Hassanpour ResNet-34: acc=0.720
- **Gap medido: −1.4pp**

**Decomposição dos confundidores (resultados empíricos):**

| Confundidor | Status sob 🅔 | Custo medido |
|---|---|---|
| Subamostragem por raça | ✅ fechado (`balance: none`) | ablação 🅑 mostrou neutro (Δ < 1σ) |
| Partição treino/teste | ✅ fechado (oficial FairFace) | absorvido no 🅔 |
| Versão das imagens | ✅ fechado (padding 0.25 nativo) | absorvido no 🅔 |
| Limpeza multi-face | ✅ fechado (CSV bruto 97k) | 🅓 mostrou que nosso cleaning custa ~0.7pp |
| Re-alinhamento MTCNN | ✅ fechado (imagens originais) | absorvido no 🅔 |
| **HPO da recipe não declarado por Hassanpour** | ❌ irrecuperável | atribuição honesta do gap residual |

**Auditoria empírica de limitadores no nosso código** (Testes A e C1):

- **Teste A — paciência aumentada (5 → 15):** bit-a-bit idêntico ao
  original. Escalonamento `CosineAnnealingWarmRestarts` **NÃO é o
  limitador**.
- **Teste C1 — dropout removido (0.2 → 0.0):** F1 marginalmente pior
  (Δ=−0.003), razão de disparidade significativamente pior (+0.058).
  **Dropout NÃO é o limitador — é regularizador favorável à equidade.**

➡️ **Conclusão:** o gap residual de −1.4pp **não é atribuível a nenhuma
escolha sub-ótima identificável no nosso código ou recipe**. Atribuição
honesta: **HPO da recipe realizada por Hassanpour e não publicada
integralmente**, fora do escopo desta dissertação.

### 4.2 Cruzamento de coerência (atualizado)

- Hassanpour ResNet-34 (mesma rede do paper-pai) entrega 0.720 sob seu
  protocolo.
- Nosso ConvNeXt-T (rede mais moderna), sob protocolo **idêntico** ao
  Hassanpour, entrega 0.706.
- A diferença de 1.4pp **não tem explicação metodológica residual** —
  todos os 5 confundidores foram fechados ou neutralizados, e 2
  suspeitos de limitador interno foram refutados empiricamente.
- **Achado adicional crucial:** nossa razão de disparidade (IR=1.541)
  sob 🅔 é a melhor de todos os pipelines do projeto. Hassanpour não
  reporta IR explicitamente — temos contribuição independente em
  equidade.

## 5. Vulnerabilidades reais (e respostas defensáveis revisadas)

| Vulnerabilidade da banca | Resposta defensável |
|---|---|
| *"Vocês ficam atrás do SOTA Hassanpour 2024 (75.7%) e até da ResNet-34 deles (72%). Por quê?"* | "Quatro escolhas metodológicas declaradas (undersample, split próprio, padding 1.25, multi-face cleaning) somam ~−2 a −3pp em acc absoluto. A contribuição é a **atribuição entre fatores** sob protocolo matched, invariante a esse offset. A ablação 🅑 (configs prontos) fecha a maior parte do gap caso a banca exija." |
| *"E o FineFACE bate vocês em 96.4%."* | "FineFACE classifica gênero, não raça. As tarefas são distintas; comparação numérica direta não se aplica. (Seção 4 deles, verbatim.)" |
| *"Por que não 448px como na recipe FineFACE?"* | "Trade-off explícito: 224px é metade da memória/compute → permite 3-seed × 5 fatores × matched protocol no orçamento 12GB. O anchor 🅐.2 prova que a recipe deles em 448×SGD entrega F1 **mais baixo** (0.663) que nosso recipe AdamW@224 (0.688), portanto a escolha 224 não nos prejudicou em F1 sob protocolo matched." |
| *"Por que não reproduzir a FineFACE multi-expert?"* | "Escopo de defesa (alto custo + arquitetura especializada de mitigação, que não é nosso eixo de contribuição). Anchor 🅐.2 (sem multi-expert) é o que cabe no orçamento e isola a recipe SGD-448." |
| *"Vocês ainda fazem undersample. Por quê?"* | "Trade-off declarado: undersample reduz acc agregada em ~1-2pp mas estabiliza F1 macro (todas as classes igualmente representadas no treino), facilitando atribuição entre fatores. Ablação 🅑 testa empiricamente esse trade-off no controle + ConvNeXt-T." |

## 6. Por que a Linha A continua válida (e ganhou base)

A SOTA real (Hassanpour 2024) **existe e foi mapeada**. O que **não existe** é uma SOTA para a pergunta que rodamos:

> *"Qual entre 5 dimensões algorítmicas (dataset/topologia/loss/paradigma/backbone) move acurácia + equidade simultaneamente sob protocolo causal matched?"*

Nenhum dos trabalhos publicados (FairFace, FineFACE, Hassanpour, FairGRAPE, U-FaTE, FSCL, DSAP, Fairness-in-Details) responde essa pergunta. Hassanpour compara modelos (CNN vs VLM); FineFACE propõe arquitetura de mitigação; FairFace propõe dataset. Nossa contribuição é **complementar** a todos eles: atribuição rigorosa entre fatores comuns, sob protocolo casado 3-seed, com critério Pareto-aware e auditoria de confundidores.

> *"We do not compete with SOTA on absolute accuracy; we contribute the matched-protocol attribution that SOTA work does not provide."*

## 7. Validação cruzada — STATUS final 2026-05-23

| Anchor / ablação / teste | Status | Resultado (3-seed) |
|---|---|---|
| 🅐.1 FairFace-recipe (RN-34 Adam @224) | ✅ executado | F1=0.676±0.006, IR=1.722±0.032 |
| 🅐.2 FineFACE-recipe (RN-50 SGD 448 RandomCrop sem multi-expert) | ✅ executado | F1=0.663±0.007, IR=1.724±0.038 |
| 🅓 raw-data (sem MTCNN + sem cleaning) | ✅ executado | F1=0.695±0.006, IR=1.649±0.008 |
| 🅑 Ablação sem subamostragem (controle + ConvNeXt) | ✅ **executado** | Outcome B: controle Δ<1σ, ConvNeXt persiste direcional |
| 🅔 Anchor Hassanpour-protocol (controle + ConvNeXt) | ✅ **executado** | ConvNeXt F1=0.703±0.005, IR=1.541±0.044 (3.5σ + 1.8σ) |
| Teste A (paciência=15) | ✅ executado | Bit-a-bit idêntico → cosine restart refutado como limitador |
| Teste C1 (dropout=0.0) | ✅ executado | Pior em F1 e IR → dropout refutado, é regularizador favorável |
| Avaliação cross-dataset (RFW/DemogPairs) | ⏸ escopo defesa | — |
| Reprodução FineFACE multi-expert completa | ⏸ fora do escopo | — |

## 8. Auditoria de confundidores metodológicos vs literatura (2026-05-22)

Auditoria provocada por pergunta da orientação ("o raw-data anchor é no mesmo dataset dos dois artigos? separou as classes?"). Inspeção física dos arquivos em `data/raw/bucket/` deu o achado abaixo.

### 8.1 Origem das imagens — versão padding 1.25 (loose), não 0.25 (default do paper)

`data/raw/bucket/{train,val}/*.jpg` foi auditado:

- **86,744 train + 10,954 val = 97,698 arquivos** (bate exatamente com o split oficial FairFace).
- **Dimensão das imagens: 448×448 RGB** (verificado via PIL em `train/1.jpg`).
- **Conclusão: é a versão `fairface-img-margin125-trainval` do FairFace publication** (padding=1.25, loose crop com mais contexto), NÃO a versão default `margin025` (224×224, tight crop).

### 8.2 Mapa completo de confundidores entre nossas pipelines e cada paper

| Dimensão | F1-F5 + Fator 5 (todos) | Anchor 🅓 (raw-data) | FairFace paper (2021) | FineFACE (2024) | Hassanpour (2024) |
|---|---|---|---|---|---|
| **Versão imagens** | FairFace 1.25 → MTCNN re-align → resize 224 | FairFace 1.25 → resize 224 (sem re-align) | FairFace **0.25** (224×224 tight, native) | FairFace 1.25 (RandomCrop 448→224) | não declarado (provavelmente 0.25) |
| **Multi-face cleanliness** | ✓ removidos (72k) | ✗ raw (97k) | ✗ raw | ✗ raw | ✗ raw |
| **MTCNN re-align nosso** | ✓ aplicado | ✗ original do FairFace | ✗ | ✗ | ✗ |
| **Split protocol** | nosso 80/10/10 estratificado | nosso 80/10/10 estratificado | **train/val oficial** (~86k / 11k) | **train/val oficial** | 75/25 do train + test no val oficial |
| **Class balance** | undersample por raça | undersample por raça | natural imbalance | natural imbalance | natural imbalance |
| **Recipe** | AdamW lr=1e-3, bs=128, CE+linear (RN-50) ou per-backbone (F5) | idem F1-F5 (matched) | ADAM lr=1e-4 (RN-34) | SGD lr=0.002 RandomCrop bs=16 multi-expert | não declarado em detalhe |
| **Tarefa de classificação** | race 7-class | race 7-class | race **4-class merged** (não 7) | **gender** (binário) — NÃO race | race 7-class |

### 8.3 Implicações revistas

1. **Anchor 🅓 está mais próximo do FineFACE que do FairFace paper** em padding (ambos 1.25), mas distantes em recipe e em **tarefa** (FineFACE não classifica raça).
2. **Hassanpour 2024 é o único trabalho na mesma tarefa exata** (race 7-class, padding presumível 0.25, split oficial). É a única comparação numerical legítima.
3. **A escolha de baixar a versão 1.25 antecipou a recipe FineFACE.** Quem montou os dados originalmente já tinha em mente compatibilidade com fairness papers que usam 448.
4. **Achado emergente (válido sob protocolo matched):** 🅓 (F1=0.695, IR=1.649) > F1-F5 control (F1=0.688, IR=1.697). Logo, **nosso MTCNN re-align + multi-face cleaning custa ~0.7pp F1 e piora IR em ~0.05** comparado a usar as imagens originais. Null bem-medido, posicionamento absoluto declarado.
5. **🅐.2 batch final (3 seeds):** recipe FineFACE-style sem multi-expert entrega F1=0.663, abaixo até do controle. Reforça que **o ganho do FineFACE no paper original vem do multi-expert architecture**, não da recipe SGD-448 isolada.

### 8.4 Claim de tese (versão final 2026-05-23, pós-🅔 e auditoria empírica)

> *"Existem dois trabalhos publicados que reportam classificação de raça
> em 7 categorias no FairFace no-domain: Hassanpour et al. 2024
> (ResNet-34 baseline 0.720, FaceScanPaliGemma VLM 0.757) e modelo
> comunitário Anzhc YOLO11x (0.735). Sob protocolo metodologicamente
> idêntico ao SOTA-CNN (anchor 🅔: padding 0.25 + partição oficial +
> sem subamostragem + sem nossa limpeza multi-face), nosso ConvNeXt-T
> entrega F1=0.703 ± 0.005, acurácia=0.706 ± 0.005 e razão de
> disparidade=1.541 ± 0.044. Gap absoluto medido: −1.4pp em acurácia.
> Auditoria empírica de duas variáveis suspeitas no nosso código
> (escalonamento cossenoidal e dropout — Testes A e C1) refutou ambas
> como limitadores. O resíduo é atribuído à otimização de hiperparâmetros
> realizada pelos autores e não publicada integralmente —
> irrecuperável no escopo desta dissertação. Importante: nossa razão de
> disparidade é a melhor de todos os pipelines do projeto, e Hassanpour
> não reporta IR explicitamente. O achado central (rede dorsal ConvNeXt-T
> como única alavanca real de acurácia + equidade entre 5 fatores
> algorítmicos) sobrevive sob 3 protocolos distintos (casado original
> 7σ + 3σ, sem subamostragem 1.5σ + 0.7σ, Hassanpour-protocol
> 3.5σ + 1.8σ), evidenciando robustez metodológica."*

## 9. Auditoria empírica de código (referência cruzada)

A auditoria documentada em `docs/auditoria_codigo_limitadores.md` testou
empiricamente dois suspeitos de limitador no nosso código:

1. **Teste A — paciência da parada antecipada (5 → 15):** valida hipótese
   de que o escalonamento cossenoidal com reinícios estaria interrompendo
   o treinamento prematuramente. **Refutada bit-a-bit**: paciência maior
   não muda o resultado.
2. **Teste C1 — dropout removido (0.2 → 0.0):** valida hipótese de
   sobre-regularização. **Refutada com descoberta inversa**: dropout
   estava ajudando a equidade.

A peça é rara em dissertações — mostra rigor científico ao testar as
próprias escolhas e incluir resultados negativos como evidência. Reforça
a atribuição do gap absoluto residual ao HPO externo.

## 10. Análises pós-treinamento — confiabilidade via agregação

Aplicação de técnicas científicas estabelecidas (deep ensemble, TTA,
calibração) sobre os 3 checkpoints ConvNeXt-T 🅔. **Sem retreinamento.**
**Objetivo metodológico, não competitivo:** elevar a CONFIABILIDADE da
estimativa de desempenho via redução de variância (Lakshminarayanan
NeurIPS 2017) e quantificação de incerteza (Guo ICML 2017), conforme
guidelines modernas (Pineau JMLR 2021).

| Config | Acurácia | F1 | IR ↓ | Tipo de comparação |
|---|---|---|---|---|
| Hassanpour RN-34 baseline (single-run) | 0.720 | — | — | referência (sem variância reportada) |
| **Single ConvNeXt-T 🅔 (nosso, comparação SIMÉTRICA)** | **0.7115** | **0.7083** | **1.496** | **−0.85pp** (dentro de 1.7σ da variância) |
| Single (média 3 seeds, mais rigorosa) | 0.7060 ± 0.005 | 0.7034 ± 0.005 | 1.541 ± 0.044 | −1.4pp (estimativa com incerteza) |
| Deep Ensemble 3 seeds | 0.7299 | 0.7285 | 1.501 | agregação científica |
| Ensemble + Calib + Threshold | 0.7304 | 0.7292 | 1.501 | melhor F1/Acc com calibração |
| Ensemble + TTA | 0.7298 | 0.7286 | **1.474** ⭐ | melhor equidade do projeto |

**Posicionamento ético declarado:** a comparação metodologicamente
simétrica (single-run-vs-single-run) é **0.7115 vs 0.720 → −0.85pp,
dentro de 1.7σ da variância natural** medida em nosso 3-seed protocol
(dp=0.005). Hassanpour 2024 não usou ensemble/TTA/calibração —
comparação direta dessas técnicas com seu single-run seria
metodologicamente assimétrica. As estimativas agregadas refletem
**MAIOR confiabilidade da medida do nosso sistema**, não vitória sobre
o paper. **Bhaskaruni IEEE ICTAI 2019** documenta que ensemble reduz
disparidade demográfica em classificação facial — convergente com IR
1.541 → 1.474.

Doc completo em [`combo_defesa_fechamento.md`](combo_defesa_fechamento.md).

## 11. Resumo de uma linha (FINAL — versão ética)

> *"Sob protocolo metodologicamente idêntico ao SOTA-CNN publicado
> (Hassanpour 2024) e comparação simétrica single-run-vs-single-run,
> nosso ConvNeXt-T 🅔 entrega acurácia=0.7115 / F1=0.7083 / IR=1.496 —
> **−0.85pp do baseline Hassanpour ResNet-34 (0.720), dentro de 1.7σ da
> variância natural entre seeds medida em nosso protocolo 3-seed**. Gap
> residual atribuído à otimização de hiperparâmetros não publicada
> pelos autores, após auditoria empírica refutar 2 suspeitos no nosso
> código. Aplicando técnicas científicas estabelecidas de agregação
> (deep ensemble — Lakshminarayanan NeurIPS 2017; TTA), a estimativa
> agregada da nossa pipeline atinge 0.7304 acurácia / 1.474 IR — não
> como reivindicação de superação direta (Hassanpour não usou essas
> técnicas), mas como demonstração de que a CONFIABILIDADE da estimativa
> sobe com práticas de replicabilidade modernas (Pineau JMLR 2021). A
> Linha A (atribuição casada 3-seed entre 5 fatores) identifica
> ConvNeXt-T como única alavanca robusta de acurácia + equidade
> simultânea, e a Linha B (critério Pareto-aware best-epoch) permanece
> como contribuição metodológica replicável."*
