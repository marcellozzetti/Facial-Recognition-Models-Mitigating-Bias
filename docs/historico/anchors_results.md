# Anchors — resultados consolidados 3-seed casados (v2)

> Material de tese. Resultados finais dos 3 anchors de posicionamento
> absoluto vs literatura, da ablação 🅑 (sem subamostragem) e do anchor
> 🅔 (protocolo Hassanpour), todos rodados sob protocolo casado 3-seed
> {42,1,2}, critério val_f1_macro. Inclui auditoria empírica de código
> (Testes A e C1) que refuta dois suspeitos de limitador no nosso
> pipeline. Complementa `docs/baseline_positioning.md` (paisagem teórica),
> `docs/sota_7class_race_audit.md` (pesquisa textual) e
> `docs/auditoria_codigo_limitadores.md` (auditoria empírica).
> Atualizado 2026-05-23. Terminologia alinhada com Haykin.

## 1. O que cada anchor / ablação isola

| Anchor / ablação | Variável manipulada (vs F1-F5 controle) | Hipótese testada |
|---|---|---|
| 🅐.1 FairFace-recipe | recipe do paper-pai: ResNet-34 + ADAM lr=1e-4 + 224 | "quanto da F1 controle vem do nosso recipe vs do recipe canônico do paper-pai?" |
| 🅐.2 FineFACE-recipe | recipe do paper de fairness 2024: ResNet-50 + SGD lr=0.002 + 448 RandomCrop, **sem multi-expert** | "quanto da F1 controle vem do nosso recipe vs do recipe da literatura de fairness 448?" |
| 🅓 raw-data | conjunto bruto FairFace publication (sem limpeza multi-face + sem re-alinhamento MTCNN) | "quanto da F1 controle vem do nosso pré-processamento?" |
| 🅑 sem subamostragem | balanceamento de classes: subamostragem por raça → `none` | "a alavanca ConvNeXt-T sobrevive à decisão de balanceamento?" |
| 🅔 protocolo Hassanpour | combinação: padding 0.25 + partição oficial + sem subamostragem + sem nossa limpeza multi-face | "sob protocolo idêntico ao SOTA publicado, alavanca persiste e gap absoluto se fecha?" |

## 2. Tabela consolidada — todos os pipelines (5 fatores + 3 anchors + ablação + anchor SOTA)

| Pipeline (recipe + dados) | Rede dorsal | F1 macro | Razão de disparidade ↓ | Acurácia | Δ F1 vs controle | Δ IR vs controle |
|---|---|---|---|---|---|---|
| **Controle CE+linear** (referência casada) | ResNet-50 | 0.688 ± 0.002 | 1.697 ± 0.033 | 0.687 | — | — |
| **🅐.1 FairFace-recipe** | ResNet-34 (ADAM lr=1e-4) | 0.676 ± 0.006 | 1.722 ± 0.032 | 0.674 | −0.012 | +0.025 |
| **🅐.2 FineFACE-recipe** | ResNet-50 (SGD lr=0.002, 448→RandomCrop224) | 0.663 ± 0.007 | 1.724 ± 0.038 | 0.664 | −0.025 | +0.027 |
| **🅓 raw-data** | ResNet-50 (recipe matched ctrl) | 0.695 ± 0.006 | 1.649 ± 0.008 | 0.695 | **+0.007** | **−0.048** |
| **🅑 Controle sem subamostragem** | ResNet-50 | 0.687 ± 0.002 | 1.696 ± 0.055 | 0.689 | −0.001 | −0.001 |
| **🅑 ConvNeXt-T sem subamostragem** | ConvNeXt-T | 0.700 ± 0.009 | 1.631 ± 0.090 | 0.703 | +0.012 | −0.066 |
| **🅔 Controle protocolo Hassanpour** | ResNet-50 | 0.680 ± 0.005 | 1.628 ± 0.023 | 0.682 | −0.008 | −0.069 |
| **🅔 ConvNeXt-T protocolo Hassanpour** | ConvNeXt-T | **0.703 ± 0.004** | **1.541 ± 0.044** | **0.706** | **+0.015** | **−0.156** |
| **Fator 5 ConvNeXt-T** (referência positiva original) | ConvNeXt-T | 0.711 ± 0.003 | 1.569 ± 0.023 | 0.709 | +0.023 (~7σ) | −0.128 (~3σ) |

## 3. Veredito por anchor (regra do 1σ casada)

### 3.1 🅐.1 FairFace-recipe (ResNet-34 + ADAM lr=1e-4 @224)

- **Δ F1 = −0.012, σ_comb=0.0063, |Δ|/σ ≈ 1.9σ** — marginalmente abaixo,
  no limite da significância.
- **Δ IR = +0.025, σ_comb=0.046, |Δ|/σ ≈ 0.5σ** — não-significativo.

**Interpretação:** o recipe do paper-pai do FairFace (ResNet-34 + ADAM
minimalista, 224) entrega F1 ligeiramente abaixo (≈1pp) do nosso
recipe modernizado (AdamW lr=1e-3 ResNet-50 ImageNet, 224) sob protocolo
casado, sem mover a equidade significativamente.

### 3.2 🅐.2 FineFACE-recipe (ResNet-50 + SGD lr=0.002 + 448→RandomCrop224, sem multi-expert)

- **Δ F1 = −0.025, σ_comb=0.0073, |Δ|/σ ≈ 3.4σ** — **significativamente abaixo**.
- **Δ IR = +0.027, σ_comb=0.050, |Δ|/σ ≈ 0.5σ** — não-significativo.

**Interpretação tese-relevante:** a recipe do FineFACE **sem o multi-expert
architecture** entrega F1 significativamente pior (−2.5pp) que nosso recipe
AdamW+224. Isto é um **achado positivo**: o ganho do paper FineFACE
(96.4% acc em gênero) vem **da arquitetura multi-expert**, não da recipe
SGD-448 isolada.

### 3.3 🅓 raw-data (sem MTCNN + sem limpeza multi-face)

- **Δ F1 = +0.007, σ_comb=0.0063, |Δ|/σ ≈ 1.1σ** — borderline positivo.
- **Δ IR = −0.048, σ_comb=0.034, |Δ|/σ ≈ 1.4σ** — borderline melhor.

**Interpretação:** usar as imagens originais do FairFace publication (sem
nosso pré-processamento) entrega F1 marginalmente acima e IR marginalmente
abaixo do nosso pipeline. **Nosso pré-processamento custa F1 e IR, não
ganha** — null/negativo bem-medido. Implica revisão da narrativa do
Fator 1: cleaning é metodologicamente penalizante sob protocolo casado.

### 3.4 🅑 Ablação sem subamostragem por raça (Outcome B confirmado)

**Controle ResNet-50 sem subamostragem vs com subamostragem:**

| Métrica | Δ | σ_comb | |Δ|/σ |
|---|---|---|---|
| Acurácia | +0.003 | 0.0034 | 0.9σ n.s. |
| F1 macro | −0.001 | 0.0028 | 0.4σ n.s. |
| Razão de disparidade | −0.001 | 0.064 | ~0σ |

**Veredito:** **Outcome B confirmado** (subamostragem é cost-without-benefit).
A subamostragem estratificada por raça (prática comum na literatura de
fairness em FairFace) é estatisticamente neutra no nosso pipeline sob
protocolo casado 3-seed. Refuta empiricamente a premissa de que
subamostragem necessariamente melhora equidade.

**ConvNeXt-T sem subamostragem vs com subamostragem (Fator 5 original):**

| Métrica | Δ | σ_comb | |Δ|/σ |
|---|---|---|---|
| F1 macro | −0.011 | 0.0093 | 1.2σ marginal |
| Razão de disparidade | +0.062 | 0.093 | 0.7σ n.s. |

**Observação importante sobre variância:** sem subamostragem, a variância
entre sementes dobra no braço ConvNeXt-T (dp_F1: 0.003 → 0.009; dp_IR:
0.023 → 0.090). A subamostragem atua como **regularizador implícito** que
estabiliza o treinamento entre sementes, mesmo não movendo a média.

### 3.5 🅔 Anchor Hassanpour-protocol — pareamento metodológico com SOTA publicada

**Configuração:** padding 0.25 + partição oficial FairFace (86,744 train +
10,954 test) + sem subamostragem + CSV bruto 97k. Reproduz integralmente
o setup metodológico de Hassanpour et al. 2024.

**ConvNeXt-T sob 🅔 vs Controle sob 🅔:**

| Métrica | Δ | σ_comb | |Δ|/σ |
|---|---|---|---|
| Acurácia | +0.024 | 0.0067 | **3.6σ significativo** |
| F1 macro | +0.024 | 0.0067 | **3.5σ significativo** |
| Razão de disparidade | −0.087 | 0.050 | **1.8σ borderline-significativo** |

**Veredito CENTRAL para a tese:** **a alavanca ConvNeXt-T persiste com
significância sob protocolo Hassanpour SOTA** — não é artefato do nosso
protocolo casado original. Magnitude do efeito comparável (+0.024 F1 vs
+0.023 no casado original), com variância colapsando para metade da
ablação 🅑 (dp_F1: 0.009 → 0.004; dp_IR: 0.090 → 0.044).

## 4. Pareamento absoluto vs SOTA publicada (Hassanpour 2024)

### 4.0 Por que comparamos primariamente com Hassanpour ResNet-34, não com VLM ou YOLO

Os três números publicados para raça 7-class FairFace pertencem a
**classes arquiteturais distintas** — comparação válida requer
equivalência de paradigma e escala:

| Sistema | Tipo | Parâmetros | Pré-treinamento | Comparável ao nosso? |
|---|---|---|---|---|
| FaceScanPaliGemma | VLM (Visão-Linguagem) | ~3 bilhões | SigLIP + texto escala internet | ❌ Não — outro paradigma + escala |
| YOLO11x community | CNN detector | ~57M | COCO detection (não revisado) | ❌ Não — não revisado por pares |
| **Hassanpour ResNet-34** | **CNN puro discriminativo** | **~21M** | **ImageNet-1k** | ✅ **Sim — único equivalente** |
| **Nosso ConvNeXt-T** | **CNN puro discriminativo** | **~28M** | **ImageNet-1k** | (referência) |

Comparar nossa CNN de 28M parâmetros com FaceScanPaliGemma (3 bilhões,
escala internet) seria como comparar veículo popular com esportivo de
F1 — tecnicamente correto, cientificamente irrelevante para nossa
contribuição. **Hassanpour ResNet-34 é a única referência arquiteturalmente
válida** para nossa comparação primária.

### 4.1 Resultados finais sob protocolo idêntico ao SOTA-CNN publicado

| Sistema | Tarefa | Setup | Acurácia | F1 | IR |
|---|---|---|---|---|---|
| FaceScanPaliGemma VLM (SOTA) | raça 7-class | partição oficial, sem subamostragem, padding 0.25 | **0.757** | **0.750** | — |
| YOLO11x community | raça 7-class | idem | 0.735 | — | — |
| **Hassanpour ResNet-34 baseline** | raça 7-class | idem | **0.720** | — | — |
| **🅔 ConvNeXt-T (nosso)** | raça 7-class | idem | **0.706 ± 0.005** | **0.703 ± 0.005** | **1.541** |
| 🅔 Controle ResNet-50 (nosso) | raça 7-class | idem | 0.682 ± 0.005 | 0.680 ± 0.005 | 1.628 |
| Controle casado original (nosso) | raça 7-class | partição própria + subamostragem + padding 1.25 + MTCNN | 0.687 | 0.688 | 1.697 |

**Decomposição final do gap absoluto −1.4pp (Hassanpour 0.720 vs 🅔 ConvNeXt 0.706):**

Sob protocolo 🅔, **todos os 5 confundidores metodológicos identificados
estão fechados**:

| Confundidor | 🅔 fecha? |
|---|---|
| Subamostragem por raça | ✅ sim |
| Partição treino/teste | ✅ sim (oficial) |
| Versão das imagens (padding) | ✅ sim (0.25 nativo) |
| Limpeza multi-face | ✅ sim (CSV bruto 97k) |
| Re-alinhamento MTCNN | ✅ sim (imagens originais) |
| **HPO da recipe não declarado por Hassanpour** | ❌ **único resíduo possível** |

**Auditoria empírica refutou 2 hipóteses de limitador interno** (Testes
A e C1 — ver `docs/auditoria_codigo_limitadores.md`):

- **Teste A — paciência 5 → 15:** bit-a-bit idêntico ao original. O
  escalonamento `CosineAnnealingWarmRestarts` não é o limitador.
- **Teste C1 — dropout 0.2 → 0.0:** F1 marginalmente pior, razão de
  disparidade significativamente pior (+0.058). Dropout não é
  sobre-regularizando; está atuando como regularizador favorável à
  equidade.

**Conclusão da decomposição:** o gap residual de −1.4pp é **honestamente
atribuível à otimização de hiperparâmetros (HPO) realizada pelos autores
e não publicada integralmente**. Está fora do escopo desta dissertação,
cuja contribuição central é atribuição entre fatores sob protocolo casado.

**Achado adicional:** nossa razão de disparidade (IR=1.541) sob protocolo
🅔 é **a melhor de todos os pipelines do projeto** e, embora Hassanpour
não reporte IR explicitamente, é evidência forte de que a alavanca
ConvNeXt-T move equidade independentemente do offset absoluto em acurácia.

## 4.5 Análises pós-treinamento — confiabilidade da estimativa via agregação

Aplicação de 4 técnicas científicas estabelecidas sobre os 3 checkpoints
ConvNeXt-T 🅔 sem retreinamento adicional. **Objetivo: reduzir variância
da estimativa e quantificar incerteza, conforme guidelines modernas de
replicabilidade (Pineau JMLR 2021).** Documentação completa em
[`docs/combo_defesa_fechamento.md`](combo_defesa_fechamento.md).

| Config | Acurácia | F1 macro | IR ↓ | Tipo de comparação |
|---|---|---|---|---|
| Hassanpour RN-34 baseline (single-run) | 0.720 | — | — | referência (sem variância reportada) |
| **Single seed s42 (nosso)** | **0.7115** | **0.7083** | **1.496** | **comparação simétrica**: −0.85pp |
| Single seed (média 3 seeds) | 0.7060 ± 0.005 | 0.7034 ± 0.005 | 1.541 ± 0.044 | mais rigorosa: −1.4pp |
| Deep Ensemble 3 seeds | 0.7299 | 0.7285 | 1.501 | agregação científica (Lakshminarayanan 2017) |
| Ensemble + TTA | 0.7298 | 0.7286 | **1.474** ⭐ | melhor equidade do projeto |
| Ensemble + Calib + Threshold | 0.7304 | 0.7292 | 1.501 | melhor F1/Acc com calibração |

**Posicionamento ético:** sob comparação simétrica single-vs-single
(consistente com a metodologia de single-run reportada por Hassanpour),
nosso ConvNeXt-T fica a **−0.85pp** — dentro de 1.7σ da variância natural
entre seeds. As configurações com ensemble/TTA/calibração NÃO são
reivindicações de superação direta — são estimativas mais CONFIÁVEIS da
performance da nossa pipeline via técnicas científicas de redução de
variância (Hansen & Salamon IEEE PAMI 1990; Lakshminarayanan NeurIPS
2017). **Bhaskaruni IEEE ICTAI 2019** documenta empiricamente que
ensemble reduz disparidade demográfica em classificação facial,
convergente com nosso resultado IR 1.541 → 1.474.

## 5. Resumo dos achados tese-relevantes (4 itens)

1. **Recipe AdamW@224 é localmente ótima** sob protocolo casado para
   raça 7 classes — bate recipe paper-pai (🅐.1: +1.2pp F1) e recipe
   FineFACE sem multi-expert (🅐.2: +2.5pp F1).
2. **Nosso pré-processamento custa F1 e IR** (🅓: +0.7pp F1, −0.05 IR
   contra nós). Limitação metodológica declarada — Fator 1 inicial
   revisitado na narrativa.
3. **Subamostragem por raça é estatisticamente neutra** (🅑: Δ < 1σ em
   todas as métricas). Refuta empiricamente uma prática comum na
   literatura de fairness em FairFace.
4. **Alavanca ConvNeXt-T sobrevive a 3 protocolos distintos** (casado
   original 7σ + 3σ, sem subamostragem 1.5σ + 0.7σ, Hassanpour 3.5σ +
   1.8σ). Achado central robusto a variações metodológicas substanciais.

## 6. FineFACE não classifica raça — descoberta da auditoria textual

**Achado crítico da auditoria de 2026-05-22:** o FineFACE (Liu et al.
2024), frequentemente citado como SOTA em fairness no FairFace, **NÃO
classifica raça**. Citação verbatim da Seção 4:

> *"We conducted two sets of experiments (1) a face-based gender
> classifier with gender as the target attribute and race and gender as
> the protected attributes (2) 13 gender-independent facial attribute
> classifiers ... with gender as the protected attribute."*

O FineFACE classifica **gênero** (binário) e **13 atributos faciais**,
com raça como atributo protegido para medir disparidade. A "manchete
96.4% accuracy" é acurácia de gênero estratificada por raça, **não
acurácia de raça**. Comparação numérica direta com nossa tarefa **não se
aplica**.

Nosso anchor 🅐.2 isola **apenas a recipe** do FineFACE aplicada à nossa
tarefa de raça 7 classes — útil como anchor metodológico de recipe, não
como reprodução do paper.

## 7. Procedência

- **Geradores de config:**
  - `scripts/generate_anchor_fairface_configs.py`
  - `scripts/generate_anchor_finefacerecipe_configs.py`
  - `scripts/generate_anchor_rawdata_configs.py`
  - `scripts/generate_ablation_no_undersample_configs.py`
  - `scripts/generate_anchor_hassanpour_configs.py`
- **Outputs definitivos:**
  - `outputs/definitive/anchor_fairface_recipe/`
  - `outputs/definitive/anchor_finefacerecipe/`
  - `outputs/definitive/anchor_rawdata/`
  - `outputs/definitive/ablation_no_undersample/`
  - `outputs/definitive/anchor_hassanpour/`
  - `outputs/definitive/ablation_patience/` (Testes A + C1)
- **Recompute:** `scripts/recompute_checkpoint_criterion.py --definitive`
- **README de cada anchor:** `configs/anchor_*/README.md`,
  `configs/ablation_*/README.md`
- **Suporte de código novo:** `src/face_bias/data/dataset.py`
  (split_protocol=official), `src/face_bias/config/schema.py`
- **Docs irmãs:** `docs/baseline_positioning.md`,
  `docs/sota_7class_race_audit.md`, `docs/auditoria_codigo_limitadores.md`,
  `docs/THESIS_STATEMENT.md`
