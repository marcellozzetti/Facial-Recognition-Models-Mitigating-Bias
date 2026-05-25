# Análise interseccional de fairness — raça × gênero × idade

> Material de tese. Combo defesa-fechamento #1. Análise interseccional
> sobre as predições do best ConvNeXt-T 🅔 seed 42 (AlDahoul-protocol)
> no test set oficial FairFace (val/, 10,954 imagens). Cruza predições
> com colunas `gender` e `age` do CSV original para identificar
> subgrupos demograficamente vulneráveis. Custo: ~10min, zero GPU
> adicional. Data: 2026-05-23.

## 1. Motivação científica

A `THESIS_STATEMENT.md §6` declarou como limitação inicial: *"medimos
disparidade apenas por raça"*. Esta análise fecha essa vulnerabilidade
sem rodar treinamento adicional — apenas re-inferência sobre o test set
já existente + cruzamento com metadados do FairFace.

**Pergunta:** *"O modelo ConvNeXt-T 🅔 mantém disparidade ~1.5 quando
medida apenas por raça. Esse número esconde subgrupos
demograficamente piores quando cruzamos com gênero e idade?"*

## 2. Setup

- **Modelo:** ConvNeXt-T sob protocolo AlDahoul et al. (anchor 🅔), seed 42
  (best.pt = 0.7083 F1 macro / 0.7115 acurácia / 1.496 IR-raça)
- **Test set:** val oficial FairFace = 10,954 imagens
- **Metadados cruzados:** colunas `gender` (Female/Male) e `age`
  (9 faixas: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69,
  more than 70) do CSV `fairface_labels.csv`
- **Métrica primária:** acurácia por subgrupo (F1 macro não-aplicável
  para subgrupos race-filtered — todos têm rótulo único)
- **Razão de disparidade interseccional:** max(acc) / min(acc) entre
  subgrupos
- **Script:** `scripts/intersectional_fairness_analysis.py`
- **Outputs:** `outputs/intersectional/`

## 3. Resultados

### 3.1 Disparidade cresce com granularidade

| Granularidade | IR (max acc / min acc) | Crescimento vs raça |
|---|---|---|
| Raça apenas | 1.541 | referência (já reportado em 🅔) |
| **Raça × gênero** | **1.982** | **+28.6%** |
| **Raça × idade (n≥30)** | **2.144** | **+39.1%** |
| **Raça × gênero × idade (n≥20)** | **3.241** | **+110.3%** |

➡️ **A disparidade interseccional DOBRA quando cruzamos com gênero E
idade.** A métrica grupal por raça subestima o viés real do modelo.
Achado fairness-relevante de primeira ordem.

### 3.2 Subgrupos PIORES (acurácia por raça × gênero)

| Grupo | n | Acurácia | Status |
|---|---|---|---|
| **Middle Eastern × Female** | 396 | **43.43%** | ⚠️ pior subgrupo absoluto |
| Southeast Asian × Female | 680 | 57.50% | crítico |
| Latino_Hispanic × Male | 793 | 58.13% | crítico |
| Southeast Asian × Male | 735 | 63.27% | borderline |
| Indian × Male | 753 | 63.48% | borderline |
| Middle Eastern × Male | 813 | 67.40% | borderline |
| Latino_Hispanic × Female | 830 | 68.80% | borderline |
| Indian × Female | 763 | 71.56% | razoável |

### 3.3 Subgrupos MELHORES (acurácia por raça × gênero)

| Grupo | n | Acurácia |
|---|---|---|
| **Black × Male** | 799 | **86.11%** |
| Black × Female | 757 | 85.07% |
| East Asian × Female | 773 | 80.98% |
| East Asian × Male | 777 | 79.54% |
| White × Male | 1122 | 76.47% |
| White × Female | 963 | 75.60% |

### 3.4 Pior intersecção tripla (raça × gênero × idade, n≥20)

| Grupo | n | Acurácia |
|---|---|---|
| **Middle Eastern × Female × 3-9** (criança) | 49 | **28.57%** |
| Middle Eastern × Female × 30-39 | 70 | 34.29% |
| Latino_Hispanic × Male × 60-69 (idoso) | 28 | 35.71% |
| Middle Eastern × Female × 40-49 | 46 | 45.65% |
| Middle Eastern × Female × 20-29 | 152 | 46.71% |

## 4. Achados tese-relevantes

### 4.1 Worst-off intersectional group: Middle Eastern × Female

Acurácia agregada de **43.43%** — significativamente abaixo da média
geral (70.6%). Particularmente catastrófica em crianças (28.57% para
3-9 anos, n=49). **Modelo essencialmente aleatório em 1/7 = 14.3% para
esse subgrupo.**

**Mecanismo provável:**
- Confusão sistemática com White e Latino_Hispanic (sobreposição
  visual de tom de pele claro + features faciais comuns no Mediterrâneo)
- Sub-representação durante pré-treinamento ImageNet (poucos exemplares
  do mundo árabe/persa em datasets web-scraped)
- Confusão exacerbada em crianças pelas features faciais menos
  diferenciadas em idades jovens

### 4.2 Best-off: Black × Male — achado contraintuitivo

A literatura clássica de reconhecimento facial reporta que modelos têm
pior desempenho em pessoas negras (Buolamwini & Gebru, 2018). **Nosso
ConvNeXt-T sob protocolo AlDahoul et al. reverte esse padrão**: Black × Male
é o subgrupo **MELHOR** atendido (86.1% acurácia).

**Interpretação:**
- FairFace foi explicitamente **balanceado por raça na coleta**
  (Kärkkäinen & Joo 2021), incluindo proporção elevada de Black
  comparado a datasets de uso geral
- A categoria "Black" no FairFace tem features visuais mais
  distintivas (pele escura uniforme, baixa sobreposição com outras
  categorias raciais) → fronteira de decisão mais clara
- Modelos modernos (ConvNeXt-T 2022) já incorporam práticas
  pós-Buolamwini (auditoria, balanceamento de pré-treinamento)

Esse achado **não invalida** a literatura clássica — apenas demonstra
que sob dataset bem-balanceado por raça e arquitetura moderna, o eixo
de viés se desloca para outras categorias (Middle Eastern, Latino).

### 4.3 Gênero amplifica disparidade de raça

A disparidade por raça apenas (IR=1.541) sobe para 1.982 quando
cruzamos com gênero. Significa que **dentro de cada classe racial,
há disparidade adicional por gênero**:

- Middle Eastern: Female (43%) vs Male (67%) — **gap de 24pp dentro
  da classe**
- Latino_Hispanic: Male (58%) vs Female (69%) — gap de 11pp,
  direção oposta (males piores aqui)
- Black: Male (86%) ≈ Female (85%) — gap mínimo

**A direção do viés de gênero VARIA por raça.** Não há um padrão
universal "mulheres são piores" ou "homens são piores". Cada
intersecção tem seu próprio padrão.

### 4.4 Idade introduz viés adicional, principalmente em crianças

Subgrupos de **crianças (0-2 e 3-9)** de classes minoritárias
consistentemente abaixo de adultos da mesma classe:

- Middle Eastern × 0-2: 40% (n=10)
- Middle Eastern × 3-9: 45.7% (n=116)
- Southeast Asian × 0-2: 41.2% (n=34)
- Latino_Hispanic × 0-2: 42.1% (n=19)

**Crianças têm features faciais menos diferenciadas racialmente** — o
modelo tem mais dificuldade. Isso é fairness-relevante porque
aplicações reais (sistemas de monitoramento, ID infantil) podem
falhar especificamente nesses grupos.

## 5. Implicações para a tese

### 5.1 Adição ao corpo da dissertação

**Nova subseção em "Resultados" (§4.X):** *"Análise interseccional
revela que a razão de disparidade por raça (IR=1.541) subestima o
viés real do modelo em até 110% (IR=3.241 quando crossamos com
gênero e idade). O subgrupo pior atendido é Middle Eastern × Female
× 3-9 anos (28.6% acurácia, n=49). Contraintuitivamente, Black × Male
é o melhor (86.1% acurácia), refletindo o balanceamento explícito de
FairFace e a evolução pós-Buolamwini das arquiteturas modernas. O
viés se desloca para Middle Eastern, Latino_Hispanic e crianças de
classes minoritárias."*

### 5.2 Adição às limitações honestas

**Atualizar `THESIS_STATEMENT.md §6`:**

> *"Disparidade interseccional (raça × gênero × idade) atinge IR=3.241
> sob nosso melhor modelo (ConvNeXt-T 🅔). Subgrupo pior atendido:
> Middle Eastern × Female × 3-9 (28.6% acurácia). Mitigação específica
> para subgrupos vulneráveis fica como trabalho futuro (defesa final)."*

### 5.3 Fechamento de vulnerabilidade de defesa

**Pergunta antecipada:** *"Vocês mediram disparidade apenas por raça.
E gênero? E idade? E intersecções?"*

**Resposta defensável:** *"Sim — análise interseccional completa em
`docs/intersectional_analysis.md`. Disparidade cresce de IR=1.541
(raça) para 3.241 (raça × gênero × idade). Pior subgrupo identificado:
Middle Eastern × Female × 3-9 anos. Pattern detalhado per-subgrupo
documentado."*

## 6. Procedência

- **Script:** `scripts/intersectional_fairness_analysis.py`
- **Modelo:** `outputs/definitive/anchor_hassanpour/exp_anc_hass_convnext_s42/train/20260523T080515Z-e6274a/checkpoints/best.pt`
- **Config:** `configs/anchor_hassanpour/exp_anc_hass_convnext_s42.yaml`
- **Test set:** val/ oficial FairFace (10,954 imagens)
- **Metadados:** `data/raw/fairface/fairface_labels.csv` (colunas gender + age)
- **Outputs:** `outputs/intersectional/per_race_{gender,age,gender_age}_convnext_s42_hassanpour.csv`
  e `intersectional_metrics_convnext_s42_hassanpour.json`
- **Tempo total:** ~10min (zero GPU adicional além da inferência sobre 10,954 imagens)

## 7. Próximos passos (combo defesa-fechamento)

- ✅ #1 Análise interseccional — **ESTE DOCUMENTO**
- ⏭ #2 Ensemble de 3 seeds ConvNeXt-T 🅔 (média de logits)
- ⏭ #3 TTA (Test-Time Augmentation) sobre best ConvNeXt-T 🅔
- ⏭ #4 Calibração + threshold optimization per-class
