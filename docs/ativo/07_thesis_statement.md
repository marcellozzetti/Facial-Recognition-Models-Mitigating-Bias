# Thesis Statement v3

> Reformulação final da tese, baseada na decisão de escopo em
> [`06_gap.md`](06_gap.md). Substitui [`historico/THESIS_STATEMENT.md`](historico/THESIS_STATEMENT.md)
> (v1 + v2) — versões anteriores foram arquivadas no pivot
> estratégico de 2026-05-25 por terem sido construídas sobre
> "evolução do MBA", framing recusado pelo orientador.
>
> Esta v3 é construída sobre **gap identificado na literatura**, não
> sobre continuidade do trabalho anterior.
>
> Versão: v3.0 — 2026-05-25.

## 1. Tese central

> **O ceiling de 72-75.7% F1 macro observado em classificação racial
> de 7 categorias sobre o FairFace dataset não é primariamente
> arquitetural nem metodologicamente solúvel apenas via mitigação
> algorítmica.**
>
> **Existe componente fenotípico irredutível, derivado da sobreposição
> distribucional de tom de pele entre categorias raciais — sobreposição
> particularmente aguda para a categoria Latinx/Hispanic, que ocupa
> faixa contínua compartilhada com White, Middle Eastern e Indian no
> espaço Fitzpatrick/Monk Skin Tone.**
>
> **A presente dissertação contribui originalmente com:**
>
> **(a) a primeira matriz pública Fitzpatrick/MST × FairFace 7-race
> construída via classifier automatizado validado por anotação manual
> regionalmente diversa;**
>
> **(b) a primeira benchmark sistemática de mitigações algorítmicas
> de fairness (FSCL+ multi-classe, Group DRO + strong regularization,
> deep ensemble + temperature scaling + group reweighting) aplicadas
> ao problema de race classification multi-classe in-domain;**
>
> **(c) decomposição empírica do "erro Latinx" em componente
> fenotipicamente irredutível vs componente algoritmicamente redutível,
> quantificando a fração de cada um;**
>
> **(d) protocolo metodológico de triangulação de métricas
> (DR max/min + worst-class F1 + CV) como padrão defensável para
> reportar fairness em classificação multi-classe.**

## 2. Justificativa

### 2.1 Por que o problema importa

Sistemas de classificação facial demográfica são deployed em
contextos sensíveis (segurança, marketing, ciências sociais). O
[NIST FRVT 2019](04_pesquisa_bibliografica/grother_2019.md) documenta
diferenciais de FPR entre raças variando em **uma a duas ordens de
magnitude** em sistemas comerciais. O **EU AI Act** (entrada em vigor
2024) exige auditoria sistemática de fairness ([Lafargue et al., 2025](04_pesquisa_bibliografica/lafargue_2025.md)).
Compreender **por que** sistemas de classificação racial falham — e
**em qual fração esse erro é remediável** — é pré-requisito para
deployment ético.

### 2.2 Por que esta tese (vs alternativas)

**Alternativa A — focar apenas em mitigação algorítmica:** insuficiente.
Vários papers ([Park 2022](04_pesquisa_bibliografica/park_2022.md),
[Sagawa 2020](04_pesquisa_bibliografica/sagawa_2020.md),
[Manzoor 2024](04_pesquisa_bibliografica/manzoor_2024.md)) sugerem
direção, e Track A (race classification) é demonstravelmente vazio
nesta direção. Mas mitigar **sem diagnosticar** o erro é otimização
no escuro: se 50% do erro é fenotipicamente irredutível, métodos
agressivos podem **piorar accuracy sem reduzir fairness real**.

**Alternativa B — focar apenas em matriz skin tone × race:**
contribuição forte mas incompleta. Mostra **por que existe gap** mas
não responde **se pode ser reduzido**.

**Alternativa C — combinar mitigação + diagnóstico fenotípico:**
adotada por esta dissertação. **Única decomposição na literatura**
que separa as duas fontes de erro e quantifica cada uma.

### 2.3 Por que o FairFace especificamente

Conforme [Q02](04_pesquisa_bibliografica/_perguntas.md):

- **FairFace é o dataset mais utilizado** para race fairness
  classification (6/14 papers do corpus original, expandido para
  6/19 com Rodada 3).
- **Único dataset com 7 categorias raciais** (inclui Middle Eastern
  e Latinx, ausentes em RFW/BFW/UTKFace).
- **Validação cruzada disponível** via UTKFace (cross-dataset 5-class).

Qualquer contribuição em fairness facial racial **precisa pousar no
FairFace** para conexão com a literatura. Esta dissertação não
foge dessa âncora.

## 3. Contribuições originais

| # | Contribuição | Originalidade | Material de evidência |
|---|---|---|---|
| **C1** | Matriz Fitzpatrick/MST × FairFace 7-race (CSV público + análise) | **Totalmente original** — 0 precedentes em fairness/biometria | [Q10](04_pesquisa_bibliografica/_perguntas.md), [landscape §6.4](05_landscape.md) |
| **C2** | Benchmark sistemática de FSCL+/Group DRO/ensemble em FairFace race 7-class (3-seed casado) | **Execução original** — 7 papers sugerem, 0 executam | [Q04](04_pesquisa_bibliografica/_perguntas.md), [landscape §6.1](05_landscape.md) |
| **C3** | Decomposição empírica erro Latinx: fenotípico irredutível vs algoritmicamente redutível | **Conceitualmente original** — combina C1 + C2 | [gap §4.3 H3+H4](06_gap.md) |
| **C4** | Decomposição empírica do ceiling 72%: arquitetural / metodológico / dados | **Execução controlada original** — vários variam, ninguém isola | [Q06](04_pesquisa_bibliografica/_perguntas.md) |
| **C5** | Triangulação DR + worst-class F1 + CV como métrica padrão | **Combinação proposta** — componentes existem | [Q05](04_pesquisa_bibliografica/_perguntas.md) |

## 4. Hipóteses falsificáveis

A tese **só é defensível se** as seguintes hipóteses se confirmarem
quantitativamente:

| ID | Hipótese | Confirmação | Refutação |
|---|---|---|---|
| **H1** | ≥1 técnica algorítmica (FSCL+ multi-classe, Group DRO, ensemble + reweighting) reduz DR em FairFace race 7-class em **≥30%** sobre baseline ResNet-34, sem perder >2 pp F1 macro | reducao≥30% E perda F1 ≤2 pp | nenhuma técnica atinge esse threshold |
| **H2** | Troca ResNet-34→ConvNeXt-T mantendo pipeline ganha +2 a +5 pp F1 macro; **Latinx F1 permanece ≈60% (±3 pp)** sem mitigação específica | ganho 2-5 pp E Latinx invariante | ganho fora da faixa OU Latinx muda mais que ±3 pp |
| **H3** | Spread MST de Latinx em FairFace **cobre ≥5 categorias MST** com sobreposição forte com White, ME, Indian | spread ≥5 categorias com pico distribuído | spread <5 ou distribuição concentrada |
| **H4** | **≥50% das misclassificações Latinx→outras classes** estão em zonas MST de sobreposição (interseção das distribuições marginais ≥30%) | %_overlap ≥50% | %_overlap <50% (sugere que o erro é predominantemente de modelo, não fenotípico) |
| **H5** | Ceiling efetivo F1 macro em FairFace race 7-class é **80-82%** (não 99%): modelos ≥78% saturados | maior modelo testado fica ≤82% e plateau-shape visível | modelo passa de 82% (refuta) |

**H4 é a hipótese central da tese.** Se refutada, a tese é
fundamentalmente errada e precisa reformulação.

## 5. Escopo

### 5.1 Dentro do escopo

- Race classification **in-domain** no FairFace 7-class (train→val).
- Backbones: **ResNet-34** (baseline) + **ConvNeXt-T** (modernização).
- Mitigações: **FSCL+ adaptado multi-classe**, **Group DRO + strong
  ℓ2 + early stopping**, **deep ensemble 3-seed + temperature scaling
  + group reweighting**. Possivelmente também **FineFACE cross-layer
  attention adaptado** (escopo expandido se cronograma permitir).
- Skin tone: **MST 10-pt** (preferencial) via [Google MST classifier](https://skintone.google)
  + validação manual em subset.
- Métricas: **F1 macro + accuracy** + tripla **DR + worst-class F1 + CV**.
- Multi-seed: **3 seeds (42, 1, 2)** com **média ± std** e
  **comparação pareada**.

### 5.2 Fora do escopo (declarado)

- **Continuous demographic labels** ([Neto 2025](04_pesquisa_bibliografica/neto_2025.md))
  — mencionar como direção futura, não implementar.
- **Re-anotação racial completa do FairFace** — apenas validação MST.
- **Cross-dataset RFW/BFW evaluation** — taxonomias diferentes.
- **Mitigação de gender / age** — foco em race.
- **Multi-task simultâneo** (race × gender × age) — single-task.
- **ViT-B / DINOv2 / EfficientNet** como backbones — ConvNeXt-T já
  cobre "moderno" sem explodir escopo.
- **Fairness em geração / VLM zero-shot** — fora do problema.
- **Dimensão temporal / longitudinal** — single-snapshot.
- **Privacy-preserving fairness** — não relevante.

## 6. Limitações honestas declaradas

### 6.1 Limitações estruturais (não-elimináveis pela dissertação)

- **Taxonomia 7-class do FairFace é socialmente construída.** Não
  representa "verdade biológica" nem cobre todas as identidades
  raciais possíveis. ([Q09](04_pesquisa_bibliografica/_perguntas.md))
- **Anotação MTurk do FairFace** sem inter-annotator agreement por
  classe ([Q01](04_pesquisa_bibliografica/_perguntas.md)) — sub-estimamos
  ruído de label.
- **In-domain evaluation** (FairFace val) — generalização para
  outros datasets é limitada por incompatibilidade taxonômica.

### 6.2 Limitações metodológicas (assumidas conscientemente)

- **3-seed casado** é o mínimo defensável; **5-10 seeds** seriam
  ideais para CI mais estreito. Tradeoff de compute.
- **Validação manual MST** sobre subset, não dataset completo —
  matriz aproximada, não exata.
- **Classifier MST automatizado** ([Schumann 2023](04_pesquisa_bibliografica/schumann_2023.md))
  pode ter viés próprio — circularidade reconhecida.
- **Single backbone moderno** (ConvNeXt-T) — comparação com ViT-B
  / DINOv2 deixada como trabalho futuro.

### 6.3 Riscos de falsificação

Se H4 for **refutada** (i.e., overlap MST não explica ≥50% do erro
Latinx), a tese **precisa ser reformulada**. Plano B:

- Manter C2 (benchmark de mitigações) e C5 (métrica triangular)
  como contribuições estáveis.
- Reformular C1+C3 como **observação negativa** ("matriz construída
  e diagnóstico não confirma hipótese; erro é majoritariamente
  algorítmico").
- A dissertação **continua publicável** mesmo com H4 refutada —
  observação negativa é informação cientificamente válida.

## 7. Conexão com plano experimental

Esta tese é executada em **3 capítulos experimentais** definidos em
[`06_gap.md` §4.1](06_gap.md):

```
Cap 1 (Q06) — Decomposição ceiling 72%
    ResNet-34 → ConvNeXt-T, 3-seed casado, HPO modesto
    Teste de H2

Cap 2 (Q04) — Mitigação algorítmica
    FSCL+ adaptado, Group DRO + strong reg, ensemble + reweighting
    3 técnicas × 3 seeds = 9 runs comparáveis ao baseline
    Teste de H1

Cap 3 (Q10) — Matriz skin tone × race
    MST automatizado sobre val (10 954 imgs)
    Anotação manual subset (500-700 imgs × 3 anotadores)
    Matriz P(MST | race) + análise overlap
    Cross-reference com confusion matrix do melhor modelo
    Teste de H3 e H4

Síntese — Decomposição final
    erro_total = erro_fenotípico_irredutível + erro_redutível_modelo
    Quantificação por classe (especialmente Latinx)
    Teste de H5
```

Cronograma: ver [`06_gap.md` §4.5](06_gap.md). Estimativa total **6-8
meses** de pesquisa ativa.

## 8. Status do thesis statement

- **v1**: histórico (pre-pivot), construído sobre evolução do MBA.
- **v2**: histórico (versão 2.0 de 2026-05-23), incorporando achados
  iniciais, mas ainda framing MBA.
- **v3 (este arquivo)**: construído sobre gap identificado em corpus
  de 19 fichas + 10 perguntas de pesquisa + 5 frentes 🔬 ranqueadas.

**Status:** **PRONTO PARA APROVAÇÃO COM ORIENTADOR** (Prof. Marcos
Quiles).

**Próximos arquivos a produzir:**

- [`02_metodologia.md`](02_metodologia.md): protocolo experimental
  detalhado (sementes, splits, hiperparâmetros, padding, multi-seed).
- [`03_metricas.md`](03_metricas.md): definição formal de DR,
  worst-class F1, CV, com derivação e propriedades.
- [`08_experimentos.md`](08_experimentos.md): tabela de fatores,
  ablações e rationale para cada run.
- [`09_resultados.md`](09_resultados.md): a ser preenchido durante
  execução.
