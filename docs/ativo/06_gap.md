# Gap — identificação e ranqueamento de lacunas

> Consolida as **5 frentes 🔬** abertas em
> [`04_pesquisa_bibliografica/_perguntas.md`](04_pesquisa_bibliografica/_perguntas.md)
> e sintetizadas em [`05_landscape.md`](05_landscape.md). Decide o
> **escopo experimental da dissertação** ranqueando por viabilidade
> (tempo, recursos), originalidade (precedente na literatura) e
> impacto (relevância para o campo).
>
> Insumo direto para [`07_thesis_statement.md`](07_thesis_statement.md).

## 1. Metodologia de ranqueamento

Cada frente é avaliada em **3 eixos**:

- **Originalidade (O):** 0–5. Quantos papers já tentaram?
  - 5 = totalmente original (0 papers).
  - 3 = direção sugerida mas não executada (vários papers sugerem).
  - 1 = densamente coberto.
- **Viabilidade (V):** 0–5. Tempo, recursos, acesso a dados/código.
  - 5 = executável com infra existente em ≤ 3 meses.
  - 3 = exige infra adicional (anotação humana, compute).
  - 1 = inviável no horizonte da dissertação.
- **Impacto (I):** 0–5. Quanto importa para o campo?
  - 5 = abre nova subarea / responde pergunta seminal.
  - 3 = consolida convergência da literatura.
  - 1 = refinamento incremental.

**Score composto:** O + V + I (max 15).

## 2. Avaliação das 5 frentes

### 2.1 Q04 — Mitigação algorítmica em FairFace race 7-class

**Pergunta:** Qual técnica de Track D (FSCL+, Group DRO, ensemble +
reweighting, FineFACE cross-layer attention, U-FaTE adaptado) melhora
mais a razão de disparidade em FairFace race 7-class in-domain?

| Eixo | Avaliação | Justificativa |
|---|---|---|
| O | **3** | 7 papers sugerem direção; **0 papers executaram em FairFace race 7-class**. Não é "original" no sentido de "ninguém pensou", mas é original no sentido de "ninguém executou". |
| V | **5** | Código open-source disponível para FSCL+ (GitHub Park), Group DRO (GitHub Sagawa), FineFACE (GitHub Manzoor). Backbone ConvNeXt-T padrão. Compute existe. Dataset FairFace local. |
| I | **5** | Estabelece **primeira benchmark sistemática** de mitigações em race 7-class. Ranking entre técnicas é informação que faltava. Cita-se em 7 papers futuros como "finalmente alguém fez". |

**Score: 13/15** ★★

**Riscos:** adaptação multi-classe das técnicas (Park, Sagawa
binarized) exige cuidado metodológico — não é plug-and-play.

### 2.2 Q10 — Matriz skin tone × race em FairFace

**Pergunta:** Como a distribuição de skin tone (Fitzpatrick ou MST)
varia entre as 7 classes raciais do FairFace? Qual fração das
misclassificações é explicada por sobreposição fenotípica
estrutural vs falha de modelo?

| Eixo | Avaliação | Justificativa |
|---|---|---|
| O | **5** | **0 papers** fazem esse cross-reference. Draelos 2025 é dermatologia + dados não-públicos. Buolamwini sugere conceitualmente mas não executa. Frente **totalmente original**. |
| V | **3** | **Anotação manual de subset (500-700 imagens × 3 anotadores)** é o gargalo. Recrutamento de anotadores regionalmente diversos é não-trivial. Mitigação possível: começar com classifier MST automatizado (Google MST API) sobre val set inteiro, depois validar com subset menor. |
| I | **5** | Resposta direta à pergunta seminal "**por que Latinx é difícil**?" — que aparece em 4+ papers sem diagnóstico. **Reframing potencial do ceiling 72%** como combinação de fenótipo + modelo, não apenas modelo. Conecta tracks paralelos (A + C) pela primeira vez. |

**Score: 13/15** ★★

**Riscos:** logística de anotação manual; circularidade do classifier
automatizado (treinado sobre que dataset?).

### 2.3 Q05 — Métrica fairness multi-classe

**Pergunta:** A triangulação **DR (max/min F1) + worst-class F1 +
CV (std/mean F1)** captura mais robustamente fairness em
multi-classe que qualquer métrica isolada?

| Eixo | Avaliação | Justificativa |
|---|---|---|
| O | **3** | Cada paper inventa sua métrica; **triangulação proposta é nova combinação**, mas componentes individuais existem. |
| V | **5** | Apenas calcular as três sobre experimentos já feitos. Custo zero adicional. |
| I | **3** | Contribuição metodológica útil mas não disruptiva. Útil como **subseção** de capítulo, não como contribuição central. |

**Score: 11/15** ★

**Encaminhamento:** **adotar como métrica padrão** em todos os
experimentos da dissertação. **Não merece capítulo dedicado**, mas
deve ser **defendida explicitamente** em `02_metodologia.md` e
`03_metricas.md`.

### 2.4 Q06 — Decomposição do ceiling 72%

**Pergunta:** Da margem entre baseline ResNet-34 (72%) e teto teórico
(99%), qual fração é arquitetural, qual é metodológica, qual é
limite de dados/anotação?

| Eixo | Avaliação | Justificativa |
|---|---|---|
| O | **2** | Nenhum paper faz decomposição controlada, mas vários variam um eixo. Não é original conceitualmente, mas é original na execução sistemática. |
| V | **5** | Variações de backbone (ResNet-34, ConvNeXt-T, ViT-B), seed (single vs 3-seed), e HPO (default vs grid) são experimentos baratos. |
| I | **3** | Insight valioso (sabermos quanto cada eixo contribui), mas não disruptivo. |

**Score: 10/15**

**Encaminhamento:** **incluir como ablação dentro do capítulo Q04**.
Não merece capítulo separado.

### 2.5 Q01 — Confiabilidade da anotação racial em FairFace (Latinx)

**Pergunta:** Inter-annotator agreement (κ de Fleiss) por classe
racial em subset re-anotado do FairFace é uniforme? Especificamente,
κ_Latinx ≪ κ_Black?

| Eixo | Avaliação | Justificativa |
|---|---|---|
| O | **4** | Schumann 2023 faz para MST; **ninguém faz para race** sobre FairFace. |
| V | **2** | **Gargalo logístico ainda maior que Q10** — exige anotadores treinados em taxonomia racial complexa (7 classes), não apenas skin tone. Risco metodológico de anotadores **divergirem mais ainda** sem treinamento adequado. |
| I | **3** | Auditaria a confiabilidade do dataset central. **Mas se Q10 mostrar overlap fenotípico, Q01 fica parcialmente respondida** — anotação ambígua reflete ambiguidade fenotípica, não erro humano. |

**Score: 9/15**

**Encaminhamento:** **fundir parcialmente com Q10** — protocolo de
anotação MST do Q10 pode opcionalmente pedir aos anotadores que
classifiquem também race (esforço marginal). **Não merece frente
independente** dado o gargalo logístico.

## 3. Ranqueamento consolidado

| Posição | Frente | Score | Recomendação |
|---|---|---|---|
| 🥇 | **Q04** — Mitigação algorítmica race 7-class | 13/15 | **CAPÍTULO PRINCIPAL** |
| 🥇 | **Q10** — Matriz skin tone × race | 13/15 | **CAPÍTULO PRINCIPAL** |
| 🥈 | **Q05** — Métrica fairness multi-classe | 11/15 | Adotar transversalmente |
| 🥉 | **Q06** — Decomposição ceiling | 10/15 | Ablação dentro de Q04 |
| 4º | **Q01** — Confiabilidade anotação | 9/15 | Fundir com Q10 |

**Empate Q04 e Q10 é estratégico:** as duas frentes são
**complementares**:

- **Q04** ataca o "**como mitigar**" (técnicas).
- **Q10** ataca o "**por que existe erro**" (diagnóstico fenotípico).

> **ATUALIZAÇÃO v3.2 (2026-06-04):** após reunião com orientador e
> Rodada 5 (mecanismos ML/RN), Q04 + Q10 **deixam de ser capítulos
> paralelos** e tornam-se **pipeline integrado**: o classifier de
> tom de pele (saída do Q10 Fase 1) é usado como **sinal
> condicionante** para o race classifier via FiLM ([[perez_2018]]).
> A combinação cria pipeline operacional único, ancorado em
> [[zemel_2013]] (FRL) e [[madras_2018]] (fair transferência).

## 4. Escopo experimental adotado (v3.2 — pós-reunião)

### 4.1 Estrutura definitiva (3 capítulos + síntese)

```
Capítulo 1 — Classifier MST + matriz P(MST | race)
    ↓ f_MST(x) ∈ ℝ^10 (logits MST)
    ↓ matriz pública como CSV
    Testa H3 (spread Latinx ≥ 5 MST)

Capítulo 2 — Race classifier condicionado por MST
    Backbone ConvNeXt-T (28M) + camadas FiLM condicionadas em f_MST
    Baselines: ResNet-34 vanilla; ConvNeXt-T vanilla; FSCL+; Group DRO;
               Adversarial debiasing (Zhang 2018)
    Cross-reference confusion matrix × matriz MST × race
    Testa H1 (pipeline supera baseline), H2 (Latinx invariante a
    backbone), H4 (overlap MST explica misclassificações)

Capítulo 3 — Extensão a face recognition (RFW ou BFW)
    Pipeline análogo: encoder + FiLM condicionado por MST
    Métrica primária: TAR @ FAR fixo por raça, foco em Black/African
    Testa H5 (fair transferência LAFTR-style)

Síntese — Decomposição final
    erro_total = irredutível_fenotípico + redutível_algorítmico
    Análise por classe (especialmente Latinx)
```

### 4.2 Hipóteses centrais v3.2 (testáveis)

Detalhamento em [`07_thesis_statement.md` §4](07_thesis_statement.md).

| ID | Hipótese | Critério de confirmação |
|---|---|---|
| **H1** (CENTRAL) | Pipeline MST→FiLM→ConvNeXt-T supera baseline ResNet-34 em F1 macro ≥ **+2 pp** E reduz DR ≥ **20%** | Ambos satisfeitos simultaneamente |
| **H2** | ConvNeXt-T vanilla ganha **+2 a +5 pp** F1; Latinx F1 ≈ **60% (±3 pp)** | Ganho no range E Latinx invariante |
| **H3** | Spread MST de Latinx cobre **≥ 5 categorias** com pico distribuído | Spread ≥ 5 com pico não-concentrado |
| **H4** (CENTRAL) | **≥ 50%** das misclassificações Latinx em zonas MST de sobreposição | %_overlap ≥ 50% |
| **H5** (CENTRAL) | Pipeline em face recognition melhora accuracy Black/African **≥ +3 pp** | Ganho ≥ 3 pp sobre baseline sem MST |

### 4.3 Contribuições originais declaradas (v3.2)

| # | Contribuição | Originalidade | Ancoragem R5 |
|---|---|---|---|
| **C1** | **Pipeline MST → FiLM → race classifier** | Primeira instância documentada em race classification multi-classe | [[perez_2018]], [[zemel_2013]] |
| **C2** | **Matriz P(MST \| race) sobre FairFace val + análise overlap** | Zero precedentes públicos (Draelos 2025 é dermatologia, dados não-públicos) | [[schumann_2023]] (protocolo) |
| **C3** | **Demonstração empírica de fair transferência race → face recognition** | Primeira aplicação documentada em FairFace + RFW/BFW | [[madras_2018]] (LAFTR teoria) |
| **C4** | **Triangulação DR + worst-class F1 + CV** | Combinação proposta para multi-classe | [[kleinberg_2017]] (impossibilidade), [[hardt_2016]] (EO/EOD) |
| **C5** | **Quantificação fenotípico vs algorítmico** do erro Latinx | Diagnóstico inédito (C1 × C2 cruzados) | [[fuentes_2019]], [[lewontin_1972]] |

### 4.4 Não-escopo declarado (v3.2)

- **Continuous demographic labels** ([[neto_2025]]): direção futura.
- **Re-anotação completa do FairFace**: apenas validação MST em subset.
- **Cross-dataset 4-class race classification** (RFW/BFW como
  classification, não verification): taxonomias incompatíveis.
- **Fairness em gender ou age** isoladamente: foco em race.
- **Multi-task simultâneo** (race × gender × age): single-task.
- **Backbones acima de ConvNeXt-T**: foco em Pareto-eficiência.
- **VLM fine-tuning** (FaceScanPaliGemma): citado como SOTA, não
  replicado.

### 4.5 Cronograma estimado (v3.2)

| Bloco | Conteúdo | Duração |
|---|---|---|
| Setup metodológico (02, 03, 08) | Especificações executáveis | 2 sem |
| Cap 1 (MST classifier + matriz) | Treino + auditoria + validação manual subset | 4 sem |
| Cap 2 (Race condicionado) | Pipeline + 4 baselines comparativos, 3-seed | 10–12 sem |
| Cap 3 (Face recognition) | Extensão a RFW ou BFW + verificação Black/African | 6 sem |
| Síntese | Decomposição final + análise por classe | 4 sem |
| Escrita (paralelo) | Capítulos completos | 8–12 sem |
| **Total** | | **~28–32 semanas (~6–8 meses ativos)** |

**Defesa prevista**: Jan–Mar 2027.

## 5. Decisão final (v3.2)

**Escopo experimental v3.2:** pipeline integrado Q10+Q04+Q06 (com Q05
transversal). Q01 fundida em Q10 (validação manual MST inclui
anotação adicional de race onde feasível).

**Tese central v3.2 — formulação completa em [`07_thesis_statement.md`](07_thesis_statement.md):**

> *"A incorporação explícita de informação de tom de pele (Monk
> Skin Tone) como sinal auxiliar condicionante no treinamento de
> classificadores faciais profundos melhora métricas de fairness
> demográfica em classificação racial multi-classe e estende essa
> melhoria a tarefas downstream de reconhecimento facial em grupos
> sub-representados."*

A tese é:

- **Falsificável** — H1, H4, H5 têm critérios binários explícitos.
- **Original** — pipeline MST→FiLM→race classifier é primeira
  instância documentada em race classification 7-class; matriz
  P(MST | race) sobre FairFace é zero precedentes; fair transferência
  para face recognition é primeira aplicação empírica.
- **Viável em 6–8 meses** com infra existente + código open-source
  (FiLM, FSCL+, Group DRO, Schumann MST).
- **Operacionalmente útil** — pipeline é artefato deployável, não
  apenas análise post-hoc.

### 5.1 Histórico de versões

- **v3.0 (06/01)**: pivot inicial, diagnóstica empírica.
- **v3.1 (06/02)**: fundamentação teórica AAPA + Lewontin.
- **v3.2 (06/04 — atual)**: **prescritiva**, pipeline integrado, com
  extensão a face recognition.

---

**Próximo arquivo:** [`07_thesis_statement.md`](07_thesis_statement.md) —
formulação final v3.2, justificativa, escopo, limitações, plano B.
