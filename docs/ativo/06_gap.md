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

Combinadas: Q04 mostra **quanto** se pode reduzir; Q10 mostra
**quanto sobra como irredutível**. Esta é uma decomposição
explanatória **inédita** na literatura de race classification.

## 4. Escopo experimental proposto para a dissertação

### 4.1 Estrutura sugerida (3 capítulos experimentais)

```
Capítulo experimental 1 — Q06 (decomposição ceiling) [ablação]
    ↓ estabelece patamar
Capítulo experimental 2 — Q04 (mitigação algorítmica)
    ↓ explora teto
Capítulo experimental 3 — Q10 (matriz skin tone × race)
    ↓ explica resíduo
Síntese: dissertação como única decomposição publicada
```

### 4.2 Hipóteses centrais (testáveis)

**H1 (Q04):** **Pelo menos uma técnica algorítmica de Track D
adaptada (FSCL+ multi-classe OU Group DRO + strong reg OU deep
ensemble + reweighting + temperature)** reduz DR em FairFace race
7-class em **≥ 30%** sobre baseline ResNet-34, sem perda significativa
de F1 macro.

**H2 (Q06):** A troca isolada de backbone **ResNet-34 → ConvNeXt-T**
mantendo todo o resto do pipeline ganha **+2 a +5 pp** de F1 macro;
**Latinx F1 permanece ≈ 60%** sem mitigação algorítmica específica
— demonstrando que **ceiling é parcialmente arquitetural, mas Latinx
é refratário à arquitetura**.

**H3 (Q10):** Construindo a matriz P(MST_k | race_j), Latinx terá
**spread** ≥ 5 categorias MST com sobreposição forte com White,
Middle East e Indian. **≥ 50% das misclassificações Latinx→outras
classes** estarão em zonas MST de sobreposição — confirmando que
parte substancial do "erro Latinx" é **fenotipicamente irredutível**.

**H4 (síntese):** O **ceiling efetivo de F1 macro em FairFace
race 7-class é ≈ 80-82%** (não 100%), por limite combinado de
sobreposição fenotípica + anotação MTurk. Modelos atingindo ≥ 78%
estão essencialmente saturados em informação útil; ganhos adicionais
exigem **mudança de paradigma** (continuous labels, MST, multi-task
com skin tone).

### 4.3 Contribuições originais declaradas

| # | Contribuição | Originalidade |
|---|---|---|
| 1 | **Primeira benchmark sistemática de mitigações algorítmicas em FairFace race 7-class** | 0 precedentes |
| 2 | **Primeira matriz pública Fitzpatrick/MST × FairFace 7-race** | 0 precedentes (Draelos é dermatologia) |
| 3 | **Decomposição empírica do ceiling 72%** em componentes arquitetural / metodológico / fenotípico | 0 precedentes (cada paper varia um eixo isolado) |
| 4 | **Triangulação DR + worst-class + CV como métrica padrão** para race classification multi-classe | Componentes existem; combinação proposta |
| 5 | **Quantificação do erro fenotípico irredutível** vs erro de modelo em Latinx | Pergunta seminal de 4+ papers; nenhuma resposta |

### 4.4 Não-escopo declarado

Para evitar diluição:

- **Continuous demographic labels** (Neto 2025): mencionar mas não
  implementar. Requer infra adicional.
- **Re-anotação completa do FairFace por dermatologistas**: inviável.
- **MST classifier from scratch**: usar Google MST API ou model
  pretrained — não treinar o nosso.
- **Test em RFW/BFW cross-dataset**: limitado a 4-class; não
  comparável diretamente.
- **Mitigações de Track C (skin tone) como protected attribute**:
  conceitualmente interessante mas explode escopo.
- **Fairness em multi-task (race × gender × age simultâneo)**:
  potencial trabalho futuro.

### 4.5 Cronograma estimado (Fase 5+)

| Etapa | Conteúdo | Duração |
|---|---|---|
| Setup | Pipeline de treino estável, 3-seed casado | já feito (Rodada 1 experimental do MBA) |
| Cap 1 (Q06) | Variação backbone + multi-seed + HPO | 4 semanas |
| Cap 2 (Q04) | FSCL+ multi-classe + Group DRO + ensemble; cada uma 3-seed | 8-10 semanas |
| Cap 3 (Q10) — Fase 1+3+4 | Pipeline MST automatizado + matriz + diagnóstico | 4 semanas |
| Cap 3 (Q10) — Fase 2 | Validação manual (anotadores) | 6 semanas (paralelo a outros) |
| Escrita | Capítulos + síntese | 8-12 semanas |
| **Total** | | **~6-8 meses de pesquisa ativa** |

## 5. Decisão final

**Escopo experimental adotado:** Q04 + Q10 + Q06 (como ablação) +
Q05 (metodologia transversal). Q01 fundida em Q10.

**Tese central a desenvolver em [`07_thesis_statement.md`](07_thesis_statement.md):**

> *"O ceiling de 72-75.7% F1 macro em classificação racial 7-class
> sobre o FairFace não é primariamente arquitetural nem
> metodologicamente solúvel apenas via mitigação algorítmica:
> contém componente **fenotípico irredutível**, derivado de
> sobreposição de tom de pele entre categorias raciais, especialmente
> aguda para Latinx/Hispanic. A presente dissertação constrói a
> primeira matriz pública Fitzpatrick/MST × FairFace 7-race,
> demonstra que ≥ 50% do erro Latinx é fenotipicamente irredutível,
> e quantifica simultaneamente o efeito de mitigações algorítmicas
> (FSCL+ multi-classe, Group DRO, ensemble + reweighting) sobre o
> componente restante, redutível de modelo."*

Esta tese é **falsificável** (H3 é binária: spread ≥ 5 categorias
MST ou não; misclassificações ≥ 50% em overlap ou não), **original**
(nenhum precedente combina os dois lados), e **viável em 6-8 meses**.

---

**Próximo arquivo:** [`07_thesis_statement.md`](07_thesis_statement.md) —
formulação final, justificativa, escopo, limitações.
