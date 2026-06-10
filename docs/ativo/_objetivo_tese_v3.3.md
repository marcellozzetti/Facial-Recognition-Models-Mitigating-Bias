# Objetivo da tese — v3.3 (pós-reunião 2026-06-08)

> **Versão**: 3.3 — pós-aprovação do pipeline pelo orientador.
> **Atualizada**: 2026-06-10.
> **Propósito**: âncora narrativa para escrita da qualificação
> (deadline 15-jul-2026).

## 1. Storytelling aprovado pelo orientador

A tese deve seguir a estrutura narrativa (confirmada na reunião):

1. **Contexto**
2. **Problemas existentes**
3. **O que tem sido feito** (estudos recentes)
4. **O que falta ser explorado** (gap)
5. **Objetivo**
6. **Como será feito** (metodologia)

## 2. Objetivo geral (1 parágrafo)

> **Esta dissertação tem como objetivo desenvolver e avaliar um
> pipeline de classificação racial em imagens faciais que incorpora
> tom de pele (escala Monk Skin Tone) como sinal auxiliar condicionante
> via mecanismo arquitetural, com o propósito de mitigar viés racial
> demonstrável no estado-da-arte atual — particularmente a
> disparidade severa entre classes raciais bem representadas
> (Black F1 ≈ 90%) e classes sub-representadas (Latinx F1 ≈ 60%)
> documentada em modelos como FaceScanPaliGemma sobre o dataset
> FairFace. A contribuição principal é a primeira instância
> empírica documentada do uso de tom de pele explícito como contexto
> arquitetural para race classification multi-classe, com avaliação
> rigorosa via triangulação de métricas (DR + worst-class F1 + F1
> macro, complementadas por EO_h e EOD por classe em ablation
> intersectional race × gender), e demonstração de fair transferência
> do mecanismo para downstream face recognition.**

## 3. Objetivos específicos (5)

### Objetivo específico 1 — Auditoria fenotípica do FairFace (Cap 1)

Quantificar a distribuição cruzada **MST × race classes** sobre o
FairFace via SkinToneNet (Pereira 2026) com validação humana em
subset. **Entregar a primeira matriz pública dessa distribuição**.

- **Hipótese testada**: H3 — Latinx tem spread MST amplo (≥ 5 das
  10 classes MST).
- **Métricas**: histograma MST por raça, % overlap entre raças,
  validação manual via Prolific (~700 imgs × 3 anotadores).

### Objetivo específico 2 — Avaliar modelos pré-treinados de skin tone (Cap 1)

> **Recomendação do orientador (2026-06-08)**.

Conduzir avaliação metodológica de modelos pré-treinados disponíveis
para classificação MST (SkinToneNet, Casual Conversations baseline,
Google API, alternativas HuggingFace) e **justificar a escolha do
modelo adotado** com critérios de desempenho, generalização e
disponibilidade.

- **Saída**: tabela comparativa + decisão fundamentada + protocolo
  de validação.

### Objetivo específico 3 — Pipeline condicionado para race classification (Cap 2)

Implementar e avaliar pipeline **SkinToneNet → ConvNeXt-T + FiLM →
race classifier** sobre FairFace, comparando contra 6 baselines
(ConvNeXt-T puro, ResNet-34, FSCL+, Group DRO, FineFACE, Adversarial
debiasing).

**Adicional** (pós-reunião): comparar **FiLM-conditioning vs
CLIP-conditioning** em ablation arquitetural — endereçar
recomendação do orientador de testar mecanismo moderno alternativo.

- **Hipóteses testadas**: H1 (pipeline funciona), H2 (ConvNeXt-T
  puro vs ResNet-34), H4 (erros Latinx em zonas overlap MST).
- **Métricas**: F1 macro, DR, worst-class F1, EO_h por classe e EOD
  por classe (ablation race × gender).

### Objetivo específico 4 — Fair transferência para face recognition (Cap 3)

Demonstrar empíricamente que o backbone fair-treinado no Cap 2
transfere a propriedade fair para tarefa downstream de **face
recognition** sobre RFW ou BFW, com controle explícito de **pixel
information** como confounder (resposta a Pangelinan 2023).

- **Hipóteses testadas** (revisadas pós-Pangelinan):
  - **H5** (cautelosa): condicionamento MST melhora fairness em FR
    *quando pixel info é controlada*.
  - **H6** (nova): disparity residual em Black/African é
    predominantemente explicada por pixel info, NÃO por skin tone
    (decomposição de variância).
- **Métricas**: TAR @ FAR fixo por raça, ε-DEO transferido.

### Objetivo específico 5 — Síntese metodológica (Cap 4)

Decompor o erro de classificação de Latinx em **componente
fenotípico (irredutível pelo overlap MST)** vs **componente
algorítmico (mitigável)** via análise cruzada dos resultados dos
Caps 1-3. **Quantificar quanto da disparidade é estrutural** (fronteira
de classificação ambígua por sobreposição MST) **vs quanto é
mitigável** (resíduo pós-conditioning).

- **Métrica**: % variance accounted by MST overlap vs % attributable
  to model bias.
- **Resultado esperado**: decomposição que sustenta a discussão
  ética/social sobre limites estruturais de race classification
  multi-classe.

## 4. Contribuições científicas (revisadas v3.3)

| ID | Contribuição | Originalidade |
|---|---|---|
| **C1** | Avaliação metodológica de modelos pré-treinados MST + protocolo de escolha | Recomendação orientador 2026-06-08 |
| **C2** | Matriz pública MST × race do FairFace | Não publicado: Pereira 2026 só agrega |
| **C3** | Primeira aplicação de FiLM-conditioning a fairness facial | Sem precedente direto |
| **C4** | Triangulação de métricas multi-classe (DR + worst-class F1 + EO_h/EOD por classe) | Baseado em Hardt 2016, original em race 7-class |
| **C5** | Demonstração empírica de fair transferência classification → FR | LAFTR é teórico; Aguirre 2023 é NLP; CV é nova |
| **C6** | Decomposição variância (fenotípico × algorítmico) | Diagnóstico inédito |
| **C7** | Comparativo FiLM vs CLIP-conditioning para race fairness | Recomendação orientador 2026-06-08 |

## 5. Hipóteses revisadas v3.3 (6 hipóteses, era 5)

| ID | Hipótese | Critério de confirmação | Critério de refutação |
|---|---|---|---|
| **H1** | Pipeline MST + FiLM → ConvNeXt-T supera baseline ResNet-34 em F1 macro ≥+2pp E reduz DR ≥20% | Ambos satisfeitos | Qualquer um abaixo |
| **H2** | ConvNeXt-T puro ganha +2 a +5pp sobre ResNet-34; Latinx F1 ≈60% (±3pp invariante) | Ganho no range E Latinx invariante | Fora do range OU Latinx muda |
| **H3** | Matriz MST × race mostra Latinx com spread ≥ 5 das 10 classes MST | Spread ≥ 5 | Spread < 5 |
| **H4** | ≥ 50% dos erros Latinx no baseline estão em zonas MST de overlap | %_overlap ≥ 50% | %_overlap < 50% |
| **H5** (revisada) | Condicionamento MST mantém ou melhora fairness em FR **com quality control de pixel info** | ≥+3pp em Black/African após normalização pixel info | < +3pp ou degradação |
| **H6** (NOVA) | Disparity residual após conditioning é explicada predominantemente por pixel info (Pangelinan 2023) | ≥ 70% variance explicada por pixel info | < 70% (sugere bias algorítmico residual) |

## 6. Foco da tese (do orientador)

**Melhorar a acurácia dos modelos de reconhecimento de imagem para
mitigar vieses, com foco em viés racial / cor de faces.**

- **Tarefa central**: race classification 7-class no FairFace.
- **Mecanismo central**: tom de pele MST como sinal auxiliar.
- **Resultado esperado**: redução documentada de disparity entre
  raças.
- **Aplicação downstream**: face recognition.

## 7. Storytelling — esqueleto para escrita

### Contexto

Sistemas de reconhecimento facial estão em uso massivo (catraca,
banco, fronteira, identificação policial). A literatura documenta há
quase uma década (Buolamwini & Gebru 2018) que esses sistemas
**falham desproporcionalmente em grupos sub-representados**.

### Problemas existentes

- **Estado-da-arte tem disparity severa**: FaceScanPaliGemma
  (AlDahoul 2026) atinge F1 macro 75% mas F1 Latinx 60%, F1 Black
  90% — gap de 30pp entre as raças.
- **Balanceamento de dados não basta**: FairFace é balanceado por
  design, mas a disparidade persiste (Karkkainen 2021, Kolla 2022).
- **Mitigação algorítmica atual** (FSCL, Group DRO, Adversarial)
  não foi sistematicamente testada em race classification multi-classe.

### O que tem sido feito (estudos recentes)

- **Datasets balanceados**: FairFace (2021), RFW (2019), BFW (2020).
- **Skin tone como dimensão alternativa**: Schumann 2023 (MST),
  Pereira 2026 (SkinToneNet).
- **Mitigação algorítmica**: Park 2022 (FSCL+), Sagawa 2020 (Group
  DRO), Manzoor 2024 (FineFACE), Liu 2025 (BNMR).
- **Vision-language models**: AlDahoul 2024/26 (FaceScanPaliGemma),
  FairCLIP (Luo 2024).

### O que falta ser explorado (gap)

- **Skin tone como sinal arquitetural condicionante** em race
  classification multi-classe — sem precedente direto.
- **Matriz pública MST × race** para FairFace — Pereira 2026 audita
  mas não cruza.
- **Decomposição irredutível (fenotípico) vs redutível
  (algorítmico)** do erro Latinx — inédito.
- **Fair transferência empírica em face recognition CV** — LAFTR é
  teórico, Aguirre é NLP.

### Objetivo

Ver Seção 2 acima.

### Como será feito

Pipeline em 6 etapas (aprovado pelo orientador). Cap 1 (MST audit) +
Cap 2 (race + conditioning) + Cap 3 (fair transfer to FR) + Cap 4
(síntese decompositiva).

Triangulação de métricas (Hardt 2016 base, multi-classe nossa).

Comparação contra 6 baselines + ablation FiLM vs CLIP-conditioning.

## 8. Próximas ações imediatas

1. ✅ Registrar reunião (2026-06-08 doc criado).
2. ✅ Criar 7 fichas R6 (Aguirre VERIFIED + 6 OVERVIEW_ONLY).
3. ✅ Planejar Rodada 7 (este documento + _rodada_07_planejamento).
4. ✅ Atualizar objetivo e hipóteses para v3.3.
5. **PENDENTE**: executar buscas Rodada 7 e triar candidatos.
6. **PENDENTE**: setup Overleaf com template Unifesp/ICT.
7. **PENDENTE**: começar escrita da Introdução (Cap 1).
