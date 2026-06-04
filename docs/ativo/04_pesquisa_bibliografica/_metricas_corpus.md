# Métricas do corpus — big numbers

> Métricas consolidadas da pesquisa bibliográfica para uso em
> reuniões com orientador (Prof. Marcos Quiles) e demais avaliadores.
> **Manter atualizado a cada nova rodada de triagem.**
>
> Última atualização: **2026-06-04** (pós-Rodada 5 + Rodada 2.6).

## Sumário em uma linha

> **~57 papers avaliados → 29 aprovados → 10 selecionados para leitura
> primária. SOTA validado em 2 rodadas independentes.**

## 1. Espaço de busca e aprovações por rodada

| Rodada | Foco | Candidatos avaliados (estimado) | Aprovados | Rejeitados / Substituídos / Standby |
|---|---|---|---|---|
| **R1** | Seeds iniciais (cobertura temática direta) | ~15 | **9** | 1 standby (S10 fair generation) |
| **R2** | Snowballing das R1 (técnicas metodológicas) | ~10 | **5** | — |
| **R2.5** | Verificação dedicada de SOTA | (re-verificação) | 0 (sem novas fichas) | — |
| **R3** | Broadening (não-FairFace) | ~15 | **5** | **5 rejeitados explicitamente** (BUPT-Balancedface, DemogPairs, CC v2, Draelos 2025, AI-Face Lin 2025) |
| **R4** | Fundamentação científica de raça e tom de pele | 5 | **4** | 1 substituído (Sparks-Jantz 2002 — direção oposta da hipótese) |
| **R5** | Mecanismos ML / Redes Neurais (pós-feedback orientador) | ~12 | **6** | — (alguns adjacentes considerados: AdaIN, conditional BN, SPADE) |
| **R2.6** | Re-verificação SOTA pós-reunião | 12 citantes do FaceScanPaliGemma | 0 (SOTA mantido) | — |
| **TOTAL** | | **~57** | **29** | **~28** (estimativa) |

**Números defensáveis:**

- **29 fichas catalogadas** ✅ (exato — verificável em `04_pesquisa_bibliografica/*.md`)
- **6 rejeições documentadas explicitamente** em `_triagem.md`
- **~57 papers avaliados** (estimativa, ±5)

## 2. Distribuição por track temático

| Track | Foco | Fichas |
|---|---|---|
| **A** | Race classification | 3 (FairFace, AlDahoul, FairGRAPE) |
| **B** | Face recognition fairness | 4 (RFW, BFW, NIST FRVT 8280, Neto continuous) |
| **C** | Skin tone (Fitzpatrick / MST) | 4 (Gender Shades, Casual Conv, Schumann MST, Lafargue) |
| **D** | Mitigação algorítmica | 6 (FSCL, Group DRO, FineFACE, U-FaTE, Bhaskaruni, FairGRAPE) |
| **E** | Auditoria e metodologia | 4 (DSAP, Lafargue, Mehrabi survey, Kotwal survey) |
| **F** | Fundamentação científica | 4 (AAPA 2019, Lewontin 1972, Fitzpatrick 1988, Massey-Martin 2003) |
| **G** | Mecanismos ML / Redes Neurais (NOVO R5) | 6 (Hardt 2016, FiLM, Zemel 2013, LAFTR, Zhang 2018, Kleinberg 2017) |

**Observação:** alguns papers aparecem em mais de um track (e.g., Lin
em A e D; Lafargue em C e E). Total ≠ soma direta por causa dessa
sobreposição interdisciplinar.

## 3. Distribuição por tipo de publicação

| Tipo | Quantidade | Exemplos |
|---|---|---|
| **Conference (top venue)** | 17 | CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, WACV, FAccT, ITCS |
| **Conference workshop** | 3 | CVPRW (Robinson BFW, Hazirbas Casual Conv) |
| **Journal (top)** | 4 | Nature Sci Reports (AlDahoul), Information Fusion (DSAP), IEEE TBIOM (Kotwal), Arch Dermatol (Fitzpatrick) |
| **Survey journal** | 2 | ACM CSur (Mehrabi), IEEE TBIOM (Kotwal) |
| **Technical report** | 2 | NISTIR 8280, NIS Massey-Martin |
| **Statement institucional** | 1 | AAPA Statement (Am J Phys Anthropol) |
| **Book chapter** | 1 | Lewontin 1972 (Evolutionary Biology, Springer) |
| **Preprint (under review)** | 1 | Neto 2025 (arXiv) |

## 4. Cobertura temporal

| Período | Fichas |
|---|---|
| 1972 | 1 (Lewontin) |
| 1988 | 1 (Fitzpatrick) |
| 2003 | 1 (Massey-Martin) |
| 2013 | 1 (Zemel LFR — Test-of-Time Award) |
| 2016 | 1 (Hardt EO) |
| 2017 | 1 (Kleinberg impossibility) |
| 2018 | 4 (Buolamwini, FiLM, LAFTR, Zhang adversarial) |
| 2019 | 4 (Wang RFW, Grother NISTIR, Bhaskaruni, Fuentes AAPA) |
| 2020 | 2 (Robinson BFW, Sagawa Group DRO) |
| 2021 | 3 (Karkkainen FairFace, Hazirbas Casual Conv, Mehrabi survey) |
| 2022 | 2 (Lin FairGRAPE, Park FSCL) |
| 2023 | 1 (Schumann MST) |
| 2024 | 4 (AlDahoul, Manzoor FineFACE, Dehdashtian U-FaTE, Dominguez DSAP) |
| 2025 | 3 (Lafargue, Neto continuous, Kotwal survey) |
| 2026 | 1 (AlDahoul Nature SR — versão journal de 2024) |

**Span**: 54 anos (1972–2026). **Mediana**: 2020-2021 (campo está em
evolução acelerada pós-Gender Shades 2018).

## 5. Rigor metodológico — métricas

| Métrica | Valor |
|---|---|
| **Fichas com autoria 100% verificada em fonte primária** | 29/29 (100%) |
| **PDFs open access versionados no repo** | 26 |
| **Papers paywalled (referência apenas, R4 fundacional)** | 4 (Fuentes 2019 Wiley; Lewontin 1972 Springer; Fitzpatrick 1988 JAMA; Massey-Martin 2003 técnico) |
| **Fichas com Seção 12 (Análise crítica do método)** | 10/29 (priorizadas; expandir após v3.2) |
| **Fichas com Seção 11 (Future work)** | 29/29 (100% — template normativo) |
| **Perguntas Q&A respondidas** (`_perguntas.md`) | 14 (Q01–Q14) |
| **Frentes 🔬 abertas** | 5 (Q01, Q04, Q05, Q06, Q10) |
| **Rodadas de validação SOTA** | 2 (R2.5 + R2.6) — independentes |
| **Tracks temáticos cobertos** | 7 (A–G) |

## 6. Sub-seleção de 10 papers para leitura prioritária

**Critério:** cobrir o **mínimo necessário** para defender a tese
v3.1 com leitura primária de cada peça argumentativa.

| # | Paper | Função na tese |
|---|---|---|
| 1 | Fuentes et al. 2019 (AAPA Statement) | Fundamentação teórica — race ≠ biologia |
| 2 | Lewontin 1972 (Apportionment) | Fundamentação genética — partição 85/6/8 |
| 3 | Buolamwini & Gebru 2018 (Gender Shades) | Marco histórico — founding paper do campo |
| 4 | Kärkkäinen & Joo 2021 (FairFace) | Dataset central da dissertação |
| 5 | AlDahoul 2024/2026 (FaceScanPaliGemma) | SOTA atual: 75.7% Acc / 75% F1 macro |
| 6 | Lin et al. 2022 (FairGRAPE) | Validação cruzada do baseline ResNet-34 72% |
| 7 | Schumann et al. 2023 (MST consensus) | Protocolo de tom de pele para Q10 |
| 8 | Hazirbas et al. 2021 (Casual Conversations) | Paradigma alternativo (self-reported) |
| 9 | Park et al. 2022 (FSCL) | Técnica central de mitigação para Cap 2 |
| 10 | Sagawa et al. 2020 (Group DRO) | Técnica alternativa de mitigação |

**Observação importante:** essa lista de 10 **NÃO É filtro de 29 → 10**.
É **priorização de leitura** com critério de **cobertura mínima
suficiente para defesa**. Outras 19 fichas são complementares
(surveys, técnicas especializadas, datasets de outras tarefas,
fundamentações históricas).

## 7. Comparação corpus pré-reunião vs pós-reunião 2026-06-04

| Métrica | Pré-reunião | Pós-reunião | Delta |
|---|---|---|---|
| Fichas catalogadas | 23 | **29** | +6 (Rodada 5 — ML/RN) |
| Tracks cobertos | 6 (A–F) | **7** (+ G) | +1 |
| Perguntas respondidas | 14 | 14 | 0 |
| Frentes 🔬 abertas | 5 | 5 | 0 (escopo mantido) |
| Rodadas de SOTA | 1 (R2.5) | **2** (+ R2.6) | +1 |
| Fichas com Seção 12 | 0 | **10** | +10 (priorizadas) |
| PDFs open access versionados | 20 | **26** | +6 |

## 8. Para a próxima reunião — perguntas antecipadas sobre rigor

Caso o orientador perguntar:

| Pergunta provável | Big number a usar |
|---|---|
| Quantos papers você efetivamente avaliou? | ~57 (estimativa); 29 aprovados; 6 rejeições explícitas documentadas em `_triagem.md` |
| Por que rejeitou BUPT-Balancedface? | Citado em [[neto_2025]] sem ficha dedicada; cobertura sobreposta com RFW para verification. Reavaliar se gap em Cap 3 v3.2 exigir |
| Quantos venues distintos cobriu? | Top venues: CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, WACV, FAccT, ITCS, IEEE TBIOM, Nature Sci Reports, Information Fusion, ACM CSur |
| O SOTA é definitivo? | Verificado em 2 rodadas independentes (R2.5 maio/26, R2.6 junho/26). FaceScanPaliGemma 75.7% F1 sem competidor identificado |
| O corpus está temporalmente atualizado? | Sim: cobertura 1972–2026, com 8 fichas de 2024-2026 (28% do corpus) |
| Como você validou autoria após o incidente? | 29/29 fichas com autoria 100% verificada em fonte primária (arXiv, DOI, PMC, página oficial do venue) |

## 9. Como atualizar este arquivo

Após cada nova rodada de triagem:

1. Atualizar contagem em §1 (Tabela de rodadas).
2. Atualizar §2 (tracks) e §3 (tipos de publicação).
3. Atualizar §4 (cobertura temporal).
4. Atualizar §5 (métricas de rigor).
5. Se atualizar a lista de 10 leituras: atualizar §6.
6. Atualizar §7 (delta pré/pós) se a rodada coincide com reunião.
7. Atualizar §8 (perguntas antecipadas) com novas defesas
   conforme o feedback acumular.
