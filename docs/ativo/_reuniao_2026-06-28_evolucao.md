---
data: 2026-06-28
tipo: material-apresentacao
participantes: [Marcello Ozzetti, Prof. Marcos Quiles]
reuniao_anterior: 2026-06-15
proxima_entrega: 2026-07-15
status: preparacao
---

# Reunião com o orientador — Evolução das últimas 2 semanas

> **13 dias** após a reunião de 15/jun onde o embasamento teórico
> foi aprovado e a escrita foi liberada. **17 dias** restantes até
> a primeira revisão ao orientador (15/jul/2026).

---

## 1. Status das decisões da reunião anterior (15/jun)

| # | Decisão | Status | Observação |
|---|---|---|---|
| 1 | Embasamento aprovado — escrita liberada | ✅ Mantido | Corpus expandido para 104 fichas (+3 R8 Latinx) |
| 2 | Deadline 15/jul/2026 (primeira revisão) | ✅ Em curso | 17 dias restantes |
| 3 | ResNet+FiLM como linha tradicional + CLIP avaliação | ✅ Formalizado | Decisão arquitetural registrada com 4 configs no Cap 2 |
| 4 | Mais uma passada no corpus | ✅ **Concluída** | Camadas 2-5 completadas; auditoria sistemática feita |

**Resultado**: 4/4 decisões executadas conforme planejado.

---

## 2. Principais entregas das 2 semanas

### 2.1 Revisão bibliográfica completa (Camadas 2-5)

| Métrica | 15/jun | 28/jun |
|---|---|---|
| Fichas VERIFIED | 14/101 (Camada 1) | **103/104 (99 %)** |
| PDFs no repositório | 26 | **103/104 (99 %)** |
| Auditoria de qualidade | — | 29 A / 14 B / 57 C / 4 D |
| Bibliografia consolidada | — | **`referencias.bib` 104 entradas** |

Detalhamento da auditoria: 100 % das fichas têm autoria correta
verificada via PDF (era 43 % antes).

### 2.2 Cross-reference sistemático

Documento `_validacao_cross_reference_v3.md`: cruzamento de cada
elemento da tese (objetivo, 6 OEs, 7 contribuições, 6 hipóteses,
storytelling) contra as 104 fichas. **Veredito**: tese fundamentada,
sem conflitos não endereçados, pronta para escrita.

### 2.3 Validação externa via NotebookLM

Análise paralela conduzida em **NotebookLM (Google AI Plus)**:
7 perguntas-chave avaliadas contra ~100 fontes do corpus.
**Resultado convergente** com nossa análise interna; 2 sugestões
novas valiosas identificadas e incorporadas.

### 2.4 Rodada 8 — diversidade fenotípica intra-Latinx

Resposta direta a uma das sugestões da análise NotebookLM. **3 novas
fichas** integradas para sustentar empíricamente a hipótese H3 +
contribuição C6 com **tripé antropologia + genética + sociologia**.

---

## 3. Validação externa via NotebookLM (Google AI Plus)

### 3.1 Por que usar NotebookLM como ferramenta de validação

- **Independente** da minha análise interna — segunda opinião externa.
- **Operacional**: 4 tiers de relevância de PDFs preparados para
  importação (limite ~300 fontes/notebook).
- **Auditável**: cada resposta cita as fontes específicas.

### 3.2 Cobertura da análise

7 perguntas-chave avaliadas:

1. A tese realmente é nova?
2. Tem valor científico?
3. Existem estudos que divergem ou já mais avançados?
4. Faltam artigos mais relevantes?
5. Existem furos e/ou gaps no estudo?
6. Existem estudos mais recentes?
7. O pipeline desenhado é defensável?

### 3.3 Veredito do NotebookLM

> *"O trabalho está muito bem estruturado, possui contribuições
> claras e está fundamentado em literatura de ponta. O ponto crítico
> da defesa será a decomposição do erro Latinx, que é onde reside a
> sua maior contribuição intelectual e diagnóstica."*

### 3.4 Convergência com minha análise interna

| Asserção | Análise interna | NotebookLM |
|---|---|---|
| FiLM em fairness é inédito | ✅ | ✅ |
| Matriz MST × FairFace preenche gap real | ✅ | ✅ |
| Pangelinan é a refutação central | ✅ | ✅ |
| Pipeline tem sequenciamento lógico | ✅ | ✅ |
| Decomposição erro Latinx é maior diferencial | ✅ | ✅ |
| Corpus na fronteira (papers 2026) | ✅ | ✅ |

### 3.5 Sugestões novas (2) — INCORPORADAS

| Sugestão | Ação tomada |
|---|---|
| **Diversidade fenotípica intra-Latinx** — categoria é tratada como bloco monolítico | ✅ Rodada 8 com 3 fichas (Telles + Bryc + Pew) integradas |
| **Sensitivity analysis ao SkinToneNet** — risco de propagação de viés | ✅ OE-2 expandido em v3.5 com validação a 2-3 classificadores MST |

### 3.6 Furos identificados — RESPONDIDOS

| Furo NotebookLM | Resposta documentada |
|---|---|
| Dependência do SkinToneNet | Mitigado via sensitivity analysis (OE-2 v3.5) |
| Escalabilidade (FairFace 108k vs NIST 18M) | Reconhecido na Seção 9 do objetivo v3.5 como limite de escopo (mestrado) — replicação industrial declarada como trabalho futuro |

---

## 4. Rodada 8 — diversidade fenotípica intra-Latinx

### 4.1 Motivação

O F1 ≈ 60 % persistente para Latinx em modelos do estado da arte
(AlDahoul 2024) não tinha **fundamentação empírica explícita** no
nosso corpus de heterogeneidade intra-categoria. Esta rodada
fecha esse argumento.

### 4.2 Três novas fichas integradas (Camada 1)

| Ficha | Autor / Venue | Aporte ao argumento |
|---|---|---|
| **`telles_2014`** (Pigmentocracies) | Edward Telles + PERLA Project — UNC Press 320 pp | **Antropologia**: pigmentocracia documentada em 4 países (Brasil, Colômbia, México, Peru) — tom de pele como dimensão social distinta da raça |
| **`bryc_2015`** (Genetic Ancestry) | Bryc, Reich et al — AJHG (Harvard + 23andMe) | **Genética**: amostra de 162.721 indivíduos — Latinos exibem composição altamente variável de ancestralidade Native American + European + African |
| **`pew_2017_hispanic_identity`** | Lopez, Gonzalez-Barrera et al — Pew Research | **Sociologia**: identidade Hispanic declina 97 % → 50 % cross 4 gerações nos EUA — categoria sociopolítica fluida |

### 4.3 Tripé empírico consolidado para H3 + C6

```
   ANTROPOLOGIA          GENÉTICA            SOCIOLOGIA
   (Telles 2014)        (Bryc 2015)         (Pew 2017)
         ↓                   ↓                   ↓
   pigmentocracia      heterogeneidade     fluidez identitária
   em 4 países         162k indivíduos     cross gerações
         ↓                   ↓                   ↓
         └─────────────────┴───────────────────┘
                           ↓
              H3 + C6 (esta dissertação)
        patamar computacional sobre fundamento
            empírico de 3 disciplinas
```

---

## 5. Estado consolidado do corpus

| Métrica | Valor |
|---|---|
| Total de fichas | **104** |
| VERIFIED | **103** (99 %) |
| OVERVIEW_ONLY | 1 (Springer paywall) |
| PDFs no repositório | 103 |
| Tracks temáticos | 11 (A-L) |
| Bibliografia `.bib` | 104 entradas |
| Período mais recente | 5 fichas de 2026 + 22 de 2025 (26 %) |
| Marcos canônicos | 7 fichas 2018-2021 (Buolamwini, Perez, Madras, Fuentes, etc.) |

---

## 6. Decisões a alinhar hoje

### 6.1 Promover OE-6 (formalizar H6)

Após Pangelinan 2023 ser lido integralmente e Bryc 2015 ter sido
incorporado (Rodada 8), recomenda-se **formalizar a decomposição
quantitativa de variância pixel info × skin tone** como
**Objetivo Específico OE-6** (era apenas H6 auxiliar).

**Pergunta**: o orientador concorda em promover H6 para OE-6
formal?

### 6.2 Estrutura final do estudo comparativo (4 configurações)

Cap 2 conduzirá ablation com 4 configurações:

| Config | Arquitetura |
|---|---|
| A | ConvNeXt-T baseline (sem conditioning) |
| **B** | **ConvNeXt-T + FiLM (MST 10-dim)** — proposta principal |
| C | ConvNeXt-T + Gated FiLM (variante não-linear) — ablação |
| D | ConvNeXt-T + FiLM (CLIP-text embedding) — avaliação alternativa |

**Pergunta**: este desenho atende à recomendação de "avaliar CLIP
como alternativa moderna" feita na reunião de 15/jun?

### 6.3 Próximos passos da escrita

- **Esta semana**: começar **Capítulo 1 — Introdução**
  (estrutura no repositório base do GitHub).
- **Semanas 2-3 (29/jun-12/jul)**: Cap 2 (Revisão) + Cap 3
  (Objetivos) + Cap 4 (Metodologia + ampliação técnicas).
- **Semana 4 (13-15/jul)**: revisão final + entrega.

**Pergunta**: alguma orientação adicional sobre estrutura LaTeX,
template Overleaf ou padrão ABNT específico da Unifesp/ICT?

---

## 7. Próximas ações imediatas (semana 29/jun-5/jul)

1. **Setup Overleaf** com template institucional (esta semana).
2. **Importar bibliografia consolidada** (`referencias.bib`).
3. **Iniciar escrita do Capítulo 1** baseado em
   `_pre_qualificacao_narrativa.md` v1.2.
4. **Promover OE-6** se aprovado.
5. **Aprofundar leitura** seletiva da Camada 2 conforme escrita
   for evoluindo (paralelo).

---

## Anexos

- [_validacao_cross_reference_v3.md](_validacao_cross_reference_v3.md) — auditoria sistemática 104 fichas × tese
- [_objetivo_tese_v3.3.md](_objetivo_tese_v3.3.md) — v3.5 atualizada com OE-6 e sensitivity analysis
- [_pre_qualificacao_narrativa.md](_pre_qualificacao_narrativa.md) — v1.2 com 3 novas seções (Latinx, escala, ponte genética/CV)
- [_mapa_citacoes_por_capitulo.md](_mapa_citacoes_por_capitulo.md) — 104 fichas alocadas por capítulo
- [_tiers_relevancia_pdfs.md](_tiers_relevancia_pdfs.md) — organização para NotebookLM
- [_rodada_08_latinx_candidatos.md](_rodada_08_latinx_candidatos.md) — Rodada 8 parcialmente integrada
- `docs/tese/referencias.bib` — bibliografia consolidada (104 entradas)
