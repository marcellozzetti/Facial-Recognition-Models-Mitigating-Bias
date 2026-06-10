---
name: aguirre-2023
status_verificacao: VERIFIED
autores: [Carlos Aguirre, Mark Dredze]
ano: 2023
titulo: "Transferring Fairness using Multi-Task Learning with Limited Demographic Information"
venue: "arXiv preprint 2305.12671 (versão v2 revisada abril 2024)"
tipo_publicacao: preprint
arxiv_id: "2305.12671"
doi: null
url_primario: https://arxiv.org/abs/2305.12671
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: HTML integral via arxiv.org/html/2305.12671v2, lido em 2026-06-10.
---

# Transferring Fairness via Multi-Task Learning (Aguirre & Dredze, 2023/2024)

> **Reforço empírico do paradigma fair transfer** (Madras 2018 LAFTR).
> Demonstra que fairness aprendida em uma tarefa **transfere via
> framework multi-task** mesmo quando demographic labels não estão
> disponíveis na tarefa-alvo. Domínio: NLP (clinical, reviews, Twitter).
> Princípio é independente da modalidade — sustenta etapas 3 e 5 do
> nosso pipeline v3.2.

## 1. Resumo do problema atacado

**Trade-off operacional em fair ML**: técnicas de mitigação de viés
demográfico exigem rótulos demográficos no conjunto de treino. Mas
para a maioria das tarefas reais, esses rótulos **não estão
disponíveis** (custo, privacidade, regulação).

Pergunta de pesquisa: **podemos transferir fairness de uma tarefa
relacionada COM labels demográficos para uma tarefa-alvo SEM esses
labels** via multi-task learning com encoder compartilhado?

## 2. Método

### 2.1 MTL-fair — arquitetura

- **Encoder compartilhado** θ_s entre 2 tarefas.
- **Cabeças task-specific**: θ_A (alvo, sem demographic labels) e
  θ_B (auxiliar, com labels).
- **Fairness loss** aplicado apenas à tarefa B (que tem labels).

### 2.2 Métrica ε-DEO (ε-Differential Equalized Odds)

Adaptação diferenciável de Equalized Odds (Hardt 2016) para uso em
training time. Equaliza recall e specificity entre subgrupos
demográficos, incluindo subgrupos intersectional.

### 2.3 Objetivo de treino

```
Loss = (1/|A||B|) · [Σ L(x_A; θ_s∪θ_A) + L(x_B; θ_s∪θ_B)]
       + λ · max(0, ε(B; θ_s∪θ_B) − ε_t)
```

- **λ**: hiperparâmetro de balanço prediction/fairness.
- **ε_t**: nível-alvo de fairness (0 = paridade perfeita).

### 2.4 MTL-inter — extensão intersectional

Quando tarefas A e B têm **eixos demográficos diferentes** (e.g., A
tem gender, B tem race), aplica ε-DEO em ambas simultaneamente →
**fairness intersectional sem labels intersection explícitos**.

### 2.5 Detalhes de otimização

- SGD com autodiferenciação.
- Período de **burn-in** + stochastic approximation updates.
- Hiperparâmetro **ρ** para smoothing de demographic counts em
  mini-batches.

## 3. Datasets e setup experimental

6 tarefas em 3 domínios:

| Domínio | Tarefa | Demographic | Tamanho train |
|---|---|---|---|
| **Clínico** | In-hospital Mortality | Gender (2) | 13.191 |
| Clínico | Phenotyping | Gender (2) | 13.839 |
| **Reviews online** | Sentiment | Gender+Age (4) | 58.259 |
| Reviews | Topic | Gender+Age (4) | 14.744 |
| **Mídia social** | Twitter Sentiment | Race (2: AAE/SAE) | 156.000 |
| Mídia social | HateXplain | Race (5) | 5.376 |

**Fontes**: MIMIC-III (ICU notes), Trustpilot (reviews), Twitter/Gab
(race proxies linguísticos/geo).

## 4. Métricas reportadas

- **F1** (utility).
- **ε-DEO** (fairness; menor é melhor).
- Per-subgroup F1 para intersection cases.

## 5. Resultados principais (valores numéricos)

### 5.1 Within-domain MTL-fair (Tabela 3)

| Tarefa | STL-base F1 | STL-base ε-DEO | MTL-fair F1 | MTL-fair ε-DEO | Redução ε-DEO |
|---|---|---|---|---|---|
| In-hospital Mortality | 62.1% | 0.25 | 64.0% | 0.19 | **−24%** |
| Phenotyping | 53.6% | 0.28 | 53.0% | 0.21 | **−25%** |
| Twitter Sentiment | 76.4% | 0.33 | 75.5% | 0.28 | **−15%** |

**Achado-bandeira**: em 3 de 4 tarefas, MTL-fair atinge fairness
**igual ou superior** a STL-fair (que tem acesso a labels da própria
tarefa).

### 5.2 Intersectional (Tabela 4 — Reviews)

| Tarefa | STL-base | STL-fair | MTL-inter | Redução vs STL-fair |
|---|---|---|---|---|
| Sentiment ε-DEO | 0.77 | 0.77 | **0.58** | **−25%** |
| Topic ε-DEO | 1.04 | 1.04 | **0.82** | **−21%** |

Performance F1 mantida (84.1 vs 84.5 sentiment; 91.6 vs 91.9 topic).

### 5.3 Cross-domain transfer (Tabela 5 — Twitter Sentiment com auxiliares)

| Auxiliar | ε-DEO target | Redução |
|---|---|---|
| HateXplain | 0.28 | −15% |
| Review sentiment | 0.23 | −18% |
| Review topic | 0.23 | −18% |

### 5.4 STILT vs MTL ablation (HateXplain)

- STILT-fair (consecutivo): ε-DEO = 1.42
- MTL-fair (simultâneo): ε-DEO = 0.80
- **Redução: −44%** → **treinamento simultâneo é essencial**, não
  pretrain → finetune.

## 6. Limitações declaradas pelos autores

1. **Critério de seleção de modelo** exige demographic labels no
   test set da tarefa-alvo — contradição parcial com motivação
   "limited demographic".
2. **Compatibilidade entre tarefas**: phenotyping teve desempenho
   ruim em MTL — fairness loss pode não ter efeito.
3. **Validação sem demographic da target**: em 2/4 datasets, seleção
   sem demographic da target degrada fairness vs single-task.
4. **MIMIC-III muito skewed** — intersection groups muito pequenos
   inviabilizam fairness intersectional clínica.
5. **Escassez geral de datasets NLP com demographics** restringe
   aplicabilidade.

## 7. Limitações que identifiquei

- **NLP only**: não validado em CV/face. Princípio se transfere mas
  precisamos demonstrar empíricamente em imagem facial.
- **Magnitude da transferência depende de correlação entre tarefas**:
  para nosso caso (race em FairFace) precisamos tarefa auxiliar
  com demographic labels que se correlacione — Casual Conversations
  (Hazirbas 2021) ou MST-E (Schumann 2023) são candidatos.
- **Sem teste sob distribution shift** — robustez não caracterizada.
- **Apenas atributo sensível único por experimento principal** —
  intersectional só em ablation.
- **Encoder compartilhado** é monolítico — não há condicionamento
  arquitetural como FiLM. Nossa abordagem é diferente: FiLM injeta
  contexto, MTL-fair compartilha encoder. **Mecanismos
  complementares.**

## 8. Relação com nossa pesquisa

### 8.1 Centralidade conceitual

Aguirre & Dredze 2023 é a **demonstração empírica** que faltava à
prova teórica do LAFTR (Madras 2018). Confirma que **fairness é
propriedade da representação**, transferível entre tarefas via
encoder compartilhado.

### 8.2 Aplicação ao pipeline v3.2

| Etapa do pipeline | Como Aguirre se aplica |
|---|---|
| Etapa 3 (race + tom como contexto) | Justifica usar SkinToneNet como tarefa auxiliar implícita — fairness aprendida em "skin tone classification" transfere para "race classification" via FiLM-conditioning |
| Etapa 5 (transferência para FR) | Aguirre confirma que fairness aprendida no Cap 2 (race) pode transferir para Cap 3 (face recognition) sem re-treinar fairness loss do zero |

### 8.3 Diferenciação metodológica

- **Aguirre**: encoder compartilhado, multi-task simultâneo.
- **Nossa abordagem**: FiLM-conditioning explícito via vetor MST
  como contexto. Mais granular (modulação por canal) e mais
  expressivo (transformação afim aprendida).

Combinação possível mas não testada: FiLM-conditioned encoder em
framework multi-task.

### 8.4 Risco D mitigado (parcialmente)

Risco D de [[../_validacao_cientifica_pipeline]] era "multi-task naive
falha (Raumanns 2024)". Aguirre demonstra que **multi-task COM
fairness loss explícito funciona**. Confirma necessidade de
mecanismo fairness-aware, não apenas multi-task naive.

## 9. Pontos para citar

- *"Aguirre & Dredze (2023) demonstram empiricamente que objetivos
  de fairness demográfico transferem entre tarefas em um framework
  multi-task com encoder compartilhado, com reduções de 15-25% em
  ε-DEO mantendo performance, e até 44% de redução em ablation
  contra STILT (treinamento sequencial). Esta evidência empírica
  sustenta o princípio teórico de fair transfer estabelecido por
  Madras et al. (LAFTR, 2018)."*

- *"A configuração MTL-inter de Aguirre & Dredze (2023), que combina
  fairness losses sobre eixos demográficos distintos em tarefas
  paralelas, alcança fairness intersectional (gender × race) sem
  exigir labels intersection — evidência metodológica diretamente
  relevante para a etapa de condicionamento por tom de pele no
  pipeline desta dissertação."*

- *"Aguirre & Dredze (2023) demonstram (Apêndice B) que treinamento
  multi-task SIMULTÂNEO supera treinamento sequencial (STILT) em
  44% na redução de ε-DEO, sustentando arquiteturalmente nossa
  decisão de inserir blocos FiLM end-to-end no backbone race
  classifier, em vez de fine-tuning sequencial sobre representações
  pre-trained."*

## 10. Arquivos relacionados

- HTML: arxiv.org/html/2305.12671v2 (lido em 2026-06-10).
- PDF: pendente download para `pdfs/aguirre_2023_mtl_fair.pdf`.
- Code: a verificar disponibilidade pública.

### Entradas relacionadas

- [[madras_2018]] LAFTR — fundamento teórico que Aguirre valida
  empiricamente.
- [[zemel_2013]] LFR — paradigma fair representation learning.
- [[hardt_2016]] EO_h/EOD — ε-DEO é variante diferenciável.
- [[raumanns_2024]] — multi-task naive falha; Aguirre mostra que
  multi-task COM fairness loss funciona.
- [[pereira_2026]] SkinToneNet — provê classificador auxiliar para
  nossa aplicação do princípio em CV facial.
- [[perez_2018]] FiLM — mecanismo arquitetural alternativo (mais
  granular que encoder compartilhado).

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído do paper:

- **Estender para mais axis demográficos** (não só gender, race) —
  parcialmente alinhado com nosso Cenário B (race × gender).
- **Aplicar a domínios não-NLP** (CV é candidato natural). ✅
  Diretamente alinhado com nossa pesquisa.
- **Investigar dependência da correlação entre tarefas** — paper
  observa que phenotyping falha; gap a estudar.
- **Métricas de seleção de modelo sem demographic da target** —
  parcialmente atacado no Apêndice E, sem solução robusta.

## 12. Análise crítica do método

### (a) Rigor formal

- ε-DEO matematicamente bem definido como variante diferenciável de
  Equalized Odds. Pressupostos claros.
- Demonstração empírica forte: 3 domínios × 2 tarefas cada =
  6 experimentos.
- **Limitação**: sem prova teórica sobre quando transferência
  funciona vs falha (é empírico).

### (b) Reprodutibilidade

- ✅ Datasets públicos (MIMIC-III, Trustpilot, Twitter).
- ⚠ Sem código mencionado no abstract (a verificar).
- ⚠ Hiperparâmetros ε_t, λ, ρ mencionados mas valores específicos
  não detalhados na extração HTML.
- ⚠ Sem multi-seed mencionado.

### (c) Aplicabilidade ao pipeline v3.2

- **Conceitualmente alta**: princípio "fairness transfere via
  shared representation" é diretamente aplicável.
- **Operacionalmente diferente**: nossa abordagem usa FiLM
  (condicionamento granular), Aguirre usa shared encoder
  (compartilhamento monolítico).
- **Combinação possível mas não testada**: FiLM-conditioned
  multi-task framework. Direção futura.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| ε-DEO sobre demographic parity | ✅ Justificada (Hardt 2016) |
| Encoder compartilhado | ✅ Justificada — testa princípio LAFTR |
| Fairness loss apenas em B | ✅ Justificada — modela limited demographic |
| 3 domínios NLP | ✅ Diversidade boa |
| MTL simultâneo vs STILT | ✅ Empíricamente justificada (Apêndice B) |
| Sem teste em CV | ❌ Choice — escopo |
| Single seed implícito | ❌ Assumida — sem multi-seed |

### (e) Conexão com R5/R6

- [[madras_2018]]: Aguirre é **prova empírica** do que Madras
  prova teoricamente. Par fundamental.
- [[hardt_2016]]: ε-DEO é variante diferenciável.
- [[raumanns_2024]]: Aguirre mostra QUANDO multi-task funciona
  (com fairness loss explícito); Raumanns mostra quando falha
  (sem fairness loss). Complementares.
- [[perez_2018]] FiLM: mecanismo alternativo de condicionamento.
- [[pereira_2026]] SkinToneNet: insumo para aplicação CV do
  princípio.
- **Implicação para v3.2**: Aguirre **fortalece a tese** de fair
  transfer. Reforço empírico do princípio LAFTR.
