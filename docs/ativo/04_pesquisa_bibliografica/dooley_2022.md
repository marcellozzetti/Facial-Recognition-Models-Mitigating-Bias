---
name: dooley-2022
status_verificacao: OVERVIEW_ONLY
autores: [Samuel Dooley, Rhea Sanjay Sukthanker, John P. Dickerson, Colin White, Frank Hutter, Micah Goldblum]
ano: 2022
titulo: "Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition"
venue: "arXiv preprint 2210.09943 (versão final dez 2023)"
tipo_publicacao: preprint
arxiv_id: "2210.09943"
doi: null
url_primario: https://arxiv.org/abs/2210.09943
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: Apenas abstract via arXiv (PDF integral pendente). Contexto adicional via [[../_validacao_cientifica_pipeline]] R6-3.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**
>
> Construída apenas a partir do abstract. PDF integral pendente.
> Promover a VERIFIED requer leitura do PDF + reescrita das seções
> de método, datasets, hiperparâmetros e resultados.

# Fairer Architectures Make for Fairer FR (Dooley et al., 2022)

> **Reformulação paradigmática**: "biases são inerentes a arquiteturas
> neurais, não apenas aos dados". **First NAS for fairness** —
> arquitetura encontrada Pareto-domina baselines de mitigação.
> Reforça H2 da nossa tese e valida ConvNeXt-T como backbone moderno.

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

Sabedoria convencional atribui viés em FR aos dados de treino.
Mitigações tradicionais focam em pre-/in-/post-processing dos dados,
com sucesso limitado.

**Tese central do paper**: viés é também propriedade da
**arquitetura neural** — e Neural Architecture Search (NAS) pode
encontrar arquiteturas Pareto-superiores em accuracy E fairness.

## 2. Método

> *Fonte: abstract verbatim.*

- **First NAS for fairness** — busca de arquitetura jointly com
  hiperparâmetros.
- Objetivo bi-critério: accuracy + fairness.
- Encontra "uma suíte de modelos" que Pareto-dominam baselines.

> **[PENDENTE PDF]** Espaço de busca exato. Métrica de fairness usada
> no objetivo (EOD? DR?). Compute para NAS. Convergência.

## 3. Datasets e setup

- **CelebA** e **VGGFace2** (datasets primários para face identification).
- Generalização para "outros datasets e sensitive attributes"
  declarada.

> **[PENDENTE PDF]** Detalhe de RFW/BFW se usados. Protocolos.

## 4-6. Métricas, resultados específicos, limitações

> **[PENDENTE PDF]** Abstract reporta apenas qualitativamente:
> "Pareto-dominam por margens grandes" e "generalizam para outros
> datasets". Sem números absolutos no abstract.
>
> Código em github.com/dooleys/FR-NAS.

## 7. Limitações que identifiquei (a partir do abstract apenas)

- **Custo computacional altíssimo** — NAS é proibitivo para a maioria
  dos labs.
- **Arquiteturas vencedoras** podem não ter interpretação semântica
  clara — caixa-preta nova.
- **CelebA e VGGFace2** como datasets primários — race classification
  7-class do FairFace não testado especificamente.
- **ConvNeXt-T** não mencionado especificamente — não está no espaço
  de busca testado.

## 8. Relação com nossa pesquisa

### 8.1 Reforça H2

Nossa H2: "trocar de ResNet-34 para ConvNeXt-T por si só reduz
disparity (ou não? — investigado experimentalmente)".

Dooley fornece **suporte conceitual**: arquitetura por si só importa.
Justifica investigar H2 isoladamente como ablation no Cap 2.

### 8.2 Valida escolha de ConvNeXt-T

- ResNet-34 (FairFace original) é arquitetura de 2015.
- ConvNeXt-T (Liu 2022) é arquitetura moderna que Pareto-domina
  ResNet em accuracy/fairness em outros contextos.
- Dooley **não testa ConvNeXt-T**, mas o paradigma "arquitetura
  importa" valida nossa escolha.

### 8.3 Diferenciação metodológica

- Dooley: NAS para descobrir arquitetura **inerentemente fair**.
- Nossa pesquisa: arquitetura fixa (ConvNeXt-T) + **conditioning
  mechanism** (FiLM com MST) — mecanismo de mitigação separado.

Não conflita com Dooley — é abordagem complementar.

## 9. Pontos para citar

- *"Dooley et al. (2022) reformulam o paradigma de bias mitigation
  em FR ao demonstrar, via Neural Architecture Search, que
  arquiteturas neurais carregam viés inerente independente dos
  dados de treino. Sua suíte de arquiteturas Pareto-domina modelos
  ResNet e MobileNet em accuracy e fairness simultaneamente. Esta
  observação justifica conceitualmente a escolha desta dissertação
  por ConvNeXt-T (Liu et al. 2022) como backbone moderno, em
  contraste com o ResNet-34 do FairFace original (Kärkkäinen & Joo,
  2021)."*

## 10. Arquivos relacionados

- PDF: pendente em `pdfs/dooley_2022.pdf`.
- Code: github.com/dooleys/FR-NAS.
- Análise R6 em [[../_validacao_cientifica_pipeline]] (R6-3).
- Entradas relacionadas: [[manzoor_2024]] (FineFACE — outra
  abordagem arquitetural), [[lin_2022]] (FairGRAPE — modificação
  via pruning), [[dataset_karkkainen_2021]] (FairFace ResNet-34
  baseline).

## 11-12. Pendente PDF

> **[BLOQUEADO]** Future work e análise crítica detalhada requerem
> leitura integral.
