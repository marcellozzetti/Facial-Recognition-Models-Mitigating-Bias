---
name: raumanns-2024
status_verificacao: VERIFIED
autores: [Ralf Raumanns (Fontys University of Applied Science, Eindhoven), Gerard Schouten, Josien P. W. Pluim (Eindhoven University of Technology), Veronika Cheplygina (IT University of Copenhagen)]
ano: 2024
titulo: "Dataset Distribution Impacts Model Fairness: Single vs. Multi-Task Learning"
venue: "FAIMI EPIMI 2024 Workshop — Lecture Notes in Computer Science vol 15198"
tipo_publicacao: workshop_conference
arxiv_id: "2407.17543"
doi: null
url_primario: https://arxiv.org/abs/2407.17543
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/raumanns_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract com 5 observações lido via pdftotext.
---

# Single vs Multi-Task Learning Fairness (Raumanns et al., 2024)

> **Cautela contra multi-task naive**: paper demonstra que
> reinforcement multi-task NÃO remove sex bias, mas adversarial
> learning REMOVE em alguns cenários. Domínio: skin lesion
> classification. Reforça nossa decisão de incluir Adversarial
> debiasing como baseline forte.

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

A influência de viés em datasets sobre a fairness das predições é
tópico ativo. Avalia performance de skin lesion classification via
ResNet-based CNNs, focando em **variações de patient sex no treino**
e **3 estratégias de aprendizado distintas**.

## 2. Método

> *Fonte: abstract verbatim.*

- **Linear programming method** para gerar datasets com distribuições
  variáveis de sex × class labels (com correlations modeladas).
- **3 learning strategies** avaliadas:
  1. **Single-task model** — ResNet baseline.
  2. **Reinforcing multi-task model** — multi-task com objetivo de
     reinforço.
  3. **Adversarial learning scheme** — adversarial debiasing.

> **[PENDENTE PDF]** Hiperparâmetros. Arquitetura específica do
> adversarial scheme. Datasets-fonte (ISIC? HAM10000?).

## 3-6. Datasets, métricas, resultados, limitações

> *Fonte: abstract verbatim, 5 observações:*

1. **Sex-specific training data yields better results.**
2. **Single-task models exhibit sex bias.**
3. **Reinforcement multi-task approach does NOT remove sex bias.**
4. **Adversarial model eliminates sex bias** (apenas com female-only
   patients).
5. **Datasets com male patients melhoram performance for male
   subgroup** mesmo quando female patients são maioria.

> **[PENDENTE PDF]** Números específicos. Múltiplas raças? Apenas
> sex tested.

## 7. Limitações que identifiquei (a partir do abstract apenas)

- **Domínio é skin lesion**, não face — translação para nossa
  pesquisa requer cuidado.
- **Apenas sex binário** testado — multi-classe não validado.
- **Adversarial model funciona APENAS em female-only setup** —
  limitação importante reconhecida.
- **Workshop paper** — escopo menor que main conference.
- **Reinforcing multi-task** definido sem detalhes no abstract.

## 8. Relação com nossa pesquisa

### 8.1 Cautela contra multi-task naive

Reforça nossa decisão (registrada em [[../_validacao_cientifica_pipeline]]
Risco D): **multi-task sem mecanismo fairness-aware NÃO basta**.

Em conjunto com Aguirre & Dredze 2023 (que mostra que multi-task COM
fairness loss funciona), conclusão metodológica é clara: **mecanismo
de fairness explícito é necessário**.

### 8.2 Adversarial como baseline forte

Raumanns demonstra que adversarial scheme **elimina sex bias** em
cenário específico. Reforça inclusão de **Adversarial debiasing
(Zhang 2018)** na lista de baselines do Cap 2 — decisão tomada
na Rodada 6.

### 8.3 Insight sobre "sex-specific data yields better results"

Achado 1 (sex-specific data is best) é contra-intuitivo mas
consistente com **Kolla & Savadamuthu 2022** (skewed-toward-Black
supera uniforme para Black group).

Sugere padrão geral: **distribuição alvo-específica supera distribuição
uniforme** para grupos sub-representados. Direção a explorar como
ablation no Cap 2.

## 9. Pontos para citar

- *"Raumanns et al. (2024), em estudo apresentado no FAIMI EPIMI
  2024 Workshop, demonstram que reinforcement multi-task learning
  NÃO remove sex bias em classificação de lesão de pele, mas que
  esquemas adversariais podem eliminar viés em cenários específicos.
  Esta evidência sustenta a inclusão de adversarial debiasing
  (Zhang et al., 2018) como baseline forte da presente dissertação,
  em contraposição a abordagens multi-task naive que demonstram
  ser insuficientes para mitigação efetiva de viés."*

## 10. Arquivos relacionados

- PDF: pendente em `pdfs/raumanns_2024.pdf`.
- Análise R6 em [[../_validacao_cientifica_pipeline]] (R6-8).
- Entradas relacionadas: [[aguirre_2023]] (multi-task COM fairness
  loss funciona — complementar), [[zhang_2018]] (Adversarial
  debiasing original), [[madras_2018]] LAFTR, [[kolla_2022]]
  (skewed > uniform — achado paralelo).

## 11-12. Pendente PDF

> **[BLOQUEADO]** Future work e análise crítica detalhada requerem
> leitura integral.
