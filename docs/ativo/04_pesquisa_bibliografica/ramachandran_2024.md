---
name: ramachandran-2024
status_verificacao: VERIFIED
autores: [Sreeraj Ramachandran (Wichita State University), Ajita Rattani (University of North Texas)]
ano: 2024
titulo: "A Self-Supervised Learning Pipeline for Demographically Fair Facial Attribute Classification"
venue: "IEEE International Joint Conference on Biometrics (IJCB 2024)"
tipo_publicacao: conference
arxiv_id: "2407.10104"
doi: null
url_primario: https://arxiv.org/abs/2407.10104
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/ramachandran_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

# SSL Pipeline for Fair Facial Attribute Classification (Ramachandran & Rattani, 2024)

> **Self-Supervised Learning sem manual labels** + pseudo-labels via
> pre-trained encoders + meta-learning contrastive learning.
> Estabelece **new benchmark for SSL in fairness of facial attribute
> classification**. Baseline competitivo em FairFace + CelebA.

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

Mitigação de viés em facial attribute classification tem dependido de
**supervised learning** com labels demográficos manuais. Mas labels
manuais têm 3 problemas:
1. Custo de anotação alto
2. Riscos de privacidade
3. Podem perpetuar viés humano

**Self-supervised learning (SSL)** capitaliza unlabeled data, mas
SSL **também pode introduzir viés** via false negative pairs em
low-data regimes (~200K imgs).

## 2. Método

> *Fonte: abstract verbatim.*

**Pipeline fully self-supervised** com 3 componentes:

1. **Pseudo-labeling** via pre-trained encoders.
2. **Diverse data curation techniques**.
3. **Meta-learning-based weighted contrastive learning**.

> **[PENDENTE PDF]** Algoritmo exato de pseudo-labeling. Detalhes
> do meta-learning weighting. Architecture do encoder.

## 3-6. Datasets, métricas, resultados, limitações

- **Datasets**: FairFace + CelebA.
- **Resultado declarado**: *"significantly outperforms existing SSL
  approaches"* e *"sets a new benchmark for SSL in fairness of
  facial attribute classification."*

> **[PENDENTE PDF]** Números específicos. Comparação contra
> baselines supervised (FSCL+, FineFACE, etc.). Métricas de fairness
> exatas.

## 7. Limitações que identifiquei (a partir do abstract apenas)

- **Self-Supervised Learning** ainda pode introduzir bias via false
  negative pair sampling (autores reconhecem).
- **Quality assurance** de dados web pode degradar performance.
- **Sem comparação contra supervised SOTA** explicitamente — abstract
  só fala em "outperforms existing SSL approaches".
- **CelebA** é facial attribute padrão; **FairFace race target** não
  explícito no abstract — pode ser gender ou age.

## 8. Relação com nossa pesquisa

### 8.1 Baseline SSL para Cap 2

Nossa pesquisa usa supervised learning (FairFace train com race
labels). SSL não substitui nossa abordagem, mas é candidato a
baseline alternativo para mostrar Pareto-superioridade da nossa
abordagem condicionada (MST + FiLM).

### 8.2 Insight sobre pseudo-labeling

Ramachandran usa pseudo-labels via pre-trained encoders. Nossa
Pereira 2026 (SkinToneNet) faz papel similar: classifier
pré-treinado que provê pseudo-rótulos MST para FairFace.

Conceitualmente próximo, ainda que aplicação seja diferente.

### 8.3 Trade-off SSL vs supervised

Se nossa pesquisa quiser argumentar a favor de **supervised**, vale
citar Ramachandran como evidência de que **SSL puro tem limitações
inerentes em fairness** (false negatives, quality assurance).

## 9. Pontos para citar

- *"Ramachandran & Rattani (2024), apresentado no IJCB 2024,
  estabelecem benchmark para self-supervised learning em fairness
  de facial attribute classification, combinando pseudo-labeling
  via encoders pré-treinados, curadoria diversa de dados e
  meta-learning weighted contrastive learning. Reconhecem
  explicitamente que SSL pode introduzir viés via false negative
  pair sampling em regimes de baixos dados (~200K imagens),
  limitando aplicabilidade no estado-da-arte de fairness."*

## 10. Arquivos relacionados

- PDF: pendente em `pdfs/ramachandran_2024.pdf`.
- Análise R6 em [[../_validacao_cientifica_pipeline]] (R6-7).
- Entradas relacionadas: [[park_2022]] (FSCL — contrastive
  supervised), [[pereira_2026]] (SkinToneNet — pre-trained
  encoder análogo), [[dataset_karkkainen_2021]] FairFace.

## 11-12. Pendente PDF

> **[BLOQUEADO]** Future work e análise crítica detalhada requerem
> leitura integral.
