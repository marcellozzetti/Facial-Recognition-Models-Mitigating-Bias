---
name: fairer-datasets-2024
status_verificacao: VERIFIED
autores: [Alexandre Fournier-Montgieux (Université Paris-Saclay, CEA, LIST), Michael Soumm (Paris-Saclay CEA LIST), Adrian Popescu (Paris-Saclay CEA LIST), Bertrand Luvison (Paris-Saclay CEA LIST), Hervé Le Borgne (Paris-Saclay CEA LIST)]
ano: 2024
titulo: "Toward Fairer Face Recognition Datasets"
venue: "arXiv preprint (jun 2024)"
tipo_publicacao: preprint
arxiv_id: "2406.16592"
doi: null
url_primario: https://arxiv.org/abs/2406.16592
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado de arXiv (pdfs/fairer_datasets_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract lido via pdftotext. Correção de framing: paper é sobre datasets GERADOS por GenAI, não auditoria de datasets reais.
---

# Toward Fairer FR Datasets (2024) — generative AI + demographic balancing

> **Paper sobre datasets SINTÉTICOS** (não auditoria de datasets reais).
> Propõe **mecanismo de balanceamento demográfico** em datasets
> gerados por GenAI (incluindo diffusion-based) para mitigar viés em
> FR e verification.

## 1-2. Resumo

- Generative AI cria identidades fictícias e endereça privacy, mas
  **problemas de fairness persistem** mesmo em datasets gerados.
- **Contribuição**: mecanismo de balanceamento de atributos
  demográficos aplicado a datasets gerados.
- Experimentos com:
  - 1 dataset real
  - 3 datasets gerados
  - Versões balanceadas de dataset diffusion-based
- Avaliação **simétrica accuracy + fairness** com regressão
  estatística rigorosa.

## 3-5. Achados

- **Balanceamento reduz** demographic unfairness.
- **Gap de performance persiste** mesmo com generation cada vez
  mais acurada — coerente com Pangelinan 2023.

## 7. Aplicação ao pipeline v3.2

- **Não usamos GenAI** em nossa tese — domínio diferente.
- **Relevância indireta**: confirma que mesmo em condições controladas
  de geração balanceada, gap persiste → suporta C5 (mitigação
  algorítmica além de balanceamento).
- **Cap 2** (Revisão): citação contextual sobre direções de mitigação
  por geração de dados (Track L synthetic).

## 8. Citar

- *"Trabalho recente sobre construção fairer de datasets via GenAI
  (arXiv:2406.16592, 2024) demonstra empiricamente que mesmo
  datasets sintéticos diffusion-based exigem mecanismo explícito de
  balanceamento de atributos demográficos para mitigar viés, e que
  gap de performance entre grupos persiste apesar do avanço na
  qualidade de geração. Este resultado reforça a posição desta
  dissertação de que mitigação algorítmica (FiLM-conditioning) é
  necessária complementarmente ao balanceamento de dados."*

## 9-12.

PDF: `pdfs/fairer_datasets_2024.pdf`. Conexões:
[[synthetic_face_2024]], [[frcsyn_2024]], [[variface_2024]]
(família synthetic data — Track L), [[pangelinan_2023]]
(balance não basta).
