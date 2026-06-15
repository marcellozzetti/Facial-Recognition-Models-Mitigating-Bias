---
name: voidface-2025
status_verificacao: VERIFIED
autores: [Ajnas Muhammed (Univ Coimbra), Iurii Medvedev (Univ Coimbra), Nuno Gonçalves (Univ Coimbra ISR)]
ano: 2025
titulo: "VOIDFace: A Privacy-Preserving Multi-Network Face Recognition With Enhanced Security"
venue: "arXiv preprint 2508.07960v1 (ago 2025) — Institute of Systems and Robotics, University of Coimbra, Portugal"
tipo_publicacao: preprint
arxiv_id: "2508.07960"
doi: null
url_primario: https://arxiv.org/abs/2508.07960
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/voidface_2025.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e motivação regulatória (AI Act, GDPR, CCPA, PIPL) lidos via pdftotext.
---

# VOIDFace — Privacy-Preserving Multi-Network FR (Muhammed, Medvedev & Gonçalves — Univ Coimbra 2025)

> **Framework privacy-preserving para FR** focado em **eliminar
> replicação de dados** e proteger contra exposição visual.
> Universidade de Coimbra (ISR), Portugal. **Pesquisa portuguesa**.

## 1-2. Resumo + método

- **Problema duplo identificado**:
  1. **Data replication**: datasets de FR são tipicamente replicados
     em múltiplas workstations — complica gestão e oversight.
  2. **Perda de controle**: uma vez submetida a face, usuário não
     controla mais o uso.
- **Contexto regulatório**: AI Act (EU), EU-GDPR, USA CCPA, China
  PIPL — exigem face data safety.
- **Contribuição VOIDFace**: framework para:
  - Eliminar data replication
  - Melhorar data control
  - Armazenamento seguro
  - Faces visualmente ocultas + difíceis de retrieve

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — federated/privacy) — paralelo a
  [[dp_fedface_2024]], [[federated_fairness_survey_2025]].
- **Não usamos privacy-preserving** — nossa tese é centralized
  classification.
- **Cap 1** (Introdução): contexto regulatório (LGPD análoga a
  GDPR/CCPA/PIPL).
- Conexão com Portugal: outro paper português da nossa lista é
  [[massively_annotated_2024]] (Univ. Porto).

## 8. Citar

- *"VOIDFace (Muhammed, Medvedev & Gonçalves — Institute of Systems
  and Robotics, University of Coimbra, arXiv 2508.07960, 2025)
  endereça privacy em face recognition via framework multi-network
  que elimina data replication e protege contra exposição visual,
  motivado pelo arcabouço regulatório do AI Act europeu, GDPR,
  California CCPA e PIPL chinês — direção complementar à mitigação
  algorítmica de viés explorada nesta dissertação."*

## 9-12.

PDF: `pdfs/voidface_2025.pdf`. Conexões: [[dp_fedface_2024]]
(privacy + FR paralelo), [[federated_fairness_survey_2025]] (survey
federated), [[massively_annotated_2024]] (Univ. Porto — outra
pesquisa portuguesa), [[lafargue_2025]] (regulatório EU AI Act).
