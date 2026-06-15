---
name: survey-racial-bias-fr-2024
status_verificacao: VERIFIED
autores: [Seyma Yucer (Durham University), Furkan Tektas (VisAI Istanbul), Noura Al Moubayed (Durham), Toby Breckon (Durham)]
ano: 2024
titulo: "Racial Bias within Face Recognition: A Survey"
venue: "ACM Computing Surveys (CSur), vol 57, no 4, Article 105, December 2024 — 39 páginas"
tipo_publicacao: journal
arxiv_id: null
doi: "10.1145/3705295"
url_primario: https://dl.acm.org/doi/10.1145/3705295
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: 39
lente_disrupcao: nenhuma
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN institucional (pdfs/survey_racial_bias_fr_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução e estrutura por pipeline de FR lidos via pdftotext. CC BY 4.0.
---

# Survey: Racial Bias within Face Recognition (Yucer, Tektas, Al Moubayed & Breckon — ACM CSur 2024)

> **Survey extenso e taxonômico de viés racial em FR** (39 páginas) —
> Durham University. Estruturado por **4 estágios do pipeline de FR**
> (acquisition, localisation, representation, verification/identification).
> Complementa [[survey_kotwal_2025]] (broader demographic) com foco
> específico em raça.

## 1-2. Resumo + estrutura

O paper organiza a literatura em três partes:

1. **Definição do problema de viés racial**:
   - Definição de raça
   - Estratégias de grouping
   - Implicações sociais de usar raça (ou grupos raciais)

2. **Revisão por estágio do pipeline FR (4 estágios)**:
   - Image acquisition
   - Face localisation
   - Face representation
   - Face verification / identification

3. **Pitfalls e limitações** das estratégias atuais de mitigação para
   pesquisa futura e aplicações comerciais.

## 7. Aplicação ao pipeline v3.2

- **Citação canônica** para Cap 1 (problemas existentes) — viés
  racial é tema central da nossa dissertação.
- **Estrutura por estágios do pipeline** alinha com a nossa
  organização do Cap 4 (Metodologia) por etapas.
- Complementa [[survey_kotwal_2025]] (broader demographic) e
  [[survey_mehrabi_2021]] (broader ML).
- ACM CSur — venue top (IF alto). Cita de mãos cheias.

## 8. Citar

- *"Yucer, Tektas, Al Moubayed e Breckon (2024), em survey de 39
  páginas publicado em ACM Computing Surveys, oferecem revisão
  taxonômica extensa do problema de viés racial em face recognition
  estruturada por estágio do pipeline de processamento facial:
  image acquisition, face localisation, face representation, face
  verification e identification. A revisão consolida tanto causas
  documentadas quanto limitações de estratégias contemporâneas de
  mitigação, configurando-se como referência canônica do tema."*

## 9-12.

PDF: `pdfs/survey_racial_bias_fr_2024.pdf` (Creative Commons
Attribution 4.0). Durham University + VisAI Istanbul. Conexões:
[[survey_kotwal_2025]] (FR fairness broader), [[survey_mehrabi_2021]]
(ML fairness taxonomia), [[pangelinan_2023]] (causas), [[grother_2019]]
(NIST escala).
