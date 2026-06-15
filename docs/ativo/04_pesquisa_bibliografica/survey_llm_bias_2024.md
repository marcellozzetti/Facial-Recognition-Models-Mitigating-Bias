---
name: survey-llm-bias-2024
status_verificacao: VERIFIED
autores: [Isabel O. Gallegos (Stanford), Ryan A. Rossi (Adobe), Joe Barrow (Pattern Data), Md Mehrab Tanjim (Adobe), Sungchul Kim (Adobe), Franck Dernoncourt (Adobe), Tong Yu (Adobe), Ruiyi Zhang (Adobe), Nesreen K. Ahmed]
ano: 2024
titulo: "Bias and Fairness in Large Language Models: A Survey"
venue: "Computational Linguistics, MIT Press / ACL Anthology, 2024"
tipo_publicacao: journal
arxiv_id: null
doi: null
url_primario: https://aclanthology.org/2024.cl-3.8/
citacoes_semantic_scholar: 36
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: nenhuma
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN institucional (pdfs/survey_llm_bias_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

# Survey: Bias and Fairness in LLMs (Gallegos, Rossi et al. — Stanford + Adobe Research, 2024)

> Survey extenso sobre bias e fairness em **Large Language Models**
> publicado em **Computational Linguistics (MIT Press / ACL)** — venue
> top da área NLP. Autoria majoritariamente Adobe Research +
> Stanford. Cobertura de métricas, técnicas de avaliação e mitigação
> em LLMs.

## 1-2. Resumo

- Survey amplo sobre viés em LLMs.
- Técnicas de avaliação e mitigação organizadas por taxonomia.
- 36 citações (Semantic Scholar 2026-06).
- **Modalidade text-only** — não cobre vision direto.

## 7. Aplicação ao pipeline v3.2

- **Track I (VLM fairness)** — referência auxiliar; VLMs como CLIP
  e BLIP herdam encoders de texto com viés documentado nos LLMs.
- **Cap 1** (Introdução): contexto para discussão de viés herdado
  cross-modal.
- **Não substitui** nossas referências core de FR fairness.

## 8. Citar

- *"Gallegos, Rossi et al. (2024), survey publicado em Computational
  Linguistics (MIT Press / ACL Anthology), consolidam técnicas de
  avaliação e mitigação de viés em large language models na
  modalidade text-only — base relevante para entender vieses
  herdados nos componentes textuais de vision-language models como
  CLIP e BLIP, considerados como abordagem alternativa de
  conditioning na presente dissertação."*

## 9-12.

PDF: `pdfs/survey_llm_bias_2024.pdf`. Conexões: [[luo_2024_fairclip]]
(FairCLIP — debiasing VL específico), [[survey_multimodal_fairness_2024]]
(multimodal fairness — paralelo), [[survey_fairness_vision_lang_2024]]
(survey vision+language por PUCRS).
