---
name: survey-long-tail-2022
status_verificacao: VERIFIED
autores: [Lu Yang, He Jiang, Qing Song (corresponding, BUPT), Jun Guo (BUPT)]
ano: 2022
titulo: "A Survey on Long-Tailed Visual Recognition"
venue: "International Journal of Computer Vision (IJCV), Springer, vol 130, pp 1837-1872, 2022 (publ. online maio 2022)"
tipo_publicacao: journal
arxiv_id: null
doi: "10.1007/s11263-022-01622-8"
url_primario: https://link.springer.com/article/10.1007/s11263-022-01622-8
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: 36
lente_disrupcao: nenhuma
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN institucional (pdfs/survey_long_tail_2022.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

# Survey: Long-Tailed Visual Recognition (Yang, Jiang, Song & Guo — IJCV 2022 / BUPT)

> Survey IJCV (top venue Springer) sobre long-tailed visual
> recognition pela **Beijing University of Posts and Telecommunications
> (BUPT)** — mesma instituição autora do BUPT-Balancedface
> ([[dataset_bupt_2019]]). Propõe **Gini coefficient** como métrica
> quantitativa para avaliar long-tailedness de datasets.

## 1-2. Resumo + contribuições

- **Problema**: long-tailed distribution é prevalente devido a
  power law na natureza; performance dominada por head classes,
  tail classes sub-aprendidas.
- **Contribuições do survey**:
  1. Sumariza datasets representativos de long-tailed visual
     recognition
  2. Organiza estudos em **10 categorias** sob perspectiva de
     representation learning
  3. Avalia **4 métricas quantitativas** de imbalance
  4. **Propõe Gini coefficient** como métrica recomendada
  5. **Análise de 20 datasets large-scale** dos últimos 10 anos
     usando Gini — long-tailedness é generalizada e ainda não
     plenamente estudada
  6. Direções futuras

## 7. Aplicação ao pipeline v3.2

- **Paradigma adjacente** — long-tail é problema diferente de
  demographic fairness, mas as técnicas (reweighting, resampling,
  margin adjustment) se sobrepõem com baselines do Cap 2 (Sagawa
  Group DRO, FSCL, etc.).
- **Cap 2** (Revisão): citação contextual sobre fronteira
  long-tail × demographic fairness.
- **Gini coefficient** como métrica auxiliar — possível
  ferramenta para reportar imbalance do FairFace 7-class.
- Conexão com [[range_loss_2016]] (Range Loss — long-tail FR).

## 8. Citar

- *"Yang, Jiang, Song & Guo (IJCV 2022, 36 páginas), em survey da
  Beijing University of Posts and Telecommunications (BUPT) sobre
  long-tailed visual recognition, organizam 10 categorias de
  abordagens de representation learning e propõem o coeficiente
  de Gini como métrica recomendada para quantificar
  long-tailedness em datasets visuais. A análise de 20 datasets
  large-scale dos últimos dez anos via Gini coefficient
  demonstra que o fenômeno long-tail é prevalente e ainda não
  plenamente endereçado — paradigma adjacente ao de demographic
  fairness em face recognition, com sobreposição parcial nas
  técnicas de mitigação."*

## 9-12.

PDF: `pdfs/survey_long_tail_2022.pdf`. BUPT. Conexões:
[[range_loss_2016]] (Range Loss — long-tail FR), [[sagawa_2020]]
(Group DRO — worst-case loss), [[dataset_bupt_2019]] (mesma
instituição), [[survey_mehrabi_2021]] (ML fairness broader).
