---
name: kim-2022-adaface
status_verificacao: OVERVIEW_ONLY
autores: [Minchul Kim, Anil K. Jain, Xiaoming Liu]
ano: 2022
titulo: "AdaFace: Quality Adaptive Margin for Face Recognition"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022 — Oral)"
tipo_publicacao: conference
arxiv_id: "2204.00964"
doi: null
url_primario: https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# AdaFace — Quality Adaptive Margin (Kim et al., CVPR 2022 Oral)

> **Quality-adaptive margin** baseado em feature norm como proxy de
> image quality. Loss SOTA em FR pós-ArcFace; **considera qualidade
> de imagem**, dimensão crítica para fairness (cf. Pangelinan 2023).

## 1-2. Resumo + método

- **Hipótese central**: margin deve ser adaptado por qualidade da
  imagem.
- **Feature norm** usado como proxy de quality (sem módulo
  computacionalmente caro de IQA).
- **Margin adaptativo**: enfatiza hard yet recognizable samples,
  de-enfatiza low-quality unidentifiable.

## 7. Aplicação ao pipeline v3.2

- **Diretamente relevante para H6** (Pangelinan 2023): pixel info
  domina disparity em FR. AdaFace **incorpora quality explicitamente**.
- **Cap 3 baseline**: AdaFace vs ArcFace, controlando para quality.
- **Insight para FiLM-conditioning**: poderíamos condicionar FiLM
  por quality também, não só por skin tone.

## 8. Citar

- *"AdaFace (Kim, Jain & Liu, CVPR 2022 Oral) introduz quality-
  adaptive margin baseado em feature norm como proxy de image
  quality, demonstrando que considerar qualidade de imagem é
  essencial para face recognition robusto. Esta observação alinha-
  se com o achado de Pangelinan et al. (2023) sobre pixel
  information como confounder primário em disparidade racial,
  motivando que o Capítulo 3 desta dissertação reporte resultados
  com controle explícito de qualidade da imagem."*

## 9-12.

PDF pendente. Conexões: [[schroff_2015_facenet]], [[wang_2018_cosface]],
[[deng_2019_arcface]], [[pangelinan_2023]].
