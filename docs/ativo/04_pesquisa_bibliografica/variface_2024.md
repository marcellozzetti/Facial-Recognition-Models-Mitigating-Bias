---
name: variface-2024
status_verificacao: VERIFIED
autores: [Michael Yeung (Sony Group), Toya Teramoto (Sony Group), Songtao Wu (Sony R&D Center China), Tatsuo Fujiwara (Sony Group), Kenji Suzuki (Sony Group), Tamaki Kojima (Sony Group)]
ano: 2024
titulo: "VariFace: Fair and Diverse Synthetic Dataset Generation for Face Recognition"
venue: "arXiv preprint 2412.06235v2 (atualizado abr 2025) — Sony Group Corporation"
tipo_publicacao: preprint
arxiv_id: "2412.06235"
doi: null
url_primario: https://arxiv.org/abs/2412.06235
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/variface_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract lido via pdftotext.
---

# VariFace — Fair Synthetic Dataset Generation (Yeung, Teramoto, Wu, Fujiwara, Suzuki & Kojima — Sony Group 2024)

> **Geração sintética de datasets de FR fair e diverso** pela Sony
> Group Corporation. Endereça privacy + bias concerns do uso de
> web-scraped datasets reais.

## 1-2. Resumo + método

- **Problema**: web-scraped datasets de FR têm **privacy + bias
  concerns** significativos.
- **Synthetic methods** mitigam essas concerns mas frequentemente
  trazem outros problemas (drift vs real, diversidade reduzida).
- **VariFace**: framework para gerar synthetic datasets **fair e
  diverse** especificamente para treinar FR.

## 7. Aplicação ao pipeline v3.2

- **Track L** (auxiliar — synthetic data) — paralelo a
  [[synthetic_face_2024]], [[massively_annotated_2024]],
  [[frcsyn_2024]], [[fairimagen_neurips2025]],
  [[fairer_datasets_2024]].
- **Não usamos synthetic data** — FairFace é real.
- **Cap 2** (Revisão): citação contextual sobre direções privacy-
  preserving + fair em data generation.
- **Procedência industrial** (Sony Group) — relevância prática
  comercial.

## 8. Citar

- *"VariFace (Yeung, Teramoto, Wu, Fujiwara, Suzuki & Kojima — Sony
  Group Corporation, arXiv 2412.06235, 2024) propõe framework de
  geração sintética de datasets de face recognition simultaneamente
  fair e diverse, endereçando preocupações de privacy e bias
  inerentes ao uso de datasets web-scraped reais. A procedência
  industrial — Sony Group — reforça a relevância prática
  de pipelines synthetic-first em deploy comercial de FR."*

## 9-12.

PDF: `pdfs/variface_2024.pdf`. Conexões: [[synthetic_face_2024]],
[[massively_annotated_2024]] (mesmo grupo análise sintético),
[[frcsyn_2024]], [[fairimagen_neurips2025]],
[[fairer_datasets_2024]] (família Track L synthetic).
