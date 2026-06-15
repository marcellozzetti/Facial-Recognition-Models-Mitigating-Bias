---
name: image-distortions-2021
status_verificacao: VERIFIED
autores: [Puspita Majumdar (IIIT-Delhi + IIT Jodhpur), Surbhi Mittal (IIT Jodhpur), Richa Singh (IIT Jodhpur), Mayank Vatsa (IIT Jodhpur)]
ano: 2021
titulo: "Unravelling the Effect of Image Distortions for Biased Prediction of Pre-trained Face Recognition Models"
venue: "arXiv preprint 2108.06581v1 (ago 2021)"
tipo_publicacao: preprint
arxiv_id: "2108.06581"
doi: null
url_primario: https://arxiv.org/abs/2108.06581
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/image_distortions_2021.pdf). Validação Nível 2 (Camada 2 - conflito moderado) em 2026-06-15 — abstract, motivação (Fig 1 Gaussian blur cross gender+race) e setup experimental lidos via pdftotext.
---

# Effect of Image Distortions on FR Bias (Majumdar, Mittal, Singh & Vatsa — IIT Jodhpur 2021)

> **Primeira análise sistemática** do efeito de **image distortions**
> (blur, noise, JPEG) em **biased prediction** de pre-trained FR
> models cross gender + race. IIT Jodhpur. Argumento central:
> mesmo modelos "unbiased" no original podem se tornar **biased
> sob distortions**.

## 1-2. Resumo + método

- **Pergunta central**: "can seemingly unbiased pre-trained model
  become biased when input data undergoes certain distortions?"
- **Setup**: avaliação de **4 SOTA face recognition models** sob
  distortions (Gaussian blur, noise, JPEG compression) cross
  diferentes subgrupos de gender e race.
- **Métrica**: Verification Accuracy @ 0.01 FAR + cosine similarity
  com imagem clean.

## 3-5. Achados

- **Image distortions têm relação com performance gap** cross
  subgroups demográficos.
- **Gaussian blur** com diferentes σ mostra impactos
  não-uniformes em gender × race subgroups (Fig 1 do paper).
- **Implicação**: avaliar fairness apenas em condições "clean"
  é insuficiente — distortions são confounders.

## 7. Aplicação ao pipeline v3.2

- **Conflito moderado** (do pente fino): reforça que **quality
  control é necessário** no protocolo experimental.
- **Suporte direto à Hipótese H6** ([[pangelinan_2023]]): pixel
  info / qualidade como confounder primário.
- **Cap 3** (Metodologia): nosso protocolo deve declarar como
  controlamos para image quality (face crop + normalização).

## 8. Citar

- *"Majumdar, Mittal, Singh & Vatsa (arXiv 2108.06581, 2021),
  pesquisadoras da IIT Jodhpur, conduzem primeira análise
  sistemática do efeito de image distortions (Gaussian blur, noise,
  JPEG) sobre biased prediction de modelos pre-treinados de face
  recognition, demonstrando que mesmo modelos aparentemente
  unbiased em condição clean podem exibir performance gap
  significativo cross subgroups demográficos sob distortions —
  evidência adicional que sustenta a hipótese de pixel-information
  como confounder de Pangelinan et al. (2023) e a Hipótese H6 da
  presente dissertação."*

## 9-12.

PDF: `pdfs/image_distortions_2021.pdf`. Conexões:
[[pangelinan_2023]] (refutação central — pixel info > skin tone),
[[occlusion_bias_2024]] (confounders paralelos), [[grother_2019]]
(NIST FRVT — avaliações em scale).
