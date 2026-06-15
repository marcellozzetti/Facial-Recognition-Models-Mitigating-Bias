---
name: fairimagen-neurips2025
status_verificacao: VERIFIED
autores: [Zihao Fu (Chinese University of Hong Kong), Ryan Brown (Oxford), Shun Shao (Cambridge), Kai Rawal (Oxford), Eoin Delaney (Trinity College Dublin), Chris Russell (Oxford OII)]
ano: 2025
titulo: "FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models"
venue: "NeurIPS 2025 (Neural Information Processing Systems)"
tipo_publicacao: conference
arxiv_id: null
doi: null
url_primario: https://blog.neurips.cc/category/2025-conference/
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN/OpenReview (pdfs/fairimagen_neurips2025.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract lido via pdftotext. Título correto identificado: 'Post-Processing for Bias Mitigation in Text-to-Image Models' (não 'Image Generation' como rascunho anterior).
---

# FairImagen — Post-Processing for Bias Mitigation in T2I Diffusion (Fu et al. — NeurIPS 2025)

> **Framework de debiasing post-hoc** sobre prompt embeddings em
> modelos text-to-image diffusion (e.g., Stable Diffusion). Opera
> nos **embeddings CLIP** via **Fair PCA** projetando em subespaço
> que minimiza informação demográfica preservando conteúdo semântico.
> **Sem retreinamento** do modelo de diffusion subjacente.

## 1-2. Resumo + método

- **Problema**: text-to-image diffusion models (Stable Diffusion, etc.)
  replicam e amplificam vieses sociais por gender e race.
- **Contribuição**: FairImagen — framework post-hoc que opera **em
  prompt embeddings** para mitigar viés sem retreinamento.
- **Núcleo técnico**:
  1. **Fair Principal Component Analysis** projeta embeddings CLIP
     em subespaço que **minimiza informação demográfica** preservando
     conteúdo semântico.
  2. **Empirical noise injection** para reforçar debiasing.
  3. **Unified cross-demographic projection** — debiasing simultâneo
     em múltiplos atributos demográficos.

## 3-5. Datasets e avaliação

- Experimentos em **gender, race, e settings intersectional**.
- **Achado**: FairImagen melhora significativamente fairness com
  **trade-off moderado** em qualidade de imagem e prompt fidelity.

## 7. Aplicação ao pipeline v3.2

- **Não usamos T2I diffusion** nesta dissertação — diferente domínio.
- **Track L (auxiliar)** — direção generativa fairness, paralelo a
  [[synthetic_face_2024]], [[variface_2024]], [[fairer_datasets_2024]].
- **Conexão técnica relevante**: Fair PCA em embeddings CLIP é
  conceitualmente próxima a **bias subspace projection**
  ([[bias_subspace_2025]]) — mesma família de técnica.
- Citação contextual no Cap 2 sobre **direções de debiasing
  pós-hoc** em VLM contexts.

## 8. Citar

- *"FairImagen (Fu, Brown, Shao, Rawal, Delaney & Russell, NeurIPS
  2025) introduz framework post-hoc de debiasing para modelos
  text-to-image baseados em diffusion (e.g., Stable Diffusion),
  operando sobre prompt embeddings CLIP via Fair Principal Component
  Analysis para projetar em subespaço que minimiza informação
  demográfica preservando semântica. A abordagem alcança debiasing
  cross gender + race + intersectional com trade-off moderado em
  qualidade de imagem, sem retreinar o modelo diffusion
  subjacente — direção representativa de debiasing pós-hoc em
  vision-language pipelines."*

## 9-12.

PDF: `pdfs/fairimagen_neurips2025.pdf`. Autoria distribuída: CUHK +
Oxford (3 autores) + Cambridge + Trinity College Dublin. Conexões:
[[bias_subspace_2025]] (técnica análoga — projeção de subespaço),
[[luo_2024_fairclip]] (FairCLIP — debiasing CLIP embeddings),
[[bendvlm_2024]] (test-time debiasing), [[synthetic_face_2024]],
[[variface_2024]], [[fairer_datasets_2024]] (família synthetic data).
