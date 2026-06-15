---
name: debiasing-neural-interventions-2025
status_verificacao: OVERVIEW_ONLY
autores: [a verificar - PDF Springer atrás de paywall institucional]
ano: 2025
titulo: "Debiasing CLIP with Neural Interventions"
venue: "Springer Nature Lecture Notes in Computer Science / Computer Vision 2025 — DOI 10.1007/978-3-032-21324-2_32 — capítulo de livro"
tipo_publicacao: book_chapter
arxiv_id: null
doi: "10.1007/978-3-032-21324-2_32"
url_primario: https://link.springer.com/chapter/10.1007/978-3-032-21324-2_32
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: Busca web + abstract da página Springer. PDF integral aguarda VPN institucional Unifesp/ICT. Status mantido em OVERVIEW_ONLY até leitura do PDF.
---

> ℹ️ **Status OVERVIEW_ONLY mantido**: PDF está em Springer chapter
> com paywall institucional. Conteúdo baseado em abstract público da
> página Springer + busca web. **Pendência registrada para baixar via
> VPN Unifesp/ICT.**

# Debiasing CLIP with Neural Interventions (Springer Lecture Notes 2025)

> **Targeted interventions** em attention heads do CLIP identificadas
> como "expertas" em demographic information. **Sem retraining** —
> intervention em inference time. Track I — método eficiente.

## 1-2. Resumo + método (a partir do abstract público)

- Identifica **attention heads específicas** que encodam demographic
  information via análise sistemática de representações internas do
  CLIP.
- **Intervention at inference time**:
  - Substitui ativações dessas heads por **demographic prototypes**.
  - Ou **neutraliza** essas heads completamente.
- **Sem retraining** — lightweight, plug-and-play.
- Bias reduction comparável a SOTA em benchmarks SISPI, So-B-IT.

## 7. Aplicação ao pipeline v3.2

- **Demonstra que viés tem localização específica** nas representações
  internas do CLIP (attention heads identificáveis).
- **Paradigma conceitual relacionado**: intervenção em representações
  intermediárias (como o FiLM-conditioning aplicado nesta dissertação),
  mas com **mecanismo diferente** (head-level vs feature-wise affine).
- **Track I** (VLM fairness) — baseline lightweight para Cap 2.
- **Não usamos esta abordagem** — operamos em ConvNeXt-T com FiLM,
  não em CLIP attention heads.

## 8. Citar

- *"Trabalhos de debiasing via neural interventions em CLIP
  publicados em Springer Lecture Notes 2025 (DOI 10.1007/
  978-3-032-21324-2_32) demonstram que viés demográfico tem
  localização identificável em attention heads específicas do
  encoder, validando o paradigma de intervenção em representações
  intermediárias da rede — paradigma diferente mas conceitualmente
  paralelo ao FiLM-conditioning empregado nesta dissertação para
  modular features do ConvNeXt-T por sinal de tom de pele."*

## 9-12.

PDF: pendente — aguarda VPN institucional. Conexões: [[bendvlm_2024]]
(test-time debiasing CLIP — paralelo), [[luo_2024_fairclip]]
(FairCLIP — Sinkhorn-based), [[dehdashtian_2024_fairerclip]]
(FairerCLIP — RKHS), [[fairimagen_neurips2025]] (Fair PCA em CLIP
embeddings — relacionado conceitualmente).
