---
name: deng-2019-arcface
status_verificacao: OVERVIEW_ONLY
autores: [Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou]
ano: 2019
titulo: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)"
tipo_publicacao: conference
arxiv_id: "1801.07698"
doi: null
url_primario: https://arxiv.org/abs/1801.07698
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: Abstract + busca web.
---

> ⚠️ **OVERVIEW_ONLY** — PDF pendente.

# ArcFace — Additive Angular Margin (Deng et al., CVPR 2019)

> **SOTA de face recognition** desde 2019. Additive angular margin
> com correspondência geométrica direta à geodesic distance na
> hipersfera. Loss canônico para FR; usado em virtualmente todos
> os papers subsequentes.

## 1-2. Resumo + método

- **ArcFace**: cos(θ + m) — margin aditivo no ângulo geodésico.
- **Diferencial vs CosFace**: ArcFace usa margin angular, CosFace
  cosine margin.
- **Vantagem**: correspondência exata entre angular margin e
  geodesic distance na hipersfera.
- **SOTA**: MegaFace Challenge (1M faces).

## 7. Aplicação ao pipeline v3.2

- **Loss canônico** para FR baselines no Cap 3.
- **Modelos auditados** em Pangelinan 2023, Kolla 2022, Dooley
  2022, Wang 2019 RFW usam ArcFace.
- **Para Cap 3 (RFW/BFW)**: ArcFace é a escolha padrão.

## 8. Citar

- *"ArcFace (Deng et al., CVPR 2019) é o estado-da-arte canônico
  para face recognition desde 2019, introduzindo additive angular
  margin com correspondência geométrica direta à distância
  geodésica na hipersfera. Praticamente todos os estudos de
  fairness em FR pós-2019 (incluindo Pangelinan et al. 2023, Kolla
  & Savadamuthu 2022, Dooley et al. 2022) auditam ArcFace como
  modelo de referência."*

## 9-12.

PDF pendente. Conexões: [[schroff_2015_facenet]], [[wang_2018_cosface]],
[[pangelinan_2023]], [[kolla_2022]], [[dataset_wang_2019]] RFW.
