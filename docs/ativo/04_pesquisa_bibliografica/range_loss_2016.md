---
name: range-loss-2016
status_verificacao: VERIFIED
autores: [Xiao Zhang (Tianjin University), Zhiyuan Fang (Southern Univ of Science and Technology), Yandong Wen (Carnegie Mellon), Zhifeng Li (SIAT CAS), Yu Qiao (SIAT CAS)]
ano: 2016
titulo: "Range Loss for Deep Face Recognition with Long-tail"
venue: "arXiv preprint 1611.08976v1 (nov 2016) — versão estendida da ICCV 2017"
tipo_publicacao: preprint
arxiv_id: "1611.08976"
doi: null
url_primario: https://arxiv.org/abs/1611.08976
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/range_loss_2016.pdf). Validação Nível 2 (Camada 2 - Track K) em 2026-06-15 — abstract e motivação long-tail lidos via pdftotext.
---

# Range Loss — Long-tail FR (Zhang, Fang, Wen, Li & Qiao — TJU + SUSTC + CMU + SIAT, 2016)

> **Loss específico para FR sob long-tail distribution**. Predecessor
> conceitual dos margin-based losses modernos (FaceNet, CosFace,
> ArcFace). Track K — FR fundadores. Mesmo time SIAT que CosFace
> (Wang 2018).

## 1-2. Resumo + método

- **Problema**: dados de FR reais têm **long-tail distribution** —
  poucas identidades dominam, maioria das identidades tem poucas
  imagens.
- **Range Loss**: optimiza simultaneamente:
  1. **Reduz intra-class variance** (samples da mesma identidade
     próximos)
  2. **Aumenta inter-class distance** (identidades diferentes
     separadas)
- Específico para long-tail — não assume distribuição uniforme.

## 3-5. Datasets e resultados

- Datasets long-tail benchmarks.
- Compara com baselines de DeepID2+ e DeepFace.

## 7. Aplicação ao pipeline v3.2

- **Track K** (FR fundadores) — completa lista com FaceNet, CosFace,
  ArcFace, MagFace, AdaFace.
- **Cap 2** (Revisão): citação contextual sobre evolução de losses
  margin-based em FR — Range Loss antecipa lógica de margin.
- **Não usamos Range Loss** — nossa tarefa é classification 7-class,
  não verification long-tail.

## 8. Citar

- *"Zhang, Fang, Wen, Li & Qiao (arXiv 1611.08976, 2016 / ICCV
  2017) — colaboração Tianjin University + Southern University of
  Science and Technology + Carnegie Mellon + Shenzhen Institutes
  of Advanced Technology — propõem Range Loss como abordagem
  específica para face recognition sob long-tail distribution,
  otimizando simultaneamente intra-class variance reduction e
  inter-class distance maximization, paradigma que antecipa
  conceitualmente os margin-based losses modernos (CosFace, ArcFace,
  MagFace, AdaFace)."*

## 9-12.

PDF: `pdfs/range_loss_2016.pdf`. Conexões: [[schroff_2015_facenet]]
(FaceNet — Triplet loss anterior), [[wang_2018_cosface]] (mesmo grupo
SIAT — Zhifeng Li, Yu Qiao co-autores), [[deng_2019_arcface]],
[[meng_2021_magface]], [[kim_2022_adaface]] (sucessores margin-based),
[[survey_long_tail_2022]] (survey BUPT — Gini coefficient).
