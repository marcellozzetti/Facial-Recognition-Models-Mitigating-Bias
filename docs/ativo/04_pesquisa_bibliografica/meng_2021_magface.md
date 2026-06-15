---
name: meng-2021-magface
status_verificacao: VERIFIED
autores: [Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou]
ano: 2021
titulo: "MagFace: A Universal Representation for Face Recognition and Quality Assessment"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021 — Oral)"
tipo_publicacao: conference
arxiv_id: "2103.06627"
doi: null
url_primario: https://arxiv.org/abs/2103.06627
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/meng_2021_magface.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução e formulação matemática lidos via pdftotext.
---

# MagFace — Universal Embedding with Quality (Meng et al., CVPR 2021 Oral)

> **Magnitude do embedding = qualidade da imagem**. Predecessor
> direto do AdaFace. Aprende within-class feature distribution
> bem-estruturada via mecanismo adaptativo.

## 1-2. Resumo + método

- **Loss MagFace**: aprende embeddings cuja **magnitude `l` mede
  qualidade** da imagem facial. Prova matemática: `l` cresce
  monotonicamente com a similaridade coseno ao centro da classe.
- **Mecanismo adaptativo**: puxa samples fáceis para próximo do
  centro da classe, empurra samples difíceis para longe da origem.
- **Efeito**: previne overfitting em low-quality samples noisy;
  estrutura melhor a distribuição within-class.
- **Aplicação tripla** em um único modelo:
  1. Face recognition
  2. Face quality assessment (FQA)
  3. Clustering

## 3-5. Datasets + métricas + resultados

- Treino e avaliação em IJB-B, IJB-C, LFW, CFP-FP, MegaFace.
- Compara contra ArcFace e variantes — supera SOTA em recognition
  e quality assessment simultaneamente.

## 7. Aplicação ao pipeline v3.2

- **Predecessor histórico do AdaFace** (Kim 2022).
- Quality-aware sem módulo dedicado — análogo conceitual ao FiLM
  que injeta sinal MST sem rede dedicada por estágio.
- **Não é baseline direto** da nossa tarefa (classification 7-class),
  mas estabelece a linha "quality-aware FR" para o Cap 2.

## 8. Citar

- *"MagFace (Meng et al., CVPR 2021 Oral) estabeleceu que a
  magnitude do embedding pode ser usada como proxy de qualidade
  da imagem, aprendendo simultaneamente representação para face
  recognition e quality assessment. Esta abordagem precede AdaFace
  (Kim et al., 2022) e consolida quality-awareness como dimensão
  necessária em face recognition robusto."*

## 9-12.

Arquivo PDF: `pdfs/meng_2021_magface.pdf`. Código:
github.com/IrvingMeng/MagFace. Conexões: [[deng_2019_arcface]]
(margin baseline), [[kim_2022_adaface]] (sucessora quality-adaptive),
[[schroff_2015_facenet]] (Triplet ancestral).
