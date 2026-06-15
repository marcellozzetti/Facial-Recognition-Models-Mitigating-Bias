---
name: deng-2019-arcface
status_verificacao: VERIFIED
autores: [Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou]
ano: 2019
titulo: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019) — versão estendida em IEEE TPAMI (arXiv v4, 2022)"
tipo_publicacao: conference
arxiv_id: "1801.07698"
doi: null
url_primario: https://arxiv.org/abs/1801.07698
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar (versão v4 estendida com sub-center e face inversion)
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/deng_2019_arcface.pdf, v4 2022). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução, método e tabelas principais lidos via pdftotext.
---

# ArcFace — Additive Angular Margin (Deng et al., CVPR 2019 / TPAMI 2022)

> **Loss canônico de face recognition desde 2019.** Additive angular
> margin com correspondência geométrica direta à distância geodésica
> na hipersfera. Versão v4 (2022) estende com **sub-center ArcFace**
> (robustez a noisy labels) e **face inversion** (geração condicionada
> via BN priors).

## 1-2. Resumo + método

- **ArcFace standard**: `cos(θ + m)` — margin aditivo no ângulo
  geodésico. Embedding 512-d normalizado, cosine similarity.
- **Diferencial vs CosFace** (Wang 2018): ArcFace usa margin angular
  (`cos(θ + m)`); CosFace usa cosine margin (`cos(θ) - m`).
- **Vantagem**: correspondência exata entre angular margin e
  geodesic distance na hipersfera unitária.
- **Sub-center ArcFace** (v4): cada classe contém K sub-centers; o
  sample só precisa estar próximo de algum sub-center positivo —
  reduz o efeito de noisy labels em datasets web (MS1MV0, Celeb500K).
- **Face inversion**: pretrained ArcFace pode gerar imagens face
  identity-preserved usando gradient + BN priors, sem treinar
  generator/discriminator.

## 7. Aplicação ao pipeline v3.2

- **Loss canônico** para FR baselines reportados na revisão (Cap 2).
- **Modelos auditados** em Pangelinan 2023, Kolla 2022, Dooley 2022,
  Wang 2019 RFW são todos ArcFace ou variantes.
- **Para Cap 4 (Metodologia)**: ArcFace é mencionado como **loss
  canônico de FR moderno**, contexto para apresentar nossa escolha
  (cross-entropy em classification 7-class, não verification).
- **Não usamos ArcFace diretamente** — nossa tarefa é classification
  7-class, não verification (ArcFace é loss para embedding learning,
  não classification multi-classe).

## 8. Citar

- *"ArcFace (Deng et al., CVPR 2019) é o estado-da-arte canônico
  para face recognition desde 2019, introduzindo additive angular
  margin com correspondência geométrica direta à distância
  geodésica na hipersfera. Praticamente todos os estudos de
  fairness em FR pós-2019 (Pangelinan et al. 2023, Kolla &
  Savadamuthu 2022, Dooley et al. 2022) auditam ArcFace como
  modelo de referência."*

## 9-12.

Arquivo PDF: `pdfs/deng_2019_arcface.pdf` (11.2 MB, versão v4 2022).
Conexões: [[schroff_2015_facenet]] (Triplet loss, paradigma anterior),
[[wang_2018_cosface]] (cosine margin alternativa),
[[meng_2021_magface]] (extensão com magnitude = quality),
[[kim_2022_adaface]] (quality-adaptive margin),
[[pangelinan_2023]] (audita ArcFace+VGGFace2),
[[kolla_2022]] (audita ArcFace).
