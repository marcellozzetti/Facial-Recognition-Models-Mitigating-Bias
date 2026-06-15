---
name: fairface-challenge-eccv2020
status_verificacao: VERIFIED
autores: [Tomas Sixta, Julio Silveira, Sergio Escalera, Neil Robertson, Eduardo Vazquez, et al.]
ano: 2020
titulo: "FairFace Challenge at ECCV 2020: Analyzing Bias in Face Recognition"
venue: "European Conference on Computer Vision Workshops (ECCV 2020 — ChaLearn Looking at People)"
tipo_publicacao: workshop_conference
arxiv_id: "2009.07838"
doi: null
url_primario: https://arxiv.org/abs/2009.07838
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado de arXiv (pdfs/fairface_challenge_eccv2020.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

# FairFace Challenge ECCV 2020 — ChaLearn Looking at People

> **Challenge oficial ECCV 2020 (ChaLearn)**: 1:1 face verification em
> gender + skin color sob confounding attributes. Dataset baseado em
> **IJB-C reanotado + 12.5K novas imagens** com labels adicionais.

## 1-2. Resumo

- Workshop **ChaLearn Looking at People** no ECCV 2020.
- Avaliar **acurácia e viés** em gender + skin color em 1:1 face
  verification.
- Dataset baseado em **IJB-C reanotado + 12.5K novas imagens**.
- **Dataset não-balanceado** intencionalmente para simular cenário
  real onde modelos fair devem ser treinados/avaliados em dados
  desbalanceados.

## 3-5. Métricas e resultados

- **151 participantes**, mais de 1.800 submissões totais.
- **36 times ativos** na fase final.
- **Top 10 times excederam 0.999 AUC-ROC** com baixos scores de bias.
- Estratégias comuns: pré-processamento de faces, homogeneização de
  distribuições, loss functions bias-aware, ensembles.
- Análise top-10: **maior false positive rate (e menor false negative)
  para mulheres de pele escura**. Eyeglasses e idade jovem também
  aumentam FPR.

## 7. Aplicação ao pipeline v3.2

- **Cap 2** (Revisão): challenge histórico, contexto da literatura
  fairness FR pós-2020.
- **Não é dataset que usamos diretamente** — focamos em FairFace
  classification (Kärkkäinen & Joo 2021), não em verification 1:1.
- Demonstra empiricamente que **bias persiste mesmo nos melhores
  modelos** (top-10 com 0.999 AUC ainda mostram FPR maior para
  mulheres de pele escura).

## 8. Citar

- *"O FairFace Challenge no ECCV 2020 Workshops (ChaLearn,
  arXiv:2009.07838) liberou benchmark anotado large-scale baseado em
  IJB-C reanotado e expandido com 12.5K imagens adicionais. A
  análise dos 10 times com maior AUC (>0.999) demonstrou
  empiricamente que viés persiste mesmo em modelos de alta acurácia,
  com mulheres de pele escura apresentando false positive rate
  sistematicamente superior."*

## 9-12.

PDF: `pdfs/fairface_challenge_eccv2020.pdf`. Conexões:
[[dataset_karkkainen_2021]] (FairFace classification dataset —
mesma família de nome mas dataset diferente),
[[buolamwini_2018]] (Gender Shades — referência histórica do
challenge).
