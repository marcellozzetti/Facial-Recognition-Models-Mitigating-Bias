---
name: dataset-bupt-2019
status_verificacao: VERIFIED
autores: [Mei Wang, Yaobin Zhang, Weihong Deng (Beijing University of Posts and Telecommunications - BUPT)]
ano: 2022
titulo: "Meta Balanced Network for Fair Face Recognition (introduz BUPT-Globalface, BUPT-Balancedface e Identity Shades datasets + MBN algorithm)"
venue: "arXiv 2205.06548 (maio 2022) — versão expandida estende trabalho anterior de Wang & Deng 2019 (RFW/CVPR)"
tipo_publicacao: preprint
arxiv_id: "2205.06548"
doi: null
url_primario: http://www.whdeng.cn/RFW/index.html
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN (pdfs/dataset_bupt_2019.pdf — nome do arquivo herdado da nomenclatura original do projeto, mas o paper canônico que descreve BUPT-Balancedface + BUPT-Globalface + IDS + MBN é de 2022). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract e introdução lidos via pdftotext.
---

> ℹ️ **Nota sobre nomenclatura**: o slug `dataset_bupt_2019` foi
> herdado de uma referência anterior ao **RFW dataset (Wang & Deng,
> CVPR 2019)** — esse outro trabalho cobre o RFW. O paper aqui
> documentado é a **continuação por Wang, Zhang & Deng (2022)** que
> introduz: BUPT-Globalface, BUPT-Balancedface, IDS e o algoritmo MBN.

# Meta Balanced Network + BUPT datasets (Wang, Zhang & Deng — BUPT, arXiv 2022)

> **Trabalho consolidado em 2022** que introduz três datasets e um
> algoritmo, tudo voltado a fairness por skin tone (não por race
> direto). Disponível em http://www.whdeng.cn/RFW/index.html.

## 1-2. Resumo + contribuições

Paper apresenta **4 contribuições integradas**:

1. **IDS (Identity Shades)** — dataset de **teste** balanceado por
   skin tone via Fitzpatrick Skin Type + Individual Typology Angle
   (ITA). Permite avaliar fairness por tom de pele (não raça).
2. **BUPT-Globalface** — dataset de **treino** representando a
   distribuição global de skin tones.
3. **BUPT-Balancedface** — dataset de **treino** balanceado por
   skin tone (1.3M imagens, 28k indivíduos).
4. **MBN (Meta Balanced Network)** — algoritmo de **meta-learning
   com adaptive margins** em large margin loss; otimiza meta
   skewness loss em meta set unbiased; usa **backward-on-backward
   automatic differentiation** para segundo gradient descent step
   nas margens.

## 3-5. Achados experimentais

- **Bias documentado em IDS**: error rates em faces **dark-skinned
  são ~2× das light-skinned** (commercial APIs + SOTA).
- **Bias vem de data E algorithm aspects**.
- **Argumento metodológico chave**: race labels são instáveis →
  paper opta por **skin tone via FST + ITA** como label mais
  preciso e científico.
- **MBN supera baselines** em IDS, aprendendo performance balanceada
  cross-skin-tone.

## 7. Aplicação ao pipeline v3.2

- **Dataset alternativo** para Cap 2 (Revisão) — citamos
  BUPT-Balancedface como exemplo de treino balanceado por skin
  tone.
- **Argumento alinhado com nossa tese**: skin tone como sinal mais
  preciso que race — converge com nossa decisão de usar **MST**
  (Schumann 2023, evolução do FST).
- **Não usamos os datasets BUPT** — operamos em FairFace
  (classification 7-class), não BUPT (verification).
- **MBN como baseline conceitual** para Cap 2 — meta-learning com
  margins é direção paralela ao FiLM-conditioning.

## 8. Citar

- *"Wang, Zhang & Deng (BUPT, arXiv 2022, 2205.06548) consolidam
  esforço de fairness em face recognition introduzindo três datasets
  (Identity Shades como conjunto de teste balanceado por skin tone,
  BUPT-Globalface e BUPT-Balancedface como conjuntos de treino) e
  o algoritmo Meta Balanced Network que aprende adaptive margins
  via meta-learning. Documentam que faces de pele escura têm
  taxa de erro aproximadamente duas vezes maior que faces de pele
  clara em modelos comerciais e SOTA, evidência que motivou a
  escolha desta dissertação por skin tone (Monk Skin Tone, Schumann
  2023) como sinal de condicionamento auxiliar à classificação
  racial."*

- *"O BUPT-Balancedface (Wang, Zhang & Deng, 2022) provê 1.3 milhões
  de imagens estritamente balanceadas por skin tone em 28 mil
  indivíduos — referência canônica para datasets de treino
  fairness-aware em face recognition."*

## 9-12.

PDF: `pdfs/dataset_bupt_2019.pdf` (paper canônico de 2022 que
introduz BUPT-Balancedface). Conexões: [[dataset_wang_2019]] RFW
(trabalho anterior dos mesmos autores), [[schumann_2023]] (MST —
sucessor moderno do FST usado pelo paper), [[kolla_2022]]
(balanceamento não basta), [[deng_2019_arcface]] (mesmo Weihong Deng
co-autor — ArcFace é base do MBN), [[survey_long_tail_2022]] (mesma
instituição BUPT — Gini coefficient como métrica adjacente).
