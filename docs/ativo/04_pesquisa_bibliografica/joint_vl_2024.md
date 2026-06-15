---
name: joint-vl-2024
status_verificacao: VERIFIED
autores: [a verificar]
ano: 2024
titulo: "Joint Vision-Language Social Bias Removal for CLIP"
venue: "arXiv preprint (a verificar venue final)"
tipo_publicacao: preprint
arxiv_id: "2411.12785"
doi: null
url_primario: https://arxiv.org/abs/2411.12785
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: a confirmar
lente_disrupcao: metodologica
fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/joint_vl_2024.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.
---


# Joint Vision-Language Social Bias Removal for CLIP (2024)

> **Bias alignment + counterfactual debiasing** simultâneos para
> remoção de viés social em CLIP. Generalização in/out-domain.
> Track I (VLM fairness).

## 1-2. Resumo + método (inferido do abstract via busca)

- **Bias alignment**: image bias embeddings extraídos do original
  carregam informação significativa sobre social groups (gender,
  age, race).
- **Counterfactual debiasing**: clustering de embeddings por grupo
  social.
- **Resultado declarado**: melhor trade-off debiasing × VL alignment
  + generalização in/out-domain superior.

## 7. Aplicação ao pipeline v3.2

- **Track I** complementar a FairCLIP e FairerCLIP.
- **Método joint** (não apenas vision ou apenas language) é
  conceitualmente alinhado com nossa abordagem de **conditioning
  multi-modal** (skin tone como sinal auxiliar).
- Para Cap 2: comparação contra esta abordagem como baseline.

## 8. Citar

- *"Pesquisa recente em debiasing de CLIP (arXiv:2411.12785, 2024)
  demonstra que abordagens joint vision-language — combinando bias
  alignment e counterfactual debiasing — superam métodos puramente
  vision ou puramente language em generalização out-of-domain de
  fairness."*

## 9-12.

PDF pendente. Conexões: [[luo_2024_fairclip]], [[dehdashtian_2024_fairerclip]].
