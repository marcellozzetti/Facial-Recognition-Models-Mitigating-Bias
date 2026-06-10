---
name: pangelinan-2023
status_verificacao: OVERVIEW_ONLY
autores: [Gabriella Pangelinan, K.S. Krishnapriya, Vitor Albiero, Grace Bezold, Kai Zhang, Kushal Vangara, Michael C. King, Kevin W. Bowyer]
ano: 2023
titulo: "Exploring Causes of Demographic Variations In Face Recognition Accuracy"
venue: "arXiv preprint (submetido abril 2023)"
tipo_publicacao: preprint
arxiv_id: "2304.07175"
doi: null
url_primario: https://arxiv.org/abs/2304.07175
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-10
n_paginas: a confirmar (PDF não lido)
lente_disrupcao: paradigma
fonte_leitura: Apenas abstract via arXiv (PDF e HTML integrais não disponíveis). Contexto adicional via [[../_validacao_cientifica_pipeline]] R6-1.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**
>
> Esta ficha foi construída **apenas a partir do abstract público
> no arXiv**. PDF integral pendente de leitura. Promover a VERIFIED
> requer: (1) download do PDF para `pdfs/pangelinan_2023.pdf`, (2)
> leitura integral, (3) reescrita das seções 3-7 com base no PDF.

# Causas de Variação Demográfica em FR Accuracy (Pangelinan et al., 2023)

> **Refutação potencial de H5 da nossa tese.** Argumenta que para FR
> (verificação 1:1), **pixel info da face** é o fator dominante de
> disparity, NÃO skin tone direto. Motivou reformulação de H5 na
> reunião de 2026-06-08.

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

Reports midiáticos sobre viés racial em FR proliferaram sem isolar a
causa mecanística. O paper revisa resultados experimentais explorando
**4 hipóteses causais** para a assimetria cross-demografia: skin tone,
face size/shape, imbalance no treino, e quantidade de pixels de face
visível no teste.

## 2. Método

> *Fonte: abstract verbatim.*

- Tarefa: **face matching 1-to-1** (NÃO classification).
- Análise: distribuições mated vs non-mated por grupo demográfico.
- Variáveis investigadas:
  - Skin tone (diferenças)
  - Face size/shape
  - Imbalance em identidades e imagens de treino
  - "Face pixels" (área útil de pixels)

> **[PENDENTE PDF]** Datasets específicos (RFW? MORPH?). FR systems
> testados (ArcFace? AdaFace?). Protocolo estatístico exato.

## 3-6. Datasets, métricas, resultados, limitações

> **[PENDENTE PDF]** Apenas a CONCLUSÃO está no abstract:
>
> *"demographic differences in face PIXEL INFORMATION of the test
> images appear to most directly impact the resultant differences
> in face recognition accuracy."*

## 7. Limitações que identifiquei (a partir do abstract apenas)

- Trabalha em **FR (verification 1:1)**, NÃO em classification.
  Conclusão "pixel info > skin tone" pode não generalizar para race
  classification.
- Análise correlacional, não causal estrita.
- Sem ablation intervencional para mitigar pixel info.

## 8. Relação com nossa pesquisa

### 8.1 Risco crítico para H5

H5 atual: "fairness transfere para FR; Black/African melhora ≥+3pp".

Pangelinan sugere que em FR, **pixel info é confounder dominante**.
Se isso for verdade, mitigação via skin-tone conditioning pode ter
efeito menor do que esperado em FR.

### 8.2 Reformulação de H5 (decidida na reunião de 2026-06-08)

Versão preferida (V3): separar em **H5 + H6**.

- **H5**: condicionamento por MST melhora fairness em FR.
- **H6 nova**: disparity residual em Black/African após
  condicionamento é predominantemente explicada por diferenças de
  pixel info (Pangelinan 2023).

Transforma a ameaça em **contribuição quantitativa** (decomposição
de variância).

### 8.3 Mitigação no plano experimental

- Cap 3 deve adicionar **quality control de imagem** (face crop +
  alinhamento + normalização de pixel info) ANTES de medir o efeito
  de MST-conditioning isoladamente.

## 9. Pontos para citar

- *"Pangelinan et al. (2023) argumentam, com base em análise
  empírica de modelos de face recognition estado-da-arte, que
  diferenças de face pixel information (área útil de pixels após
  detecção) explicam a maior parte da variação cross-demográfica
  em accuracy de FR. Esta observação motiva, na presente
  dissertação, a separação entre um efeito de pixel quality
  (Hipótese H6, formulada após Pangelinan) e o efeito de skin tone
  conditioning (Hipótese H5 original)."*

## 10. Arquivos relacionados

- PDF: pendente em `pdfs/pangelinan_2023.pdf`.
- Análise R6 completa em [[../_validacao_cientifica_pipeline]].
- Material reunião 2026-06-08 em [[../_reuniao_2026-06-08]].
- Entradas relacionadas: [[buolamwini_2018]], [[grother_2019]],
  [[dataset_wang_2019]], [[dooley_2022]].

## 11. Trabalhos sugeridos pelos autores (Future Work)

> **[PENDENTE PDF]** Não disponível no abstract.

## 12. Análise crítica do método

> **[BLOQUEADO ATÉ LEITURA INTEGRAL]** Análise crítica em 5
> dimensões requer inspeção do método, datasets, baselines e
> resultados completos. Versão preliminar:
>
> - **Aplicabilidade ao v3.2**: ALTA — refutação potencial de H5
>   é razão suficiente para citação obrigatória.
> - **Risco metodológico**: tarefa é FR verification, não
>   classification multi-classe. Tradução para nosso caso requer
>   cuidado.
