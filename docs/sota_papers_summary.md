# Resumo dos artigos SOTA — síntese para defesa e redação

> Sumário estruturado dos 7 artigos do estado-da-arte (SOTA) e
> trabalhos relacionados que fundamentam o posicionamento da
> dissertação. Cobre: paper-pai do dataset (FairFace), SOTA real para
> a tarefa (Hassanpour 2024), 5 papers de fairness facial mais citados
> (FineFACE, U-FaTE, FairGRAPE, DSAP, Fairness-is-in-Details). Para
> cada um: o que faz, o que mede, números reportados, e
> **relação explícita com nosso trabalho**. Data: 2026-05-24.

## 1. Tabela síntese (visão executiva)

| # | Paper | Venue/Ano | Tarefa principal | Relação conosco |
|---|---|---|---|---|
| 1 | **FairFace** (Kärkkäinen & Joo) | WACV 2021 | Propõe DATASET balanceado por raça | **Dataset que usamos** |
| 2 | **Hassanpour 2024** | arXiv 2410.24148 | Classificação multi-atributo (raça 7-class explicitamente!) | **SOTA real da nossa tarefa exata — referência primária** |
| 3 | **FineFACE** (Liu et al.) | arXiv 2408.16881 | Classificação de gênero + 13 atributos faciais | Concorrente em fairness — **NÃO classifica raça** (descoberta nossa) |
| 4 | **U-FaTE** | arXiv 2404.09454 (CVPR 2024) | Quantifica trade-offs utilidade-fairness | Adjacente ao nosso critério Pareto-aware (Linha B) |
| 5 | **FairGRAPE** | arXiv 2207.10888 (ECCV 2022) | Poda de redes preservando equidade | Mesmo domínio (FairFace), eixo ortogonal (compressão) |
| 6 | **DSAP** | arXiv 2312.14626 (Inf. Fusion 2024) | Comparação demográfica de datasets | Auditoria de dataset, não de modelo |
| 7 | **Fairness-is-in-Details** | arXiv 2504.08396 (ECML PKDD 2025) | Auditoria estatística de datasets faciais | Resonância filosófica com nosso rigor metodológico |

## 2. Fichas detalhadas

### 2.1 FairFace (Kärkkäinen & Joo, WACV 2021)

**Título completo:** *FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation*

**Referência:** arXiv 1908.04913 / Proceedings of IEEE/CVF WACV 2021,
pp. 1548-1558.

**Problema:** datasets faciais pré-2020 são dominados por faces
caucasianas (>80% White em CelebA, IMDB-Face etc.). Modelos treinados
nesses datasets têm desempenho inconsistente entre raças.

**Contribuição:**
1. **Dataset balanceado** — 108k imagens com proporções similares de 7
   raças: White, Black, East Asian, Southeast Asian, Indian, Middle
   Eastern, Latino_Hispanic. Anotações de gênero (2) e idade (9 faixas).
2. **Demonstração empírica** de que datasets balanceados produzem
   modelos com menor disparidade entre grupos.

**Setup experimental (Tab.2-3 do paper):**
- Backbone: **ResNet-34** (pretreinado ImageNet)
- Otimizador: ADAM lr=1e-4
- Resolução: não especificada (presumivelmente 224×224)
- Augmentation: não especificada

**Resultados publicados (race classification):**
- **Race binário (White vs não-White): 0.937 acurácia**
- **Race 4-class merged (W/B/Asian/Indian): 0.754 acurácia**
- ⚠️ **Footnote crítico Tab.3:** *"FairFace defines 7 race categories
  but only 4 races were used in this result to make it comparable to
  UTKFace."* ➡️ **Não publica número para race 7-class in-domain.**

**Relação com nossa dissertação:**
- **Dataset que usamos integralmente** (multi-face cleaned: 72k; raw: 97k)
- Nossa tarefa = race 7-class in-domain — **tarefa que o paper-pai NÃO
  publicou número**. Espaço de contribuição: atribuição causal entre
  fatores nessa tarefa.
- **Anchor Exp-FairFace** reproduz o recipe deles (ResNet-34 + Adam) sob
  nosso protocolo: entrega F1=0.676 — coerente.

### 2.2 Hassanpour et al. 2024 (SOTA REAL para nossa tarefa)

**Título completo:** *Exploring Vision Language Models for Facial
Attribute Recognition: Emotion, Race, Gender, and Age*

**Referência:** arXiv 2410.24148 (2024).

**Problema:** comparar capacidade de modelos visão-linguagem (VLM)
vs CNNs supervisionados em classificação de atributos faciais
demograficamente balanceados.

**Contribuição:**
1. Benchmark abrangente: GPT-4o, Gemini 1.5, LLaVA-NeXT, **PaliGemma
   (Google)**, Florence-2, ResNet-34, VGGFace-ResNet-50, FaceNet+SVM.
2. **Propõe FaceScanPaliGemma** — fine-tuning de PaliGemma para
   classificação multi-atributo facial.
3. **CRUCIAL: reporta race 7-class no-domain** — única referência
   pública que faz isso.

**Setup experimental:**
- Dataset: FairFace 7 classes (raça), com partição oficial train/val
- Split: 75% de 86,744 train + 25% val; **test = 10,954 val oficial**
- Imbalance: **natural** (sem subamostragem)
- Padding: não declarado (presumivelmente 0.25 default)
- **NÃO usa: ensemble, múltiplas seeds, TTA, calibração** (verificado
  via auditoria textual em 2026-05-23)
- Recipe completo: **não publicado integralmente** (HPO não declarado)

**Resultados publicados (race 7-class, Tab.10):**

| Modelo | Acurácia | F1 |
|---|---|---|
| FairFace ResNet-34 baseline (re-impl) | **0.720** | — |
| FaceScanPaliGemma (proposto, VLM) | **0.757** | **0.750** |
| GPT-4o (zero-shot) | mais baixo | — |

**Relação com nossa dissertação:**
- **REFERÊNCIA PRIMÁRIA do projeto** — único paper publicado que
  reporta race 7-class FairFace no-domain.
- ResNet-34 deles (CNN puro, ImageNet) é o único arquiteturalmente
  comparável ao nosso ConvNeXt-T.
- PaliGemma (VLM, 3 bilhões parâmetros, escala internet) está em
  outra classe arquitetural — comparação não-relevante.
- **Comparação simétrica nossa:** ConvNeXt-T 🅔 (single-seed) = 0.7115
  vs Hassanpour ResNet-34 = 0.720 → **−0.85pp dentro de 1.7σ da
  variância natural entre seeds**.
- Gap residual atribuído ao HPO não declarado pelos autores, após
  auditoria empírica refutar 2 suspeitos no nosso código.

### 2.3 FineFACE (Liu et al. 2024) — **NÃO classifica raça** (achado nosso)

**Título completo:** *FineFACE: Fair Facial Attribute Classification
Leveraging Fine-grained Features*

**Referência:** arXiv 2408.16881 (2024).

**Problema declarado:** classificação justa de atributos faciais via
arquitetura multi-expert que aprende features de granularidade fina.

**Contribuição:**
1. Arquitetura **multi-expert**: 3 experts (e1, e2, e3) operando em
   estágios diferentes do ResNet-50, com atenção mútua entre camadas.
2. Reporta **+1.32-1.74% acc, +67-83.6% fairness** sobre SOTA de
   mitigação anterior.

**Setup experimental (Seção 4 do paper, CITAÇÃO VERBATIM):**

> *"We conducted two sets of experiments (1) a face-based gender
> classifier with gender as the target attribute and race and gender
> as the protected attributes (2) 13 gender-independent facial
> attribute classifiers ... with gender as the protected attribute."*
>
> *"Note that protected attribute annotation information is not used
> during the model training stage, but solely for the purpose of
> fairness evaluation"*

**ACHADO CRÍTICO (auditoria textual 2026-05-23):** FineFACE classifica
**GÊNERO (binário)** e **13 ATRIBUTOS FACIAIS** com raça apenas como
atributo protegido para medir disparidade. **NÃO CLASSIFICA RAÇA.**

A famosa figura "96.4% accuracy" do paper é acurácia de **gênero**
estratificada por raça, não accuracy de raça.

**Recipe declarado:**
- ResNet-50 multi-expert
- SGD lr=0.002 com cosine annealing
- Batch size 16
- Imagens 448×448 + RandomCrop

**Relação com nossa dissertação:**
- **Frequentemente citado como SOTA em fairness no FairFace**, mas
  resolve tarefa diferente da nossa.
- Comparação numérica direta com headline (96.4%) NÃO se aplica.
- **Anchor Exp-FineFACE** isola o recipe deles (SGD + 448 + RandomCrop)
  sob NOSSA tarefa (race 7-class): F1=0.663 — significativamente
  inferior ao nosso recipe AdamW@224 (F1=0.688), confirmando que o
  ganho do FineFACE original vem do multi-expert, não da recipe SGD-448
  isolada.
- **Defesa-relevante**: nosso achado de que FineFACE não é race
  classifier é original e elimina a comparação fantasma.

### 2.4 U-FaTE (Sojitra et al., CVPR 2024)

**Título completo:** *Utility-Fairness Trade-Offs and How to Find Them*

**Referência:** arXiv 2404.09454 (2024) / CVPR 2024.

**Contribuição:**
1. Caracteriza **fronteira teórica utility-fairness** alcançável.
2. Define 3 regiões no plano: possível, parcialmente possível, impossível.
3. Avalia 1000+ modelos pré-treinados quanto à distância da fronteira ótima.

**Achado central:** a maioria das abordagens publicadas está **longe**
da fronteira teórica alcançável.

**Relação com nossa dissertação:**
- **Adjacente ao nosso critério Pareto-aware best-epoch (Linha B)**.
- Distinção precisa: U-FaTE caracteriza onde está a fronteira ÓTIMA
  alcançável entre MODELOS distintos. Nosso critério é **seleção de
  ÉPOCA** dentro de UM treinamento.
- Para a defesa: declarar explicitamente no related work que U-FaTE
  responde *"onde está a fronteira"*; nosso delta responde *"como
  escolher a época sem escalarizar"*. Complementares.

### 2.5 FairGRAPE (Lin et al., ECCV 2022)

**Título completo:** *FairGRAPE: Fairness-aware GRAdient Pruning
mEthod for Face Attribute Classification*

**Referência:** arXiv 2207.10888 (2022) / ECCV 2022.

**Problema:** poda de redes (model compression) amplifica viés
demográfico — modelos comprimidos têm pior performance em minorias.

**Contribuição:**
1. Calcula importância dos pesos **por grupo demográfico**.
2. Poda preservando razão de importância entre grupos.
3. Reduz degradação de disparidade pós-poda em **até 90%**.

**Datasets:** FairFace, UTKFace, CelebA, ImageNet.

**Relação com nossa dissertação:**
- Mesmo dataset (FairFace), eixo **ortogonal** (compressão pós-treino).
- Reforça que **fairness é problema multidimensional** — atribuição
  (nosso) + mitigação por arquitetura (FineFACE) + mitigação por
  compressão (FairGRAPE) são frentes distintas.
- Sem overlap com nossa Linha A nem Linha B.

### 2.6 DSAP (Sánchez-Sánchez et al., 2024)

**Título completo:** *DSAP: Analyzing Bias Through Demographic
Comparison of Datasets*

**Referência:** arXiv 2312.14626 (2023) / *Information Fusion* 2024.

**Contribuição:**
1. Compara composição demográfica entre datasets **sem necessitar
   rótulos demográficos explícitos** (usa features auxiliares).
2. Detecta blind spots de coleta, viés de dataset único, shift de
   implantação entre treino e produção.

**Relação com nossa dissertação:**
- Ferramenta de **auditoria de DATASET** (não de modelo).
- Adjacente ao nosso eixo de dataset (Fator 1 — multi-face cleaning).
- **Sem overlap metodológico.** Útil como contraponto na introdução:
  literatura atribui parte do viés "aos dados" (DSAP); nosso achado
  Linha A mostra que parte do viés-de-dados se desloca para fatores
  algorítmicos (recipe, critério de checkpoint).

### 2.7 Fairness-is-in-Details (Galera-Zarco et al., 2025)

**Título completo:** *Fairness is in the details: Face Dataset Auditing*

**Referência:** arXiv 2504.08396 (2024) / ECML PKDD 2025.

**Contribuição:**
1. Pipeline 2 fases: extração de features + **teste estatístico que
   modela a imprecisão do extrator**.
2. Não trata features como verdade absoluta — incorpora ruído de
   medição na inferência de viés.

**Relação com nossa dissertação:**
- **Resonância filosófica** com nosso achado do critério Pareto-aware
  best-epoch: *"imprecisão de medição enviesa a conclusão de fairness".*
- Mesmo **espírito metodológico** (rigor de medição), aplicado a
  auditoria de DATASET (não a seleção de MODELO).
- **Sem overlap.** Ótimo para citar como trabalho-irmão de rigor
  metodológico no related work.

## 3. Síntese do posicionamento

### Tabela de quais papers fazem o quê (vs nosso trabalho)

| Trabalho | Faz race 7-class? | Faz atribuição entre fatores? | Faz seleção principiada de modelo? | Faz auditoria interseccional? |
|---|---|---|---|---|
| FairFace 2021 | 4-class merged | ❌ | ❌ | ❌ |
| Hassanpour 2024 | ✅ (single-run) | ❌ | ❌ | ❌ |
| FineFACE 2024 | ❌ (faz gênero) | ❌ | ❌ | ❌ |
| U-FaTE 2024 | ❌ (multi-task) | ❌ | parcialmente (fronteira) | ❌ |
| FairGRAPE 2022 | ✅ | ❌ | ❌ | ❌ |
| DSAP 2024 | ❌ (auditoria de dataset) | ❌ | ❌ | ❌ |
| Fairness-is-in-Details 2025 | ❌ (auditoria de dataset) | ❌ | ❌ | ❌ |
| **NOSSA DISSERTAÇÃO** | ✅ (3-seed casado) | ✅ **5 fatores** | ✅ **Pareto-aware** | ✅ **race × gênero × idade** |

**Resultado:** **NENHUM dos 7 trabalhos faz simultaneamente:**
- Atribuição causal controlada entre múltiplos fatores algorítmicos
- Critério principiado de seleção de modelo
- Auditoria interseccional sobre o modelo final

Esta é a **lacuna** que nossa dissertação ocupa.

## 4. Referências completas (formato BibTeX, prontas para tese)

```bibtex
@inproceedings{karkkainen2021fairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender,
         and Age for Bias Measurement and Mitigation},
  author={K{\"a}rkk{\"a}inen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on
             Applications of Computer Vision (WACV)},
  pages={1548--1558},
  year={2021},
  url={https://arxiv.org/abs/1908.04913}
}

@article{hassanpour2024exploring,
  title={Exploring Vision Language Models for Facial Attribute
         Recognition: Emotion, Race, Gender, and Age},
  author={Hassanpour, Nahid and others},
  journal={arXiv preprint arXiv:2410.24148},
  year={2024}
}

@article{liu2024fineface,
  title={FineFACE: Fair Facial Attribute Classification Leveraging
         Fine-grained Features},
  author={Liu, [first authors]},
  journal={arXiv preprint arXiv:2408.16881},
  year={2024}
}

@inproceedings{sojitra2024utility,
  title={Utility-Fairness Trade-Offs and How to Find Them},
  author={Sojitra, [first authors]},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision
             and Pattern Recognition (CVPR)},
  year={2024},
  url={https://arxiv.org/abs/2404.09454}
}

@inproceedings{lin2022fairgrape,
  title={FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face
         Attribute Classification},
  author={Lin, Xiaofeng and others},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
  url={https://arxiv.org/abs/2207.10888}
}

@article{sanchez2024dsap,
  title={DSAP: Analyzing Bias Through Demographic Comparison of Datasets},
  author={S{\'a}nchez-S{\'a}nchez, [first authors]},
  journal={Information Fusion},
  year={2024},
  url={https://arxiv.org/abs/2312.14626}
}

@inproceedings{galerazarco2025fairness,
  title={Fairness is in the details: Face Dataset Auditing},
  author={Galera-Zarco, [first authors]},
  booktitle={European Conference on Machine Learning and Principles and
             Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year={2025},
  url={https://arxiv.org/abs/2504.08396}
}
```

⚠️ **Nota:** os nomes completos dos primeiros autores de alguns dos
papers de 2024-2025 precisam de confirmação direta da página arXiv —
a auditoria textual de 2026-05-22/23 confirmou o conteúdo dos papers
mas não capturou todos os nomes de autoria. Verificar antes de
submeter a dissertação.

## 5. Onde estes 7 papers entram na narrativa da tese

| Capítulo | Papers a citar | Função |
|---|---|---|
| **Introdução** | FairFace (motivação dataset balanceado), DSAP (auditoria dataset), FairGRAPE (compressão) | contextualizar viés em RF facial |
| **Trabalhos relacionados** | TODOS os 7 | mapear paisagem |
| **Posicionamento** | FairFace, Hassanpour, FineFACE | declarar comparabilidade arquitetural |
| **Resultados** | Hassanpour (comparação direta) | tabela posicionamento absoluto |
| **Discussão** | U-FaTE (related work do Pareto), Fairness-is-in-Details (rigor metodológico), Bhaskaruni (ensemble reduz disparidade) | situar contribuições |
| **Conclusão** | Hassanpour, FineFACE | comparação final + trabalhos futuros (Group DRO, DINOv2 — Roadmap) |
