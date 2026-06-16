---
data: 2026-06-16
tipo: mapa-citacoes
escopo: distribuicao das 101 fichas pelos 5 capitulos + adicoes v3.5/v1.2
status: ferramenta-de-escrita
versao: 1.1 (post-analise-NotebookLM)
---

# Mapa de citações por capítulo

> **Ferramenta operacional para a escrita LaTeX.** Lista, capítulo por
> capítulo e seção por seção, **quais fichas citar onde**. Cada entrada
> indica a chave BibTeX (de `docs/tese/referencias.bib`) e o motivo
> da citação.
>
> **Propósito**: acelerar a escrita evitando releitura do corpus para
> identificar quais papers entram em cada parte.
>
> **Atualizado em v1.1** (2026-06-16): incorporadas novas seções do
> objetivo v3.5 e narrativa v1.2:
> - Sensitivity analysis MST (OE-2 expandido)
> - Diversidade fenotípica intra-Latinx (OE-1 + H3)
> - Limites de escala (Cap 4 Discussão)
> - Ponte genética/CV (Cap 4 — Lewontin/Fuentes ↔ C6/OE-5/OE-6)

## Convenções

- `\cite{chave2024exemplo}` → chave da entrada em `referencias.bib`
- 🔴 = citação **obrigatória** (omissão é falha grave)
- 🟢 = citação **recomendada** (fortalece o argumento)
- 🟡 = citação **opcional** (contexto adicional, se couber espaço)
- **negrito** = ficha de Camada 1 (leitura integral verificada)

---

## Capítulo 1 — Introdução

### 1.1 Contexto (FR em uso massivo, motivação)

| Citação | Por que |
|---|---|
| 🔴 `\cite{grother2019face}` | NIST FRVT — escala industrial (189 algoritmos, 18M imagens, 8.5M pessoas) |
| 🔴 `\cite{buolamwini2018gender}` | **Marco fundador** Gender Shades (>10k citações) |
| 🟢 `\cite{lafargue2025fairness}` | Contexto regulatório EU AI Act, GDPR |
| 🟢 `\cite{voidface2025voidface}` | Regulações citadas (CCPA, PIPL, EU AI Act) |
| 🟡 `\cite{dpfedface2024dp}` | Privacy biométrico |

### 1.2 Problemas (disparidade documentada)

| Citação | Por que |
|---|---|
| 🔴 `\cite{aldahoul2024exploring}` | **SOTA atual** — F1 macro 75%, Latinx 60% vs Black 90% |
| 🔴 `\cite{karkkainen2021fairface}` | **Dataset central** + baseline 72% ResNet-34 |
| 🔴 `\cite{lin2022fairgrape}` | Validação independente baseline 72% |
| 🟢 `\cite{wang2019racial}` | RFW dataset — disparidade cross-race |
| 🟢 `\cite{robinson2020face}` | BFW — race × gender intersectional |

### 1.3 Balanceamento não resolve

| Citação | Por que |
|---|---|
| 🔴 **`\cite{pangelinan2023exploring}`** | **Refutação central** — 4 experimentos |
| 🔴 `\cite{rethinking2021rethinking}` (Hanson) | Refutação empírica: 75% male > balanced |
| 🟢 `\cite{kolla2022impact}` | 16 experimentos distribuição racial |
| 🟢 `\cite{wangzhang2022bupt}` (BUPT/MBN) | Precedente: skin tone > race labels |

### 1.4 Skin tone como dimensão alternativa

| Citação | Por que |
|---|---|
| 🔴 `\cite{schumann2023monk}` | **Monk Skin Tone** — escala padrão moderna |
| 🔴 `\cite{pereiramatias2026large}` | **SkinToneNet** — insumo Etapa 1 |
| 🔴 `\cite{fitzpatrick1988validity}` | Origem histórica FST |
| 🟢 `\cite{wangzhang2022bupt}` | Já argumentou em 2022 skin tone > race labels |
| 🟢 `\cite{hazirbas2021casual}` | Casual Conversations v1 — FST anotado |
| 🟢 `\cite{porgali2023casual}` | CCv2 Meta — FST + MST anotado |
| 🟡 `\cite{masseymartin2003nis}` | NIS Scale (precedente sociológico MST) |

### 1.5 Raça como construto social

| Citação | Por que |
|---|---|
| 🔴 `\cite{fuentes2019aapa}` | **Statement oficial AABA** — race ≠ biology |
| 🔴 `\cite{lewontin1972apportionment}` | Apportionment 85/6/8 — refutação genética |
| 🟢 `\cite{neto2025continuous}` | Continuous labels — questiona discretização |

---

## Capítulo 2 — Revisão de Literatura (11 tracks)

### 2.1 Track A — Race classification multi-classe

| Citação | Por que |
|---|---|
| 🔴 `\cite{karkkainen2021fairface}` | Dataset + baseline |
| 🔴 `\cite{aldahoul2024exploring}` | SOTA FaceScanPaliGemma |
| 🔴 `\cite{lin2022fairgrape}` | FairGRAPE + validação baseline |
| 🔴 `\cite{pereiramatias2026large}` | SkinToneNet — audita FairFace |

### 2.2 Track B — FR fairness (datasets + auditoria)

| Citação | Por que |
|---|---|
| 🔴 `\cite{wang2019racial}` (RFW), `\cite{robinson2020face}` (BFW), `\cite{wangzhang2022bupt}` (BUPT-Balancedface) | Datasets balanceados |
| 🔴 `\cite{grother2019face}` (NIST) | Auditoria industrial |
| 🔴 **`\cite{pangelinan2023exploring}`** | Refutação central |
| 🟢 `\cite{kolla2022impact}`, `\cite{rethinking2021rethinking}` | Balanceamento não basta |
| 🟢 `\cite{dooley2022rethinking}` | NAS arquitetural |
| 🟢 `\cite{image2021unravelling}` (distortions), `\cite{occlusion2024fairness}` | Confounders |
| 🟡 `\cite{mllm2026demographic}`, `\cite{mst2024mst}` | Trabalhos recentes |
| 🟡 `\cite{fairface2020fairface}` (ECCV challenge) | Challenge histórico |

### 2.3 Track C — Skin tone alternativo

| Citação | Por que |
|---|---|
| 🔴 `\cite{schumann2023monk}` (MST), `\cite{pereiramatias2026large}` (SkinToneNet) | Escala + classificador |
| 🔴 `\cite{fitzpatrick1988validity}`, `\cite{masseymartin2003nis}` | Precedentes históricos |
| 🟢 `\cite{hazirbas2021casual}` (CCv1), `\cite{porgali2023casual}` (CCv2) | Datasets anotados |
| 🟢 `\cite{buolamwini2018gender}` | Origem FST em fairness |
| 🟡 `\cite{lafargue2025fairness}` | Aplicação regulatória |

### 2.4 Track D — Mitigação algorítmica (baselines do Cap 2)

| Citação | Por que |
|---|---|
| 🔴 `\cite{park2022fair}` (FSCL+) | Contrastive baseline |
| 🔴 `\cite{sagawa2020distributionally}` (Group DRO) | DRO baseline |
| 🔴 `\cite{manzoor2024fineface}` (FineFACE) | Arquitetural baseline |
| 🔴 `\cite{zhang2018mitigating}` (Adversarial) | Adversarial baseline |
| 🟢 `\cite{liu2025component}` (BNMR) | Mais recente — meta-learning |
| 🟢 `\cite{bhaskaruni2019improving}` | Ensemble |
| 🟢 `\cite{dehdashtian2024utility}` (U-FaTE) | Trade-off theory |
| 🟡 `\cite{ramachandran2024self}` (SSL), `\cite{provable2024provable}` | SSL baselines |
| 🟡 `\cite{raumanns2024dataset}` | Multi-task fair |
| 🟡 `\cite{enhancing2022enhancing}` | DP/EO losses |

### 2.5 Track E — Surveys e auditoria

| Citação | Por que |
|---|---|
| 🔴 `\cite{mehrabi2021survey}` | Taxonomia canônica ML fairness |
| 🔴 `\cite{kotwal2025fairness}` | Survey específico FR fairness (TBIOM) |
| 🔴 `\cite{yucer2024racial}` | Survey racial bias FR (ACM CSur Durham) |
| 🟢 `\cite{parraga2025fairness}` (PUCRS) | **Survey brasileiro** ACM CSur |
| 🟢 `\cite{dominguezcatena2024dsap}` (DSAP) | Auditoria datasets |
| 🟢 `\cite{wang2022survey}` (OPPO) | Survey industrial |
| 🟡 `\cite{survey2024fairness}` (CV broader) | Contexto CV fairness |
| 🟡 `\cite{adewumi2024fairness}` (multimodal) | Multimodal fairness |
| 🟡 `\cite{gallegos2024bias}` (LLM bias) | LLM bias (para discussão CLIP) |
| 🟡 `\cite{yang2022survey}` (long-tail) | Gini coefficient como métrica auxiliar |

### 2.6 Track F — Fundamentação ética

| Citação | Por que |
|---|---|
| 🔴 `\cite{fuentes2019aapa}`, `\cite{lewontin1972apportionment}`, `\cite{neto2025continuous}` | Race construto + discretização |

### 2.7 Track G — Mecanismos paradigmáticos

| Citação | Por que |
|---|---|
| 🔴 `\cite{perez2018film}` (FiLM) | **Mecanismo central da tese** |
| 🔴 `\cite{hardt2016equality}` (EO/EOD), `\cite{kleinberg2017inherent}` (impossibility) | Métricas + triangulação |
| 🔴 `\cite{madras2018learning}` (LAFTR) | Fair transferência |
| 🟢 `\cite{zemel2013learning}` (LFR) | Paradigma fundador (Test-of-Time) |
| 🟢 `\cite{zhang2018mitigating}` (Adversarial) | Paradigma alternativo |
| 🟢 `\cite{aguirre2023transferring}` (multi-task NLP) | Suporte empírico LAFTR em NLP |
| 🟡 `\cite{counterfactual2025towards}` | Causal fairness |
| 🟡 `\cite{demographic2025demographic}` | Demographic-agnostic |

### 2.8 Track I — VLM/CLIP fairness (apoio ao C7 ablation)

| Citação | Por que |
|---|---|
| 🔴 `\cite{luo2024fairclip}` (FairCLIP CVPR 2024) | **Baseline principal C7** |
| 🔴 `\cite{dehdashtian2024fairerclip}` (FairerCLIP ICLR 2024) | Baseline leve C7 |
| 🟢 `\cite{bendvlm2024test}` (BendVLM) | Test-time |
| 🟢 `\cite{tan2026benchmarking}` (Benchmark LVLM ICLR 2026) | Benchmark |
| 🟢 `\cite{wu2024evaluating}` (Evaluating LVLM EMNLP 2024) | Evidência empírica |
| 🟡 outras Track I — citar conforme couber | Cobertura ampla |

### 2.9 Track J — Conditioning moderno

| Citação | Por que |
|---|---|
| 🟢 `\cite{tian2024fairvit}` (FairViT ECCV 2024) | ViT mechanism |
| 🟢 `\cite{bian2025lora}` (LoRA-FAIR ICCV 2025) | LoRA mechanism |
| 🟡 `\cite{ding2024fairness}` (LoRA fairness COLM 2024) | LoRA não enviesa |
| 🟡 `\cite{sukumaran2024fairlora}` (FairLoRA — Mila) | LoRA pesquisa canadense |

### 2.10 Track K — Fundadores de FR

| Citação | Por que |
|---|---|
| 🔴 `\cite{schroff2015facenet}` (FaceNet) | Paradigma triplet |
| 🔴 `\cite{deng2019arcface}` (ArcFace) | Loss canônico moderno |
| 🟢 `\cite{wang2018cosface}`, `\cite{meng2021magface}`, `\cite{kim2022adaface}` | Família margin-based |
| 🟡 `\cite{zhang2016range}` (Range Loss) | Long-tail FR |

### 2.11 Track L — Auxiliar/complementar

| Citação | Por que |
|---|---|
| 🟡 `\cite{terhorst2020post}` (post-hoc), `\cite{linghu2024score}` (score normalization) | Post-processing |
| 🟡 `\cite{yeung2024variface}` (Sony), `\cite{deandresttame2024second}` (FRCSyn) | Synthetic data |
| 🟡 `\cite{neto2024massively}` (MAC — **Univ. Porto**) | Synthetic + autoria portuguesa |
| 🟡 `\cite{caldeira2024mstkd}` (MST-KD — **Univ. Porto**) | KD para skin tone |
| 🟡 `\cite{mamede2024fairness}` (Occlusion bias — **Univ. Porto**) | Occlusion confounder |
| 🟡 `\cite{voidface2025voidface}` (**Univ. Coimbra**) | Privacy |
| 🟡 `\cite{tian2024fairdomain}` (FairDomain — Harvard) | Cross-domain |

---

## Capítulo 3 — Objetivos (recapitula referências do Cap 1-2)

Capítulo majoritariamente sem citações novas — referencia OEs e
Hipóteses já fundamentadas. Reusar chaves dos Caps 1-2.

---

## Capítulo 4 — Metodologia / Ampliação e busca de novas técnicas

### 4.1 Pipeline 6 etapas — fundamentação

| Etapa | Citações principais |
|---|---|
| 1 (SkinToneNet) | 🔴 `\cite{pereiramatias2026large}`, 🟢 `\cite{schumann2023monk}` |
| 2 (Auditoria MST × race) | 🔴 `\cite{pereiramatias2026large}` (audita mas não cross-tab), 🟢 `\cite{dominguezcatena2024dsap}` |
| 3 (ConvNeXt-T + FiLM) | 🔴 `\cite{perez2018film}`, 🟢 `\cite{dooley2022rethinking}` (arquitetura importa) |
| 4 (Triangulação métricas) | 🔴 `\cite{hardt2016equality}`, 🔴 `\cite{kleinberg2017inherent}`, 🟢 `\cite{dominguezcatena2024dsap}` |
| 5 (Fair transferência → FR) | 🔴 `\cite{madras2018learning}`, 🟢 `\cite{aguirre2023transferring}` |
| 6 (Decomposição variância) | 🔴 **`\cite{pangelinan2023exploring}`**, 🟢 `\cite{kolla2022impact}` |

### 4.2 Estudo comparativo de mecanismos (4 configurações)

| Config | Citações |
|---|---|
| A (baseline) | (sem citação específica) |
| B (FiLM-MST proposta) | 🔴 `\cite{perez2018film}` |
| C (Gated FiLM ablação) | 🟢 variantes não-lineares de FiLM |
| D (FiLM-CLIP avaliação) | 🔴 `\cite{radford2021learning}` (CLIP), `\cite{luo2024fairclip}` (FairCLIP) |

### 4.3 Baselines de mitigação (6)

| Baseline | Chave |
|---|---|
| ResNet-34 | `\cite{karkkainen2021fairface}` (canônico) |
| ConvNeXt-T puro | (controle) |
| FSCL+ | `\cite{park2022fair}` |
| Group DRO | `\cite{sagawa2020distributionally}` |
| FineFACE | `\cite{manzoor2024fineface}` |
| Adversarial | `\cite{zhang2018mitigating}` |

### 4.4 Datasets de teste

| Dataset | Chave |
|---|---|
| FairFace train+test | `\cite{karkkainen2021fairface}` |
| RFW (fair transfer) | `\cite{wang2019racial}` |
| BFW (fair transfer) | `\cite{robinson2020face}` |

### 4.5 Métrica auxiliar (Gini coefficient)

| Citação | Por que |
|---|---|
| 🟢 `\cite{yang2022survey}` (BUPT IJCV) | Gini coefficient como métrica de long-tailedness |

---

## Capítulo 5 — Cronograma

Citações administrativas/metodológicas (sem novas referências
científicas). Pode citar:

- 🟡 `\cite{lafargue2025fairness}` (EU AI Act timeline)
- 🟡 `\cite{kotwal2025fairness}` (state of art mais recente)

---

## Capítulo 4 — adições da v3.5/v1.2

### 4.6 Sub-análise intra-Latinx (NOVO v3.5)

| Citação | Por que |
|---|---|
| 🔴 `\cite{fuentes2019aapa}` | Posição AAPA — heterogeneidade fenotípica |
| 🔴 `\cite{lewontin1972apportionment}` | 85% variação intra-populacional |
| 🟢 `\cite{aldahoul2024exploring}` | F1 Latinx 60% — dado a explicar |
| 🟢 (R8 — candidatos a verificar) | Telles, Bonilla-Silva, Mora — diversidade Latinx (sociologia) — ver `_rodada_08_latinx_candidatos.md` |

### 4.7 Limites de escala (NOVO v3.5)

| Citação | Por que |
|---|---|
| 🔴 `\cite{grother2019face}` (NIST FRVT) | Escala industrial (18M imagens, 8.5M pessoas) — contraste com FairFace |
| 🟢 `\cite{karkkainen2021fairface}` | FairFace 108k — escala mestrado |
| 🟢 `\cite{pereiramatias2026large}` | Limitação reconhecida em audit datasets |
| 🟢 `\cite{yucer2024racial}` (Durham) | Survey limitações datasets |

### 4.8 Ponte genética ↔ visão computacional (NOVO v3.5)

| Citação | Por que |
|---|---|
| 🔴 **`\cite{lewontin1972apportionment}`** | Patamar 1 — genético-populacional |
| 🔴 **`\cite{fuentes2019aapa}`** | Patamar 2 — antropológico-institucional |
| 🟢 `\cite{wangzhang2022bupt}` (BUPT/MBN) | Skin tone > race labels (precedente) |
| 🟢 `\cite{neto2025continuous}` | Continuous labels (direção futura) |

### 4.9 Sensitivity analysis MST (NOVO v3.5 — OE-2 expandido)

| Citação | Por que |
|---|---|
| 🔴 **`\cite{pereiramatias2026large}`** (SkinToneNet) | Classificador #1 (principal) |
| 🟢 `\cite{schumann2023monk}` (MST baseline) | Classificador #2 (validação) |
| 🟢 `\cite{wangzhang2022bupt}` (BUPT IDS + FST+ITA) | Classificador #3 (cross-paradigm) |

## Apêndices / Discussão ética

| Citação | Por que |
|---|---|
| 🔴 `\cite{fuentes2019aapa}`, `\cite{lewontin1972apportionment}` | Posição ética sobre raça |
| 🟢 `\cite{neto2025continuous}` | Limitação reconhecida — discretização |
| 🟢 `\cite{parraga2025fairness}` (PUCRS) | Produção brasileira |
| 🟢 `\cite{lafargue2025fairness}` | Regulamentação |

---

## Estatísticas do mapa

| Categoria | Quantidade |
|---|---|
| Citações 🔴 obrigatórias | ~25 chaves canônicas |
| Citações 🟢 recomendadas | ~40 chaves de suporte |
| Citações 🟡 opcionais | ~36 chaves contextuais |
| **Total** | **101 fichas alocadas** |

---

## Observações operacionais para a escrita

1. **Chaves BibTeX** estão em `docs/tese/referencias.bib`. Para
   verificar a chave exata de uma ficha, abrir o `.bib` e localizar
   pela linha `% origem: <ficha>.md` que precede cada entrada.
2. **Camada 1 (14 fichas críticas)** em negrito devem ser citadas
   com profundidade nos respectivos capítulos.
3. **Pesquisas brasileiras/portuguesas** estão sinalizadas como
   diferencial nacional — incluir em pelo menos uma seção
   (e.g., 3.7 ou conclusão).
4. **Quantidade típica de citações por capítulo**:
   - Cap 1: 25-30 citações (introdução panorâmica)
   - Cap 2: 60-80 citações (revisão extensa por tracks)
   - Cap 3: 5-10 citações (objetivos, reusa chaves)
   - Cap 4: 15-20 citações (metodologia + baselines)
   - Cap 5: 3-5 citações (cronograma)
5. **Conflitos endereçados defensivamente** devem ser citados
   explicitamente com a resposta defensiva mapeada em
   `_revisao_critica_corpus_v2.md`.
