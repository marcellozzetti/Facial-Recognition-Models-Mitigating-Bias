# INDEX — Pesquisa Bibliográfica

> Navegação central das **37 fichas do corpus** (R1-R6 completas).
> Para conteúdo integral do paper consultar PDF local. Para resumo
> estruturado em 11–12 seções consultar ficha `.md`.
>
> Atualizado: 2026-06-10 (Rodada 6 COMPLETA — 8 fichas R6 fichadas:
> [pereira_2026](pereira_2026.md) e [aguirre_2023](aguirre_2023.md)
> como VERIFIED; demais 6 como OVERVIEW_ONLY pendentes leitura PDF
> integral. Análise R6 completa em
> [_validacao_cientifica_pipeline.md](../_validacao_cientifica_pipeline.md)).

## Ordem de leitura recomendada

Priorização para **construir entendimento progressivo** da literatura
de fairness racial em biometria facial — começa pela motivação
ética, passa pelo dataset central, vai para mitigações, e termina
com surveys + papers críticos.

| # | Paper | Por que ler aqui |
|---|---|---|
| 1 | [Buolamwini & Gebru 2018](buolamwini_2018.md) — Gender Shades | Marco fundador; estabelece o problema |
| 2 | [Karkkainen & Joo 2021](dataset_karkkainen_2021.md) — FairFace | Dataset central da dissertação |
| 3 | [AlDahoul et al. 2024/2026](aldahoul_2024.md) — VLMs/FaceScanPaliGemma | SOTA atual para race 7-class FairFace |
| 4 | [Lin, Kim & Joo 2022](lin_2022.md) — FairGRAPE | Validação cruzada do baseline 72% + mitigação |
| 5 | [Wang et al. 2019](dataset_wang_2019.md) — RFW | Dataset paralelo (4-class), evidência convergente |
| 6 | [Robinson et al. 2020](dataset_robinson_2020.md) — BFW | Outro dataset balanceado, threshold adaptativo |
| 7 | [Grother, Ngan & Hanaoka 2019](grother_2019.md) — NISTIR 8280 | Escala industrial, distinção FR vs classification |
| 8 | [Manzoor & Rattani 2024](manzoor_2024.md) — FineFACE | Armadilha textual (gender ≠ race); Pareto-efficient em CelebA |
| 9 | [Park et al. 2022](park_2022.md) — FSCL | Fair contrastive learning; **NÃO testou em FairFace** |
| 10 | [Sagawa et al. 2020](sagawa_2020.md) — Group DRO | Worst-group + strong regularization |
| 11 | [Dehdashtian et al. 2024](dehdashtian_2024.md) — U-FaTE | Esqueleto teórico do trade-off accuracy×fairness |
| 12 | [Bhaskaruni et al. 2019](bhaskaruni_2019.md) — Ensemble fair | Ensemble naive ≠ fairness; AdaBoost-fair funciona |
| 13 | [Hazirbas et al. 2021](dataset_hazirbas_2021.md) — Casual Conversations | Self-reported + Fitzpatrick; gold standard de anotação |
| 14 | [Schumann et al. 2023](schumann_2023.md) — MST consensus | Monk Skin Tone Scale + protocolo de anotação |
| 15 | [Dominguez-Catena et al. 2024](dominguez_2024.md) — DSAP | Auditoria de datasets; métricas unificadas [0,1] |
| 16 | [Lafargue, Claeys & Loubes 2025](lafargue_2025.md) — Fairness in Details | EU AI Act + uncertainty-aware testing |
| 17 | [Neto et al. 2025](neto_2025.md) — Continuous Labels | Questiona discretização per se |
| 18 | [Mehrabi et al. 2021](survey_mehrabi_2021.md) — Survey ACM CSur | Taxonomia canônica de bias e fairness |
| 19 | [Kotwal & Marcel 2025](survey_kotwal_2025.md) — Survey TBIOM | Survey mais recente focado em FR fairness |
| 20 | [Fuentes et al. 2019](fuentes_2019.md) — AAPA Statement | Statement oficial sobre raça (fundamento teórico) |
| 21 | [Lewontin 1972](lewontin_1972.md) — Apportionment of Human Diversity | Evidência genética clássica (85/6/8 partition) |
| 22 | [Fitzpatrick 1988](fitzpatrick_1988.md) — Sun-Reactive Skin Types | Origem dermatológica da escala Fitzpatrick |
| 23 | [Massey & Martin 2003](massey_martin_2003.md) — NIS Skin Color Scale | Precedente sociológico do MST |

## Por track temático

### Track A — Race classification (alvo direto da dissertação)

| Paper | Tipo |
|---|---|
| [Karkkainen & Joo 2021](dataset_karkkainen_2021.md) | Dataset central |
| [AlDahoul et al. 2024/2026](aldahoul_2024.md) | SOTA |
| [Lin et al. 2022](lin_2022.md) | Mitigação + validação |

### Track B — Face recognition fairness (paralelo)

| Paper | Tipo |
|---|---|
| [Wang et al. 2019](dataset_wang_2019.md) | RFW |
| [Robinson et al. 2020](dataset_robinson_2020.md) | BFW |
| [Grother et al. 2019](grother_2019.md) | NIST FRVT |
| [Neto et al. 2025](neto_2025.md) | Continuous labels |

### Track C — Skin tone como dimensão alternativa

| Paper | Tipo |
|---|---|
| [Buolamwini & Gebru 2018](buolamwini_2018.md) | Fitzpatrick (PPB) |
| [Hazirbas et al. 2021](dataset_hazirbas_2021.md) | Casual Conversations |
| [Schumann et al. 2023](schumann_2023.md) | MST scale |
| [Lafargue et al. 2025](lafargue_2025.md) | Fitzpatrick + ITA |

### Track D — Mitigação algorítmica (técnicas)

| Paper | Técnica |
|---|---|
| [Park et al. 2022](park_2022.md) | Fair Supervised Contrastive Loss (FSCL+) |
| [Sagawa et al. 2020](sagawa_2020.md) | Group DRO + strong regularization |
| [Manzoor & Rattani 2024](manzoor_2024.md) | Cross-layer mutual attention (FineFACE) |
| [Bhaskaruni et al. 2019](bhaskaruni_2019.md) | Ensemble AdaBoost-fair |
| [Dehdashtian et al. 2024](dehdashtian_2024.md) | U-FaTE (estimador trade-off) |
| [Lin et al. 2022](lin_2022.md) | FairGRAPE pruning |

### Track E — Auditoria e metodologia

| Paper | Foco |
|---|---|
| [Dominguez-Catena et al. 2024](dominguez_2024.md) | DSAP — auditoria de datasets |
| [Lafargue et al. 2025](lafargue_2025.md) | Statistical tests uncertainty-aware |
| [Mehrabi et al. 2021](survey_mehrabi_2021.md) | Survey geral ML |
| [Kotwal & Marcel 2025](survey_kotwal_2025.md) | Survey específico FR |

### Track F — Fundamentação científica de raça e tom de pele (Rodada 4)

| Paper | Foco |
|---|---|
| [Fuentes et al. 2019](fuentes_2019.md) | AAPA Statement — race ≠ biology |
| [Lewontin 1972](lewontin_1972.md) | Apportionment 85/6/8 — refutação genética |
| [Fitzpatrick 1988](fitzpatrick_1988.md) | Origem dermatológica Fitzpatrick (PUVA) |
| [Massey & Martin 2003](massey_martin_2003.md) | NIS Skin Color Scale (precedente MST) |

### Track H — Validação científica do pipeline v3.2 (Rodada 6 COMPLETA)

#### Fichas VERIFIED (leitura integral)

| Paper | Foco | Relevância v3.2 |
|---|---|---|
| [Pereira et al. 2026](pereira_2026.md) — SkinToneNet + STW | Classificador MST SOTA (ViT-Small) + dataset STW (42k imgs) auditando 8 datasets faciais incluindo FairFace | **Insumo direto da Etapa 1**: SkinToneNet pré-treinado. Contaminação treino-teste: STW agrega FairFace |
| [Aguirre & Dredze 2023](aguirre_2023.md) — Multi-task fair transfer | MTL-fair + ε-DEO; 15-44% redução de bias mantendo F1; demonstração empírica de fair transfer | Reforço empírico do LAFTR. Justifica etapas 3 e 5 |

#### Fichas OVERVIEW_ONLY (pendentes leitura PDF integral)

| Paper | Foco | Relevância v3.2 |
|---|---|---|
| [Pangelinan et al. 2023](pangelinan_2023.md) | Causas de variação demográfica em FR accuracy | **Refutação potencial de H5**: pixel info > skin tone. Motivou H6 nova |
| [Dooley et al. 2022](dooley_2022.md) — Fairer architectures | NAS bi-objective fairness+accuracy | Reforça H2: arquitetura importa; valida ConvNeXt-T |
| [Kolla & Savadamuthu 2022](kolla_2022.md) | Impacto da distribuição racial no treino de FR | Balanceamento NÃO basta; "racial gradation" novo |
| [Liu et al. 2025 FAccT](liu_2025.md) — BNMR | Bayesian Network meta-learning + face component fairness | Baseline competitivo recente; surrogate fairness alinhada |
| [Ramachandran & Rattani 2024 IJCB](ramachandran_2024.md) | SSL pipeline para fair facial attribute | Baseline SSL em FairFace + CelebA |
| [Raumanns et al. 2024 FAIMI](raumanns_2024.md) | Single vs multi-task fairness em skin lesion | Cautela contra multi-task naive; adversarial > reinforce |

Detalhes em [_validacao_cientifica_pipeline.md](../_validacao_cientifica_pipeline.md).
Auditoria de leitura em [_auditoria_fichas.md](_auditoria_fichas.md).

### Track G — Mecanismos ML / Redes Neurais (NOVO Rodada 5)

| Paper | Foco | Relevância v3.2 |
|---|---|---|
| [Hardt, Price & Srebro 2016](hardt_2016.md) | NeurIPS 2016 — Equal Opportunity / Equalized Odds | Fonte das métricas EO_h/EOD usadas pela literatura |
| [Perez et al. 2018 (FiLM)](perez_2018.md) | AAAI 2018 — conditional layer | Mecanismo para condicionar race classifier com saída MST |
| [Zemel et al. 2013 (LFR)](zemel_2013.md) | ICML 2013 (Test-of-Time 2023) — Fair Representations | Paradigma fundador do campo |
| [Madras et al. 2018 (LAFTR)](madras_2018.md) | ICML 2018 — Adversarial Fair Transferable | Fundamenta extensão race → face recognition |
| [Zhang et al. 2018](zhang_2018.md) | AAAI/ACM AIES 2018 — Adversarial debiasing | Baseline mitigação alternativo a FSCL+ |
| [Kleinberg et al. 2017](kleinberg_2017.md) | ITCS 2017 — Impossibility theorem | Justifica triangulação DR + worst-class + CV |

## Localização de PDFs (gitignored — local apenas)

Todos os PDFs em `pdfs/`:

```
pdfs/
├── aldahoul_2024_vlm.pdf
├── aldahoul_2026_naturesr.pdf       # follow-up Nature SR
├── bhaskaruni_2019_ensemble.pdf
├── buolamwini_2018_gendershades.pdf
├── dehdashtian_2024_ufate.pdf
├── dominguez_2024_dsap.pdf
├── grother_2019_nistir8280.pdf
├── hazirbas_2021_casual.pdf
├── karkkainen_2021_fairface.pdf
├── kotwal_2025_survey.pdf
├── lafargue_2025_fairdetails.pdf
├── lin_2022_fairgrape.pdf
├── manzoor_2024_fineface.pdf
├── mehrabi_2021_survey.pdf
├── neto_2025_continuous.pdf
├── park_2022_fscl.pdf
├── robinson_2020_bfw.pdf
├── sagawa_2020_groupdro.pdf
├── schumann_2023_mst.pdf
└── wang_2019_rfw.pdf
```

Tamanho total: ~83 MB. Não-versionados (gitignored por política de
copyright editorial).

## Artefatos administrativos

- [README.md](README.md) — metodologia da Pesquisa Bibliográfica.
- [_triagem.md](_triagem.md) — log de decisões editoriais (R1, R2, R2.5, R3, R4, R5, R2.6).
- [_perguntas.md](_perguntas.md) — Q01–Q14 respondidas, 5 frentes 🔬 abertas.
- [_metricas_corpus.md](_metricas_corpus.md) — **big numbers do corpus** para uso em reuniões com orientador.
- [`../00_referencias.md`](../00_referencias.md) — lista canônica de citações verificadas.

## Próximos arquivos (Fase 4)

A serem produzidos:

- `../05_landscape.md` — síntese transversal das 19 fichas (em construção).
- `../06_gap.md` — gaps consolidados, 5 frentes 🔬 ranqueadas.
- `../07_thesis_statement.md` — v3 reformulado sobre o gap escolhido.
