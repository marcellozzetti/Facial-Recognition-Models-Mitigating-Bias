# INDEX — Pesquisa Bibliográfica

> Navegação central das **101 fichas do corpus** distribuídas em
> **11 tracks temáticos**. Para conteúdo integral do paper consultar
> PDF local em `pdfs/`. Para resumo estruturado em 11–12 seções
> consultar ficha `.md`.
>
> Atualizado: 2026-06-15 (pós-reunião com orientador — corpus
> consolidado, pente fino crítico concluído, 14 fichas Camada 1
> todas VERIFIED). Estrutura final em 11 tracks: A-G (originais
> R1-R5) + I, J, K (criados na ampliação recente do corpus) + L
> (auxiliar). Track K (fundadores de FR) e Track L (auxiliar)
> adicionados ao INDEX nesta atualização.

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

> **11 tracks consolidados após pente fino crítico**. Distribuição
> total: 101 fichas. Tracks A-G originais (R1-R5); Tracks I, J, K
> criados na ampliação recente; Track L é o auxiliar/complementar.

### Track A — Race classification multi-classe (4 fichas)

Tarefa central da dissertação.

| Paper | Foco |
|---|---|
| [Karkkainen & Joo 2021](dataset_karkkainen_2021.md) | FairFace — dataset central |
| [AlDahoul et al. 2024/2026](aldahoul_2024.md) | FaceScanPaliGemma — SOTA atual |
| [Lin et al. 2022](lin_2022.md) | FairGRAPE — pruning + valida baseline 72% |
| [Pereira et al. 2026](pereira_2026.md) | SkinToneNet + STW + auditoria FairFace |

### Track B — Face recognition fairness (16 fichas)

Datasets, causas e crítica em FR fairness.

| Paper | Foco |
|---|---|
| [Wang et al. 2019](dataset_wang_2019.md) | RFW |
| [Robinson et al. 2020](dataset_robinson_2020.md) | BFW |
| [BUPT 2019](dataset_bupt_2019.md) | BUPT-CBFace |
| [Grother et al. 2019](grother_2019.md) | NIST FRVT Part 3 |
| [Pangelinan et al. 2023](pangelinan_2023.md) | Pixel info > skin tone (refutação central) |
| [Kolla & Savadamuthu 2022](kolla_2022.md) | Distribuição racial no treino |
| [Dooley et al. 2022](dooley_2022.md) | NAS bi-objective fairness+accuracy |
| [Image Distortions 2021](image_distortions_2021.md) | Distortions e bias |
| [Occlusion Bias 2024](occlusion_bias_2024.md) | Occlusion como confounder |
| [Robustness Face Detection 2022](robustness_face_detection_2022.md) | Robustez vs fairness |
| [Rethinking Assumptions 2021](rethinking_assumptions_2021.md) | Crítica metodológica |
| [Fairer Datasets 2024](fairer_datasets_2024.md) | Construção de datasets fairer |
| [Racial Bias Dataset 2017](racial_bias_dataset_2017.md) | Primeiro audit racial |
| [MLLM Face Verification 2026](mllm_face_verification_2026.md) | MLLM para verificação |
| [MST-KD 2024](mst_kd_2024.md) | Knowledge distillation MST |
| [FairFace Challenge ECCV2020](fairface_challenge_eccv2020.md) | Challenge oficial |

### Track C — Skin tone alternativo (7 fichas)

Escalas e classificadores de tom de pele.

| Paper | Foco |
|---|---|
| [Fitzpatrick 1988](fitzpatrick_1988.md) | Origem dermatológica FST |
| [Massey & Martin 2003](massey_martin_2003.md) | NIS Skin Color Scale (precedente MST) |
| [Schumann et al. 2023](schumann_2023.md) | Monk Skin Tone (MST) scale |
| [Hazirbas et al. 2021](dataset_hazirbas_2021.md) | Casual Conversations v1 |
| [Porgali et al. 2023 CVPRW](porgali_2023_ccv2.md) | Casual Conversations v2 (Meta) |
| [Buolamwini & Gebru 2018](buolamwini_2018.md) | Gender Shades — pioneiro |
| [Lafargue et al. 2025](lafargue_2025.md) | Fitzpatrick + ITA |

### Track D — Mitigação algorítmica (10 fichas)

Técnicas para reduzir viés.

| Paper | Técnica |
|---|---|
| [Park et al. 2022](park_2022.md) | Fair Supervised Contrastive Loss (FSCL+) |
| [Sagawa et al. 2020](sagawa_2020.md) | Group DRO + strong regularization |
| [Manzoor & Rattani 2024](manzoor_2024.md) | Cross-layer mutual attention (FineFACE) |
| [Bhaskaruni et al. 2019](bhaskaruni_2019.md) | Ensemble AdaBoost-fair |
| [Dehdashtian et al. 2024](dehdashtian_2024.md) | U-FaTE (estimador trade-off) |
| [Ramachandran & Rattani 2024 IJCB](ramachandran_2024.md) | SSL para fair facial attribute |
| [Provable Adversarial SSL 2024](provable_adversarial_ssl_2024.md) | SSL fair com garantias |
| [Raumanns et al. 2024 FAIMI](raumanns_2024.md) | Single vs multi-task fairness |
| [Liu et al. 2025 FAccT](liu_2025.md) | BNMR — Bayesian Network meta-learning |
| [Enhancing Visual Attributes 2022](enhancing_visual_attributes_2022.md) | Atributos visuais fair |

### Track E — Auditoria e Surveys (14 fichas)

Infraestrutura intelectual.

| Paper | Foco |
|---|---|
| [Dominguez-Catena et al. 2024](dominguez_2024.md) | DSAP — auditoria de datasets |
| [Reliable Demographic Inference 2025](reliable_demo_inference_2025.md) | DAI pipeline modular |
| [Evaluating LVLM Fairness 2024 EMNLP](evaluating_lvlm_2024.md) | FACET+UTKFace audit |
| [Benchmark LVLM 2026](benchmark_lvlm_2026.md) | Benchmark VLM fairness |
| [Mehrabi et al. 2021](survey_mehrabi_2021.md) | Survey geral ML — taxonomia canônica |
| [Kotwal & Marcel 2025](survey_kotwal_2025.md) | Survey específico FR fairness |
| [Survey Racial Bias FR 2024](survey_racial_bias_fr_2024.md) | Survey racial bias |
| [Survey Face Recognition 2022](survey_face_recognition_2022.md) | Survey FR geral |
| [Survey CV Fairness 2024](survey_cv_fairness_2024.md) | Survey CV fairness |
| [Survey LLM Bias 2024](survey_llm_bias_2024.md) | Survey LLM bias |
| [Survey Multimodal Fairness 2024](survey_multimodal_fairness_2024.md) | Survey multimodal |
| [Survey Fairness Vision-Language 2024](survey_fairness_vision_lang_2024.md) | Survey VL fairness |
| [Survey Long-Tail 2022](survey_long_tail_2022.md) | Long-tail recognition |
| [Federated Fairness Survey 2025](federated_fairness_survey_2025.md) | Survey federated fairness |

### Track F — Fundamentação científica e ética (3 fichas)

Posição da tese sobre raça como construto.

| Paper | Foco |
|---|---|
| [Fuentes et al. 2019](fuentes_2019.md) | AAPA Statement — race ≠ biology |
| [Lewontin 1972](lewontin_1972.md) | Apportionment 85/6/8 — refutação genética |
| [Neto et al. 2025](neto_2025.md) | Continuous labels — questiona discretização |

### Track G — Mecanismos ML paradigmáticos (9 fichas)

Esqueleto teórico — métricas, conditioning, representação.

| Paper | Foco | Relevância |
|---|---|---|
| [Hardt, Price & Srebro 2016](hardt_2016.md) | Equal Opportunity / Equalized Odds | Métricas canônicas |
| [Kleinberg et al. 2017](kleinberg_2017.md) | Impossibility theorem | Justifica triangulação |
| [Zemel et al. 2013](zemel_2013.md) | LFR — Learning Fair Representations | Paradigma fundador |
| [Madras et al. 2018](madras_2018.md) | LAFTR — Adversarial Fair Transferable | Fundamenta fair transfer |
| [Zhang et al. 2018](zhang_2018.md) | Adversarial debiasing | Baseline alternativo |
| [Perez et al. 2018](perez_2018.md) | FiLM — mecanismo de conditioning | **Mecanismo central da tese** |
| [Aguirre & Dredze 2023](aguirre_2023.md) | Multi-task fair transfer | Reforço empírico do LAFTR |
| [Counterfactual Fairness ICLR 2025](counterfactual_fairness_iclr2025.md) | Causal fairness | Direção alternativa |
| [Demographic-Agnostic Fairness 2025](demographic_agnostic_2025.md) | Fairness sem labels | Direção futura |

### Track I — VLM / CLIP em fairness (14 fichas) — ampliação recente

Resposta direta à recomendação do orientador sobre CLIP.

| Paper | Foco |
|---|---|
| [Luo et al. 2024 CVPR — FairCLIP](luo_2024_fairclip.md) | Sinkhorn distance + Harvard-FairVLMed |
| [Dehdashtian et al. 2024 ICLR — FairerCLIP](dehdashtian_2024_fairerclip.md) | RKHS-based debiasing zero-shot |
| [Joint VL Bias Removal 2024](joint_vl_2024.md) | Alignment + counterfactual |
| [Unified Debiasing VLMs 2024](unified_debiasing_vlm_2024.md) | Framework cross-modal |
| [BendVLM 2024](bendvlm_2024.md) | Test-time debiasing |
| [Debiasing CLIP Neural Interventions 2025](debiasing_neural_interventions_2025.md) | Attention heads |
| [Closed-form Debias 2026](closed_form_debias_2026.md) | Solução analítica |
| [Bias Subspace 2025](bias_subspace_2025.md) | Projeção em subespaço |
| [Fair Residuals VLM 2025](fair_residuals_vlm_2025.md) | Residual fairness |
| [BioPro 2025](biopro_2025.md) | Biometric prototypes |
| [Lin et al. 2025 CVPR — AI-Face](lin_2025_aiface.md) | Million-scale dataset |
| [GRAS Benchmark 2025](gras_2025.md) | 2.5M queries multi-eixo |
| [IndicFairFace 2026](indicfairface_2026.md) | Foco em Indian faces |
| [Survey Fairness Vision-Language 2024](survey_fairness_vision_lang_2024.md) | Survey transversal |

### Track J — Conditioning moderno (5 fichas) — ampliação recente

Adaptadores leves alternativos ao FiLM.

| Paper | Foco |
|---|---|
| [Zhao et al. 2025 CVPR — AIM-Fair](zhao_2025_aimfair.md) | Selective fine-tuning |
| [Bian et al. 2025 ICCV — LoRA-FAIR](bian_2025_lorafair.md) | Federated LoRA fairness |
| [FairLoRA 2024](fairlora_2024.md) | Fairness-driven LoRA vision |
| [On Fairness of LoRA 2024](fairness_lora_2024.md) | LoRA não enviesa per se |
| [Tian et al. 2024 ECCV — FairViT](tian_2024_fairvit.md) | Adaptive masking ViT |

### Track K — Fundadores de FR (6 fichas) — ampliação recente

Losses fundadoras de reconhecimento facial — preenche gap óbvio.

| Paper | Foco |
|---|---|
| [Schroff et al. 2015](schroff_2015_facenet.md) | FaceNet — Triplet loss |
| [Wang et al. 2018](wang_2018_cosface.md) | CosFace — LMCL |
| [Deng et al. 2019](deng_2019_arcface.md) | ArcFace — Additive Angular Margin |
| [Meng et al. 2021](meng_2021_magface.md) | MagFace — Quality + Magnitude |
| [Kim et al. 2022](kim_2022_adaface.md) | AdaFace — Quality-adaptive margin |
| [Range Loss 2016](range_loss_2016.md) | Range Loss — long-tail FR |

### Track L — Auxiliar / complementar (13 fichas)

Direções adjacentes — post-hoc, synthetic, federated, cross-domain, explainability.

#### Post-hoc / Calibration (4)

| Paper | Foco |
|---|---|
| [Post-hoc Comparison 2020](post_comparison_2020.md) | Comparação de métodos post-hoc |
| [FairCal 2021](faircal_2021.md) | Fair calibration |
| [Score Normalization 2024](score_normalization_2024.md) | Normalização de scores |
| [FairSight 2025](fair_sight_2025.md) | Auditoria visual |

#### Synthetic data (5)

| Paper | Foco |
|---|---|
| [Synthetic Face 2024](synthetic_face_2024.md) | Synthetic faces para fairness |
| [VariFace 2024](variface_2024.md) | Variational synthetic |
| [FRCSyn 2024](frcsyn_2024.md) | Challenge synthetic FR |
| [Massively Annotated 2024](massively_annotated_2024.md) | Large-scale synthetic |
| [FairImagen NeurIPS 2025](fairimagen_neurips2025.md) | Fair image generation |

#### Federated / Privacy (2)

| Paper | Foco |
|---|---|
| [DP-FedFace 2024](dp_fedface_2024.md) | Differential privacy + federated |
| [VoIDFace 2025](voidface_2025.md) | Privacy-preserving FR |

#### Cross-domain (2)

| Paper | Foco |
|---|---|
| [FairDomain 2024](fairdomain_2024.md) | Cross-domain fairness |
| [Face4FairShifts 2025](face4fairshifts_2025.md) | Distribution shifts |

#### Explainability (1)

| Paper | Foco |
|---|---|
| [Explainable FR 2024](explainable_fr_2024.md) | Explicabilidade em FR |

## Localização de PDFs

PDFs em `pdfs/`, versionados no repositório (~400 MB).

**Cobertura atual: 92 de 101 fichas (91 %) com PDF presente.**

| Categoria | Total |
|---|---|
| Fichas com PDF no repositório | 92 |
| Fichas sem PDF (paywall institucional ou sem URL pública) | 9 |

Os 9 PDFs ausentes estão em fontes históricas (Fitzpatrick 1988,
Lewontin 1972, Massey-Martin 2003) ou em paywalls de Springer / ACM /
Wiley (a serem obtidos via VPN institucional).

Inventário detalhado em [_pdfs_inventario.md](../_pdfs_inventario.md).

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
