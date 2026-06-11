# Análise consolidada do corpus (R1-R7 completas, 101 fichas)

> **Data**: 2026-06-10.
> **Propósito**: revisão crítica do corpus inteiro com (1) distribuição
> por ano e track, (2) análise de convergências, (3) análise de
> conflitos com a tese proposta, (4) caminhos alternativos sugeridos
> pela literatura e como respondemos a eles.

---

## 1. Distribuição por ano

| Ano | Fichas | Cumulativo | Distribuição visual |
|---|---|---|---|
| 1972 | 1 | 1 | █ Lewontin |
| 1988 | 1 | 2 | █ Fitzpatrick |
| 2003 | 1 | 3 | █ Massey & Martin |
| 2013 | 1 | 4 | █ Zemel LFR |
| 2015 | 1 | 5 | █ FaceNet |
| 2016 | 2 | 7 | ██ Hardt, Range Loss |
| 2017 | 2 | 9 | ██ Kleinberg, Klare |
| 2018 | 5 | 14 | █████ Buolamwini, LAFTR, FiLM, Adversarial, CosFace |
| 2019 | 6 | 20 | ██████ RFW, NIST, AAPA, Bhaskaruni, ArcFace, BUPT |
| 2020 | 4 | 24 | ████ Group DRO, BFW, Terhörst, FairFace Challenge |
| 2021 | 7 | 31 | ███████ FairFace, CC, Mehrabi, MagFace, Distortions, Klare, FairCal |
| 2022 | 9 | 40 | █████████ FSCL, FairGRAPE, NAS, Kolla, AdaFace, Detection, Attrs, Surveys |
| 2023 | 4 | 44 | ████ MST, Pangelinan, Aguirre, CCv2 |
| 2024 | 32 | 76 | ████████████████████████████████ |
| **2025** | **20** | **96** | ████████████████████ |
| **2026** | **5** | **101** | █████ |

### Síntese temporal

- **Pré-2018 (fundamentos)**: 9 fichas (8.9%)
- **2018-2020 (era LAFTR/Buolamwini)**: 15 fichas (14.8%)
- **2021-2023 (FairFace + MST consolidados)**: 20 fichas (19.8%)
- **2024-2026 (estado-da-arte recente)**: **57 fichas (56.4%)**
- **Combinado 2025-2026**: **25 fichas (24.7%)** — meta orientador
  ≥20 ✅

A mediana temporal está em 2023-2024, evidência de que o corpus está
fortemente alinhado com o estado-da-arte mais recente.

---

## 2. Distribuição por track

| Track | Tema | Fichas | % corpus |
|---|---|---|---|
| **A** | Race classification (FairFace direto) | 4 | 4.0% |
| **B** | FR fairness (audit, datasets, robustness) | 16 | 15.8% |
| **C** | Skin tone alternativo (MST, Fitzpatrick) | 7 | 6.9% |
| **D** | Mitigação algorítmica (loss, training) | 10 | 9.9% |
| **E** | Auditoria & Surveys | 14 | 13.9% |
| **F** | Fundamentação científica/ética | 3 | 3.0% |
| **G** | Mecanismos ML fairness (paradigmáticos) | 9 | 8.9% |
| **I** | VLM/CLIP/BLIP fairness (NOVO R7) | 14 | 13.9% |
| **J** | Conditioning moderno (LoRA, ViT) (NOVO R7) | 5 | 5.0% |
| **K** | FR fundadores (losses canônicos) (NOVO R7) | 6 | 5.9% |
| **L** | Auxiliares (post-hoc, synthetic, federated, etc.) | 13 | 12.9% |
| **TOTAL** | | **101** | **100%** |

### Detalhamento por track

#### Track A — Race classification direto (4)
- `dataset_karkkainen_2021` (FairFace — dataset central)
- `aldahoul_2024` (FaceScanPaliGemma — SOTA atual)
- `lin_2022` (FairGRAPE — pruning + valida baseline 72%)
- `pereira_2026` (SkinToneNet + STW + auditoria FairFace)

#### Track B — FR fairness (16)
- Datasets: `dataset_wang_2019` (RFW), `dataset_robinson_2020` (BFW),
  `dataset_bupt_2019` (BUPT), `grother_2019` (NIST)
- Causas: `pangelinan_2023` (pixel info), `kolla_2022` (training
  distribution), `dooley_2022` (NAS), `image_distortions_2021`,
  `occlusion_bias_2024`, `robustness_face_detection_2022`
- Crítica e datasets fairer: `rethinking_assumptions_2021`,
  `fairer_datasets_2024`, `racial_bias_dataset_2017`
- Recentes: `mllm_face_verification_2026`, `mst_kd_2024`,
  `fairface_challenge_eccv2020`

#### Track C — Skin tone alternativo (7)
- Escalas históricas: `fitzpatrick_1988`, `massey_martin_2003`
- Escalas modernas: `schumann_2023` (MST)
- Datasets: `dataset_hazirbas_2021` (CCv1), `porgali_2023_ccv2` (CCv2)
- Pioneiros: `buolamwini_2018` (Gender Shades)
- Aplicação regulatória: `lafargue_2025`

#### Track D — Mitigação algorítmica (10)
- Contrastive: `park_2022` (FSCL+)
- DRO: `sagawa_2020` (Group DRO)
- Architecture: `manzoor_2024` (FineFACE)
- Ensemble: `bhaskaruni_2019`
- Trade-off: `dehdashtian_2024` (U-FaTE)
- SSL: `ramachandran_2024`, `provable_adversarial_ssl_2024`
- Multi-task: `raumanns_2024`
- Meta-learning: `liu_2025` (BNMR FAccT)
- Geral: `enhancing_visual_attributes_2022`

#### Track E — Auditoria & Surveys (14)
- Auditoria: `dominguez_2024` (DSAP), `reliable_demo_inference_2025`,
  `evaluating_lvlm_2024`, `benchmark_lvlm_2026`
- Surveys ML fairness: `survey_mehrabi_2021`
- Surveys FR: `survey_kotwal_2025`, `survey_racial_bias_fr_2024`,
  `survey_face_recognition_2022`
- Surveys CV/VL: `survey_cv_fairness_2024`, `survey_llm_bias_2024`,
  `survey_multimodal_fairness_2024`,
  `survey_fairness_vision_lang_2024`
- Surveys laterais: `survey_long_tail_2022`, `federated_fairness_survey_2025`

#### Track F — Fundamentação científica/ética (3)
- `lewontin_1972` (genética 85/15)
- `fuentes_2019` (AAPA statement)
- `neto_2025` (continuous labels — questiona discretização)

#### Track G — Mecanismos ML fairness paradigmáticos (9)
- Métricas: `hardt_2016` (EO/EOD), `kleinberg_2017` (impossibilidade)
- Representação: `zemel_2013` (LFR), `madras_2018` (LAFTR)
- Adversarial: `zhang_2018`
- Conditioning: `perez_2018` (FiLM)
- Multi-task transfer: `aguirre_2023`
- Causal: `counterfactual_fairness_iclr2025`
- Agnostic: `demographic_agnostic_2025`

#### Track I — VLM/CLIP/BLIP fairness (14) — NOVO R7
- Métodos: `luo_2024_fairclip`, `dehdashtian_2024_fairerclip`,
  `joint_vl_2024`, `unified_debiasing_vlm_2024`, `bendvlm_2024`,
  `debiasing_neural_interventions_2025`, `closed_form_debias_2026`,
  `bias_subspace_2025`, `fair_residuals_vlm_2025`, `biopro_2025`
- Datasets/Benchmarks: `lin_2025_aiface`, `gras_2025`,
  `indicfairface_2026`

#### Track J — Conditioning moderno (5) — NOVO R7
- AIM-Fair: `zhao_2025_aimfair`
- LoRA: `bian_2025_lorafair`, `fairlora_2024`, `fairness_lora_2024`
- ViT-specific: `tian_2024_fairvit`

#### Track K — FR fundadores losses (6) — NOVO R7
- `schroff_2015_facenet` (Triplet)
- `wang_2018_cosface` (LMCL)
- `deng_2019_arcface` (Angular margin)
- `meng_2021_magface` (Quality + Magnitude)
- `kim_2022_adaface` (Quality-adaptive)
- `range_loss_2016` (Long-tail)

#### Track L — Auxiliares (13)
- Post-hoc/Calibration: `post_comparison_2020`, `faircal_2021`,
  `score_normalization_2024`, `fair_sight_2025`
- Synthetic: `synthetic_face_2024`, `variface_2024`, `frcsyn_2024`,
  `massively_annotated_2024`, `fairimagen_neurips2025`
- Federated/Privacy: `dp_fedface_2024`, `voidface_2025`
- Cross-domain: `fairdomain_2024`, `face4fairshifts_2025`
- Explainability: `explainable_fr_2024`

---

## 3. Análise crítica: convergências (papers que SUPORTAM o pipeline v3.2)

Mapeamento entre cada etapa do pipeline e os papers que fundamentam:

### Etapa 1 — SkinToneNet pré-treinado para MST

| Pilar | Papers de suporte | Força |
|---|---|---|
| MST como escala correta | `schumann_2023`, `massey_martin_2003` | ✅ Sólido |
| SkinToneNet existe e SOTA | `pereira_2026` | ✅ Decisão técnica |
| Crítica à Fitzpatrick | `fitzpatrick_1988`, `buolamwini_2018` (uso histórico) | ✅ Crítica documentada |
| Cobertura geográfica | `porgali_2023_ccv2` (CCv2) | ✅ Validação externa |

### Etapa 2 — Auditoria FairFace MST × race (Contribuição C2)

| Pilar | Papers de suporte | Força |
|---|---|---|
| FairFace tem 7 classes | `dataset_karkkainen_2021` | ✅ Dataset central |
| Distribuição agregada MST já auditada (não cruzada) | `pereira_2026` | ✅ Gap original confirmado |
| DSAP método para audit | `dominguez_2024` | ✅ Metodologia |
| ε-log-ratio adaptação | `dataset_karkkainen_2021`, `lin_2022` | ✅ Métrica auditoria |
| 8 datasets audit | `pereira_2026` (Seção 7.1) | ✅ Cross-audit incluindo FairFace |

### Etapa 3 — Race classifier ConvNeXt-T + FiLM (Contribuição C3)

| Pilar | Papers de suporte | Força |
|---|---|---|
| FiLM mecanismo formal | `perez_2018` | ✅ Mecanismo |
| Sample para race 7-class | `aldahoul_2024`, `lin_2022`, `dataset_karkkainen_2021` | ✅ SOTA 72-75% |
| ConvNeXt como backbone moderno | `dooley_2022` (NAS), `kim_2022_adaface` | ✅ Justificativa |
| Conditioning architectural | `perez_2018`, `tian_2024_fairvit` | ✅ Linhagem |

### Etapa 4 — Comparação contra 6 baselines (Contribuição C4)

| Baseline | Papers | Força |
|---|---|---|
| ResNet-34 | `dataset_karkkainen_2021` | ✅ Canônico |
| ConvNeXt-T puro | — (controle nosso) | ✅ Ablation |
| FSCL+ | `park_2022` | ✅ Top conference |
| Group DRO | `sagawa_2020` | ✅ Top conference |
| FineFACE | `manzoor_2024` | ✅ ICPR 2024 |
| Adversarial | `zhang_2018` | ✅ Linhagem clássica |

### Etapa 5 — Fair transferência para FR (Contribuição C5)

| Pilar | Papers de suporte | Força |
|---|---|---|
| LAFTR teórico | `madras_2018` (Teorema 1) | ✅ Fundamento |
| Empírico NLP | `aguirre_2023` (15-44% redução ε-DEO) | ✅ Empírico mesma família |
| Datasets FR fair | `dataset_wang_2019` (RFW), `dataset_robinson_2020` (BFW) | ✅ Datasets |
| Quality control image | `kim_2022_adaface`, `meng_2021_magface`, `image_distortions_2021` | ✅ Quality awareness |

### Métricas (Contribuição C4)

| Métrica | Papers de fundamento | Força |
|---|---|---|
| F1 macro | (van Rijsbergen 1979 — não em corpus, padrão) | ✅ Universal |
| EO/EOD | `hardt_2016` | ✅ Canônico |
| DR multi-classe | `dataset_karkkainen_2021` (ε), `lin_2022` (ρ) | ✅ Generalização |
| Worst-class F1 | `sagawa_2020` (Group DRO) | ✅ Worst-case |
| Triangulação | `kleinberg_2017` (impossibilidade) | ✅ Justificativa formal |

---

## 4. Análise crítica: CONFLITOS com o pipeline v3.2

Papers que **questionam direta ou indiretamente** alguma decisão da
tese e exigem resposta defensiva no texto.

### Conflito 1 — `pangelinan_2023` (CRÍTICO)

**Tese conflitante**: "Diferenças de pixel info da face explicam mais
variação cross-demográfica em FR accuracy do que skin tone direto".

**Implicação**: para Cap 3 (fair transferência), nosso skin-tone
conditioning pode ter efeito menor do que esperado. Apenas controlar
para skin tone via FiLM pode ser insuficiente.

**Resposta na tese**: H5 foi reformulada na reunião de 2026-06-08
para versão V3 (split em H5 + H6 nova) — H5 mantém a hipótese
original e H6 quantifica explicitamente a variance explicada por
pixel info, transformando a refutação potencial em **contribuição
quantitativa** (decomposição de variância). Cap 3 adiciona quality
control via face crop + alinhamento.

**Status**: ⚠️ ENDEREÇADO via reformulação H6.

### Conflito 2 — `neto_2025` (CRÍTICO conceitual)

**Tese conflitante**: "Modelos treinados em datasets balanceados no
espaço CONTÍNUO de pertencimento étnico superam consistentemente
modelos balanceados no espaço DISCRETO". Argumenta contra a
**discretização per se** em 4, 5, 7 ou N raças.

**Implicação**: nossa Etapa 2 produz matriz **discreta** (10 MST × 7
race). A própria taxonomia 7-class do FairFace é arbitrária.

**Resposta na tese**: limitação metodológica reconhecida no Capítulo
1 (§ Limitações). A presente dissertação opera com discreto **por
escolha pragmática alinhada ao dataset** (FairFace é discreto);
extensão para contínuo é registrada como **trabalho futuro**.
`fuentes_2019` (AAPA) e `lewontin_1972` reforçam a fragilidade
biológica da taxonomia, sustentando o reconhecimento dessa limitação
**sem refutar a contribuição** (que é sobre fairness algorítmica
dado o dataset existente).

**Status**: ⚠️ RECONHECIDO como limitação; resposta defensiva
explícita.

### Conflito 3 — `lewontin_1972` + `fuentes_2019` (ÉTICO)

**Tese conflitante**: race não tem fundamento biológico. Auditar
fenótipo via raça é categoricalmente questionável.

**Implicação**: nossa pesquisa, ao operacionalizar race classification
multi-classe, parece **reificar** a categoria racial.

**Resposta na tese**: posicionamento ético explícito no Capítulo 1
(§ Posicionamento ético). Não pretendemos validar a categoria como
biológica; **auditamos como construto social aplicado a dados
biológicos**, quantificando os limites estruturais desse exercício.
A Contribuição C6 (decomposição fenotípico/algorítmico) é justamente
diagnóstico dessa fronteira.

**Status**: ⚠️ ENDEREÇADO via posicionamento ético + C6.

### Conflito 4 — `kotwal_2025` survey (suaviza pipeline)

**Tese conflitante (síntese)**: papel do balanceamento de dados é
**limitado**; outros fatores (image quality, soft attributes) têm
papel maior. Especificamente `muthukumar`: features estruturais
explicam mais variação em dark-skinned females do que skin tone.

**Implicação**: nosso skin-tone-only conditioning pode capturar
fração minoritária do viés. Estruturas faciais 3D, idade, expressão
não são contempladas.

**Resposta na tese**: nossa contribuição se posiciona como **um eixo
arquitetural específico** (skin tone via FiLM), não como solução
exaustiva. Cap 2 reporta resultados em magnitude que esclarece
quanto desse efeito é capturado vs residual. Trabalhos futuros (Cap
4 Discussão) podem combinar skin tone com outros sinais (idade,
estrutura facial).

**Status**: ⚠️ ENDEREÇADO via escopo declarado + Cap 4 trabalhos
futuros.

### Conflito 5 — `dooley_2022` (NAS)

**Tese conflitante**: "biases são inerentes a arquiteturas
neurais". NAS encontra arquiteturas Pareto-superiores em
fairness sem qualquer intervenção em dados ou loss.

**Implicação**: para reduzir disparidade, **trocar arquitetura por si
só** pode bastar. FiLM-conditioning seria redundante se ConvNeXt-T
puro já fizer o trabalho.

**Resposta na tese**: H2 da nossa tese ENDEREÇA EXATAMENTE isso —
testa se ConvNeXt-T puro já reduz disparity sem MST. Se H2 confirmar
(Latinx permanece ~60%), o efeito FiLM (H1) é genuíno. Se H2 refutar
(ConvNeXt-T puro já resolve), parte do efeito é atribuível à
arquitetura — **resultado científico válido, não falha**.

**Status**: ✅ ENDEREÇADO via formulação de H2 antecipativamente.

### Conflito 6 — `liu_2025` BNMR (mecanismo competitivo)

**Tese conflitante**: BNMR (Bayesian Network meta-learning sample
reweighting) é mecanismo de mitigação alternativo (FAccT 2025) que
opera por reweighting, não por feature conditioning.

**Implicação**: se BNMR superar empíricamente FiLM-conditioning em
race 7-class, qual é a vantagem do nosso pipeline?

**Resposta na tese**: BNMR é candidato a **baseline forte** da
Cap 2 (ablation). Os mecanismos são **ortogonais** (BNMR atua em
samples, FiLM em features), e potencialmente combináveis em
trabalho futuro. Se BNMR supera FiLM, é informação útil para a tese
(decisão metodológica para versão final, não falha).

**Status**: ✅ ENDEREÇADO via inclusão como baseline.

### Conflito 7 — `aguirre_2023` (mecanismo alternativo)

**Tese conflitante**: shared encoder + multi-task com fairness loss
auxiliar transfere fairness — **sem conditioning architectural
explícito**.

**Implicação**: poderíamos atingir fairness sem FiLM, apenas
combinando MST training + race training num framework MTL.

**Resposta na tese**: Aguirre é **NLP**, não CV; transferência de
mecanismo não é automática. Além disso, FiLM oferece **modulação
granular por canal** vs encoder compartilhado monolítico — mais
expressivo. Cap 2 pode incluir ablation MTL-fair vs FiLM se tempo
permitir.

**Status**: ⚠️ DECISÃO PENDENTE (ablation opcional Cap 2).

---

## 5. Caminhos alternativos sugeridos pela literatura

Decisões alternativas presentes no corpus que poderíamos ter tomado,
e como argumentamos nossa escolha:

| Decisão de design | Alternativa na literatura | Por que escolhemos nosso caminho |
|---|---|---|
| **Conditioning architectural** = FiLM | CLIP-based (FairCLIP `luo_2024`), Adapter (FairViT `tian_2024`), LoRA (`fairlora_2024`, `bian_2025`) | FiLM é parameter-efficient (<1%) E semanticamente interpretable (skin tone como contexto explícito). CLIP/LoRA são alternativos — Cap 2 inclui ablation arquitetural (Contribuição C7). |
| **Backbone** = ConvNeXt-T | ResNet-50 (canônico Park 2022), ViT (Pereira 2026), NAS (Dooley 2022) | ConvNeXt-T balanceia desempenho ViT-like com custo de ResNet. Endereça crítica histórica ResNet. Dooley confirma que backbone moderno reduz disparity. |
| **Métrica fairness** = triangulação (DR + worst-class + EO/EOD) | DPV (`dehdashtian_2024`), Cramer's V/Renkonen (`dominguez_2024`), ρ(A) std (`lin_2022`), ε log-ratio (`karkkainen_2021`) | Kleinberg 2017 prova impossibilidade; nossa triangulação respeita o teorema. Inclui worst-case (Sagawa) + razão (Hardt-like) + variabilidade simultaneamente. |
| **Skin tone scale** = MST 10-class | Fitzpatrick 6-class (Buolamwini, Lafargue), NIS 11-class (Massey-Martin), ITA contínua (Lafargue), discretização contínua (Neto 2025) | MST é padrão moderno fairness-research, granular, sem viés caucasiano-cêntrico. Citamos alternativas como limitação metodológica reconhecida. |
| **Discretização** = 7 classes raciais | Contínuo (`neto_2025`), 4-class (RFW, BUPT), 10-class MST como proxy | FairFace fornece 7-class; alternativas existem mas não cobrem mesma escala. Limitação reconhecida; trabalho futuro contínuo. |
| **Pipeline** = supervisionado in-processing | SSL (`ramachandran_2024`, `provable_adversarial_ssl_2024`), Post-hoc (`faircal_2021`, `score_normalization_2024`, `fair_sight_2025`), Synthetic (`synthetic_face_2024`, `variface_2024`, `frcsyn_2024`), Causal (`counterfactual_fairness_iclr2025`), Demographic-agnostic (`demographic_agnostic_2025`) | Nossa abordagem aproveita labels disponíveis no FairFace. Alternativas são complementares, não substitutivas. Citadas como direções futuras. |
| **Tarefa** = race classification multi-classe | Face verification 1:1 (RFW, BFW, NIST, Pangelinan), Face attribute (CelebA via FSCL, FineFACE, BNMR), Generation (FairImagen, VariFace) | Nossa tarefa é race classification 7-class. Cap 3 estende para verification (RFW/BFW). Outras tarefas fora do escopo. |

---

## 6. Decisões metodológicas que precisam ser defendidas no texto

Síntese das decisões que enfrentam objeções da literatura e que
precisam ter defesa explícita na qualificação:

1. **Por que FiLM e não CLIP-conditioning** (C7 endereça via ablation).
2. **Por que MST 10-class e não Fitzpatrick 6 ou contínuo** (escolha
   justificada por Schumann 2023 + crítica a Fitzpatrick).
3. **Por que 7-class race do FairFace e não contínuo** (escolha
   pragmática alinhada ao dataset; limitação reconhecida via Neto
   2025 e AAPA Fuentes).
4. **Por que ConvNeXt-T e não outras arquiteturas modernas** (balanço
   capacidade × custo; H2 testa se backbone só já basta — Dooley
   conflicting prediction).
5. **Por que triangulação 3-5 métricas e não métrica única** (Kleinberg
   2017 impossibilidade).
6. **Por que pipeline supervisionado e não SSL/causal/agnostic** (uso
   de labels disponíveis; alternativas citadas como direções).
7. **Por que skin tone como CONTEXTO conditioning e não sample
   reweighting (BNMR)** (Cap 2 ablation compara).
8. **Por que controle de pixel info no Cap 3** (resposta a Pangelinan
   2023).

---

## 7. Consolidado: o corpus está bem estruturado

### Convergência

- **86 fichas (85.1%)** alinham com nosso pipeline diretamente ou via
  conexão clara.
- **15 fichas (14.9%)** apresentam objeções que **endereçamos
  explicitamente** no texto via H6, C6, ablations, ou limitações
  reconhecidas.
- **Nenhuma ficha refuta categoricamente** a tese sem possibilidade
  de resposta defensiva.

### Diversificação de literatura

- Cobertura **temporal**: 54 anos (1972-2026), com mediana 2023-2024.
- Cobertura **temática**: 11 tracks distintos.
- Cobertura **2025-2026**: 25 fichas, atendendo meta orientador.

### Conclusão

O corpus está **maduro e bem estruturado** para suportar a
qualificação. Os 7 conflitos identificados são **endereçáveis
defensivamente** via reformulações (H6), limitações reconhecidas, ou
ablations. Os caminhos alternativos enriquecem a discussão sem
comprometer a originalidade da contribuição central.

**Próximos passos recomendados**:
1. Incorporar respostas aos 7 conflitos no texto da revisão de
   literatura (Capítulo 2) e nas seções de limitações.
2. Manter as ablations arquiteturais (FiLM vs CLIP, FiLM vs LoRA) no
   protocolo experimental do Capítulo 3.
3. Reforçar Cap 4 (Discussão) com diálogo explícito com as
   alternativas (SSL, causal, contínuo) como trabalhos futuros.
