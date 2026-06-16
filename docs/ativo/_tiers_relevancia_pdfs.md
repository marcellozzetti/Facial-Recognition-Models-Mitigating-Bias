---
data: 2026-06-16
tipo: organizacao-pdfs
escopo: classificacao das 100 fichas com PDF em 4 tiers de relevancia
status: ativo
proposito: importacao em NotebookLM (Google AI Plus 400 GB) ou outras ferramentas
---

# Tiers de relevância dos PDFs — para importação em NotebookLM

> **Classificação das 97 fichas com PDF em 4 níveis de relevância**
> baseada no pente fino crítico, cross-reference tese × corpus e
> auditoria de qualidade. PDFs físicos copiados para subpastas em
> `pdfs_por_tier/` (gitignored — pasta de trabalho local).

## Limites do NotebookLM por plano

| Plano | Fontes/notebook | Notebooks | Custo |
|---|---|---|---|
| Free | até 50-100 | até 100 | gratuito |
| **Google AI Plus** (seu) | **até ~300** (verificar atual) | mais notebooks | incluído no plano com 400 GB |
| Google AI Pro / Ultra | até 300+ | muitos | tier mais alto |

> Para confirmar o limite exato do seu plano, ver
> https://support.google.com/notebooklm/answer/15724963 (verificar atualizado).

## Estratégia de importação recomendada

### 🎯 Opção A — Notebook único (recomendado se plano permite ≥100 fontes)
Importar **Tier 1 + Tier 2 + Tier 3 = 81 PDFs** em um único notebook.
- Cobertura: forte favorável + favorável + caminhos alternativos (VLM/CLIP).
- Suficiente para responder qualquer pergunta sobre a tese.

### 🎯 Opção B — 2 notebooks temáticos (se limite for 50)
- **Notebook 1 — Núcleo**: Tier 1 + parte do Tier 2 (datasets + métricas + FR fundadores) = ~30 PDFs
- **Notebook 2 — VLM/CLIP e mitigação**: Track I + Track J + baselines mitigação = ~30 PDFs

### 🎯 Opção C — Notebook por capítulo (se quiser usar para escrever)
- **Notebook Cap 1**: Tier 1 + datasets + skin tone (~25 PDFs)
- **Notebook Cap 2**: Track I + Track J + baselines mitigação (~30 PDFs)
- **Notebook Cap 4**: Tier 1 + mecanismos + métricas (~20 PDFs)

---

## Tier 1 — Crítico (14 PDFs)

**Camada 1 do pente fino**: 12 forte favorável + 2 conflito forte.
**Leitura integral verificada** (fichas em Nível A da auditoria).
**Imprescindíveis** — sem esses 14, a tese não se sustenta.

### Forte favorável (12)
1. `pereira_2026` — **SkinToneNet** (Etapa 1 do pipeline)
2. `dataset_karkkainen_2021` — **FairFace** (dataset central)
3. `aldahoul_2024` — **FaceScanPaliGemma** (SOTA atual 7-class)
4. `schumann_2023` — **Monk Skin Tone** (escala padrão)
5. `perez_2018` — **FiLM** (mecanismo central da tese)
6. `hardt_2016` — **EO/EOD** (métricas canônicas)
7. `kleinberg_2017` — **Impossibility theorem**
8. `madras_2018` — **LAFTR** (fair transferência)
9. `sagawa_2020` — **Group DRO** (baseline)
10. `park_2022` — **FSCL+** (baseline contrastive)
11. `lin_2022` — **FairGRAPE** (validação baseline)
12. `luo_2024_fairclip` — **FairCLIP** (C7 baseline principal)

### Conflito forte (2)
13. `pangelinan_2023` — **Refutação central** (motiva H6/OE-6)
14. `neto_2025` — **Continuous labels** (limitação reconhecida)

---

## Tier 2 — Alto (34 PDFs)

**Camada 2 favorável + marcos fundadores do campo**. Apoio direto à
tese. Importante para Cap 1 (introdução) e Cap 2 (revisão).

### Marcos fundadores (7)
- `buolamwini_2018` — Gender Shades (marco fundador do campo)
- `fuentes_2019` — AAPA Statement (race construto)
- `lewontin_1972` — Apportionment 85/6/8 ⚠️ **PDF não disponível**
- `grother_2019` — NIST FRVT (auditoria industrial)
- `zemel_2013` — LFR (Test-of-Time Award)
- `zhang_2018` — Adversarial debiasing (baseline)
- `aguirre_2023` — Multi-task fair empírico NLP

### Datasets adicionais (6)
- `dataset_wang_2019` — RFW
- `dataset_robinson_2020` — BFW
- `dataset_bupt_2019` — BUPT/MBN (precedente skin tone)
- `dataset_hazirbas_2021` — CCv1
- `porgali_2023_ccv2` — CCv2 Meta
- `fitzpatrick_1988` — FST histórico ⚠️ **PDF não disponível**

### Baselines de mitigação (4)
- `manzoor_2024` — FineFACE
- `bhaskaruni_2019` — Ensemble fair
- `dehdashtian_2024` — U-FaTE
- `liu_2025` — BNMR FAccT

### FR fundadores (5)
- `schroff_2015_facenet` — FaceNet (triplet)
- `wang_2018_cosface` — CosFace
- `deng_2019_arcface` — ArcFace (canônico)
- `meng_2021_magface` — MagFace
- `kim_2022_adaface` — AdaFace

### Conflitos moderados (5)
- `dooley_2022` — NAS arquitetural
- `kolla_2022` — 16 experimentos distribuição racial
- `rethinking_assumptions_2021` — Refutação balanceamento
- `image_distortions_2021` — Confounder distortions
- `occlusion_bias_2024` — Confounder occlusions

### Surveys top (4)
- `survey_kotwal_2025` — FR fairness específico
- `survey_mehrabi_2021` — ML fairness canônico
- `survey_racial_bias_fr_2024` — Durham (ACM CSur)
- `survey_fairness_vision_lang_2024` — PUCRS Brasil

### VLM principais (2)
- `dehdashtian_2024_fairerclip` — FairerCLIP ICLR
- `bendvlm_2024` — BendVLM test-time

### Regulatório (1)
- `lafargue_2025` — EU AI Act

---

## Tier 3 — Médio (33 PDFs)

**Caminhos alternativos + cobertura ampla**. Útil para Cap 2 (revisão
completa por 11 tracks) e defesa contra perguntas da banca.

### VLM/CLIP Track I (10)
- `joint_vl_2024`, `unified_debiasing_vlm_2024`, `closed_form_debias_2026`
- `bias_subspace_2025`, `fair_residuals_vlm_2025`, `biopro_2025`
- `lin_2025_aiface`, `gras_2025`, `indicfairface_2026`
- `mllm_face_verification_2026`, `evaluating_lvlm_2024`, `benchmark_lvlm_2026`

### Conditioning moderno Track J (5)
- `tian_2024_fairvit`, `bian_2025_lorafair`, `fairlora_2024`
- `fairness_lora_2024`, `zhao_2025_aimfair`

### Mitigação adicional (4)
- `ramachandran_2024`, `provable_adversarial_ssl_2024`
- `raumanns_2024`, `enhancing_visual_attributes_2022`

### Auditoria e dataset audit (3)
- `dominguez_2024` — DSAP
- `reliable_demo_inference_2025` — DAI
- `robustness_face_detection_2022` — Dooley parallel

### Surveys cobertura (4)
- `survey_cv_fairness_2024`, `survey_multimodal_fairness_2024`
- `survey_llm_bias_2024`, `survey_face_recognition_2022`

### Causal/agnostic (2)
- `counterfactual_fairness_iclr2025`, `demographic_agnostic_2025`

### Skin tone classifier (2)
- `mst_kd_2024` — MST-KD (Univ. Porto)
- `fairface_challenge_eccv2020` — Challenge histórico

### FR fundador adicional (1)
- `range_loss_2016` — Range Loss long-tail

---

## Tier 4 — Baixo (19 PDFs)

**Track L auxiliar + neutras contextuais**. Cobertura para banca,
não-essencial para argumentação central.

### Synthetic data (6)
- `synthetic_face_2024`, `variface_2024`, `frcsyn_2024`
- `massively_annotated_2024` (Univ. Porto), `fairimagen_neurips2025`
- `fairer_datasets_2024`

### Federated/privacy (3)
- `dp_fedface_2024`, `voidface_2025` (Univ. Coimbra)
- `federated_fairness_survey_2025`

### Cross-domain (2)
- `fairdomain_2024`, `face4fairshifts_2025`

### Explainability (1)
- `explainable_fr_2024`

### Post-hoc/calibration (4)
- `post_comparison_2020`, `faircal_2021`
- `score_normalization_2024`, `fair_sight_2025`

### Histórico social (1)
- `massey_martin_2003` — NIS Scale ⚠️ **PDF não disponível**

### Contextuais (2)
- `racial_bias_dataset_2017` — Hong Kohno Morgenstern EAAMO
- `survey_long_tail_2022` — BUPT IJCV (Gini coefficient)

---

## Como importar no NotebookLM

### Passo 1 — Gerar a pasta organizada localmente
```bash
python docs/ativo/_organizar_pdfs_por_tier.py
```
Isso cria `docs/ativo/04_pesquisa_bibliografica/pdfs_por_tier/` com
4 subpastas (tier_1_critico, tier_2_alto, tier_3_medio, tier_4_baixo).

### Passo 2 — Abrir o NotebookLM
1. Acessar https://notebooklm.google.com
2. Criar novo notebook (e.g., "Tese Marcello — Bibliografia")
3. Clicar em "Add sources" / "Adicionar fontes"

### Passo 3 — Importar por tier
Dependendo do limite do seu plano:

**Se ≥100 fontes** (provável Plus):
- Selecionar tudo de `tier_1_critico/` (14)
- + tudo de `tier_2_alto/` (34) → totaliza 48
- + selecionar de `tier_3_medio/` quanto couber → até 81 total

**Se 50 fontes**:
- Selecionar tudo de `tier_1_critico/` (14)
- + tudo de `tier_2_alto/` (34) → totaliza 48 ✓

### Passo 4 — Configurar prompts úteis no NotebookLM
Sugestões de prompts (após upload):
- *"Qual é o estado da arte atual em classificação racial 7-class sobre FairFace?"*
- *"Como Pangelinan et al. (2023) refutam skin tone como causa única do gap demográfico?"*
- *"Liste todos os mecanismos de conditioning testados em fairness vision-language."*
- *"Qual o gap específico que esta tese endereça vs Pereira et al. 2026 (SkinToneNet)?"*

---

## Observações

- **3 PDFs ausentes**: `lewontin_1972`, `fitzpatrick_1988`, `massey_martin_2003`. São fontes históricas (1972, 1988, 2003) sem PDF acessível. Conteúdo central já está documentado nas fichas correspondentes; pode-se contornar copiando as `.md` para o NotebookLM como fontes textuais.
- **Pasta `pdfs_por_tier/` é gitignored** — não vai para o repositório (são duplicatas dos PDFs originais).
- **PDFs originais** continuam em `docs/ativo/04_pesquisa_bibliografica/pdfs/` (versionados).
- **Para regenerar**: basta rodar `python docs/ativo/_organizar_pdfs_por_tier.py` (clean rebuild).
