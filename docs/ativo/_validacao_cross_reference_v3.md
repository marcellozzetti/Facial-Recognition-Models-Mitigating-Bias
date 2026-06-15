---
data: 2026-06-15
tipo: validacao-cross-reference
versao: v3.4 (post-revisao-completa)
escopo: cruzamento sistemático tese × pipeline × premissas × 101 fichas
status: ativo
---

# Validação cross-reference — tese × pipeline × 101 fichas

> **Auditoria sistemática pós-revisão bibliográfica completa.** Cada
> elemento da tese (objetivo geral, OEs, pipeline, contribuições,
> hipóteses) é cruzado contra as fichas do corpus para verificar
> embasamento, conflitos e gaps remanescentes.
>
> **Premissa**: 100 fichas VERIFIED + 1 OVERVIEW_ONLY (Springer
> paywall) — corpus blindado. Bibliografia consolidada em
> `docs/tese/referencias.bib`.

## Sumário executivo

| Elemento da tese | Embasamento | Conflitos | Recomendação |
|---|---|---|---|
| Tese central (skin tone como sinal arquitetural para race classification multi-classe) | ✅ Sólido | Nenhum sem resposta | **Manter como está** |
| Pipeline 6 etapas | ✅ Aprovado pelo orientador (08/jun) | — | Manter; ajustar Cap 2 com 4 configs |
| 7 contribuições (C1-C7) | ✅ C1-C6 sólidas; C7 reforçado por leituras | — | **Promover H6→OE-6 (decisão pendente)** |
| 6 hipóteses (H1-H6) | ✅ Todas fundamentadas | H6 endossada por Pangelinan VERIFIED | **Quantificar H6 explicitamente** |
| Storytelling 6 partes | ✅ Aprovado | — | Manter |
| **Decisão arquitetural FiLM** | ✅ Justificada vs 8 alternativas | Nenhuma fatal | Adicionar ablação **Gated FiLM** |

**Veredito geral**: corpus blindado, tese consistente, **3 ajustes táticos** recomendados (não estruturais).

---

## 1. Tese central — embasamento

> *"Esta dissertação tem como objetivo desenvolver e avaliar um
> pipeline de classificação racial em imagens faciais que incorpora
> tom de pele (MST) como sinal auxiliar condicionante via mecanismo
> arquitetural, para mitigar viés racial."*

### 1.1 Fundamentação (fichas que sustentam diretamente)

| Premissa | Fichas que fundamentam | Evidência específica |
|---|---|---|
| **Raça é construto social, não biológico** | `fuentes_2019` (AAPA), `lewontin_1972` (85/6/8), `neto_2025` | Statement oficial + apportionment + continuous labels |
| **Skin tone é label mais preciso/científico que raça** | `wang+deng_2022` (BUPT/MBN), `schumann_2023` (MST), `pereira_2026` (SkinToneNet) | Wang explicitamente argumenta race labels "instáveis" — opta por FST+ITA |
| **Disparidade racial documentada em SOTA** | `aldahoul_2024` (FaceScanPaliGemma 7-class), `buolamwini_2018` (Gender Shades), `grother_2019` (NIST FRVT) | Latinx F1 60%, Black F1 90% em SOTA atual |
| **Balanceamento não basta** | `kolla_2022`, `rethinking_assumptions_2021` (Hanson), `pangelinan_2023`, `dataset_karkkainen_2021` (FairFace já balanceado mas gap persiste) | Empírico: skewed > uniforme em alguns casos |
| **Mecanismo arquitetural justificado** | `perez_2018` (FiLM), `madras_2018` (LAFTR), `aguirre_2023` (multi-task fair) | FiLM como conditioning + LAFTR transferência + Aguirre empírico |

### 1.2 Conflitos potenciais → todos endereçados

| Conflito potencial | Ficha | Resposta defensiva mapeada |
|---|---|---|
| "Skin tone não é causa direta do gap" | `pangelinan_2023` (Camada 1 VERIFIED) | **H5 reformulada + H6 nova** (decomposição variância pixel info × skin tone) |
| "Balanceamento desbalanceado supera balanceado" | `rethinking_assumptions_2021`, `kolla_2022` | **Suporta a tese**: justifica mitigação algorítmica além de balanceamento de dados |
| "Discretização racial é o problema" | `neto_2025` | **Limitação reconhecida** no Cap 1 + parte do framing ético |

### 1.3 Conclusão da seção 1

✅ **Tese central tem fundamentação direta em ≥12 fichas, sem conflitos não endereçados.** Manter como está.

---

## 2. Pipeline 6 etapas — cross-reference

### Pipeline aprovado pelo orientador (reunião 2026-06-08)

```
Etapa 1: SkinToneNet (pré-treinado, Pereira 2026) → vetor MST 10-dim
Etapa 2: Auditoria FairFace MST × race (matriz pública C2)
Etapa 3: ConvNeXt-T + FiLM(MST) → race classifier 7-class
Etapa 4: Triangulação de métricas (DR + worst-class F1 + EO_h/EOD)
Etapa 5: Fair transferência → face recognition (RFW/BFW)
Etapa 6: Decomposição variância (fenotípico × algorítmico)
```

### Embasamento etapa por etapa

| Etapa | Fundamentação | Fichas | Status |
|---|---|---|---|
| **1. SkinToneNet pré-treinado** | Insumo direto, lido integralmente | `pereira_2026` (VERIFIED, leitura HTML+PDF) | ✅ Sólido |
| **2. Auditoria FairFace MST × race** | Matriz nova (C2) sem precedente direto | `pereira_2026` audita 8 datasets mas NÃO cross-tabula MST×race; `dominguez_2024` (DSAP framework auditoria) | ✅ Gap confirmado |
| **3. ConvNeXt-T + FiLM** | Mecanismo central da tese | `perez_2018` (FiLM original), `dooley_2022` (NAS — arquitetura importa), `wang+deng_2022` (BUPT/MBN — adaptive margins paralelo) | ✅ Justificado |
| **4. Triangulação métricas** | Defesa contra single-metric criticism | `hardt_2016` (EO/EOD), `kleinberg_2017` (impossibility — triangulação necessária), `dominguez_2024` (DSAP) | ✅ Robusto |
| **5. Fair transferência → FR** | Demonstração empírica inédita em CV | `madras_2018` (LAFTR — teórico), `aguirre_2023` (empírico NLP), `dataset_wang_2019` RFW, `dataset_robinson_2020` BFW | ✅ Gap CV confirmado |
| **6. Decomposição variância** | Diagnóstico inédito | `pangelinan_2023` (motivação H6), `kolla_2022` ("racial gradation"), `image_distortions_2021`, `occlusion_bias_2024` (confounders) | ✅ Inédito |

### 2.1 Decisão arquitetural pós-avaliação técnica

**Após avaliação crítica de 8 mecanismos candidatos** (registrada em `_decisao_arquitetural_film.md`):

| Mecanismo | Aplicabilidade | Decisão |
|---|---|---|
| **FiLM standard** (Perez 2018) | Sweet spot para sinal 10-dim, ~1% params | ✅ **LINHA PRINCIPAL (Config B)** |
| **Gated FiLM** (variante não-linear) | Testa expressividade | ✅ **ABLAÇÃO (Config C — Cap 2)** |
| **CLIP-conditioning** (Radford 2021) | Sinal alternativo via embedding | ✅ **AVALIAÇÃO ALTERNATIVA (Config D — orientador)** |
| Cross-attention | Overkill para 10-dim | ❌ Descartado |
| AdaIN | Voltado para style transfer | ❌ Descartado |
| SPADE | Requer mapa denso | ❌ Não aplicável |
| HyperNetworks | Instabilidade | ❌ Descartado |
| LoRA / Adapter | Categoria diferente (fine-tuning) | ❌ Não é conditioning |
| Concat / CBN | Casos particulares ou ineficientes | ❌ Descartado |

**Estudo comparativo final (Cap 2)**: 4 configurações (A baseline / B FiLM-MST / C Gated FiLM / D FiLM-CLIP).

---

## 3. Contribuições (C1-C7) — cross-reference

| ID | Contribuição | Fichas que fundamentam o gap | Originalidade verificada? |
|---|---|---|---|
| **C1** | Avaliação metodológica de modelos pré-treinados MST + protocolo de escolha | `pereira_2026` (SkinToneNet), `schumann_2023` (MST), `porgali_2023_ccv2` (CCv2), `wang+deng_2022` (IDS BUPT) | ✅ Sem precedente unificado |
| **C2** | Matriz pública **MST × race** do FairFace | `pereira_2026` (audita 8 datasets mas não cross-tabula), `dominguez_2024` (DSAP) | ✅ Inédito |
| **C3** | Primeira aplicação de **FiLM-conditioning** a fairness facial | `perez_2018` (FiLM não testou fairness em §11 Future Work), busca extensa confirma lacuna | ✅ Inédito |
| **C4** | Triangulação multi-classe (DR + worst-class F1 + EO_h/EOD por classe) | `hardt_2016` (canônico), `kleinberg_2017` (impossibility), `dominguez_2024` (DSAP) | ✅ Original em race 7-class |
| **C5** | Demonstração empírica de **fair transferência** classification → FR | `madras_2018` (LAFTR teórico), `aguirre_2023` (empírico mas NLP) | ✅ Inédito em CV |
| **C6** | Decomposição variância fenotípico × algorítmico | `pangelinan_2023` (motiva H6), `kolla_2022` ("racial gradation"), `image_distortions_2021` | ✅ Diagnóstico inédito |
| **C7** | Comparativo FiLM × CLIP-conditioning para race fairness | `luo_2024_fairclip` (FairCLIP), `dehdashtian_2024_fairerclip` (FairerCLIP), `bendvlm_2024`, `bian_2025_lorafair`, `tian_2024_fairvit` | ✅ Inédito em race 7-class FairFace |

### 3.1 Recomendação: promover H6 → OE-6 (decisão pendente)

> **Após Pangelinan VERIFIED + Wang+Deng BUPT 2022 VERIFIED**, fica
> claro que **pixel info como confounder** é tema central da
> literatura recente. Promover H6 de "hipótese auxiliar" para
> **Objetivo Específico OE-6 formal** com decomposição quantitativa
> de variância. Decisão aguarda orientador (item na pauta).

---

## 4. Hipóteses H1-H6 — cross-reference

| ID | Hipótese | Fichas que sustentam | Fichas em conflito | Status |
|---|---|---|---|---|
| **H1** | Pipeline MST+FiLM supera ResNet-34 baseline | `perez_2018`, `aguirre_2023`, `madras_2018`, `aldahoul_2024` (SOTA) | Nenhum direto | ✅ Testável |
| **H2** | ConvNeXt-T puro ganha 2-5pp sobre ResNet-34 | `dooley_2022` (NAS — arquiteturas Pareto-superiores), `manzoor_2024` (FineFACE — arquitetural) | Nenhum | ✅ Testável |
| **H3** | Matriz MST×race mostra Latinx spread ≥5 das 10 MST | `aldahoul_2024` (Latinx F1=60% sugere ambiguidade), `pereira_2026` (subrepresentação MST 6-10) | Nenhum | ✅ Testável (auditoria empírica) |
| **H4** | ≥50% dos erros Latinx em zonas MST de overlap | Derivado das fichas acima | Nenhum | ✅ Testável |
| **H5** (revisada) | Conditioning MST melhora fairness em FR **com pixel info controlada** | `aguirre_2023`, `madras_2018`, [[pereira_2026]] | `pangelinan_2023` (parcial — endereçada via H6) | ✅ Reformulada |
| **H6** (nova) | Disparity residual explicada predominantemente por pixel info | `pangelinan_2023` (refutação central), `image_distortions_2021`, `occlusion_bias_2024`, `kolla_2022` | Nenhum | ✅ **Endossada por leitura integral Pangelinan** |

### 4.1 Achado novo do corpus (não estava no escopo original)

> **Pangelinan reporta** (Fig 26 do paper) que **female impostors clusterizam em distâncias menores que male impostors** — "images of two different females are inherently more similar than images of two different males". O FMR gap persiste após equalização de pixels.
>
> **Implicação para nossa tese**: pode haver **análogo intra-racial** — imagens de duas pessoas diferentes da mesma raça podem ser intrinsecamente mais similares que entre raças. Isso seria **achado novo da nossa C7** se aparecer no estudo comparativo. **Não muda design, muda interpretação**.

---

## 5. Storytelling 6 partes — cross-reference

### 5.1 Contexto

| Asserção | Fichas que sustentam |
|---|---|
| FR em uso massivo (catraca, banco, fronteira) | `survey_kotwal_2025`, `survey_face_recognition_2022` (OPPO — perspectiva industrial), `lafargue_2025` (EU AI Act regulação) |
| Falhas desproporcionais em grupos sub-representados desde 2018 | `buolamwini_2018` (Gender Shades), `grother_2019` (NIST), `survey_racial_bias_fr_2024` (Yucer — Durham) |

### 5.2 Problemas existentes

| Asserção | Fichas que sustentam |
|---|---|
| Disparidade severa: F1 Black 90% vs Latinx 60% | `aldahoul_2024` (FaceScanPaliGemma Tabela 16) |
| Balanceamento não basta | `kolla_2022`, `rethinking_assumptions_2021`, `pangelinan_2023`, `dataset_karkkainen_2021` |
| Mitigação algorítmica atual não foi testada em 7-class | `park_2022` (FSCL+ — 2-class), `sagawa_2020`, `manzoor_2024` (gender 2-class) |

### 5.3 Estado da arte (o que tem sido feito)

| Linha de pesquisa | Fichas representativas |
|---|---|
| Datasets balanceados | `dataset_karkkainen_2021` (FairFace), `dataset_wang_2019` (RFW), `dataset_robinson_2020` (BFW), `wang+deng_2022` (BUPT) |
| Skin tone como dimensão alternativa | `schumann_2023` (MST), `pereira_2026` (SkinToneNet), `porgali_2023_ccv2` (CCv2) |
| Mitigação algorítmica | `park_2022`, `sagawa_2020`, `manzoor_2024`, `liu_2025` (BNMR), `zhang_2018` (adversarial), `bhaskaruni_2019` (ensemble), `dehdashtian_2024` (U-FaTE) |
| Vision-language models | `aldahoul_2024`, `luo_2024_fairclip`, `dehdashtian_2024_fairerclip`, `bendvlm_2024`, **+10 outras Track I** |
| Loss-based FR fundadores | `schroff_2015_facenet`, `wang_2018_cosface`, `deng_2019_arcface`, `meng_2021_magface`, `kim_2022_adaface`, `range_loss_2016` (Track K completo) |

### 5.4 Gaps (o que falta ser explorado)

| Gap | Fichas que confirmam o vazio |
|---|---|
| Skin tone como **sinal arquitetural condicionante** em race 7-class | `perez_2018` (FiLM não em fairness), nenhuma outra ficha cobre |
| Matriz pública **MST × race** | `pereira_2026` audita mas não cross-tabula |
| **Decomposição irredutível × redutível** do erro Latinx | Inédito — nenhuma ficha cobre |
| Fair transferência empírica em **CV** (não NLP) | `aguirre_2023` é NLP, `madras_2018` é teórico |

### 5.5 Objetivo

Coberto na Seção 2 do `_objetivo_tese_v3.3.md` — fundamentado pelos gaps acima.

### 5.6 Como será feito

| Componente | Fichas que validam o método |
|---|---|
| Pipeline 6 etapas | Aprovado orientador 2026-06-08 (`_reuniao_2026-06-08.md`) |
| 4 configs no Cap 2 | `_decisao_arquitetural_film.md` (Opção 2 aprovada com você) |
| Triangulação de métricas | `hardt_2016`, `kleinberg_2017`, `dominguez_2024` |
| Comparação com 6 baselines | Park (FSCL+), Sagawa (DRO), Manzoor (FineFACE), Zhang (Adversarial), ConvNeXt-T puro, ResNet-34 |
| Ablação FiLM × CLIP | `luo_2024_fairclip`, `dehdashtian_2024_fairerclip`, `bian_2025_lorafair` |

---

## 6. GAPS remanescentes identificados na revisão completa

### 6.1 Gaps que reforçam contribuições (boas notícias)

| Gap | Implicação |
|---|---|
| `pereira_2026` audita FairFace mas **não cross-tabula MST × race** | C2 confirmada como inédita |
| `perez_2018` (FiLM) explicitamente diz em §11 Future Work que **fairness é direção não testada** | C3 confirmada como inédita |
| `madras_2018` (LAFTR) é puramente teórico; `aguirre_2023` é NLP | C5 confirmada como inédita em CV |

### 6.2 Sinais novos do corpus (achados pós-revisão)

| Sinal | Fichas | Implicação |
|---|---|---|
| Pesquisas portuguesas crescentes na nossa área | `massively_annotated_2024`, `mst_kd_2024`, `occlusion_bias_2024`, `voidface_2025` (4 do grupo Univ. Porto + INESC TEC + Univ. Coimbra) | Considerar citar como produção nacional/lusófona |
| Pesquisa brasileira top-tier | `survey_fairness_vision_lang_2024` (Parraga et al. — PUCRS MALTA Lab) | Referência canônica brasileira para Cap 2 |
| Same-class similarity intrínseca (Pangelinan Fig 26) | `pangelinan_2023` Fig 26 | Possível análogo intra-racial — monitorar no estudo comparativo C7 |
| Wang+Deng BUPT 2022 já argumentava skin tone > race labels | `wang+deng_2022` (paper canônico que introduz BUPT) | **Reforço explícito** da decisão metodológica da tese — usar como citação central |

### 6.3 Gaps remanescentes (não bloqueantes)

| Gap | Impacto |
|---|---|
| `debiasing_neural_interventions_2025` ainda OVERVIEW_ONLY (Springer paywall) | Track I tem outras 13 fichas cobrindo o tema; baixo impacto |
| `survey_long_tail_2022` (BUPT) introduz **coeficiente de Gini** como métrica de long-tailedness | Possível métrica auxiliar para reportar imbalance do FairFace 7-class — **considerar adoção** |

---

## 7. Recomendações de ajustes (não estruturais)

### 7.1 Ajustes táticos no documento de objetivos (v3.4)

1. **Formalizar H6 como OE-6**: decomposição quantitativa de variância pixel info × skin tone como objetivo específico.
2. **Citar Wang+Deng BUPT 2022** explicitamente como precedente metodológico que já argumentava skin tone > race labels (reforço externo da decisão).
3. **Adicionar referências brasileiras/portuguesas** na seção de "estado da arte": Parraga (PUCRS), Neto/Mamede/Sequeira (Porto), Muhammed (Coimbra) — produção lusófona.
4. **Considerar Gini coefficient** (Yang et al. IJCV 2022) como métrica auxiliar de long-tailedness.

### 7.2 Ajustes táticos na narrativa

1. **Cap 1 — Contexto regulatório**: incluir GDPR/LGPD/EU AI Act citando `lafargue_2025`, `voidface_2025` (lista regulações), `dp_fedface_2024` (privacy).
2. **Cap 2 — Estudo comparativo**: incluir as 4 configurações da decisão arquitetural (não 3 nem 5).
3. **Cap 3 — Quality control**: protocolo explícito controlando pixel info (motivado por Pangelinan).
4. **Cap 4 — Decomposição variância**: nova seção formal se OE-6 for aprovado.

### 7.3 Sem necessidade de mudança

- **Pipeline 6 etapas**: mantido como aprovado.
- **Backbone ConvNeXt-T**: confirmado (CNN moderna).
- **FiLM como mecanismo central**: justificado vs 8 alternativas.
- **Datasets**: FairFace (Cap 1-2) + RFW/BFW (Cap 3).

---

## 8. Veredito final

| Pergunta | Resposta |
|---|---|
| A tese tem fundamentação bibliográfica suficiente? | ✅ **SIM** — ≥12 fichas centrais + 89 de suporte |
| Há conflitos não endereçados? | ❌ **NÃO** — 7 conflitos identificados, todos com resposta defensiva |
| O pipeline 6 etapas é defensável? | ✅ **SIM** — etapa por etapa fundamentada |
| As 7 contribuições são inéditas? | ✅ **SIM** — gap confirmado para cada uma |
| As 6 hipóteses são testáveis? | ✅ **SIM** — métricas claras + critérios de refutação |
| Precisa de mudanças estruturais? | ❌ **NÃO** — apenas 3-4 ajustes táticos de framing |
| Pode começar a escrita do Cap 1 com segurança? | ✅ **SIM — IMEDIATAMENTE** |

---

## Anexos

- **Bibliografia consolidada**: `docs/tese/referencias.bib` (101 entradas, ~1100 linhas)
- **Auditoria das fichas**: `_auditoria_fichas_relatorio.md` (29A / 14B / 57C / 1D)
- **Pente fino crítico**: `_revisao_critica_corpus_v2.md` (mapeamento por impacto)
- **Decisão arquitetural**: `_decisao_arquitetural_film.md` (8 candidatos avaliados)
- **Objetivos consolidados**: `_objetivo_tese_v3.3.md` (v3.3, pré-pós-reunião 08-jun)

> **Próxima ação recomendada**: atualizar `_objetivo_tese_v3.3.md` para
> **v3.4** incorporando recomendações da Seção 7. Aguarda decisão do
> orientador sobre formalização de OE-6 (H6).
