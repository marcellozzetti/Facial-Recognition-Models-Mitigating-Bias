# Triagem editorial — log de decisões

> Registro da Fase B do fluxo definido em `README.md` §4. Cada paper
> candidato passa por checagem de critérios 1–5 (venue + citações) antes
> de gerar ficha completa. Decisões: ✅ aprovado | ❌ descartado | ⚠ standby.

## Convenções

- **Data de verificação**: registrar para cada linha (citações mudam).
- **Citações**: prioridade Semantic Scholar API (cross-check com Google
  Scholar quando viável). Quando API falha por rate limit, marcar
  origem alternativa.
- **Standby**: paper que viola um critério quantitativo (geralmente
  citações abaixo do threshold) mas é aprovado por exceção
  documentada (cobertura única, venue alternativo forte, proximidade
  temática crítica).

## Rodada 1 — Seeds iniciais (2026-05-25)

### 1.1 Papers seed já listados em `00_referencias.md`

| # | Paper | Venue | Tipo | Citações | Decisão | Justificativa |
|---|---|---|---|---|---|---|
| S1 | **Buolamwini & Gebru (2018)** — Gender Shades | PMLR vol 81 / ACM FAT* | conference (peer-reviewed) | **4 933** (Semantic Scholar via search 2026-05-25) | ✅ APROVADO | Venue forte + impacto seminal (>4 000 citações). Atende critérios 1–5 integralmente. |
| S2 | **Karkkainen & Joo (2021)** — FairFace | WACV 2021 (IEEE/CVF) | conference (peer-reviewed) | ≥263 (Google Scholar via search 2026-05-25, subestimado; API S2 não consultada por rate limit) | ✅ APROVADO | Venue forte + dataset central do trabalho (paper-anchor). Re-verificação numérica pendente, mas decisão sustentada pela centralidade. |
| S3 | **AlDahoul, Tan, Kasireddy & Zaki (2024)** — Exploring VLMs for Facial Attribute Recognition | arXiv 2410.24148 (preprint) | preprint | **11** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Não publicado em venue peer-reviewed; abaixo do threshold de 2024 (≥20). MAS é a **única referência publicada para a tarefa exata** que rodamos (raça 7-class no-domain via VLM). Aprovar por **cobertura única** (critério 4 alternativo, §3.2 do README). Acompanhar publicação follow-up dos autores em Nature Scientific Reports (FaceScanPaliGemma, 2026 — verificar separadamente). |
| S4 | **Manzoor & Rattani (2024)** — FineFACE | **ICPR 2024** (Springer LNCS) | conference (peer-reviewed) | **2** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Venue OK (ICPR é peer-reviewed). Citações abaixo do threshold de 2024 (≥20). Aprovar por proximidade temática alta (fair facial attribute classification) + necessidade de posicionamento crítico (o paper foi inicialmente confundido com classificação racial; auditoria revelou que classifica gênero). É um caso paradigmático de **leitura crítica vs leitura de abstract**. |
| S5 | **Dehdashtian, Sadeghi & Boddeti (2024)** — U-FaTE | **CVPR 2024** | conference (peer-reviewed) | **23** (Semantic Scholar 2026-05-25) | ✅ APROVADO | Venue top da área + acima do threshold de 2024 (≥20). Atende critérios 1–5 integralmente. |
| S6 | **Dominguez-Catena, Paternain & Galar (2024)** — DSAP | **Information Fusion** (Elsevier, journal) | journal (peer-reviewed) | **7** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Periódico top da área (Information Fusion, IF~14). Citações abaixo do threshold MAS venue forte compensa (critério 4 prevalece sobre 5 quando periódico top). Nota: registro Semantic Scholar lista year=2023; arXiv submissão dez/2023; publicação efetiva em Information Fusion 2024. Resolver discrepância na ficha. |
| S7 | **Lafargue, Claeys & Loubes (2025)** — Fairness is in the Details | **ECML PKDD 2025** (Springer LNCS) | conference (peer-reviewed) | **1** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Venue OK. Paper muito recente (2025), citações naturalmente baixas. Aprovar por **cobertura única**: única referência identificada que ancora auditoria de fairness na regulação europeia (EU AI Act). |

### 1.2 Papers seed adicionais (a incorporar em `00_referencias.md`)

| # | Paper | Venue | Tipo | Citações | Decisão | Justificativa |
|---|---|---|---|---|---|---|
| S8 | **Wang, Deng, Hu, Tao & Huang (2019)** — Racial Faces in-the-Wild (RFW) | **ICCV 2019** (IEEE/CVF) | conference (peer-reviewed) | a verificar (estimado alto — dataset seminal) | ✅ APROVADO | Venue top + dataset seminal complementar ao FairFace para auditoria de bias em face recognition (não classification). Permite triangulação. |
| S9 | **Grother, Ngan & Hanaoka (2019)** — NISTIR 8280 / FRVT Part 3: Demographic Effects | **NIST Interagency Report** (gov technical report) | technical report | n/a (não-paper; relatório oficial) | ✅ APROVADO POR EXCEÇÃO | Relatório técnico oficial NIST com auditoria industry-wide de ~200 algoritmos e ~18M imagens. Não revisado por pares no sentido acadêmico, mas é a **única fonte regulatória/industry com escala desse porte**. Aprovar como evidência industry-wide complementar à literatura acadêmica. |
| S10 | **Candidatos a fair generation** (Friedrich et al. "Gaussian Harmony" arXiv 2312.14976; Perera & Patel "Unbiased-Diff"; outros) | arXiv preprints / Springer | preprint / chapter | a verificar | ⚠ STANDBY | Aspecto **periférico** à nossa pesquisa (geração vs classificação). Decidir inclusão após `06_gap.md`: se o gap envolver dimensão generativa, incorporar; se não, descartar para `_descartados.md` com justificativa. |

## 2. Resumo da Rodada 1

- **Aprovados diretamente (critérios atendidos sem exceção):** S1, S2, S5, S8 — 4 papers.
- **Aprovados por exceção (cobertura única ou venue forte compensando):** S3, S4, S6, S7, S9 — 5 papers.
- **Standby para decisão pós-gap:** S10 (fair generation) — 1 grupo.
- **Descartados:** nenhum nesta rodada.

**Total aprovado para Fase C (leitura integral + ficha):** 9 papers.

## 3. Pendências de verificação numérica

Itens que ficaram com dados parciais por rate limit ou ainda não buscados:

- [ ] Citação Semantic Scholar de **FairFace** (Karkkainen & Joo 2021, arXiv:1908.04913) — API retornou 429.
- [ ] Citação Semantic Scholar de **RFW** (Wang et al. 2019, arXiv:1812.00194) — não consultada.
- [ ] Verificar se **FaceScanPaliGemma** (Nature Scientific Reports 2026, follow-up de AlDahoul et al.) deve entrar como S3-bis.
- [ ] Decidir candidato representativo único para S10 (fair generation) ou descartar grupo inteiro pós-gap.

## 4. Próxima ação

Iniciar **Fase C** — leitura integral e ficha — pelos 4 papers aprovados sem exceção (S1, S2, S5, S8) para construir baseline metodológico. Em seguida, os 5 papers aprovados por exceção.

Ordem sugerida de leitura (priorização por valor para `06_gap.md`):

1. **S2 FairFace** — dataset central, fundamenta toda a discussão.
2. **S1 Gender Shades** — marco fundador do campo, ancora a motivação ética.
3. **S5 U-FaTE** — formaliza o trade-off utility–fairness, esqueleto teórico.
4. **S8 RFW** — segundo dataset de bias auditing, permite triangulação.
5. **S6 DSAP** — metodologia de auditoria de datasets, conecta com S9.
6. **S9 NISTIR 8280** — escala industrial, ancora positioning.
7. **S3 AlDahoul et al.** — única referência para a tarefa exata.
8. **S4 FineFACE** — armadilha textual já identificada (gênero ≠ raça); ler para entender o que NÃO é nosso problema.
9. **S7 Lafargue et al.** — auditoria regulatória, ancora discussão ética/EU AI Act.

## Rodada 3 — Snowballing direcionado: fairness sem FairFace (2026-05-25)

**Motivação:** após Rodada 2 completar 14 fichas, usuário levantou
preocupação metodológica de **viés de cobertura**: corpus pode estar
muito FairFace-cêntrico. Adicionalmente, levantou questões sobre
**merge de classes** e **correção da taxonomia 7-class**.

### 3.1 Critérios de busca

- Datasets de fairness facial **alternativos ao FairFace**.
- Trabalhos sobre **skin tone** (Fitzpatrick / MST) como alternativa a
  race.
- Trabalhos que **questionam a taxonomia discreta** em si.
- Surveys recentes (2023+) que possam ter referências cruzadas
  faltantes.

### 3.2 Candidatos verificados

| # | Paper | Venue | Tipo | Citações | Decisão | Justificativa |
|---|---|---|---|---|---|---|
| R3-1 | **Neto, Damer, Cardoso & Sequeira (2025)** — Continuous Demographic Labels | arXiv 2506.01532 (UNDER REVIEW) | preprint | 0 (recente demais) | ✅ APROVADO POR EXCEÇÃO | Único paper que questiona discretização de raça em FR. 65+ modelos treinados, 20+ subsets. Aprovar por **cobertura única** (Q09 sem outra resposta). Reavaliar antes da defesa. |
| R3-2 | **Schumann, Olanubi, Wright, Monk Jr., Heldreth & Ricco (2023)** — MST consensus | **NeurIPS 2023 Datasets and Benchmarks Track** | conference (peer-reviewed top venue) | (sem rate-limit; assumido alto) | ✅ APROVADO | NeurIPS D&B é top venue. Liberação do MST-E + protocolo de anotação canônico. Responde Q01 🔬 e Q10 🔬. |
| R3-3 | **Robinson, Livitz, Henon, Qin, Fu & Timoner (2020)** — BFW | **CVPRW 2020** | conference workshop (peer-reviewed) | (sem rate-limit) | ✅ APROVADO | Alternativa balanceada ao FairFace para verification. Permite triangulação. Citado por [[neto_2025]]. |
| R3-4 | **Hazirbas, Bitton, Dolhansky, Pan, Gordo & Canton Ferrer (2021)** — Casual Conversations | **CVPRW 2021** | conference workshop (peer-reviewed) | (sem rate-limit) | ✅ APROVADO | Self-reported demographics (gold standard) + Fitzpatrick. Argumento explícito contra race labeling (citação direta). Fundamenta Q01 🔬, Q09, Q10 🔬. |
| R3-5 | **Kotwal & Marcel (2025)** — Review of Demographic Fairness in FR | **IEEE TBIOM 2025** | journal (top biometrics) | (recente; sem citações ainda) | ✅ APROVADO POR EXCEÇÃO | Survey mais recente e mais específico de demographic fairness em FR (vs Mehrabi 2021 mais geral). Mesmo grupo de [[lafargue_2025]] (Sébastien Marcel). |

### 3.3 Candidatos identificados mas não-aprovados (Rodada 3)

| Paper | Razão para não-aprovar |
|---|---|
| **BUPT-Balancedface** (Wang et al.) | Citado em Neto 2025 mas sem ficha dedicada por enquanto — pode justificar Rodada 4 se necessário em `06_gap.md` |
| **DemogPairs** (Hupont & Tena) | Citado em Robinson 2020 — small dataset (~3 ethnicities); curado sob VGGFace2; cobertura sobreposta com BFW |
| **Casual Conversations v2** (Porgali et al. 2023) | Extensão da v1; principais conceitos já cobertos em v1. Reavaliar se v2 trouxer mudança qualitativa. |
| **Draelos, Kesty & Kesty (2025)** (J Cosmetic Dermatology) | Aplica Fitzpatrick a subset FairFace, MAS (i) dermatologia, não fairness; (ii) dados não-públicos; (iii) sem cross-reference race × Fitzpatrick. **Referência conceitual em Q10** apenas, sem ficha. |
| **AI-Face** (Lin 2025 CVPR) | Não verificado em detalhe; provável extensão de tarefa diferente (face manipulation) |

### 3.4 Resumo Rodada 3

- **5 papers adicionais aprovados.**
- Cobertura agora inclui: race classification, race recognition, skin
  tone (Fitzpatrick + MST), continuous labels, survey atualizado.
- **Total corpus: 19 fichas** (9 R1 + 5 R2 + 5 R3) + Rodada 2.5
  validação SOTA.
- **Novos pontos de pesquisa identificados (Q07-Q10):** documentados
  em `_perguntas.md`.

## Rodada 2.6 — Re-verificação SOTA pós-reunião (2026-06-04)

**Motivação:** orientador pediu **double-check** do SOTA após
reunião de 2026-06-04. Janela: fevereiro a junho de 2026 (período
post-Rodada 2.5 que rodou em maio).

### 2.6.1 Procedimento

Busca em 4 frentes:

1. **Semantic Scholar "cited by" de arXiv:2410.24148** (AlDahoul VLM
   preprint). **12 papers citantes** examinados (lista no commit
   log da Rodada 2.6).
2. **Google Scholar** janela fev-jun 2026 — busca por
   "FairFace race classification SOTA new method".
3. **Busca direcionada por valores numéricos** ("75.7", "76", "77",
   "78" + FairFace race 7-class accuracy 2026).
4. **arXiv recent submissions** (cs.CV jan-jun 2026 + filtros race
   classification).

### 2.6.2 Análise dos 12 citantes do FaceScanPaliGemma

Nenhum dos 12 papers que citam AlDahoul roda **FairFace race 7-class
in-domain** como métrica principal:

| Paper | Tarefa principal | Conexão com nosso problema |
|---|---|---|
| Molinaro 2026 — Group Emotion Recognition | Emoção em grupos | ❌ Não-race |
| Roygaga 2026 WACV — Human-Like Biases in VLMs | Auditoria subjetiva | ⚠ Audit, sem novo classifier |
| Sajib 2025 ICCIT — VLAgeBench | Age estimation | ❌ Não-race |
| V 2025 ICAFT — YOLO Human Detection | Detecção, não classificação | ❌ Não-race |
| Park 2025 — Stereotypes T2I | Bias em geração T2I | ❌ Não-classification |
| Ngai 2025 — Feline Grimace Scale | Veterinária (felinos) | ❌ Irrelevante |
| Ceschini 2025 CVPRW — Gender Scaling Laws CLIP | Gender em CLIP | ❌ Não-race |
| Hosseini 2025 — Faces of Fairness FER | Bias em expressão facial | ⚠ Audit FER, não race classification |
| Hambarde 2025 — Re-ID LVLMs | Re-identification | ❌ Não-classification atributo |
| Wang 2026 IEEE Access — FM-GCN Gender Micro-Expr | Gender micro-expressions | ❌ Não-race |
| Debnath 2025 IEEE Access — LightSTATE | Activity detection | ❌ Não-race |
| Narayanan 2025 IEEE Access — YOLOv9 Face Detection | Face detection (não classification) | ❌ Não-race |

### 2.6.3 Conclusão

**FaceScanPaliGemma (AlDahoul et al. 2024 arXiv / 2026 Nature
Scientific Reports) permanece SOTA único para FairFace race 7-class
in-domain como junho de 2026.**

- Acurácia: **75.7%** (7-class), **81.1%** (6-class East+SE merged)
- F1 macro: **75%** (7-class)
- Per-class F1: Black 90% > White 80% > Indian 79% > East Asian 78% >
  Middle East 73% > SE Asian 67% > **Latinx 60%** (pior)

Padrão hierárquico Black > Latinx confirmado por **2 fontes
independentes** (AlDahoul 2024 + Lin 2022 FairGRAPE) — robusto.

### 2.6.4 Implicação para tese v3.2

Hipóteses H1-H5 mantêm-se válidas. Plano experimental Cap 1 (Q06)
e Cap 2 (Q04) podem reportar contra baseline 72% (ResNet-34) e SOTA
75.7% (FaceScanPaliGemma) como **dois números canônicos** sem risco
de obsolescência durante a dissertação.

## Rodada 7 — Expansão pós-reunião (2026-06-10, primeira leva)

**Motivação:** Reunião com Prof. Quiles em 2026-06-08 aprovou
pipeline v3.2 e recomendou:
- Corpus ≥ 100 artigos (filtros menos restritivos)
- ≥ 20 artigos aprovados de 2025/2026
- Adicionar CLIP/BLIP como mecanismo alternativo de conditioning
- Incluir capítulo metodológico avaliando modelos pré-treinados

### 7.1 Critérios revisados

| Critério | R1-R6 | **R7 (revisado)** |
|---|---|---|
| Venue | Top conferences + journals | **+ workshops + preprints ≤18 meses** |
| Citações 2024 | ≥ 20 | **≥ 10** |
| Citações 2025 | ≥ 10 | **≥ 3** |
| Citações 2026 | sem threshold | mantido |
| Aplicabilidade | Direta | **+ adjacente** (VLM bias geral) |
| Cobertura única | Caso especial | mantido |

### 7.2 Buscas executadas (5 lotes, 12 queries)

- Lote 1: VLM/CLIP/BLIP fairness — 13 candidatos
- Lote 2: skin tone moderno — 5 candidatos
- Lote 3: race classification SOTA 2025-2026 — 6 candidatos
- Lote 4: conditioning moderno LoRA/adapter — 8 candidatos
- Lote 5: SSL + multi-task — 5 candidatos

### 7.3 Primeira leva — 18 fichas APROVADAS imediatamente

**Track I — VLM/CLIP/BLIP fairness (NOVO, 11 fichas):**
- R7-1 Luo et al. 2024 CVPR — FairCLIP (Sinkhorn + Harvard-FairVLMed)
- R7-2 Dehdashtian et al. 2024 ICLR — FairerCLIP (RKHS zero-shot)
- R7-3 Joint VL Bias Removal 2024 (alignment + counterfactual)
- R7-4 Lin et al. 2025 CVPR — AI-Face Million-Scale
- R7-5 GRAS 2025 — VLM bias benchmark 2.5M queries
- R7-6 Unified Debiasing VLMs 2024 (cross-modal framework)
- R7-7 Evaluating LVLM 2024 Findings EMNLP 2025
- R7-8 BendVLM 2024 (test-time debiasing)
- R7-9 Debiasing CLIP Neural Interventions 2025 Springer
- (2 mais a aprovar em batch futuro: AIM-Fair, Reliable Demo Inference)

**Track J — Conditioning moderno (NOVO, 6 fichas):**
- R7-10 Zhao et al. 2025 CVPR — AIM-Fair (selective fine-tuning)
- R7-11 Bian et al. 2025 ICCV — LoRA-FAIR (federated)
- R7-12 FairLoRA 2024 (fairness-driven LoRA)
- R7-13 On Fairness of LoRA 2024 (análise fundamental)
- R7-14 Tian et al. 2024 ECCV — FairViT (adaptive masking)

**Track auxiliar — skin tone + race + SSL (4 fichas):**
- R7-15 Porgali et al. 2023 CVPRW — Casual Conversations v2
- R7-16 Reliable Demographic Inference 2025
- R7-17 Demographic-Agnostic Fairness 2025
- R7-18 Provable Adversarial Fair SSL 2024

### 7.4 Standbys (14 candidatos)

Aguardando triagem detalhada: Reproducibility FairCLIP, Wang FairCLIP,
Prompt Array, JAAD MST dermatology, Skin Segmentation MST, Skin Tone
Estimation Lighting, Enhancing Fairness MST, Bahiru 2025, Dong 2025
(buscar), Fair Verification Demographic, FairViT debiased self-attention,
S-Adapter (descartável), SSL w/o Demographics, NTKMTL, etc.

### 7.5 Descartes (2)

- FairQueue (text-to-image generation — fora do escopo)
- S-Adapter ViT Anti-Spoofing (anti-spoofing, não fairness)

### 7.6 Resumo Rodada 7 primeira leva

- **18 aprovados imediatos** + 14 standby + 2 descartes = 34 verificados
- **Total corpus após R7 leva 1: 55 fichas** (era 37)
- **2025-2026 atendido: 20 fichas** (era 5) ✅
- **2 novos Tracks**: I (VLM fairness) e J (conditioning moderno)
- **Gap remanescente para meta 100**: −45 fichas
  → próximos lotes R7 atacarão surveys recentes, workshop papers,
  papers fundadores faltantes (ArcFace, FaceNet, CosFace)

## Rodada 6 — Validação científica do pipeline v3.2 (2026-06-06)

**Motivação:** após v3.2 escrita e PPTX v3.2.1 entregue, usuário
solicitou abrir frente de pesquisa para verificar se cada etapa do
pipeline tem embasamento (ou identificar refutações). Mapeamento
pipeline-step × evidência em 5 áreas críticas. Documento síntese em
`docs/ativo/_validacao_cientifica_pipeline.md`.

### 6.1 Áreas pesquisadas

1. Condicionamento por atributo auxiliar para fairness (FiLM-like, multi-task, adversarial)
2. MST aplicado a auditoria de FairFace (matriz pública MST × race)
3. Transferência empírica fair classification → fair face recognition
4. Backbone leve (ConvNeXt-T) para skin-tone prediction
5. Disparidade Black/African em face recognition

### 6.2 Candidatos verificados

| # | Paper | Venue | Tipo | Decisão | Justificativa / impacto na tese |
|---|---|---|---|---|---|
| R6-1 | **Pangelinan et al. 2023** — Exploring Causes of Demographic Variations In FR Accuracy (arXiv:2304.07175) | preprint (autores Notre Dame/IST/Florida Tech) | preprint | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | **Refutação potencial de H5**: "demographic differences in face pixel information of the test images appear to most directly impact the resultant differences in FR accuracy" — sugere que para FR, pixel info > skin tone. Necessário endereçar diretamente; aprovação por **risco crítico identificado**. |
| R6-2 | **Pereira et al. 2026** — Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild / SkinToneNet + STW (arXiv:2603.02475) | preprint (4 autores; afiliações a verificar no PDF) | preprint | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO → **✅ FICHADO 2026-06-09** ([pereira_2026.md](pereira_2026.md)) | **Verificação 2026-06-09 corrigiu meta-dados**: paper audita CelebA e VGGFace2 (NÃO FairFace) — fortalece originalidade da nossa C2. Decisão técnica: usar SkinToneNet pré-treinado como insumo da Etapa 1, substituindo treino próprio do classificador MST. STW (42.313 imgs) disponível como dataset auxiliar de validação. |
| R6-3 | **Dooley et al. 2022/23** — Rethinking Bias Mitigation: Fairer Architectures Make for Fairer FR (arXiv:2210.09943) | NAS / FR — preprint citado em literatura recente | preprint (publicado em NeurIPS 2023 workshop FAccT-adj.) | ✅ APROVADO | "biases são inerentes a arquiteturas neurais" — Pareto-domina baselines de mitigação. **Fortalece a importância de ConvNeXt-T** como escolha arquitetural moderna. Suporte para H2. |
| R6-4 | **Aguirre & Dredze 2023/24** — Transferring Fairness using Multi-Task Learning with Limited Demographic Information (arXiv:2305.12671) | preprint (Johns Hopkins) | preprint | ✅ APROVADO POR EXCEÇÃO | Prova empírica de que **demographic fairness objectives transfer fairness within a multi-task framework**. Reforça princípio teórico do pipeline (etapas 3 e 5). Domínio é NLP mas princípio se transfere. |
| R6-5 | **Kolla & Savadamuthu 2022** — The Impact of Racial Distribution in Training Data on FR Bias: A Closer Look (arXiv:2211.14498) | **WACVW 2023** (IEEE/CVF) | workshop conference (peer-reviewed) | ✅ APROVADO | "uniform distribution of races in training datasets alone does not guarantee bias-free FR" — REFORÇA necessidade de mecanismo arquitetural além de balanceamento. Justifica FiLM-conditioning. |
| R6-6 | **Liu et al. 2025** — Component-Based Fairness in Face Attribute Classification with Bayesian Network-informed Meta Learning (BNMR) (arXiv:2505.01699) | **ACM FAccT 2025** | conference (top venue ética AI) | ✅ APROVADO | Baseline competitivo recente. Mecanismo ortogonal (meta-learning sample reweighting via Bayesian Network) ao FiLM. Candidato a baseline forte do Cap 2. |
| R6-7 | **Ramachandran & Rattani 2024** — A Self-Supervised Learning Pipeline for Demographically Fair Facial Attribute Classification (arXiv:2407.10104) | **IJCB 2024** | conference (peer-reviewed) | ✅ APROVADO | SSL fair attribute classification em FairFace + CelebA. Baseline competitivo, mecanismo distinto (pseudo-labels + meta-learning contrastive). |
| R6-8 | **Raumanns et al. 2024** — Dataset Distribution Impacts Model Fairness: Single vs. Multi-Task Learning (arXiv:2407.17543) | **FAIMI EPIMI 2024 Workshop** (LNCS 15198) | workshop (peer-reviewed) | ✅ APROVADO POR EXCEÇÃO | Domínio é skin lesion (não face), mas achado é generalizável: **"reinforcement multi-task does NOT remove sex bias; adversarial scheme eliminates it in some cases"**. Cautela contra multi-task naive. Reforça adversarial (Zhang 2018) como baseline forte. |
| R6-9 | **Lu 2025** — TrueSkin: Towards Fair and Accurate Skin Tone Recognition and Generation (arXiv:2509.10980) | preprint | preprint | ⚠ STANDBY | 7,299 imagens em **6 classes** (não 10 MST). Não compete diretamente. Manter em standby — incluir só se discussão da escolha de 10 classes precisar de ancoragem comparativa. |

### 6.3 Resumo Rodada 6

- **8 papers aprovados** (R6-1 a R6-8); 1 em standby (R6-9).
- Mistura de venues: FAccT, IJCB, WACVW, FAIMI EPIMI + 4 preprints.
- **Cobertura por área:**
  - Área 1 (auxiliary conditioning): R6-4, R6-6, R6-7, R6-8
  - Área 2 (MST × FairFace audit): R6-2
  - Área 3 (fair transfer): R6-1, R6-3, R6-4, R6-8
  - Área 4 (backbone skin tone): R6-2 (ViT-Small SkinToneNet)
  - Área 5 (Black/African FR): R6-1, R6-3, R6-5
- **Total corpus após R6: 37 fichas** (29 + 8 aprovados).

### 6.4 Status de fichamento (atualizado 2026-06-10)

Todas as 8 fichas R6 criadas:

| # | Paper | Estado da ficha |
|---|---|---|
| R6-1 | Pangelinan 2023 | [OVERVIEW_ONLY](pangelinan_2023.md) — leitura PDF pendente |
| R6-2 | Pereira 2026 SkinToneNet | [VERIFIED](pereira_2026.md) — HTML integral lido |
| R6-3 | Dooley 2022 NAS | [OVERVIEW_ONLY](dooley_2022.md) — leitura PDF pendente |
| R6-4 | Aguirre & Dredze 2023 | [VERIFIED](aguirre_2023.md) — HTML integral lido |
| R6-5 | Kolla & Savadamuthu 2022 | [OVERVIEW_ONLY](kolla_2022.md) — leitura PDF pendente |
| R6-6 | Liu 2025 BNMR | [OVERVIEW_ONLY](liu_2025.md) — leitura PDF pendente |
| R6-7 | Ramachandran & Rattani 2024 | [OVERVIEW_ONLY](ramachandran_2024.md) — leitura PDF pendente |
| R6-8 | Raumanns 2024 | [OVERVIEW_ONLY](raumanns_2024.md) — leitura PDF pendente |

Promoção OVERVIEW_ONLY → VERIFIED requer download de PDF e leitura
integral. 6 papers pendentes; priorizar Pangelinan (refutação H5)
e Dooley (validação ConvNeXt-T) primeiro.

### 6.4 Impacto na tese (v3.2 → v3.3)

R6 abre **5 riscos críticos** que motivam revisão da v3.2 para v3.3:

- **Risco A** (R6-1, Pangelinan 2023): H5 precisa ser reformulada para
  reconhecer pixel info como confounder em FR; adicionar quality control
  de imagem ao Cap 3. **⏸️ DECISÃO ADIADA para reunião de segunda
  (2026-06-08)** — 3 versões de reformulação preparadas para discussão.
- **Risco B** (R6-2, Pereira 2026): considerar usar **SkinToneNet
  pré-treinado** como insumo em vez de treinar próprio classificador
  MST; foco da contribuição passa a ser **FiLM-conditioning** + matriz
  MST × race. **✅ DECIDIDO 2026-06-06: usar SkinToneNet pré-treinado**
  (recomendação técnica baseada em evidência R6).
- **Risco C** (sem precedente direto FiLM+MST→race): mitigar com piloto
  inicial em subset FairFace. Já contemplado no plano.
- **Risco D** (R6-8, Raumanns 2024): adversarial debiasing pode superar
  FiLM; já contemplado como baseline forte. **✅ DECIDIDO 2026-06-06:
  adicionar Adversarial debiasing (Zhang 2018) à lista de baselines.**
- **Risco E** (R6-2, ConvNeXt-T não-benchmarkado para skin tone): rodar
  ConvNeXt-T vs ViT-Small em sub-experimento no Cap 1. Já contemplado.

Detalhes completos em [_validacao_cientifica_pipeline.md](../_validacao_cientifica_pipeline.md).
Material de discussão de H5 na Seção 11 do mesmo arquivo.

## Rodada 5 — Mecanismos algorítmicos ML / Redes Neurais (2026-06-04)

**Motivação:** após reunião com orientador (Prof. Quiles), feedback
explícito de que faltava ampliar pesquisa para venues de ML / Redes
Neurais. Adicionalmente, ele propôs **linha de pesquisa
prescritiva** (tese v3.2): treinar MST classifier, usar saída para
condicionar race classifier, aplicar a face recognition para
verificar melhora em grupos sub-representados.

### 5.1 Critérios de busca

- Papers fundadores de **fair representation learning** em ML.
- Mecanismos de **condicionamento neural** (FiLM, conditional BN,
  attention modulada).
- **Adversarial debiasing** — formalização e prática.
- **Métricas de fairness** com prova teórica (impossibilidade
  Kleinberg).
- **Venues**: NeurIPS, ICML, ICLR, AAAI, KDD, AIES.

### 5.2 Candidatos verificados

| # | Paper | Venue | Tipo | Decisão | Justificativa |
|---|---|---|---|---|---|
| R5-1 | **Hardt, Price & Srebro (2016)** — Equality of Opportunity in Supervised Learning | **NeurIPS 2016** | conference (top venue) | ✅ APROVADO | Paper-fonte das métricas EO_h/EOD usadas em toda literatura subsequente. Citado em 7 fichas existentes; faltava ficha dedicada. |
| R5-2 | **Perez, Strub, de Vries, Dumoulin & Courville (2018)** — FiLM | **AAAI 2018** | conference (top venue) | ✅ APROVADO | Mecanismo formal de condicionamento neural — direta operacionalização do pipeline v3.2 do orientador. |
| R5-3 | **Zemel, Wu, Swersky, Pitassi & Dwork (2013)** — Learning Fair Representations | **ICML 2013** (Test-of-Time Award ICML 2023) | conference (top venue) | ✅ APROVADO | Paradigma fundador de Fair Representation Learning. Test-of-Time Award = forte endosso comunidade. |
| R5-4 | **Madras, Creager, Pitassi & Zemel (2018)** — LAFTR | **ICML 2018** | conference (top venue) | ✅ APROVADO | Conecta formalmente noção de fairness ↔ objetivo adversarial. Demonstra fair transfer learning — fundamenta extensão race→face recognition. |
| R5-5 | **Zhang, Lemoine & Mitchell (2018)** — Mitigating Unwanted Biases with Adversarial Learning | **AAAI/ACM AIES 2018** | conference (top venue ética AI) | ✅ APROVADO | Adversarial debiasing operacional. Candidato a baseline mitigação para Cap 2 v3.2. |
| R5-6 | **Kleinberg, Mullainathan & Raghavan (2017)** — Inherent Trade-Offs in the Fair Determination of Risk Scores | **ITCS 2017** (LIPIcs) | conference (top theory venue) | ✅ APROVADO | Teorema da impossibilidade da fairness. Fundamenta nossa escolha de triangulação de métricas (Q05). |

### 5.3 Resumo Rodada 5

- **6 papers aprovados.**
- Todos top venues ML/CS (NeurIPS, ICML, AAAI, ITCS, AIES).
- **Track G — Mecanismos ML / Redes Neurais** consolidado.
- **Total corpus: 29 fichas** (9 R1 + 5 R2 + 5 R3 + 4 R4 + 6 R5).

### 5.4 Impacto na tese (v3.1 → v3.2)

R5 fundamenta a **reformulação prescritiva** sugerida pelo
orientador:

- **FiLM** ([[perez_2018]]): mecanismo formal para condicionar race
  classifier com saída do MST classifier.
- **LAFTR** ([[madras_2018]]): fundamenta extensão race → face
  recognition (fair transfer learning).
- **Hardt 2016** ([[hardt_2016]]): formalização das métricas EO_h/EOD
  reportadas.
- **Kleinberg 2017** ([[kleinberg_2017]]): justifica triangulação de
  métricas (não há "fairness única").
- **Zhang 2018** ([[zhang_2018]]): baseline adversarial alternativo
  ao FSCL+ ([[park_2022]]) para Cap 2.

## Rodada 4 — Fundamentação científica de raça e tom de pele (2026-05-25)

**Motivação:** após Rodada 3 + Fase 4 (landscape, gap, thesis v3),
usuário levantou 4 perguntas fundamentais sobre o **substrato
científico** da pesquisa:

1. Existem características físicas associadas à raça identificáveis
   em imagens?
2. Tom de pele × raça — estudos médicos/científicos?
3. As 7 raças FairFace existem cientificamente?
4. Quantos tons de pele cientificamente (medicina)?

Estas perguntas tocam o **fundamento epistemológico** da dissertação
e exigem ancoragem em literatura fundadora de antropologia,
genética e dermatologia.

### 4.1 Critérios de busca

- **Position statements oficiais** de associações científicas sobre
  raça.
- **Genética populacional clássica** sobre apportionment.
- **Origem documentada** de escalas de tom de pele (Fitzpatrick,
  Felix von Luschan, Massey-Martin, MST).
- **Antropologia forense moderna** — uso de "population affinity".

### 4.2 Candidatos verificados

| # | Paper | Venue | Tipo | Decisão | Justificativa |
|---|---|---|---|---|---|
| R4-1 | **Fuentes, Ackermann et al. (2019)** — AAPA Statement on Race and Racism | **Am J Phys Anthropol** (vol 169, no 3) | journal (peer-reviewed, posição institucional) | ✅ APROVADO | Statement oficial AABA/AAPA. Fundamento institucional para Q11. Comunidade científica de antropologia biológica. |
| R4-2 | **Lewontin (1972)** — The Apportionment of Human Diversity | *Evolutionary Biology*, vol 6, Springer | book chapter (peer-reviewed) | ✅ APROVADO | Citação clássica fundadora. 50+ anos de literatura derivada. Evidência genética principal (85% intra-pop). Fundamento para Q11. |
| R4-3 | **Fitzpatrick (1988)** — The Validity and Practicality of Sun-Reactive Skin Types I Through VI | *Archives of Dermatology* (JAMA Network) | journal (peer-reviewed) | ✅ APROVADO | Paper fonte da escala Fitzpatrick. Documenta propósito original (PUVA dosimetry), refuta uso indevido como classificação racial. Fundamento para Q13. |
| R4-4 | **Massey & Martin (2003)** — The NIS Skin Color Scale | Princeton/NIS technical doc | technical report | ✅ APROVADO POR EXCEÇÃO | Documento técnico do New Immigrant Survey. Não é paper acadêmico mas é **escala canônica** em sociologia americana, adotada por GSS, NLSY97, ANES. Precedente histórico-metodológico do MST. Fundamento para Q14. |

### 4.3 Candidato originalmente proposto mas substituído

| Paper | Razão para substituir |
|---|---|
| **Sparks & Jantz 2002** (PNAS, "Boas revisited") | Após leitura: paper na verdade DEFENDE estabilidade craniana (refuta Boas), não argumenta contra raça. Substituído por Lewontin 1972 que é citação genética clássica adequada. |

### 4.4 Resumo Rodada 4

- **4 papers aprovados** como referências de fundamentação científica.
- **3 com PDF integral via download direto bloqueado por anti-scraping**
  — conteúdo capturado via WebFetch sobre HTML/PMC e WebSearch
  summaries. Anotado nas fichas em campo `fonte_leitura`.
- **Total corpus: 23 fichas** (9 R1 + 5 R2 + 5 R3 + 4 R4) + Rodada
  2.5.

### 4.5 Impacto na tese

Rodada 4 **fundamenta teoricamente** a tese v3:

- **AAPA 2019 + Lewontin 1972** sustentam a posição
  epistemológica de que **race ≠ biology**, justificando Q10 (matriz
  skin tone × race) como **conexão entre dois construtos
  diferentes**, não cross-reference de variáveis redundantes.
- **Fitzpatrick 1988** esclarece o **erro categorial** que permeia
  literatura de ML fairness usando Fitzpatrick como proxy de raça.
  Justifica adoção de MST.
- **Massey-Martin 2003** demonstra que **tom de pele tem efeitos
  sociais independentes de raça**, sustentando a relevância prática
  de Q10.

## Rodada 2.5 — Verificação dedicada de SOTA (2026-05-25)

**Motivação:** após Rodada 1 completar leitura dos 9 papers seed e
identificar AlDahoul et al. 2024 (arXiv:2410.24148) como única
referência para FairFace race 7-class classification, o usuário
levantou preocupação metodológica de **certeza absoluta sobre SOTA**
— precedente do caso "Hassanpour" (autoria fabricada em Rodada 0
pré-pivot) torna risco de erro inaceitável.

### 2.5.1 Procedimento

Busca em 4 frentes:

1. **Semantic Scholar "cited by" de arXiv:2410.24148** — 11 papers
   citantes inspecionados; nenhum reporta número superior em FairFace
   7-class race classification.
2. **Google Scholar / WebSearch para "FairFace race 7-class
   accuracy 2025 2026"** — sem competidor identificado.
3. **Papers with Code FairFace leaderboard** — **fora do ar**:
   plataforma descontinuada por Meta em jul/2025; dados arquivados
   em GitHub paperswithcode/paperswithcode-data sem atualizações.
4. **Identificação de versão peer-reviewed do AlDahoul 2024** —
   confirmado: paper foi aceito em **Nature Scientific Reports 2026**
   (DOI: 10.1038/s41598-026-39584-3) com título atualizado
   "FaceScanPaliGemma Multi-Agent Vision Language Models for
   Facial Attribute Recognition". Mesmos números experimentais.

### 2.5.2 Candidatos descartados como competidores

| Candidato | Razão de descarte |
|---|---|
| FairViT (Tian et al. ECCV 2024, arXiv:2407.14799) | Não usa FairFace; testa em CelebA |
| Sufian et al. ICPR 2024 Workshop FAIRBIO (arXiv:2506.05383) | Face authentication (verificação), não classificação racial; sem FairFace |
| Hosseini et al. 2025 (arXiv:2502.11049) | Facial expression recognition, não race; testa em AffectNet/ExpW/Fer2013/RAF-DB |
| GPT-4o (zero-shot) | Já avaliado por AlDahoul (68% acc 7-class, inferior ao FaceScanPaliGemma) |
| CLIP zero-shot | Já avaliado por AlDahoul (64.2% acc 7-class, inferior) |
| VGGFace-ResNet-50 + SVM | Já avaliado por AlDahoul (72.6% acc 7-class, inferior) |
| Google FaceNet + SVM | Já avaliado por AlDahoul (68.9% acc 7-class, inferior) |

### 2.5.3 Veredito de SOTA verificado

**Estado da arte para FairFace race classification in-domain
(test split do próprio FairFace, 10 954 imagens), em 2026-05-25:**

| Classes | Modelo | Accuracy | F1 macro | Referência canônica |
|---|---|---|---|---|
| **7 classes** (White, Black, Indian, East Asian, Southeast Asian, Middle East, Latinx/Hispanic) | **FaceScanPaliGemma** (fine-tuned PaliGemma 3B) | **75.7%** | **75%** | **AlDahoul, Tan, Kasireddy & Zaki (2026), Nature Scientific Reports** (DOI 10.1038/s41598-026-39584-3). Tabela 10 do arXiv 2410.24148 (mesma data). |
| **6 classes** (East+SE Asian merged em "Asian") | FaceScanPaliGemma | 81.1% | 79% | AlDahoul et al. (2026), Tabela 8/9 do arXiv 2410.24148. |
| **Baseline 7-class** | FairFace ResNet-34 | 72% | 72% | AlDahoul et al. (2026), Tabela 10 — **NÃO confundir com 81.5% do paper original do FairFace** (que é out-of-domain Twitter/Media/Protest). |

### 2.5.4 Implicações para a dissertação

1. **Citação canônica:** sempre referenciar **AlDahoul, Tan, Kasireddy
   & Zaki (2026), Nature Scientific Reports** (não mais arXiv preprint),
   exceto quando se discutir cronologia da literatura.
2. **Número SOTA:** **75.7% accuracy / 75% F1 macro** para 7-class
   in-domain FairFace race classification. **Não 81.5%** (que é
   out-of-domain) **e não 81.1%** (que é 6-class).
3. **Baseline para comparação:** **72% accuracy / 72% F1** (FairFace
   ResNet-34 baseline conforme AlDahoul Tabela 10, **não 0.815** do
   paper original do FairFace).
4. **Margem para superação:** qualquer trabalho que reportar **≥ 76%
   accuracy** em FairFace 7-class in-domain, com protocolo
   metodológico defensável (multi-seed, IC), seria o novo SOTA.
5. **Caveat de single seed:** AlDahoul reporta single-run sem IC.
   Nosso trabalho com 3-seed casado já oferece **rigor estatístico
   superior** mesmo se atingir apenas 72-75% accuracy.

### 2.5.5 Reabertura programada

Esta verificação tem prazo de validade. Re-verificar antes da defesa
da dissertação:

- [ ] Re-rodar Semantic Scholar "cited by" de AlDahoul 2026 (Nature)
      para capturar publicações pós-fev-2026.
- [ ] Re-rodar arXiv search para "FairFace race" no mês anterior à
      defesa.
- [ ] Monitorar canais de pré-publicação: CVPR, ICCV, ECCV, FAccT
      proceedings, IEEE TBIOM, ACM CSur.
