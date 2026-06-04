# Thesis Statement v3.2 — Prescritiva

> Reformulação **prescritiva** da tese após reunião com orientador
> (Prof. Marcos Quiles, 2026-06-04) e Rodada 5 de pesquisa
> bibliográfica (mecanismos ML / redes neurais).
>
> **Substitui v3.1** ("diagnóstica") por orientação direta do
> orientador para postura mais executiva e com extensão a face
> recognition.
>
> Versão: **v3.2** — 2026-06-04.

## 1. Tese central

> **A incorporação explícita de informação de tom de pele (Monk
> Skin Tone — MST) como sinal auxiliar condicionante no treinamento
> de classificadores faciais profundos melhora métricas de fairness
> demográfica em classificação racial multi-classe e estende essa
> melhoria a tarefas downstream de reconhecimento facial em grupos
> sub-representados.**
>
> A dissertação contribui originalmente com:
>
> **(a) um pipeline operacional MST→FiLM→classifier — primeira
> instância documentada de condicionamento explícito por tom de
> pele em race classification multi-classe sobre FairFace, ancorado
> em FiLM (Perez et al., AAAI 2018) e Fair Representation Learning
> (Zemel et al., ICML 2013 Test-of-Time);**
>
> **(b) a primeira matriz pública P(MST | race) sobre FairFace,
> construída via classifier MST validado contra anotação manual
> regionalmente diversa (protocolo Schumann et al., NeurIPS 2023);**
>
> **(c) demonstração empírica da propriedade de fair transferência
> (Madras et al., LAFTR ICML 2018) do pipeline para face
> recognition — aplicação a RFW (Wang et al., ICCV 2019) ou BFW
> (Robinson et al., CVPRW 2020) com avaliação de melhora em accuracy
> para grupos sub-representados (Black/African);**
>
> **(d) protocolo de métrica triangular (Disparity Ratio + worst-class
> F1 + Coefficient of Variation) para fairness em classificação
> multi-classe, justificado pela impossibilidade matemática de
> métrica única (Kleinberg, Mullainathan & Raghavan, ITCS 2017).**

## 2. Pipeline experimental

```
Fase 1 — Classifier MST auxiliar
  Treinar/validar sobre MST-E (Schumann 2023) +
  Casual Conversations (Hazirbas 2021)
                  ↓
            f_MST(x) ∈ ℝ^10 (logits MST)
                  ↓
Fase 2 — Auditoria da matriz P(MST | race) sobre FairFace val
  Aplicar f_MST sobre 10 954 imagens
  Validação manual em subset (≥3 anotadores regionais)
  Matriz pública como CSV
                  ↓
Fase 3 — Race classifier condicionado
  Backbone: ConvNeXt-T (28M params)
  Camadas FiLM (Perez et al. 2018) condicionadas por f_MST(x)
  Treino 3-seed casado sobre FairFace train
                  ↓
Fase 4 — Comparação fairness com vs sem MST
  Métricas: F1 macro, DR (max/min F1), worst-class F1, CV
  Baselines: ConvNeXt-T sem MST (vanilla), FSCL+ (Park 2022),
             Group DRO (Sagawa 2020), Adversarial (Zhang 2018)
                  ↓
Fase 5 — Extensão a face recognition
  Aplicar pipeline análogo a RFW ou BFW (verification 1:1)
  Condicionar embedding network por f_MST
                  ↓
Fase 6 — Validação da fair transferência (LAFTR style)
  Métrica primária: accuracy de verificação para Black/African
  Comparação com baseline sem MST
```

## 3. Justificativa

### 3.1 Por que abordagem prescritiva (vs diagnóstica de v3.1)

**v3.1** propunha decompor erro Latinx em componente irredutível vs
algoritmicamente redutível — postura **diagnóstica**. O orientador
solicitou postura mais **prescritiva**: demonstrar empiricamente
que **uma intervenção específica** (MST como sinal condicionante)
produz melhora mensurável em fairness, e que essa melhora
**transfere** para face recognition.

Vantagens da v3.2:

- **Output operacional**: pipeline reproduzível e aplicável.
- **Critério de sucesso mais claro**: melhora medida em métricas
  concretas, não apenas observação descritiva.
- **Maior impacto prático**: classifier com MST conditioning é
  **artefato deployável**, não apenas análise post-hoc.

### 3.2 Por que MST especificamente (não Fitzpatrick)

- **Schumann et al. NeurIPS 2023**: MST tem distribuição balanceada
  entre tons claros e escuros (não cluster artificial em "perceived
  White" como Fitzpatrick).
- **Origem purpose-built**: MST foi desenhado especificamente para
  ML fairness research (Monk + Google), não dermatologia clínica.
- **API + dataset públicos**: `skintone.google` + MST-E facilitam
  Fase 1 do pipeline sem necessidade de treinar classifier MST do
  zero.
- **Fitzpatrick é apenas referência histórica** ([[fitzpatrick_1988]]):
  foi criada para dosimetria PUVA em 1975, não para fairness em ML.

### 3.3 Por que FiLM como mecanismo de condicionamento

- **Perez et al. AAAI 2018**: provem que feature-wise linear
  modulation é **simples, eficiente e expressivo** — γ_i, β_i têm
  dimensão C (não C×H×W).
- **Compatível com ConvNeXt-T**: FiLM atua após normalização e
  antes de não-linearidade, ortogonal à escolha de backbone.
- **Parameter-efficient**: overhead < 1% sobre ConvNeXt-T (28M
  params).
- **Ablação trivial**: comparar accuracy/fairness **com vs sem**
  FiLM, mesma arquitetura — isolamento limpo do efeito do
  conditioning.

### 3.4 Por que estender a face recognition

- **Madras et al. LAFTR ICML 2018**: prova teórica + demonstração
  empírica de **fair transferência** — representação fair em uma
  tarefa permanece fair em tarefas downstream.
- **Linha do orientador**: aplicar o mesmo pipeline a face
  recognition operacionalmente verifica se **o ganho em fairness
  generaliza** ou é específico de classification.
- **Datasets disponíveis**: RFW (4 etnias, ICCV 2019) ou BFW (4×2
  raça × gênero, CVPRW 2020) — ambos já catalogados em nosso corpus.
- **Métrica concreta**: accuracy de verificação para Black/African
  é **diretamente comparável** com baselines existentes (NIST FRVT
  8280 documenta 10-100× FPR differentials em sistemas comerciais).

## 4. Hipóteses falsificáveis

A tese é defensável **se** as seguintes hipóteses forem confirmadas
quantitativamente. Cada uma tem critério binário de aceitação/
refutação.

| ID | Hipótese | Critério de confirmação | Critério de refutação |
|---|---|---|---|
| **H1** | Pipeline MST→FiLM→ConvNeXt-T supera baseline ResNet-34 em F1 macro **≥ +2 pp** E reduz DR (max/min F1) **≥ 20%** | Ambos satisfeitos simultaneamente | Qualquer um abaixo do threshold |
| **H2** | ConvNeXt-T vanilla (sem MST) ganha **+2 a +5 pp** F1 macro sobre ResNet-34; Latinx F1 permanece **≈ 60% (±3 pp)** | Ganho no range E Latinx invariante | Ganho fora do range OU Latinx muda >±3 pp |
| **H3** | Matriz P(MST \| race) revela spread Latinx em **≥ 5 categorias MST** com sobreposição forte com White, Middle East, Indian | Spread ≥ 5 com pico distribuído | Spread < 5 ou pico concentrado |
| **H4** | **≥ 50%** das misclassificações Latinx→outras classes do baseline vanilla estão em zonas MST de sobreposição | %_overlap ≥ 50% | %_overlap < 50% (sugere erro algoritmico, não fenotípico) |
| **H5** | Pipeline análogo aplicado a RFW (ou BFW) **face recognition** melhora accuracy em Black/African em **≥ +3 pp** sobre baseline sem MST | Ganho ≥ 3 pp | Ganho < 3 pp ou negativo |

**H1, H4 e H5 são hipóteses CENTRAIS:**

- H1 — sustenta a tese prescritiva (intervenção funciona em race classification).
- H4 — sustenta a interpretação fenotípica (não é apenas mais um trick algorítmico).
- H5 — sustenta a transferência para face recognition (linha do orientador).

**Plano se refutadas**: ver §6.4.

## 5. Contribuições originais declaradas

| # | Contribuição | Originalidade | Ancoragem |
|---|---|---|---|
| **C1** | Pipeline operacional MST → FiLM → race classifier | **Primeira instância** documentada em race classification 7-class | [[perez_2018]] (FiLM), [[zemel_2013]] (FRL paradigma) |
| **C2** | Matriz P(MST \| race) sobre FairFace val + análise overlap | **Zero precedentes** públicos em fairness/biometria | [[schumann_2023]] (MST protocolo); Draelos 2025 é dermatologia |
| **C3** | Demonstração empírica de fair transferência race classification → face recognition | **Primeira aplicação** documentada em FairFace + RFW/BFW | [[madras_2018]] (LAFTR transferência teórica) |
| **C4** | Triangulação DR + worst-class F1 + CV como métrica padrão multi-classe | **Combinação proposta** | [[kleinberg_2017]] (impossibilidade); [[hardt_2016]] (EO/EOD base) |
| **C5** | Quantificação do componente fenotípico vs algorítmico do erro Latinx | **Diagnóstico inédito** combinando C1+C2 | [[fuentes_2019]], [[lewontin_1972]] (fundamento teórico) |

## 6. Escopo e limitações

### 6.1 Dentro do escopo

- **Classification**: race em 7 classes (FairFace) — in-domain (train→val).
- **Face recognition**: RFW OU BFW (a decidir após Cap 1 — RFW tem
  4 etnias compatíveis; BFW tem 4×2).
- **Backbone**: ResNet-34 (baseline) + **ConvNeXt-T** (modernização).
- **Mitigação alternativa para baseline comparativo**: FSCL+ (Park
  2022), Group DRO + strong ℓ2 (Sagawa 2020), Adversarial debiasing
  (Zhang 2018).
- **Conditioning**: **FiLM** (Perez 2018) é mecanismo escolhido.
- **Métrica MST**: 10 pontos via Google API + validação manual em subset.
- **Métricas**: F1 macro + accuracy + **triangulação DR + worst-class
  F1 + CV** + EO_h/EOD por classe.
- **Multi-seed**: **3 seeds (42, 1, 2)** com média ± std + comparação
  pareada.

### 6.2 Fora do escopo (declarado)

- **Continuous demographic labels** ([[neto_2025]]): mencionar como
  direção futura, não implementar.
- **Re-anotação racial completa do FairFace**: apenas validação MST
  em subset.
- **Cross-dataset 4-class race classification** (RFW/BFW como
  classification, não verification): taxonomias incompatíveis.
- **Fairness em gender ou age** isoladamente: foco em race.
- **Multi-task simultâneo** (race × gender × age): single-task.
- **Backbones acima de ConvNeXt-T** (ViT-L, DINOv2): foco em
  Pareto-eficiência.
- **VLM fine-tuning** (FaceScanPaliGemma): citado como SOTA, não
  replicado.
- **Geração** (fair face generation): grupo S10 mantido em standby.
- **Dimensão temporal / longitudinal**: single-snapshot.

### 6.3 Limitações estruturais (não-elimináveis)

- **Taxonomia 7-class é socialmente construída** ([[fuentes_2019]],
  [[lewontin_1972]]) — qualquer "fairness" opera sobre categoria
  socio-política, não biológica.
- **Anotação MTurk do FairFace sem κ por classe** ([[Q01]]) —
  ruído de label sub-estimado.
- **In-domain evaluation** — generalização para outros datasets é
  limitada por incompatibilidade taxonômica.
- **Impossibilidade Kleinberg-Mullainathan-Raghavan** ([[kleinberg_2017]]):
  não é possível satisfazer simultaneamente calibration,
  balance-positive e balance-negative. Tese opta por DR + EO_h
  conscientemente.

### 6.4 Plano B (se hipóteses refutadas)

| Hipótese refutada | Plano B |
|---|---|
| **H1 falha** (pipeline não melhora) | Manter C2 (matriz) + C4 (triangulação) + C5 (decomposição). Reframing como **observação negativa**: "MST conditioning não melhora — implica que erro Latinx é majoritariamente algorítmico" |
| **H2 falha** (Latinx muda com backbone modernizado) | Achado interessante: contradiz hipótese fenotípica. **Aprofundar análise**: por que ConvNeXt-T move Latinx? Eventual reformulação parcial |
| **H3 falha** (Latinx não tem spread MST amplo) | Refuta hipótese central de sobreposição fenotípica. Reframing: Latinx tem outras dimensões fenotípicas (forma craniana, features faciais) — direção futura |
| **H4 falha** (overlap MST não explica misclassificações) | Refuta interpretação fenotípica. Manter C1+C4+C5 como contribuição metodológica; reformular C2+C3 como achado negativo |
| **H5 falha** (face recognition não melhora) | Limitação de transferência. Reframing: pipeline funciona em classification mas não em recognition; abre questão de research futura |

**Observação científica importante**: tese permanece publicável
mesmo se H1, H3, H4 ou H5 forem refutadas — observações negativas
são cientificamente válidas e raramente publicadas.

## 7. Conexão com plano experimental

Esta tese é executada em **3 capítulos experimentais + síntese**:

### 7.1 Capítulo 1 — Classifier MST e matriz MST × race
(~4 semanas)

- Treinar/validar classifier MST sobre MST-E + Casual Conversations.
- Aplicar sobre FairFace val (10 954 imgs).
- Validação manual em subset 500–700 imgs × 3 anotadores
  regionalmente diversos (protocolo Schumann 2023).
- Reportar matriz P(MST | race) + κ de Fleiss por classe.
- **Testar H3**.

### 7.2 Capítulo 2 — Race classifier condicionado por MST
(~10–12 semanas)

- **Baseline ResNet-34** sem MST (validação cruzada do 72%
  reportado em AlDahoul/Lin).
- **Vanilla ConvNeXt-T** sem MST (teste de H2).
- **Pipeline ConvNeXt-T + FiLM** condicionado por MST (teste de H1).
- **Baselines comparativos**:
  - FSCL+ (Park 2022) adaptado multi-classe.
  - Group DRO + strong ℓ2 (Sagawa 2020).
  - Adversarial debiasing (Zhang 2018).
- 3 seeds casados, comparação pareada com paired t-test.
- Cross-reference matriz × confusion matrix → **testar H4**.

### 7.3 Capítulo 3 — Extensão a face recognition
(~6 semanas)

- Selecionar dataset (RFW preferível por escala e história; BFW
  secundário se BFW for mais simples).
- Aplicar pipeline análogo (encoder + FiLM condicionado por MST).
- Comparar com/sem MST conditioning em verification 1:1.
- Métrica primária: TAR @ FAR fixo por raça, foco em **Black/African**.
- **Testar H5**.

### 7.4 Síntese — Decomposição e conclusões
(~4 semanas + escrita)

- Joint analysis das 5 hipóteses.
- Decomposição final: erro_total = irredutível_fenotípico +
  redutível_algorítmico, por classe.
- Implicações para deployment, ética, e direções futuras.

### 7.5 Cronograma estimado

| Bloco | Duração | Marco |
|---|---|---|
| Setup metodológico (02, 03, 08) | 2 semanas | Especificações executáveis |
| Cap 1 (Classifier MST + matriz) | 4 semanas | H3 testada |
| Cap 2 (Race com MST conditioning) | 10–12 semanas | H1, H2, H4 testadas |
| Cap 3 (Face recognition) | 6 semanas | H5 testada |
| Síntese | 4 semanas | Decomposição final |
| Escrita capítulos (paralelo) | 8–12 semanas | Defesa |
| **Total estimado** | **~28–32 semanas** | **~Jan–Mar 2027** |

## 8. Status do thesis statement

- **v1**: histórico (pre-pivot 25/05/2026, framing MBA-evolution).
- **v2**: histórico (versão 2.0 de 23/05, ainda framing MBA).
- **v3**: histórico (pós-pivot 25/05 — diagnóstica empírica).
- **v3.1**: histórico (06/01 — fundamentação teórica AAPA + Lewontin).
- **v3.2 (este arquivo)**: **prescritiva** — pós-reunião 04/06 + Rodada 5.

**Status:** **PRONTO PARA REVISÃO PELO ORIENTADOR** (próxima reunião).

## 9. Documentos relacionados (estado pós-reunião 2026-06-04)

- [`05_landscape.md`](05_landscape.md): síntese transversal do
  corpus (em v3.1 — pendente nota sobre Track G).
- [`06_gap.md`](06_gap.md): ranqueamento das 5 frentes 🔬 (em v3.1
  — Q04 + Q10 viram pipeline único em v3.2; ranqueamento mantém).
- [`04_pesquisa_bibliografica/_perguntas.md`](04_pesquisa_bibliografica/_perguntas.md):
  Q01–Q14 respondidas, 5 frentes 🔬 abertas.
- [`04_pesquisa_bibliografica/_metricas_corpus.md`](04_pesquisa_bibliografica/_metricas_corpus.md):
  big numbers do corpus para reuniões.
- [`04_pesquisa_bibliografica/_triagem.md`](04_pesquisa_bibliografica/_triagem.md):
  log de decisões editoriais R1–R5 + R2.5/R2.6.

## 10. Pendências pós-aprovação v3.2

- [ ] Atualizar `06_gap.md` para refletir pipeline integrado Q04+Q10
      (em v3.1, eram capítulos paralelos).
- [ ] Adicionar nota em `05_landscape.md` sobre Track G (Rodada 5).
- [ ] Regenerar PPTX para v3.2 (script `_gerar_apresentacao.py`
      arquivado em `docs/historico/reuniao_2026-06/`; replicar e
      ajustar para nova tese).
- [ ] Criar `02_metodologia.md`, `03_metricas.md`, `08_experimentos.md`
      com especificações detalhadas dos 3 capítulos experimentais.
