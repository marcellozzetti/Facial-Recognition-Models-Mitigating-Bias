# Validação científica do pipeline v3.2

> **Frente aberta:** Rodada 6 da Pesquisa Bibliográfica.
> **Data:** 2026-06-06.
> **Objetivo:** verificar, etapa por etapa, se cada decisão do pipeline
> v3.2 tem embasamento científico, ou identificar papers que **refutam**
> a abordagem.
> **Metodologia:** mapeamento pipeline-step × evidência, buscando
> EXPLICITAMENTE tanto reforço quanto refutação.

---

## Sumário executivo

Foram investigadas as **5 áreas críticas** levantadas após a v3.2:

| # | Área | Status | Risco identificado |
|---|---|---|---|
| 1 | Condicionamento por atributo auxiliar para fairness | ✓ embasada | Mecanismos alternativos competitivos |
| 2 | MST aplicado a auditoria de FairFace | ⚠️ overlap parcial | **SkinToneNet (Pereira 2026) chegou perto** |
| 3 | Transferência empírica fair clf → fair FR | ✓ gap confirmado | LAFTR é só teórico; ninguém demonstrou em FR |
| 4 | Backbone leve (ConvNeXt-T) para skin tone | ⚠️ não-benchmarkado | Risco de revisor: por que não ViT-Small? |
| 5 | Disparidade Black/African em FR | ⚠️ enquadramento | **Pangelinan 2023 refuta skin-tone como causa primária** |

**Conclusão geral:** o pipeline v3.2 **tem base científica suficiente**
para todas as 6 etapas, mas **dois pontos precisam de resposta direta**
ao orientador antes de defender a tese:

1. **Reenquadrar H5** para reconhecer que skin tone é UM dos fatores
   causadores de disparity em FR, não A causa única (Pangelinan 2023).
2. **Citar e diferenciar de SkinToneNet** (Pereira 2026), que auditou
   FairFace mas **não publicou a matriz MST × race** que é nossa C2.

Foram identificados **9 papers novos** que devem entrar no corpus
(Rodada 6) — listados na Seção 7.

---

## 1. Etapa 1 — Classificador MST treinado em MST-E + Casual Conversations

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| Usar escala MST e não Fitzpatrick | Schumann 2023, Hazirbas 2021, Buolamwini 2018 | ✓ sólido | Nenhuma refutação encontrada |
| Treinar em MST-E + Casual Conversations | Schumann 2023 (MST-E); Hazirbas 2021 (CC) | ⚠️ parcial | **Pereira 2026 usa STW (42k imagens) — alternativa nova** |
| Backbone ConvNeXt-T | — | ❌ ausente | **SkinToneNet usa ViT-Small fine-tuned** |
| Validação por anotadores em Prolific | Schumann 2023 (protocolo de consenso) | ✓ sólido | — |

### Papers novos identificados

- **Pereira et al. 2026** — *Large-Scale Dataset and Benchmark for Skin
  Tone Classification in the Wild* (arXiv:2603.02475). Propõe **SkinToneNet**
  (ViT-Small fine-tuned) e o dataset STW (42,313 imagens, 3,564 indivíduos,
  MST 10-classe). **Risco:** se ignorarmos, revisor questionará a
  escolha do backbone e o tamanho da base de treino.

- **Lu 2025** — *TrueSkin: Towards Fair and Accurate Skin Tone Recognition
  and Generation* (arXiv:2509.10980). 7,299 imagens em **6 classes**
  (não 10 MST), benchmarka LMMs. **Não compete diretamente** com nossa
  abordagem MST 10-classe.

### Resposta aos riscos

**RISCO:** Pereira 2026 já tem o classificador MST treinado em escala maior.

**RESPOSTA:** Nossa contribuição NÃO é o classificador MST em si — é o
**uso dele como sinal auxiliar para condicionar classificação racial via
FiLM**. SkinToneNet é INSUMO, não competidor. **Devemos avaliar usar
SkinToneNet pré-treinado em vez de treinar do zero** — economiza
~3 semanas de trabalho experimental e o STW dataset é maior que
MST-E + CC.

---

## 2. Etapa 2 — Auditoria FairFace: matriz pública MST × raça

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| Auditar FairFace via classificador MST automático | Dominguez 2024 (DSAP); Schumann 2023 | ✓ embasado | — |
| Publicar matriz tom × raça (Contribuição C2) | — | ⚠️ overlap parcial | **Pereira 2026 auditou MAS não publicou matriz cruzada** |
| Validação humana sobre subset (700 imgs × 3 anotadores) | Schumann 2023 (protocolo consenso 27% subjetividade) | ✓ sólido | Neto 2025 questiona discretização |

### Papers novos identificados

- **Pereira et al. 2026** (mesmo paper acima) — auditou FairFace usando
  SkinToneNet e reportou que **"FairFace exibe alta ausência de classes
  MST 6-10"**. **Crítico:** publicou distribuições agregadas por dataset,
  **MAS NÃO publicou cross-tabulation MST × race**. Nossa C2 segue
  original.

### Resposta aos riscos

**RISCO:** Pereira 2026 já fez a auditoria geral.

**RESPOSTA:** A diferença é **granularidade**. Pereira reporta
"FairFace tem subrepresentação de MST 6-10 globalmente". Nossa C2 vai
um passo além: "MST 6-10 está concentrado em quais raças? Qual o spread
MST por raça? Onde Latinx se distribui?". A **matriz cruzada** é
mecanismo causal para H3 e H4 — não duplica Pereira.

**Ação:** citar Pereira como base, declarar diferencial em C2.

---

## 3. Etapa 3 — Race classifier com tom como contexto (FiLM)

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| FiLM como mecanismo de condicionamento | Perez 2018 | ✓ sólido | — |
| Usar skin tone como sinal auxiliar para race | — | ⚠️ original | Sem precedente direto |
| Backbone ConvNeXt-T para race classifier | AlDahoul 2024/26 (precedente leve) | ⚠️ parcial | **Dooley 2022: arquitetura por si só importa** |
| Treinar multi-task (skin tone + race) como alternativa | — | ⚠️ a investigar | Raumanns 2024 (mistos) |

### Papers novos identificados

- **Liu et al. 2025 (FAccT)** — *Component-Based Fairness in Face Attribute
  Classification with Bayesian Network-informed Meta Learning*
  (arXiv:2505.01699). Mecanismo **BNMR (sample reweighting via meta-learning
  Bayesiano)**. **NÃO usa FiLM, NÃO usa skin tone**. Compete como
  baseline alternativo. **Mecanismo ortogonal ao nosso.**

- **Aguirre & Dredze 2023/24** — *Transferring Fairness using Multi-Task
  Learning with Limited Demographic Information* (arXiv:2305.12671).
  **Prova empírica** de que "demographic fairness objectives transfer
  fairness within a multi-task framework". **Reforça a viabilidade
  teórica** de usar skin tone como tarefa auxiliar.

- **Ramachandran & Rattani 2024 (IJCB)** — *A Self-Supervised Learning
  Pipeline for Demographically Fair Facial Attribute Classification*
  (arXiv:2407.10104). Baseline competitivo em FairFace + CelebA, mas
  **não usa skin tone como sinal**.

- **Raumanns et al. 2024 (FAIMI EPIMI)** — *Dataset Distribution Impacts
  Model Fairness: Single vs. Multi-Task Learning* (arXiv:2407.17543).
  Achado crítico: "reinforcement multi-task approach does NOT remove sex
  bias" e "adversarial scheme eliminates bias in some cases". **Sugere
  que multi-task naive é INSUFICIENTE e adversarial pode ser superior**.
  **Implica que FiLM-conditioning precisa ser comparado contra
  adversarial debiasing como baseline forte.**

### Resposta aos riscos

**RISCO 1:** Ninguém ainda fez FiLM+MST→race.

**RESPOSTA:** Esse é o ponto de **originalidade** da tese. A composição
é justificada por: (a) FiLM funciona como mecanismo (Perez 2018),
(b) transferência de fairness via multi-task funciona (Aguirre 2023),
(c) skin tone é a dimensão certa para race (Schumann 2023). A composição
das três é nossa contribuição.

**RISCO 2:** Multi-task naive pode falhar (Raumanns 2024).

**RESPOSTA:** Nossa abordagem **NÃO é multi-task naive** — é condicionamento
explícito via FiLM com skin tone como contexto, não como output paralelo.
Mas devemos adicionar **adversarial debiasing (Zhang 2018)** como baseline
forte, já contemplado no plano de baselines.

**RISCO 3:** Dooley 2022 mostra que arquitetura por si só dirige fairness.

**RESPOSTA:** ConvNeXt-T já é nossa escolha arquitetural moderna. **H2 da
nossa tese** explicita: "trocar ResNet-34 por ConvNeXt-T por si só NÃO
resolve Latinx". Se Dooley estiver certo, H2 será refutada e a contribuição
arquitetural se torna parte do efeito. Esse risco já está coberto.

---

## 4. Etapa 4 — Comparação de fairness COM vs SEM condicionamento

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| Baseline FSCL+ (Park 2022) | Park 2022 (ficha) | ✓ sólido | — |
| Baseline Group DRO (Sagawa 2020) | Sagawa 2020 (ficha) | ✓ sólido | — |
| Baseline FineFACE (Manzoor 2024) | Manzoor 2024 (ficha) | ✓ sólido | — |
| Baseline Adversarial (Zhang 2018) | Zhang 2018 (ficha) | ✓ sólido | — |
| Reportar F1 macro + DR + worst-class simultaneamente | Kleinberg 2017 (impossibility); Hardt 2016 (EO) | ✓ sólido | — |
| 3 seeds para significância | Política do projeto + Lafargue 2025 (uncertainty-aware) | ✓ sólido | — |

### Achado crítico

- **Kolla & Savadamuthu 2022 (WACVW)** — *The Impact of Racial Distribution
  in Training Data on Face Recognition Bias: A Closer Look*
  (arXiv:2211.14498). **Reforça a necessidade de mecanismo
  arquitetural** além de balanceamento: *"a uniform distribution of races
  in the training datasets alone does not guarantee bias-free face
  recognition"*. Como FairFace É balanceado e o gap Latinx persiste,
  isso reforça a tese de que precisamos de intervenção arquitetural
  (FiLM-conditioning) E não só de dados balanceados.

### Status

**Sem refutação. Plano experimental robusto.** Vale adicionar Kolla 2022
como motivação adicional no Cap 2.

---

## 5. Etapa 5 — Aplicar pipeline a face recognition (RFW ou BFW)

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| Estender pipeline a FR | Madras 2018 (LAFTR, teórico) | ⚠️ só teórico | **Empiricamente ainda não demonstrado em FR** |
| Métrica TAR @ FAR fixo por raça | Grother 2019 (NIST) | ✓ sólido | — |
| Esperar transferência empírica de fairness | Aguirre 2023 (em texto, não FR) | ⚠️ parcial | **Pangelinan 2023: pixel info > skin tone em FR** |
| Foco em Black/African como métrica de sucesso | Grother 2019; Wang 2019 (RFW) | ✓ sólido | — |

### Papers novos identificados

- **Pangelinan et al. 2023** — *Exploring Causes of Demographic Variations
  In Face Recognition Accuracy* (arXiv:2304.07175). **ACHADO QUE
  POTENCIALMENTE REFUTA H5**: "demographic differences in face PIXEL
  INFORMATION of the test images appear to most directly impact the
  resultant differences in face recognition accuracy". Sugere que para
  FR (verificação 1:1), skin tone é secundário a image quality.

- **Dooley et al. 2022/23** — *Rethinking Bias Mitigation: Fairer
  Architectures Make for Fairer Face Recognition* (arXiv:2210.09943).
  Argumenta que **biases são inerentes a arquiteturas**. Pareto-domina
  baselines de mitigação. Fortalece a importância de ConvNeXt-T.

- **Aguirre & Dredze 2023/24** (já citado em §3) — fortalece o
  princípio de fair transfer via multi-task.

### Resposta aos riscos

**RISCO CRÍTICO:** Pangelinan 2023 sugere que para FR, pixel info > skin
tone. Isso pode refutar H5 empiricamente.

**RESPOSTA TÉCNICA:**

1. Pangelinan trabalha com **verificação 1:1** em FR. Nossa Cap 1-2
   trabalha com **classificação 7-class de raça** — tarefa diferente.
2. Para Cap 3 (FR), aceitar explicitamente que skin tone é UM fator,
   não O fator. **Reformular H5 mais cuidadosamente:**
   - Versão atual: "A propriedade fair transfere para FR; Black/African
     melhora ≥+3pp"
   - **Versão revisada (proposta):** "A representação aprendida com
     condicionamento por MST mantém ou melhora as métricas de fairness
     em downstream FR, ESPECIALMENTE quando combinada com pré-processamento
     de qualidade da imagem (face crop + alinhamento). A magnitude
     de melhora em Black/African é proporcional à diferença pixel-info
     residual após o alinhamento."
3. **Adicionar etapa de quality control de imagem** no Cap 3 — não estava
   no plano original.

**STATUS:** H5 precisa de reformulação para resistir à refutação de
Pangelinan.

---

## 6. Etapa 6 — Medir se Black/African melhora especificamente

### Decisões de design avaliadas

| Decisão | Base no corpus | Status | Refutação? |
|---|---|---|---|
| Focar em Black/African como métrica de sucesso | Buolamwini 2018; Grother 2019; NIST | ✓ sólido | — |
| Worst-class F1 como métrica central | Sagawa 2020 (Group DRO) | ✓ sólido | — |
| TAR @ FAR fixo por subgrupo | Grother 2019 (NIST FRVT) | ✓ sólido | — |

### Achado adicional

- **Treinamento skewed-toward-African pode ser mais eficaz que balanceado**
  (resultado citado mas não com referência direta na busca; Robinson 2020
  toca em threshold adaptativo). **Vale buscar referência direta para
  esta afirmação.**

### Status

**Sem refutação direta.** Mas vale documentar que Black/African não é o
único subgrupo vulnerável — Latinx também aparece. **O foco específico
em Black/African deve ser justificado** pelo histórico industrial (NIST,
Microsoft pós-Gender Shades) e não como hipótese científica isolada.

---

## 7. Papers a adicionar ao corpus na Rodada 6

Em ordem de prioridade:

| # | Paper | Área | Por que adicionar |
|---|---|---|---|
| 1 | **Pangelinan et al. 2023** (arXiv:2304.07175) | 5 | Refutação potencial de H5; precisa ser endereçada |
| 2 | **Pereira et al. 2026** (arXiv:2603.02475) — SkinToneNet/STW | 1, 2 | Estado-da-arte em MST classifier; auditou FairFace antes de nós |
| 3 | **Dooley et al. 2022/23** (arXiv:2210.09943) | 5 | Fairness arquitetural em FR; valida ConvNeXt-T |
| 4 | **Aguirre & Dredze 2023/24** (arXiv:2305.12671) | 1, 3 | Multi-task fair transfer empírico |
| 5 | **Kolla & Savadamuthu 2022** (arXiv:2211.14498), WACVW | 4, 5 | Distribuição balanceada não basta |
| 6 | **Liu et al. 2025 FAccT** (arXiv:2505.01699) — BNMR | 1, 3 | Baseline competitivo recente |
| 7 | **Ramachandran & Rattani 2024 IJCB** (arXiv:2407.10104) | 1, 3 | SSL fair attribute clf, baseline |
| 8 | **Raumanns et al. 2024 FAIMI EPIMI** (arXiv:2407.17543) | 3 | Cautela contra multi-task naive |
| 9 | **Lu 2025** (arXiv:2509.10980) — TrueSkin | 1 | Alternativa de dataset; não compete diretamente |

---

## 8. Riscos críticos consolidados

### Risco A — Pangelinan 2023 ameaça H5

- **Natureza:** skin tone pode não ser causa primária em FR (pixel info domina)
- **Impacto:** H5 pode ser refutada em todos os cenários
- **Mitigação:** reformular H5 para considerar pixel info como confounder;
  adicionar etapa de quality control no Cap 3
- **Status:** decisão pendente — reformulação proposta na Seção 5

### Risco B — Sobreposição com SkinToneNet (Pereira 2026)

- **Natureza:** trabalho concorrente já auditou FairFace e tem classifier MST
- **Impacto:** C1 (classificador MST) deixa de ser contribuição se usarmos
  SkinToneNet pré-treinado
- **Mitigação 1:** usar SkinToneNet, focar contribuição em FiLM-conditioning
- **Mitigação 2:** treinar próprio classificador MST com STW + MST-E + CC
  e reportar SOTA
- **Status:** decisão pendente — recomendo Mitigação 1 (acelera cronograma)

### Risco C — Mecanismo FiLM + MST nunca foi testado para race classification

- **Natureza:** sem precedente direto na literatura
- **Impacto:** pode ser originalidade ou pode esconder problema técnico
- **Mitigação:** rodar piloto com FairFace subset (~5,000 imagens) antes
  de plano experimental completo
- **Status:** plano original já contempla isso na primeira fase

### Risco D — Adversarial debiasing pode superar FiLM-conditioning

- **Natureza:** Raumanns 2024 mostra adversarial > multi-task naive
- **Impacto:** se adversarial > FiLM-MST, contribuição central enfraquece
- **Mitigação:** já contemplado — adversarial debiasing (Zhang 2018) está
  como baseline forte. **Aceitar se for o caso**.

### Risco E — Justificar ConvNeXt-T sem benchmark direto

- **Natureza:** ConvNeXt-T não foi benchmarkado para skin tone prediction
- **Impacto:** revisor pode pedir comparação com ViT-Small (SkinToneNet)
- **Mitigação:** rodar sub-experimento ConvNeXt-T vs ViT-Small como parte
  do Cap 1

---

## 9. Decisões — status atualizado 2026-06-06

### 9.1 H5 — Reformular? ⏸️ PENDENTE — DISCUTIR NA REUNIÃO DE SEGUNDA

- **Status:** decisão adiada para discussão com orientador (Prof. Quiles).
- **Material para a reunião:** ver Seção 11 abaixo (alternativas
  formuladas com pros/contras de cada).
- **Versões em mesa:**
  - V1 (atual, otimista): "fairness transfere para FR; Black/African ≥+3pp"
  - V2 (cautelosa): "transferência condicionada a quality control de pixel info"
  - V3 (separada): manter H5 atual + adicionar H6 (controle de pixel info)

### 9.2 Classificador MST — SkinToneNet ✅ DECIDIDO (recomendação para orientador)

- **Decisão:** **usar SkinToneNet pré-treinado** (Pereira 2026) como
  insumo do pipeline.
- **Justificativa:**
  - Economiza ~3 semanas de cronograma (treino + validação manual).
  - SkinToneNet já é SOTA em skin-tone-in-the-wild (treinado no STW
    com 42k imagens, validado em 6 datasets out-of-domain).
  - Contribuição central da tese passa a ser **FiLM-conditioning +
    matriz MST × race**, não o classificador MST em si.
- **Risco assumido:** dependência externa de um modelo de pesquisa
  recente (preprint mar/2026). Mitigação: verificar disponibilidade de
  pesos/código antes do início do Cap 1; se não disponível, treinar
  próprio com mesma arquitetura (ViT-Small).
- **Comunicar ao orientador:** registrar como decisão informada (não
  é decisão para o orientador aprovar — é nossa escolha técnica
  baseada na evidência da Rodada 6).

### 9.3 Adversarial debiasing como baseline forte ✅ DECIDIDO

- **Decisão:** **adicionar Adversarial debiasing (Zhang 2018) à lista
  de baselines.**
- **Justificativa:** Raumanns 2024 mostra empiricamente que adversarial
  > multi-task naive para fairness. Sem esse baseline, revisor pode
  questionar se FiLM-conditioning não está sendo comparado contra
  alternativa fraca.
- **Lista final de baselines:**
  1. **ResNet-34** (Karkkainen 2021) — baseline histórico do FairFace
  2. **ConvNeXt-T puro (sem FiLM)** — controle arquitetural; testa H2
  3. **FSCL+** (Park 2022) — fair supervised contrastive
  4. **Group DRO** (Sagawa 2020) — worst-group robust
  5. **FineFACE** (Manzoor 2024) — cross-layer attention fair
  6. **Adversarial debiasing** (Zhang 2018) — **NOVO** — adversarial classifier
  7. **BNMR** (Liu 2025 FAccT) — opcional, paper muito recente
- **Impacto no cronograma:** Adversarial debiasing tem implementação
  pública (AIF360 / código próprio do paper); estimar +1-2 semanas
  no Cap 2.

### 9.4 Datasets MST de treino — N/A após decisão 9.2

- **Status:** decisão obsoleta após escolha de SkinToneNet pré-treinado.
- **Por que a pergunta existia:** se treinássemos nosso próprio
  classificador MST, precisaríamos escolher entre MST-E (1.5k imgs),
  Casual Conversations (45k frames), STW (42k imgs) ou combinação.
- **Por que não importa mais:** SkinToneNet já vem treinado no STW. Os
  outros datasets entram apenas em **validação externa** opcional do
  Cap 1, não no treino.
- **Sub-decisão técnica que pode esperar:** se em algum momento for
  necessário fine-tunar o SkinToneNet em dados adicionais (ex: melhor
  generalização para faces brasileiras), avaliar então qual fonte
  usar. Por enquanto: deixar como veio.

---

## 10. Próximas ações (atualizadas)

1. **Esta semana** — Criar fichas R6 para os 8 papers aprovados
2. **Esta semana** — Atualizar `06_gap.md` com Risco A explicitamente endereçado
3. **Esta semana** — Preparar slide específico para discussão de H5 (Seção 11)
4. **Reunião de segunda** — Apresentar reformulação proposta de H5;
   informar decisão sobre SkinToneNet; confirmar adversarial baseline
5. **Pós-reunião** — Atualizar `07_thesis_statement.md` v3.2 → v3.3
   com H5 reformulada conforme decisão do orientador
6. **Pós-aprovação** — Detalhar `02_metodologia.md` com:
   - SkinToneNet integrado como insumo
   - 6 baselines incluindo Adversarial debiasing
   - Sub-experimento ConvNeXt-T vs ViT-Small no Cap 1

---

## 11. Material para discussão de H5 na reunião de segunda (08-jun-2026)

### 11.1 H5 atual (v3.2)

> A representação aprendida com condicionamento por MST mantém ou melhora
> as métricas de fairness em downstream face recognition.
> **Critério de sucesso:** Black/African melhora ≥+3pp em RFW ou BFW.

### 11.2 A refutação que motiva reformular

**Pangelinan et al. 2023** (arXiv:2304.07175, autores Notre Dame/Florida Tech):
*"demographic differences in face PIXEL INFORMATION of the test images
appear to most directly impact the resultant differences in face
recognition accuracy."*

Em outras palavras: **para FR (verificação 1:1), o fator dominante de
disparity é qualidade/quantidade de pixels de face, não tom de pele
direto**. Como nossa H5 assume que skin tone-conditioning vai melhorar
FR, esse achado é uma ameaça direta.

### 11.3 Embasamento teórico que SUSTENTA H5

| Fonte | O que dá apoio à H5 |
|---|---|
| **Madras 2018 (LAFTR)** | Prova teórica de fair transferência via representação |
| **Aguirre & Dredze 2023** | Empírico em NLP: multi-task fair transfer FUNCIONA |
| **Kotwal & Marcel 2025** (survey FR fairness) | Tom de pele é citado como dimensão relevante |
| **Buolamwini 2018** | Tom de pele × gênero é causa documentada de disparity |
| **Dooley 2022/23** | Arquitetura impacta fairness em FR; mecanismo arquitetural funciona |

### 11.4 Alternativas de reformulação (3 versões)

#### Versão A (manter, ignorar Pangelinan) — NÃO RECOMENDADA

> H5 atual sem alteração.

**Pros:** simples, hipótese clara.
**Contras:** ignora evidência conflitante; revisor pode questionar.

#### Versão B (cautelosa, controle de qualidade)

> A representação aprendida com condicionamento por MST mantém ou melhora
> as métricas de fairness em downstream face recognition **quando a
> qualidade de imagem (face crop + alinhamento) é controlada e
> equalizada entre subgrupos**.
> **Critério:** Black/African melhora ≥+3pp em RFW ou BFW, **após
> normalização de pixel info**.

**Pros:** epistemologicamente honesta; endereça Pangelinan; adiciona
contribuição menor (controle de qualidade).
**Contras:** adiciona ~2 semanas ao Cap 3 para implementar quality
control rigoroso.

#### Versão C (separar em duas hipóteses)

> **H5:** A representação aprendida com condicionamento por MST mantém
> ou melhora fairness em FR. *(Black/African ≥+3pp em RFW/BFW)*
>
> **H6 (nova):** A disparity residual em Black/African após
> condicionamento por MST é explicada predominantemente por diferenças
> de pixel info (Pangelinan 2023), não por tom de pele.
> *(Decomposição: % variance explicada por skin tone vs pixel quality)*

**Pros:** transforma a refutação em **contribuição quantitativa**
(decomposição de variância); permite que H5 seja parcialmente refutada
sem refutar a tese inteira.
**Contras:** adiciona uma hipótese ao plano; relatório de Cap 3 fica
mais complexo (mas mais rico).

### 11.5 Recomendação técnica

**Versão C** porque:
1. Transforma uma ameaça em contribuição.
2. Se Pangelinan estiver certo (pixel info > skin tone), nossa H6
   demonstra isso quantitativamente — virou achado, não falha.
3. Se Pangelinan estiver parcialmente certo, H5 e H6 capturam o
   trade-off.
4. Se Pangelinan estiver errado, H5 confirma; H6 vira nota.

**Decisão final fica com o orientador.**
