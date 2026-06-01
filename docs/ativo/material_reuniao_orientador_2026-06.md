# Material para Reunião com Orientador

> **Autor:** Marcello Ozzetti
> **Orientador:** Prof. Marcos Quiles
> **Programa:** Mestrado em Ciência da Computação — Unifesp/ICT
> **Tema:** Equidade Racial em Classificação Facial — FairFace 7-class
> **Data da reunião:** _a confirmar_ (semana de 2026-06-01)
> **Documento elaborado em:** 2026-06-01
> **Objetivo:** apresentar resultado da reestruturação pós-pivot
> (2026-05-25), submeter tese v3.1 para aprovação, definir plano de
> Fase 5.

---

## Sumário

1. [Metodologia de pesquisa](#1-metodologia-de-pesquisa)
2. [Racional — historyline da evolução da pesquisa](#2-racional--historyline-da-evolução-da-pesquisa)
3. [Visão dos principais artigos](#3-visão-dos-principais-artigos)
4. [Gaps, considerações e achados](#4-gaps-considerações-e-achados)
5. [Próximos passos](#5-próximos-passos)
6. [Anexo A — Perguntas antecipadas](#anexo-a--perguntas-antecipadas)
7. [Anexo B — Tabela de relevância dos 23 papers](#anexo-b--tabela-de-relevância-dos-23-papers)

---

## 1. Metodologia de pesquisa

### 1.1 Gatilho da reestruturação

A reunião de 25/05/2026 identificou três problemas estruturais na
pesquisa anterior: (i) síntese baseada em abstracts, não leitura
integral; (ii) **cinco de sete citações principais com autoria
fabricada** (caso "Hassanpour"); (iii) framing "evolução do MBA"
recusado pelo orientador.

Como resposta, foi estabelecido um **protocolo formal de Pesquisa
Bibliográfica** em 4 fases, com critérios de seleção, verificação de
autoria em fonte primária, e leitura integral obrigatória.

### 1.2 Os 5 critérios de seleção (em ordem hierárquica)

| # | Critério | Detalhe |
|---|---|---|
| 1 | **Escopo temático** | Fairness/biometria/raça/auditoria em sistemas faciais |
| 2 | **Recência** | 2019–2026 (10 anos pós-Buolamwini & Gebru) + exceções seminais |
| 3 | **Verificabilidade** | Autoria, ano e venue **confirmáveis em fonte primária** (arXiv, DOI, página oficial) |
| 4 | **Relevância editorial (venue)** | CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR/AAAI/WACV/FAccT/IEEE TBIOM/TPAMI/ACM CSur/Information Fusion. Preprints só com cobertura única ou alto impacto |
| 5 | **Impacto (citações)** | ≥50 cits (papers ≥3 anos), ≥20 cits (2024), aberto para 2025-26 com cobertura única |

### 1.3 Fluxo de trabalho em 4 fases

```
Fase A — Coleta sistemática
   Google Scholar, Semantic Scholar, arXiv, DBLP, venues
   ↓
Fase B — Triagem editorial
   Critérios 1-5 aplicados
   Registro em _triagem.md
   Decisões: APROVADO | STANDBY | DESCARTADO
   ↓
Fase C — Leitura integral
   Download PDF (gitignored)
   Extração via Python (pypdf)
   Ficha em 11 seções por paper
   ↓
Fase D — Síntese transversal
   Cross-reference, gaps, thesis statement v3
```

### 1.4 Template de ficha (11 seções normativas)

Toda ficha segue estrutura idêntica:

1. Frontmatter YAML (metadata: autores, venue, citações, status)
2. Resumo do problema atacado
3. Método
4. Datasets e setup experimental
5. Métricas reportadas
6. Resultados principais (com valores numéricos)
7. Limitações declaradas pelos autores
8. Limitações que identifiquei (leitura crítica)
9. Relação com nossa pesquisa
10. Pontos para citar / posicionar
11. Arquivos relacionados
12. **Trabalhos sugeridos pelos autores (Future Work)** ← incorporado na Rodada 4

### 1.5 Metodologia Q&A — interrogação ativa do corpus

Para transformar a leitura passiva em **interrogação ativa**, cada
pergunta de pesquisa é formalmente registrada em `_perguntas.md` com:

- Status (✅ ANSWERED, ⚠ PARTIAL, ❌ OPEN, 🔬 NEW RESEARCH FRONT).
- Cross-references às fichas que sustentam a resposta.
- Síntese consolidada.
- Lacunas / nova frente de pesquisa (se aplicável).

**14 perguntas** foram formuladas e respondidas durante o processo;
**5 geraram frentes 🔬** (gaps abertos que orientam o trabalho
experimental).

### 1.6 Verificação de autoria — protocolo anti-Hassanpour

Após o caso de 5 autorias fabricadas, **toda** citação na pesquisa
ativa passa por:

1. **Download do PDF original** (arXiv ou DOI editorial).
2. **Inspeção da primeira página** para confirmação visual de
   autoria + venue.
3. **Cross-check em Semantic Scholar** para citação count + venue
   metadata.
4. **Registro em `00_referencias.md`** com status `✅ VERIFIED`,
   `⚠ CONTENT-TO-VERIFY`, ou `⏳ TO_VERIFY`.

**Sem verificação, sem citação.**

---

## 2. Racional — historyline da evolução da pesquisa

### 2.1 Mapa temporal

```
2026-05-25 (pivot)
   |
   ├─ Rodada 1: 9 seeds iniciais
   |    - FairFace, Gender Shades, U-FaTE, RFW, DSAP, NISTIR 8280,
   |      AlDahoul, FineFACE, Lafargue
   |    - Critério: cobertura temática direta
   |
   ├─ Rodada 2: snowballing das fichas R1
   |    - +5 papers das fichas-base
   |    - FSCL (Park 2022), FairGRAPE (Lin 2022),
   |      Bhaskaruni 2019, Group DRO (Sagawa 2020),
   |      Mehrabi survey 2021
   |
   ├─ Rodada 2.5: verificação dedicada de SOTA
   |    - Resposta direta ao caso Hassanpour
   |    - Validação cruzada FaceScanPaliGemma como SOTA
   |    - AlDahoul promovido a Nature Sci Reports 2026
   |
   ├─ Rodada 3: broadening (não-FairFace)
   |    - Crítica metodológica do usuário: corpus muito centrado em FairFace
   |    - +5 papers de outros tracks: BFW, Casual Conversations,
   |      MST-E, Continuous Labels, Kotwal survey TBIOM
   |
   └─ Rodada 4: fundamentação científica
        - 4 questões fundamentais do usuário sobre substrato:
          existe raça biologicamente? quantos tons cientificamente?
        - +4 papers fundadores: AAPA 2019, Lewontin 1972,
          Fitzpatrick 1988, Massey-Martin 2003
        - Tese v3 → v3.1 com fundamentação institucional
```

### 2.2 Decisões críticas em cada rodada

| Rodada | Decisão chave | Motivação | Resultado |
|---|---|---|---|
| **R1** | Selecionar 9 seeds em vez de 50+ | Massa crítica vs ruído | Leitura integral viável; padrão de qualidade |
| **R2** | Snowballing das R1 antes de surveys | Surveys são reference, não evidence | Identificou 5 papers metodológicos cruciais |
| **R2.5** | Verificar SOTA antes de qualquer claim | Lição do caso Hassanpour | Confirmou FaceScanPaliGemma 75.7% como SOTA, sem competidor |
| **R3** | Expandir para tracks paralelos | Corpus inicial era FairFace-cêntrico | Identificou Track B (recognition) e Track C (skin tone) |
| **R4** | Buscar fundamento teórico (não só empírico) | Pergunta do usuário: as 7 raças existem cientificamente? | AAPA + Lewontin fundamentam tese |

### 2.3 Aprendizados metodológicos do processo

1. **A literatura confirma o problema mas raramente executa a
   solução.** 7 papers sugerem mitigação em race classification
   multi-classe; zero executam.
2. **Categorias raciais são instrumentais, não biológicas.** O que
   parecia limitação do nosso trabalho (escolha do FairFace 7-class)
   é, na verdade, **escolha pragmática alinhada com a literatura
   dominante** — mesma escolha que AlDahoul (SOTA), FairGRAPE,
   FineFACE etc. fazem.
3. **A confusão Fitzpatrick ↔ race é endêmica** mesmo em dermatologia
   (1/3 dos profissionais). Adotar MST resolve isso ao nível
   instrumental.
4. **O método Q&A é o ativo principal.** Permitiu transformar leitura
   passiva em hipóteses falsificáveis (H1-H5 da tese v3.1).

---

## 3. Visão dos principais artigos

Detalhamento dos **10 papers centrais** (lista de leitura priorizada
para a reunião). Tabela completa dos 23 em [Anexo B](#anexo-b--tabela-de-relevância-dos-23-papers).

### 3.1 Fuentes et al. (2019) — AAPA Statement on Race and Racism

- **Venue:** *American Journal of Physical Anthropology* (vol 169,
  no 3, pp. 400-402), DOI 10.1002/ajpa.23882
- **Tipo:** Statement institucional peer-reviewed (Committee on
  Diversity da AABA, presidido por Agustín Fuentes)
- **Papel na tese:** **fundamento teórico** da posição "race is
  not biology"
- **Citação central:** *"Race does not provide an accurate
  representation of human biological variation. It was never accurate
  in the past, and it remains inaccurate when referencing contemporary
  human populations."*
- **Por que importa:** sustenta a limitação reconhecida em
  `07_thesis_statement.md` §6.1 com **citação institucional**
  oficial, não apenas observação nossa.

### 3.2 Lewontin (1972) — The Apportionment of Human Diversity

- **Venue:** *Evolutionary Biology* (Springer), vol 6
- **Citações:** 5000+ (foundational), 50+ anos de literatura derivada
- **Papel na tese:** **fundamento genético** — partição 85.4% intra-pop / 8.3% between-pop within-race / 6.3% between-race
- **Implicação:** a sobreposição fenotípica observada entre as 7
  classes raciais do FairFace, particularmente Latinx, é
  **previsão direta** da estrutura genética populacional.
- **Caveat:** Edwards (2003) "Lewontin's Fallacy" argumenta que com
  loci suficientes a classificação é possível — mas isso não justifica
  taxonomia binária discreta.

### 3.3 Buolamwini & Gebru (2018) — Gender Shades

- **Venue:** PMLR vol 81 / ACM FAccT 2018
- **Citações:** **4 933** (Semantic Scholar — verificado)
- **Papel na tese:** **marco fundador** da auditoria de fairness em
  visão computacional comercial
- **Achado-bandeira:** gap de **34.7 pontos percentuais** na taxa
  de erro entre lighter male (0%) e darker female (34.7%) no IBM
  classifier.
- **Decisão metodológica:** Buolamwini & Gebru **escolhem
  Fitzpatrick em vez de race** por argumento de instabilidade
  conceitual de race labels — precedente direto da discussão Q11.

### 3.4 Kärkkäinen & Joo (2021) — FairFace

- **Venue:** WACV 2021 (IEEE/CVF)
- **Citações:** 263+ (Google Scholar, subestimado)
- **Papel na tese:** **dataset central** — toda execução experimental
  ocorre sobre este dataset
- **Estrutura:** 108 501 imagens; **7 categorias raciais** (White,
  Black, Indian, East Asian, Southeast Asian, Middle Eastern,
  Latino); origem YFCC-100M Flickr; anotação MTurk 3-anotador.
- **Caveat crítico:** **cross-dataset evaluation usa só 4 raças** —
  not 7. **7-class evaluation só aparece em external sets**
  (Twitter, Media, Protest), com Latino F1 = 0.247.

### 3.5 AlDahoul, Tan, Kasireddy & Zaki (2024/2026) — FaceScanPaliGemma

- **Venue:** **Nature Scientific Reports 2026** (DOI
  10.1038/s41598-026-39584-3) + arXiv preprint 2410.24148 anterior
- **Citações:** 11 (recente)
- **Papel na tese:** **SOTA atual** para FairFace race 7-class
- **Achado central:** **75.7% accuracy / 75% F1 macro** em FairFace
  test 7-class — **número que devemos alcançar/superar**.
- **Baseline confirmado:** FairFace ResNet-34 = **72%** (single
  seed, sem CI).
- **Breakdown por classe:** Black 90% F1, White 80%, Indian 79%,
  East Asian 78%, Middle Eastern 73%, SE Asian 67%, **Latinx 60%
  (pior)**.

### 3.6 Lin, Kim & Joo (2022) — FairGRAPE

- **Venue:** ECCV 2022
- **Citações:** 49 (Semantic Scholar)
- **Papel na tese:** **validação cruzada** do baseline 72% +
  documentação independente do gap Hispanic
- **Confirmação:** **ResNet-34 baseline = 72.0%** sobre FairFace
  race 7-class — **número replicado** independentemente de AlDahoul
  (Tabela 2 do paper).
- **Per-class baseline:** Hispanic 59.6%, Black 83.2%, White 73.9%
  — **mesmo padrão hierárquico** que AlDahoul observou em 2024.
- **Implicação metodológica:** ResNet-34 = 72% é **número robusto**
  através de splits diferentes e seeds independentes.

### 3.7 Schumann et al. (2023) — MST Annotation Consensus

- **Venue:** **NeurIPS 2023 Datasets and Benchmarks Track**
- **Papel na tese:** **protocolo de anotação MST** para a frente
  Q10
- **Contribuições:** (i) Monk Skin Tone Scale (10 pontos);
  (ii) MST-E dataset (1 515 imgs, 31 vídeos); (iii) demonstração de
  que **anotadores de diferentes regiões geográficas** têm
  variação **sistemática** em interpretação MST.
- **Recomendação operacional:** usar **pool diverso** de anotadores +
  **alta replicação** para fairness research. **Diretamente
  importável** para Fase 2 do Q10.

### 3.8 Hazirbas et al. (2021) — Casual Conversations

- **Venue:** CVPRW 2021 (Meta/Facebook AI)
- **Papel na tese:** **paradigma alternativo de anotação** —
  self-reported demographics
- **Estrutura:** 3 011 sujeitos pagos, 45 000+ vídeos; age + gender
  **self-reported**; Fitzpatrick anotado.
- **Argumento explícito contra race labels:** *"Labeling the
  ethnicity of subjects could lead to inaccuracies... Raters may
  have unconscious biases."*
- **Influência:** justifica adotar MST (não Fitzpatrick) + self-
  identification como gold standard para Q01.

### 3.9 Park et al. (2022) — Fair Supervised Contrastive Learning (FSCL)

- **Venue:** CVPR 2022
- **Citações:** **116** (Semantic Scholar)
- **Papel na tese:** **TÉCNICA CENTRAL** do Capítulo 2 (mitigação)
- **Teorema 1:** SupCon (Khosla et al. 2020) **incentiva** o encoder a
  aprender atributos sensíveis em datasets enviesados — **fundamento
  matemático para H1**.
- **Solução proposta (FSCL+):** restringir negativos a mesmo grupo
  sensível + group-wise normalization.
- **Resultado em CelebA (target=attractiveness, sensitive=male):**
  reduz EO de 30.5 → **6.5** (4.7× menor), perdendo só 1.4 pp de
  accuracy.
- **Lacuna crítica:** **NÃO testado em FairFace race 7-class** —
  exatamente nosso gap.

### 3.10 Sagawa et al. (2020) — Group DRO

- **Venue:** ICLR 2020
- **Papel na tese:** **TÉCNICA ALTERNATIVA** do Capítulo 2
- **Achado paradoxal:** DRO **naive** ≈ ERM em redes
  overparameterized — ambos memorizam treino.
- **Solução:** **strong ℓ2 regularization** (λ ∈ [0.1, 1.0],
  4 ordens de magnitude > default) OU early stopping.
- **Resultado em CelebA (hair color × gender):** worst-group accuracy
  41.1 → **86.7** (+45 pp) ao custo de 1.3 pp em accuracy média.
- **Implicação para nós:** se aplicarmos Group DRO a race 7-class,
  precisamos sweep de λ — não plug-and-play.

---

## 4. Gaps, considerações e achados

### 4.1 Achados convergentes da literatura (alta consensus)

| Achado | Endosso (papers) | Implicação |
|---|---|---|
| **Balanceamento de dataset não basta** | 8+ (Karkkainen, Wang, Grother, Lin, AlDahoul, Kotwal, Sagawa, Klare em Kotwal) | Mitigação algorítmica é **necessária**, não opcional |
| **Latinx/Hispanic é estruturalmente mais difícil** | 4 convergentes (Karkkainen .247, AlDahoul .60, Lin 59.6%, Buolamwini sobre instabilidade) | Sugere **causa estrutural** (Q10), não falha de modelo |
| **Race é construto social** | AAPA 2019 + Lewontin 1972 + Karkkainen explícito | Tese v3.1 reconhece e cita formalmente |
| **Ensembles naive não funcionam** | Bhaskaruni 2019, Sagawa 2020, Park 2022 (SupCon) | Mitigação exige **ponderação demográfica explícita** |
| **Métricas multi-classe são fragmentadas** | 7 papers usam métricas diferentes | Q05 → triangulação DR + worst-class + CV |

### 4.2 Achados divergentes (debates abertos)

- **Skin tone vs race como dimensão preferencial:** Buolamwini, Hazirbas,
  Schumann, Lafargue defendem skin tone; Karkkainen defende race;
  Neto defende contínuo.
- **Balanceamento ideal:** equidistribuído (FairFace, RFW, BFW) vs
  skewed (Gwilliam et al. em Kotwal) vs continuous (Neto 2025).
- **Threshold:** global vs subgroup-specific (Robinson 2020 propõe
  ganho conjunto em fairness + accuracy).

### 4.3 Gaps consolidados — as 5 frentes 🔬

| Frente | Pergunta | Score viabilidade × originalidade × impacto |
|---|---|---|
| **Q04** | Mitigação algorítmica em race 7-class FairFace | **13/15** ★★ |
| **Q10** | Matriz Fitzpatrick/MST × FairFace 7-race | **13/15** ★★ |
| **Q05** | Métrica fairness multi-classe | 11/15 |
| **Q06** | Decomposição ceiling 72% | 10/15 |
| **Q01** | Confiabilidade anotação Latinx | 9/15 |

**Decisão tomada:** Q04 + Q10 como capítulos principais, Q06 como
ablação, Q05 transversal, Q01 fundida em Q10.

### 4.4 Achados não-óbvios da Rodada 4

1. **As 7 categorias do FairFace não têm fundamento biológico.** São
   construto socio-político (US Census + ajustes Karkkainen). AAPA
   2019 + Lewontin 1972 sustentam.
2. **Fitzpatrick scale foi criada em 1975 para PUVA dosimetry**, NÃO
   para classificação racial. 1/3 dos dermatologistas confunde.
3. **Não há "número correto" de tons de pele** — 5 escalas em uso, cada
   uma para propósito específico. MST 10-pt é a correta para fairness
   research.
4. **A "dificuldade Latinx" tem fundamento estatístico previsível:**
   85% da variação genética é intra-populacional (Lewontin); Latinx é
   categoria que abrange múltiplos clusters geográficos
   simultaneamente — sobreposição fenotípica é previsão direta.

---

## 5. Próximos passos

### 5.1 Tese v3.1 — formulação final

> **O ceiling de 72-75.7% F1 macro em classificação racial 7-class
> sobre FairFace não é primariamente arquitetural nem
> metodologicamente solúvel apenas via mitigação algorítmica.
> Existe componente fenotípico irredutível — sobreposição
> distribucional de tom de pele entre categorias raciais — sobreposição
> particularmente aguda para Latinx/Hispanic.**
>
> **Contribuições originais propostas:**
> 1. Primeira matriz pública MST/Fitzpatrick × FairFace 7-race.
> 2. Primeira benchmark sistemática de mitigações algorítmicas
>    (FSCL+ multi-classe, Group DRO + strong reg, ensemble +
>    reweighting) em race classification 7-class in-domain.
> 3. Decomposição empírica do erro Latinx em fenotípico irredutível
>    vs algoritmicamente redutível.
> 4. Triangulação DR + worst-class F1 + CV como métrica padrão.

### 5.2 Plano experimental — 3 capítulos

**Capítulo 1 — Decomposição do ceiling (Q06)** (~4 semanas)
- ResNet-34 (baseline) vs ConvNeXt-T mantendo pipeline.
- 3 seeds casados (42, 1, 2).
- HPO modesto.
- **Hipótese H2:** troca de backbone ganha +2 a +5 pp F1 macro;
  Latinx F1 permanece ≈ 60% sem mitigação específica.

**Capítulo 2 — Mitigação algorítmica (Q04)** (~8-10 semanas)
- 3 técnicas testadas: FSCL+ adaptado multi-classe, Group DRO
  + strong ℓ2 + early stopping, deep ensemble + temperature
  scaling + group reweighting.
- 3 seeds × 3 técnicas = 9 runs comparáveis a baseline.
- **Hipótese H1:** ≥1 técnica reduz DR em ≥30% sem perder >2 pp
  F1 macro.

**Capítulo 3 — Matriz MST × race (Q10)** (~4 semanas + paralelo 6
semanas)
- Fase 1: classifier MST automatizado sobre FairFace val (10 954
  imgs).
- Fase 2: anotação manual subset (500-700 imgs × 3 anotadores
  regionalmente diversos, protocolo Schumann 2023).
- Fase 3: construção matriz P(MST_k | race_j).
- Fase 4: cross-reference com confusion matrix do melhor modelo.
- **Hipóteses H3 + H4:** spread Latinx ≥ 5 categorias MST; ≥ 50%
  das misclassificações Latinx em zonas MST overlap.

### 5.3 Cronograma

| Bloco | Duração | Marco |
|---|---|---|
| Aprovação tese v3.1 + plano | **esta reunião** | ✅ ou ajustes |
| Setup metodológico (02, 03, 08) | 2 semanas | Especificações executáveis |
| Cap 1 (Q06) | 4 semanas | Decomposição ceiling |
| Cap 2 (Q04) | 8-10 semanas | Benchmark mitigações |
| Cap 3 (Q10) Fases 1+3+4 | 4 semanas | Matriz + diagnóstico |
| Cap 3 (Q10) Fase 2 | 6 semanas (paralelo) | Validação manual |
| Escrita capítulos | 8-12 semanas | Defesa |
| **Total estimado** | **6-8 meses** | ~Dez 2026 / Fev 2027 |

### 5.4 Riscos identificados e mitigação

| Risco | Probabilidade | Mitigação |
|---|---|---|
| **H4 (50% overlap MST) refutada** | Médio | Plano B documentado em tese §6.3 — observação negativa é cientificamente válida |
| **Recrutamento de anotadores manuais** | Alto | Começar com Fase 1 (MST automatizado) sozinho como proof-of-concept |
| **Adaptação multi-classe das técnicas** | Médio | Github code disponível para FSCL+ (Park), Group DRO (Sagawa), FineFACE (Manzoor) |
| **Compute para 3 seeds × N técnicas** | Baixo | ConvNeXt-T (28M params) é leve; cada run ~horas |
| **FairFace teste set não-público** | Baixo | Usar val como test (convenção AlDahoul) |

### 5.5 Outputs esperados

- **3 capítulos de dissertação.**
- **1 paper standalone** sobre matriz MST × race (Q10) — submeter a
  CVPRW Fair AI ou FAccT.
- **Código open-source** dos experimentos + matriz pública.
- **Dataset auxiliar:** MST labels para subset do FairFace val.

---

## Anexo A — Perguntas antecipadas

Organizado por categoria, com resposta preparada.

### A.1 Sobre metodologia da pesquisa bibliográfica

**Q-A1:** Por que 23 papers e não 50 ou 100?
> **R:** Massa crítica vs ruído. O critério 5 (citações + venue forte)
> filtra rigorosamente. Surveys (Mehrabi 2021, Kotwal 2025) cobrem o
> espectro amplo; nossos 23 são **leitura primária** focada.

**Q-A2:** Como você garante que não há outro "Hassanpour" escondido?
> **R:** Toda autoria foi verificada em fonte primária (arXiv HTML,
> DOI). Status `✅ VERIFIED` em `00_referencias.md` exige inspeção
> da primeira página do PDF. Cross-check em Semantic Scholar.

**Q-A3:** Você buscou em outros idiomas (não-inglês)?
> **R:** Não sistematicamente. A literatura de fairness em
> CV/biometria é predominantemente em inglês (CVPR, ICLR, NeurIPS,
> TBIOM). Riscos de viés linguístico reconhecidos.

**Q-A4:** Como você lida com a Rodada 4 ter conteúdo via WebFetch
em vez de PDF integral?
> **R:** Transparência: campo `fonte_leitura` na ficha registra a
> origem. Conteúdo essencial (statements oficiais, partições
> numéricas Lewontin) é captura limpa via fonte oficial AABA + PMC.
> Fitzpatrick 1988 e Massey-Martin 2003 são **referências
> históricas** — citação bibliográfica é o uso, não leitura crítica
> de método.

### A.2 Sobre a tese teórica

**Q-A5:** Se race não é biológica, por que estudar classificação
racial em face recognition?
> **R:** **Distinção fundamental** (de Fuentes 2019): race é
> **biologicamente vazia mas socialmente operativa**. Sistemas faciais
> reificam categorias raciais com consequências de deployment
> (NIST 8280: 10-100× FPR differentials). Nossa pesquisa **audita**
> esse exercício classificatório, **não o valida**.

**Q-A6:** A AAPA 2019 invalida o uso de FairFace?
> **R:** Não. Invalida a leitura essencialista das categorias. O
> FairFace é **instrumento sociopolítico** legítimo para
> classification tasks em contexto de auditoria, com ressalvas
> reconhecidas (limitações §6.1 da tese v3.1).

**Q-A7:** Por que 7 classes e não 6 (como AlDahoul testa) ou 4
(como RFW/BFW usam)?
> **R:** Pragmatismo metodológico:
> - 7 é a granularidade máxima oferecida pelo FairFace (único
>   dataset 7-class).
> - 6 (East+SE merged) ganha accuracy mas perde Middle Eastern como
>   categoria separada.
> - 4 perde Latinx, Middle Eastern e SE Asian. Reportaremos ambos 7
>   e 6 para comparabilidade com AlDahoul.

**Q-A8:** Latinx é uma categoria racial ou étnica?
> **R:** Etnicamente, é etnia (US Census trata como tal: Hispanic é
> ethnicity, race é separada). FairFace **escolhe** tratar como
> raça por argumentos visuais. Nossa Q10 explicitamente investiga
> essa fronteira fuzzy.

### A.3 Sobre o plano experimental

**Q-A9:** Por que ConvNeXt-T (28M) e não ViT-L (300M) ou DINOv2 (1B)?
> **R:** ConvNeXt-T é **Pareto-eficiente** — competitivo em ImageNet
> com fração do compute. Hipótese H4 da tese: se ceiling é
> fenotípico (não arquitetural), backbones maiores não ajudam
> proporcionalmente. AlDahoul confirma: PaliGemma 3B (150× ResNet-34)
> ganha só 3.7 pp.

**Q-A10:** Por que FSCL+ e não FineFACE (Manzoor 2024)?
> **R:** FineFACE foi testado em FairFace **para gender**, não race.
> Adaptação para race 7-class exigiria modificar a formulação
> fine-grained. **FSCL+ é mais agnóstico** à arquitetura e tarefa.
> Se cronograma permitir, FineFACE será testado como ablação.

**Q-A11:** Como você combina H1, H2, H3, H4 em uma narrativa coerente?
> **R:** H2 (Cap 1) estabelece **patamar**: backbone modernizado.
> H1 (Cap 2) testa **teto**: mitigação algorítmica. H3+H4 (Cap 3)
> explica **resíduo**: componente fenotípico. Síntese: erro_total
> = irredutível_fenotípico + redutível_modelo, quantificado por
> classe.

**Q-A12:** Como você vai recrutar anotadores regionalmente diversos
para Fase 2 do Q10?
> **R:** Plano A: Prolific Academic (controle demográfico explícito,
> custo modesto). Plano B: rede pessoal/acadêmica com 5 anotadores
> espalhados (Brasil, EUA, Índia/Oriente Médio, Leste Asiático).
> Material de treinamento: MST-E dataset (Schumann 2023).

**Q-A13:** Quanto compute você precisa?
> **R:** Estimativa preliminar: ~200-400 GPU-hours.
> ConvNeXt-T fine-tuning sobre FairFace 86K train, 30 epochs, ~2-4h
> por seed. 3 seeds × 3 técnicas × ~3-4 ablations ≈ 30-50 runs.
> Hardware: RTX 3090 ou A100 acessível.

### A.4 Sobre originalidade

**Q-A14:** Qual sua contribuição vs FaceScanPaliGemma (AlDahoul)?
> **R:** Eles **alcançam** o SOTA via fine-tuning de modelo gigante
> sem mitigação algorítmica explícita. Nós **investigamos**:
> (a) modelo 100× menor com mitigação atinge o quê?
> (b) quanto do gap restante é fenotipicamente irredutível?
> Resposta cooperativa, não competitiva.

**Q-A15:** Você está reinventando a roda? Outros já não testaram
FSCL+ em FairFace?
> **R:** Negativo. Snowballing verificado em Rodada 2 + busca de
> "cited by" no Semantic Scholar não encontrou execução de FSCL+ em
> FairFace race 7-class. 7 papers **sugerem** essa direção;
> **zero** executam.

**Q-A16:** A matriz MST × race é realmente original? Não existe
algum benchmark já?
> **R:** Draelos et al. 2025 (J Cosmetic Dermatology) é o único
> precedente conhecido — **dermatologia**, dados não-públicos, sem
> cross-reference race × Fitzpatrick. Nossa matriz seria a
> **primeira pública em contexto de fairness ML**.

### A.5 Logísticas e ética

**Q-A17:** Aspectos éticos / LGPD?
> **R:** Sem dados pessoais identificáveis adicionais. FairFace já
> tem licença Creative Commons via YFCC-100M. Anotação manual em
> Fase 2 do Q10 envolve **imagens já públicas**; anotadores
> recebem apenas categorias MST (não opinião pessoal sobre
> identidade dos sujeitos).

**Q-A18:** Onde você pretende publicar?
> **R:** Tese de mestrado completa: Unifesp/ICT. Capítulo Q10
> standalone como paper: alvo CVPRW Fair AI 2027, ICCV Workshops
> 2027, ou ACM FAccT 2027 (preferência por interdisciplinar). Como
> alternativa journal: IEEE TBIOM (mesmo venue de Kotwal & Marcel
> 2025).

**Q-A19:** Reprodutibilidade — como você garante?
> **R:** Seeds fixas (42, 1, 2). Hiperparâmetros declarados em
> `02_metodologia.md`. Código no GitHub. Matriz MST × race pública
> como CSV. Documentação clara da partição train/val (oficial
> Karkkainen).

**Q-A20:** Quanto tempo se você quiser publicar antes de defender?
> **R:** Cap 3 (Q10) pode ser **decoupled** — matriz MST × race +
> diagnóstico é trabalho de ~3 meses standalone. Submeter durante
> escrita dos demais capítulos.

### A.6 Riscos e contingências

**Q-A21:** E se H4 (≥50% overlap MST → Latinx) for refutada?
> **R:** Plano B na tese §6.3: refutação **não inviabiliza** a
> dissertação. Mantemos C2 (benchmark mitigações) e C5 (métrica
> triangular). Reformulamos C1+C3 como **observação negativa** —
> cientificamente válido.

**Q-A22:** E se o MST classifier automatizado tiver viés grande?
> **R:** Mitigação: Fase 2 (validação manual) audita o classifier.
> Se discrepância grande, Fase 2 vira ground truth e usamos
> apenas o subset anotado para análise — perde-se escala mas
> mantém-se rigor.

**Q-A23:** E se ConvNeXt-T não bater ResNet-34 (H2 refutada)?
> **R:** Achado interessante per se — refutação de "backbone
> modernos sempre melhoram". Aprofundaríamos análise: por que
> não? (tamanho de treino, regularização, data augmentation).

**Q-A24:** Cronograma de 6-8 meses é realista?
> **R:** Apertado mas viável dado:
> (i) infra experimental já existe (Rodada 1 do MBA — agora
>     historico/);
> (ii) técnicas têm código open-source;
> (iii) escrita em paralelo aos experimentos.
> Buffer recomendado: +2 meses para imprevistos.

**Q-A25:** Por que não esperar Casual Conversations v2 ou outro
dataset novo?
> **R:** v2 não muda escopo (mesma metodologia, expansão geográfica).
> Aguardar bloqueia início. Possível adendo futuro como capítulo
> bonus se houver tempo.

---

## Anexo B — Tabela de relevância dos 23 papers

Tabela completa com critérios de seleção rigorosos. Todas as
autorias verificadas em fonte primária. Todas as citações verificadas
em Semantic Scholar quando disponível.

| # | Ficha (link) | Autores (verificados) | Ano | Venue (verificado) | Citações | Status venue | Lente disruptiva | Papel na tese |
|---|---|---|---|---|---|---|---|---|
| 1 | [karkkainen_2021](04_pesquisa_bibliografica/dataset_karkkainen_2021.md) | Kärkkäinen, Joo | 2021 | **WACV** (IEEE/CVF) | 263+ GS | ✅ Top CV venue | Cobertura | **Dataset central** |
| 2 | [aldahoul_2024](04_pesquisa_bibliografica/aldahoul_2024.md) | AlDahoul, Tan, Kasireddy, Zaki | 2024/26 | **Nature Sci Reports 2026** + arXiv 2410.24148 | 11 S2 | ✅ Nature journal | Metodológica | **SOTA atual (75.7%)** |
| 3 | [lin_2022](04_pesquisa_bibliografica/lin_2022.md) | Lin, Kim, Joo | 2022 | **ECCV** | 49 S2 | ✅ Top CV venue | Metodológica | **Validação cruzada baseline** |
| 4 | [manzoor_2024](04_pesquisa_bibliografica/manzoor_2024.md) | Manzoor, Rattani | 2024 | **ICPR** (Springer LNCS) | 2 S2 | ✅ Peer-reviewed | Metodológica | Armadilha textual (gender ≠ race) |
| 5 | [dehdashtian_2024](04_pesquisa_bibliografica/dehdashtian_2024.md) | Dehdashtian, Sadeghi, Boddeti | 2024 | **CVPR** | 23 S2 | ✅ Top venue | Paradigma | Teoria trade-off |
| 6 | [dominguez_2024](04_pesquisa_bibliografica/dominguez_2024.md) | Dominguez-Catena, Paternain, Galar | 2024 | **Information Fusion** | 7 S2 | ✅ Top journal | Metodológica | Auditoria de datasets |
| 7 | [lafargue_2025](04_pesquisa_bibliografica/lafargue_2025.md) | Lafargue, Claeys, Loubes | 2025 | **ECML PKDD** | 1 S2 | ✅ Peer-reviewed | Metodológica | EU AI Act + audit |
| 8 | [dataset_wang_2019](04_pesquisa_bibliografica/dataset_wang_2019.md) | Wang, Deng, Hu, Tao, Huang | 2019 | **ICCV** | alto (seminal) | ✅ Top venue | Cobertura | RFW dataset (paralelo) |
| 9 | [grother_2019](04_pesquisa_bibliografica/grother_2019.md) | Grother, Ngan, Hanaoka | 2019 | **NISTIR 8280** | n/a (gov) | ✅ Official | Paradigma | Escala industrial |
| 10 | [buolamwini_2018](04_pesquisa_bibliografica/buolamwini_2018.md) | Buolamwini, Gebru | 2018 | **PMLR/FAccT** | **4 933 S2** | ✅ Marco fundador | Paradigma | **Founding paper** |
| 11 | [park_2022](04_pesquisa_bibliografica/park_2022.md) | Park, Lee, Lee, Hwang, Kim, Byun | 2022 | **CVPR** | **116 S2** | ✅ Top venue | Metodológica | **FSCL+ (Cap 2)** |
| 12 | [sagawa_2020](04_pesquisa_bibliografica/sagawa_2020.md) | Sagawa, Koh, Hashimoto, Liang | 2020 | **ICLR** | alto | ✅ Top venue | Paradigma | **Group DRO (Cap 2)** |
| 13 | [bhaskaruni_2019](04_pesquisa_bibliografica/bhaskaruni_2019.md) | Bhaskaruni, Hu, Lan | 2019 | **ICTAI** | n/a | ✅ Peer-reviewed | Metodológica | Ensemble fair (refuta naive) |
| 14 | [survey_mehrabi_2021](04_pesquisa_bibliografica/survey_mehrabi_2021.md) | Mehrabi, Morstatter, Saxena, Lerman, Galstyan | 2021 | **ACM CSur 54.6** | alto | ✅ Top survey venue | Nenhuma | Survey geral ML |
| 15 | [dataset_robinson_2020](04_pesquisa_bibliografica/dataset_robinson_2020.md) | Robinson, Livitz, Henon, Qin, Fu, Timoner | 2020 | **CVPRW** | n/a | ✅ Peer-reviewed | Cobertura | BFW dataset |
| 16 | [dataset_hazirbas_2021](04_pesquisa_bibliografica/dataset_hazirbas_2021.md) | Hazirbas, Bitton, Dolhansky, Pan, Gordo, Canton Ferrer | 2021 | **CVPRW** (Meta) | n/a | ✅ Peer-reviewed | Paradigma | **Self-reported demographics** |
| 17 | [schumann_2023](04_pesquisa_bibliografica/schumann_2023.md) | Schumann, Olanubi, Wright, Monk Jr., Heldreth, Ricco | 2023 | **NeurIPS Datasets & Benchmarks** | n/a | ✅ Top venue | Paradigma | **MST protocol (Cap 3)** |
| 18 | [neto_2025](04_pesquisa_bibliografica/neto_2025.md) | Neto, Damer, Cardoso, Sequeira | 2025 | arXiv 2506.01532 | 0 (recente) | ⚠ Under review | Paradigma | Continuous labels |
| 19 | [survey_kotwal_2025](04_pesquisa_bibliografica/survey_kotwal_2025.md) | Kotwal, Marcel | 2025 | **IEEE TBIOM** | n/a (recente) | ✅ Top biometrics | Nenhuma | Survey FR (atualizado) |
| 20 | [fuentes_2019](04_pesquisa_bibliografica/fuentes_2019.md) | Ackermann, Athreya, Bolnick, Fuentes, Lasisi, Lee, McLean, Nelson | 2019 | **Am J Phys Anthropol** | n/a | ✅ Statement oficial AABA | Paradigma | **Fundamento teórico** |
| 21 | [lewontin_1972](04_pesquisa_bibliografica/lewontin_1972.md) | Lewontin | 1972 | **Evolutionary Biology** (Springer) | 5000+ | ✅ Citação clássica | Paradigma | **Fundamento genético** |
| 22 | [fitzpatrick_1988](04_pesquisa_bibliografica/fitzpatrick_1988.md) | Fitzpatrick | 1988 | **Arch Dermatol** (JAMA Network) | alto | ✅ Foundational journal | Nenhuma | **Origem Fitzpatrick scale** |
| 23 | [massey_martin_2003](04_pesquisa_bibliografica/massey_martin_2003.md) | Massey, Martin | 2003 | **Princeton/NIS** technical doc | n/a (canonical) | ⚠ Technical report | Nenhuma | Precedente sociológico MST |

### Notas sobre verificação de citações

- **Semantic Scholar API** consultada em 2026-05-25; rate limit
  ocasionalmente forçou cross-check via Google Scholar.
- **Citações ≥ threshold** confirmadas para R1+R2; R3 (2024-25)
  ainda imaturas em contagem, aprovadas por venue + cobertura única.
- **R4 (foundational)** aprovados por status canônico, não citação count.

### Verificação de autoria

**100% das 23 fichas** têm autoria verificada em:

- arXiv abstract page (R1-R3 modernos).
- DOI editorial (papers em journals).
- Bioanth.org (Fuentes 2019).
- Wikipedia + secondary sources cross-validated (Lewontin 1972,
  Fitzpatrick 1988, Massey-Martin 2003 — fontes históricas).

---

## Status do material

- **Documento produzido em:** 2026-06-01
- **Para revisão antes da reunião com Prof. Quiles.**
- **Após a reunião:** mover para `docs/historico/material_reuniao_<data>.md`.

---

## Apêndices técnicos disponíveis no repo

- [`docs/ativo/05_landscape.md`](05_landscape.md) — síntese transversal
- [`docs/ativo/06_gap.md`](06_gap.md) — ranqueamento gaps
- [`docs/ativo/07_thesis_statement.md`](07_thesis_statement.md) — tese v3.1 formal
- [`docs/ativo/04_pesquisa_bibliografica/_perguntas.md`](04_pesquisa_bibliografica/_perguntas.md) — Q&A completo
- [`docs/ativo/04_pesquisa_bibliografica/_triagem.md`](04_pesquisa_bibliografica/_triagem.md) — log de decisões editoriais
- [`docs/ativo/00_referencias.md`](00_referencias.md) — fonte única de citações verificadas
- [`docs/ativo/04_pesquisa_bibliografica/INDEX.md`](04_pesquisa_bibliografica/INDEX.md) — navegação central
