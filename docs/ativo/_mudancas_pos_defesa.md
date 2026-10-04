# Mudanças pós-defesa — notas para incorporar aos capítulos

> Documento vivo criado em **2026-10-04** (véspera da qualificação).
> Consolida as revisões identificadas na verificação das 5 lacunas
> (contra o catálogo local em `04_pesquisa_bibliografica/` + 7 buscas
> web complementares em 04/10). Nada aqui é bloqueador para a defesa
> de **05/10/2026**; tudo deve ser aplicado ao texto da tese nas
> semanas seguintes.

---

## 1. Cap. 1 — Introdução (prioridade: alta)

### 1.1 Posicionar Pangelinan (2023) como precedente metodológico

**Onde inserir:** §1.3 (Problema de pesquisa) ou §1.5 (Delimitação).

**O que escrever (sugestão de parágrafo):**

> *"A estratégia metodológica de decomposição do erro adotada por
> esta pesquisa (Capítulo 6, Síntese) herda, em espírito, o
> framework de Pangelinan et al. (2023), que decompõem o gender gap
> em face recognition em quatro componentes isoláveis — skin tone,
> geometria facial, dados de treino balanceados e pixel information.
> Esta pesquisa estende essa lógica em duas direções: (i) o eixo
> target passa a ser o erro de classificação racial multi-classe
> em Latinx (não o gender gap em verificação), e (ii) a escala de
> tom de pele é a Monk Skin Tone de 10 pontos (não a Fitzpatrick de
> 6 pontos usada por Pangelinan)."*

**Por que importa:** antecipa a objeção de que a decomposição seria
"não original". A originalidade está na extensão (classificação
multi-classe + MST), não na técnica.

### 1.2 Reconhecer Schumann (2023) e Pereira (2026) na motivação da matriz MST × raça

**Onde inserir:** §1.3 (Problema de pesquisa), ao justificar a
Contribuição 2 (matriz MST × raça pública).

**O que escrever:**

> *"Schumann et al. (NeurIPS 2023) estabelecem o protocolo de
> anotação da escala Monk Skin Tone, mas o aplicam ao dataset
> interno MST-E (1 515 imagens, 19 sujeitos). Pereira et al.
> (2026) liberam o Skin Tone in the Wild (STW, 42 313 imagens)
> sobre coleta própria. Nenhum dos dois cruza a escala MST com as
> sete classes raciais do FairFace — tabulação que esta pesquisa
> publica como Contribuição 2."*

---

## 2. Cap. 4 — Metodologia (prioridade: alta)

### 2.1 Reconhecer Conditional Batch Normalization como mecanismo equivalente

**Onde inserir:** §4.3 (Racional do FiLM), antes da subseção que
compara FiLM a alternativas (concat, attention, AdaIN).

**O que escrever (sugestão de parágrafo):**

> *"Em termos formais, o mecanismo FiLM (Perez et al., 2018, AAAI)
> é equivalente ao Conditional Batch Normalization (CBN) proposto
> por de Vries et al. (2017, NeurIPS) — a diferença residual é que
> FiLM admite inserção fora de camadas de normalização. Aplicações
> prévias de CBN em contextos de equidade restringem-se a atributos
> binários (notavelmente, debiasing por gênero em
> [Wang et al., 2020, CVPR; Alvi et al., 2018]), nos quais o sinal
> condicionante tem dimensão 2. A originalidade desta pesquisa é a
> instanciação em condicionamento contínuo de alta dimensão (vetor
> MST softmax, 10-dim) sobre tarefa de classificação racial
> multi-classe — configuração não documentada na literatura
> revisada."*

**Por que importa:** uma banca atenta provavelmente perguntará
*"então o que vocês propõem é CBN com sinal MST?"*. A resposta deve
ser direta: *"sim, é essa a formulação, e a contribuição é a
instanciação em classificação multi-classe com sinal fenotípico
contínuo"*.

### 2.2 Reformular hipótese H5 — transferência arquitetural (não re-treinamento)

**Status:** já aplicado em [`06_gap.md`](06_gap.md) linha 185 e no
PPTX (slide 9) em 04/10.

**Resumo da nova H5:**

> *"O módulo FiLM+ConvNeXt-T pré-treinado em classificação racial
> (Cap. 2), transferido via feature-freeze para face recognition
> (RFW/BFW), melhora TAR @ FAR = 1 × 10⁻⁴ do grupo African em
> ≥ +3 pp sobre backbone equivalente sem condicionamento —
> demonstrando transferência arquitetural (não re-treinamento) na
> linha de Madras et al. (2018, LAFTR, Teorema 1)."*

**Impacto experimental no Cap. 3:**

- Protocolo explícito de **feature-freeze**: encoder MST e camadas
  FiLM congelados; apenas a cabeça de verificação é treinada.
- Métrica primária: **TAR @ FAR = 1e-4** por raça (não accuracy).
- Comparação: backbone sem FiLM, mesmo tamanho de dados, mesma
  cabeça de verificação, mesmo protocolo de treino.
- Checkpoint do classificador racial do Cap. 2 publicado como
  artefato (Contribuição 6).

---

## 3. Cap. 2 — Revisão da literatura (prioridade: média)

### 3.1 Atualizar seção de métricas de equidade

**Onde:** §2.5 (Métricas e protocolos de reporte).

**O que acrescentar:** citação a `Review of Demographic Fairness in
Face Recognition` (arXiv 2502.02309, 2025) como survey recente que
documenta a prevalência de reporte com métrica única e defende
triangulação. Fundamenta L5 com fonte adicional.

### 3.2 Atualizar seção de baselines

**Onde:** §2.6 (Baselines de mitigação).

**O que acrescentar:** FairCal (Salvador et al., 2021), MixFairFace
(2022) e score normalization (Linghu et al., 2024) como trabalhos
que treinam mitigação diretamente em RFW/BFW — contraste com a
proposta de transferência arquitetural desta pesquisa.

---

## 4. Resumo executivo das mudanças

| Mudança | Local | Status | Prioridade |
|---|---|---|---|
| Pangelinan como precedente metodológico | Cap 1 §1.3 | ⏳ a fazer | Alta |
| Schumann/Pereira na motivação Contribuição 2 | Cap 1 §1.3 | ⏳ a fazer | Alta |
| CBN (de Vries 2017) como equivalente formal de FiLM | Cap 4 §4.3 | ⏳ a fazer | Alta |
| H5 reformulada — transferência arquitetural | gap.md, PPTX s.9 | ✅ aplicado 04/10 | — |
| Survey 2025 e FairCal/MixFairFace citados | Cap 2 §2.5, §2.6 | ⏳ a fazer | Média |

---

## 5. Referências a adicionar em `references.bib`

Entradas que precisarão ser incluídas no `.bib` para suportar as
mudanças acima:

- `@inproceedings{deVries2017CBN}` — Conditional Batch Norm (NeurIPS 2017)
- `@article{Pereira2026STW}` — Skin Tone in the Wild (STW)
- `@article{ReviewFairness2025}` — Review of Demographic Fairness
  (arXiv 2502.02309)
- `@inproceedings{Salvador2021FairCal}` — FairCal (ICLR 2022)
- `@inproceedings{WangLiu2022MixFairFace}` — MixFairFace
- `@inproceedings{Linghu2024ScoreNorm}` — score normalization (IJCB 2024)

Nota: `Mehrabi2021Survey` e `Kleinberg2017Impossibility` já
existem no bib.
