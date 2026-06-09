---
name: madras-2018
status_verificacao: VERIFIED
autores: [David Madras, Elliot Creager, Toniann Pitassi, Richard Zemel]
ano: 2018
titulo: "Learning Adversarially Fair and Transferable Representations"
venue: "International Conference on Machine Learning (ICML) — Proceedings of Machine Learning Research vol 80"
tipo_publicacao: conference
arxiv_id: "1802.06309"
doi: null
url_primario: https://arxiv.org/abs/1802.06309
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~50
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/madras_2018_laftr.pdf)
---

# LAFTR: Learning Adversarially Fair and Transferable Representations (Madras, Creager, Pitassi & Zemel, 2018)

> Evolução **adversarial** do paradigma de [[zemel_2013]] (Fair
> Representation Learning). Conecta formalmente noções de group
> fairness (Demographic Parity, EOD, EO_h) a **objetivos adversariais
> distintos**. Demonstra fair **transfer learning** — representação
> treinada para uma tarefa preserva fairness em tarefas downstream.

## 1. Resumo do problema atacado

Em 2018, técnicas de adversarial debiasing existiam (Ganin & Lempitsky
2015 domain adaptation; Beutel et al. 2017; Edwards & Storkey 2016) mas:
(i) escolha do objetivo adversarial era ad hoc; (ii) sem teoria
conectando objetivo adversarial à noção de fairness garantida;
(iii) sem validação de **transferência** — representação fair para uma
tarefa permanece fair para outra?

## 2. Método

### 2.1 Arquitetura LAFTR

Três componentes treinados conjuntamente:

- **Encoder f**: X → Z (representação fair).
- **Classifier g**: Z → Ŷ (predição de target).
- **Adversary h**: Z → Â (predição do atributo protegido).

Objetivo:

min_{f,g} max_h [L_y(g(f(X)), Y) − α · L_a(h(f(X)), A)]

Encoder + classifier tentam minimizar erro de Y; adversary tenta
predizer A de Z. Em equilíbrio, Z não contém info sobre A
suficiente para predição.

### 2.2 Conexão objetivo adversarial ↔ noção de fairness

**Teorema central**: a escolha de **loss function adversarial L_a**
determina **qual noção de fairness** é garantida:

- **L_a = cross-entropy** → Demographic Parity (statistical parity).
- **L_a = group-balanced cross-entropy** → Equalized Odds.
- **L_a = conditional cross-entropy (Y=1)** → Equal Opportunity.

**Implicação prática**: para reportar EOD, treine com adversary L_a
group-balanced.

### 2.3 Transferência

Após treinar Z para tarefa Y_1, **congelar Z** e treinar novo classifier
para tarefa Y_2 sobre mesmo Z. Se Z é fair, classifier Y_2 herda
fairness sem re-treinar adversary.

## 3. Datasets e setup experimental

- **Adult** (UCI tabular, Y = >50K income, A = sex).
- **Heritage Health Prize** (tabular, Y = uso hospitalar, A = idade).
- Para experimento de transferência: treinar Z em uma tarefa, testar em
  outra (ex.: treinar em "tem charlson > median", transferir para
  "tem CCI > median" ou outra).

## 4. Métricas reportadas

- **Accuracy** (utility).
- **Demographic Parity Gap, EO Gap, EOD Gap** — gaps específicos da
  noção de fairness alvo.

## 5. Resultados principais

- **Demonstra trade-off Pareto-eficiente** entre accuracy e fairness,
  com curvas explícitas.
- **LAFTR-EO supera baselines** em EO gap mantendo accuracy.
- **Transferência funciona**: representação fair para tarefa-fonte
  permanece fair na tarefa-target sem re-treinamento do adversary.
- **Worst-case guarantees** teóricos com prova matemática.

## 6. Limitações declaradas pelos autores

- Tabular only.
- Atributo protegido A binário.
- Treinamento adversarial é **instável** (problema clássico do
  GAN/min-max).

## 7. Limitações que identifiquei

- **Instabilidade adversarial**: para FairFace race 7-class, treinar
  adversary com 7 classes pode ser ainda mais frágil que o caso
  binário do paper. Mitigação: usar variantes mais estáveis
  (FairHSIC, FSCL+) — ver [[park_2022]] que cita Madras como baseline
  superado em CelebA gender.
- **Não cobre conditional architectures** (FiLM-like). LAFTR é
  representation **monolítica**; nossa v3.2 usa MST como contexto
  condicional explícito.

## 8. Relação com nossa pesquisa

**Papel na v3.2:**

LAFTR é o **paradigma representational** que precede a abordagem
**conditional** (FiLM) que adotaremos. Importante para:

1. **Mostrar evolução metodológica**: Zemel 2013 → Madras 2018 → Park
   2022 → conditional architectures (nossa v3.2).
2. **Conectar formalmente** noções de fairness a objetivos adversariais
   (Teorema central de Madras). Quando reportarmos EOD/EO_h em
   nossos resultados, a interpretação está ancorada nesse vocabulário.
3. **Justificativa de transferência**: a tese v3.2 propõe aplicar o
   classifier MST a face recognition (RFW/BFW). LAFTR fornece
   **prova de conceito** de que representação fair transfere entre
   tarefas — sustenta a hipótese de que MST→race generaliza para
   MST→face recognition.

## 9. Pontos para citar

- *"Madras et al. (ICML 2018) formalizam a conexão entre escolha de
  objetivo adversarial e noção de fairness garantida: cross-entropy
  padrão induz Demographic Parity, cross-entropy group-balanced induz
  Equalized Odds, e cross-entropy condicional induz Equal
  Opportunity."*
- *"A propriedade de transferência demonstrada por Madras et al.
  (LAFTR, 2018) — uma representação treinada para ser fair em uma
  tarefa permanece fair em tarefas downstream — fundamenta
  empiricamente a extensão da abordagem desta dissertação de
  classificação racial (Capítulo 2) para reconhecimento facial
  (Capítulo 3, sobre RFW/BFW)."*

## 10. Arquivos relacionados

- PDF: `pdfs/madras_2018_laftr.pdf` (gitignored).
- Código: github.com/VectorInstitute/laftr
- Entradas relacionadas: [[zemel_2013]] (predecessor direto, mesmo
  grupo), [[hardt_2016]] (Definições EO_h/EOD que Madras conecta a
  objetivos), [[zhang_2018]] (adversarial debiasing paralelo),
  [[park_2022]] (FSCL — alternativa contrastiva ao adversarial).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Estender para multi-classe A** (atributo protegido multi-categórico).
  ✅ Diretamente alinhado com **Q04** e v3.2 (race 7-class).
- **Aplicar a domínios não-tabulares** (imagens, texto). ✅ Realizado
  pelos autores posteriormente e pela comunidade.
- **Melhorar estabilidade do treinamento adversarial**. ⚠ Trabalho
  em curso na comunidade — adoção de variantes não-adversariais
  (contrastive, conditional) é uma resposta.
- **Conectar com causal fairness**. ⚠ Direção paralela.

## 12. Análise crítica do método

### (a) Rigor formal

- **Formulação min-max** matematicamente clara: 3 componentes
  (encoder, classifier, adversary) treinados conjuntamente.
- **Teorema central** conecta escolha de L_adv a noção de fairness:
  cross-entropy → DP, group-balanced → EOD, conditional → EO_h.
  **Contribuição teórica importante.**
- **Garantia de transferência** (Teorema 1) — limite superior
  herdado por classificadores downstream.
- **Worst-case guarantees** com prova matemática.
- **Limitação**: treinamento adversarial inerentemente instável
  (problema GAN clássico).

### (b) Reprodutibilidade

- ✅ Código público: github.com/VectorInstitute/laftr.
- ✅ Datasets públicos (Adult, Heritage Health Prize).
- ⚠ Tabular only — generalização para imagens não no paper original.
- ⚠ Hiperparâmetros de α (peso adversarial) exigem tuning.

### (c) Aplicabilidade ao pipeline v3.2

- **Esqueleto teórico para Cap 3** (face recognition): Teorema 1
  fundamenta extensão race classification → FR.
- **Mecanismo conditional (FiLM)** é alternativo a representational
  monolítico de LAFTR.
- **Adversarial debiasing como baseline** (Zhang 2018 é alternativa
  operacional).

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Adversarial vs alternativas | ✅ Justificada — generaliza Zemel |
| 3 escolhas de L_adv | ✅ Teoricamente justificadas (Teorema 1) |
| Tabular datasets | ⚠ Choice — escopo histórico |
| A binário | ❌ Limitação reconhecida |
| Transferência via encoder freeze | ✅ Justificada — Teorema 1 |
| Stabilidade adversarial | ⚠ Limitação reconhecida |

### (e) Conexão com R5/R6

- [[zemel_2013]] LFR: predecessor direto. LAFTR é evolução adversarial.
- [[hardt_2016]] EO_h/EOD: Madras conecta formalmente objetivos
  adversariais a estas noções.
- [[zhang_2018]] Adversarial debiasing: paralelo metodológico,
  Madras é mais teórico, Zhang mais operacional.
- [[park_2022]] FSCL+: alternativa contrastive (não-adversarial) com
  estabilidade superior.
- [[perez_2018]] FiLM: mecanismo conditional alternativo ao
  representational. Combinação não testada na literatura.
- [[aguirre_2023]] (R6, ficha pendente): demonstra empíricamente
  fair transfer em multi-task NLP — confirma princípio de LAFTR.
- **Implicação para v3.2**: LAFTR é **fundamentação teórica** da
  transferência fair classification → FR (Cap 3). Cita-se como
  referência canônica de fair representation learning adversarial.
