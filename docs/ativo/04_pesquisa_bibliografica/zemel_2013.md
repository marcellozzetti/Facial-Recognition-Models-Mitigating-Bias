---
name: zemel-2013
status_verificacao: VERIFIED
autores: [Rich Zemel, Yu Wu, Kevin Swersky, Toniann Pitassi, Cynthia Dwork]
ano: 2013
titulo: "Learning Fair Representations"
venue: "International Conference on Machine Learning (ICML) — Proceedings of Machine Learning Research vol 28"
tipo_publicacao: conference
arxiv_id: null
doi: null
url_primario: https://proceedings.mlr.press/v28/zemel13.html
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~30
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de PMLR (pdfs/zemel_2013_lfr.pdf)
---

# Learning Fair Representations (Zemel, Wu, Swersky, Pitassi & Dwork, 2013)

> **Test-of-Time Award ICML 2023** — paper fundador do subcampo de
> **Fair Representation Learning** em ML. Estabelece o framework de
> aprender representação intermediária que (i) preserva utilidade
> para predição e (ii) ofusca informação sobre atributo protegido.

## 1. Resumo do problema atacado

Em 2013, fairness em ML era discutido sobretudo em termos de
**output adjustment** (post-hoc threshold tuning). Zemel et al.
propõem deslocar a discussão para o **espaço de representação**:
aprender mapeamento Z = f(X) tal que:

- **Utility**: Z preserva informação para predizer Y.
- **Group fairness**: Z é (aproximadamente) **independente do
  atributo protegido A**.
- **Individual fairness**: indivíduos similares em X têm Zs
  similares.

## 2. Método

### 2.1 Formulação probabilística

Mapeamento estocástico X → Z (vetor de protótipos), onde Z é
discretizado em K "clusters" prototípicos. Treinamento minimiza
loss composta:

L = A_z · L_z + A_x · L_x + A_y · L_y

- **L_y**: erro de classificação de Y a partir de Z.
- **L_x**: reconstrução de X a partir de Z (preserva informação).
- **L_z**: **statistical parity** — P(Z=k | A=0) ≈ P(Z=k | A=1) para
  cada protótipo k.

### 2.2 Trade-off

Hiperparâmetros A_z, A_x, A_y controlam tradeoff utility ↔ fairness.
**Curva Pareto** é explicitamente parametrizada.

### 2.3 Garantia de Lipschitz

Mapeamento é parametrizado para satisfazer condição de Lipschitz
sobre métrica de similaridade — operacionaliza "individual
fairness" (Dwork et al. 2012 "Fairness Through Awareness").

## 3. Datasets e setup experimental

- **German Credit** (UCI, 1000 instâncias) — Y = aprovação de crédito,
  A = sex.
- **Adult** (UCI) — Y = >50K income, A = sex.
- **Heritage Health Prize** — Y = uso hospitalar, A = idade.

## 4. Métricas reportadas

- **Accuracy** (utility).
- **Discrimination**: |P(Ŷ=1 | A=0) − P(Ŷ=1 | A=1)| (statistical
  parity violation).
- **Consistency**: similaridade de output para vizinhos próximos
  em X.

## 5. Resultados principais

- LFR (Learning Fair Representations) atinge **menor
  discrimination que baselines** sem perda significativa de
  accuracy nos 3 datasets.
- Pareto curve é favorável: ponto operacional com discrimination ≈
  0 (estatístico) ainda preserva accuracy boa.

## 6. Limitações declaradas pelos autores

- **Statistical parity** como definição de fairness — pré-Hardt 2016
  (que mostra suas limitações teóricas).
- **A binário** (sex).
- **Tabular data only** — não testa imagens ou texto.
- **K (número de protótipos)** é hyperparameter — não há prescrição.

## 7. Limitações que identifiquei

- **Statistical parity é métrica problemática** (Hardt 2016 mostra
  que força desvio de Bayes-optimal). Nossa pesquisa usa EOD-like
  (max/min ratio de F1 por classe) que evita essa armadilha.
- **Tabular focus** — para imagens (nosso caso), técnicas mais
  modernas (adversarial, contrastive como [[park_2022]]) dominam.
- **K protótipos discretos** — limita capacidade representacional
  para domínios complexos.

## 8. Relação com nossa pesquisa

**Centralidade histórica:**

Estabelece o **paradigma** de fair representation learning que
fundamenta:
- [[park_2022]] FSCL (contrastive fair representations)
- [[madras_2018]] LAFTR (adversarial fair representations)
- [[dehdashtian_2024]] U-FaTE (formaliza trade-off accuracy ↔
  fairness)

**Para v3.2:**

1. **Justificativa do paradigma**: a abordagem do orientador
   ("treinar para reconhecer tonalidade, aplicar no race classifier")
   está conceitualmente na **linhagem Zemel 2013** — aprender
   representação informativa sobre tom de pele para condicionar
   classifier de raça.
2. **Test-of-Time Award (ICML 2023)** = forte endosso comunidade.
   Pode ser citado como "este é o paradigma estabelecido" sem
   necessidade de defesa.
3. **Limitação reconhecida**: nosso pipeline v3.2 não é fair
   representation learning **pura** — é mais próximo de
   **conditional architecture** (FiLM-based). Zemel é precursor
   teórico, não o método operacional.

## 9. Pontos para citar

- *"Zemel, Wu, Swersky, Pitassi e Dwork (ICML 2013, Test-of-Time Award
  2023) estabeleceram o paradigma de Learning Fair Representations:
  aprender mapeamento intermediário que preserva utilidade para a
  tarefa-alvo enquanto ofusca informação sobre atributo protegido."*
- *"O pipeline experimental adotado nesta dissertação posiciona-se
  na linhagem de Zemel et al. (2013) — incorporar informação
  demográfica (tom de pele MST) como representação intermediária
  para condicionar o classifier downstream (raça e, posteriormente,
  reconhecimento facial)."*

## 10. Arquivos relacionados

- PDF: `pdfs/zemel_2013_lfr.pdf` (gitignored — open access PMLR).
- Test-of-Time recognition: ICML 2023 ceremony.
- Entradas relacionadas: [[hardt_2016]] (critica statistical parity),
  [[park_2022]] (FSCL como descendente moderno), [[madras_2018]]
  (LAFTR como evolução adversarial), [[dehdashtian_2024]] (formaliza
  trade-off accuracy/fairness com novo arcabouço).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Definições de fairness mais sofisticadas** que statistical
  parity. ✅ Realizado por Hardt 2016 (EO_h/EOD) e literatura
  posterior — alinhado com nossa Q05.
- **A multi-categórico**. ✅ **Alinhado com nossa pesquisa** (race
  7-class).
- **Aplicações em domínios não-tabulares** (imagens, texto). ✅
  Realizado por Madras 2018, Park 2022, e outros.
- **Combinar individual + group fairness** rigorosamente. ⚠ Direção
  teórica aberta.

## 12. Análise crítica do método

### (a) Rigor formal

- **Mapeamento probabilístico X → Z** com decomposição em K protótipos
  — clareza matemática.
- **Loss composta** (L_z + L_x + L_y) com hiperparâmetros explícitos
  controlando trade-off.
- **Garantia Lipschitz** para individual fairness — operacionaliza
  Dwork 2012.
- **Test-of-Time Award ICML 2023** = forte endosso da comunidade.
- **Limitação**: statistical parity é métrica pré-Hardt (limitações
  teóricas conhecidas).

### (b) Reprodutibilidade

- ✅ Datasets tabulares públicos (German Credit, Adult, Heritage
  Health).
- ⚠ K (número de protótipos) é hyperparameter sem prescrição.
- ⚠ Sem reportar multi-seed ou IC.
- ✅ Adoção ampla em literatura subsequente — método replicado.

### (c) Aplicabilidade ao pipeline v3.2

- **Paradigma fundador** de Fair Representation Learning — nossa
  abordagem com FiLM-conditioning está conceitualmente na linhagem
  Zemel.
- **Conditional architecture (FiLM)** é mais flexível que
  representational pura — evolução metodológica.
- **Statistical parity como métrica** é limitada — usamos DR + EOD
  generalizados.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Representation learning intermediário | ✅ Justificada — paradigma fundador |
| Statistical parity como métrica | ⚠ Pré-Hardt — limitação conceitual |
| K protótipos discretos | ⚠ Choice — limita capacidade representacional |
| Garantia Lipschitz | ✅ Justificada — operacionaliza individual fairness |
| Tabular only | ✅ Choice — escopo histórico |
| Hiperparâmetros A_z, A_x, A_y | ✅ Justificada — controle explícito do trade-off |

### (e) Conexão com R5/R6

- [[hardt_2016]]: critica statistical parity que Zemel usa. Evolução
  metodológica.
- [[madras_2018]] LAFTR: evolução adversarial direta de Zemel.
- [[park_2022]] FSCL+: evolução contrastive direta de Zemel.
- [[dehdashtian_2024]] U-FaTE: formaliza o trade-off accuracy ↔
  fairness que Zemel propôs implicitamente.
- [[perez_2018]] FiLM: mecanismo alternativo (conditional) ao
  representational puro de Zemel.
- **Implicação para v3.2**: Zemel é **citação fundamental** para
  contextualização teórica. Nossa abordagem (FiLM-conditioning) é
  evolução conditional do paradigma representational.
