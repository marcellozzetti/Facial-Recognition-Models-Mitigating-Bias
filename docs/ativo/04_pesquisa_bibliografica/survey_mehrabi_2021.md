---
name: survey-mehrabi-2021
status_verificacao: VERIFIED
autores: [Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, Aram Galstyan]
ano: 2021
titulo: "A Survey on Bias and Fairness in Machine Learning"
venue: "ACM Computing Surveys (CSur), vol 54, no 6, article 115"
tipo_publicacao: journal
arxiv_id: "1908.09635"
doi: "10.1145/3457607"
url_primario: https://arxiv.org/abs/1908.09635
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 170+
lente_disrupcao: nenhuma
fonte_leitura: PDF integral extraído via pypdf (pdfs/mehrabi_2021_survey.pdf); este é um survey, não paper de método
---

# Mehrabi et al. (2021) — A Survey on Bias and Fairness in Machine Learning

## 1. Resumo do problema atacado

Survey amplo sobre **viés e equidade em aprendizado de máquina**,
publicado em **ACM Computing Surveys** (venue top — IF~14+).
Apresenta a taxonomia canônica de (a) **tipos de viés** organizados
no ciclo Data → Algorithm → User Interaction, (b) **definições
formais de fairness** (10 definições), e (c) **abordagens de
mitigação** classificadas como pre-/in-/post-processing. Não traz
contribuição experimental nova — é referência obrigatória para
contextualização teórica.

## 2. Método

(Não aplicável — survey.) Estrutura organizada em 6 seções:

- **Seção 2:** Exemplos reais de unfairness algorítmica (COMPAS,
  beauty pageant, blink detection).
- **Seção 3:** Taxonomia de tipos de bias no ciclo
  Data ↔ Algorithm ↔ User.
- **Seção 4:** Taxonomia de 10 definições formais de fairness.
- **Seção 5:** Métodos de mitigação por domínio (classification,
  regression, PCA, clustering, NLP, etc.).
- **Seção 6:** Direções futuras.

### 2.1 Taxonomia de bias (Seção 3)

**Data → Algorithm:**
- **Measurement Bias:** features mal escolhidas/proxy ruim (ex.:
  arrest rate como proxy de criminalidade em COMPAS).
- **Omitted Variable Bias:** variável importante ausente.
- **Representation Bias:** subgrupos faltantes (ex.: ImageNet
  ocidental-cêntrico).
- **Aggregation Bias:** conclusões individuais a partir do agregado
  (ecological fallacy).
- Sampling, Longitudinal Data Fallacy, Linking.

**Algorithm → User:**
- **Algorithmic Bias:** introduzido por design do algoritmo.
- **Popularity / Ranking / Emergent / Self-selection.**

**User → Data:**
- **Historical Bias:** dados refletem desigualdades passadas.
- **Population / Behavioral / Content Production / Temporal.**

### 2.2 Definições de fairness (Seção 4, Tabela 1)

**Individual** (similar predictions for similar individuals):
- **Fairness Through Awareness** (Dwork et al.).
- **Counterfactual Fairness** (Kusner et al.) — contrafactual.

**Group** (equalizar métricas entre grupos):
- **Demographic Parity / Statistical Parity:** P(Ŷ | A=0) = P(Ŷ | A=1).
- **Conditional Statistical Parity:** condiciona em legitimate factors L.
- **Equal Opportunity** (Hardt et al.): P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1) — equal TPR.
- **Equalized Odds** (Hardt et al.): equal TPR AND equal FPR.
- **Treatment Equality:** equal FN/FP ratio.
- **Test Fairness** (Chouldechova): equal P(Y=1 | S=s, R=r).
- **Fairness Through Unawareness:** não usar atributo sensível.

**Subgroup:**
- **Subgroup Fairness** (Kearns et al.): satisfaz fairness em todos
  os subgrupos definidos por intersecção.

### 2.3 IMPOSSIBILIDADE FUNDAMENTAL (Kleinberg-Mullainathan-Raghavan
2017)

O survey enfatiza que **algumas definições de fairness são
mutuamente incompatíveis**. Especificamente, **calibration** e
**balance for positive/negative class** **não podem ser satisfeitas
simultaneamente**, exceto em casos especiais altamente restritos.
**Consequência prática:** ao escolher uma métrica de fairness, o
analista está implicitamente sacrificando outra.

### 2.4 Métodos de mitigação (Seção 5)

Classificação em 3 categorias:

- **Pre-processing:** transforma dados antes do treino (reamostragem,
  reweighting, fair representation learning, disparate impact removal,
  preferencial sampling).
- **In-processing:** modifica o algoritmo durante o treino
  (regularization, adversarial debiasing, fair constraints in
  optimization).
- **Post-processing:** modifica predições após o treino (calibração,
  threshold tuning, label flipping).

## 3. Datasets e setup experimental

Não aplicável.

## 4. Métricas reportadas

Não aplicável. O survey **define** métricas mas não roda experimentos.

## 5. Resultados principais (valores numéricos)

Não aplicável.

## 6. Limitações declaradas pelos autores

- **Cobertura por domínio é desigual.** NLP recebe mais atenção que
  visão computacional.
- **Snapshot temporal:** survey publicado em 2021, papers pós-2021
  não cobertos.
- **Foco em definições formais** — discussão sociotécnica menos
  desenvolvida.

## 7. Limitações que identifiquei (leitura crítica)

- **Não inclui facial recognition / classification specificamente.**
  Section 5 lista classification em geral, mas a literatura específica
  de face attribute classification (FairFace, RFW, FaceScanPaliGemma)
  é posterior ou não citada.
- **Análise da impossibilidade Kleinberg simplifica:** menciona o
  teorema mas não discute como ele se manifesta em diferentes
  domínios (e.g., classification vs ranking).
- **Survey de definições é exaustivo** mas não oferece **diretrizes
  práticas** para quando usar cada uma. Para nossa pesquisa, isso
  significa que precisamos buscar guias mais aplicados em outros
  papers (Hardt et al., Pleiss et al.).
- **Discussão de "future work" é genérica.** Não identifica gaps
  específicos como "fair race classification em 7 classes" — o que
  é justamente nosso interesse.
- **Estilo descritivo, não crítico.** Apresenta abordagens sem
  argumentar quais são mais robustas em settings específicos.
- **Múltiplos termos para o mesmo conceito** (ex.: statistical
  disparity vs demographic parity) reforçam confusão em literatura
  derivada.

## 8. Relação com nossa pesquisa

**Papel:** referência canônica para **contextualização teórica**
do conceito de viés e fairness. Toda dissertação séria em fairness
deve citar Mehrabi et al. (2021) na **revisão de literatura
fundamental**, junto com Buolamwini & Gebru (2018) e Barocas &
Selbst (2016).

**Pontos de ancoragem:**

1. **Taxonomia canônica de fairness:** nossa razão de disparidade DR
   é uma forma de **group fairness** (especificamente, max-min
   accuracy disparity). Para nossa pesquisa, citar Mehrabi como
   referência da taxonomia de **group fairness vs individual fairness
   vs subgroup fairness** ancora a discussão sobre escolha de métrica.
2. **Categorização pre-/in-/post-processing:** nossa metodologia
   experimental opera em **in-processing** (deep ensemble) e
   **post-processing** (calibração, temperature scaling). O survey
   nos dá a linguagem padrão para descrever isso.
3. **Impossibilidade Kleinberg:** ao discutir nossa razão de
   disparidade, podemos invocar Mehrabi para justificar que **não
   tentamos satisfazer todas as fairness definitions simultaneamente
   — apenas a razão de disparidade**, e isso é uma escolha de design
   consciente.
4. **Vocabulário técnico:** "protected attribute", "sensitive
   attribute", "demographic group", "intersectional analysis" —
   nossa dissertação deve usar a terminologia consistente com o
   survey.
5. **Para `06_gap.md`:** o survey **não cobre race classification
   facial específica** (publicado antes de FairFace classifier ter
   visibilidade pós-AlDahoul). Isto sugere que nosso gap pode ser
   apresentado como **lacuna explícita na literatura de fairness
   geral** — não apenas em fairness facial.

## 9. Pontos para citar / posicionar

- *"O conceito de fairness algorítmico em aprendizado de máquina é
  formalizado em pelo menos dez definições matemáticas distintas,
  classificáveis nas categorias individual, group e subgroup
  fairness (Mehrabi, Morstatter, Saxena, Lerman & Galstyan, 2021).
  A presente dissertação opera no eixo group fairness, especificamente
  via uma medida de max-min disparity entre classes raciais,
  alinhada com a noção de Equalized Odds (Hardt et al., 2016)
  particularizada para classificação multi-classe."*
- *"Mehrabi et al. (2021) categorizam técnicas de mitigação de viés
  em pré-processamento (modificação dos dados antes do treino),
  in-processamento (modificação do algoritmo durante o treino) e
  pós-processamento (modificação das predições após o treino). Esta
  dissertação investiga predominantemente abordagens de in-
  processing (ensemble profundo, regularização) e post-processing
  (calibração de temperatura), sem alterar o dataset FairFace
  original."*
- *"É importante reconhecer, conforme registrado por Mehrabi et al.
  (2021) e originalmente demonstrado por Kleinberg, Mullainathan e
  Raghavan (2017), que satisfazer múltiplas definições de fairness
  simultaneamente é matematicamente impossível exceto em casos
  altamente restritos. A escolha de uma métrica específica — razão
  de disparidade entre classes raciais — é, portanto, uma decisão
  consciente que prioriza paridade entre grupos demográficos sobre
  outras formulações como counterfactual fairness ou test fairness."*

## 10. Arquivos relacionados

- PDF local: `pdfs/mehrabi_2021_survey.pdf` (gitignored).
- Texto extraído: `pdfs/mehrabi_2021_survey.txt` (gitignored).
- Versão publicada: ACM CSur 54.6 (2021), DOI 10.1145/3457607.
- Entradas relacionadas: [[buolamwini_2018]] (caso real citado no
  survey), [[sagawa_2020]] (instância de in-processing por DRO),
  [[park_2022]] (instância de in-processing por contrastive learning),
  [[bhaskaruni_2019]] (instância de ensemble como mitigação).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 3.2, linha F11.
