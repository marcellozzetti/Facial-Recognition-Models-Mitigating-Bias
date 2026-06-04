---
name: sagawa-2020
status_verificacao: VERIFIED
autores: [Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang]
ano: 2020
titulo: "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization"
venue: "International Conference on Learning Representations (ICLR)"
tipo_publicacao: conference
arxiv_id: "1911.08731"
doi: null
url_primario: https://arxiv.org/abs/1911.08731
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~50
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/sagawa_2020_groupdro.pdf)
---

# Group DRO (Sagawa, Koh, Hashimoto & Liang, 2020)

## 1. Resumo do problema atacado

Redes neurais sobreparametrizadas treinadas via **ERM (Empirical
Risk Minimization)** podem ter **acurácia média alta** em test set
i.i.d. mas **falhar consistentemente em subgrupos atípicos** —
aprendendo correlações espúrias que valem em média mas não em todos
os grupos. Exemplo: rede que aprende "background aquático →
waterbird" tem alta acurácia média mas falha em waterbirds em
background terrestre.

O paper investiga **Group Distributionally Robust Optimization
(group DRO)**: minimizar **worst-case loss sobre grupos pré-definidos**
em vez da média. **Achado-bandeira:** group DRO ingênuo **falha** em
redes sobreparametrizadas — qualquer modelo que zera loss em treino
zera worst-case loss também, mas generaliza mal. **Solução: forte
regularização** (ℓ2 penalty maior que default + early stopping)
permite group DRO superar ERM em **10-40 pontos percentuais** na
worst-group accuracy.

## 2. Método

### 2.1 Formulação ERM vs Group DRO

- **ERM:** θ* = argmin E_P[ℓ(θ; (x,y))]
- **Group DRO:** θ* = argmin max_{g ∈ G} E_{P_g}[ℓ(θ; (x,y,g))]

onde G = {1, ..., m} são grupos pré-definidos baseados em atributos
espúrios (ou demográficos no nosso contexto).

### 2.2 Otimização estocástica

Section 5 introduz algoritmo de otimização estocástico para DRO com:
- Garantias de convergência no caso convexo.
- Estabilidade empírica em modelos não-convexos (deep networks).
- Tracking de pesos por grupo via running average exponencial.

### 2.3 Regularização "forte" (descoberta central)

- **Default ℓ2 penalty** (λ=0.0001 para ResNet50) é insuficiente.
- **Strong ℓ2:** λ = 1.0 (Waterbirds) ou λ = 0.1 (CelebA) — **4
  ordens de magnitude acima do default**.
- **Early stopping:** parar antes de zero training loss.
- **Group adjustments:** ajustar a métrica de loss por grupo para
  refletir diferenças no generalization gap.

### 2.4 Setup experimental — 3 aplicações

| Dataset | Task | Y (target) | A (spurious attribute) | m groups |
|---|---|---|---|---|
| **Waterbirds** (CUB + Places) | bird classification | waterbird/landbird | water/land background | 4 |
| **CelebA** | face attribute | hair color (blond/dark) | gender (M/F) | 4 |
| MultiNLI | NLI | entailment/neutral/contradiction | negation presence | 6 |

### 2.5 Arquiteturas e hiperparâmetros

- **Waterbirds/CelebA:** ResNet50 pre-trained ImageNet, fine-tuned.
- **MultiNLI:** BERT.
- **Otimizador:** minibatch SGD para ERM, algoritmo estocástico
  próprio para DRO.
- **Default ℓ2:** 1e-4. **Strong ℓ2:** 1.0 (W) / 0.1 (CelebA).
- **Sem reportar seed multipla** — single run conforme excerto lido.

## 3. Datasets e setup experimental

Ver §2.4. Nenhum dos datasets é FairFace.

CelebA é o **único dataset facial**: hair color como target, gender
como atributo "espúrio". Configuração diferente de race classification,
mas mesma estrutura demográfica.

## 4. Métricas reportadas

- **Average accuracy** (em test i.i.d.).
- **Worst-group accuracy:** acurácia mínima entre os m grupos no test
  set.
- **Training accuracy** (overall + worst-group).

## 5. Resultados principais (valores numéricos)

### 5.1 CelebA (hair color target, gender protected) — Tabela 1

| Regularização | Método | Train acc all | Train acc worst | Test acc all | **Test acc worst** |
|---|---|---|---|---|---|
| Standard | ERM | 100.0 | 99.9 | 94.8 | **41.1** |
| Standard | DRO | 100.0 | 100.0 | 94.7 | **41.1** ← idêntico! |
| **Strong ℓ2** | ERM | 95.7 | 40.4 | 95.8 | 37.8 |
| **Strong ℓ2** | **DRO** | 95.0 | **93.4** | 93.5 | **86.7** ← saltou 45 pp! |

**Achado-bandeira:** DRO standard NÃO funciona. **DRO + strong ℓ2**
multiplica worst-group accuracy em **~2.1×** (41.1 → 86.7) ao custo de
apenas 1.3 pp em accuracy média.

### 5.2 Waterbirds — Tabela 1

| Regularização | Método | Test acc all | **Test acc worst** |
|---|---|---|---|
| Standard | ERM | 97.3 | 60.0 |
| Standard | DRO | 97.4 | 76.9 |
| **Strong ℓ2** | DRO | 96.6 | **84.6** |
| Early stopping | DRO | 93.2 | **86.0** |

Worst-group accuracy: **60.0 → 86.0 com early stopping DRO**.

### 5.3 MultiNLI

| Regularização | Método | Test acc all | Test acc worst |
|---|---|---|---|
| Standard | ERM | 82.5 | 65.7 |
| Standard | DRO | 82.0 | 66.4 |
| Early stopping | DRO | 81.4 | **77.7** |

### 5.4 Padrão consistente

- **Default DRO ≈ ERM** em todos 3 datasets — paradoxal mas explicado
  por overparameterization.
- **Strong reg + DRO** → 10-40 pp de melhoria em worst-group.

## 6. Limitações declaradas pelos autores

- **Conhecimento prévio dos grupos é necessário** durante treino —
  precisa-se de annotations (a, y, g) por exemplo.
- **No test time, group label não é usado** — modelo só usa x. Mas
  precisa de g no treino.
- **Hiperparâmetro ℓ2 precisa ser tunado** por dataset — não é "drop-in".
- **Métrica é worst-group accuracy** — não captura distribuição
  completa de disparidade (apenas extremo).

## 7. Limitações que identifiquei (leitura crítica)

- **CelebA hair color target + gender protected ≠ FairFace race
  7-class.** Tarefa específica testada é binária por binária — m=4
  groups. Para FairFace race 7-class com gender também protected, m
  seria 14. Comportamento de DRO com m grande não é testado.
- **Strong ℓ2 = 0.1 ou 1.0** é **4 ordens de magnitude acima** do
  default ImageNet (1e-4). Para ConvNeXt-T (que tem LayerNorm
  em vez de BatchNorm), o efeito de ℓ2 forte pode ser **diferente**
  — não validado.
- **Worst-group accuracy** é métrica diferente de DR ou Equalized
  Odds. Mapear nossa razão de disparidade a worst-group exige
  cuidado:
  - Worst-group = min(acc_g) — extremo absoluto.
  - DR = max(F1)/min(F1) — razão de extremos.
  - Não-intercambiáveis sem tradução explícita.
- **Single run reportado** (parece). Sem desvio padrão.
- **Apenas binary protected attribute** efetivamente testado em
  facial (CelebA). Group DRO em multi-class race não trivial:
  - 7 race × 2 gender = 14 grupos pode ter grupos pequenos demais.
  - Conditioning em y também (race × y) explodiu para 7×Y_attribute.
- **Early stopping subjetivo** — depende de validation set bem
  representado em todos os grupos. Para FairFace, validation set tem
  Latino sub-representado (~1.6K imagens), o que pode enviesar
  early stopping.

## 8. Relação com nossa pesquisa

**Centralidade conceitual:** Group DRO formaliza matematicamente o
princípio "otimizar para o pior grupo, não para a média" — princípio
**implícito** em nossa razão de disparidade (que reporta o gap entre
grupos extremos). É um candidato **forte** para mitigação em FairFace
race 7-class.

**Pontos de ancoragem:**

1. **Worst-group como métrica:** nossa razão de disparidade DR =
   max(F1)/min(F1) é uma **razão**, enquanto worst-group de Sagawa é
   um **mínimo absoluto**. Relacionados mas distintos. **Reportar
   worst-group F1 ao lado de DR** seria adição valiosa.
2. **Lição metodológica "strong regularization":** se Group DRO for
   testado em nossa pesquisa, default weight_decay (1e-4) **não vai
   funcionar**. Precisa varredura de λ entre 0.01 e 1.0.
3. **Gap identificado para `06_gap.md`:** **Group DRO não foi testado
   em race 7-class FairFace**. Apenas binary hair color × binary
   gender (CelebA). Aplicar com m=7 (raças) ou m=14 (raça × gênero)
   seria contribuição experimental tangível.
4. **CelebA worst-group 41.1 → 86.7 com DRO + strong ℓ2** é uma
   **prova de conceito de magnitude** — ganho de 45 pp em worst-group
   sem perda significativa em média. Justifica testar a técnica.
5. **Anatomia do failure mode do ERM em overparameterized:** Sagawa
   demonstra que **modelos com vanishing training loss não generalizam
   no worst case** mesmo quando treinados com DRO. Isto pode explicar
   por que **deep ensemble naive** (que tende a sobreajustar todos os
   seeds) não melhora fairness em nossas Rodadas 1 experimentais —
   conecta com [[bhaskaruni_2019]].

## 9. Pontos para citar / posicionar

- *"Sagawa, Koh, Hashimoto e Liang (2020), publicado nos anais do
  ICLR 2020, demonstram que group Distributionally Robust
  Optimization (group DRO) aplicado ingenuamente a redes neurais
  sobreparametrizadas não melhora a worst-group accuracy: modelos
  ERM e DRO atingem desempenho indistinguível porque ambos zeram a
  loss de treino. A solução é forte regularização — ℓ2 penalty
  amplificado em quatro ordens de magnitude sobre o default ou
  early stopping — que permite group DRO superar ERM em 10 a 40
  pontos percentuais em worst-group accuracy."*
- *"Em experimento sobre CelebA com classificação de cor de cabelo
  como alvo e gênero como atributo protegido, Sagawa et al. (2020,
  Tabela 1) reportam worst-group accuracy de 41.1% para ERM
  standard, 41.1% para DRO standard, e 86.7% para DRO combinado com
  ℓ2 penalty forte (λ=0.1) — ganho de 45 pontos percentuais ao
  custo de apenas 1.3 pontos em average accuracy."*
- *"O princípio 'otimizar para o pior grupo' formalizado por group
  DRO (Sagawa et al., 2020) está conceitualmente alinhado com a
  razão de disparidade DR=max(F1)/min(F1) adotada nesta dissertação,
  embora as métricas sejam distintas: worst-group é um mínimo
  absoluto, DR é uma razão entre extremos. Ambas refletem o
  compromisso com equidade demográfica como prioridade sobre acurácia
  agregada."*

## 10. Arquivos relacionados

- PDF local: `pdfs/sagawa_2020_groupdro.pdf` (gitignored).
- Texto extraído: `pdfs/sagawa_2020_groupdro.txt` (gitignored).
- Código: github.com/kohpangwei/group_DRO (mencionado no paper).
- Entradas relacionadas: [[buolamwini_2018]] (motivação ética, citado
  pelos próprios Sagawa et al.),
  [[bhaskaruni_2019]] (ensembles ingênuos não funcionam — mesma
  classe de achado),
  [[park_2022]] (técnica algorítmica alternativa, contrastive),
  [[dehdashtian_2024]] (U-FaTE — formaliza o trade-off entre
  worst-case e average).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 3.2, linha F9.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section 6 (Discussion) + Limitations:

- **Group DRO sem labels demográficos no treino** — atualmente
  exige (x, y, g). Direção: descobrir g via clustering. ⚠ Paralela,
  emergente em SensitiveNets.
- **Aplicar a tarefas multi-classe complexas** — paper testa
  worst-group em 4-6 grupos binários. Extensão para FairFace
  race 7-class não trivial. ✅ **Alinhada com Q04 e Q05**.
- **Estudar interação Group DRO × outras técnicas de regularização**
  (não só ℓ2 e early stopping) — autores enfatizam que strong reg é
  essencial. ✅ **Alinhada com Q04**.
- **Aprimorar algoritmo estocástico** — autores propõem como
  convergence-guaranteed mas não otimizado. ⚠ Engenharia, não
  pesquisa.
- **Aplicar a face race classification** — autores testam CelebA
  hair color × gender mas não FairFace race 7-class. ✅ **GAP
  CENTRAL — Q04**.

## 12. Análise crítica do método (revisão pós-reunião 2026-06-04)

### (a) Rigor formal

- **Formulação min-max bem definida**: argmin_θ max_g E[ℓ(θ;
  (x,y,g))]. Pressupõe partição de grupos conhecida no treino.
- **Achado paradoxal formalmente verificado**: em regime
  overparametrizado, ERM e DRO convergem ao mesmo limite (Bayes
  risk zero em treino implica worst-group loss zero em treino).
  Empíricamente demonstrado (Tabela 1) + interpretado.
- **Algoritmo estocástico** com garantia de convergência no caso
  convexo (Section 5). Caso não-convexo (deep networks) é
  empírico — convergência observada mas sem prova.

### (b) Reprodutibilidade

- ✅ **Hiperparâmetros declarados**: SGD minibatch para ERM,
  algoritmo próprio para DRO. λ ℓ2 reportado em (1.0 Waterbirds,
  0.1 CelebA). Default ℓ2 do ResNet50 = 0.0001.
- ⚠ **Single run reportado** — sem desvio padrão entre seeds.
  Sagawa et al. apenas dizem "trained to convergence".
- ✅ **Código público**: github.com/kohpangwei/group_DRO.
- Critério de seleção: dois cenários separados — (a) treinar até
  convergência total (regime degenerado), (b) early stopping
  (regime útil).

### (c) Aplicabilidade ao pipeline v3.2

- **Strong ℓ2 = 0.1-1.0** é específico de ResNet50 com BatchNorm.
  **ConvNeXt-T usa LayerNorm** — efeito de ℓ2 forte pode ser
  diferente. **Sweep de λ necessário** (não plug-and-play).
- **m = grupos**: para race 7-class × gender = 14 grupos. Para
  race 7-class apenas (sem gender), m=7. Tamanho de pior grupo
  fica pequeno (Latinx test = 1623 — gerenciável).
- **Worst-group accuracy** é métrica natural para reportar; mapeia
  bem para worst-class F1 que adotamos.
- **Conexão com FiLM** ([[perez_2018]]) **não-trivial**: Group DRO
  opera no nível de loss, FiLM no nível de feature. Combinação seria
  ortogonal.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Group DRO (worst-case sobre grupos) | ✅ **Formalmente justificada** como caso especial de DRO (Ben-Tal et al. 2013) |
| Strong ℓ2 regularization | ✅ Justificada — Section 3.1 demonstra que sem reg, DRO ≈ ERM |
| Early stopping como alternativa | ✅ Justificada empíricamente |
| ResNet50 + BatchNorm | ❌ Assumida — sem comparação com LayerNorm/ConvNeXt |
| Single run reportado | ❌ Assumida (custo? não declarado) |
| Group labels conhecidos no treino | ⚠ Reconhecida como limitação |

### (e) Conexão com R5

- [[hardt_2016]] EO_h/EOD: Group DRO **não otimiza diretamente** EO.
  Worst-group é métrica relacionada mas distinta (mínimo absoluto vs
  razão). Pode-se computar EO post-hoc.
- [[zemel_2013]]: Group DRO **não aprende fair representation** —
  treina classifier robusto sem alterar features. Paradigma
  diferente.
- [[madras_2018]] LAFTR: Group DRO e LAFTR são **complementares**:
  Madras representa, Sagawa otimiza. Combinação testável (LAFTR
  + DRO loss) **não explorada na literatura**.
- [[perez_2018]] FiLM: ortogonal ao Group DRO — pode ser usado em
  conjunto, FiLM modula features condicionadas a MST, DRO escolhe o
  pior caso de grupo.
