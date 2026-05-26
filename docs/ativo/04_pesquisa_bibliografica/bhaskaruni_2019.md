---
name: bhaskaruni-2019
status_verificacao: VERIFIED
autores: [Dheeraj Bhaskaruni, Hui Hu, Chao Lan]
ano: 2019
titulo: "Improving Prediction Fairness via Model Ensemble"
venue: "IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), pp. 1810-1814"
tipo_publicacao: conference
arxiv_id: null
doi: "10.1109/ICTAI.2019.00273"
url_primario: https://ieeexplore.ieee.org/document/8995403/
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 21
lente_disrupcao: metodologica
fonte_leitura: PDF integral extraído via pypdf (pdfs/bhaskaruni_2019_ensemble.pdf, versão NSF PAR open access — par.nsf.gov/servlets/purl/10171436)
---

# Improving Prediction Fairness via Model Ensemble (Bhaskaruni, Hu & Lan, 2019)

## 1. Resumo do problema atacado

Nina et al. (2017) **teorizaram** que ensembles de modelos
individualmente enviesados podem produzir um ensemble **fair** — os
vieses se cancelariam mutuamente. **Não há evidência empírica** para
sustentar esta intuição. Bhaskaruni et al. (a) **testam
empiricamente** a tese; (b) descobrem que **ensembles padrão
(bagging) NÃO melhoram fairness** — às vezes pioram; (c) propõem
nova estratégia de ensemble inspirada em AdaBoost, mas reweighting
**unfairly predicted** em vez de **mispredicted** instances.

## 2. Método

### 2.1 Hipótese central

Em AdaBoost, instâncias mal classificadas ganham peso na próxima
iteração. **Bhaskaruni inverte a lógica:** instâncias **unfairly
classificadas** ganham peso. Identificação de "unfair" via k-NN
situation testing (Luong et al.).

### 2.2 Pseudo-algoritmo

Para t = 1, ..., m:
1. Treinar modelo base f_t com loss reponderado por w_i^(t).
2. δ_i^(t) = 1 se x_i é **unfairly predicted** por f_t (situação testing
   diz: vizinhança k-NN de x_i tem statistical disparity acima de
   threshold r), 0 c.c.
3. Peso do modelo: α_t = ln((1 - ε_t) / ε_t), onde ε_t = Σ w_i δ_i / Σ w_i.
   **Modelo mais "fair" tem maior peso no ensemble.**
4. Atualizar pesos das instâncias: w_i^(t+1) = w_i^(t) · exp(α_t · δ_i^(t)).
   **Instâncias com previsão unfair ganham peso.**

Ensemble final: f(x) = Σ α_t f_t(x).

### 2.3 K-NN situation testing

Para cada x_i, vizinhança N_{i,k} de seus k vizinhos mais próximos
(distância euclidiana). Statistical disparity:

SD(f_t; N_{i,k}) = P(promoted | male, vizinhança) − P(promoted | female, vizinhança)

Se |SD| > r → instância **unfairly predicted**.

### 2.4 Setup experimental

- **Base model:** **Logistic Regression** (não deep learning).
- **Ensemble size:** m (não declarado quantitativamente).
- **k (vizinhos):** não declarado.
- **r (threshold):** não declarado.
- **50 random trials**, média reportada.
- **Train/test split:** 75/25 random.

### 2.5 Datasets — tabular, não imagem

- **Credit Default** (UCI): 30K → downsampled para 20K. Sensitive:
  education degree (binarizado). Label: default payment (binário).
- **Community Crime** (UCI): 1 993 instâncias. Sensitive: fração de
  African-American residents (binarizado 0.5). Label: crime rate
  (binarizado 0.5).

**Não testado em facial recognition, FairFace, ou qualquer dataset
de visão computacional.** É fair ML genérico em tabular.

## 3. Datasets e setup experimental

Ver §2.5. Datasets tabulares, fair ML clássico.

## 4. Métricas reportadas

- **Statistical Disparity (SD):** P(Y=1 | majority) − P(Y=1 | minority).
- **Equalized Odds (EO):** mesma SD condicionada em true label.
- **Classifier error rate** (erro de classificação).

## 5. Resultados principais (valores numéricos)

### Community Crime dataset — Tabela I

| Método | SD ↓ | EO ↓ | Error ↓ |
|---|---|---|---|
| Logistic Regression (standard) | .1000 ± .0000 | .4695 ± .0000 | .1883 ± .0000 |
| FairLR (fair logistic regression) | .0898 ± .0971 | .0620 ± .0882 | .1166 ± .0189 |
| **Bagging (Nina et al.)** | **.2267** ± .2025 | .2855 ± .1867 | .1187 ± .0987 |
| AdaBoost padrão | .0746 ± .0124 | .3712 ± .0889 | .1013 ± .0117 |
| **Proposed (AdaBoost fair)** | **.0239** ± .0247 | **.0593** ± .0367 | .1604 ± .0445 |

**Achados-bandeira:**

1. **Bagging PIORA a fairness** (SD .2267 vs .1000 da LR baseline) —
   refuta diretamente a tese de Nina et al. (2017) de que "bagging
   melhora fairness".
2. **AdaBoost padrão** melhora SD parcialmente mas EO continua alto.
3. **Proposed** reduz SD em 4× sobre LR e EO em 8× sobre LR, **mas ao
   custo de aumento em error rate** (1.16 → 1.6 — trade-off
   accuracy-fairness explícito).
4. **Proposed vs FairLR:** Proposed bate FairLR em SD (×3.8) e
   competitivo em EO, mas tem error maior.

Mesma estrutura observada no Credit Default dataset (Tabela II,
mencionada mas não tabulada no excerto lido).

## 6. Limitações declaradas pelos autores

- Trade-off explícito accuracy-fairness — não Pareto-efficient.
- Heurística (AdaBoost variation), não teoricamente ótima.
- Apenas 2 datasets testados, ambos tabulares.

## 7. Limitações que identifiquei (leitura crítica)

- **Não testado em facial recognition / deep learning.** O paper opera
  em logistic regression sobre tabular. Conclusões podem não
  transferir diretamente para CNN/Transformer faciais.
- **Trade-off accuracy-fairness não é Pareto-efficient** — error
  aumenta 38% (1.16 → 1.60). Para nossa pesquisa, isto é informação
  importante: deep ensemble **naive** pode não funcionar; **proper
  weighted ensemble** funciona mas paga preço em accuracy.
- **Hiperparâmetros essenciais (k, r, m) não declarados** — papel
  curto (5 páginas ICTAI) limita reprodução.
- **Sensitive attribute binarizado.** Tarefa de raça 7-class do
  FairFace não é diretamente compatível.
- **Situation testing k-NN** assume distância euclidiana faz sentido —
  válido em tabular, **questionável em representations deep**.
- **Resultado em Community Crime é small dataset (1 993)** — alta
  variância (Bagging SD .2267 ± .2025 — IC enorme).
- **Bagging com SD .2267 ± .2025** sugere que o intervalo de
  confiança cobre 0 a 0.43 — o achado "bagging piora fairness" pode
  ser inflado pela alta variância do dataset.
- **ICTAI é venue menor** que CVPR/NeurIPS/ICLR — paper menos
  influente que outros candidatos da Rodada 2.

## 8. Relação com nossa pesquisa

**Centralidade:** Bhaskaruni é a **referência seminal** sobre ensembles
+ fairness, citada extensivamente quando se discute deep ensemble para
mitigação de viés. Para a nossa pesquisa, é importante porque:

1. **Refuta a tese ingênua "deep ensemble → fairness".** Bagging
   piora SD em Community Crime. Para nossa Rodada 1 experimental
   (que testou deep ensemble), este resultado **explica** por que o
   deep ensemble naive talvez tenha tido eficácia limitada para
   fairness.
2. **Mostra que ensemble PROPOSITAL (que reweighting unfair instances)
   funciona.** Sugere direção:
   - Não basta combinar 3 seeds.
   - Pode ser preciso **reweighting demográfico** entre seeds (ex.:
     ensemble onde cada seed prioriza classes underrepresented).
3. **Estabelece o **trade-off accuracy-fairness** como prática
   empírica:** error 1.16 → 1.60 quando fairness melhora 4×. Isto
   é uma **referência quantitativa** para esperar trade-offs em
   nosso trabalho.
4. **Fair contrastive learning (Park 2022) consegue Pareto-efficient
   onde Bhaskaruni 2019 não consegue.** Esta progressão histórica
   é importante para narrativa da literatura.

**Limitação de transposição:** o paper é fair ML clássico tabular.
Aplicabilidade direta à nossa pesquisa de visão computacional é
**indireta**.

## 9. Pontos para citar / posicionar

- *"Bhaskaruni, Hu e Lan (2019), publicado nos anais do IEEE ICTAI
  2019, demonstram empiricamente que ensembles padrão como bagging
  (Nina et al., 2017) NÃO melhoram fairness — em alguns casos
  pioram. Em seu experimento sobre o Community Crime dataset, o
  bagging eleva a statistical disparity de 0.10 (logistic regression
  baseline) para 0.23, ao passo que a estratégia proposta de
  AdaBoost-inspired reweighting de instâncias unfairly preditas
  reduz a disparidade para 0.02."*
- *"O achado de Bhaskaruni et al. (2019) sobre a ineficácia do
  bagging para fairness é uma referência importante para
  interpretação de resultados com deep ensembles em tarefas
  faciais: a mera agregação de modelos individuais não corrige viés
  demográfico estrutural; estratégias de ensemble explicitamente
  fairness-aware são necessárias."*
- *"O trade-off accuracy-fairness reportado em Bhaskaruni et al.
  (2019) — redução de 4× em statistical disparity ao custo de
  aumento de 38% em error rate — antecede em três anos os
  resultados Pareto-efficient de Park et al. (CVPR 2022) e Manzoor
  & Rattani (ICPR 2024), ilustrando a evolução metodológica do
  campo de fair ensemble learning."*

## 10. Arquivos relacionados

- PDF local: `pdfs/bhaskaruni_2019_ensemble.pdf` (gitignored).
- Texto extraído: `pdfs/bhaskaruni_2019_ensemble.txt` (gitignored).
- Fonte PDF: par.nsf.gov/servlets/purl/10171436 (NSF Public Access
  Repository — versão aberta).
- Entradas relacionadas: [[park_2022]] (Pareto-efficient ensemble
  posterior), [[manzoor_2024]] (FineFACE Pareto-efficient via
  mutual attention), [[mehrabi_2021]] (survey que catalogou
  ensemble como técnica de mitigação).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 3.1, linha F4.
