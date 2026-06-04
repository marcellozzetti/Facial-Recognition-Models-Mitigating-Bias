---
name: hardt-2016
status_verificacao: VERIFIED
autores: [Moritz Hardt, Eric Price, Nathan Srebro]
ano: 2016
titulo: "Equality of Opportunity in Supervised Learning"
venue: "Advances in Neural Information Processing Systems 29 (NeurIPS / NIPS 2016)"
tipo_publicacao: conference
arxiv_id: "1610.02413"
doi: null
url_primario: https://arxiv.org/abs/1610.02413
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~25
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/hardt_2016_eqopp.pdf)
---

# Equality of Opportunity in Supervised Learning (Hardt, Price & Srebro, 2016)

> **Paper-fonte** das métricas Equal Opportunity (EO_h) e Equalized
> Odds (EOD) usadas em **toda a literatura subsequente** de fairness
> em ML. Citado em [[survey_mehrabi_2021]], [[park_2022]],
> [[sagawa_2020]], [[dehdashtian_2024]], [[manzoor_2024]],
> [[bhaskaruni_2019]], entre outros — agora com ficha dedicada.

## 1. Resumo do problema atacado

Em 2016, definições formais de fairness em ML eram fragmentadas:
"demographic parity" tinha problemas teóricos (forçava classifier
errar em GT para satisfazer paridade); "fairness through unawareness"
era ingênua (atributos sensíveis podem ser inferidos de proxies). O
paper introduz **critério interpretável** baseado em **igualdade de
taxas condicionais ao ground truth Y**, e mostra como **ajustar
qualquer predictor existente** para satisfazê-lo.

## 2. Método

### 2.1 Definição formal: Equal Opportunity

Predictor Ŷ satisfaz **Equal Opportunity** com respeito a A e Y se:

P(Ŷ=1 | A=0, Y=1) = P(Ŷ=1 | A=1, Y=1)

**Interpretação:** igual **True Positive Rate** (TPR) entre grupos
demográficos. Modelo deve ser igualmente bom em identificar Y=1
("oportunidade") independente de A.

### 2.2 Definição formal: Equalized Odds

Predictor Ŷ satisfaz **Equalized Odds** se:

P(Ŷ=1 | A=0, Y=y) = P(Ŷ=1 | A=1, Y=y), ∀y ∈ {0, 1}

**Interpretação:** **TPR E FPR iguais** entre grupos. Mais
restritivo que EO_h.

### 2.3 Post-hoc fix

Algoritmo para ajustar qualquer predictor pre-treinado **sem
re-treinar**, via **derived predictor** com threshold por grupo. Garante
satisfação de EO_h/EOD com perda mínima de Bayes risk.

## 3. Datasets e setup experimental

Caso de uso: **FICO score** dataset (avaliação de risco de crédito
nos EUA) — demonstra na prática que pode-se calibrar o modelo
existente para EO_h sem perder predictive power significativo.

## 4. Métricas reportadas

- **TPR / FPR / Equalized Odds Difference (EOD)** — formalização
  dada no próprio paper.
- **Bayes risk** (loss) comparado entre predictor original e ajustado.

## 5. Resultados principais

- Demonstram que **demographic parity** força classifiers a desviar do
  Bayes-optimal mesmo quando GT é uniforme entre grupos —
  matematicamente indefensável.
- **EO_h e EOD** são satisfatórios sem perda significativa de Bayes
  risk em casos realistas.
- Post-hoc adjustment é **eficiente** (closed-form).

## 6. Limitações declaradas pelos autores

- **Caso binário** (Y ∈ {0,1}, A binário). Extensão multi-classe
  não trivial.
- **Post-hoc** assume score do predictor original é informativo —
  funciona pior se predictor base é fraco.
- **Não é causal** — não distingue discriminação direta de proxy.

## 7. Limitações que identifiquei

- **Y e A binários** — para nosso caso (race 7-class + skin tone como
  conditioning), formulação direta exige adaptação. Várias generalizações
  posteriores: EOD generalizado em [[park_2022]] (somatório sobre pares),
  worst-class em [[sagawa_2020]].
- **Threshold-based** — em classification softmax como o nosso, não há
  threshold único; precisamos adaptação para argmax multi-classe.
- **Pressupõe que existe ground truth confiável Y** — para race
  classification em FairFace, Y já é socialmente construído
  (ver [[fuentes_2019]]).

## 8. Relação com nossa pesquisa

**Centralidade fundacional:** é a **fonte original** da nossa principal
métrica de fairness (EO/EOD), embora usemos forma generalizada
multi-classe (triangulação DR + worst-class F1 + CV proposta em Q05).

**Pontos de ancoragem para v3.2:**

1. **Vocabulário formal**: TPR/FPR/EO_h/EOD são **citados nas fichas**
   de Park, Sagawa, Manzoor, Dehdashtian, Bhaskaruni — Hardt 2016 é o
   ancestral comum.
2. **Justificativa da nossa métrica**: nossa razão de disparidade
   DR=max(F1)/min(F1) por classe é descendente direto do "max gap em
   TPR entre grupos" de Hardt 2016, adaptado para multi-classe.
3. **Justificativa do plano experimental**: ao reportar accuracy +
   DR + worst-class F1, estamos respeitando o **arcabouço Hardt
   2016** (TPR/FPR por grupo) generalizado para multi-classe.

## 9. Pontos para citar

- *"O critério de Equal Opportunity, proposto por Hardt, Price e
  Srebro (NeurIPS 2016), exige que a taxa de verdadeiros positivos
  seja igual entre grupos demográficos condicionada ao ground truth
  positivo. Esta formulação substitui a Demographic Parity como
  métrica preferida em fairness em ML supervisionado por evitar
  desvio do Bayes-optimal classifier."*
- *"A formulação multi-classe da razão de disparidade adotada nesta
  dissertação — max sobre classes da F1 macro dividida pelo mínimo —
  é generalização direta do critério de Hardt et al. (2016) para
  contexto de classification multi-classe."*

## 10. Arquivos relacionados

- PDF: `pdfs/hardt_2016_eqopp.pdf` (gitignored).
- Entradas relacionadas: [[survey_mehrabi_2021]] (Hardt cited como
  Definitions 1 e 2), [[park_2022]] (EOD generalizado),
  [[sagawa_2020]] (worst-group accuracy como complemento),
  [[dehdashtian_2024]] (U-FaTE para EOD em CelebA), [[bhaskaruni_2019]]
  (EO em fair LR).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Estender para multi-classe** (Y multi-categórico): autores
  reconhecem como direção futura. ✅ **Diretamente alinhado com Q05**
  e nosso plano v3.2.
- **Estudos de fairness em settings dinâmicos** (decisões em tempo).
  ❌ Fora do escopo.
- **Fairness causal**: distinguir discriminação direta de proxy.
  ❌ Fora do escopo (Mehrabi 2021 também cita).
- **Aplicar a classifiers complexos** (deep networks). ✅ **Alinhado
  com toda nossa pesquisa** — feito amplamente desde 2016.
