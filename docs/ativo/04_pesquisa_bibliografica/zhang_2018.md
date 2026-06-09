---
name: zhang-2018
status_verificacao: VERIFIED
autores: [Brian Hu Zhang, Blake Lemoine, Margaret Mitchell]
ano: 2018
titulo: "Mitigating Unwanted Biases with Adversarial Learning"
venue: "AAAI/ACM Conference on AI, Ethics, and Society (AIES 2018)"
tipo_publicacao: conference
arxiv_id: "1801.07593"
doi: "10.1145/3278721.3278779"
url_primario: https://arxiv.org/abs/1801.07593
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~25
lente_disrupcao: metodologica
fonte_leitura: PDF integral baixado de arXiv (pdfs/zhang_2018_adversarial.pdf)
---

# Mitigating Unwanted Biases with Adversarial Learning (Zhang, Lemoine & Mitchell, 2018)

> **Adversarial debiasing operacional** — formula explicitamente como
> adversarial training pode satisfazer **Demographic Parity, Equal
> Opportunity OU Equalized Odds** dependendo do tipo de input do
> adversary. Trabalho aplicado em word embeddings e classifiers em
> tabular. Paralelo a [[madras_2018]] (LAFTR) em formulação, mais
> orientado a engenharia.

## 1. Resumo do problema atacado

Como integrar mitigação de viés em pipelines de produção sem alterar
significativamente o modelo base? Zhang et al. (Google) propõem
**framework adversarial mínimo** que pode ser adicionado a qualquer
modelo supervisionado treinado por gradient descent.

## 2. Método

### 2.1 Arquitetura

- **Predictor**: produz Ŷ a partir de X (modelo principal).
- **Adversary**: produz Â a partir de Ŷ (saída do predictor) e
  opcionalmente Y (ground truth).

Treinamento min-max:

- Predictor minimiza L_pred(Ŷ, Y) E maximiza L_adv(Â, A).
- Adversary minimiza L_adv(Â, A).

### 2.2 Escolha do input do adversary determina noção de fairness

- **Adversary recebe só Ŷ**: força independência Ŷ ⊥ A →
  **Demographic Parity**.
- **Adversary recebe Ŷ E Y**: força Ŷ ⊥ A | Y →
  **Equalized Odds**.
- **Adversary recebe Ŷ, Y só quando Y=1**: força TPR igual →
  **Equal Opportunity**.

**Insight chave**: a estrutura do adversary controla qual fairness é
satisfeita — análogo ao Teorema central de [[madras_2018]].

### 2.3 Gradiente do predictor

Crucial: para garantir treinamento estável, **gradiente do predictor
sobre L_pred é projetado** ortogonalmente ao gradiente sobre L_adv
(predictor não aprende a "ajudar" o adversary, apenas evita ser
predito por ele).

## 3. Datasets e setup experimental

- **Word embeddings**: Word2Vec — Y = analogia (man:king ::
  woman:?), A = gender stereotype.
- **UCI Adult**: Y = >50K, A = sex.

## 4. Métricas reportadas

- **Accuracy** em Y.
- **Bias** = mutual information de Â com A (mensurada
  empiricamente).

## 5. Resultados principais

- **Word embeddings**: análogos completados com **menos evidência de
  stereotyping** (e.g., "man : computer programmer :: woman : X" não
  produz "homemaker" como resposta enviesada).
- **Adult**: reduz EOD significativamente mantendo accuracy.
- **Projeção do gradiente** é crítica — sem ela treinamento colapsa.

## 6. Limitações declaradas pelos autores

- Atributo A binário no experimento principal.
- Tabular + word embeddings (não imagens).
- Adversarial training é **sensível a hiperparâmetros** (learning
  rates ratio entre predictor e adversary).

## 7. Limitações que identifiquei

- Para FairFace race 7-class, **adversary multi-classe é frágil**
  como em LAFTR. Mitigação possível: usar one-vs-rest.
- **Não testa transferência** explicitamente — diferentemente de
  Madras 2018.
- **Não cobre conditional architectures** — adversary atua sobre Ŷ,
  não sobre features intermediárias. Para v3.2 (FiLM-based), preciso
  combinar Zhang com mecanismo conditional.

## 8. Relação com nossa pesquisa

**Papel na v3.2:**

Zhang 2018 é candidato a **baseline** de mitigação algorítmica para
o Capítulo 2 da v3.2, ao lado de FSCL+ ([[park_2022]]) e Group DRO
([[sagawa_2020]]).

**Vantagens de Zhang sobre Park/Sagawa:**

1. **Mais simples de implementar** — apenas adicionar adversary head
   ao modelo existente.
2. **Menos demanda de regularização** — não exige strong ℓ2 como
   Sagawa.
3. **Garantia formal** de qual fairness é satisfeita (escolha do
   input do adversary).

**Desvantagens:**

1. **Instabilidade adversarial** documentada.
2. **Não testado em FairFace race 7-class** — mesmo gap que outros.
3. **Não cobre conditional** — precisaria combinar com FiLM para v3.2.

## 9. Pontos para citar

- *"Zhang, Lemoine e Mitchell (Google, AIES 2018) demonstraram que
  adversarial debiasing pode satisfazer Demographic Parity, Equal
  Opportunity OU Equalized Odds dependendo da estrutura do
  adversary — uma reformulação operacional do framework teórico de
  Madras et al. (2018)."*
- *"A escolha de baseline adversarial nesta dissertação considera
  Zhang et al. (2018) por sua simplicidade de implementação e
  estabilidade superior quando o atributo protegido é multi-classe,
  embora reconheça-se a desvantagem do mecanismo agir sobre output
  (Ŷ) e não sobre features intermediárias — limitação resolvida pela
  combinação com camadas FiLM (Perez et al., 2018)."*

## 10. Arquivos relacionados

- PDF: `pdfs/zhang_2018_adversarial.pdf` (gitignored).
- Entradas relacionadas: [[madras_2018]] (LAFTR — paralelo teórico),
  [[zemel_2013]] (paradigma fundador), [[hardt_2016]] (Definições
  EO_h/EOD usadas), [[park_2022]] (FSCL — alternativa contrastiva).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Estender para A multi-classe**. ✅ **Alinhado com v3.2** (race
  7-class).
- **Aplicar a arquiteturas neurais profundas** além de tabular. ✅
  Realizado por LAFTR e literatura posterior.
- **Combinar com pre-processing** (data augmentation). ⚠ Direção
  paralela.
- **Investigar fairness em pipelines de produção**. ✅ Adoção em
  Google interno e outras companhias (BAIR Berkeley, IBM AIF360 inclui
  Zhang 2018 como implementação canônica).

## 12. Análise crítica do método

### (a) Rigor formal

- **Framework adversarial mínimo** matematicamente claro: predictor
  + adversary com objetivos opostos.
- **Insight chave**: estrutura do adversary controla qual noção de
  fairness é satisfeita — paralelo ao Teorema central de Madras.
- **Projeção ortogonal do gradiente** é crítica para estabilidade —
  sem ela, treinamento colapsa.
- **Limitação**: heurística operacional, não teorema formal.

### (b) Reprodutibilidade

- ✅ Implementação canônica em AIF360 (IBM) — adoção ampla.
- ✅ Word embeddings + UCI Adult são datasets padrão.
- ⚠ Sensibilidade a hiperparâmetros (learning rates ratio entre
  predictor e adversary) reconhecida.
- ⚠ Sem multi-seed reportado.

### (c) Aplicabilidade ao pipeline v3.2

- **Candidato a baseline forte** de mitigação algorítmica para Cap 2.
- **Mais simples de implementar** que LAFTR (Madras 2018) — adversary
  atua sobre Ŷ, não features.
- **Limitação**: não cobre conditional architectures — combinação
  com FiLM seria mecanismo novo.
- **Multi-class adversarial frágil** para race 7-class — mitigação
  via one-vs-rest.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Adversary recebe Ŷ (não features) | ✅ Justificada — simplicidade operacional |
| 3 estruturas para 3 noções de fairness | ✅ Justificada — analogia a Madras |
| Projeção ortogonal do gradiente | ✅ Justificada — empíricamente crítica |
| Word embeddings + UCI tabular | ✅ Choice — escopo histórico |
| A binário | ❌ Limitação reconhecida |
| Sem teste em imagens | ⚠ Limitação de escopo (2018) |

### (e) Conexão com R5/R6

- [[madras_2018]] LAFTR: paralelo teórico. Zhang é mais simples e
  operacional; Madras mais formal.
- [[hardt_2016]]: estruturas de adversary de Zhang implementam
  Demographic Parity, EO, EOD de Hardt.
- [[zemel_2013]]: ambos são pós-Zemel; Zhang adota paradigma
  adversarial vs representational de Zemel.
- [[perez_2018]] FiLM: ortogonal — FiLM modula features, Zhang
  modula output. Combinação testável.
- [[park_2022]] FSCL+: alternativa contrastive não-adversarial.
- **Implicação para v3.2**: Zhang é **candidato a baseline** de
  Cap 2. Implementação via AIF360 simplifica adoção.
