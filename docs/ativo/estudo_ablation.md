# Estudo — uso correto do termo "ablation" em ML

**Data:** 31/08/2026
**Motivação:** feedback do orientador — verificar se o termo *ablation*,
usado na dissertação para descrever a comparação A/B/C, está sendo
empregado no sentido rigoroso da metodologia científica ou se estamos
inferindo algo diferente do que os artigos originais chamam de ablation.

**TL;DR:** nosso uso está **correto e rigoroso**. Podemos manter o
termo, mas vale usar linguagem defensiva na apresentação para preemptar
a pergunta da banca.

---

## 1. Definição canônica

Fonte: [Wikipedia — Ablation (AI)](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)) + Baeldung + literatura primária.

> **Ablation study** = remoção de UM componente do sistema, mantendo
> tudo o mais constante (*ceteris paribus*), com o objetivo de medir a
> contribuição isolada desse componente para o desempenho.

A analogia vem da biologia (remoção de tecidos/órgãos) e da neurociência
(cirurgia ablativa cerebral). O termo foi adotado em redes neurais
justamente por analogia à remoção de neurônios/camadas.

### 1.1 Método rigoroso

1. **Baseline fixo:** define-se um modelo completo (todas as partes) como referência.
2. **Ceteris paribus:** varia-se **apenas um elemento por vez**.
3. **Mesma métrica, mesmo dataset, mesmo protocolo:** o que muda é só o componente em análise.
4. **Interpretação:** a diferença de desempenho é atribuída ao componente removido/alterado.

---

## 2. Como usamos na dissertação

Definição atual do §Ablation (Cap. 4 §4.7):

| ID | Configuração | O que muda |
|---|---|---|
| **A** | ConvNeXt-T puro (baseline) | — (controle) |
| **B** | ConvNeXt-T + FiLM (sinal MST 10-dim) | **+ mecanismo de condicionamento** |
| **C** | ConvNeXt-T + FiLM (sinal CLIP-text 512-dim) | **+ mudança do sinal condicionante** (mantém FiLM) |

**Elementos mantidos idênticos entre A, B e C:**
- Backbone (ConvNeXt-T, mesmos pesos ImageNet iniciais)
- Dataset (FairFace, mesmos splits, mesmas seeds 42/1/2)
- Otimizador (AdamW, LR backbone/head fixos)
- Cronograma de treinamento (mesmo número de épocas, mesmo early stopping)
- Protocolo de avaliação (Cenário A + B, mesma triangulação de métricas)
- Sementes (mesmo trio 42, 1, 2)

**Elementos variados:**
- **A → B:** adição do mecanismo FiLM (isola *efeito do condicionamento arquitetural*).
- **B → C:** troca do sinal condicionante (isola *efeito da natureza discreta vs semântica do sinal*).

Cada par de comparações **isola uma única variável de decisão**. Isso
atende exatamente o critério canônico.

---

## 3. Análise da possível confusão

### 3.1 Ambiguidade popular

Em jargão informal de conferências e blogs, "ablation" às vezes é usado
como sinônimo de "comparação entre variantes". Isso é impreciso — nem
toda comparação entre variantes é ablation rigoroso; ablation exige
**controle sistemático** (ceteris paribus).

### 3.2 Onde estaríamos errados

Seríamos imprecisos se, por exemplo:
- Comparássemos ConvNeXt-T+FiLM contra **ResNet+FiLM** e chamássemos de "ablation" — aí estaríamos variando 2 coisas ao mesmo tempo (backbone E presença do condicionamento).
- Comparássemos configurações treinadas com hiperparâmetros diferentes.
- Comparássemos com número diferente de sementes ou épocas.

**Nada disso é o que fazemos.** Nosso protocolo (§4.6.5 "Detalhes de
treinamento") fixa explicitamente os hiperparâmetros entre configurações.

### 3.3 Nomenclatura alternativa (se banca preferir)

Termos aceitáveis para o que fazemos, caso a banca sugira substituição:
- **Estudo controlado de componentes** (mais preciso, menos usado)
- **Ablation arquitetural** (qualifica que a variação é no plano da arquitetura, não em hiperparâmetros — usar isso na apresentação)
- **Ablation do sinal condicionante** (para o par B→C)

Recomendação: manter "ablation" no texto (é a nomenclatura consagrada
na literatura de ML e reconhecida pela banca), mas **acrescentar sempre
"arquitetural"** para deixar claro o escopo.

---

## 4. Preparação para arguição

### Se a banca perguntar: "Você está usando ablation no sentido estrito?"

**Resposta preparada:**
> "Sim. A definição canônica exige *ceteris paribus* — variação de um
> único elemento por vez. Nas comparações A→B, mantenho tudo constante
> (backbone ConvNeXt-T, dataset FairFace, splits, sementes 42/1/2,
> AdamW com LR fixo, mesmo protocolo de avaliação) e acrescento apenas
> o mecanismo FiLM com sinal MST. Na comparação B→C, mantenho FiLM
> presente e troco apenas o sinal condicionante (MST direto vs
> embedding CLIP-text). Cada par isola uma única variável de decisão.
> Se a banca preferir, posso adotar o termo mais qualificado
> 'ablation arquitetural', enfatizando que a variação é no plano da
> arquitetura e não em hiperparâmetros."

### Se a banca perguntar: "Por que não removeu FiLM camada por camada?"

**Resposta preparada:**
> "Ablation camada-por-camada faz sentido quando o interesse é medir a
> contribuição relativa de cada componente arquitetural interno
> (ex.: em ResNet, comparar blocos com/sem skip connection). Aqui, o
> componente em investigação é o mecanismo FiLM inserido como um todo,
> condicionado no vetor MST — desativar FiLM em alguns estágios e não
> em outros produziria configurações intermediárias sem contrapartida
> na literatura. Optamos pela ablation completa (com/sem FiLM em todos
> os 4 estágios) que espelha os precedentes de Perez et al. (2018) e
> Dumoulin et al. (2018)."

### Se a banca perguntar: "E a variante gated (porta sigmoide)?"

**Resposta preparada:**
> "A implementação canônica de FiLM já é afim (γ⊙F + β). Variantes
> internas das MLPs geradoras — presença ou não de gating,
> profundidade das camadas — são escolhas empíricas dentro da
> configuração B, não configurações separadas do ablation. Foi
> exatamente essa a orientação recebida na reunião de agosto/2026:
> não inflar o ablation com micro-variações internas do FiLM."

---

## 5. Conclusão

- **Uso correto:** sim, nosso ablation é rigoroso (ceteris paribus atendido).
- **Ação recomendada:** trocar "ablation" por "ablation arquitetural" no texto para deixar mais claro o escopo — edição cirúrgica em §4.7 e no Slide 21 da apresentação.
- **Preparação da banca:** as 3 respostas preparadas cobrem os ângulos previsíveis de arguição.

**Sources principais:**
- [Wikipedia — Ablation (AI)](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence))
- [Baeldung — Ablation Study in ML](https://www.baeldung.com/cs/ml-ablation-study)
- Perez et al. (2018), FiLM: Visual Reasoning with a General Conditioning Layer, AAAI
- Dumoulin et al. (2018), Feature-wise transformations, Distill
