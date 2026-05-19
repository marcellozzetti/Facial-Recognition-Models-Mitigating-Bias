# Síntese dos 5 PDFs SOTA — veredito de novidade (fatores 1–2 + deltas)

> Material de apoio para leitura crítica (alvo: até domingo). Extração
> estruturada + posicionamento vs. nossos deltas. Complementa
> [sota_review.md](sota_review.md) §5.0 e fecha o veredito de novidade
> dos fatores 1 (dataset) e 2 (topologia). Data: 2026-05-19.

## 0. TL;DR do veredito

| Nosso delta candidato | Trabalho mais próximo (dos 5) | Sobrepõe? | Ação |
|---|---|---|---|
| **#2 Critério Pareto-aware best-epoch** | U-FaTE (CVPR'24) — caracteriza a *fronteira* utilidade-fairness | **Não** (fronteira ≠ critério de seleção em treino) | Distinguir explicitamente no related work; delta **mantém-se** |
| **Decomposição controlada cleaning×topologia** | nenhum isola fatores de treino causalmente | **Não** | Delta **mantém-se** sólido |
| **Efeito recipe-dependent da limpeza** | DSAP / Fairness-in-Details (auditam dataset, não a interação) | **Não** | Delta **mantém-se** |
| Mitigação por arquitetura (não é nosso delta) | **FineFACE (2024)** — fine-grained, +67–83% fairness | (concorrente de *mitigação*, não de *atribuição*) | Posicionar: nosso objetivo é **atribuição** (Linha A), não bater o número de mitigação |

**Conclusão:** nenhum dos 5 ataca atribuição causal controlada nem um
critério de seleção de modelo Pareto-aware. Os deltas **resistem**. O
ponto de atenção é **U-FaTE** (o adjacente mais forte ao delta #2) e
**FineFACE** (concorrente de mitigação a posicionar, não a superar).

## 1. Fichas (extração estruturada)

### FairGRAPE — arXiv 2207.10888 (ECCV 2022)
- **Título:** *FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification*.
- **Problema:** poda de rede amplifica viés demográfico na compressão.
- **Método:** scores de importância por grupo; poda preservando a razão
  de importância entre grupos; atualização iterativa.
- **Datasets:** FairFace, UTKFace, CelebA, ImageNet.
- **Resultado:** reduz a disparidade da degradação por poda em até 90%.
- **(a) Pareto** ✗ · **(b) decomp. controlada** ✗ · **(c) critério best-epoch** ✗ · **(d) limpeza** ✗
- **Posição:** intervenção *na compressão* — ortogonal. Mesmo cenário
  (FairFace, classif. de atributo) → âncora de related work do eixo
  topologia/arquitetura, **sem overlap** com nossos deltas.

### FineFACE — arXiv 2408.16881 (2024)
- **Título:** *FineFACE: Fair Facial Attribute Classification Leveraging Fine-grained Features*.
- **Método:** classif. de atributo como reconhecimento *fine-grained*;
  atenção mútua entre camadas (experts raso→profundo); sem rótulo
  demográfico no treino.
- **Resultado:** **+1,32–1,74% acc, +67–83,6% fairness sobre SOTA**;
  "equilíbrio Pareto-eficiente".
- **(a) Pareto** ~ (resultado, não busca multi-obj) · **(b)** ✗ · **(c)** ✗ · **(d)** ✗
- **Posição:** **concorrente mais direto na tarefa** (fair attribute
  classif. com números fortes). É **mitigação por arquitetura**, não
  atribuição causal nem critério de seleção. Implicação dupla: (i)
  confirma que a **alavanca está na topologia/arquitetura** (coerente
  com nosso Fator-2) e que loss-family não é (nosso Fator-3);
  (ii) a dissertação **deve posicionar** que nosso objetivo é
  *atribuir* (Linha A), não superar o número de mitigação do FineFACE.

### U-FaTE — arXiv 2404.09454 (CVPR 2024)
- **Título:** *Utility-Fairness Trade-Offs and How to Find Them*.
- **Método:** U-FaTE quantifica trade-offs ótimos; define 3 regiões no
  plano utilidade-fairness (possível/parcial/impossível); avalia
  1000+ modelos pré-treinados.
- **Achado:** a maioria das abordagens está **longe** do trade-off
  alcançável.
- **(a) Pareto/fronteira** ✓ (caracterização teórica da fronteira) · **(b)** ✗ · **(c)** ✗ · **(d)** ✗
- **Posição:** ⚠️ **o adjacente mais forte ao delta #2.** Mas é
  *caracterização da fronteira alcançável entre modelos*, **não** um
  **critério de seleção de época em HPO multi-objetivo**. Distinção a
  fazer explicitamente no related work: U-FaTE responde "onde está a
  fronteira"; nosso delta responde "como escolher a época/modelo sem
  escalarizar, evitando viés dependente de recipe". Complementares,
  não concorrentes. **É o PDF de leitura mais crítica.**

### DSAP — arXiv 2312.14626 (Information Fusion 2024)
- **Título:** *DSAP: Analyzing Bias Through Demographic Comparison of Datasets* (Demographic Similarity from Auxiliary Profiles).
- **Método:** compara composição demográfica entre datasets **sem
  rótulos demográficos explícitos**; detecta blind spots, viés de
  dataset único, shift de implantação.
- **(a)** ✗ · **(b)** ✗ · **(c)** ✗ · **(d) comparação demográfica de dataset** ✓ (foco)
- **Posição:** ferramenta de **auditoria de dataset** (espaço de dados),
  não de atribuição de fator algorítmico. Adjacente ao nosso eixo
  dataset/viés-de-cena, **sem overlap** metodológico. Atribui o viés
  "aos dados" — útil como contraponto: nosso trabalho mostra que parte
  do "viés de dados" é, na verdade, de **recipe/critério**.

### Fairness is in the details — arXiv 2504.08396 (ECML PKDD 2025)
- **Título:** *Fairness is in the details: Face Dataset Auditing*.
- **Método:** pipeline 2 fases — extração de features + **teste
  estatístico que modela a imprecisão do extrator** (não trata feature
  como verdade absoluta).
- **(a)** ✗ · **(b)** decompõe *auditoria* em fidelidade-de-extração +
  rigor estatístico (decomposição de **medição**, não de fator causal
  de treino) · **(c)** ✗ · **(d)** auditoria de dataset ✓
- **Posição:** **resonância filosófica** com nosso achado do critério
  de checkpoint — "imprecisão de medição enviesa a conclusão de
  fairness". Mesmo *espírito metodológico* (rigor de medição), aplicado
  a auditoria de dataset, não a seleção de modelo. **Sem overlap**;
  bom para citar como trabalho-irmão de rigor metodológico.

## 2. Implicações de rota (para o debate pós-leitura)

1. **Deltas resistem** — nenhum dos 5 faz atribuição causal controlada
   nem critério Pareto-aware de seleção. Reforça Linha A + paper de
   métodos (Linha B).
2. **U-FaTE é o ponto a blindar** no related work do delta #2
   (fronteira ≠ critério). Ler com atenção: a distinção precisa estar
   escrita com precisão na qualificação.
3. **FineFACE recalibra o discurso, não o objetivo:** confirma que a
   alavanca é arquitetura/topologia (nosso Fator-2) e que loss-family
   não é (nosso Fator-3). Posicionar explicitamente: *não competimos
   em "maior ganho de fairness"; atribuímos onde o ganho mora.*
4. **DSAP + Fairness-in-Details** sustentam que "viés de dados" é um
   espaço reconhecido — e nosso achado (parte do efeito atribuído a
   dados é recipe/critério) é um **contraponto valioso**, não uma
   redundância.
5. **Próximo experimento — Fator 4 (contrastivo)** ganha peso: com
   loss-family = null e topologia = alavanca, o paradigma contrastivo
   é o próximo candidato a "o que de fato ajuda" sob o mesmo protocolo
   casado 3-seed + critério correto.

## 3. Guia de leitura (prioridade até domingo)

| Ordem | PDF | Por que / o que procurar |
|---|---|---|
| 1 | **U-FaTE** 2404.09454 | Crítico p/ delta #2. Buscar: como definem a fronteira; é seleção de modelo ou caracterização? confirmar que NÃO é critério best-epoch em HPO. |
| 2 | **FineFACE** 2408.16881 | Concorrente de mitigação. Buscar: datasets exatos (FairFace?), backbone, definição de "Pareto-efficient", métricas de fairness usadas. |
| 3 | FairGRAPE 2207.10888 | Related work topologia. Buscar: backbone, se isola fatores (não deve). |
| 4 | Fairness-in-details 2504.08396 | Trabalho-irmão de rigor. Buscar: a tese de "imprecisão de medição enviesa fairness" — paralelo ao nosso critério. |
| 5 | DSAP 2312.14626 | Contexto dataset. Buscar: como atribui viés "aos dados"; usar como contraponto. |

> Limitação: extração feita das páginas/abstracts arXiv (não do PDF
> completo). Datasets/backbones exatos de FineFACE, U-FaTE e
> Fairness-in-details precisam de confirmação na leitura do texto
> integral — marcado acima onde importa para o veredito.
