# Taxonomia — nomenclatura, glossário e convenções

> Documento normativo. Estabelece os termos técnicos a serem usados
> consistentemente em todos os materiais ativos (capítulos da
> dissertação, apresentações, comunicações). Estrutura inspirada em
> Haykin, *Redes Neurais: Princípios e Práticas* (2ª ed., 2008), mas
> com preservação de termos consolidados em inglês na comunidade de
> aprendizado profundo em português brasileiro.
>
> Criado em 2026-05-25 como parte da reestruturação documental.
> Pendente preenchimento integral.

## 1. Princípio de tradução

Para cada termo técnico, adota-se a forma mais comum no discurso
acadêmico brasileiro de visão computacional / aprendizado profundo:

- **Termos com tradução consolidada em PT-BR** → usar tradução.
- **Termos cuja tradução soa artificial** (ex.: "sementes aleatórias"
  em vez de "seeds") → manter o termo em inglês.
- **Acrônimos** (CNN, GPU, SGD, etc.) → manter em inglês.

## 2. Vocabulário consolidado

### 2.1 Treinamento e otimização

| Termo (PT-BR adotado) | Tradução literal alternativa | Justificativa da escolha |
|---|---|---|
| seed (não traduzir) | semente aleatória | "seeds" é universal em ML acadêmico brasileiro |
| epoch (não traduzir) | época | "epoch" é mais usado em apresentações modernas, mas "época" é aceitável em texto formal |
| batch (não traduzir) | lote | "batch" é universal; "tamanho de batch" preferido a "tamanho de lote" |
| learning rate | taxa de aprendizagem | qualquer forma é aceita; preferimos "learning rate" em tabelas |
| loss / função de custo | função de erro | "loss" em tabelas; "função de custo" em texto corrido |
| weight decay (não traduzir) | decaimento de pesos | termo técnico estabelecido |
| early stopping (não traduzir) | parada antecipada | "early stopping" é universal |
| dropout (não traduzir) | abandono aleatório | sem tradução estabelecida |
| fine-tuning (não traduzir) | ajuste fino | qualquer forma é aceita |
| checkpoint (não traduzir) | ponto de verificação | "checkpoint" é universal |
| best.pt | melhor checkpoint | nome canônico do arquivo |

### 2.2 Arquitetura

| Termo (PT-BR adotado) | Justificativa |
|---|---|
| backbone | rede dorsal / rede de extração de características — "backbone" é mais conciso e universal |
| head | "head" em tabelas; "camada de saída" ou "classificador" em texto corrido |
| feature / vetor de características | "vetor de características" no texto corrido (Haykin §2.10) |
| embedding | "representação latente" no texto corrido; "embedding" em discussão técnica |
| layer | camada |
| residual block | bloco residual |
| skip connection | conexão de atalho |
| pooling | "pooling" em tabelas; "agrupamento" em texto corrido |
| convolution | convolução |
| transformer | transformador (raramente — geralmente mantém-se "Transformer") |

### 2.3 Métricas e avaliação

| Termo (PT-BR adotado) | Origem / Justificativa |
|---|---|
| accuracy | acurácia |
| F1 macro | F1 macro (preserva o nome técnico) |
| disparity ratio (DR) | razão de disparidade — Max-Min Ratio (Hassanpour-style; ver `00_referencias.md`) |
| inequity rate (IR) | razão de disparidade — termo legado que ainda aparece em alguns documentos antigos; preferir DR |
| coefficient of variation (CV) | coeficiente de variação |
| Gini coefficient | coeficiente de Gini |
| confusion matrix | matriz de confusão |
| recall | recall (preserva o nome técnico) |
| precision | precisão |
| TPR / FPR | TPR / FPR (siglas em inglês são universais) |
| threshold (não traduzir) | limiar | "threshold" em tabelas; "limiar" em texto corrido |

### 2.4 Aprendizado e protocolos

| Termo (PT-BR adotado) | Justificativa |
|---|---|
| matched protocol | protocolo casado |
| paired comparison | comparação pareada |
| ablation | ablação |
| baseline | linha de base / referência |
| state of the art (SOTA) | estado da arte |
| reproducibility | reprodutibilidade |
| determinism | determinismo |
| transfer learning | transferência de aprendizado |
| data augmentation | aumento de dados / data augmentation (qualquer forma) |
| Test-Time Augmentation (TTA) | aumento em tempo de teste / TTA |
| ensemble (não traduzir) | comitê de máquinas (Haykin §7.10) — mas "ensemble" é mais usado |
| deep ensemble | ensemble profundo / deep ensemble |
| calibration / temperature scaling | calibração / temperature scaling |
| oversampling / undersampling | sobreamostragem / subamostragem |
| stratified split | partição estratificada |

### 2.5 Equidade demográfica (fairness)

| Termo (PT-BR adotado) | Justificativa |
|---|---|
| fairness | equidade |
| bias (algorítmico) | viés |
| disparity | disparidade |
| protected attribute | atributo protegido |
| demographic parity | paridade demográfica |
| equal opportunity | oportunidade igual |
| equalized odds | chances equalizadas |
| intersectional analysis | análise interseccional |
| group fairness | equidade entre grupos |
| individual fairness | equidade individual |
| in-processing / pre-processing / post-processing | técnicas de mitigação (in-processing, etc. — manter em inglês) |

### 2.6 Dados e atributos faciais

| Termo (PT-BR adotado) | Justificativa |
|---|---|
| race classification | classificação de raça (no contexto da literatura específica) |
| gender classification | classificação de gênero |
| age estimation | estimativa de idade |
| facial attribute | atributo facial |
| in-domain / cross-dataset | in-domain / cross-dataset (manter em inglês) |
| MTCNN | MTCNN (sigla universal) |
| padding | padding (não traduzir) |

## 3. Convenções de citação

Detalhadas em `00_referencias.md`. Resumo:

- Citação completa em texto: **Autor1, Autor2 et al. (Ano)** — quando há 3+ autores.
- Citação com 2 autores: **Autor1 & Autor2 (Ano)**.
- Citação com 1 autor: **Sobrenome (Ano)**.
- Após primeira menção, em texto contínuo no mesmo parágrafo: **Sobrenome do primeiro autor (Ano)**.

## 4. Nomenclatura de experimentos (consolidada)

Inspirada nos documentos prévios de nomenclatura, mas a aplicabilidade
final será revisada após `06_gap.md` e `07_thesis_statement.md`. As
categorias atuais são preservadas para referência:

- **Fatores (F1–F5)** — dimensões algorítmicas testadas sob protocolo
  casado.
- **Experimentos adicionais (Exp-…)** — posicionamento absoluto vs
  literatura.
- **Análises de robustez (Rob-…)** — testes de sobrevivência do achado
  central.
- **Auditorias de hiperparâmetro (Aud-…)** — verificação empírica de
  limitadores potenciais no código.
- **Análises pós-treinamento (PT-…)** — técnicas aplicadas sobre
  checkpoints (interseccional, ensemble, TTA, calibração).

A taxonomia detalhada com mapeamento de códigos antigos para novos
pode ser consultada em `historico/nomenclatura_experimentos.md` (com
as correções de citação já aplicadas).

## 5. Pendências

- [ ] Validar este vocabulário contra o material da disciplina de
      Redes Neurais (Prof. Quiles) — pode haver convenções específicas
      a manter.
- [ ] Após `06_gap.md`, possivelmente adicionar termos específicos da
      nova abordagem.
- [ ] Refinamento de termos de fairness conforme leitura aprofundada
      dos papers verificados.
