# Reunião de kick-off com o orientador — 2026-05-11

**Programa:** Mestrado em Ciência da Computação — Unifesp / ICT
**Orientador:** Prof. Marcos
**Reuniões recorrentes:** toda segunda-feira (acompanhamento + alinhamento)
**Prazo de qualificação:** **24 de agosto de 2026** (~15 semanas a partir de hoje)
**Estado dos artefatos:** repositório `face-bias` v0.2.0.dev0, rodada limpa dos 11 experimentos do MBA já documentada em [docs/clean_results.md](clean_results.md).

---

## ⚠️ Diretriz nº 0 — TESE PRECISA SER POSITIVA (a mais importante)

> **Orientação direta do Prof. Marcos:** *"A tese não pode falar que o undersampling não funciona. Falar negativamente vai exigir que eu tenha testado todas as milhares de condições — se alguém da banca encontrar um artigo que funcione, derruba a tese. A tese precisa ser positiva (provando algo que funciona)."*

**Implicação:** **todo o framing da proposta atual precisa ser invertido**. A pergunta de pesquisa, o título de trabalho, as hipóteses — tudo organizado em torno de "balancear não basta" precisa ser **reorientado para uma afirmação positiva**.

### Por que isso é decisivo

- **Afirmação negativa** (tese atual): "X não funciona". Bar de prova é **exaustivo**: precisa cobrir todas as condições plausíveis. Banca pode derrubar com **um único contraexemplo**.
- **Afirmação positiva** (tese reformulada): "Y funciona, e aqui está como". Bar de prova é **suficiente**: precisa mostrar Y funciona em condições específicas e bem definidas. Banca só pode questionar a magnitude ou a generalização.

### Antes × Depois

| | Tese atual (precisa virar) | Tese reformulada (positiva) |
|---|---|---|
| **Pergunta** | "Se balancear não basta, o que basta?" | "Como alcançar reconhecimento facial demograficamente equitativo combinando arquitetura otimizada + aprendizado contrastivo + funções de perda?" |
| **Hipótese central** | H0 falsificada: balanceamento → equidade | H1 afirmativa: combinação X reduz IR de ~1,76 para ~1,3 mantendo acurácia |
| **Título proposto** | "Beyond Class Balancing: ..." | "Achieving Demographic Equity in Face Recognition via [solução proposta]" |
| **Diagnóstico dos 11 experimentos** | **A evidência da tese** | **A motivação/baseline** que minha solução vai superar |
| **Risco de banca** | Alto (contraexemplo derruba) | Baixo (precisam questionar magnitude) |

### O que muda na prática

- O trabalho experimental **NÃO muda** — continuo fazendo o mesmo (cabeça MLP + Optuna + contrastivo + losses).
- O **diagnóstico empírico** (clean run, IR=1,76) **NÃO some** — passa a ser **a motivação** que justifica a busca da solução.
- O **enquadramento narrativo** muda: em vez de "olha o problema persistente", a tese diz "olha a solução que descobri para este problema conhecido".

### Tarefa imediata

Reescrever na próxima semana:
1. **Pergunta de pesquisa** (§3.2 da PROPOSTA) — afirmativa, não diagnóstica.
2. **Hipóteses H1-H5** — afirmar que algo funciona, com targets numéricos concretos.
3. **Título de trabalho** — começar com verbo afirmativo: "Achieving", "Reducing", "Combining para alcançar...".
4. **Sumário executivo** — abrir com a proposta de solução, não com o diagnóstico do problema.
5. **§3.4 Contribuições** — destacar o método proposto como contribuição #1.

---

## Sumário executivo da reunião

O orientador validou a pergunta de pesquisa central (*"se balancear não basta, o que basta?"*) e o diagnóstico empírico, mas redirecionou o trabalho em quatro frentes técnicas que precisam ser endereçadas **antes da qualificação**:

1. **Estabelecer o TOP Line / SOTA** de reconhecimento facial e de classificação racial para que toda comparação tenha referência absoluta clara.
2. **Aprofundar a arquitetura do classificador**: hoje uso uma saída linear `Linear(2048, 7)` direto sobre os features da ResNet50. Ele questionou se essa é a melhor escolha e pediu **uma bateria de experimentos com cabeça densa (MLP)** sobre a ResNet, com **Optuna** otimizando os hiperparâmetros das camadas ocultas.
3. **Limpeza adicional do dataset**: manter FairFace, mas **remover imagens que o MTCNN detecta múltiplas faces** — o rótulo de raça pode estar incorreto nessas amostras (ground truth ambíguo). Não trocar dataset.
4. **Investigar paradigmas de aprendizado contrastivo** como alternativa/complemento ao classificador tradicional: **SimCLR**, **SupCon (Supervised SimCLR)** e **CLIP**.

Também aprovou explicitamente:
- **Variação de funções de perda** para benchmarking (= PQ1 da proposta atual).

---

## Diretrizes do orientador — versão fiel + tradução técnica

### Diretriz 1: definir o TOP Line / SOTA de reconhecimento de imagens

> "Preciso ter definido o TOP Line (SOTA) de reconhecimento de imagens."

**Tradução técnica:**
Estabelecer a referência absoluta de performance contra a qual qualquer resultado meu vai ser comparado. Isso significa duas coisas distintas:

- **SOTA de reconhecimento facial geral**: benchmarks consagrados (LFW, IJB-C, MegaFace, RFW). Hoje os números chave são ArcFace ~99,5% no LFW, AdaFace/TransFace++ ~98% no IJB-C.
- **SOTA de classificação racial em FairFace especificamente**: Karkkainen & Joo (2021) reportaram ~74% no paper original; trabalhos recentes (FairFace Augmented, 2024) chegam a 76–78%.

**Pergunta aberta a esclarecer com ele na próxima segunda**: ele quer que eu compare meu trabalho contra qual SOTA — o de FR de identidade (LFW/IJB-C) ou o de classificação demográfica (FairFace/RFW)? Provavelmente os dois, mas com pesos diferentes.

### Diretriz 2: rever arquitetura do classificador

> "Você está tendo saída linear de classificação e o que faz com os neurônios das camadas densas durante o treinamento? Roda uma nova bateria de testes com uma rede densa de saída da ResNet."

**Tradução técnica:**
Hoje o `LResNet50E_IR` faz:
```
backbone (ResNet50, 2048-D features) → Linear(2048, num_classes) → logits
```

Ou seja: **uma única transformação linear** entre os features ricos da ResNet e as classes. Não há camada oculta, não há não-linearidade adicional, não há regularização explícita além do dropout único.

O orientador está sugerindo testar uma **cabeça MLP** (Multi-Layer Perceptron) entre o backbone e a saída:

```
backbone (2048-D) → Linear(2048, H) → ReLU → Dropout → Linear(H, num_classes) → logits
```

Variações a investigar com Optuna:
- Tamanho da camada oculta H ∈ {128, 256, 512, 1024}
- Número de camadas ocultas ∈ {1, 2, 3}
- Função de ativação ∈ {ReLU, GeLU, SiLU}
- Dropout ∈ {0, 0.2, 0.3, 0.5}
- Normalização ∈ {None, BatchNorm1d, LayerNorm}

### Diretriz 3: usar Optuna como critério de avaliação

> "Considerar o Optuna como critério de avaliação do que melhor trabalha com camadas ocultas."

**Tradução técnica:**
[Optuna](https://optuna.org/) é um framework de otimização de hiperparâmetros baseado em TPE/CMA-ES. Em vez de tentar combinações de hyperparams manualmente, ele constrói uma distribuição sobre o espaço de busca e propõe iterativamente combinações que provavelmente maximizam o objetivo (no nosso caso, IR baixo + acc alta).

**O que isso envolve:**
- Adicionar `optuna` ao `pyproject.toml`.
- Criar um script `scripts/hpo_head.py` que:
  - Define o espaço de busca (camadas, dropouts, ativações).
  - Roda trials em paralelo na GPU.
  - Usa critério multiobjetivo (Pareto): maximizar F1 macro **e** minimizar IR simultaneamente.
- ~30–50 trials × ~30 min cada = **~15–25h de GPU**. Cabe no orçamento local com calendário ajustado.

### Diretriz 4: dataset — remover imagens com múltiplas faces (MTCNN)

> "Não recomendo mudar o dataset, apenas remover as imagens que passam pelo MTCNN que possuem mais de uma foto — porque pode ser que o gabarito esteja incorreto."

**Tradução técnica:**
FairFace tem um rótulo de raça por imagem, mas se a imagem tem **2+ rostos** detectados pelo MTCNN, ambiguidade entra: qual dos rostos é o sujeito que recebeu o label?

**Estado atual no `face_bias.preprocessing.pipeline`:**
- Já existe a flag `max_faces_per_image: 1` no config, que limita a 1 face por imagem (mantendo a primeira detectada).
- **Mas isso não dropa a imagem do CSV** — ela continua no dataset com a primeira face. Se a primeira detectada não for a do sujeito rotulado, o rótulo fica errado.

**O que precisa ser feito:**
- Criar um script `scripts/audit_multiface.py` que reprocessa o FairFace com MTCNN, anota quantas faces cada imagem tem, e **gera uma lista de exclusão** dos com >1 face.
- Atualizar a `setup_dataset` em `face_bias/data/dataset.py` para aceitar uma `exclusion_list` configurável.
- Documentar: quantas das 97.282 imagens são removidas. Estimativa pelo Cap. 4 do MBA: ~63% de detecção em rotações, mas em frontais limpas deve ser ~95% de imagens com 1 só face — ou seja, ~5% × 97k ≈ 5.000 imagens.

### Diretriz 5: avaliar arquiteturas de aprendizado contrastivo

> "Recomendou avaliar diferentes arquiteturas, colocando um aprendizado contrastivo supervisionado — SimCLR, Sup-SimCLR, CLIP."

**Tradução técnica:**
Hoje treino o modelo via **classificação direta**: dado (imagem, raça), aprende a prever raça. O orientador está abrindo um eixo paralelo: usar **representações aprendidas por contraste**.

Três paradigmas a estudar:

| Método | Tipo | Como funciona | Por que pode ajudar |
|---|---|---|---|
| **SimCLR** (Chen et al., 2020) | Auto-supervisionado | Maximiza similaridade entre duas augmentações da MESMA imagem | Backbone aprende features semânticas sem usar rótulos — útil quando rótulos são ruidosos |
| **SupCon** (Khosla et al., 2020) | Supervisionado | Maximiza similaridade entre imagens da MESMA classe | Cria clusters por classe no espaço de embedding **explicitamente** — atacando o gap representacional que o t-SNE do MBA mostrou |
| **CLIP** (Radford et al., 2021) | Cross-modal | Alinha embeddings de imagem com embeddings de texto | Permite usar prompts ("a face of a Latino_Hispanic person") como âncoras semânticas |

**Decisão metodológica importante:** essas arquiteturas substituem o classificador convencional ou são pré-treinamentos do backbone? Provavelmente **pré-treinamentos**: usar SimCLR/SupCon para treinar o ResNet50, depois colocar um classificador linear ou MLP em cima.

### Diretriz 6: variar funções de perda (APROVADO)

> "Aprovou alterar as funções de perda para benchmark."

Esse era o PQ1 da proposta atual (AdaFace, MagFace, KP-RPE). **Mantém-se inalterado** — orientador OK.

### Diretriz 7: convenção organizacional — `docs/` para tudo

> "Todo o material que formos gerando precisa ficar na pasta `docs`."

**Tradução prática:** todo artefato textual produzido durante o mestrado (relatórios, revisões de literatura, atas de reunião, análises, rascunhos de capítulo) **vive em `docs/`**, não na raiz do repositório.

**Estado atual:**
- ✅ `docs/PROPOSTA_MESTRADO.md` movido para `docs/` em 11/05/2026 (commit 5f8c1be).
- ✅ `docs/clean_results.md`, `docs/smoke_results.md`, `docs/exp01_vs_mba.md`, `docs/meeting_prep_2026-05-11.md`, `docs/meeting_2026-05-11_kickoff.md` — já estão em `docs/`.
- ⚠️ `REVIEW_AND_PLAN.md` — permanece na raiz por enquanto. Pode ser movido para `docs/` em commit separado por consistência (a decidir).

**Convenção a partir de hoje:**
- Novos relatórios técnicos → `docs/<nome>.md`
- Atas de reunião → `docs/meeting_<YYYY-MM-DD>_<tag>.md`
- Revisões de literatura → `docs/literature_<tema>.md`
- Análises de experimento → `docs/exp_<id>_<tema>.md`

### Diretriz 8: busca semântica sobre ~900 referências de FairFace

> "Fazer uma pesquisa semântica sobre todas as ~900 referências de artigos sobre o FairFace que utilizei como base, para certificar que não estou fazendo nada repetitivo e vai ter valor para a comunidade."

**Tradução prática:** posicionar este trabalho na literatura existente sobre FairFace. Garantir que:

1. Nenhum trabalho publicado já fez exatamente a combinação proposta (MLP head + Optuna + contrastivo + losses).
2. Os achados positivos esperados (ex.: "SupCon reduz IR em FairFace") não estão publicados em forma idêntica.
3. Existem **lacunas reais** que esta tese preenche.

**Por que isso é decisivo para a qualificação:**
- Banca pode questionar novidade. Sem busca semântica documentada, defesa é frágil.
- Reforça o argumento de "tese positiva" — uma vez que se mostra que ninguém combinou os 3 eixos antes, a contribuição fica clara.
- A própria busca é entregável **publicável** como tabela de literatura adjacente no capítulo de fundamentação.

**Pendência crítica para destravar essa frente:**
Onde estão as ~900 referências? Possíveis fontes:
- Export do Zotero / Mendeley
- Resultado de busca no Google Scholar / Semantic Scholar com filtro "FairFace"
- BibTeX consolidado da revisão de literatura do MBA
- Lista do próprio orientador (perguntar em 18/05)

**Fluxo proposto para a busca semântica** (a implementar quando o corpus chegar):

```
1. Input: arquivo com 900 refs (BibTeX, CSV, ou planilha com título + abstract)
2. Enriquecimento: usar API do Semantic Scholar (gratuita) para puxar abstracts faltantes
3. Embeddings: sentence-transformers (modelo all-MiniLM-L6-v2 ou BAAI/bge-base-en-v1.5)
4. Indexação: FAISS local (~5 MB para 900 papers, roda em segundos)
5. Queries semânticas alinhadas com a proposta:
     Q1. "MLP head architecture for race classification"
     Q2. "Optuna HPO face recognition fairness"
     Q3. "Supervised contrastive learning FairFace"
     Q4. "AdaFace MagFace demographic equity FairFace"
     Q5. "Combined loss adaptive + contrastive + architecture optimization"
     Q6. "Multi-face filtering label noise FairFace"
6. Output: docs/literature_semantic_audit.md com:
     - Top-20 papers mais similares por query
     - Cluster analysis (papers que se agrupam por tema)
     - Tabela de lacunas confirmadas vs trabalhos sobrepostos
     - Citação direta dos papers que precisam ser referenciados na tese
```

**Estimativa de esforço:** 3-5 dias úteis após receber o corpus.

---

## O que sai do escopo

A proposta original tinha **PQ2 (DCFace augmentation)** e **PQ3 (combinação loss × dados sintéticos)** como linhas principais. O orientador **não mencionou** DCFace nem dados sintéticos. Há três interpretações:

1. Ele esqueceu / vai retomar em outra sessão.
2. Ele considera fora de escopo para qualificação e quer foco em arquitetura.
3. Ele aprova mas é segunda prioridade.

**Pergunta crítica para a próxima segunda**: DCFace augmentation continua no escopo da qualificação, ou é trabalho futuro pós-qualificação?

**Por enquanto, plano até dia 24/08 NÃO inclui DCFace** — foco nos eixos que ele mandou.

---

## Plano de ação — 15 semanas até 24/08

Cada bloco tem **entregável principal** + **discussão na segunda seguinte**.

### Semanas 1-2 (12/05 → 25/05) — Setup + dataset audit + SOTA review

| # | Atividade | Saída concreta |
|---|---|---|
| 1.1 | Levantar SOTA de FR (LFW/IJB-C/MegaFace) e de FairFace classificação | `docs/sota_review.md` com tabela de benchmarks |
| 1.2 | Auditar FairFace para imagens com múltiplas faces | `scripts/audit_multiface.py` + `data/raw/fairface/multi_face_exclusion.csv` |
| 1.3 | Atualizar `setup_dataset` para aceitar lista de exclusão | PR no repo + 81 → 84 testes |
| 1.4 | Re-rodar Exp 5 (baseline) sobre dataset limpo (sem multi-face) | Tabela `before / after` no `docs/dataset_clean.md` |

**Entrega na reunião de 25/05:** apresentar SOTA + impacto numérico da limpeza no baseline.

### Semanas 3-4 (26/05 → 08/06) — Cabeça MLP + Optuna

| # | Atividade | Saída concreta |
|---|---|---|
| 2.1 | Refatorar `LResNet50E_IR` para suportar `head=mlp` configurável | PR no repo |
| 2.2 | Integrar Optuna no pipeline (`scripts/hpo_head.py`) | Script multiobjetivo (F1 ↑, IR ↓) |
| 2.3 | Rodar HPO com ~30 trials | `docs/hpo_head_results.md` + 4 arquiteturas finalistas |
| 2.4 | Comparar **linear (atual)** vs **melhor MLP** sobre o baseline corrigido | Tabela comparativa, 3 seeds |

**Entrega na reunião de 08/06:** "Encontrei a cabeça ideal. IR caiu de X para Y, F1 subiu de A para B."

### Semanas 5-8 (09/06 → 06/07) — Aprendizado contrastivo

| # | Atividade | Saída concreta |
|---|---|---|
| 3.1 | Estudar SimCLR/SupCon/CLIP (2 papers + tutoriais) | `docs/contrastive_review.md` |
| 3.2 | Implementar SimCLR pretrain do ResNet50 sobre FairFace | Módulo `face_bias/contrastive/simclr.py` |
| 3.3 | Implementar SupCon (provavelmente o mais promissor) | Módulo `face_bias/contrastive/supcon.py` |
| 3.4 | Avaliar CLIP zero-shot e fine-tuned em FairFace | Notebook + métricas no `docs/contrastive_results.md` |
| 3.5 | Comparar todas as três contra o melhor classificador do Sem 3-4 | Tabela final |

**Entrega na reunião de 06/07:** "SupCon dá IR=X, SimCLR=Y, CLIP=Z, melhor classificador=W."

### Semanas 9-10 (07/07 → 20/07) — Benchmark de loss functions

| # | Atividade | Saída concreta |
|---|---|---|
| 4.1 | Implementar AdaFace e MagFace na melhor arquitetura encontrada | PR no repo |
| 4.2 | Rodar 3 seeds × {CE, AdaFace, MagFace, ArcFace} = 12 treinos | Tabela média ± desvio |
| 4.3 | Análise representacional (t-SNE pré/pós) | Figuras finais |

**Entrega na reunião de 20/07:** "Loss-adaptive bate baseline com margem X% no IR."

### Semanas 11-12 (21/07 → 03/08) — Síntese + análise

| # | Atividade | Saída concreta |
|---|---|---|
| 5.1 | Consolidar todas as tabelas e gráficos | `docs/qualification_results.md` |
| 5.2 | Análise interseccional (raça × gênero × idade no melhor modelo) | Heatmaps + discussão |
| 5.3 | Grad-CAM em amostras representativas | Figuras qualitativas |

**Entrega na reunião de 03/08:** rascunho da seção de resultados completa.

### Semanas 13-15 (04/08 → 24/08) — Escrita da qualificação

| # | Atividade | Saída concreta |
|---|---|---|
| 6.1 | Escrita dos capítulos 1-2 (Introdução + Fundamentação) | Draft #1 |
| 6.2 | Sessão de revisão intensiva com orientador | Draft #2 |
| 6.3 | Escrita dos capítulos 3-5 (Metodologia + Resultados + Conclusão) | Draft #3 |
| 6.4 | Polish final + revisão de português acadêmico | **Versão final 24/08** |

**Entrega 24/08:** **qualificação completa entregue**.

---

## Perguntas em aberto a esclarecer com orientador

Para a próxima segunda (18/05) trazer respostas próprias + perguntas:

1. **SOTA-alvo**: ele quer comparação contra FR clássico (LFW/IJB-C) ou contra FairFace racial (FairFace/RFW), ou ambos com pesos diferentes?
2. **DCFace augmentation**: está fora de escopo da qualificação ou é segunda prioridade?
3. **Cross-dataset (RFW)**: ele considera necessário para qualificação?
4. **Paper concomitante**: ainda mira submissão a WACV Fair-CV em set/26, ou foco é 100% qualificação?
5. **Múltiplas seeds**: ele exige quantas seeds por condição? Mínimo defensável para banca de qualificação na Unifesp.
6. **Coautoria**: caso haja paper, prática do programa para coautoria com orientador.

---

## Formato sugerido para as reuniões semanais

Cada segunda, levar:

1. **Slide / página única de status** com:
   - O que foi entregue na semana passada
   - Bloqueios encontrados
   - Plano da semana seguinte
2. **Tabela atualizada de resultados** (cumulativa)
3. **Lista de perguntas técnicas** geradas durante a semana

Reuniões devem ter **time-boxed em 45-60 min** para serem produtivas semanalmente.

---

## Reescopo da PROPOSTA_MESTRADO.md — pendente

A proposta atual tem PQ1/PQ2/PQ3 organizados em torno de "se balancear não basta, o que basta?" — **framing negativo que precisa virar afirmativo** (ver Diretriz nº 0 no topo).

**Pergunta de pesquisa reformulada (rascunho a refinar):**

> Como uma arquitetura de classificação **otimizada por Optuna**, combinada com **pré-treinamento contrastivo** e **funções de perda quality-adaptive**, pode produzir reconhecimento facial racialmente equitativo sobre FairFace, **reduzindo o Inequity Rate (F1) do baseline de 1,76 para abaixo de 1,3**, mantendo acurácia agregada acima de 0,67?

**Hipóteses afirmativas (todas com targets numéricos concretos):**

- **H1** — Cabeça MLP otimizada por Optuna **reduz** IR do baseline em ≥15% (de 1,76 para ≤1,50) mantendo ou melhorando acurácia.
- **H2** — Pré-treinamento contrastivo supervisionado (SupCon) **eleva** F1 da classe pior (Latino_Hispanic) em ≥10pp (de 0,47 para ≥0,57).
- **H3** — Loss-adaptive (AdaFace ou MagFace) sobre arquitetura otimizada **reduz** IR para ≤1,30.
- **H4** — A combinação dos três (cabeça MLP + contrastivo + loss-adaptive) é **a melhor solução**, com IR ≤1,25 e F1 macro ≥0,70.
- **H5** — Limpeza de imagens multi-face reduz ruído de rótulo e **melhora** todas as métricas em 1-3pp.

**Eixos atualizados da tese:**

- **PQ1** — Qual configuração de cabeça densa otimizada produz melhor trade-off acc × IR?
- **PQ2** — Qual paradigma contrastivo (SimCLR, SupCon, CLIP) maximiza equidade demográfica?
- **PQ3** — Qual função de perda quality-adaptive funciona melhor na arquitetura ótima?
- **PQ4** — A combinação dos eixos é aditiva ou tem interação positiva?
- **(PQ5? — a confirmar)** — DCFace augmentation contribui marginalmente sobre a melhor combinação?

**Decisão:** **aguardar próxima reunião (18/05)** para alinhar com orientador antes de reescrever a proposta. Esta semana foco em:
1. SOTA review + dataset audit (Diretrizes 1, 2 e 4).
2. **Rascunho da pergunta de pesquisa positiva** para validação com ele.

---

*Documento gerado em 2026-05-11 após reunião de kick-off com Prof. Marcos.*
*Próxima reunião: 2026-05-18 (segunda).*
