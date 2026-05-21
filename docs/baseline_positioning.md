# Posicionamento vs. baselines da literatura — FairFace race 7-class

> Material de tese / defesa. Resposta direta à pergunta da banca *"vocês reportam F1≈0.69; isto está atrás do SOTA?"*. Documenta a ausência de benchmark canônico para a tarefa exata e justifica o porquê de nossa contribuição (atribuição matched) ser intrinsecamente robusta a essa ausência. Data: 2026-05-20.

## 1. A tarefa exata que rodamos

| Item | Valor |
|---|---|
| Dataset | FairFace (Kärkkäinen & Joo, WACV 2021) |
| Atributo | **race (7 classes)**: White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino |
| Cenário | **in-domain** (treina FairFace, testa FairFace) |
| Pré-processamento | MTCNN-aligned crops; clean dataset (multi-face removed) |
| Métrica primária | **F1 macro** (e disparity_ratio = max/min F1 per classe) |
| Resolução | **224×224** |
| Backbone padrão | ResNet-50 ImageNet pretrained |

## 2. O que o paper original do FairFace publica

Lido direto do PDF (CVF Open Access, WACV 2021, §4.2 e Tabelas 2–5):

| Aspecto | FairFace paper | Comentário |
|---|---|---|
| Backbone | **ResNet-34** | mais leve que o nosso ResNet-50 |
| Optimizador | ADAM, **lr=1e-4** | recipe simples |
| Input | não-especificado | omissão típica |
| Augmentation | não-especificada | idem |

**Resultados publicados (in-domain, FairFace → FairFace, Tabelas 2 e 3):**

| Tarefa | Acc | Observação |
|---|---|---|
| Race em White (binário: White vs outros) | **0.937** | fácil; alta separabilidade |
| **Race 4-class** (W / B / Asian / Indian, merged) | **0.754** | merge pra ser comparável com UTKFace |

> ⚠️ **Footnote crítico da Tabela 3:** *"FairFace defines 7 race categories but only 4 races (White, Black, Asian, and Indian) were used in this result to make it comparable to UTKFace."*

➡️ **O paper original do dataset NÃO publica número para race 7-class in-domain.** Eles publicam:
- Binário White / non-White
- 4-class merged
- Cross-dataset (vários)
- Gender, Age

**A tarefa exata que rodamos não tem benchmark publicado pelo próprio paper que criou o dataset.**

## 3. O que o FineFACE publica (Liu et al., 2024, arXiv 2408.16881)

Excerto verbatim do paper (§4.1 do nosso desk-check):

> *"For a fair comparison with studies in [36,12,25], we utilized ResNet50 [9] as our method's backbone CNN architecture. … e1 encompasses layers from stage 1 to stage 3, e2 stages 1 to 4, e3 stages 1 to 5. … We trained all the models using SGD with early stopping, momentum 0.9, weight decay 5e-4, mini-batch size of 16 … learning rate 0.002 with cosine annealing. We fixed the input image size as 448×448, following the common settings in existing fairness studies [7,20]."*

**Conclusões:**
1. FineFACE compara contra **outros papers de mitigação** ([36, 12, 25]), **não contra um baseline ResNet50+Linear puro**.
2. A "baseline" deles **É a arquitetura multi-expert deles** (3 experts e1/e2/e3 + atenção + concat).
3. Reportam **+1.32–1.74% acc e +67–83.6% fairness sobre essa SOTA** — números relativos, não absolutos comparáveis ao nosso plain-ResNet+Linear.
4. Recipe deles: **448px, SGD, batch=16, LR-diferencial** — totalmente diferente do nosso.

**Comparar nosso F1=0.688 (ResNet50+Linear, 224, AdamW lr=1e-3) ao headline do FineFACE é apples-to-oranges em pelo menos 5 dimensões.**

## 4. Onde isso nos posiciona honestamente

| Sistema | Backbone | Tarefa | Métrica | Valor |
|---|---|---|---|---|
| FairFace paper (2021) | ResNet-34 | race **4-class** in-domain | acc | **0.754** |
| FairFace paper (2021) | ResNet-34 | race binário (White vs not) | acc | 0.937 |
| FineFACE (2024) | ResNet-50 **multi-expert** + 448px | race 7-class FairFace | — | *baseline interno = multi-expert deles* |
| **Nosso CE+linear control** | ResNet-50 + Linear ImageNet, 224px | **race 7-class** in-domain | **f1=0.688 / acc=0.687** | |
| **Nosso ConvNeXt-T (Fator 5 parcial)** | ConvNeXt-T + Linear, 224px | race 7-class in-domain | **f1=0.697 / acc=0.695** | |

**Cruzamento de coerência:**
- Paper FairFace + ResNet-34 + 4-class = acc 0.754
- Nós + ResNet-50 + **7-class** (mais difícil; +3 grupos para discriminar) = acc 0.687
- A diferença é consistente com a dificuldade adicional do 7-class. Não há "buraco estrutural".

## 5. Vulnerabilidades reais (e respostas)

| Vulnerabilidade da banca | Resposta defensável |
|---|---|
| *"Vocês reportam 0.69 enquanto FineFACE reporta 0.7X+"* | "FineFACE compara contra outros métodos de mitigação, não baseline simples. Plain ResNet50+Linear sob nosso recipe minimalista entrega 0.69 — coerente com a literatura (paper FairFace 0.75 em 4-class mais fácil). Linha A é Δ entre fatores sob controle, invariante a offset uniforme do baseline." |
| *"Por que não 448px como na literatura de fairness?"* | "Trade-off explícito: 224px é metade da memória/compute → permite 3-seed × 5 fatores × matched protocol no orçamento 12GB. Validade da Δ-atribuição não depende da resolução absoluta." |
| *"Por que não reproduzir uma baseline publicada?"* | "Tarefa 7-class race in-domain não tem baseline canônica publicada (paper FairFace publica 4-class apenas). Os papers de mitigação não publicam plain-ResNet baseline. **Não há referência canônica para reproduzir nesta tarefa exata.**" |
| *"E o recipe FineFACE-style?"* | "Diferenças isoladas (SGD diferencial, RandomCrop, batch=16) provavelmente movem o F1 absoluto em ~2–5pp para cima, mas movem todos os fatores juntos → atribuição Δ invariante. Em escopo de defesa, fazemos sanity de recipe-anchor com 1 seed FineFACE-recipe." |

## 6. Por que isso reforça a Linha A (e não a enfraquece)

A ausência de benchmark canônico **é parte da motivação** da Linha A:
- Se houvesse um SOTA-único bem definido, o trabalho seria *mitigação* (vencer o número) — caminho saturado.
- Como **não há**, a contribuição é metodológica: **decomposição causal controlada** + **critério Pareto-aware** + **disparity_ratio honestamente nomeado** (cf. `docs/formula_desk_check.md`, `docs/checkpoint_criterion_audit.md`).

> *"We do not chase a non-existent leaderboard; we contribute the methodology to attribute, under matched control, where the bias mechanistically lives."*

## 7. Trabalho de validação cruzada planejado (escopo defesa)

1. **FairFace-recipe anchor** (1 ou 3 seeds, ResNet-34 + ADAM lr=1e-4, 224): mostra onde caímos sob o recipe do paper-pai. **Útil para incluir como linha extra na tabela de baselines.**
2. **FineFACE-recipe anchor sem multi-expert** (1 seed, ResNet-50 + 448px + SGD diferencial + Linear): isola "quanto do gap é recipe vs head architecture". Defesa.
3. **FineFACE completo** (multi-expert + 448 + SGD): reprodução da SOTA-de-mitigação no nosso pipeline; **scope de defesa**, alto custo.
4. **Cross-dataset eval** (treinar em FairFace, testar em RFW/DemogPairs) sobre os checkpoints definitivos atuais.

## 8. Resumo de uma linha

> *"Nossa F1≈0.69 em FairFace race 7-class in-domain está dentro do ballpark esperado dado o paper-pai (ResNet-34 + 4-class = 0.75) e a inexistência de baseline canônica 7-class. A Linha A (atribuição matched 3-seed entre fatores) é intrinsecamente invariante a esse offset, e a ausência de benchmark é precisamente a justificativa metodológica do trabalho."*
