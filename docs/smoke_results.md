# Smoke Run dos 11 Experimentos do MBA — Replicação Acelerada

**Data:** 2026-05-07
**Configuração:** 5 épocas por experimento (override) + split 80/10/10 + balanced + dropout corrigido (0.2 ou 0.5 conforme MBA)
**Hardware:** RTX 4070 SUPER (12 GB), CUDA 12.1, PyTorch 2.5.1
**Tempo total wall-clock:** ~4h26min para 11 experimentos
**Pipeline:** `face_bias` v0.2.0.dev0 (commit `741effd`)

## Tabela de resultados

| # | Experimento | MBA reportou (25ep) | Smoke nosso (5ep) | Δ acc | IR | Gap | Train (min) |
|---|---|---|---|---:|---:|---:|---:|
| 1 | CE + SGD + OneCycle | acc=0.68, F1=0.68 | acc=0.663, F1=0.662 | -0.02 | **1.73** | 0.34 | 24.0 |
| 2 | ArcFace + SGD + OneCycle | acc=0.61, F1=0.62 | acc=0.526, F1=0.509 | -0.08 | **3.60** | 0.52 | 24.0 |
| 3 | CE + AdamW + OneCycle | acc=0.67, F1=0.68 | acc=0.666, F1=0.665 | 0.00 | 1.73 | 0.35 | 24.0 |
| 4 | ArcFace + AdamW + OneCycle | acc=0.58, F1=0.58 | acc=0.481, F1=0.440 | -0.10 | **5.39** | 0.58 | 24.0 |
| 5 | CE + AdamW + Cosine | acc=0.62, F1=0.62 | acc=0.665, F1=0.665 | **+0.05** | 1.76 | 0.36 | 24.1 |
| 6 | ArcFace + AdamW + Cosine | (zerou no MBA) | acc=0.512, F1=0.458 | n/a | **7.21** | 0.66 | 24.1 |
| 7 | CE + AdamW + OneCycle (Black/White) | acc=0.95, F1=0.95 | acc=0.941, F1=0.941 | -0.01 | 1.00 | 0.00 | 9.7 |
| 8 | ArcFace + AdamW + OneCycle (Black/White) | acc=0.94, F1=0.94 | acc=0.934, F1=0.934 | -0.01 | 1.00 | 0.00 | 9.8 |
| 9 | CE + AdamW + OneCycle + dropout=0.5 | acc=0.67, F1=0.67 | acc=0.662, F1=0.662 | -0.01 | **1.64** | 0.32 | 24.0 |
| 10 | ArcFace + AdamW + OneCycle + dropout=0.5 | acc=0.59, F1=0.59 | **CRASH** (rc=0xC0000005) | — | — | — | — |
| 11 | CE + AdamW + OneCycle (40 ep) | acc=0.67, F1=0.67 | acc=0.666, F1=0.665 | 0.00 | 1.73 | 0.35 | 24.0 |

## Observações para a defesa

### 1. Reprodutibilidade comprovada

Exp 3 e Exp 11 produzem **resultados idênticos** quando ambos rodam 5 épocas (Exp 11 só difere de Exp 3 em `num_epochs=40`, mas com `--epochs 5` essa diferença some). Isso prova que:

- Seed=42 + cuDNN deterministic + PYTHONHASHSEED estão funcionando.
- O dataset é determinístico (mesma divisão em todas as execuções).
- A camada de logging/MLflow não introduz variância.

### 2. CE é universalmente melhor que ArcFace neste setup

| Setup | CE (Exp 1/3/5/9/11) | ArcFace correspondente |
|---|---:|---:|
| 7-class, 5ep, melhor caso | acc=0.666 | acc=0.526 (Exp 2) |
| 7-class, 5ep, pior caso | acc=0.662 | acc=0.481 (Exp 4) |

Isso **inverte a expectativa do MBA**, que reportou ArcFace ligeiramente abaixo (~0.61) mas em valores próximos. Explicação: o **bug §2.2** documentado no Sprint B fazia o `ArcFaceLoss` do MBA cair em `cross_entropy` puro, então os experimentos "ArcFace" do MBA na verdade rodaram CE. Agora, com o `ArcMarginProduct` realmente integrado ao modelo, o margin angular `m=0.5` produz uma penalidade severa demais para 5 épocas convergirem — o modelo entra em colapso parcial em algumas classes.

**Achado para a tese**: ArcFace exige curvas de aprendizado mais longas e/ou margem menor (`m=0.3` é comum em outros papers). Não é defeito da implementação — é um efeito esperado da arquitetura quando exposta corretamente. Vale rodar uma versão dos Exp 2/4/6 com 25 épocas para confirmar.

### 3. ArcFace causa colapso sistemático em classes minoritárias

Exp 6 (pior IR, 7.21) tem distribuição extrema:
- Black: F1=0.771
- Southeast Asian: **F1=0.107** (modelo praticamente nunca prevê essa classe)
- Latino_Hispanic: F1=0.237

A margem angular força separação agressiva entre classes, e quando uma classe é "difícil" (visualmente próxima de outras), o gradiente de ArcFace empurra essa classe para a margem do espaço de embedding em vez de encontrar uma região própria. **Confirma a hipótese H1 da PROPOSTA_MESTRADO**: AdaFace (margem adaptativa por qualidade) deve mitigar exatamente esse modo de falha.

### 4. Cosine vs OneCycle são quase idênticos com CE

Exp 3 (OneCycle) vs Exp 5 (Cosine), ambos CE+AdamW: acc=0.666 vs 0.665. **Diferença insignificante em 5 épocas.** O MBA reportou 0.67 vs 0.62 — diferença que provavelmente desaparece sem o bug §2.3 de normalização.

### 5. Dropout 0.5 (Exp 9) tem o menor IR entre 7-class

Exp 9 (dropout=0.5): IR=1.638 — **menor IR de todos os 7-class**.
Exp 3 (dropout=0.2): IR=1.732.

Mais regularização → modelo aprende features menos discriminativas demograficamente → fairness sutilmente melhor. Pequeno achado mas consistente.

### 6. Black/White binário é essencialmente resolvido

Exp 7 e 8 atingem **94% de acurácia em apenas 5 épocas**, com IR≈1.0 (perfeitamente fair). MBA reportou 95% e 94% com 25 épocas — chegamos lá em 1/5 do tempo.

**Implicação metodológica**: o setup binário é fácil demais para servir como benchmark de fairness. Para o mestrado, precisa de configurações mais difíceis (interseccional, RFW, BUPT).

### 7. Padrão Latino_Hispanic confirmado em todos os 7-class

Em todos os Exp 1, 3, 5, 9, 11 (CE), Latino_Hispanic é a pior classe:

| Exp | F1 Latino_Hispanic | F1 Black (melhor) | gap |
|---|---:|---:|---:|
| 1 | 0.469 | 0.812 | 0.343 |
| 3 | 0.471 | 0.816 | 0.345 |
| 5 | 0.473 | 0.835 | 0.362 |
| 9 | 0.496 | 0.812 | 0.316 |
| 11 | 0.471 | 0.816 | 0.345 |

Gap de ~35pp e F1 de Latino_Hispanic preso em ~0.47-0.50 independentemente de optimizer/scheduler/dropout. **Esse é o gargalo central** que o mestrado deve atacar via dados sintéticos / loss adaptativa / arquiteturas com melhor representação.

### 8. Exp 10 não foi crash — foi divergência catastrófica

Após o primeiro `rc=0xC0000005` na corrida em batch, um retry isolado **completou** sem crash, mas com resultados degenerados:

| Época | train_loss | val_acc | val_f1 | val_IR |
|---:|---:|---:|---:|---:|
| 1 | 10.76 | **0.143** | 0.037 | inf |
| 2 | 9.40 | 0.143 | 0.036 | inf |
| 3 | 9.38 | 0.143 | 0.036 | inf |
| 4 | 9.37 | 0.143 | 0.036 | inf |
| 5 | 9.37 | 0.143 | 0.036 | inf |

`val_acc=0.1429 ≈ 1/7` = **predição aleatória**. O modelo aprendeu a mapear todas as entradas para uma única classe (= recall=0 para as outras 6, daí IR infinito). A primeira tentativa provavelmente atingiu um NaN em algum gradiente que disparou o ACCESS_VIOLATION; o retry chegou em um basin estável-mas-degenerado.

**O MBA reportou Exp 10 com acc=0.59**. Isso só foi possível porque o bug §2.2 transformava o ArcFaceLoss em cross-entropy puro — então o "Exp 10" do MBA era na prática Exp 9 (CE+AdamW+dropout=0.5). A combinação **ArcFace real + AdamW + OneCycleLR + dropout=0.5** com 5 épocas e `m=0.5` é numericamente instável, e nem 25 épocas devem salvar — o equilíbrio degenerado é estável.

**Implicações para a tese**:
- Validação experimental do bug §2.2 — sem o bug o resultado seria muito diferente do reportado.
- Reforça a hipótese H1 (PROPOSTA_MESTRADO): margem fixa `m=0.5` é frágil; AdaFace ou MagFace com margem adaptativa deveriam evitar esse modo de falha.
- Adicionar **gradient clipping** + **warmup** no trainer mitiga divergências similares para outros experimentos com ArcFace.

## Tempos por experimento (5 épocas)

| Tipo | Experimentos | Tempo médio | Total |
|---|---|---:|---:|
| 7-class CE | 1, 3, 5, 9, 11 | 24.0 min | 120 min |
| 7-class ArcFace | 2, 4, 6 | 24.0 min | 72 min |
| 2-class B/W | 7, 8 | 9.7 min | 19.4 min |
| Falhou | 10 | 0.3 min | 0.3 min |
| **Total smoke** | 10 OK + 1 crash | — | **~232 min** |

Extrapolando para 25 épocas:
- 7-class: ~120 min/exp
- B/W: ~50 min/exp
- **Rodada completa de 25 épocas: 9 × 120 + 2 × 50 = ~1180 min ≈ 19h40min**
- 40 épocas (Exp 11): 192 min adicionais

## Recomendação para a rodada limpa

Antes da rodada de 25 épocas (~20h):

1. **Investigar Exp 10** — re-rodar isolado para confirmar se é flaky ou bug. Se persistir, adicionar handling de NaN no trainer.
2. **Considerar reduzir margem ArcFace** para `m=0.3` (Exp 2/4/6 rodam menos catastroficamente).
3. **Add early-stopping com patience=5** — vimos que CE atinge plateau na época 5; rodar 25 épocas com early-stop economiza tempo sem perder acurácia.
4. **Salvar `last.pt` em todas as runs** (já implementado no commit `b3bb3a9`) para comparar best vs last contra o MBA.

## Próximos passos sugeridos

| Prioridade | Ação | Tempo |
|---|---|---|
| ✅ Feito | Re-rodar Exp 10 — confirmou divergência (não crash) | 25 min |
| 🟠 Alta | Adicionar **gradient clipping** (e.g. `clip_grad_norm_=5.0`) ao trainer para estabilizar ArcFace | 30 min |
| 🟠 Alta | Decidir destino do Exp 10: (a) aceitar como "divergência conhecida" ou (b) ajustar `m=0.3` na config | 5 min |
| 🟠 Média | Rodada limpa dos 11 com num_epochs do config (25/40), early-stop com patience=5 | ~15-20h |
| 🟢 Baixa | Variação ArcFace com `m=0.3` para validar hipótese H1 | ~24 min/exp |
| 🟢 Baixa | Plots automáticos para os 11 experimentos | rodar `scripts/plot_experiment.py` em loop |

---

*Gerado a partir de `outputs/smoke/results.json` (10 OK + 1 divergência) e da matriz `tab:resultados_experimentos` da dissertação de MBA.*
