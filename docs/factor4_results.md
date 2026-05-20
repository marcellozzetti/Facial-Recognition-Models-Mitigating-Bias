# Fator 4 (paradigma de aprendizado) — resultados 3-seed casados

> Material de tese. Isola a contribuição do **paradigma contrastivo
> (SupCon canônico, one-stage joint)** à acurácia e disparidade
> demográfica, sob protocolo casado idêntico aos outros fatores.
> Data: 2026-05-20.

## 1. Tabela principal — DEFINITIVA

Re-run definitivo (`outputs/definitive/factor4/`, 3 runs × 25 ép.),
critério `val_f1_macro`, backbone ResNet-50 ImageNet, head linear, CE
+ 0,5·SupCon (joint). Recompute via
`scripts/recompute_checkpoint_criterion.py --definitive`.

| Loss / Paradigma (clean, casado) | Acc | **F1 macro** | disparity_ratio ↓ |
|---|---|---|---|
| **CE + linear** (controle) | 0.6865 ± 0.0019 | **0.6877 ± 0.0017** | **1.697 ± 0.033** |
| **SupCon (Fator 4)** | **0.6790 ± 0.0047** | **0.6816 ± 0.0036** | **1.699 ± 0.078** |
| AdaFace (Fator 3) | 0.6755 ± 0.0077 | 0.6765 ± 0.0067 | 1.697 ± 0.057 |
| MagFace (Fator 3) | 0.6706 ± 0.0020 | 0.6715 ± 0.0030 | 1.795 ± 0.062 |
| ArcFace (Fator 3) | 0.5717 ± 0.0059 | 0.5486 ± 0.0086 | 3.301 ± 0.365 |

## 2. Veredito (regra 1σ, casada)

1. **SupCon NÃO é alavanca de equidade.** IR essencialmente idêntico
   ao CE (Δ +0,002 ≪ 1σ). Acurácia marginalmente abaixo (Δ −0,006,
   ~1,5σ — borderline).
2. SupCon ≈ AdaFace ≈ MagFace ≈ CE na faixa F1 0,67–0,69 / IR ~1,70 —
   contribuição marginal do *paradigma* indistinguível do *loss*.
3. ⚠️ **SupCon tem o maior dp de IR** (±0,078; spread entre seeds 0,13)
   — mais sensível a seed que os demais. Sinaliza instabilidade
   subjacente, possivelmente a "amostragem majoritária dominante de
   positivos" descrita pela FSCL.

## 3. Ancoragem na literatura (predição confirmada)

A **FSCL (Park et al., CVPR 2022)** existe *justamente porque* o
SupCon canônico **não é naturalmente justo** — grupos majoritários
dominam a amostragem de positivos/negativos. Nosso resultado **confirma
empiricamente a premissa da FSCL sob protocolo casado 3-seed em FairFace
race**: SupCon canônico não move a equidade. Forte para a banca:
predição da literatura quantificada com rigor metodológico.

A FSCL e variantes (FairCL, SSL fair 2024) ficam **naturalmente
motivadas** para o programa de defesa (PLANO §5, matriz cruzada) como
"canônico não basta → contrastivo modificado para fairness".

## 4. Padrão consolidando-se nos 5 fatores

| Fator | Resultado | Move equidade? |
|---|---|---|
| 1 — Dataset (limpeza) | F1 +1,35pp (CE) | ❌ (IR não-significativo) |
| 2 — Topologia (HPO MLP) | IR −0,11 (≫1σ) | ✅ **modesto** |
| 3 — Loss family | CE≈Ada≈Mag; ArcFace pior | ❌ |
| 4 — Paradigma (SupCon) | ≡ CE em IR | ❌ |
| 5 — Backbone | (não rodado — eixo restante) | ❓ |

**Padrão emergente:** nos 4 fatores algoritmicos testados sob protocolo
casado, apenas **topologia** moveu equidade significativamente. Loss e
paradigma são **nulls atribuição-grade** (ancorados na literatura).
**Fator 5 (backbone)** é o único eixo restante com expectativa real de
mover — FineFACE / LVFace / ViT em FairFace literatura sugerem espaço
de manobra arquitetural maior que o explorado.

A **maior alavanca medida no projeto** continua sendo metodológica: o
**critério Pareto-aware best-epoch** (corrige +0,15 F1 no AdaFace; ver
[checkpoint_criterion_audit.md](checkpoint_criterion_audit.md)).

## 5. Caveats honestos

- Forma do SupCon: **single-view in-batch** (escolha de casamento de
  protocolo, ver `models/contrastive.py` docstring). Full **two-view**
  pertence à defesa (PLANO §5). Nulo aqui **não** invalida
  contrastivo-em-geral; invalida especificamente *"canônico single-view
  como termo conjunto, sob protocolo casado"*.
- Determinismo confirmado bit-a-bit (seed 42 ép.1 idêntica entre
  sanity, batch1 e batch2).
- 11 commits do arco completo desta sessão (MagFace fix →
  auditoria → desk-check → backbone rename → disparity_ratio →
  AdaFace init → re-run definitivo → SupCon infra+sanity+batch).

## 6. Procedência

- Configs: `configs/experiments_factor4/exp_f4_supcon_s{01,02,42}.yaml`
- Saídas: `outputs/definitive/factor4/` (3 runs, results+history)
- Recompute: `scripts/recompute_checkpoint_criterion.py --definitive`
- Implementação: `src/face_bias/models/contrastive.py`,
  `src/face_bias/training/trainer.py::_train_loss`
- Sanity: pegou 2 bugs antes do batch (single-forward perf;
  state_dict CLIs) — commits `58715cb`, `45f13be`
