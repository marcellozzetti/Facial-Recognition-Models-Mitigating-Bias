# Auditoria de código — critério de seleção de modelo (achado dominante)

> Material de tese. Auditoria completa do pipeline em execução, pedida
> para investigar "por que não consigo aumentar as métricas". Data:
> 2026-05-18.

## TL;DR

As métricas **estavam sendo atingidas** (F1 ≈ 0,69); o pipeline as
**descartava na seleção de modelo**. O `best.pt` era escolhido por
**`val_loss` mínimo**, que para heads de margem é **anti-correlacionado**
com F1 — e o viés é **dependente da família de loss** (corrompe
ArcFace/AdaFace/MagFace, quase não toca softmax+CE). Isso contaminou
toda comparação de loss do projeto e **invalidou** o veredito anterior
"CE vence margem / AdaFace instável".

## 1. Sintoma e prova

`exp_f3_adaface_s02` atingiu `val_f1=0,678` na ép.8, mas o F1 reportado
(`best.pt`) foi 0,585. Quantificação nos 3 seeds AdaFace concluídos:

| Run | Selecionado (min val_loss) | Melhor real (max F1) | F1 descartado |
|---|---|---|---|
| adaface s01 | ép.1: f1=0,449 IR=6,85 | ép.6: f1=**0,662** IR=1,71 | **+0,213** |
| adaface s02 | ép.3: f1=0,573 IR=2,47 | ép.8: f1=**0,678** IR=1,72 | **+0,104** |
| adaface s42 | ép.2: f1=0,552 IR=2,05 | ép.7: f1=**0,673** IR=1,71 | **+0,121** |

Em todos, o melhor F1 é a **última época** → o early-stopping também
matou o treino com F1 ainda subindo (não convergiu).

## 2. Causa raiz (dois bugs de medição, compondo)

- **B1 — checkpoint por `min val_loss`** (`trainer.py`,
  `ModelCheckpoint(mode="min")` + `fit()` passando `val_metrics["loss"]`).
  Heads de margem emitem, na avaliação, `inference_logits` = cosseno
  escalado **sem margem**; a CE-eval daí é mínima cedo (modelo ruim) e
  **sobe** enquanto F1/acc sobem.
- **B2 — early-stopping em `val_loss`, patience=5**: a paciência estoura
  na subida do val_loss e o treino para com F1 ainda crescendo.
- O scheduler `CosineAnnealingWarmRestarts (T_0=8)` **não** é o problema
  (runs param em ép.6–8, antes do 1º restart).

## 3. Recompute casado (sem re-rodar) — `scripts/recompute_checkpoint_criterion.py`

`history.json` preserva métricas por época de **todos** os runs.
Re-seleção por **max val_f1_macro** uniforme (volta a ser casado):

| grupo (dataset limpo) | n | acc OLD→NOVO | f1 OLD→NOVO | IR OLD→NOVO |
|---|---|---|---|---|
| CE+linear | 3 | 0,667→**0,687** | 0,667→**0,688** | 1,72→1,70 |
| ArcFace | 3 | 0,565→0,572 | 0,543→0,549 | 2,99→3,30 |
| AdaFace | 3 | 0,540→**0,669** | 0,525→**0,671** | 3,79→**1,71** |
| MagFace | 0 | (batch em curso) | — | — |

**Achados:**
1. **AdaFace transformado**: f1 0,525±0,067 (parecia instável/ruim) →
   **0,671±0,008** (apertado, estável). O "outlier s01 IR=11,9" era
   100 % artefato (era a época 1).
2. **Viés dependente de loss**: o ganho ao corrigir o critério foi
   **AdaFace +0,146 ≫ CE +0,021** f1. A contaminação não é uniforme —
   é exatamente o que o critério ingênuo causa.
3. **Veredito anterior RETRATADO**: sob critério casado correto,
   AdaFace (f1 0,671, IR 1,71) é **competitivo com CE** (0,688, 1,70);
   só ArcFace é genuinamente pior (0,549, IR 3,30). MagFace pendente.

## 4. Correção (runs futuros corretos)

- `TrainingConfig.checkpoint_metric ∈ {val_loss, val_f1_macro,
  val_accuracy}`, **default `val_f1_macro`** (métrica-tarefa).
- `Trainer(monitor=...)` deriva o `mode` (min p/ loss, max p/ métrica);
  `ModelCheckpoint` e `EarlyStopping` passam a usar a métrica monitorada.
- `cli/train.py` casa o `mode` do early-stopping ao monitor e loga.
- Testes 39/39 verdes (trainer/callbacks/full-pipeline/margin heads).

> Implicação metodológica: comparações casadas exigem o **mesmo
> critério** em todos os grupos. Recuperar o Fator 3 e o baseline
> `dataset_factor` exige re-seleção uniforme (feito via recompute na
> validação; números de **teste** definitivos saem dos runs futuros já
> com o critério correto, ou de `last.pt` onde coincide com max-F1).

## 5. Outras frentes auditadas — SEM erro algorítmico

- **Dataset/split**: label = `race` consistente; split estratificado,
  seed-fixo, 80/10/10, reprodutível e casado entre runs; `LabelEncoder`
  consistente; sem vazamento (tarefa é por-imagem). ✅
- **Métricas**: `f1_macro` com `zero_division=0`; IR = max/min (o `inf`
  era correto — sinalizava grupo colapsado, não bug); alinhamento
  pred/target em lockstep no loop de avaliação. ✅
- **Caveats de validade (não-bugs, registrados por honestidade):**
  - `_filter_existing_images` descarta imagens sem alinhamento MTCNN;
    se a taxa de falha for demograficamente assimétrica, enviesa as
    métricas de fairness — **quantificar taxa de descarte por raça**
    (cf. `docs/dataset_audit_findings.md`).
  - Augmentation de treino é mínima (só hflip) — **alavanca** para subir
    generalização, não defeito.

## 6. Conexão com a contribuição da tese

Este é o **delta candidato #2** da [sota_review.md](sota_review.md) §5
(critério Pareto-aware best-epoch) **comprovado empiricamente no próprio
projeto**: o critério ingênuo de best-epoch **enviesa conclusões causais
de forma dependente da família de loss**. Reforça (não enfraquece) a
Linha A — e vira um achado de Fator 3 por si só. O caso MagFace
([magface_diagnosis.md](magface_diagnosis.md)) + este caso checkpoint
formam dois exemplos independentes da mesma tese: **ablação ingênua
confunde efeito-do-fator com efeito-de-configuração-quebrada**.
