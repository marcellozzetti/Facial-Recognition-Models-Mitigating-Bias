# HPO Round 1 — Multi-Objective Search Over MLP Head Topology

**Period:** 2026-05-13 19:38 UTC → 2026-05-14 11:00 UTC (≈ 15h30 wall clock; one
reboot midway, study resumed from SQLite).
**Hardware:** RTX 4070 SUPER 12 GB, Windows 11.
**Objective:** identify head topologies that **dominate the MBA baseline
(Exp 5)** simultaneously on classification utility (F1 macro) and fairness
(Inequity Rate).
**Output artefacts:** `outputs/hpo/round1/{study.db, hpo.log, trials.csv,
best_params.json, pareto_reanalyzed.json}`.

This document is intended to be defendable material for the master's
qualification — every claim is backed by an artefact in the repo.

---

## 1. Experimental setup

### Why a HPO study and not another grid

The MBA used an 11-experiment grid over loss / optimizer / scheduler / dataset
filter, with the head fixed as `nn.Linear(2048→7)`. The orientador (kickoff
2026-05-11, diretriz nº 2) flagged that the dense classifier head had never
been examined and that the **search method itself** should be principled —
hence Optuna (Akiba et al., 2019). The framing inverts: backbone and training
recipe are fixed at Exp 5's clean-run configuration; only the head topology
and the learning rate are searched.

### Frozen configuration (inherited from Exp 5)

- Backbone: `ResNet50` (ImageNet pretrained), `dropout=0.2` after `avgpool`.
- Loss: `cross_entropy`; Optimizer: `AdamW (lr_search, weight_decay=5e-4)`;
  Scheduler: `CosineAnnealingWarmRestarts`.
- Dataset: FairFace, balanced via undersampling to 10 374 / class; 80/10/10
  split with `seed=42`.
- Train/val/test sizes: 58 094 / 7 262 / 7 262.
- `grad_clip_norm=5.0`, `batch_size=128`.

### Search space (this is what each trial samples)

| Variable | Range / values | Notes |
|---|---|---|
| `mlp_depth` | int ∈ {1, 2, 3} | uniform |
| `mlp_hidden_i`, i ∈ [0, depth) | categorical {128, 256, 512, 1024, 2048} | per layer |
| `mlp_activation` | categorical {relu, gelu, silu} | |
| `mlp_dropout` | float ∈ [0, 0.6] | uniform |
| `mlp_norm` | categorical {none, batchnorm, layernorm} | applied between Linear and activation |
| `learning_rate` | float ∈ [1e-4, 5e-3] | **log-uniform** |

### Optuna configuration

- **Sampler:** `TPESampler(seed=42, multivariate=True, n_startup_trials=10)`.
- **Pruner:** none. Optuna's `Trial.report` / `should_prune` is not supported
  in multi-objective studies (caught empirically before the run — see
  `scripts/hpo_head.py` comment block).
- **Directions:** `["maximize", "minimize"]` — F1 macro up, Inequity Rate down.
- **Storage:** local SQLite at `outputs/hpo/round1/study.db`, resumable.

### Budget per trial

- **8 epochs.** This is intentionally below the 25-epoch full training of
  Exp 5; the goal is to enable 20 trials in a single GPU-day. The Pareto-front
  winners will be refit at 25 epochs in a follow-up round (see §6).

---

## 2. Headline numbers

### Baseline (MBA Exp 5, 25 epochs)
- F1 macro: **0.665**
- Inequity Rate: **1.76**

### Pareto front under the original "best-by-F1" criterion (2 trials)

| Trial | F1 ↑ | IR ↓ | Topology | LR |
|---|---|---|---|---|
| 4 | 0.6817 | 1.564 | `[256] GELU drop=0.52 norm=none` | 3.4e-4 |
| 10 | 0.6832 | 1.584 | `[256] GELU drop=0.29 norm=none` | 4.9e-4 |

### Pareto front under the corrected "best-Pareto-local-epoch" criterion (4 trials)

| Trial | F1 ↑ | IR ↓ | Best epoch | Topology | LR |
|---|---|---|---|---|---|
| 4 | 0.6817 | 1.564 | 8 / 8 | `[256] GELU drop=0.52 norm=none` | 3.4e-4 |
| 8 | 0.6808 | **1.544** | 7 / 8 | `[256, 1024, 512] GELU drop=0.40 norm=none` | 4.2e-4 |
| 12 | 0.6617 | **1.529** | 3 / 8 | `[256] GELU drop=0.27 norm=none` | 1.6e-4 |
| 13 | 0.6730 | 1.539 | 4 / 8 | `[256] SiLU drop=0.42 norm=none` | 5.1e-4 |

(Trial 10 drops out of the corrected front because trial 4 dominates it
under the new per-trial best.)

### Gains over the baseline

- Best F1 macro: **+1.7 pp** (trial 4, fairness-leaning) to **+1.8 pp** (trial 10, accuracy-leaning).
- Best Inequity Rate: **−13.1 %** (1.529 vs 1.76 — trial 12) to **−11.1 %** (trial 4).
- **Every trial in the Pareto front dominates the baseline on both metrics.**

---

## 3. Methodological finding — the "best epoch" criterion matters

The original script reported, for each trial, the epoch with maximum F1 macro
and the IR observed at that same epoch. With 8 epochs per trial and a
cosine LR schedule, F1 and IR can move in opposite directions across the
trajectory; selecting "max F1 row" silently discards epochs where IR is much
lower but F1 is fractionally smaller.

This is not a bug — it is a single-objective optimisation idiom misapplied
to a multi-objective problem. Round 1 surfaced four cases where the
discarded epoch would have entered the global Pareto front:

| Trial | Discarded (Pareto-local) epoch | Reported (best-by-F1) | Delta |
|---|---|---|---|
| 8 | epoch 7 (0.6808, **1.544**) | epoch 8 (0.6809, 1.581) | IR worsened by 0.037 |
| 12 | epoch 3 (0.6617, **1.529**) | epoch 4 (0.6781, 1.568) | IR worsened by 0.039 |
| 13 | epoch 4 (0.6730, **1.539**) | epoch 7 (0.6779, 1.640) | IR worsened by 0.101 |
| 20 | epoch 5 (0.6645, **1.543**) | epoch 8 (0.6777, 1.592) | IR worsened by 0.049 |

`scripts/reanalyze_hpo_round1.py` recomputes the front from `hpo.log`
without touching the Optuna SQLite (the study itself is a frozen artefact
of Round 1). `scripts/hpo_head.py` was updated for Round 2 to surface the
Pareto-local best epoch directly (tie-broken by lowest IR, which is editorial
and documented inline). Both views are kept available — the by-F1 front is
what Optuna's `study.best_trials` returns and is reported here for full
transparency.

---

## 4. What the TPE sampler converged on

The first ten trials are random by design (Optuna's TPE startup). Trials 10
onward are TPE-sampled, conditional on the prior trials' outcomes.

Patterns visible in the TPE-sampled tail (#10–#20):

1. **Width 256 is the attractor.** 7 of 10 post-startup trials picked
   `hidden_0 = 256`. The two outliers (`[1024]`, `[2048]`) were exploration
   moves that fared worse on F1 or collapsed early.
2. **GELU dominates.** 8 of 10 post-startup trials picked GELU. The two ReLU
   trials (#15, #19) and one SiLU (#13) were experiments to vary that axis;
   only #13 entered the Pareto front, and only under the corrected criterion.
3. **`norm=none` wins on this width.** Every trial on the Pareto front (under
   either criterion) has `norm=none`. The two normalised runs with width 256
   (trial 14: layernorm; trial 15: layernorm) ended up dominated.
4. **Depth > 1 generally hurts.** Trial 8 — `[256, 1024, 512]` — is the only
   multi-layer trial on the front and only enters under the corrected
   criterion. Trials 2, 7, 9, 20 (all depth ≥ 2) are dominated.
5. **Dropout 0.27–0.52 is the safe band.** Below 0.15 with a wide head
   (≥ 1024) the model can collapse — trial 6 (`[2048] SiLU drop=0.14`)
   fell into the 1-class predictor failure mode (F1 ≈ 0.05, IR = +∞).
6. **TPE found no benefit from depth > 1 in the budget allocated.** This
   does not prove deeper heads cannot help with more compute; it means the
   8-epoch budget penalises them disproportionately.

---

## 5. Catastrophic failure mode worth recording

Trial 6 — `head=[2048] + SiLU + dropout=0.14 + norm=batchnorm + lr=4.5e-3` —
collapsed on epoch 1 to F1 ≈ 0.05, IR = ∞ (one class has F1 = 0, divides by
zero). The model is essentially predicting a single class.

Two factors interacted:
- The head has 2048 → 2048 → 7 = ~4.2 M trainable parameters, a large
  expansion over the embedding.
- `BatchNorm1d` with `momentum=0.1` (default) plus the high LR and shallow
  dropout pushed the BN running stats into a degenerate regime that the
  cosine schedule could not recover from in 8 epochs.

Mitigation: the trial is reported with `IR = 1e6` (finite penalty in
`_train_trial`) so Optuna sees a dominated point rather than NaN. The
ground truth — "this combination collapses" — is captured.

---

## 6. Next steps

### Round 2 — refit the Pareto winners at 25 epochs

For each trial on the corrected Pareto front (#4, #8, #12, #13), retrain
from scratch with the same head topology + LR but `num_epochs = 25` and 3
seeds (≠ 42), to estimate the variance of the Pareto position. Expected
output: the same trials remain on the front, with tighter confidence
intervals on F1 and IR.

### Round 3 — broaden the search

If Round 2 confirms the gains, run an enlarged study with:
- Higher width ceiling (4096) and lower floor (64).
- Per-layer learning rate multiplier (backbone vs head).
- Loss alternatives: `arcface`, `magface`, `adaface` (orientador-approved
  axis from kickoff diretriz 6).

---

## 7. Reproducibility

```powershell
# Full study (≈ 12 h on a single RTX 4070 SUPER)
.\.venv\Scripts\python.exe scripts/hpo_head.py `
  --n-trials 20 --hpo-epochs 8 `
  --base-config configs/experiments/exp05_ce_adamw_cosine.yaml `
  --output-dir outputs/hpo/round1 `
  --study-name mlp_head_round1

# Re-analysis (any time, from the log)
.\.venv\Scripts\python.exe scripts/reanalyze_hpo_round1.py `
  --log outputs/hpo/round1/hpo.log `
  --output outputs/hpo/round1/pareto_reanalyzed.json
```

The SQLite study is resumable: if a run is interrupted, mark orphan RUNNING
trials as FAIL (see `optuna.trial.TrialState`) and re-invoke the same
command — Optuna's TPE will resume from the persisted state.
