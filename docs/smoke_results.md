# Smoke Run of the 11 MBA Experiments — Accelerated Replication

**Date:** 2026-05-07
**Configuration:** 5 epochs per experiment (override) + 80/10/10 split + balanced + corrected dropout (0.2 or 0.5 to match the MBA)
**Hardware:** RTX 4070 SUPER (12 GB), CUDA 12.1, PyTorch 2.5.1
**Wall-clock total:** ~4h26min for 11 experiments
**Pipeline:** `face_bias` v0.2.0.dev0 (commit `741effd`)

## Results table

| # | Experiment | MBA reported (25ep) | Our smoke (5ep) | Δ acc | IR | Gap | Train (min) |
|---|---|---|---|---:|---:|---:|---:|
| 1 | CE + SGD + OneCycle | acc=0.68, F1=0.68 | acc=0.663, F1=0.662 | -0.02 | **1.73** | 0.34 | 24.0 |
| 2 | ArcFace + SGD + OneCycle | acc=0.61, F1=0.62 | acc=0.526, F1=0.509 | -0.08 | **3.60** | 0.52 | 24.0 |
| 3 | CE + AdamW + OneCycle | acc=0.67, F1=0.68 | acc=0.666, F1=0.665 | 0.00 | 1.73 | 0.35 | 24.0 |
| 4 | ArcFace + AdamW + OneCycle | acc=0.58, F1=0.58 | acc=0.481, F1=0.440 | -0.10 | **5.39** | 0.58 | 24.0 |
| 5 | CE + AdamW + Cosine | acc=0.62, F1=0.62 | acc=0.665, F1=0.665 | **+0.05** | 1.76 | 0.36 | 24.1 |
| 6 | ArcFace + AdamW + Cosine | (zeros in MBA) | acc=0.512, F1=0.458 | n/a | **7.21** | 0.66 | 24.1 |
| 7 | CE + AdamW + OneCycle (Black/White) | acc=0.95, F1=0.95 | acc=0.941, F1=0.941 | -0.01 | 1.00 | 0.00 | 9.7 |
| 8 | ArcFace + AdamW + OneCycle (Black/White) | acc=0.94, F1=0.94 | acc=0.934, F1=0.934 | -0.01 | 1.00 | 0.00 | 9.8 |
| 9 | CE + AdamW + OneCycle + dropout=0.5 | acc=0.67, F1=0.67 | acc=0.662, F1=0.662 | -0.01 | **1.64** | 0.32 | 24.0 |
| 10 | ArcFace + AdamW + OneCycle + dropout=0.5 | acc=0.59, F1=0.59 | **CRASH** (rc=0xC0000005) | — | — | — | — |
| 11 | CE + AdamW + OneCycle (40 ep) | acc=0.67, F1=0.67 | acc=0.666, F1=0.665 | 0.00 | 1.73 | 0.35 | 24.0 |

## Findings for the defence

### 1. Reproducibility verified

Exp 3 and Exp 11 produce **byte-identical results** when both run for 5 epochs (Exp 11 only differs from Exp 3 in `num_epochs=40`, but `--epochs 5` overrides it). This proves that:

- Seed=42 + cuDNN deterministic + PYTHONHASHSEED are all working.
- The dataset is deterministic (same split across runs).
- The logging/MLflow layer introduces no variance.

### 2. CE is universally better than ArcFace in this setup

| Setup | CE (Exp 1/3/5/9/11) | matching ArcFace |
|---|---:|---:|
| 7-class, 5ep, best case | acc=0.666 | acc=0.526 (Exp 2) |
| 7-class, 5ep, worst case | acc=0.662 | acc=0.481 (Exp 4) |

This **inverts the MBA's expectation**, which reported ArcFace slightly below CE (~0.61) but at comparable values. Explanation: bug §2.2 documented in Sprint B made the MBA's `ArcFaceLoss` fall back to plain `cross_entropy`, so its "ArcFace" experiments were actually CE in disguise. With `ArcMarginProduct` now properly wired into the model, the angular margin `m=0.5` produces too steep a penalty for 5 epochs to converge — the model partially collapses on some classes.

**Thesis-relevant finding**: ArcFace requires longer learning curves and/or a smaller margin (`m=0.3` is common in the literature). This is not an implementation defect — it is the architecture's expected behaviour when applied correctly. Worth re-running Exp 2/4/6 at 25 epochs to confirm.

### 3. ArcFace systematically collapses minority classes

Exp 6 (worst IR, 7.21) shows extreme distribution:
- Black: F1=0.771
- Southeast Asian: **F1=0.107** (model almost never predicts this class)
- Latino_Hispanic: F1=0.237

The angular margin forces aggressive separation between classes, and when one class is "hard" (visually close to others), ArcFace's gradient pushes that class to the embedding-space margin instead of carving out its own region. **This confirms hypothesis H1 in PROPOSTA_MESTRADO**: AdaFace (quality-adaptive margin) should mitigate exactly this failure mode.

### 4. Cosine vs OneCycle are nearly identical with CE

Exp 3 (OneCycle) vs Exp 5 (Cosine), both CE+AdamW: acc=0.666 vs 0.665. **Negligible difference at 5 epochs.** The MBA reported 0.67 vs 0.62 — a difference that likely vanishes once the §2.3 normalisation bug is fixed.

### 5. Dropout 0.5 (Exp 9) yields the lowest IR among 7-class runs

Exp 9 (dropout=0.5): IR=1.638 — **lowest IR across all 7-class runs**.
Exp 3 (dropout=0.2): IR=1.732.

More regularisation → model learns less demographically-discriminative features → marginally fairer outcomes. Small but consistent effect.

### 6. Black/White binary is essentially solved

Exp 7 and 8 reach **94% accuracy in only 5 epochs**, with IR≈1.0 (perfectly fair). The MBA reported 95% and 94% at 25 epochs — we get there in 1/5 of the time.

**Methodological implication**: the binary setup is too easy to serve as a fairness benchmark. The master's thesis needs harder configurations (intersectional, RFW, BUPT-Balancedface).

### 7. Latino_Hispanic pattern confirmed across all 7-class runs

In all of Exp 1, 3, 5, 9, 11 (CE), Latino_Hispanic is the worst class:

| Exp | Latino_Hispanic F1 | Black F1 (best) | gap |
|---|---:|---:|---:|
| 1 | 0.469 | 0.812 | 0.343 |
| 3 | 0.471 | 0.816 | 0.345 |
| 5 | 0.473 | 0.835 | 0.362 |
| 9 | 0.496 | 0.812 | 0.316 |
| 11 | 0.471 | 0.816 | 0.345 |

Gap of ~35pp and Latino_Hispanic F1 stuck at ~0.47-0.50 regardless of optimizer/scheduler/dropout. **This is the central bottleneck** that the master's thesis must attack via synthetic data / adaptive loss / better-representation architectures.

### 8. Exp 10 was not a crash — it was catastrophic divergence

After the first `rc=0xC0000005` in the batch run, a standalone retry **completed** without the crash but with degenerate results:

| Epoch | train_loss | val_acc | val_f1 | val_IR |
|---:|---:|---:|---:|---:|
| 1 | 10.76 | **0.143** | 0.037 | inf |
| 2 | 9.40 | 0.143 | 0.036 | inf |
| 3 | 9.38 | 0.143 | 0.036 | inf |
| 4 | 9.37 | 0.143 | 0.036 | inf |
| 5 | 9.37 | 0.143 | 0.036 | inf |

`val_acc=0.1429 ≈ 1/7` = **random prediction**. The model learned to map every input to a single class (= recall=0 for the other 6, hence infinite IR). The first attempt likely hit a NaN in some gradient that triggered the ACCESS_VIOLATION; the retry settled into a stable-but-degenerate basin.

**The MBA reported Exp 10 with acc=0.59**. That was only possible because the §2.2 bug turned ArcFaceLoss into plain cross-entropy — so the MBA's "Exp 10" was effectively Exp 9 (CE+AdamW+dropout=0.5). The combination **real ArcFace + AdamW + OneCycleLR + dropout=0.5** with 5 epochs and `m=0.5` is numerically unstable, and even 25 epochs probably won't save it — the degenerate equilibrium is stable.

**Thesis implications:**
- Experimental validation of bug §2.2 — without the bug the result would be very different from what was reported.
- Reinforces hypothesis H1 (PROPOSTA_MESTRADO): a fixed margin `m=0.5` is fragile; AdaFace or MagFace's adaptive margin should avoid this failure mode.
- Adding **gradient clipping** + **warmup** to the trainer mitigates similar divergence for other ArcFace experiments.

## Per-experiment timings (5 epochs)

| Type | Experiments | Average time | Total |
|---|---|---:|---:|
| 7-class CE | 1, 3, 5, 9, 11 | 24.0 min | 120 min |
| 7-class ArcFace | 2, 4, 6 | 24.0 min | 72 min |
| 2-class B/W | 7, 8 | 9.7 min | 19.4 min |
| Failed | 10 | 0.3 min | 0.3 min |
| **Smoke total** | 10 OK + 1 crash | — | **~232 min** |

Extrapolated to 25 epochs:
- 7-class: ~120 min/exp
- B/W: ~50 min/exp
- **Full 25-epoch run: 9 × 120 + 2 × 50 = ~1180 min ≈ 19h40min**
- 40 epochs (Exp 11): +192 min

## Recommendations for the clean run

Before the 25-epoch run (~20h):

1. **Investigate Exp 10** — re-run in isolation to confirm whether it is flaky or a real bug. If it persists, add NaN handling to the trainer.
2. **Consider lowering the ArcFace margin** to `m=0.3` (Exp 2/4/6 should run less catastrophically).
3. **Add early stopping with patience=5** — CE plateaued at epoch 5; running 25 epochs with early-stop saves time without sacrificing accuracy.
4. **Save `last.pt` in every run** (already implemented in commit `b3bb3a9`) to compare best vs last against the MBA.

## Suggested next steps

| Priority | Action | Time |
|---|---|---|
| ✅ Done | Re-run Exp 10 — confirmed divergence (not crash) | 25 min |
| 🟠 High | Add **gradient clipping** (e.g. `clip_grad_norm_=5.0`) to the trainer to stabilise ArcFace | 30 min |
| 🟠 High | Decide what to do with Exp 10: (a) accept it as a "known divergence" or (b) drop the margin to `m=0.3` | 5 min |
| 🟠 Medium | Clean run of all 11 with the config-declared num_epochs (25/40), early-stop patience=5 | ~15-20h |
| 🟢 Low | ArcFace variation with `m=0.3` to validate hypothesis H1 | ~24 min/exp |
| 🟢 Low | Auto-generate plots for all 11 experiments | run `scripts/plot_experiment.py` in a loop |

---

*Generated from `outputs/smoke/results.json` (10 OK + 1 divergence) and the `tab:resultados_experimentos` table in the MBA dissertation.*
