"""Recompute the loss-factor comparison under a corrected, matched
model-selection criterion — WITHOUT re-running anything.

The trainer selected best.pt by *min val_loss*. For margin heads the
eval logits are plain scaled cosine (no margin), so eval-CE is
anti-correlated with F1: best-by-val-loss picks an early bad epoch and
the bias is loss-family-dependent (corrupts ArcFace/AdaFace/MagFace,
not softmax+CE) — so the *matched comparison rule is violated*.

Every run still has the full per-epoch `history.json`. This re-selects
the epoch by **max val_f1_macro** uniformly for ALL runs (matched
again) and prints the corrected mean±std table beside the old
(min-val-loss) one to quantify the artifact. Validation-based model
selection — no GPU, no checkpoints, no re-run.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path


def runs(root: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for hp in glob.glob(f"outputs/{root}/*/train/*/history.json"):
        label = Path(hp).parents[2].name  # <label>/train/<runid>/history.json
        try:
            H = json.load(open(hp))
        except Exception:
            continue
        if H:
            out[label] = H
    return out


def pick(H, key, mode):
    return (min if mode == "min" else max)(H, key=lambda e: e[key])


def agg(hist: dict[str, dict], prefix: str, key: str, mode: str):
    labels = sorted(k for k in hist if k.startswith(prefix))
    accs, f1s, irs = [], [], []
    for k in labels:
        e = pick(hist[k], key, mode)
        accs.append(e["val_accuracy"])
        f1s.append(e["val_f1_macro"])
        # canonical key with legacy fallback (old history predates rename)
        irs.append(e.get("val_f1_disparity_ratio", e.get("val_f1_inequity_rate")))

    def ms(v):
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    return len(labels), ms(accs), ms(f1s), ms(irs)


import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument(
    "--definitive",
    action="store_true",
    help="read the definitive matched re-run (outputs/definitive/{factor3,baseline}) "
    "instead of the interim batches",
)
_args = _ap.parse_args()

if _args.definitive:
    GROUPS = [
        ("CE+linear (clean)", "definitive/baseline", "exp_r2base_exp05_ce_"),
        ("ArcFace (clean)", "definitive/baseline", "exp_r2base_exp06_af_"),
        ("AdaFace (Fator 3)", "definitive/factor3", "exp_f3_adaface_"),
        ("MagFace (Fator 3)", "definitive/factor3", "exp_f3_magface_"),
        ("SupCon (Fator 4)", "definitive/factor4", "exp_f4_supcon_"),
    ]
else:
    GROUPS = [
        ("CE+linear (clean)", "dataset_factor", "exp_r2base_exp05_ce_"),
        ("ArcFace (clean)", "dataset_factor", "exp_r2base_exp06_af_"),
        ("AdaFace (Fator 3)", "factor3", "exp_f3_adaface_"),
        ("MagFace (Fator 3)", "factor3", "exp_f3_magface_"),
    ]

H = {root: runs(root) for _, root, _ in GROUPS}

for crit_name, key, mode in [
    ("OLD  — min val_loss (contaminado)", "val_loss", "min"),
    ("NOVO — max val_f1_macro (casado)", "val_f1_macro", "max"),
]:
    print(f"\n=== {crit_name} ===")
    print(f"{'grupo':<22}{'n':>3}  {'acc':>14} {'f1_macro':>14} {'IR(f1)':>14}")
    for name, root, pfx in GROUPS:
        n, (am, ad), (fm, fd), (im, idv) = agg(H[root], pfx, key, mode)
        if n == 0:
            print(f"{name:<22}{n:>3}  (sem runs concluídos)")
            continue
        print(f"{name:<22}{n:>3}  {am:6.4f}±{ad:5.4f} {fm:6.4f}±{fd:5.4f} "
              f"{im:6.3f}±{idv:5.3f}")

print("\nRegra do 1-sigma: diferenca < 1 desvio combinado = nao-significativa.")
print("Comparacao so e casada se TODOS os grupos usam o mesmo criterio.")
