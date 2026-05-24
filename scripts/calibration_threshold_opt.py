"""Calibracao via temperature scaling + threshold optimization per-class.

Carrega o ensemble de checkpoints, computa logits no VAL e TEST,
ajusta temperatura T sobre o val (minimiza NLL), depois otimiza um
vetor de thresholds por classe sobre o val (maximiza F1 macro), e
aplica no test. Reporta metricas antes/depois.

Temperature scaling (Guo et al. ICML 2017): T = scalar > 0 que divide
todos os logits. Reduz overconfidence sem mudar argmax (acuracia ==).
Mas muda a distribuicao de probabilidades, o que importa para
threshold-based decision rules e calibracao.

Threshold optimization per-class: cada classe tem um threshold b_c;
prediz classe c se logit_c + b_c > max_{c' != c} (logit_{c'} + b_{c'}).
Equivalente a adicionar um bias treinavel por classe pos-hoc.
Maximiza F1 macro no val via grid search ou gradient (usamos grid simple).

Uso:
  python scripts/calibration_threshold_opt.py \\
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \\
    --config configs/.../exp_*.yaml \\
    --output-dir outputs/calibration \\
    --run-label ensemble3_calibrated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_bias.config.loader import load_config  # noqa: E402
from face_bias.data.dataset import setup_dataset  # noqa: E402
from face_bias.evaluation.metrics import disparity_ratio  # noqa: E402
from face_bias.models.resnet import LResNet50E_IR  # noqa: E402


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = LResNet50E_IR(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.2),
        head=cfg["model"]["head"],
        pretrained=False,
        backbone_arch=cfg["model"].get("backbone_arch", "resnet50"),
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_logits(model, loader, device):
    all_logits, all_targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images, labels=labels.to(device)) if hasattr(model, "head_type") else model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def temperature_scale_nll(logits_val: np.ndarray, y_val: np.ndarray, num_classes: int) -> float:
    """Find optimal temperature T via grid search minimizing NLL on val."""
    logits_t = torch.from_numpy(logits_val).float()
    y_t = torch.from_numpy(y_val).long()
    best_T, best_nll = 1.0, float("inf")
    for T in np.arange(0.5, 5.01, 0.05):
        logp = F.log_softmax(logits_t / T, dim=1)
        nll = F.nll_loss(logp, y_t).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


def optimize_class_biases(logits_val: np.ndarray, y_val: np.ndarray, num_classes: int,
                          n_iters: int = 200, lr: float = 0.05) -> np.ndarray:
    """Otimiza vetor de biases por classe maximizando F1 macro no val.

    Implementacao: gradiente coordinate-descent simples sobre b_c.
    Em cada iteracao tenta b_c +- lr; aceita se F1 macro sobe.
    """
    biases = np.zeros(num_classes, dtype=np.float32)

    def f1_with_biases(b):
        y_pred = (logits_val + b).argmax(axis=1)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    best_f1 = f1_with_biases(biases)
    print(f"  Inicial F1 macro (no bias): {best_f1:.4f}")

    for it in range(n_iters):
        improved = False
        for c in range(num_classes):
            for delta in (lr, -lr):
                b_new = biases.copy()
                b_new[c] += delta
                f1 = f1_with_biases(b_new)
                if f1 > best_f1:
                    biases = b_new
                    best_f1 = f1
                    improved = True
                    break
        if not improved:
            print(f"  Convergiu em iter {it+1}, F1 macro final: {best_f1:.4f}")
            break

    return biases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/calibration")
    parser.add_argument("--run-label", default="calibrated")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    print("Construindo dataloaders")
    dataloaders, label_encoder, num_classes = setup_dataset(cfg)
    class_names = list(label_encoder.classes_)
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    print(f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)} | {num_classes} classes")

    # Compute ensemble logits on VAL and TEST
    val_logits_acc = None
    test_logits_acc = None
    y_val_ref = y_test_ref = None

    for ckpt_path in args.checkpoints:
        print(f"\nCarregando {ckpt_path}")
        model = load_model(ckpt_path, cfg, device)

        logits_v, y_v = predict_logits(model, val_loader, device)
        logits_t, y_t = predict_logits(model, test_loader, device)

        if val_logits_acc is None:
            val_logits_acc = logits_v.copy()
            test_logits_acc = logits_t.copy()
            y_val_ref = y_v
            y_test_ref = y_t
        else:
            val_logits_acc += logits_v
            test_logits_acc += logits_t

        del model
        torch.cuda.empty_cache()

    n = len(args.checkpoints)
    val_logits_acc /= n
    test_logits_acc /= n

    # Baseline ensemble metrics (test)
    y_pred_baseline = test_logits_acc.argmax(axis=1)
    acc_base = accuracy_score(y_test_ref, y_pred_baseline)
    f1_base = f1_score(y_test_ref, y_pred_baseline, average="macro", zero_division=0)
    per_class_base = f1_score(y_test_ref, y_pred_baseline, labels=range(num_classes),
                              average=None, zero_division=0)
    ir_base = disparity_ratio(per_class_base)
    print(f"\n=== BASELINE (ensemble sem calibracao/threshold) ===")
    print(f"  acc={acc_base:.4f}, f1={f1_base:.4f}, IR={ir_base:.3f}")

    # Step 1: Temperature scaling (sobre val)
    print(f"\n=== Etapa 1: Temperature scaling (otimizando NLL no val) ===")
    T = temperature_scale_nll(val_logits_acc, y_val_ref, num_classes)
    print(f"  T optimo = {T:.3f}")
    # Apply to test (NAO muda argmax → mesmo F1 mas calibrado)
    test_logits_T = test_logits_acc / T
    y_pred_T = test_logits_T.argmax(axis=1)
    acc_T = accuracy_score(y_test_ref, y_pred_T)
    print(f"  acc apos T={T:.3f}: {acc_T:.4f} (== baseline porque T scalar nao muda argmax)")

    # Step 2: Per-class threshold optimization (sobre val)
    print(f"\n=== Etapa 2: Threshold per-class (otimizando F1 macro no val) ===")
    biases = optimize_class_biases(val_logits_acc, y_val_ref, num_classes)
    print(f"  Biases finais por classe: {dict(zip(class_names, [round(float(b), 3) for b in biases]))}")

    # Apply biases to TEST logits
    test_logits_final = test_logits_acc + biases
    y_pred_final = test_logits_final.argmax(axis=1)
    acc_final = accuracy_score(y_test_ref, y_pred_final)
    f1_final = f1_score(y_test_ref, y_pred_final, average="macro", zero_division=0)
    per_class_final = f1_score(y_test_ref, y_pred_final, labels=range(num_classes),
                               average=None, zero_division=0)
    ir_final = disparity_ratio(per_class_final)

    print(f"\n=== FINAL (ensemble + threshold per-class) ===")
    print(f"  acc       = {acc_final:.4f}  ({acc_final - acc_base:+.4f} vs baseline)")
    print(f"  f1_macro  = {f1_final:.4f}  ({f1_final - f1_base:+.4f} vs baseline)")
    print(f"  IR        = {ir_final:.3f}  ({ir_final - ir_base:+.3f} vs baseline)")
    print(f"  per-class F1: {dict(zip(class_names, [round(float(x), 4) for x in per_class_final]))}")

    summary = {
        "run_label": args.run_label,
        "n_checkpoints": n,
        "baseline_ensemble": {
            "accuracy": float(acc_base),
            "f1_macro": float(f1_base),
            "disparity_ratio": float(ir_base),
            "per_class_f1": {c: float(x) for c, x in zip(class_names, per_class_base)},
        },
        "temperature_T": float(T),
        "class_biases": {c: float(b) for c, b in zip(class_names, biases)},
        "final_calibrated_thresholded": {
            "accuracy": float(acc_final),
            "f1_macro": float(f1_final),
            "disparity_ratio": float(ir_final),
            "per_class_f1": {c: float(x) for c, x in zip(class_names, per_class_final)},
        },
        "delta_vs_baseline": {
            "accuracy": float(acc_final - acc_base),
            "f1_macro": float(f1_final - f1_base),
            "disparity_ratio": float(ir_final - ir_base),
        },
    }
    summary_path = out_dir / f"calibration_{args.run_label}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
