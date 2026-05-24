"""Analise interseccional de fairness (raça × gênero × idade).

Carrega o melhor checkpoint ConvNeXt-T sob protocolo Hassanpour (🅔),
roda inferência sobre o test set (val oficial FairFace, 10,954 imagens),
e cruza as predições com colunas gender + age do CSV original. Produz:

  - Tabela acurácia por (raça × gênero)
  - Tabela acurácia por (raça × grupo etário)
  - Identificação do subgrupo PIOR (worst-off intersectional group)
  - Tabela razão de disparidade interseccional

Output:
  outputs/intersectional/intersectional_metrics_{run_id}.json
  outputs/intersectional/intersectional_report.md (humano)

Uso:
  python scripts/intersectional_fairness_analysis.py \\
    --checkpoint outputs/definitive/anchor_hassanpour/exp_anc_hass_convnext_s42/train/.../checkpoints/best.pt \\
    --config configs/anchor_hassanpour/exp_anc_hass_convnext_s42.yaml \\
    --output-dir outputs/intersectional
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_bias.config.loader import load_config  # noqa: E402
from face_bias.data.dataset import setup_dataset  # noqa: E402
from face_bias.models.resnet import LResNet50E_IR  # noqa: E402


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Reconstrói o modelo conforme config e carrega pesos do best.pt."""
    model = LResNet50E_IR(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.2),
        head=cfg["model"]["head"],
        pretrained=False,  # carrega via state_dict
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
def predict_with_filenames(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    file_list: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference and return (y_true, y_pred, filenames) aligned."""
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images, labels=labels.to(device)) if hasattr(model, "head_type") else model(images)
        proba = torch.softmax(logits.float(), dim=1)
        all_preds.append(proba.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    # file_list deve estar na ordem do DataLoader (que é shuffle=False para test)
    assert len(file_list) == len(y_pred), \
        f"file_list ({len(file_list)}) != predictions ({len(y_pred)})"
    return y_true, y_pred, file_list


def compute_intersectional_metrics(
    df_results: pd.DataFrame, group_cols: list[str], class_names: list[str]
) -> pd.DataFrame:
    """Compute per-group accuracy + F1 macro.

    df_results must have columns: y_true, y_pred, race_str, plus group_cols.
    Returns DataFrame with one row per intersection.
    """
    rows = []
    for keys, sub in df_results.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        n = len(sub)
        if n < 10:  # too few samples — skip
            continue
        acc = (sub["y_true"] == sub["y_pred"]).mean()
        f1_macro = f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
        row = dict(zip(group_cols, keys))
        row.update({"n": n, "accuracy": acc, "f1_macro": f1_macro})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--config", required=True, help="Path to YAML config used for the run")
    parser.add_argument("--output-dir", default="outputs/intersectional")
    parser.add_argument("--run-label", default="convnext_s42_hassanpour",
                        help="Label for output files")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    print(f"Carregando modelo de {args.checkpoint}")
    model = load_model(args.checkpoint, cfg, device)

    print("Reconstruindo dataloaders (split_protocol oficial)")
    dataloaders, label_encoder, num_classes = setup_dataset(cfg)
    class_names = list(label_encoder.classes_)
    test_loader = dataloaders["test"]

    # Recompute test file list from CSV (mesma ordem que setup_dataset usa para test)
    csv_pd = pd.read_csv(cfg["data"]["dataset_file"])
    test_df = csv_pd.loc[csv_pd["file"].str.startswith("val/")].reset_index(drop=True)
    test_files = test_df["file"].tolist()

    print(f"Test set: {len(test_files)} imagens (val/ oficial)")
    print(f"Classes: {class_names}")

    print("Rodando inferência...")
    y_true, y_pred, file_list = predict_with_filenames(
        model, test_loader, device, test_files
    )

    # Construct results dataframe
    df = pd.DataFrame({"file": file_list, "y_true": y_true, "y_pred": y_pred})
    df["race_str"] = df["y_true"].apply(lambda i: class_names[i])

    # Merge with gender + age from CSV
    df = df.merge(test_df[["file", "gender", "age"]], on="file", how="left")
    assert df["gender"].notna().all(), "gender NaN — merge falhou"
    assert df["age"].notna().all(), "age NaN — merge falhou"

    # Overall metrics
    acc_overall = (df["y_true"] == df["y_pred"]).mean()
    f1_overall = f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0)
    print(f"\nOVERALL: acc={acc_overall:.4f}, f1_macro={f1_overall:.4f} (n={len(df)})")

    # By race × gender
    print("\n=== Per (raça × gênero) ===")
    df_rg = compute_intersectional_metrics(df, ["race_str", "gender"], class_names)
    print(df_rg.to_string(index=False))

    # By race × age
    print("\n=== Per (raça × faixa etária) ===")
    df_ra = compute_intersectional_metrics(df, ["race_str", "age"], class_names)
    print(df_ra.to_string(index=False))

    # By race × gender × age (3-way)
    print("\n=== Per (raça × gênero × faixa etária) — só grupos com n>=20 ===")
    df_rga = compute_intersectional_metrics(df, ["race_str", "gender", "age"], class_names)
    df_rga_filtered = df_rga[df_rga["n"] >= 20].copy()

    # Worst-off groups
    worst_rg = df_rg.loc[df_rg["f1_macro"].idxmin()]
    worst_ra = df_ra.loc[df_ra["f1_macro"].idxmin()]
    worst_rga = df_rga_filtered.loc[df_rga_filtered["f1_macro"].idxmin()] if len(df_rga_filtered) > 0 else None

    # Razão de disparidade interseccional (max/min over groups)
    ir_rg = df_rg["f1_macro"].max() / df_rg["f1_macro"].min() if df_rg["f1_macro"].min() > 0 else float('inf')
    ir_ra = df_ra["f1_macro"].max() / df_ra["f1_macro"].min() if df_ra["f1_macro"].min() > 0 else float('inf')
    ir_rga = df_rga_filtered["f1_macro"].max() / df_rga_filtered["f1_macro"].min() if len(df_rga_filtered) > 0 and df_rga_filtered["f1_macro"].min() > 0 else None

    print(f"\n=== Razões de disparidade interseccional ===")
    print(f"  IR (raça)              : {df_rg.groupby('race_str')['f1_macro'].first().max() / df_rg.groupby('race_str')['f1_macro'].first().min():.3f} (referência: 1-D)")
    print(f"  IR (raça × gênero)     : {ir_rg:.3f}")
    print(f"  IR (raça × idade)      : {ir_ra:.3f}")
    if ir_rga:
        print(f"  IR (raça × gênero × idade): {ir_rga:.3f}")

    print(f"\n=== Pior subgrupo ===")
    print(f"  raça × gênero  : {worst_rg['race_str']} × {worst_rg['gender']} (n={int(worst_rg['n'])}, f1={worst_rg['f1_macro']:.4f}, acc={worst_rg['accuracy']:.4f})")
    print(f"  raça × idade   : {worst_ra['race_str']} × {worst_ra['age']} (n={int(worst_ra['n'])}, f1={worst_ra['f1_macro']:.4f}, acc={worst_ra['accuracy']:.4f})")
    if worst_rga is not None:
        print(f"  raça × g × age : {worst_rga['race_str']} × {worst_rga['gender']} × {worst_rga['age']} (n={int(worst_rga['n'])}, f1={worst_rga['f1_macro']:.4f}, acc={worst_rga['accuracy']:.4f})")

    # Save artifacts
    summary = {
        "run_label": args.run_label,
        "checkpoint": args.checkpoint,
        "n_test": int(len(df)),
        "overall_acc": float(acc_overall),
        "overall_f1_macro": float(f1_overall),
        "ir_race_gender": float(ir_rg),
        "ir_race_age": float(ir_ra),
        "ir_race_gender_age": float(ir_rga) if ir_rga else None,
        "worst_race_gender": {
            "group": f"{worst_rg['race_str']} × {worst_rg['gender']}",
            "n": int(worst_rg["n"]),
            "f1_macro": float(worst_rg["f1_macro"]),
            "accuracy": float(worst_rg["accuracy"]),
        },
        "worst_race_age": {
            "group": f"{worst_ra['race_str']} × {worst_ra['age']}",
            "n": int(worst_ra["n"]),
            "f1_macro": float(worst_ra["f1_macro"]),
            "accuracy": float(worst_ra["accuracy"]),
        },
    }
    if worst_rga is not None:
        summary["worst_race_gender_age"] = {
            "group": f"{worst_rga['race_str']} × {worst_rga['gender']} × {worst_rga['age']}",
            "n": int(worst_rga["n"]),
            "f1_macro": float(worst_rga["f1_macro"]),
            "accuracy": float(worst_rga["accuracy"]),
        }

    summary_path = out_dir / f"intersectional_metrics_{args.run_label}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df_rg.to_csv(out_dir / f"per_race_gender_{args.run_label}.csv", index=False)
    df_ra.to_csv(out_dir / f"per_race_age_{args.run_label}.csv", index=False)
    df_rga_filtered.to_csv(out_dir / f"per_race_gender_age_{args.run_label}.csv", index=False)

    print(f"\nArtifacts saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
