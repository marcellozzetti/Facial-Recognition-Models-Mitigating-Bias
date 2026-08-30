"""Pipeline stage — Etapa 4: comparação triangulada contra baselines.

Cap. 4 §Baselines + §Triangulação (§4.8).

Consome os checkpoints treinados pela Etapa 3 (configs A/B/C) e demais
baselines externos (ResNet-34, FSCL+, Group DRO, FineFACE, Adversarial
debiasing) e produz:

- outputs/etapa4/per_class_metrics.csv       (F1 por raça, por modelo)
- outputs/etapa4/triangulation.csv           (DR, worst-class F1, EO gap)
- outputs/etapa4/pareto.csv                  (macro F1 x inequity)
- outputs/etapa4/pareto.png                  (visualização)
- outputs/etapa4/report.md                   (síntese textual)

Uso:
    python pipelines/06_evaluate_baselines.py \\
        --predictions-dir outputs/etapa3/predictions/ \\
        --output-dir outputs/etapa4/

Cada arquivo em ``predictions-dir`` deve ser um parquet com colunas:
    y_true (int)   — rótulo verdadeiro (0..6)
    y_pred (int)   — rótulo predito
    group (str)    — atributo sensível para EO/EqOdds (ex.: gênero)
    model (str)    — nome do modelo (ex.: "A_baseline", "B_film_mst", "fscl_plus")

Também aceita seed no nome do arquivo: agrega por (model) e reporta
média + desvio-padrão entre sementes (rigor experimental §4.10).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.fairness import (  # noqa: E402
    ModelResult,
    build_pareto_table,
    disparity_ratio,
    equal_opportunity_per_class,
    equalized_odds_per_class,
    inequity_rate,
    per_class_f1,
    plot_pareto,
    worst_class_f1,
    worst_class_f1_bootstrap,
)

logger = logging.getLogger("pipelines.06_evaluate_baselines")

FAIRFACE_RACES = [
    "White", "Black", "Indian", "East Asian",
    "Southeast Asian", "Middle Eastern", "Latino_Hispanic",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 4 — triangulação vs baselines.")
    p.add_argument("--predictions-dir", type=Path, required=True,
                   help="Diretório com parquets de predições (1 por modelo/seed).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-classes", type=int, default=7)
    p.add_argument("--class-names", nargs="+", default=FAIRFACE_RACES)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_predictions(pred_dir: Path) -> list[pd.DataFrame]:
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet em {pred_dir}.")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        required = {"y_true", "y_pred", "model"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f}: colunas ausentes {sorted(missing)}")
        dfs.append(df)
    return dfs


def _per_model_metrics(
    df: pd.DataFrame,
    n_classes: int,
    class_names: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    scores = per_class_f1(y_true, y_pred, n_classes)
    macro_f1 = float(np.nanmean(scores))
    dr = disparity_ratio(y_true, y_pred, n_classes)
    wf1, wf1_lo, wf1_hi = worst_class_f1_bootstrap(
        y_true, y_pred, n_classes, n_boot=n_boot, seed=seed,
    )
    row: dict = {
        "macro_f1": macro_f1,
        "disparity_ratio": dr,
        "inequity_rate": inequity_rate(y_true, y_pred, n_classes),
        "worst_class_f1": wf1,
        "worst_class_f1_ci_lo": wf1_lo,
        "worst_class_f1_ci_hi": wf1_hi,
    }
    for i, name in enumerate(class_names):
        row[f"f1_{name}"] = float(scores[i]) if not np.isnan(scores[i]) else None
    if "group" in df.columns:
        eo_df = equal_opportunity_per_class(
            y_true, y_pred, df["group"].to_numpy(), n_classes, class_names=class_names,
        )
        eq_df = equalized_odds_per_class(
            y_true, y_pred, df["group"].to_numpy(), n_classes, class_names=class_names,
        )
        row["eo_gap_mean"] = float(eo_df["eo_gap"].dropna().mean()) if not eo_df["eo_gap"].dropna().empty else None
        row["eqodds_gap_mean"] = float(eq_df["eqodds_gap"].dropna().mean()) if not eq_df["eqodds_gap"].dropna().empty else None
    return row


def _aggregate_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    """Média + desvio-padrão por modelo (agrupa por 'model')."""
    numeric = per_run.select_dtypes(include="number").columns.tolist()
    grouped = per_run.groupby("model")[numeric].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    return grouped


def _generate_report(agg: pd.DataFrame, pareto_tbl: pd.DataFrame) -> str:
    lines = [
        "# Etapa 4 — Triangulação vs baselines",
        "",
        "## Métricas agregadas por modelo (média entre sementes)",
        "",
        "| Modelo | F1 macro | DR | F1 pior classe | EO gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        eo = row.get("eo_gap_mean_mean")
        eo_str = f"{eo:.3f}" if eo is not None and not np.isnan(eo) else "—"
        lines.append(
            f"| {row['model']} | {row['macro_f1_mean']:.3f} | "
            f"{row['disparity_ratio_mean']:.3f} | "
            f"{row['worst_class_f1_mean']:.3f} | {eo_str} |"
        )
    lines += [
        "",
        "## Fronteira Pareto (F1 macro × inequidade)",
        "",
        "| Modelo | F1 macro | Inequidade | Pareto |",
        "|---|---:|---:|:---:|",
    ]
    for _, row in pareto_tbl.iterrows():
        mark = "★" if row["pareto"] else ""
        lines.append(
            f"| {row['model']} | {row['macro_f1']:.3f} | "
            f"{row['inequity']:.3f} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dfs = _load_predictions(args.predictions_dir)
    logger.info("Encontrados %d arquivos de predições.", len(dfs))

    per_run: list[dict] = []
    for df in dfs:
        model = str(df["model"].iloc[0])
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
        metrics = _per_model_metrics(
            df,
            n_classes=args.n_classes,
            class_names=args.class_names,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        metrics["model"] = model
        metrics["seed"] = seed
        per_run.append(metrics)

    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(args.output_dir / "per_run_metrics.csv", index=False)

    agg = _aggregate_seeds(per_run_df)
    agg.to_csv(args.output_dir / "aggregated_metrics.csv", index=False)

    # Pareto: usa média entre sementes
    pareto_input = [
        ModelResult(
            name=row["model"],
            macro_f1=float(row["macro_f1_mean"]),
            inequity=float(row["inequity_rate_mean"]),
        )
        for _, row in agg.iterrows()
    ]
    pareto_tbl = build_pareto_table(pareto_input)
    pareto_tbl.to_csv(args.output_dir / "pareto.csv", index=False)
    plot_pareto(pareto_input, save_path=args.output_dir / "pareto.png")

    report = _generate_report(agg, pareto_tbl)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("Saídas geradas em %s.", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
