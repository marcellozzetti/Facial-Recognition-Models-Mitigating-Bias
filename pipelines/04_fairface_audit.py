"""Pipeline stage — Etapa 2: auditoria fenotípica FairFace × MST.

Consome o parquet gerado pelo pipeline 03 (``fairface_val_mst.parquet``)
+ CSV de labels do FairFace, produz a matriz Contribuição 2 e testa H3.

Uso:
    python pipelines/04_fairface_audit.py \\
        --mst-predictions outputs/etapa1/fairface_val_mst.parquet \\
        --fairface-labels data/FairFace/val_labels.csv \\
        --dataset-root data/FairFace \\
        --output-dir outputs/etapa2/

Produz:
    outputs/etapa2/audit_fairface_mst.parquet   (Etapa 1 + labels casados)
    outputs/etapa2/matriz_mst_x_raca.parquet    (matriz 7×10 normalizada)
    outputs/etapa2/matriz_mst_x_raca.png        (heatmap Contribuição 2)
    outputs/etapa2/resumo_por_raca.csv          (spread + entropia + CV)
    outputs/etapa2/relatorio.md                 (interpretação + H3)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.audit import (  # noqa: E402
    audit_from_files,
    build_matrix,
    summarize,
    assess_hypothesis_h3,
    visualize,
)

logger = logging.getLogger("pipelines.04_fairface_audit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 2 — auditoria FairFace × MST.")
    p.add_argument("--mst-predictions", type=Path, required=True)
    p.add_argument("--fairface-labels", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--spread-threshold",
        type=float,
        default=0.05,
        help="Massa mínima por tom MST para contar no spread (default 0.05).",
    )
    p.add_argument(
        "--h3-min-spread",
        type=int,
        default=5,
        help="Spread mínimo requerido para confirmar H3 (Cap 3).",
    )
    return p.parse_args(argv)


def generate_report(matrix: pd.DataFrame, resumo: pd.DataFrame, h3) -> str:
    lines = [
        "# Etapa 2 — Auditoria fenotípica FairFace × MST",
        "",
        "Matriz Contribuição 2 (proporção intra-raça por tom Monk).",
        "",
        "## Resumo por classe racial",
        "",
        "| Raça | Spread (≥5%) | Entropia (bits) | CV |",
        "|---|---:|---:|---:|",
    ]
    for race, row in resumo.iterrows():
        lines.append(
            f"| {race} | {int(row['spread'])} | {row['entropy_bits']:.3f} | {row['cv']:.3f} |"
        )
    lines += [
        "",
        f"## H3 (Cap. 3): spread Latinx ≥ {h3.min_spread_required} tons MST",
        "",
        f"- Threshold: {h3.threshold:.0%} de massa por tom",
        f"- Latinx spread observado: **{h3.latinx_spread}**",
        f"- Requerido: {h3.min_spread_required}",
        f"- **{'CONFIRMADA' if h3.confirmed else 'REFUTADA'}**",
        "",
        "## Matriz completa (proporção intra-raça)",
        "",
    ]
    # matriz como markdown table
    header = "| Raça | " + " | ".join(f"MST {c}" for c in matrix.columns) + " |"
    sep = "|---|" + "|".join(["---:"] * len(matrix.columns)) + "|"
    lines += [header, sep]
    for race, row in matrix.iterrows():
        cells = " | ".join(f"{v:.3f}" if v else "—" for v in row.values)
        lines.append(f"| {race} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_df = audit_from_files(
        args.mst_predictions,
        args.fairface_labels,
        dataset_root=args.dataset_root,
    )
    audit_out = args.output_dir / "audit_fairface_mst.parquet"
    audit_df.to_parquet(audit_out, index=False)
    logger.info("Audit casado: %s (%d linhas).", audit_out, len(audit_df))

    matrix = build_matrix(audit_df, race_col="race", mst_col="mst_pred", normalize=True)
    matrix.to_parquet(args.output_dir / "matriz_mst_x_raca.parquet")

    resumo = summarize(matrix, threshold=args.spread_threshold)
    resumo.to_csv(args.output_dir / "resumo_por_raca.csv")

    h3 = assess_hypothesis_h3(
        matrix,
        threshold=args.spread_threshold,
        min_spread_required=args.h3_min_spread,
    )
    (args.output_dir / "h3_result.json").write_text(
        json.dumps(h3.as_dict(), indent=2), encoding="utf-8"
    )

    visualize(matrix, save_path=args.output_dir / "matriz_mst_x_raca.png")

    report = generate_report(matrix, resumo, h3)
    (args.output_dir / "relatorio.md").write_text(report, encoding="utf-8")
    logger.info(
        "H3 (%s): Latinx spread=%d (requerido >=%d) → %s",
        "confirmada" if h3.confirmed else "refutada",
        h3.latinx_spread,
        h3.min_spread_required,
        "CONFIRMADA" if h3.confirmed else "REFUTADA",
    )
    logger.info("Saídas em %s.", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
