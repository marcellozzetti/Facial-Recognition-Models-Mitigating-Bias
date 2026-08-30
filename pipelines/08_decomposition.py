"""Pipeline stage — Etapa 6: síntese decompositiva final.

Cap. 4 §Etapa 6. Contribuição 6.

Consome:
    - Etapa 2: matriz MST × raça (parquet)
    - Etapa 4: predições por modelo (parquet consolidado, com pixel_information)

Produz:
    - outputs/etapa6/decomposition.json       (R²: fenotípico, algorítmico, residual)
    - outputs/etapa6/h4_result.json           (concentração de erros Latinx em overlap)
    - outputs/etapa6/h6_result.json           (R² pixel information)
    - outputs/etapa6/report.md                (síntese final)

Uso:
    python pipelines/08_decomposition.py \\
        --matrix outputs/etapa2/matriz_mst_x_raca.parquet \\
        --predictions outputs/etapa4/all_predictions.parquet \\
        --output-dir outputs/etapa6/
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

from face_bias.decomposition import (  # noqa: E402
    decompose_variance,
    generate_report,
    test_h4_overlap_concentration,
    test_h6_pixel_info_explains,
)

logger = logging.getLogger("pipelines.08_decomposition")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 6 — síntese decompositiva.")
    p.add_argument("--matrix", type=Path, required=True,
                   help="Matriz MST x raça da Etapa 2 (parquet 7×10).")
    p.add_argument("--predictions", type=Path, required=True,
                   help="Predições consolidadas da Etapa 4 (parquet).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--h4-threshold", type=float, default=0.5,
                   help="Cap 3 H4: >= X dos erros Latinx em overlap.")
    p.add_argument("--h4-overlap-min-races", type=int, default=2)
    p.add_argument("--h4-prop-threshold", type=float, default=0.05)
    p.add_argument("--h6-threshold", type=float, default=0.7,
                   help="Cap 3 H6: >= X da variância por pixel info.")
    return p.parse_args(argv)


def _prepare_errors_df(predictions: pd.DataFrame) -> pd.DataFrame:
    """Deriva coluna 'error' se ausente. Espera y_true, y_pred, mst_pred, model."""
    df = predictions.copy()
    if "error" not in df.columns:
        if not {"y_true", "y_pred"}.issubset(df.columns):
            raise ValueError(
                "predictions precisa ter 'error' ou ('y_true' e 'y_pred')."
            )
        df["error"] = (df["y_true"] != df["y_pred"]).astype(int)
    return df


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(args.matrix)
    predictions = pd.read_parquet(args.predictions)
    errors = _prepare_errors_df(predictions)
    logger.info("Errors DF: %d linhas.", len(errors))

    decomp = decompose_variance(errors)
    (args.output_dir / "decomposition.json").write_text(
        json.dumps(decomp.as_dict(), indent=2), encoding="utf-8"
    )
    logger.info(
        "R² total=%.3f  fenotipico=%.3f  algoritmico=%.3f  residual=%.3f",
        decomp.r2_total, decomp.r2_phenotypic,
        decomp.r2_algorithmic, decomp.r2_residual,
    )

    h4 = None
    if "race" in errors.columns:
        h4 = test_h4_overlap_concentration(
            errors, matrix,
            threshold=args.h4_threshold,
            overlap_min_races=args.h4_overlap_min_races,
            prop_threshold=args.h4_prop_threshold,
        )
        (args.output_dir / "h4_result.json").write_text(
            json.dumps(h4.as_dict(), indent=2), encoding="utf-8"
        )
        logger.info(
            "H4 (%s): concentracao=%.1f%% (requerido >=%.0f%%)",
            "confirmada" if h4.confirmed else "refutada",
            h4.overlap_concentration * 100 if h4.overlap_concentration == h4.overlap_concentration else float("nan"),
            args.h4_threshold * 100,
        )
    else:
        logger.warning("Sem coluna 'race' nas predições — H4 pulada.")

    h6 = None
    if "pixel_information" in errors.columns:
        h6 = test_h6_pixel_info_explains(errors, threshold=args.h6_threshold)
        (args.output_dir / "h6_result.json").write_text(
            json.dumps(h6.as_dict(), indent=2), encoding="utf-8"
        )
        logger.info(
            "H6 (%s): R²=%.3f (requerido >=%.2f)",
            "confirmada" if h6.confirmed else "refutada",
            h6.r2_pixel_info, args.h6_threshold,
        )
    else:
        logger.warning("Sem coluna 'pixel_information' nas predições — H6 pulada.")

    report = generate_report(decomp, h4, h6)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("Saídas em %s.", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
