"""Pipeline stage — Etapa 1: inferência MST via SkinToneNet.

Uso:
    python pipelines/03_mst_inference.py \\
        --config configs/mestrado/stages/etapa1_skintonenet.yaml \\
        --dataset-root data/FairFace \\
        --split val \\
        --output outputs/etapa1/fairface_val_mst.parquet

Enquanto os weights oficiais do SkinToneNet (Matias 2026, arXiv 2603.02475)
não são publicados, use --allow-imagenet-only para smoke tests. NÃO
publique resultados científicos nesse modo — a head não foi treinada.

Etapa 1 do Cap. 4 §4.2. Prazo formal: Nov/2026.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.mst import SkinToneNetInference  # noqa: E402

logger = logging.getLogger("pipelines.03_mst_inference")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência MST 10-classes com SkinToneNet.")
    p.add_argument("--config", type=Path, help="YAML da Etapa 1 (opcional).")
    p.add_argument("--weights", type=Path, help="Weights STW; sobrescreve config.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Raiz de imagens.")
    p.add_argument(
        "--labels-csv",
        type=Path,
        help="CSV com coluna 'file' (padrão FairFace). Se omitido, faz glob *.jpg.",
    )
    p.add_argument("--split", default="val", help="Nome do split (tag para logging).")
    p.add_argument("--output", type=Path, required=True, help="Arquivo .parquet de saída.")
    p.add_argument("--cache-dir", type=Path, default=None, help="Diretório do cache SQLite.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help=">0 restringe a N imagens (smoke).")
    p.add_argument(
        "--allow-imagenet-only",
        action="store_true",
        help="Permite rodar sem weights STW (WARN: head aleatória).",
    )
    return p.parse_args(argv)


def _resolve_weights(args: argparse.Namespace) -> Path | None:
    if args.weights is not None:
        return args.weights
    if args.config and args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        w = (cfg.get("inputs") or {}).get("skintonenet_weights")
        if w:
            return (REPO_ROOT / w).resolve()
    return None


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    root = args.dataset_root
    if args.labels_csv:
        df = pd.read_csv(args.labels_csv)
        col = "file" if "file" in df.columns else df.columns[0]
        paths = [root / p for p in df[col].tolist()]
    else:
        paths = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    weights_path = _resolve_weights(args)
    if weights_path is not None and not weights_path.exists():
        logger.warning("weights_path apontado (%s) não existe; tratando como ausente.", weights_path)
        weights_path = None

    paths = _collect_paths(args)
    if not paths:
        logger.error("Nenhuma imagem encontrada em %s.", args.dataset_root)
        return 2
    logger.info("Split=%s | imagens=%d | weights=%s", args.split, len(paths), weights_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with SkinToneNetInference(
        weights_path=weights_path,
        device=args.device,
        cache_dir=args.cache_dir,
        allow_imagenet_only=args.allow_imagenet_only,
    ) as infer:
        df = infer.infer_batch(paths, batch_size=args.batch_size)
    elapsed = time.time() - t0
    df["split"] = args.split
    df.to_parquet(args.output, index=False)
    logger.info(
        "Salvo %s (%d linhas) em %.2fs (%.1f img/s).",
        args.output,
        len(df),
        elapsed,
        len(df) / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
