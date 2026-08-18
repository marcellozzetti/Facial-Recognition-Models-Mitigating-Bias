"""Pipeline stage — Etapa 1 (parte A): treinamento do classificador MST próprio.

Estratégia decidida na reunião Ago/2026 com o orientador:
    - Treinamento interno sobre MSTE + Casual Conversations v2
    - Independência de release do SkinToneNet (Matias 2026)
    - STW será usado como benchmark externo se acesso for concedido

Uso:
    python pipelines/03a_train_mst_classifier.py \\
        --mste-root data/MSTE \\
        --ccv2-root data/CCv2 \\
        --output outputs/etapa1_own/ \\
        --seed 42 --max-epochs 30

Executar 3 vezes com --seed 42, 1, 2 para reproduzir o rigor
experimental declarado em Cap. 4 §4.10 (3 sementes independentes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from face_bias.mst.datasets import build_mst_dataset, class_balance  # noqa: E402
from face_bias.mst.trainer import MSTTrainer, stratified_split  # noqa: E402

logger = logging.getLogger("pipelines.03a_train_mst_classifier")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Etapa 1 — treino do MSTClassifier próprio.")
    p.add_argument("--mste-root", type=Path, help="Raiz do MSTE (Google).")
    p.add_argument("--ccv2-root", type=Path, help="Raiz do Casual Conversations v2 (Meta).")
    p.add_argument("--stw-root", type=Path, help="Raiz do STW (Matias 2026) — se disponível.")
    p.add_argument("--output", type=Path, required=True, help="Diretório de saída.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="vit_b_16")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    sources = []
    if args.mste_root:
        sources.append({"source": "mste", "root": args.mste_root})
    if args.ccv2_root:
        sources.append({"source": "ccv2", "root": args.ccv2_root})
    if args.stw_root:
        sources.append({"source": "stw", "root": args.stw_root})
    if not sources:
        logger.error("Nenhum dataset informado. Passe --mste-root e/ou --ccv2-root.")
        return 2

    df = build_mst_dataset(sources)
    logger.info("Datasets combinados: %d imagens.", len(df))
    logger.info("Balanço por classe MST:\n%s", class_balance(df).to_string())

    train_df, val_df = stratified_split(df, val_frac=args.val_frac, seed=args.seed)
    logger.info("Split: train=%d val=%d (val_frac=%.2f, seed=%d)",
                len(train_df), len(val_df), args.val_frac, args.seed)

    trainer = MSTTrainer(
        device=args.device,
        backbone=args.backbone,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    result = trainer.train(
        train_df, val_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=args.output,
    )

    summary = {
        "seed": args.seed,
        "backbone": args.backbone,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_val_acc": result.best_val_acc,
        "best_val_f1_macro": result.best_val_f1_macro,
        "checkpoint": str(result.checkpoint_path) if result.checkpoint_path else None,
        "sources": [s["source"] for s in sources],
    }
    (args.output / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / f"history_seed{args.seed}.json").write_text(
        json.dumps(result.history, indent=2), encoding="utf-8"
    )
    logger.info("Concluído. best_f1=%.4f  best_acc=%.4f  ckpt=%s",
                result.best_val_f1_macro, result.best_val_acc, result.checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
