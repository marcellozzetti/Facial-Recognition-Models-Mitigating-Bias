"""Pipeline Etapa 1 — geração dos splits treino/val do STW.

Varre data/raw/stw/images/all_cropped_images_by_class/{1..10}/ e cria
splits estratificados (por classe) com razão configúravel (default 80/20).

Uso:
    python -I pipelines/03_prepare_stw_splits.py
    python -I pipelines/03_prepare_stw_splits.py --val-ratio 0.15 --seed 42
    python -I pipelines/03_prepare_stw_splits.py --dry-run

Saída:
    data/raw/stw/stw_train.csv   (substitui o anterior)
    data/raw/stw/stw_val.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "stw" / "images" / "all_cropped_images_by_class"
OUT_DIR = REPO_ROOT / "data" / "raw" / "stw"
EXTS = {".jpg", ".jpeg", ".png"}
MST_CLASSES = list(range(1, 11))


def collect_images(images_root: Path) -> dict[int, list[Path]]:
    """Retorna {classe: [paths]} para todos os arquivos encontrados."""
    per_class: dict[int, list[Path]] = {}
    for cls in MST_CLASSES:
        cls_dir = images_root / str(cls)
        if not cls_dir.exists():
            print(f"[AVISO] Pasta não encontrada: {cls_dir}")
            per_class[cls] = []
            continue
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS]
        per_class[cls] = sorted(files)
    return per_class


def make_split(
    per_class: dict[int, list[Path]],
    val_ratio: float,
    seed: int,
    images_root: Path,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Retorna (train_rows, val_rows) onde cada row = (path_relativo, classe)."""
    rng = random.Random(seed)
    train_rows: list[tuple[str, int]] = []
    val_rows:   list[tuple[str, int]] = []

    for cls, files in per_class.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_files   = shuffled[:n_val]
        train_files = shuffled[n_val:]

        stw_root = images_root.parent.parent  # data/raw/stw
        for f in train_files:
            rel = f.relative_to(stw_root).as_posix()
            train_rows.append((rel, cls))
        for f in val_files:
            rel = f.relative_to(stw_root).as_posix()
            val_rows.append((rel, cls))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_csv(rows: list[tuple[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("path,class\n")
        for rel, cls in rows:
            f.write(f"{rel},{cls}\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gera splits treino/val do STW a partir das imagens cortadas.")
    p.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    p.add_argument("--out-dir",     type=Path, default=OUT_DIR)
    p.add_argument("--val-ratio",   type=float, default=0.20, help="Fração de validação (default 0.20)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--dry-run",     action="store_true", help="Só imprime estatísticas, não salva arquivos")
    args = p.parse_args(argv)

    per_class = collect_images(args.images_root)

    total = sum(len(v) for v in per_class.values())
    print(f"\n=== STW — imagens encontradas ===")
    for cls in MST_CLASSES:
        n = len(per_class[cls])
        bar = "#" * min(40, int(n / total * 400)) if total > 0 else ""
        print(f"  MST {cls:2d}: {n:5d}  {bar}")
    print(f"  TOTAL : {total:5d}\n")

    if total == 0:
        print("ERRO: nenhuma imagem encontrada. Verifique --images-root")
        return 1

    train_rows, val_rows = make_split(per_class, args.val_ratio, args.seed, args.images_root)

    # Estatísticas do split
    from collections import Counter
    train_dist = Counter(cls for _, cls in train_rows)
    val_dist   = Counter(cls for _, cls in val_rows)
    print(f"Split {int((1-args.val_ratio)*100)}/{int(args.val_ratio*100)} (seed={args.seed}):")
    print(f"  Treino : {len(train_rows):6d} imagens")
    print(f"  Val    : {len(val_rows):6d} imagens")
    print(f"  Total  : {len(train_rows)+len(val_rows):6d} imagens")
    print()
    print(f"  {'Classe':8} {'Treino':>8} {'Val':>6} {'Ratio val':>10}")
    for cls in MST_CLASSES:
        n_tr = train_dist.get(cls, 0)
        n_vl = val_dist.get(cls, 0)
        ratio = n_vl / (n_tr + n_vl) if (n_tr + n_vl) > 0 else 0
        print(f"  MST {cls:2d}  {n_tr:8d} {n_vl:6d} {ratio:10.1%}")

    if args.dry_run:
        print("\n[dry-run] Nenhum arquivo salvo.")
        return 0

    train_path = args.out_dir / "stw_train.csv"
    val_path   = args.out_dir / "stw_val.csv"

    # Backup do split anterior se existir
    for path in [train_path, val_path]:
        if path.exists():
            backup = path.with_suffix(".csv.bak")
            path.rename(backup)
            print(f"[backup] {path.name} -> {backup.name}")

    write_csv(train_rows, train_path)
    write_csv(val_rows,   val_path)
    print(f"\n[ok] {train_path}")
    print(f"[ok] {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
