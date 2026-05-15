"""Audit FairFace images for MTCNN multi-face detections.

For each original FairFace image (in ``data/raw/bucket/{train,val}``),
run MTCNN and record the number of faces detected. The output is a CSV
the user can sort/filter to decide which images to exclude from the
training set (kickoff diretriz nº 4 — multi-face cleanup).

This script does NOT modify the on-disk aligned dataset. It only emits:

* ``outputs/multi_face_audit.csv`` — one row per original image with
  columns: ``file``, ``race``, ``age``, ``gender``, ``service_test``,
  ``n_faces``, ``status`` (``ok`` | ``multi`` | ``zero`` | ``error``).
* ``outputs/multi_face_audit_summary.md`` — histogram of ``n_faces``,
  per-race breakdown, top examples per bucket. Defendable artefact for
  the dissertation methodology section.

Usage:
    python scripts/audit_multi_face.py \\
        --labels data/raw/fairface/fairface_labels.csv \\
        --images-root data/raw/bucket \\
        --output-dir outputs/audit

Tunables:
    --limit N           run only the first N images (smoke test)
    --device cuda|cpu   default auto
    --num-workers N     CPU image-load workers (default 4)
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class OriginalImageDataset(Dataset):
    """Yields (PIL.Image, file_relative_path) for each row of the labels CSV."""

    def __init__(self, df: pd.DataFrame, images_root: Path):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["file"]  # e.g. "train/12345.jpg"
        abs_path = self.images_root / rel_path
        try:
            with Image.open(abs_path) as raw:
                img = raw.convert("RGB").copy()
            return img, rel_path, ""
        except FileNotFoundError:
            return None, rel_path, "file_not_found"
        except Exception as exc:  # noqa: BLE001
            return None, rel_path, f"open_error:{exc}"


def _collate(batch):
    """Custom collate — keep PIL images in a list (variable sizes)."""
    return list(batch)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _bucket(n_faces: int) -> str:
    if n_faces == 0:
        return "zero"
    if n_faces == 1:
        return "ok"
    return "multi"


def audit(
    labels_csv: Path,
    images_root: Path,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    limit: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    if limit:
        df = df.head(limit)
    logging.info(f"Auditing {len(df)} images from {labels_csv}")

    detector = MTCNN(keep_all=True, device=str(device), post_process=False)

    dataset = OriginalImageDataset(df, images_root)
    loader = DataLoader(
        dataset,
        batch_size=1,  # MTCNN handles single images at a time; batching with
        num_workers=num_workers,  # variable-sized PIL is messy
        collate_fn=_collate,
    )

    results: list[dict] = []
    start = time.time()
    for batch in tqdm(loader, desc="MTCNN audit"):
        item = batch[0]
        img, rel_path, error = item
        if img is None:
            results.append({"file": rel_path, "n_faces": -1, "status": error})
            continue
        try:
            boxes, _probs = detector.detect(img)
            n_faces = 0 if boxes is None else len(boxes)
            results.append({"file": rel_path, "n_faces": n_faces, "status": _bucket(n_faces)})
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"{rel_path}: detection failed ({exc})")
            results.append({"file": rel_path, "n_faces": -1, "status": f"detect_error:{exc}"})

    elapsed = time.time() - start
    logging.info(f"Done in {elapsed:.1f}s ({len(results) / max(elapsed, 1):.1f} img/s)")

    # Merge with labels and persist
    audit_df = pd.DataFrame(results)
    merged = df.merge(audit_df, on="file", how="left")

    csv_path = output_dir / "multi_face_audit.csv"
    merged.to_csv(csv_path, index=False)
    logging.info(f"wrote {csv_path}")

    _write_summary(merged, output_dir / "multi_face_audit_summary.md")


def _write_summary(df: pd.DataFrame, out_path: Path) -> None:
    """Produce a human-readable Markdown summary of the audit."""
    n_total = len(df)
    hist = Counter(df["n_faces"].astype(int).clip(lower=-1).tolist())

    lines: list[str] = []
    lines.append("# FairFace multi-face audit")
    lines.append("")
    lines.append(f"**Total images audited:** {n_total}")
    lines.append("")
    lines.append("## n_faces distribution")
    lines.append("")
    lines.append("| n_faces | count | % of total |")
    lines.append("|---:|---:|---:|")
    for k in sorted(hist):
        pct = hist[k] / n_total * 100
        label = "error" if k == -1 else str(k)
        lines.append(f"| {label} | {hist[k]} | {pct:.2f}% |")
    lines.append("")

    # Bucket counts
    bucket_counts = df["status"].value_counts()
    lines.append("## Status bucket")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---:|")
    for status, count in bucket_counts.items():
        lines.append(f"| `{status}` | {count} |")
    lines.append("")

    # Multi-face breakdown by race
    multi_df = df[df["n_faces"] > 1]
    if not multi_df.empty:
        lines.append("## Multi-face images by race")
        lines.append("")
        breakdown = (
            multi_df.groupby("race")
            .agg(n_multi=("n_faces", "count"), avg_faces=("n_faces", "mean"))
            .sort_values("n_multi", ascending=False)
        )
        # Add per-race base rate for context
        per_race_total = df.groupby("race").size()
        breakdown["base_total"] = per_race_total
        breakdown["pct_of_race"] = breakdown["n_multi"] / breakdown["base_total"] * 100
        lines.append("| race | multi-face count | total | % of race | avg n_faces |")
        lines.append("|---|---:|---:|---:|---:|")
        for race, row in breakdown.iterrows():
            lines.append(
                f"| {race} | {int(row['n_multi'])} | {int(row['base_total'])} | "
                f"{row['pct_of_race']:.2f}% | {row['avg_faces']:.2f} |"
            )
        lines.append("")

        # Top examples
        lines.append("## Top 20 images by n_faces (multi-face only)")
        lines.append("")
        top = multi_df.nlargest(20, "n_faces")[["file", "race", "n_faces"]]
        lines.append("| file | race | n_faces |")
        lines.append("|---|---|---:|")
        for _, r in top.iterrows():
            lines.append(f"| `{r['file']}` | {r['race']} | {int(r['n_faces'])} |")
        lines.append("")

    # Zero-face breakdown
    zero_df = df[df["n_faces"] == 0]
    if not zero_df.empty:
        lines.append("## Zero-face images by race (MTCNN missed them)")
        lines.append("")
        breakdown = zero_df.groupby("race").size().sort_values(ascending=False)
        per_race_total = df.groupby("race").size()
        lines.append("| race | zero-face count | total | % of race |")
        lines.append("|---|---:|---:|---:|")
        for race, count in breakdown.items():
            total = per_race_total[race]
            lines.append(f"| {race} | {count} | {total} | {count / total * 100:.2f}% |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/raw/fairface/fairface_labels.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("data/raw/bucket"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="audit only first N images")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = _resolve_device(args.device)
    logging.info(f"device={device}")

    audit(
        labels_csv=args.labels,
        images_root=args.images_root,
        output_dir=args.output_dir,
        device=device,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
