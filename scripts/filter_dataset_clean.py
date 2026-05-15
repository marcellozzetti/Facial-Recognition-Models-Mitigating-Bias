"""Generate the n_faces==1 cleaned label CSV (Option A — kickoff 2026-05-14).

Filters ``data/raw/fairface/fairface_labels.csv`` by joining with the
multi-face audit at ``outputs/audit/multi_face_audit.csv`` and keeping
only rows whose original image had exactly one face detected by MTCNN.
Writes ``data/raw/fairface/fairface_labels_clean.csv`` and reports the
per-race breakdown so the methodological choice is traceable.

This is a one-shot script; the result is checked into the repo so the
two replication runs (R2: Exp 5 + Exp 6 on clean dataset) are deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/raw/fairface/fairface_labels.csv")
    parser.add_argument("--audit", default="outputs/audit/multi_face_audit.csv")
    parser.add_argument(
        "--out", default="data/raw/fairface/fairface_labels_clean.csv"
    )
    parser.add_argument(
        "--n-faces", type=int, default=1, help="Keep rows with exactly this many faces."
    )
    args = parser.parse_args(argv)

    labels_path = Path(args.labels)
    audit_path = Path(args.audit)
    out_path = Path(args.out)

    if not labels_path.exists():
        print(f"ERROR: labels CSV not found: {labels_path}", file=sys.stderr)
        return 1
    if not audit_path.exists():
        print(f"ERROR: audit CSV not found: {audit_path}", file=sys.stderr)
        return 1

    labels = pd.read_csv(labels_path)
    audit = pd.read_csv(audit_path)[["file", "n_faces", "status"]]

    print(f"Labels rows:        {len(labels):>7d}")
    print(f"Audit rows:         {len(audit):>7d}")

    merged = labels.merge(audit, on="file", how="inner", validate="one_to_one")
    print(f"After join (inner): {len(merged):>7d}")
    missing = len(labels) - len(merged)
    if missing:
        print(f"Note: {missing} label rows had no audit entry and were dropped.")

    clean = merged.loc[merged["n_faces"] == args.n_faces].copy()
    clean = clean.drop(columns=["n_faces", "status"])
    print(f"After n_faces=={args.n_faces} filter: {len(clean):>7d}")
    print(f"Removed (multi or zero): {len(merged) - len(clean):>7d} "
          f"({(len(merged) - len(clean)) / len(merged):.2%})")

    print("\nPer-race breakdown after filter:")
    print(clean["race"].value_counts().to_string())

    print("\nFor reference — original per-race breakdown:")
    print(labels["race"].value_counts().to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(clean)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
