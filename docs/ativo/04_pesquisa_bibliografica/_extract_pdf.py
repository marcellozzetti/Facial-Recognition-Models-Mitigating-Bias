"""Extrator simples de texto de PDFs para a Pesquisa Bibliográfica.

Uso:
    python _extract_pdf.py pdfs/karkkainen_2021_fairface.pdf
    python _extract_pdf.py pdfs/karkkainen_2021_fairface.pdf --pages 1-5
    python _extract_pdf.py pdfs/karkkainen_2021_fairface.pdf --download 1908.04913

Sai texto bruto para stdout. Não interpreta — só extrai.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from pypdf import PdfReader


def parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < total]


def download_arxiv(arxiv_id: str, dest: Path) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def extract(pdf_path: Path, pages_spec: str | None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = parse_pages(pages_spec, len(reader.pages))
    chunks: list[str] = []
    for idx in pages:
        page = reader.pages[idx]
        text = page.extract_text() or ""
        chunks.append(f"\n===== PAGE {idx + 1} =====\n{text}")
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="path to PDF (relative to cwd)")
    ap.add_argument("--pages", type=str, default=None, help='e.g. "1-5" or "1,3,7-9"')
    ap.add_argument("--download", type=str, default=None, help="arXiv id to download into pdf path if missing")
    ap.add_argument("--url", type=str, default=None, help="arbitrary PDF URL to download into pdf path if missing")
    args = ap.parse_args()

    if args.download and not args.pdf.exists():
        download_arxiv(args.download, args.pdf)
    elif args.url and not args.pdf.exists():
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(args.url, args.pdf)

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    sys.stdout.reconfigure(encoding="utf-8")
    print(extract(args.pdf, args.pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
