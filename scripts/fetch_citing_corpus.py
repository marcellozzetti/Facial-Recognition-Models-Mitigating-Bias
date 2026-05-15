"""Fetch the corpus of papers that CITE the FairFace paper, via OpenAlex.

OpenAlex (https://openalex.org) is a free, fully open scholarly graph —
no API key, no account, no rate-limit headaches for this volume. This
script:

1. Resolves the FairFace paper's OpenAlex Work ID (by DOI, fallback title).
2. Pages through every Work that cites it (`filter=cites:<id>`), 200/page,
   cursor-paginated.
3. Reconstructs each abstract from OpenAlex's inverted index.
4. Appends rows to ``docs/literature_corpus.csv`` with the SAME schema as
   the FairFace-references seed (title, authors, year, venue, theme,
   source), using ``source=fairface-citing`` and ``theme=`` (left blank;
   the semantic search assigns topics later).

This is the "descendants" half of the novelty corpus (the FairFace
*references* seed — the "ancestors" — is already in the CSV). Together
they feed ``scripts/semantic_search_corpus.py`` (Diretriz 8).

Usage:
    python scripts/fetch_citing_corpus.py
    python scripts/fetch_citing_corpus.py --out docs/literature_corpus.csv
    python scripts/fetch_citing_corpus.py --mailto you@example.com  # polite pool, faster

OpenAlex etiquette: passing --mailto puts you in the "polite pool"
(faster, more reliable). It is optional but recommended.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

OPENALEX = "https://api.openalex.org"
FAIRFACE_DOI = "10.1109/WACV48630.2021.00159"  # FairFace, WACV 2021 (IEEE 9423296)
FAIRFACE_TITLE = "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age"


def _get(url: str, params: dict, mailto: str | None) -> dict:
    if mailto:
        params = {**params, "mailto": mailto}
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 503):
            wait = 2 ** attempt
            print(f"  HTTP {r.status_code}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex request failed after retries: {url}")


def resolve_fairface_id(mailto: str | None) -> str:
    """Return the OpenAlex Work ID for the FairFace paper."""
    # Try DOI first (exact).
    try:
        data = _get(f"{OPENALEX}/works/doi:{FAIRFACE_DOI}", {}, mailto)
        wid = data["id"].rsplit("/", 1)[-1]
        print(f"Resolved FairFace by DOI -> {wid} ({data.get('title')!r})")
        return wid
    except Exception as e:  # noqa: BLE001
        print(f"DOI resolution failed ({e}); falling back to title search.", file=sys.stderr)

    data = _get(
        f"{OPENALEX}/works",
        {"search": FAIRFACE_TITLE, "per-page": 1},
        mailto,
    )
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Could not resolve FairFace Work ID by title.")
    wid = results[0]["id"].rsplit("/", 1)[-1]
    print(f"Resolved FairFace by title -> {wid} ({results[0].get('title')!r})")
    return wid


def _abstract_from_inverted_index(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_citing(work_id: str, mailto: str | None) -> list[dict]:
    """Page through every Work that cites ``work_id``."""
    rows: list[dict] = []
    cursor = "*"
    page = 0
    while cursor:
        page += 1
        data = _get(
            f"{OPENALEX}/works",
            {
                "filter": f"cites:{work_id}",
                "per-page": 200,
                "cursor": cursor,
                "select": "id,title,publication_year,authorships,"
                "primary_location,abstract_inverted_index",
            },
            mailto,
        )
        batch = data.get("results", [])
        for w in batch:
            authors = "; ".join(
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])
            )
            venue = ""
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            venue = src.get("display_name", "") or ""
            rows.append(
                {
                    "title": (w.get("title") or "").replace("\n", " ").strip(),
                    "authors": authors,
                    "year": w.get("publication_year") or "",
                    "venue": venue,
                    "theme": "",  # assigned later by semantic clustering
                    "source": "fairface-citing",
                    "abstract": _abstract_from_inverted_index(
                        w.get("abstract_inverted_index")
                    ),
                }
            )
        total = data.get("meta", {}).get("count", "?")
        print(f"  page {page}: +{len(batch)} (running {len(rows)}/{total})")
        cursor = data.get("meta", {}).get("next_cursor")
        if not batch:
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/literature_corpus.csv")
    ap.add_argument(
        "--mailto",
        default=None,
        help="Your email — puts requests in OpenAlex's faster 'polite pool'.",
    )
    ap.add_argument(
        "--abstract-out",
        default="docs/literature_corpus_abstracts.csv",
        help="Separate CSV with abstracts (kept apart so the main corpus "
        "stays diff-friendly).",
    )
    args = ap.parse_args(argv)

    wid = resolve_fairface_id(args.mailto)
    print(f"Fetching works that cite {wid} ...")
    rows = fetch_citing(wid, args.mailto)
    print(f"Total citing works fetched: {len(rows)}")

    out = Path(args.out)
    existing = out.exists()
    # Append to the seed CSV (the 76 FairFace references already there).
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for r in rows:
            writer.writerow(
                [r["title"], r["authors"], r["year"], r["venue"], r["theme"], r["source"]]
            )
    print(f"{'Appended to' if existing else 'Wrote'} {out} (+{len(rows)} rows)")

    # Abstracts in a side file (large; keep the main corpus diff-friendly).
    ab = Path(args.abstract_out)
    with open(ab, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["title", "year", "abstract"])
        for r in rows:
            writer.writerow([r["title"], r["year"], r["abstract"]])
    print(f"Wrote {ab} ({len(rows)} abstracts)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
