"""Build a local semantic index over a corpus of FairFace-adjacent papers.

Given a CSV (or BibTeX) with paper metadata, this script:

1. Enriches missing abstracts via the Semantic Scholar API (free, rate-limited).
2. Encodes title+abstract into sentence embeddings.
3. Indexes them with FAISS for fast similarity search.
4. Runs a configurable set of semantic queries aligned with the master's
   research questions and writes a Markdown report listing the top-K
   most similar papers per query.

The output lives in ``docs/literature_semantic_audit.md`` — used to
position the master's contribution against ~900 published works on
FairFace and demographic fairness in face recognition.

Dependencies (not yet in pyproject.toml — add to ``[project.optional-dependencies]
literature`` if/when the corpus arrives):

    pip install sentence-transformers faiss-cpu semanticscholar bibtexparser

Usage:
    python scripts/semantic_search_corpus.py \\
        --input docs/literature_corpus.csv \\
        --output docs/literature_semantic_audit.md \\
        --top-k 20

The CSV must have at least columns: title, abstract (optional), year, authors,
venue. Missing abstracts are pulled from Semantic Scholar by title match.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Default query set — aligned with the directives from the 2026-05-11 meeting.
# Edit this list to refine the search, or pass --queries-file to override.
DEFAULT_QUERIES: list[tuple[str, str]] = [
    ("Q1_mlp_head", "Multi-layer perceptron classification head for race classification on FairFace"),
    ("Q2_optuna_hpo", "Hyperparameter optimization with Optuna for face recognition fairness"),
    ("Q3_supcon", "Supervised contrastive learning SupCon for demographic equity in face classification"),
    ("Q4_simclr", "SimCLR self-supervised pretraining for face recognition fairness"),
    ("Q5_clip_face", "CLIP contrastive language-image pretraining applied to face attribute classification"),
    ("Q6_adaface_magface", "AdaFace MagFace quality-adaptive margin loss for race fairness"),
    ("Q7_combination", "Combination of architecture optimization and contrastive learning and adaptive loss for face recognition fairness"),
    ("Q8_multiface_noise", "Multi-face image label noise filtering FairFace dataset cleaning"),
    ("Q9_inequity_rate", "Inequity rate fairness metric Gini coefficient face recognition"),
    ("Q10_undersampling_limits", "Limitations of class undersampling for demographic fairness in deep learning"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local semantic index over a FairFace literature corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with columns: title, abstract (opt), year, authors, venue")
    parser.add_argument("--output", type=Path, default=Path("docs/literature_semantic_audit.md"))
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of nearest neighbours per query (default: 20)")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5",
                        help="sentence-transformers model")
    parser.add_argument("--enrich-abstracts", action="store_true",
                        help="Fetch missing abstracts from Semantic Scholar")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logging.error(f"Input corpus not found at {args.input}")
        print("\nThis script is a scaffold — provide the corpus first.")
        print("Expected: docs/literature_corpus.csv with title/abstract/year/authors/venue.")
        print("Possible sources for the ~900 FairFace references:")
        print("  - Zotero / Mendeley export -> CSV")
        print("  - Semantic Scholar bulk download")
        print("  - BibTeX from the MBA literature review")
        return 1

    # The rest of the pipeline is intentionally not implemented yet — it lights up
    # once the corpus exists. See the module docstring for the full flow.
    print("Scaffold ready. Implement the pipeline once docs/literature_corpus.csv exists.")
    print(f"Will run {len(DEFAULT_QUERIES)} queries, top-{args.top_k} each, with {args.model}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
