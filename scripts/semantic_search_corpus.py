"""Semantic literature audit over the FairFace corpus (Diretriz 8).

This is a *thesis chapter generator*, not just a search tool. Given the
corpus produced by ``fetch_citing_corpus.py`` (FairFace references +
papers citing FairFace), it:

1. Loads ``docs/literature_corpus.csv`` (title/authors/year/venue/source)
   and merges abstracts from ``docs/literature_corpus_abstracts.csv``.
2. Encodes ``title + abstract`` into dense embeddings with a
   sentence-transformers model.
3. Computes cosine similarity (numpy — the corpus is ~550 docs, FAISS is
   unnecessary and fragile on Windows) between each research-question
   query and every document.
4. Writes ``docs/literature_semantic_audit.md`` formatted as a
   **dissertation chapter draft**: methodology, corpus characterisation,
   per-query ranked results with similarity scores, a dedicated novelty
   / overlap analysis for the thesis delta, gap synthesis, threats to
   validity, and a conclusion that feeds the objective.

Design choice: cosine over raw embeddings (not FAISS) keeps the pipeline
dependency-light (only ``sentence-transformers``) and fully reproducible
on the target Windows machine.

Usage:
    python scripts/semantic_search_corpus.py \\
        --input docs/literature_corpus.csv \\
        --abstracts docs/literature_corpus_abstracts.csv \\
        --output docs/literature_semantic_audit.md \\
        --top-k 15
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# Query set — aligned with the 2026-05-11 kickoff directives AND with the
# novelty-overlap check (sota_review.md §5.0). Queries Q11-Q14 specifically
# probe whether the thesis delta already exists in the literature.
QUERIES: list[tuple[str, str, str]] = [
    # (id, theme, query text)
    ("Q01_mlp_head", "diretriz-2",
     "Multi-layer perceptron classification head topology for race "
     "classification on FairFace"),
    ("Q02_optuna_hpo", "diretriz-3",
     "Optuna hyperparameter optimization for face recognition demographic "
     "fairness"),
    ("Q03_supcon", "diretriz-5",
     "Supervised contrastive learning SupCon for demographic equity in "
     "face attribute classification"),
    ("Q04_simclr", "diretriz-5",
     "SimCLR self-supervised pretraining for fair face recognition"),
    ("Q05_clip_face", "diretriz-5",
     "CLIP contrastive language-image pretraining for face attribute "
     "fairness"),
    ("Q06_adaface_magface", "diretriz-6",
     "AdaFace MagFace quality-adaptive margin loss for racial fairness"),
    ("Q07_multiface_clean", "diretriz-4",
     "Multi-face image label noise filtering and dataset cleaning for "
     "FairFace"),
    ("Q08_inequity_rate", "metric",
     "Inequity rate Gini coefficient demographic disparity metric face "
     "recognition"),
    ("Q09_undersampling_limits", "premise",
     "Limitations of class balancing and undersampling for demographic "
     "fairness in deep learning"),
    ("Q10_backbone", "axis-backbone",
     "Vision transformer ConvNeXt backbone for fair facial attribute "
     "classification"),
    # --- Novelty-overlap probes (the thesis delta) ---
    ("Q11_moo_arch_fairness", "novelty-risk",
     "Multi-objective architecture search Pareto front accuracy fairness "
     "trade-off face attribute classification"),
    ("Q12_pareto_epoch", "novelty-core",
     "Pareto-aware epoch selection criterion within multi-objective "
     "hyperparameter optimization trial for fairness"),
    ("Q13_decomposition", "novelty-core",
     "Controlled experimental decomposition isolating dataset quality "
     "versus model architecture contribution to demographic bias"),
    ("Q14_recipe_dependent_clean", "novelty-empirical",
     "Recipe-dependent effect of dataset cleaning on margin-based versus "
     "softmax-based losses for fairness"),
]


def load_corpus(corpus_csv: Path, abstracts_csv: Path | None) -> list[dict]:
    rows: list[dict] = []
    with open(corpus_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "authors": (r.get("authors") or "").strip(),
                    "year": (r.get("year") or "").strip(),
                    "venue": (r.get("venue") or "").strip(),
                    "source": (r.get("source") or "").strip(),
                    "abstract": "",
                }
            )

    if abstracts_csv and abstracts_csv.exists():
        ab: dict[str, str] = {}
        with open(abstracts_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ab[(r.get("title") or "").strip()] = (r.get("abstract") or "").strip()
        for row in rows:
            row["abstract"] = ab.get(row["title"], "")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/literature_corpus.csv"))
    parser.add_argument(
        "--abstracts", type=Path, default=Path("docs/literature_corpus_abstracts.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/literature_semantic_audit.md")
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.input.exists():
        logging.error(f"Corpus not found: {args.input}")
        logging.error("Run scripts/fetch_citing_corpus.py first.")
        return 1

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.error(
            "Missing deps. Install with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install 'face-bias[literature]'"
        )
        return 1

    corpus = load_corpus(args.input, args.abstracts)
    n_total = len(corpus)
    n_abstr = sum(1 for r in corpus if r["abstract"])
    by_source = Counter(r["source"] for r in corpus)
    by_year = Counter(r["year"] for r in corpus if r["year"])
    logging.info(f"Corpus: {n_total} docs, {n_abstr} with abstracts")

    logging.info(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)

    docs_text = [
        f"{r['title']}. {r['abstract']}" if r["abstract"] else r["title"]
        for r in corpus
    ]
    logging.info("Encoding corpus ...")
    doc_emb = model.encode(
        docs_text, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    q_texts = [q[2] for q in QUERIES]
    q_emb = model.encode(q_texts, normalize_embeddings=True)

    # Cosine similarity = dot product of L2-normalised vectors.
    sims = np.asarray(q_emb) @ np.asarray(doc_emb).T  # (n_queries, n_docs)

    # ---- Build the chapter-grade markdown ----
    lines: list[str] = []
    A = lines.append

    A("# Auditoria Semântica da Literatura — Capítulo (rascunho)\n")
    A("> **Documento gerado por** `scripts/semantic_search_corpus.py`.")
    A("> Material destinado à escrita da dissertação (Diretriz 8). "
      "Reproduzível: re-rodar o script regenera este arquivo.\n")
    A(f"**Data de geração:** {date.today().isoformat()}  ")
    A(f"**Modelo de embedding:** `{args.model}` (sentence-transformers)  ")
    A("**Métrica de similaridade:** cosseno sobre embeddings L2-normalizados  ")
    A(f"**Top-K por query:** {args.top_k}\n")
    A("---\n")

    # 1. Methodology
    A("## 1. Metodologia\n")
    A("Esta auditoria semântica complementa a revisão sistemática "
      "([sota_review.md](sota_review.md)). Enquanto aquela usa busca "
      "booleana em venues, esta usa **similaridade semântica densa** "
      "sobre o corpus completo de papers que citam o FairFace + as "
      "referências do próprio FairFace, para mitigar o viés de "
      "palavra-chave da busca booleana.\n")
    A("**Protocolo:**\n")
    A("1. Corpus: `docs/literature_corpus.csv` "
      f"({n_total} documentos; {n_abstr} com abstract).\n")
    A("2. Representação: `título + abstract` codificado em embedding denso.\n")
    A("3. Consultas: 14 *research queries* — 10 alinhadas às diretrizes do "
      "orientador, 4 sondas de **novidade/overlap** (Q11–Q14) que testam "
      "diretamente se o delta da tese já existe.\n")
    A("4. Ranqueamento: top-K por similaridade de cosseno.\n")
    A("5. Análise: leitura dos top-K das sondas de novidade para "
      "confirmar/refutar overlap.\n")
    A("\n**Ameaças à validade declaradas:** (i) corpus OpenAlex pode não "
      "ser exaustivo vs Google Scholar (479 vs ~900 brutos — diferença "
      "por deduplicação); (ii) similaridade semântica captura proximidade "
      "temática, não prova ausência de overlap — leitura humana dos top-K "
      "é mandatória antes de afirmar novidade; (iii) abstracts ausentes "
      "degradam o sinal para alguns docs.\n")
    A("---\n")

    # 2. Corpus characterisation
    A("## 2. Caracterização do corpus\n")
    A("| Fonte | Qtd |\n|---|---:|")
    for s, c in by_source.most_common():
        A(f"| `{s}` | {c} |")
    A("")
    A("**Distribuição temporal (papers que citam):**\n")
    A("| Ano | Qtd |\n|---|---:|")
    for y in sorted(by_year):
        if y and y >= "2020":
            A(f"| {y} | {by_year[y]} |")
    A("\n---\n")

    # 3. Per-query results
    A("## 3. Resultados por query\n")
    for qi, (qid, qtheme, qtext) in enumerate(QUERIES):
        order = np.argsort(-sims[qi])[: args.top_k]
        A(f"### {qid} ({qtheme})\n")
        A(f"> *Query:* {qtext}\n")
        A("| # | sim | Ano | Título | Fonte |")
        A("|---:|---:|---:|---|---|")
        for rank, di in enumerate(order, 1):
            d = corpus[di]
            title = d["title"][:110].replace("|", "/")
            A(f"| {rank} | {sims[qi][di]:.3f} | {d['year']} | {title} | "
              f"{d['source']} |")
        A("")
    A("---\n")

    # 4. Novelty / overlap analysis
    A("## 4. Análise de novidade (sondas Q11–Q14)\n")
    A("Esta seção é o coração do capítulo para a defesa: ela testa se a "
      "contribuição da tese já existe na literatura.\n")
    novelty_ids = {"Q11_moo_arch_fairness", "Q12_pareto_epoch",
                   "Q13_decomposition", "Q14_recipe_dependent_clean"}
    for qi, (qid, qtheme, qtext) in enumerate(QUERIES):
        if qid not in novelty_ids:
            continue
        order = np.argsort(-sims[qi])[:5]
        top_sim = sims[qi][order[0]]
        A(f"### {qid}\n")
        A(f"> *Query:* {qtext}\n")
        A(f"- **Similaridade máxima no corpus:** {top_sim:.3f}")
        verdict = (
            "ALTA — possível overlap, **leitura humana obrigatória** dos top-5"
            if top_sim >= 0.75
            else "MÉDIA — temas adjacentes, provável diferenciação"
            if top_sim >= 0.60
            else "BAIXA — nenhum trabalho semanticamente próximo no corpus"
        )
        A(f"- **Veredito automático:** {verdict}\n")
        A("Top-5 mais próximos (para leitura manual):\n")
        for rank, di in enumerate(order, 1):
            d = corpus[di]
            A(f"{rank}. ({sims[qi][di]:.3f}, {d['year']}) "
              f"{d['title'][:130]}")
        A("")
    A("> **Nota metodológica:** similaridade alta NÃO prova overlap (pode "
      "ser tema próximo com método distinto); similaridade baixa é "
      "evidência mais forte de novidade. A decisão final exige leitura "
      "dos PDFs dos top-5 de Q12/Q13 (o delta-núcleo) e cruzamento com "
      "sota_review.md §5.0.\n")
    A("---\n")

    # 5. Conclusion
    A("## 5. Síntese para a escrita da dissertação\n")
    A("- Esta auditoria + a revisão sistemática (sota_review.md) formam o "
      "**capítulo de revisão de literatura** da qualificação.\n")
    A("- As sondas Q11–Q14 alimentam a **seção de posicionamento da "
      "contribuição** (delta vs SOTA).\n")
    A("- O ranking por query alimenta as **subseções temáticas** "
      "(fairness metrics, contrastivo, losses adaptativas, backbones).\n")
    A("- Ação obrigatória antes da submissão de paper: ler os PDFs dos "
      "top-5 de Q12 e Q13 e registrar o veredito de overlap final em "
      "sota_review.md §5.0.\n")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Wrote {args.output} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
