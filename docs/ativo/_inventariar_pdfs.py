"""Inventaria PDFs do corpus e identifica gaps.

Lê todas as fichas markdown em 04_pesquisa_bibliografica/ e cruza com
os PDFs presentes em pdfs/. Gera relatório de gap e candidatos para
download a partir do arXiv.

Uso:
    python _inventariar_pdfs.py
    -> imprime relatorio + escreve _pdfs_inventario.md
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "04_pesquisa_bibliografica"
PDFS_DIR = CORPUS_DIR / "pdfs"
OUT_FILE = HERE / "_pdfs_inventario.md"


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*?)$", re.MULTILINE)


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.search(text)
    if not m:
        return {}
    body = m.group(1)
    fields = {}
    for fm in FIELD_RE.finditer(body):
        k, v = fm.group(1), fm.group(2).strip()
        v = v.strip('"').strip("'")
        if v.lower() in ("null", "none", ""):
            v = None
        fields[k] = v
    return fields


def list_fichas() -> list[Path]:
    files = sorted(CORPUS_DIR.glob("*.md"))
    # remove arquivos de meta (começam com _, ou README, INDEX)
    return [f for f in files if not f.name.startswith("_") and f.name not in {"README.md", "INDEX.md"}]


def list_pdfs() -> set[str]:
    return {p.stem for p in PDFS_DIR.glob("*.pdf")}


def stem_candidates(ficha_name: str) -> list[str]:
    """Possíveis stems de pdf para uma ficha (heurística)."""
    base = ficha_name.removesuffix(".md")
    # padrões observados: {nome}, {nome}_v2, {nome}_film, etc.
    return [base, base.replace("_", "-")]


def main() -> None:
    fichas = list_fichas()
    pdfs = list_pdfs()

    rows = []
    have_pdf_count = 0
    by_status = defaultdict(int)
    missing_with_arxiv = []
    missing_no_arxiv = []

    for f in fichas:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        name = fm.get("name") or f.stem
        status = fm.get("status_verificacao") or "?"
        arxiv = fm.get("arxiv_id")
        doi = fm.get("doi")
        url = fm.get("url_primario")

        by_status[status] += 1

        # busca PDF aproximado por stem
        stem = f.stem
        has_pdf = False
        pdf_stem = None
        for cand in pdfs:
            if cand.startswith(stem.split("_")[0]) and stem.split("_")[1:2] and stem.split("_")[1] in cand:
                has_pdf = True
                pdf_stem = cand
                break
        if not has_pdf:
            # busca mais permissiva: qualquer arquivo que comece com o primeiro segmento
            first = stem.split("_")[0]
            matches = [c for c in pdfs if c.startswith(first)]
            if matches:
                has_pdf = True
                pdf_stem = matches[0]

        if has_pdf:
            have_pdf_count += 1
        else:
            entry = {
                "ficha": stem,
                "name": name,
                "status": status,
                "arxiv": arxiv,
                "doi": doi,
                "url": url,
            }
            if arxiv:
                missing_with_arxiv.append(entry)
            else:
                missing_no_arxiv.append(entry)

        rows.append({
            "ficha": stem,
            "name": name,
            "status": status,
            "pdf": pdf_stem or "—",
            "arxiv": arxiv or "—",
        })

    # === REPORT ===
    out = []
    out.append("# Inventário de PDFs do corpus\n")
    out.append(f"Gerado em: 2026-06-15\n\n")
    out.append(f"- Total de fichas: **{len(fichas)}**\n")
    out.append(f"- Fichas com PDF identificado: **{have_pdf_count}**\n")
    out.append(f"- Fichas sem PDF: **{len(fichas) - have_pdf_count}**\n")
    out.append(f"- PDFs presentes em `pdfs/`: **{len(pdfs)}**\n\n")

    out.append("## Distribuição por status_verificacao\n\n")
    for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
        out.append(f"- `{s}`: {n}\n")

    out.append("\n## Gap A — fichas SEM PDF mas COM arxiv_id (downloadáveis automaticamente)\n\n")
    out.append(f"Total: **{len(missing_with_arxiv)}**\n\n")
    out.append("| Ficha | Status | arXiv ID | URL |\n")
    out.append("|---|---|---|---|\n")
    for e in sorted(missing_with_arxiv, key=lambda x: x["ficha"]):
        out.append(f"| `{e['ficha']}` | {e['status']} | {e['arxiv']} | {e['url'] or '—'} |\n")

    out.append("\n## Gap B — fichas SEM PDF e SEM arxiv_id (manual)\n\n")
    out.append(f"Total: **{len(missing_no_arxiv)}**\n\n")
    out.append("| Ficha | Status | DOI / URL |\n")
    out.append("|---|---|---|\n")
    for e in sorted(missing_no_arxiv, key=lambda x: x["ficha"]):
        ref = e["doi"] or e["url"] or "—"
        out.append(f"| `{e['ficha']}` | {e['status']} | {ref} |\n")

    out.append("\n## Inventário completo\n\n")
    out.append("| Ficha | Status | PDF presente | arXiv |\n")
    out.append("|---|---|---|---|\n")
    for r in rows:
        out.append(f"| `{r['ficha']}` | {r['status']} | {r['pdf']} | {r['arxiv']} |\n")

    OUT_FILE.write_text("".join(out), encoding="utf-8")

    print(f"Total de fichas: {len(fichas)}")
    print(f"Com PDF: {have_pdf_count}")
    print(f"Sem PDF: {len(fichas) - have_pdf_count}")
    print(f"  - com arxiv_id: {len(missing_with_arxiv)} (downloadáveis)")
    print(f"  - sem arxiv_id: {len(missing_no_arxiv)} (manual)")
    print(f"\nRelatório completo: {OUT_FILE}")


if __name__ == "__main__":
    main()
