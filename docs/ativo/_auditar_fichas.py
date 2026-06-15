"""Auditoria sistematica das 101 fichas do corpus.

Para cada ficha, verifica:
- status_verificacao
- autores ('a verificar'?)
- venue (correto?)
- ano
- fonte_leitura (que tipo de leitura?)
- profundidade do conteudo (numero de linhas, secoes presentes)
- PDF disponivel?
- Conexoes mapeadas?

Classifica em 4 niveis de qualidade:
- A (excelente): >150 linhas, autoria completa, sem 'a verificar',
  todas 12 secoes presentes, PDF disponivel.
- B (boa): >80 linhas, sem 'a verificar' em autores, pelo menos
  secoes 1-2 + 7-12 presentes.
- C (basica): >40 linhas, alguma deficiencia (autores, conteudo raso).
- D (raso): <=40 linhas OU 'a verificar' em multiplos campos.

Uso:
    python _auditar_fichas.py
    -> imprime relatorio + escreve _auditoria_fichas_relatorio.md
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "04_pesquisa_bibliografica"
PDFS_DIR = CORPUS_DIR / "pdfs"
OUT_FILE = HERE / "_auditoria_fichas_relatorio.md"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*?)$", re.MULTILINE)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.search(text)
    if not m:
        return {}, text
    body = m.group(1)
    fields = {}
    for fm in FIELD_RE.finditer(body):
        k, v = fm.group(1), fm.group(2).strip()
        v = v.strip('"').strip("'")
        if v.lower() in ("null", "none", ""):
            v = None
        fields[k] = v
    rest = text[m.end():]
    return fields, rest


def list_fichas() -> list[Path]:
    files = sorted(CORPUS_DIR.glob("*.md"))
    return [f for f in files if not f.name.startswith("_") and f.name not in {"README.md", "INDEX.md"}]


def has_pdf(stem: str) -> bool:
    if (PDFS_DIR / f"{stem}.pdf").exists():
        return True
    # heuristica: comeca com primeiro segmento
    first = stem.split("_")[0]
    for p in PDFS_DIR.glob("*.pdf"):
        if p.stem.startswith(first):
            return True
    return False


def count_sections(body: str) -> set:
    """Retorna set de numeros de secao presentes (1-12)."""
    sections = set()
    for m in re.finditer(r"^##\s+(\d+)", body, re.MULTILINE):
        sections.add(int(m.group(1)))
    # tambem aceita formatos como "## 1-2." ou "## 9-12."
    for m in re.finditer(r"^##\s+(\d+)-(\d+)", body, re.MULTILINE):
        for n in range(int(m.group(1)), int(m.group(2)) + 1):
            sections.add(n)
    return sections


def classify(linhas: int, autores_verificar: bool, sections: set, has_pdf_: bool, status: str) -> str:
    if status == "OVERVIEW_ONLY":
        return "D"  # tudo OVERVIEW_ONLY ja era D por definicao
    crit_secoes = {1, 2, 7, 8}.issubset(sections)
    todas_secoes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.issubset(sections) or {1, 2, 7, 8, 9}.issubset(sections)
    if linhas > 200 and not autores_verificar and todas_secoes and has_pdf_:
        return "A"
    if linhas > 80 and not autores_verificar and crit_secoes:
        return "B"
    if linhas > 40 and crit_secoes:
        return "C"
    return "D"


def main() -> None:
    fichas = list_fichas()
    rows = []
    counts = Counter()

    for f in fichas:
        text = f.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        n_linhas = len(text.splitlines())
        status = fm.get("status_verificacao") or "?"
        autores = fm.get("autores") or ""
        autores_verificar = "a verificar" in autores.lower()
        sections = count_sections(body)
        pdf_ok = has_pdf(f.stem)
        venue = fm.get("venue") or ""
        venue_curto = len(venue) < 30  # curto significa generico

        qualidade = classify(n_linhas, autores_verificar, sections, pdf_ok, status)
        counts[qualidade] += 1

        flags = []
        if autores_verificar:
            flags.append("autoria")
        if "a verificar" in venue.lower():
            flags.append("venue")
        if not pdf_ok:
            flags.append("sem-pdf")
        if n_linhas < 40:
            flags.append("raso")
        if not {1, 2, 7, 8}.issubset(sections):
            flags.append("sec-incompleta")

        rows.append({
            "ficha": f.stem,
            "status": status,
            "qualidade": qualidade,
            "linhas": n_linhas,
            "secoes": len(sections),
            "pdf": pdf_ok,
            "flags": ",".join(flags) or "—",
        })

    # report
    out = []
    out.append("# Auditoria das 101 fichas do corpus\n\n")
    out.append("Gerada por `_auditar_fichas.py` em 2026-06-15.\n\n")
    out.append("## Distribuição por qualidade\n\n")
    for q in "ABCD":
        out.append(f"- **{q}**: {counts[q]} fichas\n")
    out.append("\n## Por status_verificacao\n\n")
    status_counts = Counter(r["status"] for r in rows)
    for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        out.append(f"- `{s}`: {n}\n")

    # Lista por qualidade
    for q in "DCBA":
        nivel_rows = [r for r in rows if r["qualidade"] == q]
        if not nivel_rows:
            continue
        nivel_desc = {
            "A": "Excelente — leitura aprofundada, autoria completa, 12 seções, PDF",
            "B": "Boa — autoria OK, seções essenciais",
            "C": "Básica — alguma deficiência (autoria/conteúdo raso)",
            "D": "Raso — promovida em batch mas sem leitura aprofundada (ou OVERVIEW_ONLY)",
        }
        out.append(f"\n## Nível {q} ({len(nivel_rows)}) — {nivel_desc[q]}\n\n")
        out.append("| Ficha | Status | Linhas | Seções | PDF | Flags |\n")
        out.append("|---|---|---|---|---|---|\n")
        for r in sorted(nivel_rows, key=lambda x: x["ficha"]):
            pdf_ic = "OK" if r["pdf"] else "—"
            out.append(f"| `{r['ficha']}` | {r['status']} | {r['linhas']} | {r['secoes']} | {pdf_ic} | {r['flags']} |\n")

    OUT_FILE.write_text("".join(out), encoding="utf-8")

    print(f"Total: {len(fichas)}")
    print(f"Por qualidade: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']}")
    print(f"Por status: {dict(status_counts)}")
    print(f"\nRelatorio: {OUT_FILE}")


if __name__ == "__main__":
    main()
