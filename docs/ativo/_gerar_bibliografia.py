"""Gera bibliografia BibTeX consolidada a partir das 101 fichas.

Le todas as fichas em 04_pesquisa_bibliografica/*.md, extrai metadados
do frontmatter YAML e produz docs/tese/referencias.bib pronto para o
LaTeX.

Heuristicas:
- Tipo da entrada: @article (journal), @inproceedings (conference),
  @misc (preprint/dataset/outros).
- Chave da entrada: lastnameYYYYkeyword (e.g., perez2018film).
- Author Bib format: 'Sobrenome, Nome and ...'.
- Suporta arxiv_id e DOI no campo opcional.

Uso:
    python _gerar_bibliografia.py
    -> escreve docs/tese/referencias.bib
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from collections import OrderedDict

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "04_pesquisa_bibliografica"
TESE_DIR = HERE.parent / "tese"
OUT_FILE = TESE_DIR / "referencias.bib"

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
    return [f for f in files if not f.name.startswith("_") and f.name not in {"README.md", "INDEX.md"}]


def slugify(s: str) -> str:
    """Strip acentos e nao-alfanumericos para chaves bibtex."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9]+", "", s)
    return s.lower()


def parse_authors(raw: str | None) -> list[str]:
    """Le campo de autores e devolve lista de 'Sobrenome, Nome' para
    BibTeX. Trata bem casos onde a string original tem comentarios
    entre parenteses (afiliacoes)."""
    if not raw:
        return []
    # remove lista YAML [...]
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    # split por virgula, mas respeitando parenteses (afiliacoes)
    items = []
    depth = 0
    cur = ""
    for ch in raw:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            if cur.strip():
                items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())

    out = []
    for item in items:
        # remove (afiliacao) e anotacoes
        clean = re.sub(r"\s*\(.*?\)", "", item).strip()
        # remove anotacoes como '— primeiro autor'
        clean = re.split(r"\s+[—–-]\s+", clean)[0].strip()
        if not clean or clean.lower() in ("a verificar", "et al.", "et al"):
            continue
        # split nome em 'Nome Sobrenome' -> 'Sobrenome, Nome'
        parts = clean.split()
        if len(parts) == 1:
            out.append(parts[0])
            continue
        # se ja tiver virgula, mantem
        if "," in clean:
            out.append(clean)
            continue
        last = parts[-1]
        first = " ".join(parts[:-1])
        out.append(f"{last}, {first}")
    return out


def make_key(fm: dict, stem: str) -> str:
    authors = parse_authors(fm.get("autores"))
    year = fm.get("ano") or ""
    if authors:
        first = authors[0].split(",")[0]  # sobrenome
        key_base = slugify(first)
    else:
        key_base = slugify(stem.split("_")[0])
    # adicionar slug do titulo (primeira palavra significativa)
    title = fm.get("titulo") or stem
    skipwords = {"the", "a", "an", "on", "of", "and", "for", "in", "to", "from"}
    title_words = [w for w in re.split(r"\W+", title) if w and w.lower() not in skipwords]
    if title_words:
        keyword = slugify(title_words[0])[:12]
    else:
        keyword = ""
    parts = [key_base, year, keyword]
    return "".join(p for p in parts if p)


def escape_bibtex(s: str) -> str:
    """Escapa caracteres especiais BibTeX em strings de campo."""
    if s is None:
        return ""
    # protege uppercase: envolve {N} em palavras com upper interno (ex.: 'CLIP', 'FaceNet')
    # implementacao simples: envolve { ao redor da string inteira do title para evitar lowercase
    return s.replace("\\", r"\\").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def entry_type(fm: dict) -> str:
    t = (fm.get("tipo_publicacao") or "").lower()
    if t in ("journal",):
        return "article"
    if t in ("conference", "workshop_conference"):
        return "inproceedings"
    if t in ("book_chapter",):
        return "incollection"
    return "misc"


def build_entry(fm: dict, stem: str) -> tuple[str, str]:
    """Retorna (chave, entrada bibtex completa)."""
    key = make_key(fm, stem)
    btype = entry_type(fm)
    authors = parse_authors(fm.get("autores"))
    author_str = " and ".join(authors) if authors else "Anonymous"
    title = (fm.get("titulo") or stem).strip()
    year = fm.get("ano") or ""
    venue = (fm.get("venue") or "").strip()
    doi = fm.get("doi")
    arxiv = fm.get("arxiv_id")
    url = fm.get("url_primario")

    fields = OrderedDict()
    fields["author"] = author_str
    fields["title"] = "{" + escape_bibtex(title) + "}"
    if year:
        fields["year"] = year

    # venue mapping
    if btype == "article":
        fields["journal"] = escape_bibtex(venue)
    elif btype == "inproceedings":
        fields["booktitle"] = escape_bibtex(venue)
    elif btype == "incollection":
        fields["booktitle"] = escape_bibtex(venue)
    else:
        if venue:
            fields["howpublished"] = "\\url{" + (url or "") + "}" if url else escape_bibtex(venue)
            fields["note"] = escape_bibtex(venue)

    if doi:
        fields["doi"] = doi
    if arxiv and arxiv.replace(".", "").replace("v", "").isdigit() == False:
        # arxiv pode ter formato 1234.56789v1
        pass
    if arxiv:
        # eprint Tex padrao
        fields["eprint"] = arxiv
        fields["archivePrefix"] = "arXiv"
    if url and "howpublished" not in fields:
        fields["url"] = url

    body = ",\n  ".join(f"{k} = {{{v}}}" if not (k in ("howpublished",) and v.startswith("\\url"))
                       else f"{k} = {v}"
                       for k, v in fields.items())
    entry = f"@{btype}{{{key},\n  {body}\n}}\n"
    return key, entry


def main() -> None:
    TESE_DIR.mkdir(exist_ok=True)
    fichas = list_fichas()
    entries: list[tuple[str, str, str]] = []  # (key, stem, entry)
    seen_keys: set[str] = set()
    skipped: list[str] = []

    for f in fichas:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if not fm.get("titulo"):
            skipped.append(f.stem)
            continue
        key, entry = build_entry(fm, f.stem)
        # se key duplicada, sufixa
        base_key = key
        suffix = ord("a")
        while key in seen_keys:
            key = base_key + chr(suffix)
            # regenerar entrada com key corrigida
            entry = re.sub(r"^@(\w+)\{[^,]+,", f"@\\1{{{key},", entry, count=1)
            suffix += 1
        seen_keys.add(key)
        entries.append((key, f.stem, entry))

    # ordenar por chave
    entries.sort(key=lambda x: x[0])

    # escrever .bib
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        fh.write("% Bibliografia consolidada a partir das 101 fichas\n")
        fh.write("% Gerada por docs/ativo/_gerar_bibliografia.py em 2026-06-15\n")
        fh.write(f"% Total de entradas: {len(entries)}\n\n")
        for key, stem, entry in entries:
            fh.write(f"% origem: {stem}.md\n")
            fh.write(entry)
            fh.write("\n")

    print(f"Total de fichas: {len(fichas)}")
    print(f"Entradas BibTeX geradas: {len(entries)}")
    print(f"Puladas (sem titulo): {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"  - {s}")
    print(f"\nArquivo: {OUT_FILE}")


if __name__ == "__main__":
    main()
