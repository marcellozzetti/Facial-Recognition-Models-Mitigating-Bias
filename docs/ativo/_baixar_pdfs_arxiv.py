"""Baixa PDFs do arXiv para fichas sem PDF presente.

Lê o inventário gerado por _inventariar_pdfs.py e baixa cada PDF
em pdfs/{ficha_stem}.pdf, respeitando rate limit do arXiv.

Uso:
    python _baixar_pdfs_arxiv.py
"""

from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "04_pesquisa_bibliografica"
PDFS_DIR = CORPUS_DIR / "pdfs"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*?)$", re.MULTILINE)

# arXiv pede que clientes se identifiquem
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thesis-corpus-builder/1.0; mailto:marcello.ozzetti@gmail.com)"
}


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


def list_existing_pdfs() -> set[str]:
    return {p.stem for p in PDFS_DIR.glob("*.pdf")}


def is_clean_arxiv_id(arxiv_id: str | None) -> bool:
    if not arxiv_id:
        return False
    if arxiv_id.lower() in ("null", "none", "a verificar", "a confirmar"):
        return False
    return bool(re.match(r"^\d{4}\.\d{4,6}(v\d+)?$", arxiv_id))


def has_pdf_for_ficha(stem: str, existing: set[str]) -> bool:
    """Verifica se existe um PDF cujo nome começa com o primeiro segmento
    do stem da ficha (heurística usada no inventário)."""
    first = stem.split("_")[0]
    return any(c.startswith(first) for c in existing)


def download(url: str, dst: Path) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            return False, f"resposta nao e PDF ({len(data)} bytes)"
        dst.write_bytes(data)
        return True, f"{len(data) // 1024} KB"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL erro: {e}"
    except Exception as e:
        return False, f"erro: {e}"


def main() -> None:
    fichas = list_fichas()
    existing = list_existing_pdfs()

    targets = []
    for f in fichas:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        arxiv = fm.get("arxiv_id")
        if not is_clean_arxiv_id(arxiv):
            continue
        if has_pdf_for_ficha(f.stem, existing):
            continue
        targets.append((f.stem, arxiv))

    print(f"Total a baixar: {len(targets)}\n")

    ok, fail = [], []
    for i, (stem, arxiv) in enumerate(targets, 1):
        dst = PDFS_DIR / f"{stem}.pdf"
        url = f"https://arxiv.org/pdf/{arxiv}.pdf"
        print(f"[{i:3d}/{len(targets)}] {stem} ({arxiv}) ", end="", flush=True)
        success, info = download(url, dst)
        if success:
            print(f"OK {info}")
            ok.append((stem, arxiv, info))
        else:
            print(f"FAIL {info}")
            fail.append((stem, arxiv, info))
        # rate limit suave (arXiv aceita ~1 req/3s para clientes bem comportados)
        time.sleep(3.5)

    print(f"\n=== Resultado ===")
    print(f"Sucesso: {len(ok)}")
    print(f"Falha:   {len(fail)}")
    if fail:
        print("\nFalhas:")
        for stem, arxiv, info in fail:
            print(f"  - {stem} ({arxiv}): {info}")


if __name__ == "__main__":
    main()
