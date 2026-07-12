"""Baixa PDFs adicionais de fontes OpenAccess (CVPR/ICCV/OpenReview).

Para fichas sem arxiv_id mas com URL direta de PDF acessível.

Uso:
    python _baixar_pdfs_extras.py
"""

from __future__ import annotations

import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).parent
PDFS_DIR = HERE / "04_pesquisa_bibliografica" / "pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thesis-corpus-builder/1.0; mailto:marcello.ozzetti@gmail.com)"
}

# (ficha_stem, url_direta_pdf)
TARGETS = [
    (
        "bian_2025_lorafair",
        "https://openaccess.thecvf.com/content/ICCV2025/papers/Bian_LoRA-FAIR_Federated_LoRA_Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.pdf",
    ),
    (
        "zhao_2025_aimfair",
        "https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_AIM-Fair_Advancing_Algorithmic_Fairness_via_Selectively_Fine-Tuning_Biased_Models_with_CVPR_2025_paper.pdf",
    ),
    (
        "counterfactual_fairness_iclr2025",
        "https://openreview.net/pdf/ff6f04aebc48400e7d040f01b7a7306d9b00073b.pdf",
    ),
]


def download(url: str, dst: Path) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            return False, f"resposta nao e PDF ({len(data)} bytes, inicia com {data[:8]})"
        dst.write_bytes(data)
        return True, f"{len(data) // 1024} KB"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL erro: {e}"
    except Exception as e:
        return False, f"erro: {e}"


def main() -> None:
    ok, fail = [], []
    for i, (stem, url) in enumerate(TARGETS, 1):
        dst = PDFS_DIR / f"{stem}.pdf"
        print(f"[{i}/{len(TARGETS)}] {stem} ... ", end="", flush=True)
        if dst.exists():
            print("ja existe, pulando")
            continue
        success, info = download(url, dst)
        if success:
            print(f"OK {info}")
            ok.append((stem, info))
        else:
            print(f"FAIL {info}")
            fail.append((stem, info))
        time.sleep(3.0)

    print(f"\nSucesso: {len(ok)} / Falha: {len(fail)}")


if __name__ == "__main__":
    main()
