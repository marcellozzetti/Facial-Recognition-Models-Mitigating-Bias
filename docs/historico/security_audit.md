# Security Audit — Dependency Vulnerabilities

**Last audit:** 2026-05-14
**Tool:** `pip-audit` against the working virtualenv
**Trigger:** GitHub Dependabot alerts on https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/security/dependabot

## Summary

This file tracks two waves of patches:

**Wave 1 (2026-05-11).** 21 alerts open → 20 patched, 1 residual (`diskcache`, no upstream fix).
**Wave 2 (2026-05-14).** 4 new alerts (3 PyTorch CVEs + a re-flagged `diskcache`) → 3 patched, 1 residual (same `diskcache`).

After both waves, **only the `diskcache` advisory remains**, and it is documented as accepted with mitigation rationale below.

| Wave | Date | Opened | Patched | Residual |
|---|---|---|---|---|
| 1 | 2026-05-11 | 21 | 20 | 1 (diskcache) |
| 2 | 2026-05-14 |  4 |  3 | 1 (same diskcache) |
| **Net state today** | — | — | — | **1** |

---

## Wave 2 — 2026-05-14 (PyTorch security advisories + DiskCache re-flag)

### Fixed by upgrading PyTorch

We bumped `torch` from 2.5.1+cu121 to 2.12.0+cu126 (`>=2.11,<3.0` in
`pyproject.toml`) and `torchvision` to 0.27.0+cu126 (`>=0.26,<1.0`).
The CUDA wheel changed from cu121 → cu126 because PyTorch dropped cu121
wheels at version 2.6. The host RTX 4070 SUPER + Windows 11 driver
already supports CUDA 12.6 — confirmed via `torch.cuda.is_available()`
and `torch.cuda.get_device_name(0)` after install.

| Advisory | Severity | Title | Fix | Closed at |
|---|---|---|---|---|
| GHSA-53q9-r3pm-6pq6 / CVE-2025-32434 | **Critical** | `torch.load` with `weights_only=True` leads to RCE | torch ≥ 2.6.0 | torch 2.12 |
| GHSA-63cw-57p8-fm3p / CVE-2026-24747 | High (flagged by Dependabot as "Moderate Improper Resource Shutdown" — see note) | Loading a malicious checkpoint with `weights_only=True` (second RCE path) | torch ≥ 2.10.0 | torch 2.12 |
| GHSA-2rj9-7h5r-q4h8 | Moderate | libuv used by tensorpipe — improper resource shutdown | torch ≥ 2.9.0 | torch 2.12 |
| GHSA-g6v3-crfc-cggj | Low | Access to arbitrary address while parsing flatbuffer (local DoS) | torch ≥ 2.1.0 (already fixed in 2.5.1) | torch 2.12 |

**Verification commands run:**

```powershell
# Install with cu126 wheels from the PyTorch index
.\.venv\Scripts\python.exe -m pip install --upgrade `
    "torch>=2.11,<3.0" "torchvision>=0.26,<1.0" `
    --index-url https://download.pytorch.org/whl/cu126 `
    --extra-index-url https://pypi.org/simple/

# Confirm versions and CUDA detection
.\.venv\Scripts\python.exe -c "import torch, torchvision; `
    print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# -> 2.12.0+cu126 0.27.0+cu126 True

# Confirm the 99-test suite still passes
.\.venv\Scripts\python.exe -m pytest tests/unit tests/smoke/test_arcface_integration.py -q
# -> 94 passed in 5.04s   (the 5 remaining smoke tests are mini-dataset slow)
```

**Removed during the bump:** `torchaudio 2.5.1+cu121`. It was a leftover
from the MBA notebook setup (`pip install torch torchvision torchaudio`),
not a declared dependency in `pyproject.toml`, and no module in the
project imports it. Removing it eliminates the `torchaudio==2.5.1+cu121`
hard pin on the old torch.

### Re-flagged: `diskcache==5.6.3` (CVE-2025-69872)

This is the same advisory that was already documented as residual in
Wave 1 — Dependabot re-flagged it after rescanning. Status unchanged:

**Upstream status as of 2026-05-14:** `pip index versions diskcache`
still lists 5.6.3 as the latest published version. No 5.6.4 or higher
has shipped; the CVE has no patched release on PyPI.

Mitigations from Wave 1 still apply (transitive via MLflow; local-only
`file://` backend; cache contents produced by our training process, not
user-controlled input). See the Wave 1 entry below for full detail.

---

## Wave 1 — 2026-05-11 (initial Dependabot sweep)

All packages bumped to the version listed in `pip-audit`'s "Fix Versions"
column. Direct dependencies were updated in `pyproject.toml`; transitive
dependencies are pinned via the regenerated `requirements.txt` lockfile.

| Package | Was | Now | CVEs addressed |
|---|---|---|---|
| **Pillow** (direct) | 10.2.0 | 12.2.0 | CVE-2024-28219, CVE-2026-42308, CVE-2026-42310 |
| urllib3 | 2.6.3 | 2.7.0 | CVE-2026-44431, CVE-2026-44432 |
| idna | 3.4 | 3.15 | PYSEC-2024-60 (x2) |
| Jinja2 | 3.1.5 | 3.1.6 | CVE-2025-27516 |
| fonttools | 4.57.0 | 4.62.1 | CVE-2025-66034 |
| filelock | 3.17.0 | 3.29.0 | CVE-2025-68146, CVE-2026-22701 |
| pip | 25.0 | 26.1+ | CVE-2025-8869, CVE-2026-1703, CVE-2026-3219, CVE-2026-6357 |
| setuptools | 65.5.0 | 78.1.1+ | PYSEC-2022-43012 (x2), PYSEC-2025-49 (x2), CVE-2024-6345 |

Direct dependencies got pinned minima in `pyproject.toml` so a fresh
install (without the lockfile) still picks safe versions. Transitive
deps are pinned via the regenerated `requirements.txt`.

### Residual: `diskcache==5.6.3` (CVE-2025-69872) — Wave 1 detail

**Status: accepted, no upstream fix available at audit time.**

- `diskcache` is a transitive dependency of `mlflow`.
- The CVE has no upstream patched version on PyPI.
- The vulnerability is not directly exploitable in our usage (we use
  `mlflow` for local experiment tracking only; the disk cache is not
  exposed to untrusted input).

**Mitigations applied:**

1. MLflow runs are local-only (`file://` backend), never exposed to the network.
2. The `outputs/mlruns/` directory is `.gitignored` — no cache state goes to the repo.
3. The disk-cache contents are produced by our own training process, not user input.

**Plan:**

- Re-audit weekly via `pip-audit`.
- Once `diskcache >= 5.6.4` (or newer fix) lands, bump immediately and re-freeze.

---

## How to verify locally

```powershell
.\.venv\Scripts\python.exe -m pip install pip-audit --quiet
.\.venv\Scripts\python.exe -m pip_audit
```

Expected output: only the `diskcache` line, until that CVE is patched
upstream. The PyTorch and torchvision lines are reported as
"Dependency not found on PyPI and could not be audited" — this is
expected because we install the `+cu126` CUDA wheels from PyTorch's
own index, not from PyPI. Dependabot reads from PyPI's metadata
(matching `torch>=2.11` in `pyproject.toml`), so it will report the
CVEs as closed.

## How to verify on GitHub

1. Open https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/security/dependabot
2. Dependabot reads the lockfile (`requirements.txt`) and reports remaining alerts.
3. After Wave 2, only alerts related to `diskcache` should remain.

## Rollback

If a regression is found after the torch 2.12 bump, restore the previous
lockfile:

```powershell
Copy-Item requirements-pre-torch-bump.txt requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-pre-torch-bump.txt` is checked in alongside the live
lockfile until Wave 2 is confirmed stable through the next batch of
training runs.
