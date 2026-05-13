# Security Audit — Dependency Vulnerabilities

**Date:** 2026-05-11
**Tool:** `pip-audit` against the working virtualenv
**Trigger:** GitHub Dependabot alerts on https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/security/dependabot

## Summary

The repository's dependency tree had **21 known vulnerabilities across 9 packages** prior to this audit. After upgrading to security-patched versions, only **1 residual vulnerability** remains (no fix available upstream yet).

| Status | Count |
|---|---:|
| Vulnerabilities fixed | 20 |
| Vulnerabilities accepted (no upstream fix) | 1 |

## Fixed in this commit

All packages bumped to the version listed in `pip-audit`'s "Fix Versions" column. Direct dependencies were updated in `pyproject.toml`; transitive dependencies are pinned via the regenerated `requirements.txt` lockfile.

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

Direct dependencies got pinned minima in `pyproject.toml` so a fresh install (without the lockfile) still picks safe versions. Transitive deps are pinned via the regenerated `requirements.txt`.

## Residual: `diskcache==5.6.3` (CVE-2025-69872)

**Status: accepted, no upstream fix available at audit time.**

- `diskcache` is a transitive dependency of `mlflow`.
- The CVE has no upstream patched version on PyPI as of 2026-05-11.
- The vulnerability is not directly exploitable in our usage (we use `mlflow` for local experiment tracking only; the disk cache is not exposed to untrusted input).

**Mitigations applied:**
1. MLflow runs are local-only (`file://` backend), never exposed to the network.
2. The `outputs/mlruns/` directory is `.gitignored` — no cache state goes to the repo.
3. The disk-cache contents are produced by our own training process, not user input.

**Plan:**
- Re-audit weekly via `pip-audit`.
- Once `diskcache >= 5.6.4` (or newer fix) lands, bump immediately and re-freeze.

## How to verify locally

```powershell
.\.venv\Scripts\python.exe -m pip install pip-audit --quiet
.\.venv\Scripts\python.exe -m pip_audit
```

Expected: only the `diskcache` line, until that CVE is patched upstream.

## How to verify on GitHub

1. Open https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/security/dependabot
2. Dependabot reads the lockfile (`requirements.txt`) and reports remaining alerts.
3. After this commit, only alerts related to `diskcache` should remain.
