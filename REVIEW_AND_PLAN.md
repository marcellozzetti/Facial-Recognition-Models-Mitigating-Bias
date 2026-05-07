# Code Review and Refactor Plan

**Repository:** `Facial-Recognition-Models-Mitigating-Bias`
**Review author:** consolidated for the MBA → master's-thesis evolution
**Date:** 6 May 2026
**Stack to preserve:** Python 3.9, PyTorch 2.4.1, MTCNN (facenet-pytorch), OpenCV, pandas, scikit-learn, boto3, PyYAML

> **Status (2026-05-07):** all four sprints (A/B/C/D) below have landed — the
> repository now matches the proposed structure, the bug list in §2 is fully
> resolved, and the smoke run in [docs/smoke_results.md](docs/smoke_results.md)
> validated all 11 MBA experiments end-to-end. This document is preserved as
> the historical baseline that justifies each architectural decision.

---

## Table of contents

1. [Current state of the project](#1-current-state-of-the-project)
2. [Critical findings and bugs](#2-critical-findings-and-bugs)
3. [Best-practices assessment](#3-best-practices-assessment)
4. [Proposed structure](#4-proposed-structure)
5. [Phased refactor plan](#5-phased-refactor-plan)
6. [Per-stage test pipeline](#6-per-stage-test-pipeline)
7. [Practical next steps](#7-practical-next-steps)

---

## 1. Current state of the project

### 1.1 File map (excluding `.venv`, `.git`, `old/`, `__pycache__/`)

| File | Lines | State |
|---|---:|---|
| `configs/default.yaml` | 48 | **Functional, but with credential leak** |
| `data/bucket_dataset.py` | 106 | Functional |
| `data/face_dataset.py` | 109 | **Functional with bugs** |
| `models/arc_layers.py` | 47 | Functional (correct) |
| `models/base_model.py` | 7 | Skeleton, **unused** |
| `models/losses.py` | 22 | **Incorrect implementation** |
| `models/resnet_model.py` | 43 | Functional |
| `pipelines/pre_processing_pipeline.py` | 112 | Functional |
| `pipelines/training_pipeline.py` | 0 | **Empty** |
| `pipelines/evaluation_pipeline.py` | 0 | **Empty** |
| `preprocessing/pre_processing_images.py` | 150 | Functional |
| `tests/test_data_loading.py` | 0 | **Empty** |
| `tests/test_data_rotation.py` | 0 | **Empty** |
| `tests/test_draw_bounding.py` | 36 | **Broken** (incompatible signature) |
| `tests/test_models.py` | 0 | **Empty** |
| `utils/config.py` | 9 | Functional |
| `utils/custom_logging.py` | 19 | Functional |
| `utils/metrics.py` | 0 | **Empty** |
| `utils/versions.py` | 72 | Functional |
| `notebooks/MBA_IA_USP_marcello_ozzetti.ipynb` | — | MBA legacy |
| `pyproject.toml` / `requirements.txt` | — | **Missing** |
| `README.md` | 2 | Placeholder |
| `ROADMAP.md` | 10 | Placeholder |

### 1.2 One-sentence diagnosis

The project has **a functional preprocessing pipeline plus model skeletons**, but **the training, evaluation and test stages are empty**, with **silent bugs in the data loader** and **a critical defect in the ArcFace implementation**.

---

## 2. Critical findings and bugs

> **Sprint B status (2026-05-06):**
> §2.1 (credential), §2.3-§2.6 (dataset), §2.7 (sys.path), §2.11 (pyproject), §2.12 (crop), §2.14 (alignment) — ✅ fixed.
> §2.2 (ArcFace), §2.10 (reproducibility), §2.13 (logging) — ⏳ in progress.

### 2.1 🔴 Critical — AWS credential exposed in the repository

**File:** [configs/default.yaml:40](configs/default.yaml#L40)

```yaml
bucket_download_url: 'https://...&X-Amz-Credential=ASIAS74TL4TIRAUMFFFM%2F20250202...'
```

The S3 presigned URL contains an `AKID` (`ASIAS74TL4TIRAUMFFFM`), the signature, and is committed to Git. Presigned URLs do expire (12h here), but:

- The temporary AKID is exposed.
- If the key is long-lived, this is a permanent leak.
- Wrong pattern for a master's project (regulatory auditing).

**Action:** rotate the credential; move the URL into an env var or `configs/credentials.json` (already in `.gitignore`); rewrite history if the credential was non-rotating.

### 2.2 🔴 Critical — `ArcFaceLoss` does not implement ArcFace

**File:** [models/losses.py:6-22](models/losses.py#L6-L22)

```python
class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        ...
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        ...

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)  # ← plain cross-entropy!
```

The constructor computes the angular-margin constants, but `forward` ignores everything and returns `cross_entropy`. **Result: every "ArcFaceLoss experiment" in the MBA actually ran plain cross-entropy.**

This explains why ArcFaceLoss matched or even slightly underperformed CrossEntropyLoss in the MBA. The angular logic *does* live in [models/arc_layers.py](models/arc_layers.py) (`ArcMarginProduct`), but it **is never wired into the model** ([models/resnet_model.py](models/resnet_model.py) uses `nn.Linear` directly).

**Action:** discuss explicitly at the master's qualification. Report it as a finding from the review process. Then wire `ArcMarginProduct` into the model and re-run the MBA's key experiments — that single experiment alone is a valid contribution.

### 2.3 🟠 High — Normalization bug: `image_std` equals `image_mean`

**File:** [data/face_dataset.py:56,61,66](data/face_dataset.py#L56)

```python
transforms.Normalize(config['image']['image_mean'], config['image']['image_mean'])
#                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                   should be image_std
```

Every image was normalised with `mean=std=[0.485, 0.456, 0.406]` instead of using the `std=[0.229, 0.224, 0.225]` defined in the YAML. This directly affects the experimental results because the input to the pretrained network (ImageNet) was outside the expected scale.

### 2.4 🟠 High — Inconsistent YAML key: `image_size` vs. `input_size`

**File:** [data/face_dataset.py:53,59,64](data/face_dataset.py#L53)

```python
transforms.Resize((config['image']['input_size'], config['image']['input_size']))
```

[configs/default.yaml:18](configs/default.yaml#L18) defines `image_size`, not `input_size`. This code throws `KeyError` at runtime.

### 2.5 🟠 High — `class_to_idx` does not exist on `FaceDataset`

**File:** [data/face_dataset.py:100](data/face_dataset.py#L100)

```python
return dataloaders, datasets['train'].class_to_idx
```

`FaceDataset` (lines 15–43) does not define `class_to_idx`. `setup_dataset` therefore fails at runtime. It should return `label_encoder.classes_` or similar.

### 2.6 🟠 High — `setup_dataset` does not return `num_classes` or `label_encoder`

Without these, the training pipeline cannot:

- Know the number of classes to build the model.
- Decode predictions back to race names for the confusion matrix.

### 2.7 🟡 Medium — `sys.path.append` everywhere

```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

Python anti-pattern. Breaks when the project is installed, imported by tests or packaged. Solution: turn the project into an installable package via `pyproject.toml`.

### 2.8 🟡 Medium — `BaseModel` defined but unused

[models/base_model.py](models/base_model.py) declares `BaseModel(nn.Module)`, but `LResNet50E_IR` inherits directly from `nn.Module`. Either use it or remove it.

### 2.9 🟡 Medium — `tests/test_draw_bounding.py` has an incompatible signature

[tests/test_draw_bounding.py:24](tests/test_draw_bounding.py#L24) calls `draw_bounding_procedure(image, bbox)` with 2 args. The actual function ([preprocessing/pre_processing_images.py:53](preprocessing/pre_processing_images.py#L53)) requires 3: `(img, bbox, landmark)`. The test fails. Also it isn't really a test — it is a script with a `__main__`.

### 2.10 🟡 Medium — No reproducibility

- No `seed_everything()` for PyTorch + numpy + Python.
- No `torch.backends.cudnn.deterministic = True`.
- No dataset versioning (DVC or checksum hashes).
- No per-experiment hyperparameter tracking.

### 2.11 🟡 Medium — Missing `requirements.txt` / `pyproject.toml`

There is no way to reproduce the environment. Versions are only documented in Cap. 4 of the dissertation's `.tex`.

### 2.12 🟢 Low — `cropping_procedure` may overflow into negative indices

[preprocessing/pre_processing_images.py:96](preprocessing/pre_processing_images.py#L96):

```python
return img[y1-border:y1+height+border, x1-border:x1+width+border]
```

If `y1 - border < 0`, NumPy accepts the negative index and returns the wrong slice (from the end of the image). The commented-out sibling function (lines 98–125) already has the correct `max(0, ...)` clamping — it is commented out but is the right version.

### 2.13 🟢 Low — Unstructured logging

Each module writes to its own file (`bucket.log`, `preprocessing.log`, etc.) with no correlation timestamp or experiment ID. Hard to debug long-running runs.

### 2.14 🔴 Critical — Face alignment used absolute landmarks against the cropped image

**Discovered during Sprint B (2026-05-06).**

**File (pre-fix):** `preprocessing/pre_processing_images.py` — sequence `detect_and_adjust_faces` → `cropping_procedure` → `alignment_procedure`.

**Defective flow:**

1. MTCNN detects `bbox` and `landmarks` in **absolute** coordinates (pixels in the original image).
2. `cropping_procedure` crops the face region (with a border).
3. `alignment_procedure` was given the **cropped** image plus the **absolute** landmarks and computed:
   ```python
   eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
   M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
   cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
   ```

Because `eyes_center` was in original-image coordinates, the rotation pivot fell **outside** the cropped image (especially when the face was far from the top-left of the original frame). `warpAffine` then rotates around that external point, producing a severely shifted/cut output.

**Why the MBA missed it:** the rotation angle `arctan2(Δy, Δx)` is invariant to translation, so the angle being applied was correct. Only the **centre** was wrong. For mostly-frontal faces with small rotation (the bulk of FairFace) the visual effect is subtle, but the face is shifted enough to inject noise into training.

**Impact on the MBA narrative:**

- Every image across the 11 experiments was aligned with the wrong pivot.
- Combined with bug §2.3 (wrong normalisation), this explains part of the ~0.68 accuracy ceiling even with carefully tuned setups.
- This is a **finding to report** at the master's qualification: re-running the MBA's key experiments with the fixed pipeline is a concrete experimental contribution.

**Fix (user's Sprint A commit):** translate the landmarks into the cropped-image coordinate system before alignment:

```python
crop_x1 = max(0, x1 - border)
crop_y1 = max(0, y1 - border)
adjusted_landmarks = {
    "left_eye":  (landmark[0][0] - crop_x1, landmark[0][1] - crop_y1),
    "right_eye": (landmark[1][0] - crop_x1, landmark[1][1] - crop_y1),
}
aligned_face = alignment_procedure(cropped_face, adjusted_landmarks)
```

In addition, `alignment_procedure` now explicitly returns `None` when landmarks are missing (it previously logged a warning and silently continued).

**Open follow-up:** document the finding in Cap. 4 of the dissertation as "Limitations of the MBA execution identified during review" and in the master's introduction as a motivator for the "preprocessing rigour" axis.

---

## 3. Best-practices assessment

### 3.1 Scientific-Python checklist

| Category | Item | State |
|---|---|---|
| **Packaging** | `pyproject.toml` with pinned dependencies | ❌ |
| **Packaging** | Installable via `pip install -e .` | ❌ |
| **Packaging** | No `sys.path` hacks | ❌ |
| **Reproducibility** | Deterministic seeds | ❌ |
| **Reproducibility** | Dataset versioning | ❌ |
| **Reproducibility** | Experiment tracking | ❌ |
| **Configuration** | One YAML per experiment | ⚠️ Partial (only one) |
| **Configuration** | Schema validation (Pydantic / `jsonschema`) | ❌ |
| **Configuration** | Secrets outside the repo | ⚠️ Partial (URL leaked) |
| **Structure** | Clean separation `data/`, `models/`, `pipelines/`, `utils/` | ✅ |
| **Structure** | `__init__.py` exporting public APIs | ❌ (empty) |
| **Quality** | Type hints | ❌ |
| **Quality** | Consistent docstrings | ⚠️ Partial |
| **Quality** | Linter configured (ruff / black / isort) | ❌ |
| **Quality** | Pre-commit hooks | ❌ |
| **Tests** | `pytest` with fixtures | ❌ (placeholders) |
| **Tests** | Minimum coverage | ❌ |
| **Tests** | End-to-end smoke tests | ❌ |
| **CI/CD** | GitHub Actions or similar | ❌ |
| **Documentation** | Useful README | ❌ (2 lines) |
| **Documentation** | CHANGELOG | ❌ |

### 3.2 Neural-network / training specifics

| Category | Item | State |
|---|---|---|
| **Model** | `model.train()` / `model.eval()` switched explicitly | ❌ (no code) |
| **Model** | `torch.no_grad()` for validation/test | ❌ |
| **Model** | Mixed precision (`torch.cuda.amp`) | ❌ |
| **Model** | Per-epoch checkpoint saving | ❌ |
| **Model** | Early stopping with patience | ❌ |
| **Model** | Resume from checkpoint | ❌ |
| **Data** | Real augmentation (more than `RandomHorizontalFlip`) | ⚠️ Minimal |
| **Data** | `pin_memory=True` | ✅ |
| **Data** | `num_workers > 0` | ✅ |
| **Data** | DataLoader seeded for reproducibility | ❌ |
| **Metrics** | Per-epoch metrics logged | ❌ |
| **Metrics** | Per-demographic-subgroup metrics | ❌ |
| **Metrics** | Fairness metrics (IR/FDR/CEI) | ❌ |
| **Tracking** | TensorBoard / MLflow / W&B | ❌ |
| **Hardware** | GPU availability check | ✅ (in `versions.py`) |
| **Hardware** | Move model/data to `device` | ❌ |

---

## 4. Proposed structure

### 4.1 Principles

1. **Preserve the MBA stack** (PyTorch + MTCNN + ResNet50 + ArcFace + S3 + YAML).
2. **Each stage testable in isolation** — mirrors the MBA "experiments" but as reproducible code.
3. **No `sys.path` hacks** — installable Python package.
4. **File-based configuration + CLI overrides** — one YAML per experiment.
5. **Native tracking** — embedded MLflow (lighter than W&B, no external login).

### 4.2 Proposed directory tree

```
Facial-Recognition-Models-Mitigating-Bias/
├── pyproject.toml                       # 🆕 Packaging and dependencies
├── README.md                            # 🔄 Rewrite
├── PROPOSTA_MESTRADO.md                 # ✅ Already created
├── REVIEW_AND_PLAN.md                   # ✅ This document
├── .gitignore                           # ✅ OK (keep)
├── .pre-commit-config.yaml              # 🆕 Automatic linters
│
├── configs/
│   ├── default.yaml                     # 🔄 Sanitise credentials
│   ├── experiments/                     # 🆕
│   │   ├── exp01_crossentropy_sgd.yaml  #     One YAML per experiment
│   │   ├── exp02_arcface_sgd.yaml
│   │   └── ...
│   └── credentials.json.example         # 🆕 Template (no secrets)
│
├── src/                                 # 🆕 Main installable package
│   └── face_bias/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py                # 🔄 Was utils/config.py
│       │   └── schema.py                # 🆕 Pydantic validation
│       ├── data/
│       │   ├── __init__.py
│       │   ├── bucket.py                # 🔄 Was data/bucket_dataset.py
│       │   ├── dataset.py               # 🔄 Was data/face_dataset.py (bug fixes)
│       │   └── transforms.py            # 🆕 Augmentation pipelines
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── detection.py             # 🔄 detect_and_adjust_faces, etc.
│       │   ├── alignment.py             # 🔄 alignment_procedure
│       │   └── visualization.py         # 🔄 draw_bounding_procedure
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── resnet.py                # 🔄 LResNet50E_IR
│       │   ├── arc_margin.py            # 🔄 ArcMarginProduct (unchanged)
│       │   └── losses.py                # 🔧 Fix ArcFaceLoss
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py               # 🆕 Training loop
│       │   ├── callbacks.py             # 🆕 Early-stopping, checkpoint
│       │   └── schedulers.py            # 🆕 OneCycleLR, CosineAnnealing
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py             # 🆕 Evaluation loop
│       │   ├── metrics.py               # 🆕 Acc, F1, log-loss + IR/FDR/CEI
│       │   └── reports.py               # 🆕 Confusion matrix, report
│       └── utils/
│           ├── __init__.py
│           ├── logging.py               # 🔄 Was custom_logging
│           ├── reproducibility.py       # 🆕 seed_everything()
│           └── system.py                # 🔄 Was versions.py
│
├── pipelines/                           # 🔄 Entry-point scripts (CLI)
│   ├── 01_download_dataset.py           # 🔄 Bucket wrapper
│   ├── 02_preprocess.py                 # 🔄 pre_processing_pipeline wrapper
│   ├── 03_train.py                      # 🆕 Orchestrated training
│   ├── 04_evaluate.py                   # 🆕 Evaluation
│   └── 05_fairness_audit.py             # 🆕 IR/FDR/CEI on saved runs
│
├── tests/                               # 🆕 Full pytest tree
│   ├── __init__.py
│   ├── conftest.py                      # 🆕 Shared fixtures
│   ├── unit/                            #     Fast, no GPU
│   │   ├── test_config.py
│   │   ├── test_dataset.py
│   │   ├── test_preprocessing.py
│   │   ├── test_models.py
│   │   ├── test_losses.py
│   │   └── test_metrics.py
│   ├── integration/                     #     Touch the filesystem
│   │   ├── test_bucket_download.py
│   │   ├── test_preprocessing_pipeline.py
│   │   └── test_dataset_pipeline.py
│   └── smoke/                           #     End-to-end with mini data
│       ├── test_train_one_step.py
│       ├── test_evaluate_one_batch.py
│       └── test_full_pipeline_mini.py
│
├── notebooks/
│   ├── MBA_IA_USP_marcello_ozzetti.ipynb   # ✅ Keep as historical reference
│   └── exploratory/                       # 🆕 Analysis notebooks
│
├── scripts/                             # 🆕 Utilities
│   ├── fix_arcface_bug_demo.py
│   └── compare_experiments.py
│
├── data/                                # ⚠️ Paths only, data via DVC
│   ├── raw/                             # 🆕 (gitignored)
│   ├── processed/                       # 🆕 (gitignored)
│   └── synthetic/                       # 🆕 (gitignored, future)
│
├── outputs/                             # 🆕 (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── mlruns/                          #     MLflow tracking
│
└── docs/                                # 🆕 (future)
    └── architecture.md
```

### 4.3 Proposed `pyproject.toml` (skeleton)

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "face-bias"
version = "0.2.0-mestrado"
description = "Facial recognition bias mitigation - MBA → master's"
requires-python = ">=3.9,<3.12"
authors = [{name = "Marcello Ozzetti"}]
dependencies = [
    "torch==2.4.1",
    "torchvision==0.19.1",
    "facenet-pytorch>=2.5.3",
    "opencv-python-headless>=4.8",
    "numpy>=1.23,<2.0",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "PyYAML>=6.0",
    "pydantic>=2.0",
    "boto3>=1.28",
    "tqdm>=4.65",
    "matplotlib>=3.7",
    "mlflow>=2.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1.0",
    "black>=23.0",
    "pre-commit>=3.5",
]

[project.scripts]
face-bias-download = "face_bias.cli:download"
face-bias-preprocess = "face_bias.cli:preprocess"
face-bias-train = "face_bias.cli:train"
face-bias-evaluate = "face_bias.cli:evaluate"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast unit tests",
    "integration: tests that touch files",
    "smoke: end-to-end tests with minimal data",
    "gpu: tests that require CUDA",
]
```

---

## 5. Phased refactor plan

Four ~1-week sprints, aligned with **Month 1** of the master's plan in [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md).

### Sprint A — Hygiene and packaging (week 1)

| # | Item | Priority |
|---|---|---|
| A1 | Create `pyproject.toml` and install package (`pip install -e .`) | 🔴 |
| A2 | Move code to `src/face_bias/` and remove every `sys.path.append` | 🔴 |
| A3 | Sanitise `configs/default.yaml` — strip the URL with credential | 🔴 |
| A4 | Rotate the exposed AWS credential | 🔴 |
| A5 | Add `requirements.txt` generated from the current env (lock) | 🟠 |
| A6 | Configure `ruff` + `black` + pre-commit | 🟠 |
| A7 | Rewrite `README.md` with setup instructions | 🟠 |

### Sprint B — Bug fixes and reproducibility (week 2)

| # | Item | Priority |
|---|---|---|
| B1 | Fix `image_std` in `face_dataset.py` | 🔴 |
| B2 | Fix `image_size` (not `input_size`) in `face_dataset.py` | 🔴 |
| B3 | Fix `setup_dataset` return signature (num_classes + label_encoder) | 🔴 |
| B4 | **Fix `ArcFaceLoss`** — wire `ArcMarginProduct` into the model | 🔴 |
| B5 | Implement `utils/reproducibility.py` with `seed_everything()` | 🟠 |
| B6 | Add Pydantic validation for the YAML | 🟠 |
| B7 | Move logging to a structured format with run_id | 🟢 |

### Sprint C — Training and evaluation (week 3)

| # | Item | Priority |
|---|---|---|
| C1 | Implement `training/trainer.py` with the full loop | 🔴 |
| C2 | Implement `evaluation/evaluator.py` | 🔴 |
| C3 | Implement `evaluation/metrics.py` with Acc, F1, log-loss + **fairness (IR, FDR, CEI)** | 🔴 |
| C4 | Implement callbacks: ModelCheckpoint, EarlyStopping | 🟠 |
| C5 | Integrate MLflow tracking | 🟠 |
| C6 | Implement schedulers (OneCycleLR, CosineAnnealingWarmRestarts) | 🟠 |
| C7 | CLI pipeline `pipelines/03_train.py` with `--config` arg | 🟠 |

### Sprint D — Tests and CI (week 4)

| # | Item | Priority |
|---|---|---|
| D1 | Implement fixtures in `tests/conftest.py` | 🔴 |
| D2 | Implement unit tests (see §6) | 🔴 |
| D3 | Implement integration tests (see §6) | 🟠 |
| D4 | Implement smoke tests (see §6) | 🟠 |
| D5 | Configure GitHub Actions: lint + tests | 🟠 |
| D6 | Add coverage badge to the README | 🟢 |

---

## 6. Per-stage test pipeline

Mirrors the **experimental stages of the MBA** ([Cap. 4 of the dissertation](USPSC-Cap4-Avaliacao_Experimental.tex)):

> 1. Selection and analysis of the dataset
> 2. Preprocessing (Experiment I rotation, Experiment II MTCNN)
> 3. Model training (11 experiments)
> 4. Performance analysis

Each stage gets **three test levels**: unit (fast), integration (touches files), smoke (end-to-end with mini data).

### 6.1 Stage 0 — Configuration and environment

**Goal:** make sure the project loads config, validates the schema and detects the GPU.

| Test | Type | What it validates |
|---|---|---|
| `test_config_loads_default_yaml` | unit | YAML loads without error |
| `test_config_schema_valid` | unit | Pydantic accepts the expected fields |
| `test_config_rejects_missing_fields` | unit | Clear error if a required field is missing |
| `test_seed_everything_deterministic` | unit | Two calls produce the same random numbers |
| `test_system_report_includes_torch` | unit | `system_info_report()` returns the torch version |

### 6.2 Stage 1 — Data acquisition (`bucket`)

**Goal:** S3 download/upload work without leaking credentials.

| Test | Type | What it validates |
|---|---|---|
| `test_credentials_loaded_from_env` | unit | Function accepts a credential via `os.environ` |
| `test_credentials_not_in_yaml` | unit | YAML does not contain `X-Amz-Credential` |
| `test_bucket_client_initialized` | integration | `boto3.client('s3', ...)` returns a valid object (mock) |
| `test_download_zip_extracts` | integration | Mock S3 with `moto` → download and unzip |
| `test_download_handles_404` | integration | Invalid URL → log error, no crash |

### 6.3 Stage 2 — Preprocessing (MTCNN + alignment)

**Goal:** correct detection, cropping and alignment. **Direct mirror of the MBA's Experiments I and II.**

| Test | Type | What it validates |
|---|---|---|
| `test_detect_faces_returns_boxes` | unit | MTCNN on a dummy image returns a list (not None) |
| `test_alignment_procedure_rotates_correctly` | unit | Eyes are horizontally aligned after `alignment_procedure` |
| `test_cropping_handles_negative_indices` | unit | Bbox near the edge → no wrong slice (regression for §2.12) |
| `test_rotate_procedure_45deg` | unit | A 45° rotated image keeps the same shape and is non-empty |
| `test_resize_to_224x224` | unit | Output has shape (224, 224, 3) |
| `test_pipeline_processes_directory` | integration | Pipeline on `tests/fixtures/sample_images/` produces the expected files |
| `test_mtcnn_detection_rate_baseline` | integration | **Reproduces MBA Experiment I:** detects ≥60% of the faces in rotated fixtures |
| `test_no_quality_loss_after_pipeline` | integration | **Reproduces MBA Experiment II:** MTCNN re-detects faces after the pipeline |

### 6.4 Stage 3 — PyTorch dataset

**Goal:** `FaceDataset` loads correctly, normalises and encodes labels.

| Test | Type | What it validates |
|---|---|---|
| `test_dataset_len_matches_csv` | unit | `len(dataset) == len(csv)` |
| `test_dataset_returns_tensor_and_label` | unit | `__getitem__` returns `(Tensor, int)` |
| `test_dataset_normalization_uses_imagenet_stats` | unit | Tensor mean ≈ 0, std ≈ 1 (regression for §2.3) |
| `test_dataset_image_shape_3x224x224` | unit | Tensor has shape `[3, 224, 224]` |
| `test_label_encoder_round_trip` | unit | `encode → decode == original` |
| `test_train_val_test_split_stratified` | integration | Race distribution preserved across the 3 splits (chi² test) |
| `test_class_balance_after_undersampling` | integration | **Reproduces the `tab:raceCountBalanceado` MBA table** |
| `test_dataloader_iterates_one_batch` | smoke | `next(iter(dataloader))` works |

### 6.5 Stage 4 — Models and loss

**Goal:** the model builds, the forward pass runs, ArcFace actually applies the margin (bug §2.2).

| Test | Type | What it validates |
|---|---|---|
| `test_resnet50_constructs_with_7_classes` | unit | The model is instantiable |
| `test_resnet50_forward_returns_correct_shape` | unit | Input `[B,3,224,224]` → output `[B, 7]` |
| `test_resnet50_pretrained_weights_loaded` | unit | Conv layers have non-random weights |
| `test_resnet50_dropout_is_active_in_train_mode` | unit | `model.train()` enables dropout, `model.eval()` disables |
| `test_arcmarginproduct_forward_shape` | unit | `ArcMarginProduct` returns logits of the expected shape |
| `test_arcfaceloss_differs_from_crossentropy` | unit | **Regression for §2.2**: loss with margin > 0 differs from cross-entropy |
| `test_loss_gradient_flows_to_input` | unit | `loss.backward()` produces non-zero gradients |
| `test_arcface_decreases_intra_class_distance` | integration | Sanity: same-class pairs → embeddings closer after one update |

### 6.6 Stage 5 — Training loop

**Goal:** train one batch, one mini epoch, validate persistence.

| Test | Type | What it validates |
|---|---|---|
| `test_trainer_one_step_decreases_loss` | smoke | Loss at t1 < loss at t0 (high lr, one batch repeated) |
| `test_trainer_saves_checkpoint` | smoke | After 1 epoch the `.pt` file exists |
| `test_trainer_resumes_from_checkpoint` | smoke | Loading the checkpoint reproduces the weights |
| `test_early_stopping_halts_training` | unit | After N epochs without improvement, training stops |
| `test_scheduler_onecyclelr_warmup` | unit | LR rises in the first steps |
| `test_mlflow_logs_metrics` | smoke | After 1 epoch `mlruns/` contains a run with `train_loss` |
| `test_full_pipeline_mini_dataset` | smoke | **Reproduces MBA Experiment 1 at mini scale** (10 images, 1 epoch) |

### 6.7 Stage 6 — Evaluation and metrics

**Goal:** confusion matrix, classification report, **fairness metrics** (the MBA's gap).

| Test | Type | What it validates |
|---|---|---|
| `test_confusion_matrix_shape_7x7` | unit | For 7 classes the matrix is 7×7 |
| `test_classification_report_keys` | unit | Returns precision, recall, F1 per class |
| `test_log_loss_is_positive` | unit | Always > 0 |
| `test_inequity_rate_known_input` | unit | For FMR `[0.01, 0.05]` → IR = 5.0 |
| `test_fdr_known_input` | unit | Classic case from Pereira & Marcel |
| `test_garbe_zero_when_perfectly_balanced` | unit | Equal groups → GARBE = 0 |
| `test_cei_combines_disparities` | unit | Validation against a reference implementation |
| `test_full_evaluation_on_fixture_predictions` | integration | Pipeline reads a CSV of predictions and produces the full report |
| `test_audit_recomputes_mba_baseline` | smoke | **Recomputes fairness over the MBA matrices — Month-1 deliverable** |

### 6.8 Coverage summary by stage

| Stage | Unit | Integration | Smoke | Total |
|---|---:|---:|---:|---:|
| 0 — Config | 5 | 0 | 0 | 5 |
| 1 — Bucket | 2 | 3 | 0 | 5 |
| 2 — Preprocessing | 5 | 3 | 0 | 8 |
| 3 — Dataset | 5 | 2 | 1 | 8 |
| 4 — Models | 7 | 1 | 0 | 8 |
| 5 — Training | 2 | 0 | 5 | 7 |
| 6 — Evaluation | 7 | 1 | 1 | 9 |
| **Total** | **33** | **10** | **7** | **50** |

50 tests is an aggressive but achievable target for Month 1 — four sprints of ~12 tests each.

---

## 7. Practical next steps

### 7.1 Open decisions (need approval before implementation)

1. **Rewrite Git history to remove the credential?** If the AWS credential is long-lived, the old URL is still reachable in history even after `git rm`. Two options:
   - **Option A** — rotate the credential and move on; history keeps an invalid leak (preferred if the credential was a temporary `ASIA*` that already expired).
   - **Option B** — `git filter-repo` to rewrite history (more aggressive, requires force-push).

2. **Create a separate refactor branch?** Recommended `feat/mestrado-refactor` to preserve `main` while the refactor lands.

3. **Migrate to `src/face_bias/` now or later?** Migrating now produces a large commit but avoids rework. If you prefer incremental, the flat layout can stay.

4. **Tracking — MLflow or TensorBoard?** MLflow is more robust (compares experiments, stores artefacts), TensorBoard is lighter.

5. **DVC for dataset versioning?** Recommended for the master's but adds complexity. Alternative: SHA256 + JSON manifest.

### 7.2 Suggested execution order

Immediate (< 1 day):

- A3 + A4 — sanitise the credential and rotate.
- B1 + B2 + B3 — fix the silent dataset bugs (impact on the MBA).
- B4 — fix ArcFaceLoss (a finding with master's-narrative impact).

Sprint 1 (current week):

- A1 + A2 — pyproject.toml + reorganise to `src/`.
- B5 — `seed_everything()`.
- D1 — test fixtures.

From there, follow the §5 plan.

---
