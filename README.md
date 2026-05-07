# face-bias

A facial-recognition toolkit focused on **bias mitigation and fairness auditing**.

`face-bias` is an installable Python package that goes from raw face images to
a trained classifier and a full demographic-fairness report, with every stage
exposed as a deterministic CLI. It implements the modern building blocks of a
face-recognition pipeline (MTCNN detection, similarity-transform alignment,
LResNet50E_IR + ArcFace, MLflow tracking) plus an opinionated set of
fairness metrics — Inequity Rate, max–min disparity, coefficient of variation
and Gini — applied per demographic group on top of the usual classification
report.

The default scenario in this repository is race classification on the
[FairFace](https://github.com/joojs/fairface) dataset across seven groups, but
the pipeline is dataset-agnostic: any CSV that maps an image path to a class
label works.

## Features

- 🧱 **Installable Python package** (`pip install -e .`) — no `sys.path`
  hacks, every stage importable as `face_bias.*`.
- 🎛️ **Config-driven experiments** — one YAML per run, validated by a Pydantic
  schema. Eleven ready-made experiment configs live under
  [configs/experiments/](configs/experiments/).
- 🔁 **Reproducible by default** — single `seed_everything()` seeds Python,
  NumPy, PyTorch (CPU + CUDA), `PYTHONHASHSEED` and toggles cuDNN
  deterministic mode.
- 🧠 **Modern training loop** — train/val cycle with optional
  `EarlyStopping`, gradient-norm clipping, OneCycleLR or
  CosineAnnealingWarmRestarts schedulers, MLflow tracking and best/last
  checkpoint persistence.
- ⚖️ **Fairness audit out of the box** — every evaluation produces per-class
  precision/recall/F1 plus aggregate disparity metrics
  (`fairness_audit.json`).
- 🔍 **Interpretability** — built-in t-SNE projection of feature embeddings
  and Grad-CAM heatmaps for qualitative inspection of model behaviour.
- 🪟 **Cross-platform tested** — Windows MAX_PATH and line-ending pitfalls
  handled (centralised `mlruns/`, `.gitattributes`).
- 🧪 **80+ tests** spanning unit, integration and smoke (full-pipeline)
  coverage.

## Quick start

```bash
# 1) Create a virtualenv and install the package with dev/dvc extras
python -m venv .venv
.venv/Scripts/Activate.ps1            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,dvc]"

# 2) Drop your S3 / Azure credentials into configs/credentials.json
cp configs/credentials.json.example configs/credentials.json
# edit configs/credentials.json (or export FACE_BIAS_BUCKET_* env vars)

# 3) Verify the environment (versions + GPU)
face-bias-version --config configs/default.yaml

# 4) Run an experiment end-to-end (training + evaluation + 5 reports)
face-bias-train    --config configs/experiments/exp01_ce_sgd_onecycle.yaml \
                   --device cuda
face-bias-evaluate --config configs/experiments/exp01_ce_sgd_onecycle.yaml \
                   --checkpoint outputs/<run_id>/checkpoints/best.pt \
                   --device cuda

# 5) Optional: t-SNE and Grad-CAM
face-bias-tsne     --config configs/experiments/exp01_ce_sgd_onecycle.yaml \
                   --checkpoint outputs/<run_id>/checkpoints/best.pt
face-bias-gradcam  --config configs/experiments/exp01_ce_sgd_onecycle.yaml \
                   --checkpoint outputs/<run_id>/checkpoints/best.pt \
                   --num-samples 8

# 6) Run the entire experiment matrix in one shot
python scripts/run_all_experiments.py --device cuda --output-dir outputs/run01
```

## CLI entry-points

| Command | Purpose |
|---|---|
| `face-bias-version` | System / library / GPU report |
| `face-bias-download` | Pull the dataset archive from S3 / Azure Blob and unpack it |
| `face-bias-preprocess` | MTCNN detection + similarity-transform alignment + resize |
| `face-bias-train` | Train a model, log to MLflow, save `best.pt` and `last.pt` |
| `face-bias-evaluate` | Score a checkpoint on a split and emit five report files |
| `face-bias-tsne` | 2-D t-SNE projection of feature embeddings, coloured by class |
| `face-bias-gradcam` | Grad-CAM heatmaps over a sample of images |

Every CLI accepts `--config` and `--device` (`auto | cpu | cuda | cuda:0`).
`face-bias-train` additionally supports `--epochs N` to override the
config-declared `num_epochs` for quick smoke runs.

## Installation

`face-bias` targets Python 3.9–3.11 and PyTorch ≥ 2.4.

```bash
pip install -e ".[dev,dvc]"
```

Optional extras:

- `dev` — `pytest`, `ruff`, `black`, `pre-commit`, `moto` (S3 mocks).
- `dvc` — Data Version Control with the S3 backend.

For a frozen environment that mirrors the development setup, use the lockfile:

```bash
pip install -r requirements.txt
```

After cloning, install the pre-commit hooks once:

```bash
pre-commit install
```

## How it works

```
                 ┌──────────────────────┐      ┌────────────────────┐
   raw faces ─▶  │   preprocessing/     │ ──▶  │      data/         │ ──▶  PyTorch
   on S3         │ MTCNN + alignment    │      │ FaceDataset +      │      DataLoader
                 │  (face_bias.cli.     │      │ undersample +      │
                 │   preprocess)        │      │ class_filter +     │
                 └──────────────────────┘      │ 80/10/10 split     │
                                               └────────────────────┘
                                                         │
                                                         ▼
   ┌──────────────────────────────┐    ┌──────────────────────────────┐
   │         models/              │    │         training/            │
   │ LResNet50E_IR (ResNet50      │ ◀──│ Trainer                       │
   │  + Linear or ArcMargin head) │    │  – build_optimizer/scheduler  │
   │ ArcMarginProduct             │    │  – grad_clip_norm             │
   │ ArcFaceLoss / CrossEntropy   │    │  – EarlyStopping(patience=5)  │
   └──────────────────────────────┘    │  – MLflow + best/last.pt      │
                                        └──────────────────────────────┘
                                                         │
                                                         ▼
                                        ┌──────────────────────────────┐
                                        │         evaluation/          │
                                        │  classification_metrics      │
                                        │  per_class_report            │
                                        │  fairness_audit              │
                                        │   (IR, max-min, CV, Gini)    │
                                        │  confusion_matrix            │
                                        └──────────────────────────────┘
                                                         │
                                                         ▼
                                        ┌──────────────────────────────┐
                                        │      interpretability/       │
                                        │  t-SNE on embeddings         │
                                        │  Grad-CAM heatmaps           │
                                        └──────────────────────────────┘
```

Every stage corresponds to a Python sub-package and at least one CLI entry
point. The trainer always passes `labels` to the model on the forward pass —
`LResNet50E_IR` decides at runtime whether the `arcface` head should consume
them (training) or fall back to scaled cosine (eval/inference).

## Project layout

```
src/face_bias/         # The installable package
├── config/            # YAML loader + Pydantic schema validation
├── data/              # Bucket I/O (S3 / Blob) and PyTorch Dataset
├── preprocessing/     # MTCNN detection, similarity-transform alignment, viz
├── models/            # LResNet50E_IR, ArcMarginProduct, ArcFaceLoss
├── training/          # Trainer, optimisers, schedulers, callbacks
├── evaluation/        # Predict + classification + fairness audit
├── interpretability/  # t-SNE + Grad-CAM
├── utils/             # logging, reproducibility, system report
└── cli/               # face-bias-{download,preprocess,train,evaluate,...}

configs/
├── default.yaml                       # Canonical reference config
├── credentials.json.example           # Template for bucket credentials
└── experiments/                       # One YAML per pre-defined experiment

pipelines/             # Thin wrappers for the most common CLIs
scripts/               # Helper scripts (config generator, plotting, batch run)
tests/                 # pytest suite — unit / integration / smoke
notebooks/             # Exploratory and reference notebooks
docs/                  # Long-form technical documentation
data/                  # raw/, processed/, synthetic/ — gitignored
outputs/               # checkpoints/, logs/, mlruns/ — gitignored
```

## Configuration

Configs are plain YAML. The Pydantic schema in `src/face_bias/config/schema.py`
catches typos before training starts:

```yaml
model:
  name: "LResNet50E_IR"
  pretrained: true
  num_classes: 7
  dropout: 0.2
  head: "linear"          # one of: linear, arcface
  arcface_s: 30.0
  arcface_m: 0.5

training:
  batch_size: 128
  learning_rate: 0.001
  num_epochs: 25
  optimizer: "sgd"        # one of: sgd, adamw
  scheduler: "onecyclelr" # one of: onecyclelr, cosineannealingwarmrestarts
  loss_function: "cross_entropy"
  test_size: 0.1          # final test fraction
  val_size: 0.1           # final val fraction (-> 80/10/10)
  random_state: 42
  grad_clip_norm: 5.0
  early_stopping_patience: 5

data:
  dataset_file: "data/raw/fairface/fairface_labels.csv"
  dataset_image_output_path: "data/processed/fairface_aligned"
  balance: "undersample"  # one of: none, undersample
  class_filter: null      # e.g. ["Black", "White"] for binary subsets
```

Eleven ready-made variants — combinations of CrossEntropy/ArcFace, SGD/AdamW,
OneCycleLR/CosineAnnealingWarmRestarts, dropout 0.2 / 0.5, full vs. binary
classes — live under `configs/experiments/` and are regenerated by
`scripts/generate_experiment_configs.py`.

## Outputs of a run

`face-bias-train` writes everything under `<output-dir>/<run_id>/`:

```
outputs/<run_id>/
├── checkpoints/
│   ├── best.pt          # lowest val_loss across all epochs
│   └── last.pt          # final epoch (or last epoch before EarlyStopping)
├── history.json         # per-epoch metrics
└── ../mlruns/           # MLflow tracking (shared at the parent level)
```

`face-bias-evaluate` adds, under `<eval-output-dir>/<run_id>/`:

```
metrics.json              # accuracy, F1 macro/weighted, log_loss
per_class.csv             # precision / recall / F1 / support per class
fairness_audit.json       # IR, max-min, std, CV, Gini for precision/recall/F1
confusion_matrix.csv      # labelled DataFrame
classification_report.txt # sklearn-style text report
```

`scripts/plot_experiment.py` turns those artefacts into four publication-ready
figures (`metricas.png`, `matriz.png`, `per_class.png`, `fairness.png`) in
both PNG and PDF.

## Data versioning

The `data/` tree is gitignored. Use [DVC](https://dvc.org/) to track datasets:

```bash
dvc init
dvc remote add -d storage s3://your-bucket/dvc-store

dvc add data/raw/fairface
git add data/raw/fairface.dvc .gitignore
git commit -m "data: track FairFace v1.25.x"

dvc push          # upload to the remote
dvc pull          # fetch on a fresh clone
```

## Experiment tracking (MLflow)

```bash
mlflow ui --backend-store-uri ./outputs/mlruns
# open http://localhost:5000
```

Each `face-bias-train` run logs hyperparameters, per-epoch metrics, and the
final `history.json` and `best.pt` as artefacts.

## Testing

```bash
pytest                         # full suite
pytest -m unit                 # fast, no GPU, no I/O
pytest -m integration          # mocks + filesystem
pytest -m smoke                # end-to-end with mini fixtures
pytest -m "not gpu"            # skip CUDA-only tests
pytest --cov=face_bias         # with coverage
```

Test categories live in `tests/{unit,integration,smoke}/`. Markers are
declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Documentation

Long-form documentation under [docs/](docs/):

- [docs/exp01_vs_mba.md](docs/exp01_vs_mba.md) — sample experiment report
  with cross-baseline analysis.
- [docs/smoke_results.md](docs/smoke_results.md) — full smoke-run report
  across all 11 pre-defined experiments.

The internal code-review record and historical refactor plan are kept in
[REVIEW_AND_PLAN.md](REVIEW_AND_PLAN.md).

## License

MIT — see [LICENSE](LICENSE) (to be added).

## Citation

If this code is useful in your work, please cite:

```bibtex
@software{ozzetti_facebias,
  author  = {Ozzetti, Marcello},
  title   = {face-bias: Facial Recognition with Bias Mitigation},
  url     = {https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias}
}
```
