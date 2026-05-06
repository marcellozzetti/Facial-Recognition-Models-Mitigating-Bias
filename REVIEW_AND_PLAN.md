# Revisão de Código e Plano de Refatoração

**Repositório:** `Facial-Recognition-Models-Mitigating-Bias`
**Autor da revisão:** consolidada para evolução MBA → Mestrado
**Data:** 06 de maio de 2026
**Stack a preservar:** Python 3.9, PyTorch 2.4.1, MTCNN (facenet-pytorch), OpenCV, pandas, scikit-learn, boto3, PyYAML

---

## Sumário

1. [Estado Atual do Projeto](#1-estado-atual-do-projeto)
2. [Achados Críticos e Bugs](#2-achados-críticos-e-bugs)
3. [Avaliação por Boas Práticas](#3-avaliação-por-boas-práticas)
4. [Estrutura Proposta](#4-estrutura-proposta)
5. [Plano de Refatoração Faseado](#5-plano-de-refatoração-faseado)
6. [Pipeline de Testes por Etapa](#6-pipeline-de-testes-por-etapa)
7. [Próximos Passos Práticos](#7-próximos-passos-práticos)

---

## 1. Estado Atual do Projeto

### 1.1 Mapa de arquivos (excluindo `.venv`, `.git`, `old/`, `__pycache__/`)

| Arquivo | Linhas | Estado |
|---|---:|---|
| `configs/default.yaml` | 48 | **Funcional, com leak de credencial** |
| `data/bucket_dataset.py` | 106 | Funcional |
| `data/face_dataset.py` | 109 | **Funcional com bugs** |
| `models/arc_layers.py` | 47 | Funcional (correto) |
| `models/base_model.py` | 7 | Esqueleto, **não utilizado** |
| `models/losses.py` | 22 | **Implementação incorreta** |
| `models/resnet_model.py` | 43 | Funcional |
| `pipelines/pre_processing_pipeline.py` | 112 | Funcional |
| `pipelines/training_pipeline.py` | 0 | **Vazio** |
| `pipelines/evaluation_pipeline.py` | 0 | **Vazio** |
| `preprocessing/pre_processing_images.py` | 150 | Funcional |
| `tests/test_data_loading.py` | 0 | **Vazio** |
| `tests/test_data_rotation.py` | 0 | **Vazio** |
| `tests/test_draw_bounding.py` | 36 | **Quebrado** (assinatura incompatível) |
| `tests/test_models.py` | 0 | **Vazio** |
| `utils/config.py` | 9 | Funcional |
| `utils/custom_logging.py` | 19 | Funcional |
| `utils/metrics.py` | 0 | **Vazio** |
| `utils/versions.py` | 72 | Funcional |
| `notebooks/MBA_IA_USP_marcello_ozzetti.ipynb` | — | Legado MBA |
| `pyproject.toml` / `requirements.txt` | — | **Inexistente** |
| `README.md` | 2 | Placeholder |
| `ROADMAP.md` | 10 | Placeholder |

### 1.2 Diagnóstico em uma frase

O projeto tem **um pipeline de pré-processamento funcional + esqueletos de modelos**, mas **as etapas de treinamento, avaliação e teste estão vazias**, com **bugs silenciosos no carregamento de dados** e **uma falha crítica na implementação de ArcFace**.

---

## 2. Achados Críticos e Bugs

### 2.1 🔴 Crítico — Credencial AWS exposta no repositório

**Arquivo:** [configs/default.yaml:40](configs/default.yaml#L40)

```yaml
bucket_download_url: 'https://...&X-Amz-Credential=ASIAS74TL4TIRAUMFFFM%2F20250202...'
```

A URL pré-assinada da S3 inclui `AKID` (`ASIAS74TL4TIRAUMFFFM`), assinatura, e está versionada no Git. URLs pré-assinadas expiram (12h aqui), mas:
- O AKID temporário está exposto.
- Se a chave for de longo prazo, é leak permanente.
- Padrão errado para o mestrado (auditoria regulatória).

**Ação:** rotacionar a credencial; mover URL para variável de ambiente ou `configs/credentials.json` (já no `.gitignore`); reescrever histórico se for credencial não-rotativa.

### 2.2 🔴 Crítico — `ArcFaceLoss` não implementa ArcFace

**Arquivo:** [models/losses.py:6-22](models/losses.py#L6-L22)

```python
class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        ...
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        ...

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)  # ← apenas cross-entropy!
```

O construtor calcula constantes da margem angular, mas `forward` ignora tudo e retorna `cross_entropy`. **Resultado: todos os "experimentos com ArcFaceLoss" do MBA, na verdade, rodaram cross-entropy puro.**

Isso explica por que ArcFaceLoss teve performance similar ou pior que CrossEntropyLoss no MBA. A lógica angular existe em [models/arc_layers.py](models/arc_layers.py) (`ArcMarginProduct`), mas **não é integrada ao modelo** ([models/resnet_model.py](models/resnet_model.py) usa `nn.Linear` direto).

**Ação:** discutir explicitamente na qualificação do mestrado. Reportar como descoberta do processo de revisão. Em seguida, integrar `ArcMarginProduct` ao modelo e re-rodar os experimentos chave do MBA — esse experimento isolado já é uma contribuição válida.

### 2.3 🟠 Alto — Bug de normalização: `image_std` é igual a `image_mean`

**Arquivo:** [data/face_dataset.py:56,61,66](data/face_dataset.py#L56)

```python
transforms.Normalize(config['image']['image_mean'], config['image']['image_mean'])
#                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                   deveria ser image_std
```

Todas as imagens foram normalizadas com `mean=std=[0.485, 0.456, 0.406]`, em vez de usar `std=[0.229, 0.224, 0.225]` definido no YAML. Isso afeta diretamente os resultados experimentais, pois a entrada da rede pré-treinada (ImageNet) ficou fora da escala esperada.

### 2.4 🟠 Alto — Chave do YAML inconsistente: `image_size` vs `input_size`

**Arquivo:** [data/face_dataset.py:53,59,64](data/face_dataset.py#L53)

```python
transforms.Resize((config['image']['input_size'], config['image']['input_size']))
```

[configs/default.yaml:18](configs/default.yaml#L18) define `image_size`, não `input_size`. Esse código falha com `KeyError` ao executar.

### 2.5 🟠 Alto — `class_to_idx` não existe em `FaceDataset`

**Arquivo:** [data/face_dataset.py:100](data/face_dataset.py#L100)

```python
return dataloaders, datasets['train'].class_to_idx
```

`FaceDataset` (linhas 15-43) não define `class_to_idx`. Esse `setup_dataset` falha em runtime. Deveria retornar `label_encoder.classes_` ou similar.

### 2.6 🟠 Alto — `setup_dataset` não retorna `num_classes` nem `label_encoder`

Sem isso, o pipeline de treinamento não consegue:
- Saber o número de classes para construir o modelo;
- Decodificar predições para nomes de raça nas matrizes de confusão.

### 2.7 🟡 Médio — `sys.path.append` em todos os arquivos

```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

Anti-padrão Python. Quebra quando o projeto é instalado, importado por testes ou empacotado. Solução: tornar o projeto um pacote instalável via `pyproject.toml`.

### 2.8 🟡 Médio — `BaseModel` definido mas não usado

[models/base_model.py](models/base_model.py) declara `BaseModel(nn.Module)` mas `LResNet50E_IR` herda de `nn.Module` direto. Ou usa, ou remove.

### 2.9 🟡 Médio — `tests/test_draw_bounding.py` tem assinatura incompatível

[tests/test_draw_bounding.py:24](tests/test_draw_bounding.py#L24) chama `draw_bounding_procedure(image, bbox)` com 2 args. A função real ([preprocessing/pre_processing_images.py:53](preprocessing/pre_processing_images.py#L53)) requer 3: `(img, bbox, landmark)`. O teste falha. E não é um teste — é um script com `__main__`.

### 2.10 🟡 Médio — Sem reprodutibilidade

- Sem `seed_everything()` em PyTorch + numpy + Python.
- Sem `torch.backends.cudnn.deterministic = True`.
- Sem versionamento de dataset (DVC ou hash de checksum).
- Sem registro de hiperparâmetros por experimento.

### 2.11 🟡 Médio — Sem `requirements.txt` / `pyproject.toml`

Não há como reproduzir o ambiente. Versões só estão documentadas no Cap. 4 do `.tex` da dissertação.

### 2.12 🟢 Baixo — `cropping_procedure` pode estourar índices negativos

[preprocessing/pre_processing_images.py:96](preprocessing/pre_processing_images.py#L96):

```python
return img[y1-border:y1+height+border, x1-border:x1+width+border]
```

Se `y1 - border < 0`, NumPy aceita índice negativo e retorna fatia errada (do final da imagem). Função-irmã comentada (linhas 98-125) já tem o `max(0, ...)` correto — está comentada mas é a versão certa.

### 2.13 🟢 Baixo — Logging não estruturado

Cada módulo escreve em arquivo próprio (`bucket.log`, `preprocessing.log`, etc.) sem timestamp de correlação ou ID de experimento. Difícil depurar runs longos.

---

## 3. Avaliação por Boas Práticas

### 3.1 Boas práticas Python científico (checklist)

| Categoria | Item | Estado |
|---|---|---|
| **Empacotamento** | `pyproject.toml` com dependências fixadas | ❌ |
| **Empacotamento** | Pacote instalável via `pip install -e .` | ❌ |
| **Empacotamento** | Sem `sys.path` hacks | ❌ |
| **Reprodutibilidade** | Seeds determinísticas | ❌ |
| **Reprodutibilidade** | Versionamento de dataset | ❌ |
| **Reprodutibilidade** | Tracking de experimentos | ❌ |
| **Configuração** | YAML por experimento | ⚠️ Parcial (1 só) |
| **Configuração** | Validação de schema (Pydantic / `jsonschema`) | ❌ |
| **Configuração** | Secrets fora do repo | ⚠️ Parcial (URL leak) |
| **Estrutura** | Separação clara `data/`, `models/`, `pipelines/`, `utils/` | ✅ |
| **Estrutura** | `__init__.py` com APIs públicas | ❌ (vazios) |
| **Qualidade** | Type hints | ❌ |
| **Qualidade** | Docstrings consistentes | ⚠️ Parcial |
| **Qualidade** | Linter configurado (ruff / black / isort) | ❌ |
| **Qualidade** | Pre-commit hooks | ❌ |
| **Testes** | `pytest` com fixtures | ❌ (placeholders) |
| **Testes** | Cobertura mínima | ❌ |
| **Testes** | Smoke tests do pipeline ponta-a-ponta | ❌ |
| **CI/CD** | GitHub Actions ou similar | ❌ |
| **Documentação** | README útil | ❌ (2 linhas) |
| **Documentação** | CHANGELOG | ❌ |

### 3.2 Boas práticas específicas de redes neurais / treinamento

| Categoria | Item | Estado |
|---|---|---|
| **Modelo** | `model.train()` / `model.eval()` separados | ❌ (sem código) |
| **Modelo** | `torch.no_grad()` em validação/teste | ❌ |
| **Modelo** | Mixed precision (`torch.cuda.amp`) | ❌ |
| **Modelo** | Checkpoints salvos por época | ❌ |
| **Modelo** | Early stopping com patience | ❌ |
| **Modelo** | Resume from checkpoint | ❌ |
| **Dados** | Augmentação real (não só `RandomHorizontalFlip`) | ⚠️ Mínima |
| **Dados** | `pin_memory=True` | ✅ |
| **Dados** | `num_workers > 0` | ✅ |
| **Dados** | DataLoader com seed para reprodutibilidade | ❌ |
| **Métricas** | Métricas por época logadas | ❌ |
| **Métricas** | Métricas por subgrupo demográfico | ❌ |
| **Métricas** | Métricas de fairness (IR/FDR/CEI) | ❌ |
| **Tracking** | TensorBoard / MLflow / W&B | ❌ |
| **Hardware** | Verificação de GPU disponível | ✅ (em `versions.py`) |
| **Hardware** | Mover modelo/dados para `device` | ❌ |

---

## 4. Estrutura Proposta

### 4.1 Princípios

1. **Preservar a stack do MBA** (PyTorch + MTCNN + ResNet50 + ArcFace + S3 + YAML).
2. **Cada etapa testável isoladamente** — espelha os "experimentos" do MBA mas como código reproduzível.
3. **Sem `sys.path` hacks** — pacote Python instalável.
4. **Configuração por arquivo + override por CLI** — um YAML por experimento.
5. **Tracking nativo** — MLflow embutido (mais leve que W&B, sem login externo).

### 4.2 Árvore de diretórios proposta

```
Facial-Recognition-Models-Mitigating-Bias/
├── pyproject.toml                       # 🆕 Empacotamento e dependências
├── README.md                            # 🔄 Reescrever
├── PROPOSTA_MESTRADO.md                  # ✅ Já criado
├── REVIEW_AND_PLAN.md                    # ✅ Este documento
├── .gitignore                           # ✅ OK (manter)
├── .pre-commit-config.yaml              # 🆕 Linters automáticos
│
├── configs/
│   ├── default.yaml                     # 🔄 Sanitizar credenciais
│   ├── experiments/                     # 🆕
│   │   ├── exp01_crossentropy_sgd.yaml  #     1 YAML por experimento
│   │   ├── exp02_arcface_sgd.yaml
│   │   └── ...
│   └── credentials.json.example         # 🆕 Template (sem segredos)
│
├── src/                                 # 🆕 Pacote principal (instalável)
│   └── face_bias/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py                # 🔄 Antigo utils/config.py
│       │   └── schema.py                # 🆕 Validação Pydantic
│       ├── data/
│       │   ├── __init__.py
│       │   ├── bucket.py                # 🔄 Antigo data/bucket_dataset.py
│       │   ├── dataset.py               # 🔄 Antigo data/face_dataset.py (fix bugs)
│       │   └── transforms.py            # 🆕 Pipelines de augmentação
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── detection.py             # 🔄 detect_and_adjust_faces, etc.
│       │   ├── alignment.py             # 🔄 alignment_procedure
│       │   └── visualization.py         # 🔄 draw_bounding_procedure
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── resnet.py                # 🔄 LResNet50E_IR
│       │   ├── arc_margin.py            # 🔄 ArcMarginProduct (sem mudança)
│       │   └── losses.py                # 🔧 Corrigir ArcFaceLoss
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py               # 🆕 Loop de treino
│       │   ├── callbacks.py             # 🆕 Early stop, checkpoint
│       │   └── schedulers.py            # 🆕 OneCycleLR, CosineAnnealing
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py             # 🆕 Loop de avaliação
│       │   ├── metrics.py               # 🆕 Acc, F1, log-loss + IR/FDR/CEI
│       │   └── reports.py               # 🆕 Matriz de confusão, relatório
│       └── utils/
│           ├── __init__.py
│           ├── logging.py               # 🔄 Antigo custom_logging
│           ├── reproducibility.py       # 🆕 seed_everything()
│           └── system.py                # 🔄 Antigo versions.py
│
├── pipelines/                           # 🔄 Scripts entrypoint (CLI)
│   ├── 01_download_dataset.py           # 🔄 Wrapper de bucket
│   ├── 02_preprocess.py                 # 🔄 Wrapper de pre_processing_pipeline
│   ├── 03_train.py                      # 🆕 Treino orquestrado
│   ├── 04_evaluate.py                   # 🆕 Avaliação
│   └── 05_fairness_audit.py             # 🆕 IR/FDR/CEI sobre runs salvos
│
├── tests/                               # 🆕 Estrutura completa pytest
│   ├── __init__.py
│   ├── conftest.py                      # 🆕 Fixtures compartilhadas
│   ├── unit/                            #     Testes rápidos, sem GPU
│   │   ├── test_config.py
│   │   ├── test_dataset.py
│   │   ├── test_preprocessing.py
│   │   ├── test_models.py
│   │   ├── test_losses.py
│   │   └── test_metrics.py
│   ├── integration/                     #     Testes médios, lê arquivos
│   │   ├── test_bucket_download.py
│   │   ├── test_preprocessing_pipeline.py
│   │   └── test_dataset_pipeline.py
│   └── smoke/                           #     Smoke tests ponta-a-ponta
│       ├── test_train_one_step.py
│       ├── test_evaluate_one_batch.py
│       └── test_full_pipeline_mini.py
│
├── notebooks/
│   ├── MBA_IA_USP_marcello_ozzetti.ipynb   # ✅ Manter como referência histórica
│   └── exploratory/                       # 🆕 Notebooks de análise
│
├── scripts/                             # 🆕 Utilitários
│   ├── fix_arcface_bug_demo.py
│   └── compare_experiments.py
│
├── data/                                # ⚠️ Apenas paths, dados via DVC
│   ├── raw/                             # 🆕 (gitignored)
│   ├── processed/                       # 🆕 (gitignored)
│   └── synthetic/                       # 🆕 (gitignored, futuro)
│
├── outputs/                             # 🆕 (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── mlruns/                          #     MLflow tracking
│
└── docs/                                # 🆕 (futuro)
    └── architecture.md
```

### 4.3 `pyproject.toml` proposto (esqueleto)

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "face-bias"
version = "0.2.0-mestrado"
description = "Facial recognition bias mitigation - MBA → Mestrado"
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
    "unit: testes unitários rápidos",
    "integration: testes que tocam arquivos",
    "smoke: testes ponta-a-ponta com dados mínimos",
    "gpu: testes que requerem CUDA",
]
```

---

## 5. Plano de Refatoração Faseado

Faseado em 4 sprints de ~1 semana, alinhados ao **Mês 1** do plano de mestrado em [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md).

### Sprint A — Higienização e empacotamento (semana 1)

| # | Item | Prioridade |
|---|---|---|
| A1 | Criar `pyproject.toml` e instalar pacote (`pip install -e .`) | 🔴 |
| A2 | Mover código para `src/face_bias/` e remover todos os `sys.path.append` | 🔴 |
| A3 | Sanitizar `configs/default.yaml` — remover URL com credencial | 🔴 |
| A4 | Rotacionar credencial AWS exposta | 🔴 |
| A5 | Adicionar `requirements.txt` gerado a partir do ambiente atual (lock) | 🟠 |
| A6 | Configurar `ruff` + `black` + pre-commit | 🟠 |
| A7 | Reescrever `README.md` com setup instructions | 🟠 |

### Sprint B — Correção de bugs e reprodutibilidade (semana 2)

| # | Item | Prioridade |
|---|---|---|
| B1 | Corrigir `image_std` em `face_dataset.py` | 🔴 |
| B2 | Corrigir `image_size` (não `input_size`) em `face_dataset.py` | 🔴 |
| B3 | Corrigir retorno de `setup_dataset` (num_classes + label_encoder) | 🔴 |
| B4 | **Corrigir `ArcFaceLoss`** — integrar `ArcMarginProduct` ao modelo | 🔴 |
| B5 | Implementar `utils/reproducibility.py` com `seed_everything()` | 🟠 |
| B6 | Adicionar validação Pydantic ao YAML | 🟠 |
| B7 | Migrar logging para padrão estruturado com run_id | 🟢 |

### Sprint C — Treino e avaliação (semana 3)

| # | Item | Prioridade |
|---|---|---|
| C1 | Implementar `training/trainer.py` com loop completo | 🔴 |
| C2 | Implementar `evaluation/evaluator.py` | 🔴 |
| C3 | Implementar `evaluation/metrics.py` com Acc, F1, log-loss + **fairness (IR, FDR, CEI)** | 🔴 |
| C4 | Implementar callbacks: ModelCheckpoint, EarlyStopping | 🟠 |
| C5 | Integrar MLflow tracking | 🟠 |
| C6 | Implementar schedulers (OneCycleLR, CosineAnnealingWarmRestarts) | 🟠 |
| C7 | Pipeline CLI `pipelines/03_train.py` com `--config` arg | 🟠 |

### Sprint D — Testes e CI (semana 4)

| # | Item | Prioridade |
|---|---|---|
| D1 | Implementar fixtures em `tests/conftest.py` | 🔴 |
| D2 | Implementar testes unitários (ver §6) | 🔴 |
| D3 | Implementar testes de integração (ver §6) | 🟠 |
| D4 | Implementar smoke tests (ver §6) | 🟠 |
| D5 | Configurar GitHub Actions: lint + tests | 🟠 |
| D6 | Adicionar badge de cobertura ao README | 🟢 |

---

## 6. Pipeline de Testes por Etapa

Espelha as **etapas experimentais do MBA** ([Cap. 4 da dissertação](USPSC-Cap4-Avaliacao_Experimental.tex)):

> 1. Seleção e análise do conjunto de dados
> 2. Pré-processamento (Experimento I rotação, Experimento II MTCNN)
> 3. Treinamento do modelo (11 experimentos)
> 4. Análise de desempenho

Cada etapa terá **testes em três níveis**: unit (rápido), integration (lê arquivos), smoke (ponta-a-ponta com mini-dataset).

### 6.1 Etapa 0 — Configuração e ambiente

**Objetivo:** garantir que o projeto carrega config, valida schema e detecta GPU.

| Teste | Tipo | O que valida |
|---|---|---|
| `test_config_loads_default_yaml` | unit | YAML carrega sem erro |
| `test_config_schema_valid` | unit | Pydantic aceita campos esperados |
| `test_config_rejects_missing_fields` | unit | Erro claro se faltar campo obrigatório |
| `test_seed_everything_deterministic` | unit | 2 chamadas produzem mesmos números aleatórios |
| `test_system_report_includes_torch` | unit | `system_info_report()` retorna versão de torch |

### 6.2 Etapa 1 — Aquisição de dados (`bucket`)

**Objetivo:** download/upload S3 funcionam, sem expor credenciais.

| Teste | Tipo | O que valida |
|---|---|---|
| `test_credentials_loaded_from_env` | unit | Função aceita credencial via `os.environ` |
| `test_credentials_not_in_yaml` | unit | YAML não contém `X-Amz-Credential` |
| `test_bucket_client_initialized` | integration | `boto3.client('s3', ...)` retorna objeto válido (mock) |
| `test_download_zip_extracts` | integration | Mock S3 com `moto` → download e unzip |
| `test_download_handles_404` | integration | URL inválida → log de erro, sem crash |

### 6.3 Etapa 2 — Pré-processamento (MTCNN + alinhamento)

**Objetivo:** detecção, recorte e alinhamento corretos. **Espelho direto dos Experimentos I e II do MBA.**

| Teste | Tipo | O que valida |
|---|---|---|
| `test_detect_faces_returns_boxes` | unit | MTCNN com imagem dummy retorna lista (não None) |
| `test_alignment_procedure_rotates_correctly` | unit | Olhos alinhados horizontalmente após `alignment_procedure` |
| `test_cropping_handles_negative_indices` | unit | Bbox próximo a borda → não retorna fatia errada (bug §2.12) |
| `test_rotate_procedure_45deg` | unit | Imagem rotacionada 45° tem mesma shape, não-nula |
| `test_resize_to_224x224` | unit | Saída tem shape (224, 224, 3) |
| `test_pipeline_processes_directory` | integration | Pipeline em `tests/fixtures/sample_images/` produz arquivos esperados |
| `test_mtcnn_detection_rate_baseline` | integration | **Reproduz Experimento I do MBA**: detecta ≥60% das faces em fixtures rotacionadas |
| `test_no_quality_loss_after_pipeline` | integration | **Reproduz Experimento II do MBA**: MTCNN re-detecta faces após processamento |

### 6.4 Etapa 3 — Dataset PyTorch

**Objetivo:** `FaceDataset` carrega corretamente, normaliza, encoda labels.

| Teste | Tipo | O que valida |
|---|---|---|
| `test_dataset_len_matches_csv` | unit | `len(dataset) == len(csv)` |
| `test_dataset_returns_tensor_and_label` | unit | `__getitem__` retorna `(Tensor, int)` |
| `test_dataset_normalization_uses_imagenet_stats` | unit | Tensor médio ≈ 0, desvio ≈ 1 (regressão para bug §2.3) |
| `test_dataset_image_shape_3x224x224` | unit | Tensor tem shape `[3, 224, 224]` |
| `test_label_encoder_round_trip` | unit | `encode → decode == original` |
| `test_train_val_test_split_stratified` | integration | Distribuição de raça preservada nos 3 splits (chi² test) |
| `test_class_balance_after_undersampling` | integration | **Reproduz tabela `tab:raceCountBalanceado` do MBA** |
| `test_dataloader_iterates_one_batch` | smoke | `next(iter(dataloader))` funciona |

### 6.5 Etapa 4 — Modelos e Loss

**Objetivo:** modelo constrói, faz forward pass, ArcFace funciona de verdade (bug §2.2).

| Teste | Tipo | O que valida |
|---|---|---|
| `test_resnet50_constructs_with_7_classes` | unit | Modelo é instanciável |
| `test_resnet50_forward_returns_correct_shape` | unit | Input `[B,3,224,224]` → output `[B, 7]` |
| `test_resnet50_pretrained_weights_loaded` | unit | Camadas conv têm pesos não-aleatórios |
| `test_resnet50_dropout_is_active_in_train_mode` | unit | `model.train()` ativa dropout, `model.eval()` desativa |
| `test_arcmarginproduct_forward_shape` | unit | `ArcMarginProduct` retorna logits no shape esperado |
| `test_arcfaceloss_differs_from_crossentropy` | unit | **Regressão para bug §2.2**: loss com margin > 0 difere de cross-entropy |
| `test_loss_gradient_flows_to_input` | unit | `loss.backward()` produz gradientes não-nulos |
| `test_arcface_decreases_intra_class_distance` | integration | Sanity: pares mesma classe → embeddings mais próximos após 1 update |

### 6.6 Etapa 5 — Loop de treinamento

**Objetivo:** treinar 1 batch, 1 época mini, validar persistência.

| Teste | Tipo | O que valida |
|---|---|---|
| `test_trainer_one_step_decreases_loss` | smoke | Loss em t1 < loss em t0 (com lr alto, 1 batch repetido) |
| `test_trainer_saves_checkpoint` | smoke | Após 1 época, arquivo `.pt` existe |
| `test_trainer_resumes_from_checkpoint` | smoke | Carregar checkpoint reproduz pesos |
| `test_early_stopping_halts_training` | unit | Após N épocas sem melhora, treino para |
| `test_scheduler_onecyclelr_warmup` | unit | LR aumenta nos primeiros steps |
| `test_mlflow_logs_metrics` | smoke | Após 1 época, `mlruns/` contém run com `train_loss` |
| `test_full_pipeline_mini_dataset` | smoke | **Reproduz Experimento 1 do MBA em escala mini** (10 imagens, 1 época) |

### 6.7 Etapa 6 — Avaliação e métricas

**Objetivo:** confusion matrix, classification report, **fairness metrics** (lacuna do MBA).

| Teste | Tipo | O que valida |
|---|---|---|
| `test_confusion_matrix_shape_7x7` | unit | Para 7 classes, matriz 7×7 |
| `test_classification_report_keys` | unit | Retorna precision, recall, F1 por classe |
| `test_log_loss_is_positive` | unit | Sempre > 0 |
| `test_inequity_rate_known_input` | unit | Para FMR `[0.01, 0.05]` → IR = 5.0 |
| `test_fdr_known_input` | unit | Caso clássico do paper de Pereira & Marcel |
| `test_garbe_zero_when_perfectly_balanced` | unit | Grupos iguais → GARBE = 0 |
| `test_cei_combines_disparities` | unit | Validação contra implementação de referência |
| `test_full_evaluation_on_fixture_predictions` | integration | Pipeline lê CSV de predições e gera relatório completo |
| `test_audit_recomputes_mba_baseline` | smoke | **Recalcula fairness sobre matrizes do MBA — entregável Mês 1** |

### 6.8 Resumo: cobertura por etapa

| Etapa | Unit | Integration | Smoke | Total |
|---|---:|---:|---:|---:|
| 0 — Config | 5 | 0 | 0 | 5 |
| 1 — Bucket | 2 | 3 | 0 | 5 |
| 2 — Pré-processamento | 5 | 3 | 0 | 8 |
| 3 — Dataset | 5 | 2 | 1 | 8 |
| 4 — Modelos | 7 | 1 | 0 | 8 |
| 5 — Treinamento | 2 | 0 | 5 | 7 |
| 6 — Avaliação | 7 | 1 | 1 | 9 |
| **Total** | **33** | **10** | **7** | **50** |

50 testes é uma meta agressiva mas factível para o Mês 1. Em 4 sprints de ~12 testes/sprint.

---

## 7. Próximos Passos Práticos

### 7.1 Decisões em aberto (precisam de aprovação antes da implementação)

1. **Reescrita histórica do Git para remover credencial?** Se a credencial AWS é de longo prazo, a URL antiga continua acessível no histórico mesmo após `git rm`. Há duas opções:
   - **Opção A** — rotacionar credencial e seguir; histórico fica com leak de credencial inválida (preferida se credencial era temporária `ASIA*` que já expirou).
   - **Opção B** — `git filter-repo` para reescrever histórico (mais agressivo, exige force-push).

2. **Criar branch separado para refactor?** Recomendo `feat/mestrado-refactor` para preservar `main` enquanto a refatoração avança.

3. **Migrar para `src/face_bias/` agora ou depois?** Migrar agora gera commit grande mas evita retrabalho. Se preferir incremental, pode-se manter estrutura plana.

4. **Tracking — MLflow ou TensorBoard?** MLflow é mais robusto (compara experimentos, salva artefatos), TensorBoard é mais leve.

5. **DVC para versionar dataset?** Recomendado para mestrado mas adiciona complexidade. Alternativa: hash SHA256 + manifesto JSON.

### 7.2 Ordem sugerida de execução

Imediata (< 1 dia):
- A3 + A4 — sanitizar credencial e rotacionar.
- B1 + B2 + B3 — corrigir bugs silenciosos do dataset (impacto no MBA).
- B4 — corrigir ArcFaceLoss (descoberta com impacto na narrativa do mestrado).

Sprint 1 (semana atual):
- A1 + A2 — pyproject.toml + reorganização para `src/`.
- B5 — `seed_everything()`.
- D1 — fixtures de teste.

A partir daí, seguir o plano de §5.

---

*Documento elaborado em revisão completa do código-fonte do repositório, em 2026-05-06, alinhado à proposta de mestrado em [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md).*
