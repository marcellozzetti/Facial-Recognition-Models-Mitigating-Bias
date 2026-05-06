# Facial Recognition Models Mitigating Bias

Toolkit para reconhecimento facial com foco em mitigação de viés demográfico,
desenvolvido como evolução da dissertação de MBA em IA (USP) para o
mestrado.

- **Roadmap acadêmico:** [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md)
- **Plano de refatoração:** [REVIEW_AND_PLAN.md](REVIEW_AND_PLAN.md)
- **Branch ativa:** `feat/mestrado-refactor`

## Stack

- Python 3.9–3.11
- PyTorch 2.4+, Torchvision
- facenet-pytorch (MTCNN)
- scikit-learn, pandas, OpenCV
- Pydantic, PyYAML, boto3
- MLflow (tracking) e DVC (versionamento de dados)

## Estrutura

```
src/face_bias/        # Pacote instalável (face-bias)
├── config/           # Carregador YAML
├── data/             # Bucket S3 + Dataset PyTorch
├── models/           # LResNet50E_IR, ArcFace, ArcMargin
├── preprocessing/    # MTCNN, alinhamento, rotação
├── utils/            # Logging, system info
└── cli/              # Pontos de entrada CLI
pipelines/            # Wrappers numerados (01_, 02_, …)
configs/              # default.yaml + credentials.json (gitignored)
tests/                # pytest unit/integration/smoke
notebooks/            # Notebook MBA preservado para referência
```

## Setup

### 1. Criar e ativar o virtualenv

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar o pacote em modo editable

```bash
pip install -e ".[dev,dvc]"
```

A flag `dev` adiciona pytest, ruff, black, pre-commit; `dvc` adiciona o
suporte ao DVC com remoto S3.

### 3. Configurar credenciais

```bash
cp configs/credentials.json.example configs/credentials.json
# Edite configs/credentials.json com chaves reais.
# O arquivo está no .gitignore — nunca commitar.
```

Alternativamente, exportar variáveis de ambiente:

```bash
export FACE_BIAS_BUCKET_ACCESS_KEY_ID=...
export FACE_BIAS_BUCKET_SECRET_ACCESS_KEY=...
export FACE_BIAS_BUCKET_URL=https://...
```

### 4. Instalar pre-commit hooks (uma vez)

```bash
pre-commit install
```

### 5. Verificar instalação

```bash
face-bias-version --config configs/default.yaml
```

## Pipeline (uso)

Cada etapa tem um CLI dedicado:

| Etapa | CLI | Status |
|---|---|---|
| Aquisição (S3) | `face-bias-download` | ✅ |
| Pré-processamento (MTCNN + align) | `face-bias-preprocess` | ✅ |
| Treinamento | `face-bias-train` | ⏳ Sprint C |
| Avaliação + fairness audit | `face-bias-evaluate` | ⏳ Sprint C |
| Info de ambiente | `face-bias-version` | ✅ |

Exemplo:

```bash
face-bias-download   --config configs/default.yaml
face-bias-preprocess --config configs/default.yaml
```

## Versionamento de dados (DVC)

Datasets não são versionados pelo Git. Usar DVC:

```bash
# Configurar remoto S3 (uma vez)
dvc remote add -d storage s3://your-bucket/dvc-store
dvc remote modify storage region us-east-1

# Rastrear um dataset
dvc add data/raw/fairface
git add data/raw/fairface.dvc .gitignore
git commit -m "data: add fairface raw dataset (v1.25.x)"

# Sincronizar
dvc push    # envia ao remoto
dvc pull    # baixa do remoto
```

## Tracking de experimentos (MLflow)

```bash
mlflow ui --backend-store-uri ./outputs/mlruns
# abra http://localhost:5000
```

A integração com o pipeline de treino chega no Sprint C.

## Testes

```bash
pytest                           # roda tudo
pytest -m unit                   # apenas unit tests
pytest -m "not gpu"              # pula testes que exigem GPU
pytest --cov=face_bias           # com cobertura
```

A suíte completa (50 testes em 7 etapas) está em [REVIEW_AND_PLAN.md §6](REVIEW_AND_PLAN.md).

## Estado atual (Sprint A — concluído)

✅ Pacote instalável `pip install -e .`
✅ Credencial AWS removida do repositório
✅ Estrutura `src/face_bias/`
✅ `pyproject.toml` + `requirements.txt` lock
✅ `.pre-commit-config.yaml` com detect-secrets
✅ DVC inicializado

## Próximo passo (Sprint B)

Correção dos bugs documentados em [REVIEW_AND_PLAN.md §2](REVIEW_AND_PLAN.md):
- B1: `image_std` em `transforms.Normalize`
- B2: chave `image_size` (não `input_size`)
- B3: retorno de `setup_dataset`
- B4: integrar `ArcMarginProduct` à `LResNet50E_IR` para que `ArcFaceLoss` realmente aplique margem angular
- B5: `seed_everything()`
- B6: validação Pydantic do YAML

## Citação

Se este código for útil para seu trabalho:

```bibtex
@misc{ozzetti2024facial,
  author = {Ozzetti, Marcello},
  title  = {Facial Recognition Models Mitigating Bias},
  year   = {2024},
  url    = {https://github.com/marcello-ozzetti/Facial-Recognition-Models-Mitigating-Bias}
}
```

## Licença

MIT — ver [LICENSE](LICENSE) (a adicionar).
