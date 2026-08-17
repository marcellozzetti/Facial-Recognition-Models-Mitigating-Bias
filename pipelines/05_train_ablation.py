"""Pipeline stage — Etapa 3: treinamento das 4 configurações de ablation (A/B/C/D).

Uso:
    python pipelines/05_train_ablation.py --config-id A --seed 42
    python pipelines/05_train_ablation.py --config-id B --seed 1
    python pipelines/05_train_ablation.py --config-id C --seed 2
    python pipelines/05_train_ablation.py --config-id D --seed 42

Ou em lote:
    python pipelines/05_train_ablation.py --all  # 4 configs × 3 seeds = 12 runs

Consome:
    outputs/etapa1/*.parquet  (MST softmax para Config B/C/D)

Produz:
    outputs/etapa3/{A,B,C,D}_*/metrics_seed{42,1,2}.json
    outputs/etapa3/{A,B,C,D}_*/model_seed{42,1,2}.pt

TODO Jan-Mar/2027:
    [ ] Loader unificado com config injection
    [ ] Loop de 12 runs (4 configs × 3 seeds)
    [ ] MLflow tracking com tags {config_id, seed}
    [ ] Estimativa de tempo GPU (per Cap 5 Risco 4: 200-400h total)
"""

from __future__ import annotations

# TODO: implementação
