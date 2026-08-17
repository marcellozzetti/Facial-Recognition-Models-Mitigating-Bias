"""Pipeline stage — Etapa 4: comparação contra 6 baselines + Pareto.

Uso:
    python pipelines/06_evaluate_baselines.py --config configs/mestrado/stages/etapa4_baselines.yaml

Consome:
    outputs/etapa3/A_baseline/*, outputs/etapa3/B_film_linear/*  (nossos)
    baselines treinados (ResNet-34, FSCL+, Group DRO, FineFACE, Adv Debias)

Produz:
    outputs/etapa4/comparativo.parquet   (12 modelos × triangulação)
    outputs/etapa4/pareto.png            (visualização)
    outputs/etapa4/relatorio.md          (com aderência ISO 19795-10)

TODO Abr/2027:
    [ ] Treinar/carregar 6 baselines (usar src/face_bias/baselines/*)
    [ ] Aplicar triangulação (Cenário A + Cenário B)
    [ ] Gerar Pareto front
    [ ] Relatório ISO/IEC 19795-10:2024 compatible
"""

from __future__ import annotations

# TODO: implementação
