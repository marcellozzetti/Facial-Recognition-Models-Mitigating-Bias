"""Pipeline stage — Etapa 2: auditoria FairFace + matriz MST × raça.

Uso:
    python pipelines/04_fairface_audit.py \\
        --config configs/mestrado/stages/etapa2_fairface_audit.yaml

Consome:
    outputs/etapa1/fairface_val_mst.parquet

Produz (Contribuição 2 desta pesquisa):
    outputs/etapa2/matriz_mst_x_raca.parquet
    outputs/etapa2/matriz_mst_x_raca.png
    outputs/etapa2/relatorio.md

TODO Dez/2026:
    [ ] Ler MST predictions
    [ ] Cross-tabulate por raça (7 × 10)
    [ ] Testar H3 (Latinx spread >= 5 classes)
    [ ] Publicar matriz (primeira pública documentada)
"""

from __future__ import annotations

# TODO: implementação
