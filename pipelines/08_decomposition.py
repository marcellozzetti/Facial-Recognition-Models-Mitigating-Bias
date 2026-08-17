"""Pipeline stage — Etapa 6: síntese decompositiva final.

Uso:
    python pipelines/08_decomposition.py --config configs/mestrado/stages/etapa6_decomposition.yaml

Consome:
    outputs/etapa2/matriz_mst_x_raca.parquet
    outputs/etapa4/comparativo.parquet
    outputs/etapa5/rfw_verification.parquet

Produz (Contribuição 6):
    outputs/etapa6/decomposicao_latinx.parquet
    outputs/etapa6/decomposicao.png
    outputs/etapa6/sintese.md

TODO Jun/2027:
    [ ] Combinar Etapas 2, 4, 5
    [ ] Decomposição via ANOVA/R²:
        - Componente fenotípico (overlap MST intra-Latinx)
        - Componente algorítmico (redução vs baseline)
    [ ] Testar H4 (>= 50% Latinx errors em zonas overlap)
    [ ] Testar H6 (>= 70% variance explicada por pixel info)
    [ ] Visualização (waterfall ou barra empilhada)
    [ ] Relatório final para Cap. 5 da dissertação (a escrever pós-qualificação)
"""

from __future__ import annotations

# TODO: implementação
