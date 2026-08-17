"""Matriz cruzada Monk Skin Tone × classes raciais — Contribuição 2 desta pesquisa.

Cap. 3, Contribuição 2 (eixo fenotípico-empírico) e Cap. 4, §4.2 (Etapa 2).

Primeira matriz pública dessa distribuição. Contribui empiricamente para
o diagnóstico de heterogeneidade fenotípica intra-categorial (Cap. 2, §2.5).

Interface pública prevista:
    class CrossDistributionMatrix:
        def build(audit_df: pd.DataFrame) -> pd.DataFrame  # (7 raças) × (10 MST)
        def spread_per_race(matrix: pd.DataFrame) -> pd.Series  # #classes MST >= threshold
        def entropy_per_race(matrix: pd.DataFrame) -> pd.Series
        def visualize(matrix: pd.DataFrame, save_path: Path) -> None  # heatmap
        def test_hypothesis_H3(matrix: pd.DataFrame) -> dict  # H3: Latinx spread >= 5

Formato de publicação:
    outputs/etapa2/matriz_mst_x_raca.parquet   (dados)
    outputs/etapa2/matriz_mst_x_raca.png       (heatmap)
    outputs/etapa2/relatorio.md                (interpretação)

TODO Etapa 2 (Dez/2026):
    [ ] Implementar build (agregação por raça)
    [ ] Métricas: spread MST, entropia, coefficient of variation
    [ ] Testes formais de H3 (Latinx spread >= 5)
    [ ] Visualização heatmap com matplotlib (padrão dataviz da tese)

Ver também:
    - Cap 3, H3 (hipótese testada aqui)
    - Cap 6 (Etapa 6): consumidor da matriz para decomposição
"""

from __future__ import annotations

# TODO: implementação
