"""Disparity Ratio (DR) — razão entre desempenho do melhor e pior grupo.

Cap. 4, §4.8.1 (Cenário A). Métrica derivada usada na triangulação.

Fórmula:
    DR = min_c F1_c / max_c F1_c   ∈ (0, 1]  (1 = equidade perfeita)

Interface pública prevista:
    def disparity_ratio(f1_per_class: pd.Series) -> float
    def disparity_ratio_ci(preds, labels, n_boot: int = 10_000) -> tuple[float, float, float]

TODO Etapa 4 (Abr/2027):
    [ ] Implementação exata da fórmula
    [ ] Bootstrap IC 95%
    [ ] Unit test com dados sintéticos
