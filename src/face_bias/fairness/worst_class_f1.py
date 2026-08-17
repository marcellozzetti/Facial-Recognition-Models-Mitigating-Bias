"""Worst-class F1 — F1 do grupo demográfico mais penalizado.

Cap. 4, §4.8.1 (Cenário A). Alinhado ao princípio DRO de Sagawa (2020).

Interface pública prevista:
    def worst_class_f1(f1_per_class: pd.Series) -> tuple[float, str]  # valor, classe

TODO Etapa 4 (Abr/2027):
    [ ] Implementação + IC bootstrap
    [ ] Unit test
