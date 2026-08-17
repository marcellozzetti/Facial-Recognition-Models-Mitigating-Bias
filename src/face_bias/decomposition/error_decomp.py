"""Decomposição do erro Latinx — Etapa 6, Contribuição 6 (eixo diagnóstico-estrutural).

Cap. 3 (Contribuição 6) e Cap. 4, §4.2 (Etapa 6). Combina resultados das
Etapas 2 (matriz MST × raça) e 5 (transferência fair) para quantificar:

    ERRO_LATINX = COMPONENTE_FENOTÍPICO + COMPONENTE_ALGORÍTMICO
              (irredutível)         (mitigável)

Onde:
    - Componente fenotípico: limite estrutural pela sobreposição MST
      intra-categoria (obtido da Etapa 2)
    - Componente algorítmico: parcela atacável por conditioning
      arquitetural (obtido comparando A vs B na Etapa 4 + Etapa 5)

Este é o diagnóstico estrutural que substitui a mera redução agregada
de F1 macro (linguagem da Contribuição 6, Cap. 3).

Interface pública prevista:
    class ErrorDecomposer:
        def __init__(self, matriz_mst_raca: pd.DataFrame, resultados_config: dict)
        def phenotypic_component() -> dict  # erro atribuível a overlap MST
        def algorithmic_component() -> dict  # erro que FiLM reduz vs baseline
        def variance_decomposition() -> pd.DataFrame  # tabela final para dissertação
        def visualize(save_path: Path) -> None  # diagrama de decomposição

TODO Etapa 6 (Jun/2027):
    [ ] Definição formal do modelo de variância (usar ANOVA / R²)
    [ ] Integração matriz Etapa 2 + resultados Etapa 4
    [ ] Cálculo dos dois componentes
    [ ] Visualização (barra empilhada ou waterfall)
    [ ] Relatório para Cap 5 da dissertação (a escrever pós-qualificação)

Ver também:
    - src/face_bias/audit/cross_matrix.py — provedor da matriz MST × raça
    - src/face_bias/fairness/pareto.py — visualização complementar
    - Cap 3 (Hipótese H4): ao menos 50% dos erros Latinx concentram em MST overlap
"""

from __future__ import annotations

# TODO: implementação
