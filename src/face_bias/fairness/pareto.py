"""Visualização Pareto-eficiente — Cap. 4, §4.8.3.

Projeta configurações e baselines no plano (1 - F1_macro, 1 - DR):
    - Eixo X: erro agregado (menor = melhor)
    - Eixo Y: disparidade (menor = melhor)
    - Fronteira Pareto: pontos não-dominados

Uma configuração é Pareto-superior a outra se reduz simultaneamente
ambas as coordenadas.

Referências metodológicas:
    - Manzoor & Rattani (2024) — FineFACE Pareto-eficiente
    - Dominguez-Catena, Paternain & Galar (2024) — DSAP com Pareto

Interface pública prevista:
    def build_pareto_data(results: dict[str, dict]) -> pd.DataFrame
    def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame  # pontos não-dominados
    def plot_pareto(df: pd.DataFrame, front: pd.DataFrame, save_path: Path) -> None

TODO Etapa 4 (Abr/2027):
    [ ] Algoritmo O(n log n) para fronteira Pareto 2D
    [ ] Plot com matplotlib (paleta tese: NAVY, BLUE_MID, ACCENT)
    [ ] Anotações por configuração (labels A/B/C/D + baselines)
