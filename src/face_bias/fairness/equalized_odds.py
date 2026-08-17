"""Equalized Odds — extensão restritiva de Equal Opportunity.

Cap. 4, §4.8.2 (Cenário B). Hardt, Price & Srebro (2016).

Definição:
    EqOdds satisfeita se TPR e FPR iguais entre grupos.
    Métrica: gap = max(|ΔTPR|, |ΔFPR|)

Interface pública prevista:
    def equalized_odds_binary(preds, labels, sensitive) -> dict  # {tpr_gap, fpr_gap, max_gap}
    def equalized_odds_per_race(preds, labels, race, gender) -> pd.DataFrame

TODO Etapa 4 (Abr/2027):
    [ ] Implementação binária (gender)
    [ ] Estratificação por raça (7 tabelas 2x2)
    [ ] Unit test
