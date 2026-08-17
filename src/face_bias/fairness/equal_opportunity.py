"""Equal Opportunity — Hardt, Price & Srebro (2016).

Cap. 4, §4.8.2 (Cenário B, análise interseccional race × gender).

Definição binária original:
    EO(A) = P(Ŷ=1 | A=a, Y=1) - P(Ŷ=1 | A=b, Y=1) → 0

Adaptação multi-classe usada nesta pesquisa:
    Para cada raça c ∈ {7 classes}, gap entre TPR_c^M e TPR_c^F.
    Aplicação sobre eixo binário de gênero mantém a semântica original.

Interface pública prevista:
    def equal_opportunity_binary(preds, labels, sensitive_binary) -> float
    def equal_opportunity_per_race(preds, labels, race, gender) -> pd.Series

TODO Etapa 4 (Abr/2027):
    [ ] Cenário A: extensão multi-classe (one-vs-rest fica documentado)
    [ ] Cenário B: gap TPR_M vs TPR_F estratificado por raça
    [ ] Unit test com casos degenerados
