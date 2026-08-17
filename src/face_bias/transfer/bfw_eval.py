"""BFW (Balanced Faces in the Wild) — 8 subgrupos race × gender.

Cap. 4, §4.2 (Etapa 5) e §4.8.2 (Cenário B). Robinson et al. (2020).

Diferente do RFW, BFW é construído sobre 8 subgrupos formados por
race × gender por design — motivação do Cenário B da triangulação.

Interface pública prevista:
    def load_bfw_pairs(bfw_root: Path) -> pd.DataFrame
    def evaluate_intersectional(embeddings, pairs) -> pd.DataFrame  # métricas por subgrupo

TODO Etapa 5 (Mai/2027):
    [ ] Loader BFW pairs (8 subgrupos)
    [ ] Reuso do embed pipeline do rfw_eval
    [ ] Cross-check contra hiding bias (Cap 4 §4.8)
