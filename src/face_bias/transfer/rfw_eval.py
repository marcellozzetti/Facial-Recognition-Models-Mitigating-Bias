"""RFW (Racial Faces in-the-Wild) downstream — verificação um-para-um.

Cap. 4, §4.2 (Etapa 5). Wang et al. (2019).

Protocolo de verificação: decisão binária "mesma identidade?" sobre
pares de imagens estratificados por raça (4 categorias RFW).

Interface pública prevista:
    def load_rfw_pairs(rfw_root: Path) -> pd.DataFrame
    def compute_embeddings(backbone_fair, image_paths) -> np.ndarray
    def evaluate_verification(embeddings, pairs) -> pd.DataFrame  # FMR, FNMR por raça

TODO Etapa 5 (Mai/2027):
    [ ] Loader RFW pairs (protocol files oficiais)
    [ ] Wrapper embed→cosine sim
    [ ] Métricas: FMR@thresh, FNMR@thresh, EER por raça
    [ ] Integrar com pixel_information controller (BiSeNet)
