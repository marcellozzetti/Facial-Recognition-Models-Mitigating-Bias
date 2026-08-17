"""Descritores adicionais de qualidade de imagem — salvaguarda Etapa 5.

Cap. 4, §4.2 (Etapa 5). Estratifica resultados por 3 descritores
complementares para blindar Etapa 6 contra crítica de que erro
Latinx seria puramente sensorial:

    - Luminância média: canal L* da escala CIELAB
    - Nitidez: variância do Laplaciano
    - Resolução efetiva: pixels da face segmentada

Interface pública prevista:
    def luminance_cielab(image: np.ndarray) -> float
    def sharpness_laplacian(image: np.ndarray) -> float
    def effective_resolution(image: np.ndarray, face_mask: np.ndarray) -> int
    def compute_all(images: list[Path]) -> pd.DataFrame

TODO Etapa 5 (Mai/2027):
    [ ] Implementar 3 descritores (OpenCV + skimage)
    [ ] Batch processing com progress bar
    [ ] Análise estratificada (persistência do gap após controle)
