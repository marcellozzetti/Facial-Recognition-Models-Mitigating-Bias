"""Controle de pixel information via BiSeNet — Pangelinan et al. (2023).

Cap. 4, §4.2 (Etapa 5) e Cap. 3 (Hipótese H6). Responde à refutação central
sobre driver estrutural do gap em face recognition (Cap. 2, §2.6).

Segmentação:
    face_pixels = skin + eyebrows + eyes + ears + nose + mouth
    face_pixel_fraction = |face_pixels| / (H × W)

Uso:
    - Como confounder principal no controle da Etapa 5
    - Como teste direto de H6 (decomposição de variância)

Interface pública prevista:
    class PixelInfoAnalyzer:
        def __init__(self, bisenet_weights: Path, device: str = "cuda")
        def compute_face_fraction(images: torch.Tensor) -> torch.Tensor  # (N,)
        def stratify_by_fraction(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame

TODO Etapa 5 (Mai/2027):
    [ ] Baixar weights BiSeNet oficiais
    [ ] Wrapper de segmentação
    [ ] Cálculo face-pixel fraction
    [ ] Análise de variância explicada (H6)
