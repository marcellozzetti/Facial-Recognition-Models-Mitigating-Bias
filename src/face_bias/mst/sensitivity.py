"""Sensitivity analysis — 2 ou 3 classificadores MST alternativos.

Salvaguarda contra propagação de viés herdado do SkinToneNet único.
Cap. 4, §4.2 (Etapa 1) e Cap. 3 (Objetivo 2).

Alternativas candidatas mapeadas no corpus:
    1. MST-KD (Caldeira, Cardoso, Sequeira & Neto, 2024) - Multiple Specialized Teachers KD
    2. Casual Conversations baseline (Hazirbas 2021 + Porgali 2023) - MST/Fitzpatrick anotado
    3. Modelo público adicional a definir (verificar Hugging Face 2026)

Interface pública prevista:
    class MSTSensitivityRunner:
        def run(images: torch.Tensor, models: list[str]) -> pd.DataFrame  # (N, n_models)
        def concordance_matrix(preds: pd.DataFrame) -> np.ndarray  # κ pairwise
        def rank_stability(...) -> float

TODO Etapa 1 (Nov/2026):
    [ ] Definir 2 alternativas ao SkinToneNet com weights públicos
    [ ] Framework de comparação (mesma preprocessing, mesmo device)
    [ ] Relatório de concordância

Ver também:
    - src/face_bias/mst/skintonenet.py — modelo principal
    - Cap 5, Risco 2 (mitigação declarada)
"""

from __future__ import annotations

# TODO: implementação
