"""SkinToneNet wrapper — adota classificador MST pré-treinado como insumo da Etapa 1.

Referência: Matias, Costa, Neto & Novello de Brito (2026), arXiv 2603.02475.
Cap. 4, §4.2 (Etapa 1) e §4.9 (validação humana interna).

Este módulo carrega o SkinToneNet pré-treinado (ViT-S sobre STW),
expõe interface para inferência MST 10-classe sobre um lote de imagens
faciais (softmax → vetor R^10), e implementa cache local para inferências
repetidas (pipeline v3.x).

Interface pública prevista:
    class SkinToneNetInference:
        def __init__(self, weights_path: str, device: str = "cuda", cache_dir: Path | None = None)
        def infer(self, images: torch.Tensor) -> torch.Tensor  # (N, 10) softmax
        def infer_batch(self, image_paths: list[Path]) -> pd.DataFrame  # com hash-based cache

TODO Etapa 1 (Nov/2026):
    [ ] Baixar weights oficiais SkinToneNet (arXiv 2603.02475 - Matias 2026)
    [ ] Implementar carga do ViT-S ajustado
    [ ] Adicionar preprocessing consistente (224x224, ImageNet norm)
    [ ] Cache SQLite por hash SHA-256 da imagem
    [ ] Smoke test em subset FairFace 100 imagens

Ver também:
    - src/face_bias/mst/validation.py — protocolo humano interno
    - src/face_bias/mst/sensitivity.py — sensitivity analysis MST alternativos
    - src/face_bias/audit/fairface_mst.py — consumidor da Etapa 2
"""

from __future__ import annotations

# TODO: implementação
