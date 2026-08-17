"""Pipeline stage — Etapa 1: inferência MST via SkinToneNet.

Uso:
    python pipelines/03_mst_inference.py \\
        --config configs/mestrado/stages/etapa1_skintonenet.yaml \\
        --output outputs/etapa1/

Depende de:
    src/face_bias/mst/skintonenet.py

Etapa 1 do Cap. 4 §4.2 — Nov/2026.

TODO:
    [ ] Carregar SkinToneNet pré-treinado
    [ ] Aplicar sobre FairFace train + val + test (86.744 + 10.954 + 10.954)
    [ ] Salvar softmax MST 10-dim em parquet
    [ ] Registrar métrica em MLflow (n_images, inference_time_sec)
"""

from __future__ import annotations

# TODO: implementação
