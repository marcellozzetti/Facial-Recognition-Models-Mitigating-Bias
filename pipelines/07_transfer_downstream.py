"""Pipeline stage — Etapa 5: transferência fair para face recognition (RFW + BFW).

Uso:
    python pipelines/07_transfer_downstream.py --dataset rfw
    python pipelines/07_transfer_downstream.py --dataset bfw

Consome:
    outputs/etapa3/B_film_linear/model_seed42.pt  (backbone fair)

Produz:
    outputs/etapa5/rfw_verification.parquet
    outputs/etapa5/rfw_by_pixel_info.parquet    (estratificação Pangelinan)
    outputs/etapa5/bfw_intersectional.parquet

TODO Mai/2027:
    [ ] Carregar backbone fair
    [ ] Extrair embeddings de RFW e BFW pairs
    [ ] Cálculo similaridade coseno + limiar
    [ ] BiSeNet para face_pixel_fraction
    [ ] Estratificação por 3 descritores de qualidade
"""

from __future__ import annotations

# TODO: implementação
