"""Baseline: adversarial_debias — Etapa 4 (comparação sistemática).

Cap. 4, §4.4 (Baselines de comparação). Implementa referência para
comparação contra o pipeline FiLM+MST proposto (Config B).

TODO Etapa 4 (Abr/2027):
    [ ] Implementar treinamento fiel ao paper original
    [ ] Reproduzir hiperparâmetros documentados
    [ ] Rodar sobre FairFace 7-class in-domain
    [ ] Reportar 3 sementes (rigor: 42, 1, 2)
    [ ] Métricas: F1 macro, DR, worst-class F1, EO/EqOdds

Referências:
    - fscl_plus:            Park et al. (2022), CVPR — Fair SupCon Learning
    - group_dro:            Sagawa et al. (2020), ICLR — Distributionally Robust Optim
    - fineface:             Manzoor & Rattani (2024), ICPR — cross-layer attention
    - adversarial_debias:   Zhang, Lemoine & Mitchell (2018), AIES

Ver também:
    - src/face_bias/models/contrastive.py — SupCon base (herdar para FSCL+)
"""

from __future__ import annotations

# TODO: implementação
