"""Validação humana interna do classificador MST — Cap. 4, §4.9.

Escopo declarado:
    Subset estratificado de aproximadamente 200-300 imagens do FairFace,
    com estratificação por raça e por tom MST. Rotulagem realizada
    EXCLUSIVAMENTE pela equipe acadêmica (Mestrando + Orientador).
    NÃO haverá contratação de anotadores externos nem uso de plataformas
    de crowdsourcing.

Métricas de concordância:
    - κ de Cohen par a par
    - Exatidão categórica com IC 95% via bootstrap

Interface pública prevista:
    class HumanAgreementProtocol:
        def sample_stratified(fairface_val: pd.DataFrame, n_target: int = 250) -> pd.DataFrame
        def compute_agreement(labels_a: list[int], labels_b: list[int]) -> dict
        def bootstrap_ci(kappa: float, n_boot: int = 10_000) -> tuple[float, float]

TODO Etapa 1 (Nov/2026):
    [ ] Amostragem estratificada por (raça, MST predito)
    [ ] Interface simples para rotulagem (CLI ou notebook)
    [ ] Cálculo κ + bootstrap CI
    [ ] Relatório em Markdown para anexo da dissertação

Ver também:
    - Cap 3 (objetivos): Objetivo 1 depende deste protocolo
    - Cap 5 (Riscos): Risco 2 (dependência SkinToneNet) menciona esta validação
"""

from __future__ import annotations

# TODO: implementação
