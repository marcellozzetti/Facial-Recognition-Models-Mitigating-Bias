"""Síntese decompositiva — Etapa 6, Contribuição 6.

Cap. 4 §Etapa 6. Separa quanto do erro Latinx vem do componente
fenotípico (irredutível, dado a sobreposição MST intra-Latinx) vs
componente algorítmico (mitigável, redutível por escolha de método).

Método:
    - Ajusta modelo linear R² sobre erros por (raça × MST × modelo).
    - Decompõe variância explicada em fatores.
    - Testa H4 (>= 50% erros Latinx em zonas de sobreposição MST).
    - Testa H6 (>= 70% variância explicada por pixel information).

Interface pública:
    decompose_variance(errors_df) -> DecompositionResult
    test_h4_overlap_concentration(errors_df, threshold=0.5) -> H4Result
    test_h6_pixel_info_explains(errors_df, threshold=0.7) -> H6Result
    generate_report(decomp, h4, h6) -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecompositionResult:
    r2_total: float
    r2_phenotypic: float   # variância explicada por MST + interações com MST
    r2_algorithmic: float  # variância explicada por modelo/config
    r2_residual: float
    n_observations: int

    def as_dict(self) -> dict:
        return {
            "r2_total": self.r2_total,
            "r2_phenotypic": self.r2_phenotypic,
            "r2_algorithmic": self.r2_algorithmic,
            "r2_residual": self.r2_residual,
            "n_observations": self.n_observations,
        }


@dataclass(frozen=True)
class H4Result:
    overlap_concentration: float
    threshold: float
    confirmed: bool
    latinx_errors_total: int
    latinx_errors_in_overlap: int

    def as_dict(self) -> dict:
        return {
            "hypothesis": "H4",
            "statement": "≥50% dos erros Latinx concentram-se em zonas de sobreposição MST",
            "threshold": self.threshold,
            "overlap_concentration": self.overlap_concentration,
            "confirmed": self.confirmed,
            "latinx_errors_total": self.latinx_errors_total,
            "latinx_errors_in_overlap": self.latinx_errors_in_overlap,
        }


@dataclass(frozen=True)
class H6Result:
    r2_pixel_info: float
    threshold: float
    confirmed: bool

    def as_dict(self) -> dict:
        return {
            "hypothesis": "H6",
            "statement": "≥70% da variância do erro é explicada por pixel information",
            "threshold": self.threshold,
            "r2_pixel_info": self.r2_pixel_info,
            "confirmed": self.confirmed,
        }


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R² de um ajuste linear (com intercept)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    y_hat = x @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _onehot(series: pd.Series) -> np.ndarray:
    """One-hot simples de uma Series categórica (sem sklearn)."""
    return pd.get_dummies(series).to_numpy(dtype=np.float64)


def decompose_variance(errors_df: pd.DataFrame) -> DecompositionResult:
    """Decompõe R² total em componentes.

    ``errors_df`` deve conter colunas:
        error (float ou 0/1)  — magnitude ou indicador de erro por observação
        mst_pred (int 1..10)  — tom Monk predito
        model (str)           — configuração / baseline
        (opcional) race       — para estratificação
    """
    required = {"error", "mst_pred", "model"}
    missing = required - set(errors_df.columns)
    if missing:
        raise ValueError(f"errors_df sem colunas: {sorted(missing)}")
    y = errors_df["error"].to_numpy(dtype=np.float64)

    mst_x = _onehot(errors_df["mst_pred"])
    model_x = _onehot(errors_df["model"])
    full_x = np.hstack([mst_x, model_x])

    r2_total = _linear_r2(full_x, y)
    r2_phen = _linear_r2(mst_x, y)
    r2_algo = _linear_r2(model_x, y)
    r2_res = max(0.0, 1.0 - r2_total) if not np.isnan(r2_total) else float("nan")
    return DecompositionResult(
        r2_total=r2_total,
        r2_phenotypic=r2_phen,
        r2_algorithmic=r2_algo,
        r2_residual=r2_res,
        n_observations=int(len(errors_df)),
    )


def test_h4_overlap_concentration(
    errors_df: pd.DataFrame,
    matrix_race_mst: pd.DataFrame,
    threshold: float = 0.5,
    overlap_min_races: int = 2,
    prop_threshold: float = 0.05,
) -> H4Result:
    """H4: >= threshold dos erros Latinx concentram-se em MSTs de sobreposição.

    ``matrix_race_mst`` é a matriz da Etapa 2 (7 raças × 10 MST, normalizada
    por linha). Um tom MST é "de sobreposição" se ao menos ``overlap_min_races``
    raças têm massa >= ``prop_threshold`` naquele tom.
    """
    if "race" not in errors_df.columns:
        raise ValueError("errors_df precisa da coluna 'race' para H4.")
    latinx = errors_df[errors_df["race"] == "Latino_Hispanic"]
    if latinx.empty:
        return H4Result(
            overlap_concentration=float("nan"),
            threshold=threshold,
            confirmed=False,
            latinx_errors_total=0,
            latinx_errors_in_overlap=0,
        )

    overlap_msts = []
    for mst_col in matrix_race_mst.columns:
        n_races_present = int((matrix_race_mst[mst_col] >= prop_threshold).sum())
        if n_races_present >= overlap_min_races:
            overlap_msts.append(int(mst_col))

    latinx_errors = latinx[latinx["error"] > 0]
    total = int(len(latinx_errors))
    in_overlap = int(latinx_errors["mst_pred"].isin(overlap_msts).sum())
    concentration = float(in_overlap / total) if total > 0 else float("nan")
    return H4Result(
        overlap_concentration=concentration,
        threshold=threshold,
        confirmed=(concentration >= threshold) if not np.isnan(concentration) else False,
        latinx_errors_total=total,
        latinx_errors_in_overlap=in_overlap,
    )


def test_h6_pixel_info_explains(
    errors_df: pd.DataFrame,
    threshold: float = 0.7,
) -> H6Result:
    """H6: >= threshold da variância do erro é explicada por pixel information."""
    if "pixel_information" not in errors_df.columns:
        raise ValueError("errors_df precisa da coluna 'pixel_information' para H6.")
    x = errors_df["pixel_information"].to_numpy(dtype=np.float64)
    y = errors_df["error"].to_numpy(dtype=np.float64)
    r2 = _linear_r2(x, y)
    return H6Result(
        r2_pixel_info=r2,
        threshold=threshold,
        confirmed=(r2 >= threshold) if not np.isnan(r2) else False,
    )


def generate_report(
    decomp: DecompositionResult,
    h4: Optional[H4Result] = None,
    h6: Optional[H6Result] = None,
) -> str:
    lines = [
        "# Etapa 6 — Síntese decompositiva",
        "",
        "## Decomposição de variância do erro",
        "",
        f"- R² total (fenótipo + modelo): **{decomp.r2_total:.3f}**",
        f"- R² fenotípico (MST): **{decomp.r2_phenotypic:.3f}**",
        f"- R² algorítmico (modelo/config): **{decomp.r2_algorithmic:.3f}**",
        f"- R² residual: **{decomp.r2_residual:.3f}**",
        f"- N observações: {decomp.n_observations}",
        "",
    ]
    if h4 is not None:
        lines += [
            "## H4 — concentração de erros Latinx em zonas de sobreposição MST",
            "",
            f"- Threshold: {h4.threshold:.0%}",
            f"- Concentração observada: **{h4.overlap_concentration:.1%}**",
            f"- Erros Latinx: {h4.latinx_errors_total} (na sobreposição: {h4.latinx_errors_in_overlap})",
            f"- **{'CONFIRMADA' if h4.confirmed else 'REFUTADA'}**",
            "",
        ]
    if h6 is not None:
        lines += [
            "## H6 — pixel information explica variância do erro (Pangelinan)",
            "",
            f"- Threshold: {h6.threshold:.0%}",
            f"- R² observado: **{h6.r2_pixel_info:.3f}**",
            f"- **{'CONFIRMADA' if h6.confirmed else 'REFUTADA'}**",
            "",
        ]
    return "\n".join(lines) + "\n"
