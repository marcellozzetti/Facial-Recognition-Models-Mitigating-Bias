"""Descritores de qualidade de imagem — Cap. 4 §Etapa 5 (salvaguarda ótica).

Três descritores complementares para blindar a Etapa 6 contra a crítica
de que o gap Latinx seria puramente sensorial em vez de fenotípico:

    - luminância média (canal L* CIELAB)
    - nitidez (variância do Laplaciano)
    - resolução efetiva da face em pixels

Todos calculados a partir de crop facial (RGB HxWx3).
"""

from __future__ import annotations

import numpy as np


def luminance_lstar(rgb: np.ndarray) -> float:
    """Média do canal L* (CIELAB) — escala 0..100. Sem OpenCV."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"esperado HxWx3; recebido {rgb.shape}")
    arr = rgb.astype(np.float64) / 255.0
    # sRGB → XYZ (matriz D65)
    mask = arr > 0.04045
    linear = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ xyz_matrix.T
    # XYZ → L*  (referência D65 branco: Yn=1.0)
    y = xyz[..., 1]
    fy = np.where(y > 0.008856, y ** (1 / 3), 7.787 * y + 16 / 116)
    l_star = 116 * fy - 16
    return float(np.clip(l_star, 0, 100).mean())


def sharpness_laplacian(rgb: np.ndarray) -> float:
    """Variância do Laplaciano em cinza — proxy padrão de nitidez."""
    if rgb.ndim == 3:
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    else:
        gray = rgb
    gray = gray.astype(np.float64)
    # Laplaciano 3x3 discreto (kernel padrão)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    # descartar bordas
    inner = lap[1:-1, 1:-1]
    return float(inner.var())


def face_resolution_pixels(crop_hw: tuple[int, int]) -> int:
    """Área do crop facial em pixels (h * w). Proxy simples de resolução."""
    h, w = crop_hw
    return int(h) * int(w)


def descriptors_for_crop(rgb: np.ndarray) -> dict:
    """Todos os 3 descritores em um dict."""
    return {
        "luminance_lstar": luminance_lstar(rgb),
        "sharpness_laplacian": sharpness_laplacian(rgb),
        "face_area_pixels": face_resolution_pixels(rgb.shape[:2]),
    }
