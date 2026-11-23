"""
_validation.py

Utilidades internas (privadas) de ``stats_toolkit`` para validar las entradas
de las funciones de inferencia (Módulos 4-6). Los mensajes de error están en
español y explican qué hacer.
"""

from __future__ import annotations

import numpy as np


def as_sample(data, name: str = "la muestra", min_n: int = 2) -> np.ndarray:
    """Convierte ``data`` en un array de floats y comprueba que es utilizable.

    Lanza ``ValueError`` si hay valores ausentes (NaN) o menos de ``min_n``
    observaciones. A diferencia de ``descriptive``, aquí no se eliminan los NaN
    en silencio: en un contraste, el tamaño muestral debe ser explícito.
    """
    arr = np.asarray(data, dtype=float).ravel()
    if np.isnan(arr).any():
        raise ValueError(
            f"{name} contiene valores ausentes (NaN): elimínalos o imputa antes de aplicar el contraste."
        )
    if arr.size < min_n:
        raise ValueError(f"{name} necesita al menos {min_n} observaciones (tiene {arr.size}).")
    return arr


def require_positive_std(arr: np.ndarray, name: str = "la muestra") -> float:
    """Devuelve la desviación típica muestral (ddof=1) y exige que sea > 0."""
    sd = float(np.std(arr, ddof=1))
    if sd == 0:
        raise ValueError(
            f"La desviación típica de {name} es 0 (todos los valores son iguales): "
            "el estadístico no está definido."
        )
    return sd


def check_alternative(alternative: str) -> None:
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")


def check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence debe estar entre 0 y 1 (por ejemplo, 0.95).")
