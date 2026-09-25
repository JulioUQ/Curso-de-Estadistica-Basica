"""
descriptive.py
---------------
Funciones modulares de estadística descriptiva para el Módulo 1
del repositorio "Estadística Básica".

Todas las funciones aceptan listas, arrays de NumPy o Series de pandas,
y devuelven resultados en diccionarios simples para facilitar su uso
tanto en scripts como en notebooks.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from scipy import stats as sp_stats

ArrayLike = Union[Iterable[float], np.ndarray]


def _to_array(data: ArrayLike) -> np.ndarray:
    """Convierte cualquier ArrayLike a un np.ndarray de floats, sin NaNs."""
    arr = np.asarray(data, dtype=float)
    return arr[~np.isnan(arr)]


def central_tendency(data: ArrayLike) -> dict:
    """
    Calcula media, mediana y moda de un conjunto de datos numéricos.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos (lista, array o Serie de pandas).

    Returns
    -------
    dict
        {'mean': float, 'median': float, 'mode': float}
    """
    arr = _to_array(data)
    mode_result = sp_stats.mode(arr, keepdims=False)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "mode": float(mode_result.mode),
    }


def dispersion(data: ArrayLike, ddof: int = 1) -> dict:
    """
    Calcula rango, varianza, desviación estándar y coeficiente de
    variación de un conjunto de datos numéricos.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos.
    ddof : int, default 1
        Grados de libertad. Usa 1 para varianza/desviación muestral
        (corrección de Bessel) o 0 para la versión poblacional.

    Returns
    -------
    dict
        {'range': float, 'variance': float, 'std': float, 'cv_pct': float}
    """
    arr = _to_array(data)
    variance = float(np.var(arr, ddof=ddof))
    std = float(np.std(arr, ddof=ddof))
    mean = float(np.mean(arr))
    cv_pct = (std / mean) * 100 if mean != 0 else float("nan")

    return {
        "range": float(np.ptp(arr)),
        "variance": variance,
        "std": std,
        "cv_pct": cv_pct,
    }


def shape(data: ArrayLike) -> dict:
    """
    Calcula asimetría (skewness) y curtosis de exceso de un conjunto
    de datos numéricos.

    Returns
    -------
    dict
        {'skewness': float, 'kurtosis_excess': float}
    """
    arr = _to_array(data)
    return {
        "skewness": float(sp_stats.skew(arr)),
        "kurtosis_excess": float(sp_stats.kurtosis(arr, fisher=True)),
    }


def five_number_summary(data: ArrayLike) -> dict:
    """
    Calcula el resumen de cinco números de Tukey (mínimo, Q1, mediana,
    Q3, máximo) y el rango intercuartílico (IQR).

    Returns
    -------
    dict
        {'min': float, 'q1': float, 'median': float, 'q3': float,
         'max': float, 'iqr': float}
    """
    arr = _to_array(data)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(arr)),
        "iqr": float(q3 - q1),
    }


def detect_outliers(data: ArrayLike) -> np.ndarray:
    """
    Detecta outliers usando la regla de Tukey (1.5 * IQR).

    Returns
    -------
    np.ndarray
        Subconjunto de `data` considerado atípico.
    """
    arr = _to_array(data)
    summary = five_number_summary(arr)
    lower_bound = summary["q1"] - 1.5 * summary["iqr"]
    upper_bound = summary["q3"] + 1.5 * summary["iqr"]
    return arr[(arr < lower_bound) | (arr > upper_bound)]


def summary_statistics(data: ArrayLike, ddof: int = 1) -> dict:
    """
    Combina central_tendency, dispersion, shape y five_number_summary
    en un único diccionario de resumen completo.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos.
    ddof : int, default 1
        Grados de libertad para varianza/desviación estándar.

    Returns
    -------
    dict
        Diccionario anidado con todas las métricas descriptivas.
    """
    arr = _to_array(data)
    return {
        "n": int(arr.size),
        "central_tendency": central_tendency(arr),
        "dispersion": dispersion(arr, ddof=ddof),
        "shape": shape(arr),
        "five_number_summary": five_number_summary(arr),
    }


if __name__ == "__main__":
    # Ejemplo de uso rápido con datos simulados de salarios
    np.random.seed(42)
    salarios_ejemplo = np.random.lognormal(mean=7.5, sigma=0.4, size=200)

    resumen = summary_statistics(salarios_ejemplo)

    print("== Resumen estadístico de ejemplo (salarios simulados) ==")
    for seccion, valores in resumen.items():
        print(f"\n{seccion}:")
        if isinstance(valores, dict):
            for k, v in valores.items():
                print(f"  {k}: {v:.2f}")
        else:
            print(f"  {valores}")
