"""
regression.py

Funciones para el Módulo 5 — Correlación y Regresión Lineal.
Incluye covarianza, correlación de Pearson, ajuste de regresión lineal simple
por mínimos cuadrados ordinarios (OLS), predicción, R², y el contraste de
hipótesis / intervalo de confianza sobre la pendiente (reconectando con el
Módulo 4 de inferencia). Ampliación (fase 3): correlación de Spearman y su
contraste.

Requiere scipy (scipy.stats), ya usada en el Módulo 4.
"""

import math

import numpy as np
from scipy import stats

from ._pvalues import p_value_from_statistic


def _as_arrays(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("x o y contienen valores ausentes (NaN): elimínalos antes del análisis.")
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 pares de observaciones.")
    return x, y


def _require_variation(v: np.ndarray, name: str) -> None:
    if np.all(v == v[0]):
        raise ValueError(f"{name} es constante: la medida no está definida.")


# ---------------------------------------------------------------------------
# Covarianza y correlación
# ---------------------------------------------------------------------------

def covariance(x, y) -> float:
    """Covarianza muestral entre x e y (con corrección de Bessel, n-1)."""
    x, y = _as_arrays(x, y)
    x_mean, y_mean = x.mean(), y.mean()
    return float(np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1))


def correlation(x, y) -> float:
    """Coeficiente de correlación de Pearson entre x e y."""
    x, y = _as_arrays(x, y)
    _require_variation(x, "x")
    _require_variation(y, "y")
    return float(covariance(x, y) / (x.std(ddof=1) * y.std(ddof=1)))


# ---------------------------------------------------------------------------
# Regresión lineal simple (OLS)
# ---------------------------------------------------------------------------

def simple_linear_regression(x, y) -> tuple[float, float]:
    """
    Ajusta Y = beta0 + beta1 * X por mínimos cuadrados ordinarios.

    Devuelve (beta0, beta1).
    """
    x, y = _as_arrays(x, y)
    _require_variation(x, "x")
    beta1 = covariance(x, y) / np.var(x, ddof=1)
    beta0 = y.mean() - beta1 * x.mean()
    return float(beta0), float(beta1)


def predict(x, beta0: float, beta1: float):
    """Predice Y = beta0 + beta1 * X. Acepta un escalar o un array."""
    x = np.asarray(x, dtype=float)
    return beta0 + beta1 * x


def residuals(x, y, beta0: float, beta1: float) -> np.ndarray:
    """Residuos: y_observado - y_predicho."""
    x, y = _as_arrays(x, y)
    return y - predict(x, beta0, beta1)


def r_squared(x, y, beta0: float, beta1: float) -> float:
    """Coeficiente de determinación R^2."""
    x, y = _as_arrays(x, y)
    resid = residuals(x, y, beta0, beta1)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Inferencia sobre la pendiente (conecta con el Módulo 4)
# ---------------------------------------------------------------------------

def standard_error_slope(x, y, beta0: float, beta1: float) -> float:
    """Error estándar del estimador de la pendiente beta1_hat."""
    x, y = _as_arrays(x, y)
    n = len(x)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 observaciones (n - 2 grados de libertad).")
    _require_variation(x, "x")
    resid = residuals(x, y, beta0, beta1)
    # Error estándar residual (grados de libertad: n - 2, por estimar beta0 y beta1)
    s_residual = math.sqrt(np.sum(resid ** 2) / (n - 2))
    ss_x = np.sum((x - x.mean()) ** 2)
    return float(s_residual / math.sqrt(ss_x))


def t_test_slope(
    x, y, beta0: float, beta1: float, alternative: str = "two-sided"
) -> tuple[float, float]:
    """
    Contraste de hipótesis H0: beta1 = 0 vs H1: beta1 != 0 (u otra alternativa).

    Devuelve (t_stat, p_value).
    """
    x, y = _as_arrays(x, y)
    n = len(x)
    se = standard_error_slope(x, y, beta0, beta1)
    t_stat = beta1 / se
    df = n - 2

    p_value = p_value_from_statistic(t_stat, stats.t(df), alternative)

    return float(t_stat), p_value


def confidence_interval_slope(
    x, y, beta0: float, beta1: float, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para la pendiente beta1, usando la distribución t con n-2 grados de libertad."""
    x, y = _as_arrays(x, y)
    n = len(x)
    se = standard_error_slope(x, y, beta0, beta1)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 2)
    margin = t_crit * se
    return float(beta1 - margin), float(beta1 + margin)


# ---------------------------------------------------------------------------
# Ampliación (F3.4): correlación de Spearman
# ---------------------------------------------------------------------------

def spearman_correlation(x, y) -> float:
    """Coeficiente de correlación de Spearman.

    Parameters
    ----------
    x, y : array_like
        Muestras emparejadas, sin NaN y de la misma longitud.

    Returns
    -------
    float
        La correlación de Pearson entre los rangos de ``x`` y de ``y`` (los
        empates reciben el rango medio). Mide la fuerza de una relación
        **monótona**, no necesariamente lineal, y es poco sensible a valores
        atípicos.
    """
    x, y = _as_arrays(x, y)
    return correlation(stats.rankdata(x), stats.rankdata(y))


def spearman_test(x, y, alternative: str = "two-sided") -> tuple[float, float]:
    """Contraste de H0: no hay asociación monótona (rho de Spearman = 0).

    Parameters
    ----------
    x, y : array_like
        Muestras emparejadas, al menos 3 pares.
    alternative : {"two-sided", "greater", "less"}

    Returns
    -------
    tuple of float
        ``(rho, p_value)``. El p-valor usa la aproximación
        ``t = rho·sqrt((n-2)/(1-rho^2))`` con n - 2 grados de libertad (la de
        ``scipy.stats.spearmanr``); es razonable a partir de unos 10 pares.
    """
    x, y = _as_arrays(x, y)
    if len(x) < 3:
        raise ValueError("Se necesitan al menos 3 pares de observaciones.")
    rho = spearman_correlation(x, y)
    n = len(x)
    t_stat = math.copysign(math.inf, rho) if abs(rho) >= 1 else rho * math.sqrt((n - 2) / (1 - rho**2))
    return rho, p_value_from_statistic(t_stat, stats.t(n - 2), alternative)
