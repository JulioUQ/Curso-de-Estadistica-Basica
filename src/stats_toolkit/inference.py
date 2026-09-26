"""
inference.py

Funciones para el Módulo 4 — Inferencia Estadística.
Incluye construcción de intervalos de confianza (media con sigma conocida/
desconocida, proporciones), contrastes de hipótesis (z-test, t-test de una
muestra, z-test de dos proporciones) y una simulación de cobertura de
intervalos de confianza.

Requiere scipy (scipy.stats) para los valores críticos de las distribuciones
Normal y t de Student.
"""

import math

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Error estándar
# ---------------------------------------------------------------------------

def standard_error_mean(sigma: float, n: int) -> float:
    """Error estándar de la media muestral: SE = sigma / sqrt(n)."""
    return sigma / math.sqrt(n)


# ---------------------------------------------------------------------------
# Intervalos de confianza
# ---------------------------------------------------------------------------

def confidence_interval_mean_z(
    sample_mean: float, sigma: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para la media con sigma poblacional CONOCIDA (usa la Normal)."""
    se = standard_error_mean(sigma, n)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_crit * se
    return sample_mean - margin, sample_mean + margin


def confidence_interval_mean_t(
    sample_mean: float, sample_std: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para la media con sigma poblacional DESCONOCIDA (usa la t de Student, n-1 g.l.)."""
    se = sample_std / math.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin = t_crit * se
    return sample_mean - margin, sample_mean + margin


def confidence_interval_proportion(
    p_hat: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para una proporción (aproximación Normal)."""
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_crit * se
    return max(0.0, p_hat - margin), min(1.0, p_hat + margin)


# ---------------------------------------------------------------------------
# Contrastes de hipótesis
# ---------------------------------------------------------------------------

def _p_value_from_stat(stat: float, cdf, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * (1 - cdf(abs(stat)))
    elif alternative == "greater":
        return 1 - cdf(stat)
    elif alternative == "less":
        return cdf(stat)
    raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")


def z_test_mean(
    sample_mean: float, mu0: float, sigma: float, n: int, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Test z de una muestra para la media, con sigma poblacional conocida.

    Devuelve (z_stat, p_value).
    """
    se = standard_error_mean(sigma, n)
    z_stat = (sample_mean - mu0) / se
    p_value = _p_value_from_stat(z_stat, stats.norm.cdf, alternative)
    return z_stat, p_value


def t_test_mean(sample, mu0: float, alternative: str = "two-sided") -> tuple[float, float]:
    """Test t de una muestra para la media, con sigma poblacional desconocida.

    Devuelve (t_stat, p_value).
    """
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)
    se = sample_std / math.sqrt(n)
    t_stat = (sample_mean - mu0) / se
    df = n - 1
    cdf = lambda x: stats.t.cdf(x, df)
    p_value = _p_value_from_stat(t_stat, cdf, alternative)
    return t_stat, p_value


def two_proportion_z_test(
    x1: int, n1: int, x2: int, n2: int, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Test z para comparar dos proporciones independientes (ej. A/B testing).

    Devuelve (z_stat, p_value).
    """
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se
    p_value = _p_value_from_stat(z_stat, stats.norm.cdf, alternative)
    return z_stat, p_value


# ---------------------------------------------------------------------------
# Simulación pedagógica: cobertura de un intervalo de confianza
# ---------------------------------------------------------------------------

def simulate_ci_coverage(
    true_mu: float,
    sigma: float,
    n: int,
    confidence: float = 0.95,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Simula `n_simulations` muestras de tamaño `n` de una población Normal(true_mu, sigma),
    construye un IC para cada una, y devuelve la proporción de intervalos que
    efectivamente contienen `true_mu`.

    Sirve para verificar empíricamente la interpretación correcta del nivel
    de confianza: no es "la probabilidad de que mu esté en ESTE intervalo",
    sino la proporción de intervalos, a la larga, que contienen a mu.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_simulations):
        sample = rng.normal(true_mu, sigma, size=n)
        sample_mean = sample.mean()
        lower, upper = confidence_interval_mean_z(sample_mean, sigma, n, confidence)
        if lower <= true_mu <= upper:
            hits += 1
    return hits / n_simulations
