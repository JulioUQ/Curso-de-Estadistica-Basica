"""
distributions.py

Funciones para el Módulo 3 — Distribuciones de Probabilidad.
Incluye pmf/pdf/cdf de las distribuciones discretas y continuas más comunes,
utilidades de esperanza/varianza para variables discretas, y una simulación
del Teorema Central del Límite.
"""

from math import comb, exp, factorial, sqrt, pi, erf

import numpy as np


# ---------------------------------------------------------------------------
# Distribuciones discretas
# ---------------------------------------------------------------------------

def bernoulli_pmf(x: int, p: float) -> float:
    """P(X = x) para X ~ Bernoulli(p), x en {0, 1}."""
    if x not in (0, 1):
        raise ValueError("x debe ser 0 o 1 para una distribución de Bernoulli")
    return p if x == 1 else 1 - p


def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) para X ~ Binomial(n, p)."""
    if not (0 <= k <= n):
        raise ValueError("k debe estar entre 0 y n")
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) para X ~ Poisson(lambda)."""
    if k < 0:
        raise ValueError("k debe ser un entero no negativo")
    return (lam ** k) * exp(-lam) / factorial(k)


# ---------------------------------------------------------------------------
# Distribuciones continuas
# ---------------------------------------------------------------------------

def uniform_pdf(x: float, a: float, b: float) -> float:
    """f(x) para X ~ Uniforme(a, b)."""
    return 1 / (b - a) if a <= x <= b else 0.0


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """f(x) para X ~ Normal(mu, sigma^2)."""
    return (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """P(X <= x) para X ~ Normal(mu, sigma^2), usando la función error."""
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))


def exponential_pdf(x: float, lam: float) -> float:
    """f(x) para X ~ Exponencial(lambda)."""
    return lam * exp(-lam * x) if x >= 0 else 0.0


def exponential_cdf(x: float, lam: float) -> float:
    """P(X <= x) para X ~ Exponencial(lambda)."""
    return 1 - exp(-lam * x) if x >= 0 else 0.0


# ---------------------------------------------------------------------------
# Esperanza y varianza para variables discretas genéricas
# ---------------------------------------------------------------------------

def expected_value_discrete(values, probabilities) -> float:
    """E[X] = sum(x_i * P(X = x_i)) para una variable discreta genérica."""
    return sum(v * p for v, p in zip(values, probabilities))


def variance_discrete(values, probabilities) -> float:
    """Var(X) = E[(X - mu)^2] para una variable discreta genérica."""
    mu = expected_value_discrete(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ---------------------------------------------------------------------------
# Teorema Central del Límite
# ---------------------------------------------------------------------------

def simulate_clt(
    population: str,
    population_params: dict,
    sample_size: int,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula el Teorema Central del Límite.

    Extrae `n_samples` muestras de tamaño `sample_size` de la población
    indicada, y devuelve un array de numpy con la media de cada muestra.

    Parameters
    ----------
    population : {"exponential", "uniform", "binomial", "poisson"}
    population_params : dict
        Parámetros de la población. Ejemplos:
        - exponential: {"lam": 0.5}
        - uniform: {"a": 0, "b": 1}
        - binomial: {"n": 10, "p": 0.5}
        - poisson: {"lam": 3}
    sample_size : int
        Tamaño de cada muestra individual.
    n_samples : int
        Número de muestras a simular.
    seed : int, optional
        Semilla para reproducibilidad.
    """
    rng = np.random.default_rng(seed)

    generators = {
        "exponential": lambda size: rng.exponential(
            scale=1 / population_params.get("lam", 1), size=size
        ),
        "uniform": lambda size: rng.uniform(
            population_params.get("a", 0), population_params.get("b", 1), size=size
        ),
        "binomial": lambda size: rng.binomial(
            population_params.get("n", 10), population_params.get("p", 0.5), size=size
        ),
        "poisson": lambda size: rng.poisson(
            population_params.get("lam", 1), size=size
        ),
    }

    if population not in generators:
        raise ValueError(
            f"Población no soportada: '{population}'. "
            f"Opciones válidas: {list(generators.keys())}"
        )

    generate_sample = generators[population]
    means = np.empty(n_samples)
    for i in range(n_samples):
        sample = generate_sample(sample_size)
        means[i] = sample.mean()

    return means
