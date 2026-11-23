"""
inference.py

Funciones para el Módulo 4 — Inferencia Estadística.
Incluye construcción de intervalos de confianza (media con sigma conocida/
desconocida, proporciones), contrastes de hipótesis (z-test, t-test de una
muestra, z-test de dos proporciones) y una simulación de cobertura de
intervalos de confianza.
Ampliación (fase 3): potencia y tamaño muestral del test z, tamaño muestral
para intervalos e intervalos bootstrap por percentiles.

Requiere scipy (scipy.stats) para los valores críticos de las distribuciones
Normal y t de Student.
"""

import math

import numpy as np
from scipy import stats

from ._pvalues import p_value_from_statistic
from ._validation import as_sample, check_alternative, check_confidence, require_positive_std


# ---------------------------------------------------------------------------
# Error estándar
# ---------------------------------------------------------------------------

def standard_error_mean(sigma: float, n: int) -> float:
    """Error estándar de la media muestral: SE = sigma / sqrt(n)."""
    if sigma <= 0:
        raise ValueError("sigma debe ser positiva.")
    if n < 1:
        raise ValueError("n debe ser al menos 1.")
    return sigma / math.sqrt(n)


# ---------------------------------------------------------------------------
# Intervalos de confianza
# ---------------------------------------------------------------------------

def confidence_interval_mean_z(
    sample_mean: float, sigma: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para la media con sigma poblacional CONOCIDA (usa la Normal)."""
    check_confidence(confidence)
    se = standard_error_mean(sigma, n)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_crit * se
    return sample_mean - margin, sample_mean + margin


def confidence_interval_mean_t(
    sample_mean: float, sample_std: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para la media con sigma poblacional DESCONOCIDA (usa la t de Student, n-1 g.l.)."""
    check_confidence(confidence)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 observaciones (n >= 2).")
    se = sample_std / math.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin = t_crit * se
    return sample_mean - margin, sample_mean + margin


def confidence_interval_proportion(
    p_hat: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC para una proporción (aproximación Normal, intervalo de Wald).

    Es deficiente con n pequeño o p_hat cerca de 0 o 1 (con p_hat = 0 o 1 da un
    intervalo de anchura 0). En esos casos usa
    ``confidence_interval_proportion_wilson``.
    """
    check_confidence(confidence)
    if not 0 <= p_hat <= 1:
        raise ValueError("p_hat debe estar entre 0 y 1.")
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_crit * se
    return max(0.0, p_hat - margin), min(1.0, p_hat + margin)


# ---------------------------------------------------------------------------
# Contrastes de hipótesis
# ---------------------------------------------------------------------------

def z_test_mean(
    sample_mean: float, mu0: float, sigma: float, n: int, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Test z de una muestra para la media, con sigma poblacional conocida.

    Devuelve (z_stat, p_value).
    """
    se = standard_error_mean(sigma, n)
    z_stat = (sample_mean - mu0) / se
    p_value = p_value_from_statistic(z_stat, stats.norm(), alternative)
    return float(z_stat), p_value


def t_test_mean(sample, mu0: float, alternative: str = "two-sided") -> tuple[float, float]:
    """Test t de una muestra para la media, con sigma poblacional desconocida.

    Devuelve (t_stat, p_value).
    """
    check_alternative(alternative)
    sample = as_sample(sample)
    n = len(sample)
    sample_mean = sample.mean()
    sample_std = require_positive_std(sample)
    se = sample_std / math.sqrt(n)
    t_stat = (sample_mean - mu0) / se
    df = n - 1
    p_value = p_value_from_statistic(t_stat, stats.t(df), alternative)
    return float(t_stat), p_value


def two_proportion_z_test(
    x1: int, n1: int, x2: int, n2: int, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Test z para comparar dos proporciones independientes (ej. A/B testing).

    Devuelve (z_stat, p_value).
    """
    if not (0 <= x1 <= n1 and 0 <= x2 <= n2):
        raise ValueError("Los éxitos deben estar entre 0 y el tamaño de cada grupo.")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool in (0, 1):
        raise ValueError(
            "La proporción agrupada es 0 o 1 (todos fracasos o todos éxitos): "
            "el estadístico z no está definido."
        )
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se
    p_value = p_value_from_statistic(z_stat, stats.norm(), alternative)
    return float(z_stat), p_value


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


# ---------------------------------------------------------------------------
# Ampliación (F2.2): lo que enseña la teoría del Módulo 4
# ---------------------------------------------------------------------------

def confidence_interval_proportion_wilson(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC de Wilson para una proporción.

    Invierte el contraste z con la varianza calculada bajo cada valor candidato
    de p, en lugar de con p_hat. Se comporta mucho mejor que el intervalo de
    Wald con muestras pequeñas o proporciones cercanas a 0 o 1, y nunca se sale
    de [0, 1].

    Devuelve (límite inferior, límite superior).
    """
    check_confidence(confidence)
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Se necesita n >= 1 y 0 <= successes <= n.")
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def one_proportion_z_test(
    successes: int, n: int, p0: float, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Test z para una proporción, H0: p = p0 (aproximación Normal).

    El error estándar se calcula con p0, el valor supuesto por H0:
    z = (p_hat - p0) / sqrt(p0 (1 - p0) / n).

    Devuelve (z_stat, p_value).
    """
    check_alternative(alternative)
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Se necesita n >= 1 y 0 <= successes <= n.")
    if not 0 < p0 < 1:
        raise ValueError("p0 debe estar estrictamente entre 0 y 1.")
    z_stat = (successes / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return float(z_stat), p_value_from_statistic(z_stat, stats.norm(), alternative)


def welch_t_test(x, y, alternative: str = "two-sided") -> tuple[float, float, float]:
    """Test t de Welch para dos medias independientes (no supone varianzas iguales).

    t = (media_x - media_y) / sqrt(s_x^2/n_x + s_y^2/n_y), con los grados de
    libertad aproximados por Welch-Satterthwaite.

    Devuelve (t_stat, p_value, df).
    """
    check_alternative(alternative)
    x, y = as_sample(x, "x"), as_sample(y, "y")
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    if vx + vy == 0:
        raise ValueError("Ambas muestras son constantes: el estadístico no está definido.")
    t_stat = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1))
    p = p_value_from_statistic(t_stat, stats.t(df), alternative)
    return float(t_stat), p, float(df)


def paired_t_test(x, y, alternative: str = "two-sided") -> tuple[float, float]:
    """Test t para datos emparejados: contrasta H0: media de D = 0, con D = x - y.

    Equivale a un test t de una muestra sobre las diferencias (n - 1 g.l.).

    Devuelve (t_stat, p_value).
    """
    check_alternative(alternative)
    x, y = as_sample(x, "x"), as_sample(y, "y")
    if x.size != y.size:
        raise ValueError("x e y deben tener la misma longitud (una pareja por observación).")
    d = x - y
    sd = require_positive_std(d, "las diferencias")
    t_stat = d.mean() / (sd / math.sqrt(d.size))
    return float(t_stat), p_value_from_statistic(t_stat, stats.t(d.size - 1), alternative)


def chi2_test_variance(sample, sigma0_sq: float, alternative: str = "two-sided") -> tuple[float, float]:
    """Contraste de la varianza de una población Normal, H0: sigma^2 = sigma0^2.

    chi2 = (n - 1) s^2 / sigma0^2, con n - 1 grados de libertad. Es muy sensible
    a la falta de normalidad. El p-valor bilateral es 2 · min(cola izquierda,
    cola derecha), recortado a 1.

    Devuelve (chi2_stat, p_value).
    """
    check_alternative(alternative)
    if sigma0_sq <= 0:
        raise ValueError("sigma0_sq debe ser positiva.")
    x = as_sample(sample)
    df = x.size - 1
    chi2 = df * x.var(ddof=1) / sigma0_sq
    dist = stats.chi2(df)
    if alternative == "greater":
        p = dist.sf(chi2)
    elif alternative == "less":
        p = dist.cdf(chi2)
    else:
        p = min(1.0, 2 * min(dist.cdf(chi2), dist.sf(chi2)))
    return float(chi2), float(p)


# ---------------------------------------------------------------------------
# Ampliación (F3.2): potencia y tamaño muestral
# ---------------------------------------------------------------------------

def _z_crit(alpha: float, alternative: str) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1 (por ejemplo, 0.05).")
    check_alternative(alternative)
    return float(stats.norm.isf(alpha / 2 if alternative == "two-sided" else alpha))


def power_z_test_mean(
    effect: float, sigma: float, n: int, alpha: float = 0.05, alternative: str = "two-sided"
) -> float:
    """Potencia del test z para una media con sigma conocida.

    Parameters
    ----------
    effect : float
        Diferencia real supuesta bajo H1, ``mu1 - mu0``.
    sigma : float
        Desviación típica poblacional (conocida), > 0.
    n : int
        Tamaño muestral.
    alpha : float
        Nivel de significación.
    alternative : {"two-sided", "greater", "less"}
        Hipótesis alternativa del contraste.

    Returns
    -------
    float
        Probabilidad de rechazar H0 cuando la media real es ``mu0 + effect``.
        Con ``effect = 0`` vale ``alpha``. Con d = effect·sqrt(n)/sigma:
        derecha ``P(Z > z_{1-α} - d)``, izquierda ``P(Z < -z_{1-α} - d)`` y
        bilateral la suma de ambas colas con ``z_{1-α/2}``.
    """
    z = _z_crit(alpha, alternative)
    d = effect / standard_error_mean(sigma, n)
    norm = stats.norm()
    if alternative == "greater":
        return float(norm.sf(z - d))
    if alternative == "less":
        return float(norm.cdf(-z - d))
    return float(norm.sf(z - d) + norm.cdf(-z - d))


def _ceil(x: float) -> int:
    # Redondeo hacia arriba sin que el error de coma flotante añada una unidad
    # (por ejemplo, 96.00000000000001 debe dar 96).
    return int(math.ceil(round(x, 9)))


def sample_size_z_test_mean(
    effect: float, sigma: float, alpha: float = 0.05, power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """Tamaño muestral mínimo del test z para una media (sigma conocida).

    Parameters
    ----------
    effect : float
        Diferencia mínima que se quiere detectar, ``mu1 - mu0`` (distinta de 0;
        con alternativa unilateral, con el signo de esa alternativa).
    sigma : float
        Desviación típica poblacional supuesta, > 0.
    alpha : float
        Nivel de significación.
    power : float
        Potencia deseada, entre ``alpha`` y 1 (habitualmente 0,80 o 0,90).
    alternative : {"two-sided", "greater", "less"}

    Returns
    -------
    int
        ``n = ceil(((z_crit + z_{1-β}) · sigma / effect)^2)``. En el caso
        bilateral la fórmula ignora la cola opuesta, que solo suma potencia: la
        potencia real con ese n es siempre al menos la pedida.
    """
    z = _z_crit(alpha, alternative)
    if not alpha < power < 1:
        raise ValueError("power debe estar entre alpha y 1 (por ejemplo, 0.8).")
    if effect == 0:
        raise ValueError("effect no puede ser 0: no hay diferencia que detectar.")
    if (alternative == "greater" and effect < 0) or (alternative == "less" and effect > 0):
        raise ValueError("El signo de effect no coincide con la hipótesis alternativa.")
    if sigma <= 0:
        raise ValueError("sigma debe ser positiva.")
    z_beta = float(stats.norm.ppf(power))
    return max(1, _ceil(((z + z_beta) * sigma / effect) ** 2))


def sample_size_mean_ci(margin: float, sigma: float, confidence: float = 0.95) -> int:
    """Tamaño muestral para estimar una media con un margen de error dado.

    Parameters
    ----------
    margin : float
        Semiamplitud máxima deseada del intervalo (margen de error), > 0.
    sigma : float
        Desviación típica poblacional supuesta (de un estudio previo o piloto).
    confidence : float
        Nivel de confianza.

    Returns
    -------
    int
        ``n = ceil((z_{1-α/2} · sigma / margin)^2)``.
    """
    check_confidence(confidence)
    if margin <= 0 or sigma <= 0:
        raise ValueError("margin y sigma deben ser positivos.")
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    return max(1, _ceil((z * sigma / margin) ** 2))


def sample_size_proportion_ci(margin: float, p: float = 0.5, confidence: float = 0.95) -> int:
    """Tamaño muestral para estimar una proporción con un margen de error dado.

    Parameters
    ----------
    margin : float
        Margen de error deseado, entre 0 y 1 (por ejemplo, 0.03 = 3 puntos).
    p : float
        Proporción supuesta. Con ``p = 0.5`` (valor por defecto) se obtiene el
        tamaño más conservador, porque ``p(1 - p)`` es máximo.
    confidence : float
        Nivel de confianza.

    Returns
    -------
    int
        ``n = ceil(z_{1-α/2}^2 · p(1 - p) / margin^2)`` (aproximación Normal,
        la del intervalo de Wald).
    """
    check_confidence(confidence)
    if not 0 < margin < 1:
        raise ValueError("margin debe estar entre 0 y 1.")
    if not 0 < p < 1:
        raise ValueError("p debe estar estrictamente entre 0 y 1.")
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    return max(1, _ceil(z**2 * p * (1 - p) / margin**2))


# ---------------------------------------------------------------------------
# Ampliación (F3.3): bootstrap
# ---------------------------------------------------------------------------

def bootstrap_distribution(
    data, statistic=np.mean, n_resamples: int = 10_000, seed: int | None = None
) -> np.ndarray:
    """Distribución bootstrap de un estadístico.

    Parameters
    ----------
    data : array_like
        Muestra observada (sin NaN, al menos 2 valores).
    statistic : callable
        Función que acepta un array 2D y el argumento ``axis`` (como
        ``np.mean``, ``np.median`` o ``np.std``); se aplica a cada remuestra.
    n_resamples : int
        Número de remuestras con reemplazamiento del mismo tamaño que ``data``.
    seed : int, optional
        Semilla de ``np.random.default_rng`` (reproducibilidad).

    Returns
    -------
    numpy.ndarray
        Los ``n_resamples`` valores del estadístico.
    """
    x = as_sample(data, "data")
    if n_resamples < 1:
        raise ValueError("n_resamples debe ser al menos 1.")
    rng = np.random.default_rng(seed)
    resamples = x[rng.integers(0, x.size, size=(n_resamples, x.size))]
    return np.asarray(statistic(resamples, axis=1), dtype=float)


def bootstrap_ci(
    data, statistic=np.mean, confidence: float = 0.95, n_resamples: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Intervalo de confianza bootstrap por el método de los percentiles.

    Parameters
    ----------
    data, statistic, n_resamples, seed
        Como en :func:`bootstrap_distribution`.
    confidence : float
        Nivel de confianza.

    Returns
    -------
    tuple of float
        Percentiles ``α/2`` y ``1 - α/2`` de la distribución bootstrap
        (interpolación lineal, como ``np.percentile``). Es un método
        aproximado: con muestras pequeñas o estadísticos sesgados su cobertura
        real puede quedar por debajo de la nominal.
    """
    check_confidence(confidence)
    boot = bootstrap_distribution(data, statistic, n_resamples, seed)
    alpha = 1 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
