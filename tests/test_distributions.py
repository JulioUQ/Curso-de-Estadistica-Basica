"""
Tests de ``stats_toolkit.distributions`` (Módulo 3).

Referencias independientes: fórmulas cerradas con fracciones exactas, valores
de tablas, propiedades (la pmf suma 1, E[X] y Var(X) conocidas) y
``scipy.stats``. Las simulaciones se comparan con márgenes basados en el
error estándar teórico.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from stats_toolkit import distributions as dist

# ---------------------------------------------------------------------------
# Discretas
# ---------------------------------------------------------------------------


class TestBernoulli:
    def test_valores(self):
        assert dist.bernoulli_pmf(1, 0.3) == pytest.approx(0.3)
        assert dist.bernoulli_pmf(0, 0.3) == pytest.approx(0.7)

    def test_x_fuera_del_soporte_es_error(self):
        with pytest.raises(ValueError):
            dist.bernoulli_pmf(2, 0.3)


class TestBinomial:
    def test_valor_exacto(self):
        # P(X = 2), X ~ B(5, 1/2) = C(5,2) / 2^5 = 10/32
        assert dist.binomial_pmf(2, 5, 0.5) == pytest.approx(float(Fraction(10, 32)))

    def test_ejemplo_de_la_teoria_frente_a_scipy(self):
        assert dist.binomial_pmf(15, 50, 0.3) == pytest.approx(stats.binom.pmf(15, 50, 0.3))

    def test_pmf_suma_uno_y_media_np(self):
        n, p = 50, 0.3
        pmf = np.array([dist.binomial_pmf(k, n, p) for k in range(n + 1)])
        ks = np.arange(n + 1)
        assert pmf.sum() == pytest.approx(1.0)
        assert (ks * pmf).sum() == pytest.approx(n * p)
        assert (((ks - n * p) ** 2) * pmf).sum() == pytest.approx(n * p * (1 - p))

    @pytest.mark.parametrize("k", [-1, 6])
    def test_k_fuera_de_rango_es_error(self, k):
        with pytest.raises(ValueError):
            dist.binomial_pmf(k, 5, 0.5)


class TestPoisson:
    def test_ejemplo_de_la_teoria(self):
        # P(X = 6), X ~ Poisson(4) = 4^6 e^-4 / 6!
        assert dist.poisson_pmf(6, 4) == pytest.approx(4**6 * math.exp(-4) / 720)

    def test_pmf_suma_uno_media_y_varianza_lambda(self):
        lam = 4.0
        ks = np.arange(0, 60)
        pmf = np.array([dist.poisson_pmf(int(k), lam) for k in ks])
        assert pmf.sum() == pytest.approx(1.0)
        assert (ks * pmf).sum() == pytest.approx(lam)
        assert (((ks - lam) ** 2) * pmf).sum() == pytest.approx(lam)

    def test_k_negativo_es_error(self):
        with pytest.raises(ValueError):
            dist.poisson_pmf(-1, 2.0)

    # Corregido en F2.3: cálculo en escala logarítmica.
    @pytest.mark.parametrize("k,lam", [(200, 150.0), (1000, 900.0), (0, 0.5), (3, 0.0)])
    def test_k_grande_no_desborda(self, k, lam):
        assert dist.poisson_pmf(k, lam) == pytest.approx(stats.poisson.pmf(k, lam), rel=1e-10)

    def test_lambda_negativa_es_error(self):
        with pytest.raises(ValueError):
            dist.poisson_pmf(1, -1.0)


# ---------------------------------------------------------------------------
# Continuas
# ---------------------------------------------------------------------------


class TestUniform:
    def test_densidad_dentro_y_fuera(self):
        assert dist.uniform_pdf(0.5, 0, 2) == pytest.approx(0.5)
        assert dist.uniform_pdf(0, 0, 2) == pytest.approx(0.5)  # extremos incluidos
        assert dist.uniform_pdf(2, 0, 2) == pytest.approx(0.5)
        assert dist.uniform_pdf(-0.1, 0, 2) == 0.0
        assert dist.uniform_pdf(2.1, 0, 2) == 0.0


class TestNormal:
    def test_densidad_en_la_media(self):
        assert dist.normal_pdf(3, mu=3, sigma=2) == pytest.approx(1 / (2 * math.sqrt(2 * math.pi)))

    @pytest.mark.parametrize("x", [-2.5, -1.0, 0.0, 0.7, 3.1])
    def test_densidad_frente_a_scipy(self, x):
        assert dist.normal_pdf(x, mu=0.5, sigma=1.5) == pytest.approx(
            stats.norm.pdf(x, loc=0.5, scale=1.5)
        )

    def test_cdf_valores_de_tabla(self):
        assert dist.normal_cdf(0) == pytest.approx(0.5)
        assert dist.normal_cdf(1.96) == pytest.approx(0.9750, abs=5e-5)
        assert dist.normal_cdf(1.645) == pytest.approx(0.9500, abs=5e-5)

    @pytest.mark.parametrize("k,expected", [(1, 0.6827), (2, 0.9545), (3, 0.9973)])
    def test_regla_68_95_99_7(self, k, expected):
        mu, sigma = 10, 3
        p = dist.normal_cdf(mu + k * sigma, mu, sigma) - dist.normal_cdf(mu - k * sigma, mu, sigma)
        assert p == pytest.approx(expected, abs=5e-5)

    @pytest.mark.parametrize("x", [-5.0, -9.0, -10.0, -20.0, -37.0])
    def test_cdf_precisa_en_la_cola_izquierda(self, x):
        # Con 0.5 * (1 + erf(...)) estas probabilidades se perdían (desde -9, daban 0).
        ref = stats.norm.cdf(x)
        assert ref > 0
        assert dist.normal_cdf(x) == pytest.approx(ref, rel=1e-10, abs=0)

    def test_cdf_menos_diez_con_media_y_desviacion(self):
        # P(X <= mu - 10 sigma) = P(Z <= -10) ≈ 7,62e-24
        assert dist.normal_cdf(-25, mu=5, sigma=3) == pytest.approx(stats.norm.cdf(-10), rel=1e-10, abs=0)

    @pytest.mark.parametrize("x", [0.5, 2.0, 8.0])
    def test_cdf_simetrica(self, x):
        assert dist.normal_cdf(x) + dist.normal_cdf(-x) == pytest.approx(1.0, abs=1e-15)


class TestExponential:
    LAM = 4.0

    def test_densidad_y_cdf_formula_cerrada(self):
        x = 0.3
        assert dist.exponential_pdf(x, self.LAM) == pytest.approx(self.LAM * math.exp(-self.LAM * x))
        assert dist.exponential_cdf(x, self.LAM) == pytest.approx(1 - math.exp(-self.LAM * x))

    def test_frente_a_scipy(self):
        x = 0.8
        assert dist.exponential_cdf(x, self.LAM) == pytest.approx(
            stats.expon.cdf(x, scale=1 / self.LAM)
        )

    def test_mediana_es_ln2_entre_lambda(self):
        assert dist.exponential_cdf(math.log(2) / self.LAM, self.LAM) == pytest.approx(0.5)

    def test_cero_para_x_negativo(self):
        assert dist.exponential_pdf(-1, self.LAM) == 0.0
        assert dist.exponential_cdf(-1, self.LAM) == 0.0

    def test_cdf_en_cero_es_cero_positivo(self):
        for x in (0, 0.0):
            value = dist.exponential_cdf(x, self.LAM)
            assert value == 0.0
            assert math.copysign(1.0, value) == 1.0  # no -0.0

    @pytest.mark.parametrize("x", [1e-8, 1e-12, 1e-17, 1e-300])
    def test_cdf_precisa_con_x_muy_pequeno(self, x):
        # Con 1 - exp(-lambda x) se perdían cifras (y con x = 1e-17, daba 0).
        ref = stats.expon.cdf(x, scale=1 / self.LAM)
        assert ref > 0
        assert dist.exponential_cdf(x, self.LAM) == pytest.approx(ref, rel=1e-12, abs=0)


# ---------------------------------------------------------------------------
# Esperanza y varianza discretas
# ---------------------------------------------------------------------------


class TestDiscreteMoments:
    def test_dado_equilibrado(self):
        values, probs = range(1, 7), [1 / 6] * 6
        assert dist.expected_value_discrete(values, probs) == pytest.approx(7 / 2)
        assert dist.variance_discrete(values, probs) == pytest.approx(35 / 12)


# ---------------------------------------------------------------------------
# Teorema Central del Límite
# ---------------------------------------------------------------------------


class TestSimulateCLT:
    N_SAMPLES = 20_000

    @pytest.mark.parametrize(
        "population,params,mu,sigma",
        [
            ("exponential", {"lam": 0.5}, 2.0, 2.0),
            ("uniform", {"a": 0, "b": 1}, 0.5, math.sqrt(1 / 12)),
            ("binomial", {"n": 10, "p": 0.3}, 3.0, math.sqrt(10 * 0.3 * 0.7)),
            ("poisson", {"lam": 3}, 3.0, math.sqrt(3)),
        ],
    )
    def test_media_y_error_estandar_de_las_medias(self, population, params, mu, sigma):
        n = 30
        means = dist.simulate_clt(population, params, sample_size=n, n_samples=self.N_SAMPLES, seed=0)
        se_mean = sigma / math.sqrt(n)
        assert means.shape == (self.N_SAMPLES,)
        # E[X̄] = mu, con margen de 4 errores estándar de la media de las medias
        assert abs(means.mean() - mu) <= 4 * se_mean / math.sqrt(self.N_SAMPLES)
        # SD(X̄) = sigma / sqrt(n), con un margen relativo del 3 %
        assert means.std(ddof=1) == pytest.approx(se_mean, rel=0.03)

    def test_reproducible_con_semilla(self):
        a = dist.simulate_clt("exponential", {"lam": 1}, 10, 100, seed=5)
        b = dist.simulate_clt("exponential", {"lam": 1}, 10, 100, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_poblacion_no_soportada_es_error(self):
        with pytest.raises(ValueError):
            dist.simulate_clt("cauchy", {}, 10, 10)


class TestBinomialGrande:
    # F2.3: con n grande, C(n, k) no cabe en un float; se calcula en logaritmos.
    @pytest.mark.parametrize("k,n,p", [(1000, 2000, 0.5), (15, 50, 0.3), (0, 10, 0.0), (10, 10, 1.0)])
    def test_frente_a_scipy(self, k, n, p):
        assert dist.binomial_pmf(k, n, p) == pytest.approx(stats.binom.pmf(k, n, p), rel=1e-10)

    def test_p_fuera_de_rango_es_error(self):
        with pytest.raises(ValueError):
            dist.binomial_pmf(1, 5, 1.2)
