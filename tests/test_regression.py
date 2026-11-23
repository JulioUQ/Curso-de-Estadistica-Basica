"""
Tests de ``stats_toolkit.regression`` (Módulo 5).

Referencias independientes: ``numpy.cov``, ``scipy.stats.pearsonr`` y
``linregress``, el OLS de ``statsmodels`` y propiedades algebraicas de los
mínimos cuadrados (ecuaciones normales, R² = r² en regresión simple).
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from stats_toolkit import regression as reg

# Datos de la teoría del M5
HORAS = np.array([2, 3, 4, 5, 5, 6, 7, 8, 8, 9], dtype=float)
NOTA = np.array([4.5, 5.2, 5.8, 6.5, 6.9, 7.2, 7.8, 8.4, 8.6, 9.1])

# Datos con ruido apreciable (relación no perfecta)
_rng = np.random.default_rng(3)
X = _rng.uniform(5, 50, size=20)
Y = 20 + 3.5 * X + _rng.normal(0, 25, size=20)


@pytest.fixture(scope="module")
def ols():
    return sm.OLS(Y, sm.add_constant(X)).fit()


# ---------------------------------------------------------------------------
# Covarianza y correlación
# ---------------------------------------------------------------------------


class TestCovarianceCorrelation:
    def test_covarianza_muestral_frente_a_numpy(self):
        assert reg.covariance(X, Y) == pytest.approx(np.cov(X, Y, ddof=1)[0, 1])

    def test_correlacion_frente_a_pearsonr(self):
        assert reg.correlation(HORAS, NOTA) == pytest.approx(stats.pearsonr(HORAS, NOTA).statistic)

    def test_relacion_lineal_exacta_da_r_igual_a_mas_menos_uno(self):
        x = np.arange(10.0)
        assert reg.correlation(x, 3 * x + 1) == pytest.approx(1.0)
        assert reg.correlation(x, -2 * x + 5) == pytest.approx(-1.0)

    def test_r_invariante_a_cambios_lineales_positivos(self):
        assert reg.correlation(10 * X + 7, Y) == pytest.approx(reg.correlation(X, Y))

    def test_longitudes_distintas_es_error(self):
        with pytest.raises(ValueError):
            reg.covariance([1, 2, 3], [1, 2])

    # F2.3: x o y constantes, NaN o muy pocas observaciones dan un error claro.
    def test_x_constante_es_error(self):
        with pytest.raises(ValueError):
            reg.correlation([2, 2, 2], [1, 2, 3])
        with pytest.raises(ValueError):
            reg.simple_linear_regression([2, 2, 2], [1, 2, 3])

    def test_nan_es_error(self):
        with pytest.raises(ValueError):
            reg.correlation([1, 2, np.nan], [1, 2, 3])

    def test_pendiente_necesita_tres_puntos(self):
        with pytest.raises(ValueError):
            reg.t_test_slope([1, 2], [1, 3], 0.0, 2.0)


# ---------------------------------------------------------------------------
# Regresión lineal simple
# ---------------------------------------------------------------------------


class TestSimpleLinearRegression:
    def test_coeficientes_frente_a_linregress(self):
        b0, b1 = reg.simple_linear_regression(X, Y)
        ref = stats.linregress(X, Y)
        assert (b0, b1) == pytest.approx((ref.intercept, ref.slope))

    def test_recupera_una_recta_exacta(self):
        x = np.array([0.0, 1.0, 2.0, 5.0])
        assert reg.simple_linear_regression(x, 2 + 3 * x) == pytest.approx((2.0, 3.0))

    def test_prediccion_escalar_y_vector(self):
        assert reg.predict(2.0, 1.0, 0.5) == pytest.approx(2.0)
        np.testing.assert_allclose(reg.predict([0, 4], 1.0, 0.5), [1.0, 3.0])

    def test_residuos_cumplen_las_ecuaciones_normales(self):
        b0, b1 = reg.simple_linear_regression(X, Y)
        e = reg.residuals(X, Y, b0, b1)
        assert e.sum() == pytest.approx(0.0, abs=1e-8)
        assert (e * X).sum() == pytest.approx(0.0, abs=1e-6)

    def test_r2_igual_a_r_al_cuadrado_y_a_statsmodels(self, ols):
        b0, b1 = reg.simple_linear_regression(X, Y)
        r2 = reg.r_squared(X, Y, b0, b1)
        assert r2 == pytest.approx(reg.correlation(X, Y) ** 2)
        assert r2 == pytest.approx(ols.rsquared)

    def test_r2_vale_uno_con_ajuste_perfecto(self):
        x = np.arange(5.0)
        assert reg.r_squared(x, 1 + 2 * x, 1.0, 2.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Inferencia sobre la pendiente
# ---------------------------------------------------------------------------


class TestSlopeInference:
    def test_error_estandar_frente_a_linregress(self):
        b0, b1 = reg.simple_linear_regression(X, Y)
        assert reg.standard_error_slope(X, Y, b0, b1) == pytest.approx(stats.linregress(X, Y).stderr)

    def test_t_y_p_bilaterales_frente_a_statsmodels(self, ols):
        b0, b1 = reg.simple_linear_regression(X, Y)
        t, p = reg.t_test_slope(X, Y, b0, b1)
        assert t == pytest.approx(ols.tvalues[1])
        assert p == pytest.approx(ols.pvalues[1])

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_p_unilateral_frente_a_linregress(self, alternative):
        b0, b1 = reg.simple_linear_regression(X, Y)
        _, p = reg.t_test_slope(X, Y, b0, b1, alternative=alternative)
        assert p == pytest.approx(stats.linregress(X, Y, alternative=alternative).pvalue)

    def test_alternativa_invalida_es_error(self):
        b0, b1 = reg.simple_linear_regression(X, Y)
        with pytest.raises(ValueError):
            reg.t_test_slope(X, Y, b0, b1, alternative="distinta")

    def test_ic_de_la_pendiente_frente_a_statsmodels(self, ols):
        b0, b1 = reg.simple_linear_regression(X, Y)
        low, up = reg.confidence_interval_slope(X, Y, b0, b1, confidence=0.95)
        assert (low, up) == pytest.approx(tuple(ols.conf_int(alpha=0.05)[1]))

    def test_p_valor_extremo_no_se_anula(self):
        x = np.arange(30, dtype=float)
        y = 1 + 2 * x + 0.01 * np.random.default_rng(1).standard_normal(30)
        b0, b1 = reg.simple_linear_regression(x, y)
        _, p = reg.t_test_slope(x, y, b0, b1)
        ref = sm.OLS(y, sm.add_constant(x)).fit().pvalues[1]
        assert ref > 0
        # abs=0: la tolerancia absoluta por defecto (1e-12) aceptaría p = 0.
        assert p == pytest.approx(ref, rel=1e-6, abs=0)

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_p_valor_extremo_unilateral(self, alternative):
        # Pendiente positiva (cola derecha) o negativa (cola izquierda), con
        # |t| muy grande: el p-valor unilateral es la mitad del bilateral de OLS.
        x = np.arange(30, dtype=float)
        sign = 1 if alternative == "greater" else -1
        y = 1 + sign * 2 * x + 0.01 * np.random.default_rng(1).standard_normal(30)
        b0, b1 = reg.simple_linear_regression(x, y)
        _, p = reg.t_test_slope(x, y, b0, b1, alternative=alternative)
        ref = sm.OLS(y, sm.add_constant(x)).fit().pvalues[1] / 2
        assert ref > 0
        assert p == pytest.approx(ref, rel=1e-6, abs=0)

    def test_devuelve_floats_de_python(self):
        b0, b1 = reg.simple_linear_regression(X, Y)
        t, p = reg.t_test_slope(X, Y, b0, b1)
        assert type(t) is float
        assert type(p) is float


# ---------------------------------------------------------------------------
# F3.4: correlación de Spearman
# Referencias: scipy.stats.spearmanr y la fórmula 1 - 6·Σd²/(n(n²-1)) sin empates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_spearman_frente_a_scipy(alternative):
    ref = stats.spearmanr(X, Y, alternative=alternative)
    rho, p = reg.spearman_test(X, Y, alternative)
    assert rho == pytest.approx(ref.statistic, rel=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-9, abs=0)


def test_spearman_formula_sin_empates():
    rx, ry = stats.rankdata(X), stats.rankdata(Y)
    n = len(X)
    esperado = 1 - 6 * np.sum((rx - ry) ** 2) / (n * (n**2 - 1))
    assert reg.spearman_correlation(X, Y) == pytest.approx(esperado)


def test_spearman_con_empates_frente_a_scipy():
    a = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7]
    b = [2, 1, 3, 3, 5, 6, 5, 7, 9, 8]
    assert reg.spearman_correlation(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)


def test_spearman_relacion_monotona_no_lineal():
    x = np.arange(1, 21, dtype=float)
    y = np.exp(x / 3)
    assert reg.spearman_correlation(x, y) == pytest.approx(1.0)
    assert reg.correlation(x, y) < 0.9
    rho, p = reg.spearman_test(x, y)
    assert rho == pytest.approx(1.0) and p == 0.0
    assert reg.spearman_test(x, -y)[0] == pytest.approx(-1.0)


def test_spearman_entradas_invalidas():
    with pytest.raises(ValueError):
        reg.spearman_correlation([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError):
        reg.spearman_test([1, 2], [2, 1])
    with pytest.raises(ValueError):
        reg.spearman_test([1, 2, np.nan], [1, 2, 3])
