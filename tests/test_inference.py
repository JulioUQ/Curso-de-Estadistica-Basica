"""
Tests de ``stats_toolkit.inference`` (Módulo 4).

Referencias independientes: valores de tablas de la Normal, ``statsmodels``
(intervalos y contrastes) y ``scipy.stats.ttest_1samp``. La cobertura del IC
se contrasta con el margen binomial del nivel nominal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind
from statsmodels.stats.weightstats import DescrStatsW

from stats_toolkit import inference as inf

SAMPLE = np.array([2340, 2510, 2280, 2600, 2455, 2390, 2470, 2530, 2410, 2380], dtype=float)

# statsmodels nombra las alternativas de otra forma
SM_ALT = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}


def test_error_estandar_de_la_media():
    assert inf.standard_error_mean(sigma=15, n=36) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Intervalos de confianza
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    @pytest.mark.parametrize("confidence,z_tabla", [(0.90, 1.6449), (0.95, 1.9600), (0.99, 2.5758)])
    def test_media_sigma_conocida_con_cuantiles_de_tabla(self, confidence, z_tabla):
        low, up = inf.confidence_interval_mean_z(100, sigma=15, n=36, confidence=confidence)
        assert (low + up) / 2 == pytest.approx(100)
        assert (up - low) / 2 == pytest.approx(z_tabla * 2.5, abs=1e-3)

    def test_media_sigma_desconocida_frente_a_statsmodels(self):
        low, up = inf.confidence_interval_mean_t(
            SAMPLE.mean(), SAMPLE.std(ddof=1), SAMPLE.size, confidence=0.95
        )
        ref = DescrStatsW(SAMPLE).tconfint_mean(alpha=0.05)
        assert (low, up) == pytest.approx(ref)

    def test_intervalo_t_es_mas_ancho_que_el_z(self):
        # Con la misma desviación, los cuantiles t (n-1 g.l.) superan a los z.
        z = inf.confidence_interval_mean_z(0, 1, 10)
        t = inf.confidence_interval_mean_t(0, 1, 10)
        assert (t[1] - t[0]) > (z[1] - z[0])

    def test_proporcion_frente_a_statsmodels_normal(self):
        low, up = inf.confidence_interval_proportion(p_hat=15 / 50, n=50, confidence=0.95)
        ref = proportion_confint(15, 50, alpha=0.05, method="normal")
        assert (low, up) == pytest.approx(ref)

    # El IC de una proporción con p_hat = 0 o 1 (intervalo de Wald degenerado)
    # no tiene comportamiento especificado todavía (Wilson: tarea F2.2).


# ---------------------------------------------------------------------------
# Contrastes de hipótesis
# ---------------------------------------------------------------------------


class TestZTestMean:
    # x̄ = 105, mu0 = 100, sigma = 15, n = 36  ->  z = 5 / 2.5 = 2
    @pytest.mark.parametrize(
        "alternative,p_tabla",
        [("two-sided", 0.045500), ("greater", 0.022750), ("less", 0.977250)],
    )
    def test_valores_de_tabla(self, alternative, p_tabla):
        z, p = inf.z_test_mean(105, 100, sigma=15, n=36, alternative=alternative)
        assert z == pytest.approx(2.0)
        assert p == pytest.approx(p_tabla, abs=1e-6)

    def test_alternativa_invalida_es_error(self):
        with pytest.raises(ValueError):
            inf.z_test_mean(105, 100, 15, 36, alternative="distinto")


class TestTTestMean:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_frente_a_scipy(self, alternative):
        t, p = inf.t_test_mean(SAMPLE, mu0=2400, alternative=alternative)
        ref = stats.ttest_1samp(SAMPLE, popmean=2400, alternative=alternative)
        assert t == pytest.approx(ref.statistic)
        assert p == pytest.approx(ref.pvalue)

    def test_estadistico_frente_a_statsmodels(self):
        t, p = inf.t_test_mean(SAMPLE, mu0=2400)
        t_ref, p_ref, _ = DescrStatsW(SAMPLE).ttest_mean(2400)
        assert (t, p) == pytest.approx((t_ref, p_ref))


class TestTwoProportionZTest:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_frente_a_statsmodels(self, alternative):
        z, p = inf.two_proportion_z_test(30, 100, 45, 110, alternative=alternative)
        z_ref, p_ref = proportions_ztest([30, 45], [100, 110], alternative=SM_ALT[alternative])
        assert z == pytest.approx(z_ref)
        assert p == pytest.approx(p_ref)


# ---------------------------------------------------------------------------
# Precisión de p-valores extremos (corregido en F1.4: colas con sf)
# ---------------------------------------------------------------------------

_EXTREME_T_SAMPLE = 10 + 0.1 * np.random.default_rng(0).standard_normal(30)


@pytest.mark.parametrize(
    "case",
    [
        "z_bilateral",
        "z_unilateral_derecha",
        "t_bilateral",
        "dos_proporciones",
    ],
)
def test_p_valores_extremos_no_se_anulan(case):
    if case == "z_bilateral":
        _, p = inf.z_test_mean(10, 0, sigma=1, n=1)
        ref = 2 * stats.norm.sf(10)
    elif case == "z_unilateral_derecha":
        _, p = inf.z_test_mean(10, 0, sigma=1, n=1, alternative="greater")
        ref = stats.norm.sf(10)
    elif case == "t_bilateral":
        _, p = inf.t_test_mean(_EXTREME_T_SAMPLE, mu0=0)
        ref = stats.ttest_1samp(_EXTREME_T_SAMPLE, 0).pvalue
    else:
        _, p = inf.two_proportion_z_test(900, 1000, 100, 1000)
        ref = proportions_ztest([900, 100], [1000, 1000])[1]
    assert ref > 0
    # abs=0: la tolerancia absoluta por defecto (1e-12) aceptaría p = 0.
    assert p == pytest.approx(ref, rel=1e-6, abs=0)


@pytest.mark.parametrize(
    "case",
    ["z_cola_izquierda", "t_cola_izquierda", "t_cola_derecha", "dos_proporciones_cola_izquierda"],
)
def test_p_valores_extremos_unilaterales_con_estadistico_negativo_y_positivo(case):
    if case == "z_cola_izquierda":
        _, p = inf.z_test_mean(-10, 0, sigma=1, n=1, alternative="less")
        ref = stats.norm.cdf(-10)
    elif case == "t_cola_izquierda":
        _, p = inf.t_test_mean(-_EXTREME_T_SAMPLE, mu0=0, alternative="less")
        ref = stats.ttest_1samp(-_EXTREME_T_SAMPLE, 0, alternative="less").pvalue
    elif case == "t_cola_derecha":
        _, p = inf.t_test_mean(_EXTREME_T_SAMPLE, mu0=0, alternative="greater")
        ref = stats.ttest_1samp(_EXTREME_T_SAMPLE, 0, alternative="greater").pvalue
    else:
        _, p = inf.two_proportion_z_test(100, 1000, 900, 1000, alternative="less")
        ref = proportions_ztest([100, 900], [1000, 1000], alternative="smaller")[1]
    assert ref > 0
    assert p == pytest.approx(ref, rel=1e-6, abs=0)


def test_z_con_estadistico_negativo_valores_de_tabla():
    # x̄ = 95, mu0 = 100, sigma = 15, n = 36  ->  z = -2
    expected = {"two-sided": 0.045500, "greater": 0.977250, "less": 0.022750}
    for alternative, p_tabla in expected.items():
        z, p = inf.z_test_mean(95, 100, sigma=15, n=36, alternative=alternative)
        assert z == pytest.approx(-2.0)
        assert p == pytest.approx(p_tabla, abs=1e-6)


def test_bilateral_simetrico_en_las_funciones_publicas():
    _, p_pos = inf.z_test_mean(105, 100, sigma=15, n=36)
    _, p_neg = inf.z_test_mean(95, 100, sigma=15, n=36)
    assert p_pos == pytest.approx(p_neg, rel=1e-12)
    _, p_ab = inf.two_proportion_z_test(30, 100, 45, 110)
    _, p_ba = inf.two_proportion_z_test(45, 110, 30, 100)
    assert p_ab == pytest.approx(p_ba, rel=1e-12)


def test_contrastes_devuelven_floats_de_python():
    results = [
        inf.z_test_mean(105, 100, sigma=15, n=36),
        inf.t_test_mean(SAMPLE, mu0=2400),
        inf.two_proportion_z_test(30, 100, 45, 110),
    ]
    for stat, p in results:
        assert type(stat) is float
        assert type(p) is float


# ---------------------------------------------------------------------------
# Simulación de cobertura
# ---------------------------------------------------------------------------


class TestSimulateCICoverage:
    @pytest.mark.parametrize("confidence", [0.80, 0.95])
    def test_cobertura_proxima_al_nivel_nominal(self, confidence):
        n_sim = 4_000
        cov = inf.simulate_ci_coverage(
            true_mu=100, sigma=15, n=30, confidence=confidence, n_simulations=n_sim, seed=1
        )
        margin = 4 * math.sqrt(confidence * (1 - confidence) / n_sim)
        assert abs(cov - confidence) <= margin

    def test_reproducible_con_semilla(self):
        kwargs = dict(true_mu=0, sigma=1, n=10, n_simulations=200, seed=9)
        assert inf.simulate_ci_coverage(**kwargs) == inf.simulate_ci_coverage(**kwargs)


# ---------------------------------------------------------------------------
# F2.2: funciones nuevas
# ---------------------------------------------------------------------------

X_A = np.array([5.1, 4.8, 6.2, 5.9, 5.5, 6.8, 4.9, 5.7])
X_B = np.array([6.0, 6.9, 7.4, 5.8, 7.1, 6.6, 7.9, 6.2, 7.0, 6.4, 8.1])
ANTES = np.array([142, 150, 138, 160, 155, 147, 152, 149], dtype=float)
DESPUES = np.array([138, 145, 139, 151, 150, 140, 149, 147], dtype=float)


class TestWilson:
    @pytest.mark.parametrize("x,n", [(15, 50), (0, 20), (20, 20), (3, 7)])
    def test_frente_a_statsmodels(self, x, n):
        low, up = inf.confidence_interval_proportion_wilson(x, n, 0.95)
        ref = proportion_confint(x, n, alpha=0.05, method="wilson")
        assert (low, up) == pytest.approx(ref)

    def test_no_es_degenerado_con_p_hat_cero(self):
        low, up = inf.confidence_interval_proportion_wilson(0, 20)
        assert low == 0.0 and up > 0.1


class TestOneProportion:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_frente_a_statsmodels(self, alternative):
        z, p = inf.one_proportion_z_test(62, 100, 0.5, alternative=alternative)
        z_ref, p_ref = proportions_ztest(62, 100, value=0.5, prop_var=0.5, alternative=SM_ALT[alternative])
        assert z == pytest.approx(z_ref)
        assert p == pytest.approx(p_ref)


class TestWelch:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_frente_a_scipy(self, alternative):
        t, p, df = inf.welch_t_test(X_A, X_B, alternative=alternative)
        ref = stats.ttest_ind(X_A, X_B, equal_var=False, alternative=alternative)
        assert (t, p) == pytest.approx((ref.statistic, ref.pvalue))
        assert df == pytest.approx(ref.df)

    def test_grados_de_libertad_frente_a_statsmodels(self):
        _, _, df = inf.welch_t_test(X_A, X_B)
        assert df == pytest.approx(sm_ttest_ind(X_A, X_B, usevar="unequal")[2])


class TestPaired:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_frente_a_scipy(self, alternative):
        t, p = inf.paired_t_test(ANTES, DESPUES, alternative=alternative)
        ref = stats.ttest_rel(ANTES, DESPUES, alternative=alternative)
        assert (t, p) == pytest.approx((ref.statistic, ref.pvalue))

    def test_equivale_a_una_muestra_sobre_diferencias(self):
        assert inf.paired_t_test(ANTES, DESPUES) == pytest.approx(inf.t_test_mean(ANTES - DESPUES, 0))

    def test_longitudes_distintas_es_error(self):
        with pytest.raises(ValueError):
            inf.paired_t_test([1, 2, 3], [1, 2])


class TestVarianza:
    def test_formula_cerrada_con_dos_grados_de_libertad(self):
        # n = 3 -> 2 g.l.: P(chi2 > c) = exp(-c / 2)
        x = np.array([1.0, 4.0, 7.0])  # s^2 = 9
        chi2, p = inf.chi2_test_variance(x, sigma0_sq=4.0, alternative="greater")
        assert chi2 == pytest.approx(2 * 9 / 4)
        assert p == pytest.approx(math.exp(-chi2 / 2))

    def test_bilateral_es_doble_de_la_cola_menor(self):
        chi2, p_der = inf.chi2_test_variance(X_B, 0.5, "greater")
        _, p_izq = inf.chi2_test_variance(X_B, 0.5, "less")
        assert inf.chi2_test_variance(X_B, 0.5)[1] == pytest.approx(min(1, 2 * min(p_der, p_izq)))
        assert p_der + p_izq == pytest.approx(1)


class TestValidacionInferencia:
    def test_nan_es_error(self):
        with pytest.raises(ValueError):
            inf.t_test_mean([1.0, 2.0, np.nan], 0)

    def test_varianza_cero_es_error(self):
        with pytest.raises(ValueError):
            inf.t_test_mean([3.0, 3.0, 3.0], 0)

    def test_proporcion_agrupada_degenerada_es_error(self):
        with pytest.raises(ValueError):
            inf.two_proportion_z_test(0, 10, 0, 12)

    def test_sigma_no_positiva_es_error(self):
        with pytest.raises(ValueError):
            inf.z_test_mean(1, 0, sigma=0, n=10)

    def test_confianza_fuera_de_rango_es_error(self):
        with pytest.raises(ValueError):
            inf.confidence_interval_mean_z(0, 1, 10, confidence=95)


# ---------------------------------------------------------------------------
# F3.2: potencia y tamaño muestral
# Referencias: statsmodels NormalIndPower (ratio=0 = una muestra), fórmulas
# cerradas y simulación.
# ---------------------------------------------------------------------------

from statsmodels.stats.power import NormalIndPower  # noqa: E402


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("effect,n", [(5, 36), (-3, 50), (1, 10), (8, 100)])
def test_potencia_z_frente_a_statsmodels(effect, n, alternative):
    esperado = NormalIndPower().power(effect / 15, n, 0.05, ratio=0, alternative=SM_ALT[alternative])
    assert inf.power_z_test_mean(effect, 15, n, 0.05, alternative) == pytest.approx(esperado, rel=1e-9)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_potencia_sin_efecto_es_alpha(alternative):
    assert inf.power_z_test_mean(0, 10, 25, 0.05, alternative) == pytest.approx(0.05)


def test_potencia_crece_con_n_y_con_el_efecto():
    pot_n = [inf.power_z_test_mean(3, 15, n) for n in (10, 50, 100, 400)]
    pot_e = [inf.power_z_test_mean(e, 15, 50) for e in (1, 3, 5, 8)]
    assert pot_n == sorted(pot_n) and pot_e == sorted(pot_e)


def test_potencia_coincide_con_simulacion():
    rng = np.random.default_rng(0)
    mu0, effect, sigma, n = 100, 5, 15, 36
    medias = rng.normal(mu0 + effect, sigma, size=(20_000, n)).mean(axis=1)
    z = (medias - mu0) / (sigma / math.sqrt(n))
    empirica = np.mean(np.abs(z) > stats.norm.isf(0.025))
    assert empirica == pytest.approx(inf.power_z_test_mean(effect, sigma, n), abs=0.015)


def test_tamano_muestral_test_z_formula_y_potencia():
    # n = ((1,96 + 0,8416) · 15 / 5)^2 = 70,6 -> 71
    n = inf.sample_size_z_test_mean(5, 15)
    assert n == 71
    assert inf.power_z_test_mean(5, 15, n) >= 0.80 > inf.power_z_test_mean(5, 15, n - 2)
    # Unilateral: z_{0,95} = 1,645 -> ((1,645 + 0,8416)·15/5)^2 = 55,6 -> 56
    assert inf.sample_size_z_test_mean(5, 15, alternative="greater") == 56
    assert inf.sample_size_z_test_mean(-5, 15, alternative="less") == 56


@pytest.mark.parametrize("power", [0.8, 0.9, 0.95])
def test_tamano_muestral_frente_a_statsmodels(power):
    n_sm = NormalIndPower().solve_power(effect_size=0.4, alpha=0.05, power=power, ratio=0, alternative="larger")
    assert inf.sample_size_z_test_mean(0.4, 1, power=power, alternative="greater") == math.ceil(n_sm)


def test_tamano_muestral_ic():
    assert inf.sample_size_mean_ci(2, 15) == 217          # (1,96·15/2)^2 = 216,1
    assert inf.sample_size_proportion_ci(0.03) == 1068    # 1,96²·0,25/0,0009 = 1067,1
    assert inf.sample_size_proportion_ci(0.05) == 385     # 384,1 (el clásico 385)
    assert inf.sample_size_proportion_ci(0.05, p=0.1) < inf.sample_size_proportion_ci(0.05)
    # El margen obtenido con ese n no supera el pedido
    n = inf.sample_size_mean_ci(2, 15)
    lo, hi = inf.confidence_interval_mean_z(0, 15, n)
    assert (hi - lo) / 2 <= 2


@pytest.mark.parametrize(
    "kwargs",
    [dict(effect=0, sigma=1), dict(effect=1, sigma=0), dict(effect=1, sigma=1, power=0.01),
     dict(effect=-1, sigma=1, alternative="greater"), dict(effect=1, sigma=1, alternative="less"),
     dict(effect=1, sigma=1, alpha=1.5)],
)
def test_tamano_muestral_entradas_invalidas(kwargs):
    with pytest.raises(ValueError):
        inf.sample_size_z_test_mean(**kwargs)


def test_tamano_muestral_ic_entradas_invalidas():
    for llamada in (lambda: inf.sample_size_mean_ci(0, 1), lambda: inf.sample_size_proportion_ci(0.05, p=1),
                    lambda: inf.sample_size_proportion_ci(1.2), lambda: inf.power_z_test_mean(1, 1, 10, alternative="x")):
        with pytest.raises(ValueError):
            llamada()


# ---------------------------------------------------------------------------
# F3.3: bootstrap
# Referencias: scipy.stats.bootstrap (percentiles), el IC t clásico para la
# media y propiedades (reproducibilidad, orden de los límites).
# ---------------------------------------------------------------------------

BOOT = np.random.default_rng(1).normal(10, 2, size=40)


@pytest.mark.parametrize("statistic", [np.mean, np.median])
def test_bootstrap_frente_a_scipy(statistic):
    ref = stats.bootstrap((BOOT,), statistic, method="percentile", n_resamples=20_000, rng=0).confidence_interval
    lo, hi = inf.bootstrap_ci(BOOT, statistic, n_resamples=20_000, seed=0)
    # Distintas remuestras: coinciden salvo el error Monte Carlo
    assert lo == pytest.approx(ref.low, abs=0.05) and hi == pytest.approx(ref.high, abs=0.05)


def test_bootstrap_media_parecido_al_ic_t():
    lo, hi = inf.bootstrap_ci(BOOT, seed=3)
    lo_t, hi_t = inf.confidence_interval_mean_t(BOOT.mean(), BOOT.std(ddof=1), BOOT.size)
    assert lo == pytest.approx(lo_t, abs=0.1) and hi == pytest.approx(hi_t, abs=0.1)


def test_bootstrap_reproducible_y_ordenado():
    a = inf.bootstrap_ci(BOOT, np.median, seed=5)
    assert a == inf.bootstrap_ci(BOOT, np.median, seed=5)
    assert a[0] <= np.median(BOOT) <= a[1]
    dist = inf.bootstrap_distribution(BOOT, n_resamples=500, seed=5)
    assert dist.shape == (500,)
    assert _dentro_del_rango(dist)


def _dentro_del_rango(dist):
    return BOOT.min() <= dist.min() and dist.max() <= BOOT.max()


def test_bootstrap_entradas_invalidas():
    with pytest.raises(ValueError):
        inf.bootstrap_ci([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        inf.bootstrap_ci([1.0])
    with pytest.raises(ValueError):
        inf.bootstrap_ci(BOOT, confidence=95)
    with pytest.raises(ValueError):
        inf.bootstrap_distribution(BOOT, n_resamples=0)
