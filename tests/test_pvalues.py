"""
Tests de ``stats_toolkit._pvalues`` (utilidad interna de los Módulos 4 y 5).

Referencias independientes de SciPy: fórmulas cerradas con ``math.erfc`` para la
Normal y las funciones de distribución exactas de la t de Student con 1 y 2
grados de libertad (escritas sin restas que cancelen cifras). Además se
comprueban propiedades: simetría del bilateral, relación entre colas, rango
[0, 1] y comportamiento en el límite de representación del ``float``.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from stats_toolkit._pvalues import p_value_from_statistic

ALTERNATIVES = ["two-sided", "greater", "less"]
SQRT2 = math.sqrt(2)


# ---------------------------------------------------------------------------
# Referencias independientes
# ---------------------------------------------------------------------------


def _normal_ref(z: float, alternative: str) -> float:
    """p-valor de la Normal estándar mediante math.erfc (sin pasar por SciPy)."""
    if alternative == "two-sided":
        return math.erfc(abs(z) / SQRT2)
    if alternative == "greater":
        return 0.5 * math.erfc(z / SQRT2)
    return 0.5 * math.erfc(-z / SQRT2)


def _t_upper_tail(t: float, df: int) -> float:
    """P(T > t) exacta para df = 1 (Cauchy) y df = 2, sin cancelación para t > 0."""
    if t < 0:
        return 1 - _t_upper_tail(-t, df)
    if df == 1:
        return 0.5 if t == 0 else math.atan(1 / t) / math.pi
    if df == 2:
        s = math.sqrt(t * t + 2)
        return 1 / (s * (s + t))  # = (1 - t/s) / 2, reescrita sin resta
    raise ValueError(df)


def _t_ref(t: float, df: int, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * _t_upper_tail(abs(t), df)
    if alternative == "greater":
        return _t_upper_tail(t, df)
    return _t_upper_tail(-t, df)


STATISTICS = [-30.0, -10.0, -3.0, -1.96, -0.5, 0.0, 0.5, 1.96, 3.0, 10.0, 30.0]


# ---------------------------------------------------------------------------
# Comparación con referencias independientes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("z", STATISTICS)
def test_normal_frente_a_formula_cerrada(z, alternative):
    p = p_value_from_statistic(z, stats.norm(), alternative)
    # abs=0: con valores del orden de 1e-198 la tolerancia absoluta por
    # defecto (1e-12) aceptaría cualquier cosa, incluido 0.
    assert p == pytest.approx(_normal_ref(z, alternative), rel=1e-10, abs=0)


@pytest.mark.parametrize("df", [1, 2])
@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("t", [-1e6, -50.0, -2.0, 0.0, 2.0, 50.0, 1e6])
def test_t_frente_a_formula_cerrada(t, alternative, df):
    p = p_value_from_statistic(t, stats.t(df), alternative)
    assert p == pytest.approx(_t_ref(t, df, alternative), rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# Propiedades
# ---------------------------------------------------------------------------

DISTS = {"normal": stats.norm(), "t(5)": stats.t(5), "t(29)": stats.t(29)}


@pytest.mark.parametrize("name", DISTS)
@pytest.mark.parametrize("s", [0.3, 1.0, 2.5, 6.0, 12.0])
def test_bilateral_es_simetrico(name, s):
    dist = DISTS[name]
    assert p_value_from_statistic(s, dist) == p_value_from_statistic(-s, dist)


@pytest.mark.parametrize("name", DISTS)
@pytest.mark.parametrize("s", [-12.0, -2.5, -0.3, 0.0, 0.3, 2.5, 12.0])
def test_bilateral_es_el_doble_de_la_cola_menor(name, s):
    dist = DISTS[name]
    right = p_value_from_statistic(s, dist, "greater")
    left = p_value_from_statistic(s, dist, "less")
    assert p_value_from_statistic(s, dist) == pytest.approx(2 * min(right, left), rel=1e-12, abs=0)


@pytest.mark.parametrize("name", DISTS)
@pytest.mark.parametrize("s", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_las_dos_colas_suman_uno(name, s):
    # Solo con estadísticos habituales: en las colas extremas, una de las dos
    # probabilidades es tan pequeña que la suma es 1 por redondeo y no aporta nada.
    dist = DISTS[name]
    total = p_value_from_statistic(s, dist, "greater") + p_value_from_statistic(s, dist, "less")
    assert total == pytest.approx(1.0, abs=1e-12)


def test_estadistico_cero():
    for dist in DISTS.values():
        assert p_value_from_statistic(0.0, dist) == pytest.approx(1.0)
        assert p_value_from_statistic(0.0, dist, "greater") == pytest.approx(0.5)
        assert p_value_from_statistic(0.0, dist, "less") == pytest.approx(0.5)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_p_valores_en_el_intervalo_unidad(alternative):
    values = [-math.inf, -1e300, -40.0, -8.5, -1.0, -0.0, 0.0, 1.0, 8.5, 40.0, 1e300, math.inf]
    for dist in DISTS.values():
        for s in values:
            p = p_value_from_statistic(s, dist, alternative)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Límites de la representación en coma flotante
# ---------------------------------------------------------------------------


def test_extremo_representable_no_se_anula():
    # P(|Z| >= 37) ≈ 1,1e-299: diminuto, pero representable como float.
    ref = math.erfc(37 / SQRT2)
    assert ref > 0
    assert p_value_from_statistic(37.0, stats.norm()) == pytest.approx(ref, rel=1e-10, abs=0)
    assert p_value_from_statistic(37.0, stats.norm(), "greater") == pytest.approx(ref / 2, rel=1e-10, abs=0)
    assert p_value_from_statistic(-37.0, stats.norm(), "less") == pytest.approx(ref / 2, rel=1e-10, abs=0)


def test_subdesbordamiento_legitimo_da_cero():
    # P(Z >= 40) ≈ 3,7e-350 no cabe en un float: la propia referencia vale 0.0.
    # Aquí el 0.0 es una limitación de la aritmética, no un error del cálculo.
    assert math.erfc(40 / SQRT2) == 0.0
    assert p_value_from_statistic(40.0, stats.norm()) == 0.0
    assert p_value_from_statistic(40.0, stats.norm(), "greater") == 0.0
    assert p_value_from_statistic(-40.0, stats.norm(), "less") == 0.0


def test_estadisticos_infinitos_y_nan():
    norm = stats.norm()
    assert p_value_from_statistic(math.inf, norm) == 0.0
    assert p_value_from_statistic(-math.inf, norm) == 0.0
    assert p_value_from_statistic(math.inf, norm, "less") == 1.0
    assert p_value_from_statistic(-math.inf, norm, "greater") == 1.0
    assert math.isnan(p_value_from_statistic(math.nan, norm))


# ---------------------------------------------------------------------------
# Contrato
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_devuelve_float_de_python(alternative):
    assert type(p_value_from_statistic(1.3, stats.t(7), alternative)) is float


def test_alternativa_invalida_es_error():
    with pytest.raises(ValueError):
        p_value_from_statistic(1.0, stats.norm(), "distinto")
