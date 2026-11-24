"""
Tests de ``stats_toolkit.descriptive`` (Módulo 1).

Referencias independientes: cálculos a mano sobre conjuntos pequeños,
propiedades matemáticas, y pandas / SciPy (``bias=False``) para los
estimadores de asimetría y curtosis tipo 2 de Joanes y Gill (1998).
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stats_toolkit import descriptive as desc

# Conjunto clásico: media 5, varianza poblacional 4, desviación poblacional 2.
CLASSIC = [2, 4, 4, 4, 5, 5, 7, 9]


# ---------------------------------------------------------------------------
# Tendencia central
# ---------------------------------------------------------------------------


class TestCentralTendency:
    def test_media_mediana_moda_con_moda_unica(self):
        res = desc.central_tendency([2, 3, 3, 5, 7])
        assert res["mean"] == pytest.approx(20 / 5)
        assert res["median"] == pytest.approx(3)
        assert res["mode"] == pytest.approx(3)

    def test_mediana_con_n_par_es_promedio_de_los_centrales(self):
        # Definición de la teoría del M1: (x_(n/2) + x_(n/2+1)) / 2
        assert desc.central_tendency([7, 1, 4, 2])["median"] == pytest.approx((2 + 4) / 2)

    def test_ignora_nan(self):
        res = desc.central_tendency([1.0, 2.0, np.nan, 3.0, 3.0])
        assert res["mean"] == pytest.approx(9 / 4)
        assert res["median"] == pytest.approx(2.5)
        assert res["mode"] == pytest.approx(3)

    def test_estructura_y_tipos_se_mantienen(self):
        # Compatibilidad con el código existente: mismas claves y valores float.
        res = desc.central_tendency([2, 3, 3, 5, 7])
        assert set(res) == {"mean", "median", "mode"}
        assert all(isinstance(v, float) for v in res.values())

    @pytest.mark.parametrize(
        "x",
        [[1, 1, 2, 3, 3], [5.2, 1.7, 3.3, 9.1], [1, 1, 2, 2, 3, 3]],
        ids=["bimodal", "todos_distintos", "frecuencias_iguales"],
    )
    def test_moda_nan_si_no_es_unica_y_sin_avisos(self, x):
        # Antes devolvía arbitrariamente el valor más pequeño (1, 1.7 y 1).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = desc.central_tendency(x)
        assert isinstance(res["mode"], float)
        assert math.isnan(res["mode"])

    def test_media_y_mediana_no_dependen_de_la_moda(self):
        x = [5.2, 1.7, 3.3, 9.1]
        res = desc.central_tendency(x)
        assert res["mean"] == pytest.approx(sum(x) / 4)
        assert res["median"] == pytest.approx((3.3 + 5.2) / 2)

    def test_moda_coincide_con_modes_cuando_es_unica(self):
        x = [4, 5, 3, 5, 5, 2, 4]
        assert desc.central_tendency(x)["mode"] == desc.modes(x)[0] == 5.0


class TestModes:
    """
    Referencias independientes: ``statistics.multimode`` (biblioteca estándar)
    y recuentos hechos a mano. Cuando todos los valores distintos tienen la
    misma frecuencia, nuestra convención ("sin moda") difiere de
    ``statistics.multimode`` y de ``pandas.Series.mode``, que devuelven todos
    los valores; esos casos se comprueban por separado y se documentan.
    """

    @pytest.mark.parametrize(
        "x,expected",
        [
            ([2, 3, 3, 5, 7], [3.0]),  # moda única
            ([4, 5, 3, 5, 5, 2, 4], [5.0]),  # moda única (3 veces frente a 2)
            ([1, 1, 2, 3, 3], [1.0, 3.0]),  # bimodal
            ([1, 1, 2, 3, 3, 4, 4], [1.0, 3.0, 4.0]),  # trimodal
        ],
    )
    def test_coincide_con_multimode_cuando_hay_moda_destacada(self, x, expected):
        res = desc.modes(x)
        assert res == expected
        assert res == sorted(float(v) for v in statistics.multimode(x))

    def test_invariante_al_orden_de_las_observaciones(self):
        x = [1, 1, 2, 3, 3, 4, 7, 7]
        expected = desc.modes(x)
        rng = np.random.default_rng(0)
        for _ in range(5):
            assert desc.modes(list(rng.permutation(x))) == expected
        # statistics.multimode depende del orden de aparición; nosotros no.
        assert statistics.multimode([3, 3, 1, 1]) == [3, 1]
        assert desc.modes([3, 3, 1, 1, 2]) == [1.0, 3.0]

    @pytest.mark.parametrize(
        "x",
        [[5.2, 1.7, 3.3, 9.1], [1, 1, 2, 2, 3, 3], ["a", "b", "c"]],
        ids=["todos_distintos", "todas_frecuencias_2", "categorias_distintas"],
    )
    def test_convencion_sin_moda_si_todas_las_frecuencias_son_iguales(self, x):
        assert desc.modes(x) == []
        # Convención distinta en la biblioteca estándar y en pandas:
        assert len(statistics.multimode(x)) == len(set(x))
        assert len(pd.Series(x).mode()) == len(set(x))

    @staticmethod
    def _salarios_practica_m1() -> np.ndarray:
        """Reproduce exactamente los 300 salarios de la práctica del M1."""
        np.random.seed(42)
        medias_log = {"Ventas": 7.2, "IT": 7.5, "RRHH": 7.1, "Direccion": 8.0}
        return np.array(
            [round(s, 2) for mu in medias_log.values() for s in np.random.lognormal(mu, 0.3, 75)]
        )

    def test_salarios_de_la_practica_sin_moda(self):
        x = self._salarios_practica_m1()
        assert len(set(x)) == 300  # los 300 salarios son distintos
        assert desc.modes(x) == []
        assert math.isnan(desc.central_tendency(x)["mode"])

    def test_una_coincidencia_por_redondeo_basta_para_crear_una_moda(self):
        # Con datos continuos redondeados, dos valores pueden coincidir por
        # azar. Con esta semilla, 300 salarios en céntimos dan 299 distintos:
        # el repetido es, por definición, la moda, aunque no describa dónde
        # se concentran los datos (motivo para usar la clase modal).
        x = np.round(np.random.default_rng(42).lognormal(7.5, 0.3, 300), 2)
        values, counts = np.unique(x, return_counts=True)
        assert counts.max() == 2 and (counts == 2).sum() == 1
        assert desc.modes(x) == [float(values[counts.argmax()])]

    def test_datos_constantes_y_una_sola_observacion(self):
        assert desc.modes([4, 4, 4]) == [4.0]
        assert desc.modes([7.5]) == [7.5]
        assert desc.modes(["x", "x"]) == ["x"]

    @pytest.mark.parametrize("x", [[], [np.nan, np.nan], [None, float("nan"), pd.NA]])
    def test_sin_observaciones_validas_lista_vacia(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert desc.modes(x) == []

    def test_ignora_valores_ausentes(self):
        # NaN es lo más repetido, pero no es una observación válida.
        assert desc.modes([1, np.nan, np.nan, np.nan, 2, 2]) == [2.0]
        assert desc.modes(pd.Series([1.0, None, 2.0, 2.0])) == [2.0]
        assert desc.modes(["a", None, "b", "b", pd.NA, np.nan]) == ["b"]

    def test_datos_categoricos(self):
        assert desc.modes(["Ventas", "IT", "IT", "RRHH"]) == ["IT"]
        assert desc.modes(pd.Series(["b", "a", "b", "a", "c"])) == ["a", "b"]
        # Cuatro departamentos con 75 empleados cada uno: sin moda destacada.
        assert desc.modes(["Ventas", "IT", "RRHH", "Direccion"] * 75) == []

    def test_comparacion_exacta_de_decimales(self):
        # 1 y 1.0 son el mismo número...
        assert desc.modes([1, 1.0, 2]) == [1.0]
        # ...pero 0.1 + 0.2 y 0.3 no son exactamente iguales en coma flotante.
        assert 0.1 + 0.2 != 0.3
        assert desc.modes([0.1 + 0.2, 0.3, 0.3]) == [0.3]
        # Sin redondeo implícito: 7.20 y 7.21 son valores distintos.
        assert desc.modes([7.20, 7.21, 7.21, 7.3]) == [7.21]

    def test_tipos_numericos_de_numpy(self):
        assert desc.modes(np.array([1, 2, 2], dtype=np.int64)) == [2.0]
        assert all(type(v) is float for v in desc.modes(np.array([1.5, 1.5, 2.0])))

    # --- Se cuenta con los valores originales, sin convertir antes a float ---

    def test_enteros_mayores_que_2_elevado_a_53(self):
        # float(2**53) == float(2**53 + 1): convertir antes de contar
        # fusionaría dos valores distintos.
        big, bigger = 2**53, 2**53 + 1
        assert float(big) == float(bigger)
        x = [big, bigger, bigger]
        assert Counter(x)[bigger] == 2  # referencia: recuento exacto
        res = desc.modes(x)
        assert res == [bigger]
        assert type(res[0]) is int and res[0] == 9007199254740993

    def test_enteros_grandes_de_numpy(self):
        big = 2**60
        res = desc.modes(np.array([big, big + 1, big + 1], dtype=np.int64))
        assert res == [big + 1]
        assert type(res[0]) is int

    def test_fraccion_frente_a_decimal(self):
        # Fraction(1, 3) y 1/3 no son iguales en Python: son valores distintos.
        assert Fraction(1, 3) != 1 / 3
        res = desc.modes([Fraction(1, 3), Fraction(1, 3), 1 / 3])
        assert res == [Fraction(1, 3)]
        assert type(res[0]) is Fraction
        # Fraction(1, 2) y 0.5 sí son iguales: se cuentan juntos.
        assert desc.modes([Fraction(1, 2), 0.5, 0.25]) == [Fraction(1, 2)]

    @pytest.mark.parametrize("x", [[1, 1.0, 2], [1.0, 1, 2], [2, 1.0, 1]])
    def test_1_y_1_punto_0_cuentan_juntos_y_el_tipo_no_depende_del_orden(self, x):
        res = desc.modes(x)
        assert res == [1]
        assert type(res[0]) is int  # representante de tipo más exacto

    def test_conserva_tipo_y_valor_original(self):
        assert [type(v) for v in desc.modes([3, 3, 5])] == [int]
        assert [type(v) for v in desc.modes([2.5, 2.5, 4.0])] == [float]
        assert desc.modes(np.array(["a", "b", "b"])) == ["b"]
        assert type(desc.modes(np.array(["a", "b", "b"]))[0]) is str

    def test_central_tendency_sigue_devolviendo_float(self):
        res = desc.central_tendency([3, 3, 5])
        assert res["mode"] == 3 and type(res["mode"]) is float

    def test_datos_mezclados_es_error(self):
        with pytest.raises(TypeError):
            desc.modes([1, "a", "a"])


# ---------------------------------------------------------------------------
# Dispersión
# ---------------------------------------------------------------------------


class TestDispersion:
    def test_muestral_por_defecto_ddof_1(self):
        res = desc.dispersion(CLASSIC)
        var_muestral = 32 / 7  # suma de cuadrados = 32, n - 1 = 7
        assert res["variance"] == pytest.approx(var_muestral)
        assert res["std"] == pytest.approx(math.sqrt(var_muestral))
        assert res["range"] == pytest.approx(9 - 2)
        assert res["cv_pct"] == pytest.approx(math.sqrt(var_muestral) / 5 * 100)

    def test_poblacional_ddof_0(self):
        res = desc.dispersion(CLASSIC, ddof=0)
        assert res["variance"] == pytest.approx(4.0)
        assert res["std"] == pytest.approx(2.0)
        assert res["cv_pct"] == pytest.approx(40.0)

    def test_datos_constantes_tienen_dispersion_nula(self):
        res = desc.dispersion([5, 5, 5, 5])
        assert res["range"] == 0
        assert res["variance"] == 0
        assert res["std"] == 0

    def test_cv_invariante_a_cambio_de_escala(self):
        # El CV es adimensional: multiplicar por una constante positiva no lo cambia.
        x = np.array([12.0, 15.0, 9.0, 20.0, 14.0])
        assert desc.dispersion(x * 1000)["cv_pct"] == pytest.approx(desc.dispersion(x)["cv_pct"])


# ---------------------------------------------------------------------------
# Forma
# ---------------------------------------------------------------------------

SKEWED = np.array([1, 2, 2, 3, 3, 3, 4, 10, 15, 20], dtype=float)


def _g1_g2(x: np.ndarray) -> tuple[float, float]:
    """Coeficientes de momentos (tipo 1) calculados a mano."""
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m3 = np.mean((x - m) ** 3)
    m4 = np.mean((x - m) ** 4)
    return m3 / m2**1.5, m4 / m2**2 - 3


_SHAPE_DATASETS = {
    "cola_derecha_n10": SKEWED,
    "lognormal_n50": np.random.default_rng(1).lognormal(0, 0.5, 50),
    "normal_n200": np.random.default_rng(2).normal(10, 3, 200),
    "cola_izquierda_n4": np.array([-8.0, -4.0, -2.0, -1.0]),
}


class TestShape:
    """Estimadores corregidos G1 y G2 (tipo 2 de Joanes y Gill, 1998)."""

    def test_caso_resuelto_a_mano(self):
        # x = (0, 0, 0, 3) es una variable de dos puntos con p = 1/4, cuyos
        # coeficientes de momentos tienen forma cerrada:
        #   g1 = (1 - 2p) / sqrt(p(1-p)) = 2/sqrt(3)
        #   g2 = (1 - 6p(1-p)) / (p(1-p)) = -2/3
        # Con n = 4:
        #   G1 = sqrt(4·3)/2 · 2/sqrt(3) = 2
        #   G2 = 3/(2·1) · [5·(-2/3) + 6] = 4
        res = desc.shape([0, 0, 0, 3])
        assert res["skewness"] == pytest.approx(2.0)
        assert res["kurtosis_excess"] == pytest.approx(4.0)

    @pytest.mark.parametrize("x", _SHAPE_DATASETS.values(), ids=_SHAPE_DATASETS.keys())
    def test_coincide_con_la_formula_a_partir_de_los_momentos(self, x):
        n = x.size
        g1, g2 = _g1_g2(x)
        res = desc.shape(x)
        assert res["skewness"] == pytest.approx(g1 * math.sqrt(n * (n - 1)) / (n - 2))
        assert res["kurtosis_excess"] == pytest.approx(
            (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
        )

    @pytest.mark.parametrize("x", _SHAPE_DATASETS.values(), ids=_SHAPE_DATASETS.keys())
    def test_coincide_con_pandas_y_scipy_sin_sesgo(self, x):
        res = desc.shape(x)
        assert res["skewness"] == pytest.approx(pd.Series(x).skew())
        assert res["kurtosis_excess"] == pytest.approx(pd.Series(x).kurt())
        assert res["skewness"] == pytest.approx(stats.skew(x, bias=False))
        assert res["kurtosis_excess"] == pytest.approx(stats.kurtosis(x, fisher=True, bias=False))

    def test_factor_de_correccion_de_la_asimetria(self):
        # G1 / g1 = sqrt(n(n-1)) / (n-2): depende solo de n y tiende a 1.
        for n in (10, 30, 300):
            x = np.random.default_rng(n).exponential(size=n)
            g1, _ = _g1_g2(x)
            ratio = desc.shape(x)["skewness"] / g1
            assert ratio == pytest.approx(math.sqrt(n * (n - 1)) / (n - 2))
        assert math.sqrt(300 * 299) / 298 == pytest.approx(1.0, abs=0.006)

    def test_datos_simetricos_tienen_asimetria_nula(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert desc.shape(x)["skewness"] == pytest.approx(0.0, abs=1e-12)

    def test_signo_de_la_asimetria_segun_la_cola(self):
        assert desc.shape(SKEWED)["skewness"] > 0
        assert desc.shape(-SKEWED)["skewness"] < 0

    @pytest.mark.parametrize(
        "a,b",
        [(100, 7), (-20, 0.01), (0, -1), (50, -3), (0, -0.5), (-1000, -250)],
    )
    def test_invariante_a_cambios_de_origen_y_escala(self, a, b):
        # Ambas medidas son adimensionales. Para y = a + b·x:
        #   asimetría(y) = signo(b) · asimetría(x)   (un factor negativo la invierte)
        #   curtosis(y)  = curtosis(x)               (no depende del signo de b)
        # Tolerancia relativa 1e-9: solo cubre el redondeo en coma flotante.
        base = desc.shape(SKEWED)
        res = desc.shape(a + b * SKEWED)
        assert res["skewness"] == pytest.approx(math.copysign(1, b) * base["skewness"], rel=1e-9)
        assert res["kurtosis_excess"] == pytest.approx(base["kurtosis_excess"], rel=1e-9)

    # --- Casos en los que el estimador no está definido -------------------

    @pytest.mark.parametrize("x", [[], [5.0], [1.0, 2.0]], ids=["n0", "n1", "n2"])
    def test_menos_de_tres_observaciones_ninguna_medida(self, x):
        res = desc.shape(x)
        assert math.isnan(res["skewness"])
        assert math.isnan(res["kurtosis_excess"])

    def test_con_tres_observaciones_solo_la_asimetria(self):
        # Asimetría y curtosis se gestionan de forma independiente.
        res = desc.shape([1.0, 2.0, 4.0])
        assert res["skewness"] == pytest.approx(pd.Series([1.0, 2.0, 4.0]).skew())
        assert not math.isnan(res["skewness"])
        assert math.isnan(res["kurtosis_excess"])

    def test_con_cuatro_observaciones_ambas_medidas(self):
        res = desc.shape([1.0, 2.0, 4.0, 8.0])
        assert not math.isnan(res["skewness"])
        assert not math.isnan(res["kurtosis_excess"])

    @pytest.mark.parametrize("x", [[3.0] * 2, [3.0] * 5, [0.1] * 50], ids=["n2", "n5", "n50"])
    def test_datos_constantes_no_definidas_y_sin_avisos(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # cualquier aviso haría fallar el test
            res = desc.shape(x)
        assert math.isnan(res["skewness"])
        assert math.isnan(res["kurtosis_excess"])

    def test_n_cuenta_solo_las_observaciones_validas(self):
        # Los NaN se descartan: (1, 2, NaN, 4) son 3 observaciones válidas.
        res = desc.shape([1.0, 2.0, np.nan, 4.0])
        assert res["skewness"] == pytest.approx(desc.shape([1.0, 2.0, 4.0])["skewness"])
        assert math.isnan(res["kurtosis_excess"])
        full = desc.shape([1.0, 2.0, np.nan, 4.0, 8.0])
        assert full == pytest.approx(desc.shape([1.0, 2.0, 4.0, 8.0]))


# ---------------------------------------------------------------------------
# Resumen de cinco números y outliers
# ---------------------------------------------------------------------------


class TestFiveNumberSummary:
    # El método de cálculo de los cuartiles no está fijado por la teoría
    # (interpolación vs. bisagras de Tukey), así que solo se prueban las
    # propiedades que no dependen de él.

    def test_extremos_y_mediana(self):
        res = desc.five_number_summary([9, 1, 13, 5, 3, 11, 7])
        assert res["min"] == 1
        assert res["median"] == 7
        assert res["max"] == 13

    def test_orden_y_definicion_del_iqr(self):
        rng = np.random.default_rng(0)
        res = desc.five_number_summary(rng.normal(size=101))
        assert res["min"] <= res["q1"] <= res["median"] <= res["q3"] <= res["max"]
        assert res["iqr"] == pytest.approx(res["q3"] - res["q1"])


class TestDetectOutliers:
    # Conjuntos en los que la clasificación no depende del método de cuartiles.

    def test_detecta_atipico_superior(self):
        out = desc.detect_outliers(list(range(1, 11)) + [100])
        assert out.tolist() == [100]

    def test_detecta_atipico_inferior(self):
        out = desc.detect_outliers([-100] + list(range(1, 11)))
        assert out.tolist() == [-100]

    def test_sin_atipicos(self):
        assert desc.detect_outliers(list(range(1, 11))).size == 0


class TestSummaryStatistics:
    def test_n_excluye_nan_y_agrega_las_secciones(self):
        res = desc.summary_statistics([1.0, np.nan, 2.0, 3.0])
        assert res["n"] == 3
        assert set(res) == {"n", "central_tendency", "dispersion", "shape", "five_number_summary"}
        assert res["central_tendency"]["mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Utilidades de exploración de DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def df_ejemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "grupo": ["a", "b", "b", "a"],
            "valor": [10.0, 20.0, 20.0, np.nan],
        }
    )


class TestDescribeDf:
    def test_recuentos_nulos_y_unicos(self, df_ejemplo):
        res = desc.describe_df(df_ejemplo).set_index("Column")
        assert res.loc["valor", "Non-null Count"] == 3
        assert res.loc["valor", "% Null Values"] == pytest.approx(25.0)
        assert res.loc["id", "Unique Values"] == 3
        assert res.loc["grupo", "Unique Values"] == 2

    def test_estadisticos_numericos(self, df_ejemplo):
        res = desc.describe_df(df_ejemplo).set_index("Column")
        assert res.loc["valor", "mean"] == pytest.approx(50 / 3)
        assert res.loc["valor", "median"] == pytest.approx(20.0)
        assert res.loc["valor", "min"] == pytest.approx(10.0)

    def test_categorias_mas_frecuentes(self, df_ejemplo):
        res = desc.describe_df(df_ejemplo).set_index("Column")
        top = res.loc["grupo", "TopCounts"]
        assert isinstance(top, str)
        assert "a (2)" in top and "b (2)" in top


class TestDetectDuplicates:
    def test_devuelve_todas_las_apariciones_del_duplicado(self, df_ejemplo):
        dup = desc.detect_duplicates(df_ejemplo, subset=["id"], verbose=False)
        assert dup["id"].tolist() == [2, 2]

    def test_duplicado_exacto_de_fila_completa(self, df_ejemplo):
        dup = desc.detect_duplicates(df_ejemplo, verbose=False)
        assert len(dup) == 2

    def test_sin_duplicados(self, df_ejemplo):
        dup = desc.detect_duplicates(df_ejemplo.drop_duplicates(), subset=["id"], verbose=False)
        assert dup.empty

    def test_verbose_false_no_imprime(self, df_ejemplo, capsys):
        desc.detect_duplicates(df_ejemplo, subset=["id"], verbose=False)
        assert capsys.readouterr().out == ""


class TestUniqueCategories:
    def test_valores_unicos_por_columna(self, df_ejemplo):
        res = desc.unique_categories(df_ejemplo, columns=["grupo"], verbose=False)
        assert set(res["grupo"]) == {"a", "b"}

    def test_por_defecto_usa_columnas_de_texto(self, df_ejemplo):
        res = desc.unique_categories(df_ejemplo, verbose=False)
        assert list(res) == ["grupo"]


class TestSummarizeByGroup:
    def test_medias_por_grupo_y_columnas_aplanadas(self):
        df = pd.DataFrame({"g": ["x", "x", "y", "y", "y"], "v": [1.0, 3.0, 2.0, 4.0, 9.0]})
        res = desc.summarize_by_group(df, ["g"], ["v"], ["mean", "std"]).set_index("g")
        assert list(res.columns) == ["v_mean", "v_std"]
        assert res.loc["x", "v_mean"] == pytest.approx(2.0)
        assert res.loc["y", "v_mean"] == pytest.approx(5.0)
        assert res.loc["x", "v_std"] == pytest.approx(math.sqrt(2.0))

    def test_ordenacion(self):
        df = pd.DataFrame({"g": ["x", "y", "z"], "v": [3.0, 1.0, 2.0]})
        res = desc.summarize_by_group(df, ["g"], ["v"], ["mean"], order="v_mean")
        assert res["g"].tolist() == ["y", "z", "x"]


class TestValidacionEntradas:
    # F2.3: sin datos válidos no se devuelven NaN con avisos de NumPy, sino un error claro.
    @pytest.mark.parametrize("fn", ["central_tendency", "five_number_summary", "detect_outliers", "summary_statistics"])
    def test_sin_datos_es_error(self, fn):
        with pytest.raises(ValueError):
            getattr(desc, fn)([np.nan, np.nan])

    def test_dispersion_muestral_necesita_dos_datos(self):
        with pytest.raises(ValueError):
            desc.dispersion([5.0])
        assert desc.dispersion([5.0], ddof=0)["variance"] == 0.0


# ---------------------------------------------------------------------------
# knn_impute: imputación KNN con desempate determinista (corrección del CI de F5.1)
# Referencias: sklearn.impute.KNNImputer cuando no hay empates, medias
# calculadas a mano y ejecución completa del cuaderno de preprocesamiento.
# ---------------------------------------------------------------------------

def _datos_sin_empates():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    X.iloc[[3, 10, 20], 0] = np.nan
    X.iloc[[5, 33], 2] = np.nan
    return X


def test_knn_sin_empates_coincide_con_sklearn():
    from sklearn.impute import KNNImputer

    X = _datos_sin_empates()
    ref = KNNImputer(n_neighbors=3).fit_transform(X)
    assert np.allclose(desc.knn_impute(X).to_numpy(), ref, rtol=0, atol=1e-12)


def test_knn_empate_entre_mas_de_tres_vecinos_elige_los_de_menor_id():
    # Cinco donantes a distancia exactamente 0 del receptor (fila con ID 100).
    datos = pd.DataFrame({"v": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], "w": [0.0] * 6})
    ids = [100, 10, 3, 7, 1, 5]
    res = desc.knn_impute(datos, n_neighbors=3, ids=ids)
    # Menores ID: 1 (v=4), 3 (v=2) y 5 (v=5) -> media 11/3
    assert res.loc[0, "v"] == pytest.approx(11 / 3)
    # Otro conjunto de identificadores cambia la elección de forma predecible
    res2 = desc.knn_impute(datos, n_neighbors=3, ids=[100, 1, 2, 3, 4, 5])
    assert res2.loc[0, "v"] == pytest.approx((1 + 2 + 3) / 3)


def test_knn_independiente_del_orden_de_las_filas():
    rng = np.random.default_rng(1)
    base = pd.DataFrame({
        "x": np.r_[np.full(8, 0.5), rng.normal(size=12)],
        "y": np.r_[np.zeros(8), rng.normal(size=12)],
        "z": np.r_[np.arange(8, dtype=float), rng.normal(size=12)],
    })
    base.loc[[0, 15], "z"] = np.nan  # el receptor 0 tiene 7 donantes empatados
    ids = pd.Index(rng.permutation(np.arange(200, 220)))
    referencia = desc.knn_impute(base, ids=ids).set_axis(ids)
    for semilla in range(5):
        perm = np.random.default_rng(semilla).permutation(len(base))
        barajado = base.iloc[perm].reset_index(drop=True)
        res = desc.knn_impute(barajado, ids=ids[perm]).set_axis(ids[perm])
        pd.testing.assert_frame_equal(res.loc[referencia.index], referencia, check_exact=True)


def test_knn_repetible_y_conserva_filas_ids_y_observados():
    X = _datos_sin_empates()
    X.iloc[:6, 1] = 0.0  # crea empates adicionales
    ids = pd.Index(np.arange(1000, 1060))
    a = desc.knn_impute(X, ids=ids)
    b = desc.knn_impute(X, ids=ids)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.shape == X.shape and a.index.equals(X.index) and list(a.columns) == list(X.columns)
    assert not a.isna().any().any()
    observados = X.notna()
    pd.testing.assert_frame_equal(a[observados], X[observados], check_exact=True)


def test_knn_ids_invalidos():
    X = _datos_sin_empates()
    with pytest.raises(ValueError):
        desc.knn_impute(X, ids=range(5))
    with pytest.raises(ValueError):
        desc.knn_impute(X, ids=[1] * len(X))


# Aviso de zmq exclusivo de Windows al lanzar el kernel (no afecta al resultado)
@pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")
def test_preprocesamiento_reproduce_el_csv_versionado(tmp_path):
    """Ejecuta el cuaderno de preprocesamiento en una copia y compara su CSV
    con el versionado (la misma comprobación estricta que hace el CI)."""
    import shutil
    from pathlib import Path

    import nbformat
    from nbclient import NotebookClient

    raiz = Path(__file__).resolve().parent.parent
    origen = raiz / "notebooks" / "01_estadistica_descriptiva"
    destino = tmp_path / "notebooks" / "01_estadistica_descriptiva"
    shutil.copytree(raiz / "src", tmp_path / "src")
    (destino / "data").mkdir(parents=True)
    shutil.copy(origen / "data" / "sleephealth.xlsx", destino / "data")
    shutil.copy(origen / "practica_preprocesamiento.ipynb", destino)

    nb = nbformat.read(destino / "practica_preprocesamiento.ipynb", as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(destino)}}).execute()

    leer = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[""])  # noqa: E731
    versionado = leer(origen / "data" / "sleephealth_processed.csv")
    regenerado = leer(destino / "data" / "sleephealth_processed.csv")
    assert len(regenerado) == 373 and regenerado["ID"].is_unique
    pd.testing.assert_frame_equal(versionado, regenerado, check_exact=False, rtol=1e-9)
