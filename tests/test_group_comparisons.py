"""
Tests de ``stats_toolkit.group_comparisons`` (Módulo 6).

Referencias independientes: sumas de cuadrados y frecuencias esperadas
calculadas a mano, funciones de supervivencia en forma cerrada
(F(2, d2) y chi-cuadrado con 2 g.l.), la distribución exacta de U de
Mann-Whitney obtenida por enumeración y ``scipy.stats.f_oneway``.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stats_toolkit import group_comparisons as gc

# ---------------------------------------------------------------------------
# ANOVA de un factor
# ---------------------------------------------------------------------------


class TestAnovaOneWay:
    def test_ejemplo_resuelto_a_mano(self):
        # Medias 2, 5, 8; media global 5.
        # SS_entre = 3·(9 + 0 + 9) = 54 ; SS_dentro = 3·2 = 6 ; g.l. (2, 6)
        # F = (54/2) / (6/6) = 27 ; para F(2, d2): P(F > f) = (1 + 2f/d2)^(-d2/2)
        f, p = gc.anova_one_way([1, 2, 3], [4, 5, 6], [7, 8, 9])
        assert f == pytest.approx(27.0)
        assert p == pytest.approx((1 + 2 * 27 / 6) ** (-3))

    def test_frente_a_scipy_con_grupos_desiguales(self):
        rng = np.random.default_rng(2)
        groups = [rng.normal(5, 1, 10), rng.normal(6, 1, 12), rng.normal(5.5, 1, 8)]
        f, p = gc.anova_one_way(*groups)
        ref = stats.f_oneway(*groups)
        assert (f, p) == pytest.approx((ref.statistic, ref.pvalue))

    def test_dos_grupos_equivale_al_t_de_varianzas_iguales(self):
        # Con k = 2, F = t² del t-test agrupado y los p-valores coinciden.
        a, b = [5.1, 4.8, 6.0, 5.5, 5.2], [6.3, 6.8, 5.9, 7.1]
        f, p = gc.anova_one_way(a, b)
        t = stats.ttest_ind(a, b, equal_var=True)
        assert f == pytest.approx(t.statistic**2)
        assert p == pytest.approx(t.pvalue)

    def test_menos_de_dos_grupos_es_error(self):
        with pytest.raises(ValueError):
            gc.anova_one_way([1, 2, 3])


# ---------------------------------------------------------------------------
# Chi-cuadrado
# ---------------------------------------------------------------------------


class TestContingencyTable:
    def test_recuentos(self):
        x = ["J", "J", "A", "A", "A", "S"]
        y = ["D", "F", "D", "D", "F", "F"]
        tabla = gc.build_contingency_table(x, y)
        assert tabla.loc["J", "D"] == 1
        assert tabla.loc["A", "D"] == 2
        assert tabla.loc["S", "F"] == 1
        assert tabla.to_numpy().sum() == len(x)


class TestChi2Independence:
    # Tabla 3×2 del ejemplo de la teoría del M6 (Yates no se aplica fuera de 2×2).
    OBS = np.array([[40, 10], [20, 30], [15, 35]], dtype=float)

    def test_pearson_resuelto_a_mano(self):
        rows = self.OBS.sum(axis=1, keepdims=True)
        cols = self.OBS.sum(axis=0, keepdims=True)
        expected = rows @ cols / self.OBS.sum()
        chi2_manual = ((self.OBS - expected) ** 2 / expected).sum()

        chi2, p, dof, exp = gc.chi2_test_independence(self.OBS)
        assert dof == (3 - 1) * (2 - 1)
        np.testing.assert_allclose(exp, expected)
        assert chi2 == pytest.approx(chi2_manual)
        # Con 2 g.l., P(chi2 > x) = exp(-x / 2)
        assert p == pytest.approx(math.exp(-chi2_manual / 2))

    def test_acepta_dataframe(self):
        df = pd.DataFrame(self.OBS, index=["J", "A", "S"], columns=["D", "F"])
        assert gc.chi2_test_independence(df)[0] == pytest.approx(
            gc.chi2_test_independence(self.OBS)[0]
        )

    # Tablas 2×2 (F1.5): por defecto, Pearson sin corrección, como en la
    # teoría; con correction=True, la corrección de Yates.
    OBS_2X2 = np.array([[12, 5], [7, 16]], dtype=float)

    def _expected(self, obs):
        return obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / obs.sum()

    def test_2x2_por_defecto_sin_correccion(self):
        exp = self._expected(self.OBS_2X2)
        chi2_manual = ((self.OBS_2X2 - exp) ** 2 / exp).sum()
        chi2, p, dof, _ = gc.chi2_test_independence(self.OBS_2X2)
        assert dof == 1
        assert chi2 == pytest.approx(chi2_manual)
        # Con 1 g.l., P(chi2 > x) = erfc(sqrt(x / 2))
        assert p == pytest.approx(math.erfc(math.sqrt(chi2_manual / 2)))

    def test_2x2_con_correccion_de_yates(self):
        exp = self._expected(self.OBS_2X2)
        chi2_yates = ((np.abs(self.OBS_2X2 - exp) - 0.5) ** 2 / exp).sum()
        chi2, p, _, _ = gc.chi2_test_independence(self.OBS_2X2, correction=True)
        assert chi2 == pytest.approx(chi2_yates)
        assert p == pytest.approx(math.erfc(math.sqrt(chi2_yates / 2)))
        assert chi2 < gc.chi2_test_independence(self.OBS_2X2)[0]

    def test_correccion_no_afecta_a_tablas_mayores(self):
        assert gc.chi2_test_independence(self.OBS, correction=True)[0] == pytest.approx(
            gc.chi2_test_independence(self.OBS)[0]
        )


class TestChi2GoodnessOfFit:
    def test_dado_uniforme_resuelto_a_mano(self):
        obs = np.array([10, 12, 8, 15, 9, 6])
        e = obs.sum() / 6
        chi2_manual = ((obs - e) ** 2 / e).sum()
        chi2, p = gc.chi2_goodness_of_fit(obs)
        assert chi2 == pytest.approx(chi2_manual)
        assert p == pytest.approx(stats.chi2.sf(chi2_manual, df=5))

    def test_frecuencias_esperadas_explicitas(self):
        obs = np.array([50, 30, 20])
        exp = np.array([40, 40, 20])
        chi2, _ = gc.chi2_goodness_of_fit(obs, exp)
        assert chi2 == pytest.approx(100 / 40 + 100 / 40 + 0)


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------


def _exact_u_distribution(n1: int, n2: int) -> list[int]:
    """U del primer grupo para cada asignación posible de rangos (sin empates)."""
    ranks = range(1, n1 + n2 + 1)
    return [sum(c) - n1 * (n1 + 1) // 2 for c in combinations(ranks, n1)]


class TestMannWhitneyU:
    X = [1.2, 2.5, 3.1]
    Y = [4.0, 5.5, 6.2]

    def test_p_exacto_por_enumeracion(self):
        # Separación total: U_x = 0, el valor más extremo de los C(6,3) = 20 posibles.
        dist_u = _exact_u_distribution(3, 3)
        p_less = sum(u <= 0 for u in dist_u) / len(dist_u)  # = 1/20
        _, p_two = gc.mann_whitney_u(self.X, self.Y)
        _, p_one = gc.mann_whitney_u(self.X, self.Y, alternative="less")
        assert p_one == pytest.approx(p_less)
        assert p_two == pytest.approx(2 * p_less)

    def test_estadistico_es_un_extremo_de_su_rango(self):
        # Qué U se devuelve (U_x o U_y) no está documentado; con separación
        # total debe ser 0 o n1·n2.
        u, _ = gc.mann_whitney_u(self.X, self.Y)
        assert u in (0, 3 * 3)


# ---------------------------------------------------------------------------
# F2.2 y F2.4: tamaño del efecto, post-hoc y más pruebas no paramétricas
# ---------------------------------------------------------------------------

G1 = np.array([5, 6, 7, 5, 6, 8, 6], dtype=float)
G2 = np.array([8, 9, 7, 10, 8, 9, 9], dtype=float)
G3 = np.array([4, 5, 4, 6, 5, 5, 4], dtype=float)


class TestEtaSquared:
    def test_resuelto_a_mano(self):
        allv = np.concatenate([G1, G2, G3])
        ss_tot = ((allv - allv.mean()) ** 2).sum()
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in (G1, G2, G3))
        assert gc.eta_squared(G1, G2, G3) == pytest.approx(ss_b / ss_tot)

    def test_relacion_con_f(self):
        # eta^2 = F (k-1) / (F (k-1) + (n-k))
        f, _ = gc.anova_one_way(G1, G2, G3)
        k, n = 3, 21
        assert gc.eta_squared(G1, G2, G3) == pytest.approx(f * (k - 1) / (f * (k - 1) + (n - k)))


class TestCramersV:
    def test_frente_a_scipy(self):
        obs = np.array([[40, 10], [20, 30], [15, 35]])
        assert gc.cramers_v(obs) == pytest.approx(stats.contingency.association(obs, method="cramer"))

    def test_asociacion_perfecta_vale_uno(self):
        assert gc.cramers_v(np.array([[10, 0], [0, 10]])) == pytest.approx(1.0)


class TestTukey:
    def test_frente_a_statsmodels(self):
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        res = gc.tukey_hsd(G1, G2, G3, labels=["A", "B", "C"])
        values = np.concatenate([G1, G2, G3])
        groups = ["A"] * 7 + ["B"] * 7 + ["C"] * 7
        ref = pairwise_tukeyhsd(values, groups)
        # statsmodels da (grupo2 - grupo1); el toolkit, (grupo_1 - grupo_2)
        assert res["diferencia"].to_numpy() == pytest.approx(-ref.meandiffs)
        assert res["p_valor"].to_numpy() == pytest.approx(ref.pvalues, abs=1e-3)
        assert list(res["grupo_1"]) == ["A", "A", "B"]


class TestKruskalWallis:
    def test_h_sin_empates_resuelto_a_mano(self):
        a, b, c = [1.1, 2.3, 3.5], [4.2, 5.8, 6.1, 7.7], [8.4, 9.0]
        allv = np.array(a + b + c)
        ranks = stats.rankdata(allv)
        n = allv.size
        sizes = [len(a), len(b), len(c)]
        idx = np.cumsum([0] + sizes)
        h = 12 / (n * (n + 1)) * sum(ranks[idx[i]:idx[i + 1]].sum() ** 2 / sizes[i] for i in range(3)) - 3 * (n + 1)
        h_tk, p_tk = gc.kruskal_wallis(a, b, c)
        assert h_tk == pytest.approx(h)
        assert p_tk == pytest.approx(math.exp(-h / 2))  # chi2 con 2 g.l.


class TestWilcoxon:
    def test_p_exacto_por_enumeracion(self):
        d = np.array([1.5, -0.5, 2.0, 3.0, 2.5, -1.0, 4.0])
        w, p = gc.wilcoxon_signed_rank(d)
        ranks = stats.rankdata(np.abs(d))
        w_plus = ranks[d > 0].sum()
        # Distribución exacta de W+ bajo H0: los 2^n reparticiones de signos
        from itertools import product
        dist = [sum(r for r, s in zip(ranks, signs) if s) for signs in product([0, 1], repeat=len(d))]
        mean = ranks.sum() / 2
        p_exact = np.mean([abs(v - mean) >= abs(w_plus - mean) for v in dist])
        assert w == pytest.approx(min(w_plus, ranks.sum() - w_plus))
        assert p == pytest.approx(p_exact)

    def test_emparejado_equivale_a_diferencias(self):
        x, y = [5.0, 7.0, 6.5, 8.0, 9.5], [4.0, 7.5, 5.0, 6.0, 7.0]
        assert gc.wilcoxon_signed_rank(x, y) == pytest.approx(gc.wilcoxon_signed_rank(np.subtract(x, y)))


class TestValidacionGrupos:
    def test_anova_sin_variabilidad_dentro_es_error(self):
        with pytest.raises(ValueError):
            gc.anova_one_way([1, 1], [2, 2])

    def test_anova_con_nan_es_error(self):
        with pytest.raises(ValueError):
            gc.anova_one_way([1, 2, np.nan], [2, 3, 4])


# ---------------------------------------------------------------------------
# F3.4: prueba de signos
# Referencia: la distribución Binomial(m, 1/2) sumada a mano.
# ---------------------------------------------------------------------------

ANTES = [142, 150, 138, 160, 155, 147, 152, 149]
DESPUES = [138, 145, 139, 151, 150, 140, 149, 147]


def _binom_cdf(k, m):
    return sum(math.comb(m, i) for i in range(k + 1)) / 2**m


def test_prueba_de_signos_exacta():
    k, p = gc.sign_test(ANTES, DESPUES)
    # 7 de 8 diferencias positivas: P(K >= 7) = 9/256; bilateral, el doble
    assert k == 7
    assert p == pytest.approx(2 * 9 / 256)
    assert gc.sign_test(ANTES, DESPUES, "greater")[1] == pytest.approx(1 - _binom_cdf(6, 8))
    assert gc.sign_test(ANTES, DESPUES, "less")[1] == pytest.approx(_binom_cdf(7, 8))


def test_prueba_de_signos_descarta_ceros_y_una_muestra():
    k, p = gc.sign_test([1, -2, 0, 3, 0, 4])
    assert k == 3 and p == pytest.approx(2 * (1 - _binom_cdf(2, 4)))
    assert gc.sign_test(np.array(ANTES) - np.array(DESPUES)) == gc.sign_test(ANTES, DESPUES)


def test_prueba_de_signos_entradas_invalidas():
    with pytest.raises(ValueError):
        gc.sign_test([0, 0, 0])
    with pytest.raises(ValueError):
        gc.sign_test([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        gc.sign_test([1, np.nan])
