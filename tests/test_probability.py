"""
Tests de ``stats_toolkit.probability`` (Módulo 2).

Referencias independientes: probabilidades exactas con ``fractions.Fraction``
y márgenes de error binomiales para las simulaciones de Monte Carlo.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from stats_toolkit import probability as prob


def _within_binomial_margin(p_hat: float, p: float, n: int, k: float = 4.0) -> bool:
    """|p_hat - p| <= k * sqrt(p(1-p)/n): margen de k errores estándar."""
    return abs(p_hat - p) <= k * math.sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestRollDice:
    def test_forma_y_rango(self):
        sums_ = prob.roll_dice(n_dice=3, n_trials=5_000, seed=1)
        assert sums_.shape == (5_000,)
        assert sums_.min() >= 3 and sums_.max() <= 18

    def test_reproducible_con_semilla(self):
        np.testing.assert_array_equal(
            prob.roll_dice(2, 1_000, seed=7), prob.roll_dice(2, 1_000, seed=7)
        )

    def test_frecuencia_de_suma_7_converge_a_un_sexto(self):
        n = 200_000
        sums_ = prob.roll_dice(n_dice=2, n_trials=n, seed=42)
        p_hat = np.mean(sums_ == 7)
        assert _within_binomial_margin(p_hat, 6 / 36, n)


class TestEmpiricalProbability:
    def test_proporcion_exacta(self):
        assert prob.empirical_probability([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0) == 0.5


class TestMonteCarloEstimate:
    def test_estimacion_dentro_del_margen_binomial(self):
        n = 20_000
        p_hat = prob.monte_carlo_estimate(lambda: np.random.rand() < 0.3, n_trials=n, seed=3)
        assert _within_binomial_margin(p_hat, 0.3, n)

    def test_semilla_reproducible(self):
        # F2.3: trial_fn usa el estado global de np.random (es la API pública),
        # así que `seed` fija ese estado: misma semilla, misma estimación.
        trial = lambda: np.random.rand() < 0.3  # noqa: E731
        a = prob.monte_carlo_estimate(trial, n_trials=2_000, seed=7)
        b = prob.monte_carlo_estimate(trial, n_trials=2_000, seed=7)
        assert a == b


# ---------------------------------------------------------------------------
# Probabilidad condicional, total y Bayes
# ---------------------------------------------------------------------------


class TestConditionalProbability:
    def test_as_dada_carta_roja(self):
        # P(As | Roja) = P(As ∩ Roja) / P(Roja) = (2/52) / (26/52) = 1/13
        assert prob.conditional_probability(2 / 52, 26 / 52) == pytest.approx(1 / 13)

    def test_condicion_con_probabilidad_cero_es_error(self):
        with pytest.raises(ValueError):
            prob.conditional_probability(0.0, 0.0)


class TestTotalProbabilityAndBayes:
    # Ejemplo del test médico de la teoría del M2.
    PREV, SENS, FP = Fraction(1, 100), Fraction(95, 100), Fraction(5, 100)

    def test_probabilidad_total(self):
        expected = self.SENS * self.PREV + self.FP * (1 - self.PREV)  # = 59/1000
        res = prob.total_probability([0.95, 0.05], [0.01, 0.99])
        assert res == pytest.approx(float(expected))

    def test_posterior_del_test_medico(self):
        evidence = self.SENS * self.PREV + self.FP * (1 - self.PREV)
        expected = self.SENS * self.PREV / evidence  # = 95/590 ≈ 0.1610
        res = prob.bayes_theorem(prior=0.01, likelihood=0.95, evidence=0.059)
        assert res == pytest.approx(float(expected))

    def test_longitudes_distintas_es_error(self):
        with pytest.raises(ValueError):
            prob.total_probability([0.5, 0.5], [1.0])

    def test_priors_que_no_suman_uno_es_error(self):
        with pytest.raises(ValueError):
            prob.total_probability([0.5, 0.5], [0.5, 0.4])


# ---------------------------------------------------------------------------
# Naive Bayes de Bernoulli con suavizado de Laplace
# ---------------------------------------------------------------------------


class TestSimpleTextNaiveBayes:
    """
    Corpus mínimo, resuelto a mano:
        "a b" -> s, "a c" -> s, "d" -> h;   vocabulario {a, b, c, d}, alpha = 1.
    P(s) = 2/3, P(h) = 1/3.
    P(w | c) = (n.º docs de c con w + 1) / (n.º docs de c + 2):
        s: a = 3/4, b = 2/4, c = 2/4, d = 1/4
        h: a = 1/3, b = 1/3, c = 1/3, d = 2/3
    """

    MSGS = ["a b", "a c", "d"]
    LABELS = ["s", "s", "h"]
    LIK = {
        "s": {"a": Fraction(3, 4), "b": Fraction(1, 2), "c": Fraction(1, 2), "d": Fraction(1, 4)},
        "h": {"a": Fraction(1, 3), "b": Fraction(1, 3), "c": Fraction(1, 3), "d": Fraction(2, 3)},
    }
    PRIOR = {"s": Fraction(2, 3), "h": Fraction(1, 3)}

    def _posterior_manual(self, present: set[str]) -> dict[str, float]:
        scores = {}
        for c in ("s", "h"):
            score = self.PRIOR[c]
            for w, p in self.LIK[c].items():
                score *= p if w in present else (1 - p)
            scores[c] = score
        total = sum(scores.values())
        return {c: float(v / total) for c, v in scores.items()}

    @pytest.fixture
    def modelo(self):
        return prob.SimpleTextNaiveBayes(alpha=1.0).fit(self.MSGS, self.LABELS)

    def test_priors_y_verosimilitudes(self, modelo):
        assert modelo.class_priors["s"] == pytest.approx(2 / 3)
        for c in ("s", "h"):
            for w, p in self.LIK[c].items():
                assert modelo.word_likelihoods[c][w] == pytest.approx(float(p))

    @pytest.mark.parametrize("mensaje,presentes", [("a", {"a"}), ("d b", {"b", "d"})])
    def test_posterior_coincide_con_calculo_manual(self, modelo, mensaje, presentes):
        res = modelo.predict_proba(mensaje)
        expected = self._posterior_manual(presentes)
        assert res["s"] == pytest.approx(expected["s"])
        assert res["h"] == pytest.approx(expected["h"])
        assert sum(res.values()) == pytest.approx(1.0)

    def test_prediccion_es_la_clase_mas_probable(self, modelo):
        expected = self._posterior_manual({"a"})
        assert modelo.predict("a") == max(expected, key=expected.get)

    def test_palabras_fuera_del_vocabulario_se_ignoran(self, modelo):
        assert modelo.predict_proba("a zzz") == pytest.approx(modelo.predict_proba("a"))

    def test_tokenizacion_insensible_a_mayusculas(self, modelo):
        assert modelo.predict_proba("A B") == pytest.approx(modelo.predict_proba("a b"))
