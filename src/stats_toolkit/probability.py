"""
probability.py
---------------
Funciones modulares de probabilidad para el Módulo 2 del repositorio
"Estadística Básica": simulación de Monte Carlo, probabilidad
condicional, Teorema de Bayes y un clasificador Naive Bayes minimalista
para texto (usado en el caso de uso "filtro de spam").
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------
# Simulación de Monte Carlo
# ---------------------------------------------------------------------


def roll_dice(n_dice: int = 1, n_trials: int = 10_000, seed: int | None = None) -> np.ndarray:
    """
    Simula `n_trials` lanzamientos de `n_dice` dados de 6 caras.

    Returns
    -------
    np.ndarray de forma (n_trials,)
        Suma de los dados obtenida en cada lanzamiento.
    """
    rng = np.random.default_rng(seed)
    rolls = rng.integers(low=1, high=7, size=(n_trials, n_dice))
    return rolls.sum(axis=1)


def empirical_probability(outcomes: Iterable, event: Callable[[float], bool]) -> float:
    """
    Estima P(evento) de forma empírica a partir de una muestra de
    resultados simulados.

    Parameters
    ----------
    outcomes : Iterable
        Resultados simulados (p. ej. salida de `roll_dice`).
    event : Callable
        Función que recibe un resultado y devuelve True/False según si
        pertenece al evento de interés.

    Returns
    -------
    float
        Proporción de resultados que cumplen `event`.
    """
    outcomes = list(outcomes)
    hits = sum(1 for o in outcomes if event(o))
    return hits / len(outcomes)


def monte_carlo_estimate(
    trial_fn: Callable[[], bool], n_trials: int = 10_000, seed: int | None = None
) -> float:
    """
    Estima una probabilidad ejecutando `trial_fn` repetidamente.

    Parameters
    ----------
    trial_fn : Callable
        Función sin argumentos que simula un único ensayo y devuelve
        True si el evento de interés ocurrió.
    n_trials : int
        Número de simulaciones.
    seed : int, opcional
        Semilla para reproducibilidad (afecta a `np.random` si se usa
        dentro de `trial_fn` mediante un generador compartido).

    Returns
    -------
    float
        Proporción estimada de éxitos.
    """
    if seed is not None:
        np.random.seed(seed)
    successes = sum(1 for _ in range(n_trials) if trial_fn())
    return successes / n_trials


# ---------------------------------------------------------------------
# Probabilidad condicional y Teorema de Bayes
# ---------------------------------------------------------------------


def conditional_probability(p_joint: float, p_condition: float) -> float:
    """
    Calcula P(A | B) = P(A ∩ B) / P(B).
    """
    if p_condition == 0:
        raise ValueError("P(B) no puede ser 0 en una probabilidad condicional.")
    return p_joint / p_condition


def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """
    Aplica el Teorema de Bayes: P(A|B) = P(B|A) * P(A) / P(B).

    Parameters
    ----------
    prior : float
        P(A), la probabilidad a priori del evento A.
    likelihood : float
        P(B|A), la probabilidad de observar B dado que ocurrió A.
    evidence : float
        P(B), la probabilidad total de observar B.

    Returns
    -------
    float
        P(A|B), la probabilidad posterior.
    """
    return (likelihood * prior) / evidence


def total_probability(likelihoods: Sequence[float], priors: Sequence[float]) -> float:
    """
    Calcula P(B) = sum_i P(B|A_i) * P(A_i), dado un conjunto de
    hipótesis que particionan el espacio muestral.
    """
    if len(likelihoods) != len(priors):
        raise ValueError("likelihoods y priors deben tener la misma longitud.")
    if not math.isclose(sum(priors), 1.0, abs_tol=1e-6):
        raise ValueError("Los priors deben sumar 1 (deben formar una partición de Ω).")
    return sum(l * p for l, p in zip(likelihoods, priors))


# ---------------------------------------------------------------------
# Naive Bayes minimalista para texto (caso de uso: filtro de spam)
# ---------------------------------------------------------------------


class SimpleTextNaiveBayes:
    """
    Clasificador Naive Bayes Bernoulli minimalista para texto, pensado
    con fines didácticos (no para producción). Estima P(clase) y
    P(palabra | clase) a partir de un corpus de entrenamiento y aplica
    el Teorema de Bayes (con supuesto de independencia condicional)
    para clasificar mensajes nuevos.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float, default 1.0
            Suavizado de Laplace, para evitar probabilidades cero ante
            palabras no vistas en entrenamiento.
        """
        self.alpha = alpha
        self.class_priors: dict[str, float] = {}
        self.word_likelihoods: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    def fit(self, messages: Sequence[str], labels: Sequence[str]) -> "SimpleTextNaiveBayes":
        """Entrena el modelo a partir de mensajes y sus etiquetas de clase."""
        classes = set(labels)
        n_total = len(labels)
        tokenized = [self._tokenize(m) for m in messages]
        self.vocabulary = set.union(*tokenized) if tokenized else set()

        for c in classes:
            class_docs = [tok for tok, lab in zip(tokenized, labels) if lab == c]
            self.class_priors[c] = len(class_docs) / n_total

            word_counts = Counter()
            for doc in class_docs:
                word_counts.update(doc)

            n_docs_c = len(class_docs)
            self.word_likelihoods[c] = {
                word: (word_counts[word] + self.alpha) / (n_docs_c + 2 * self.alpha)
                for word in self.vocabulary
            }

        return self

    def predict_proba(self, message: str) -> dict[str, float]:
        """
        Devuelve la probabilidad posterior (normalizada) de cada clase
        para un mensaje nuevo.
        """
        tokens = self._tokenize(message)
        log_scores: dict[str, float] = {}

        for c, prior in self.class_priors.items():
            log_score = math.log(prior)
            for word in self.vocabulary:
                p_word_given_c = self.word_likelihoods[c][word]
                if word in tokens:
                    log_score += math.log(p_word_given_c)
                else:
                    log_score += math.log(1 - p_word_given_c)
            log_scores[c] = log_score

        # Normalización numéricamente estable (log-sum-exp)
        max_log = max(log_scores.values())
        exp_scores = {c: math.exp(v - max_log) for c, v in log_scores.items()}
        total = sum(exp_scores.values())
        return {c: v / total for c, v in exp_scores.items()}

    def predict(self, message: str) -> str:
        """Devuelve la clase más probable para un mensaje nuevo."""
        proba = self.predict_proba(message)
        return max(proba, key=proba.get)


if __name__ == "__main__":
    # --- Ejemplo rápido: Monte Carlo para P(suma de 2 dados == 7) ---
    sums_ = roll_dice(n_dice=2, n_trials=100_000, seed=42)
    p_seven = empirical_probability(sums_, lambda s: s == 7)
    print(f"P(suma == 7) estimada por Monte Carlo: {p_seven:.4f} (teórica: {6/36:.4f})")

    # --- Ejemplo rápido: Teorema de Bayes (test médico) ---
    prior_enfermo = 0.01
    sensibilidad = 0.95
    falsos_positivos = 0.05
    evidencia = total_probability(
        likelihoods=[sensibilidad, falsos_positivos],
        priors=[prior_enfermo, 1 - prior_enfermo],
    )
    posterior = bayes_theorem(prior_enfermo, sensibilidad, evidencia)
    print(f"P(Enfermo | Test+) = {posterior:.4f}")

    # --- Ejemplo rápido: filtro de spam ---
    mensajes = [
        "gana dinero rapido ahora",
        "oferta exclusiva solo hoy",
        "reunion de trabajo manana",
        "informe de ventas adjunto",
        "gana un premio ahora mismo",
    ]
    etiquetas = ["spam", "spam", "ham", "ham", "spam"]

    modelo = SimpleTextNaiveBayes().fit(mensajes, etiquetas)
    nuevo_mensaje = "gana dinero en la reunion"
    print(f"\nMensaje: '{nuevo_mensaje}'")
    print(f"Probabilidades: {modelo.predict_proba(nuevo_mensaje)}")
    print(f"Predicción: {modelo.predict(nuevo_mensaje)}")
