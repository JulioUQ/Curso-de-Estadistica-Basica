"""
stats_toolkit: librería del curso de Estadística Básica.

Un módulo por cada módulo del curso:

- ``descriptive`` (M1): tendencia central, dispersión, forma, atípicos.
- ``probability`` (M2): probabilidad empírica, Bayes, Monte Carlo, Naive Bayes.
- ``distributions`` (M3): pmf/pdf/cdf, esperanza y varianza, TCL.
- ``inference`` (M4): intervalos de confianza y contrastes de hipótesis.
- ``regression`` (M5): correlación y regresión lineal simple.
- ``group_comparisons`` (M6): ANOVA, chi-cuadrado y pruebas no paramétricas.

Uso habitual en los capítulos: ``from stats_toolkit import inference as inf``.
Los módulos que empiezan por guion bajo (``_pvalues``, ``_validation``) son
internos.
"""

from . import descriptive, distributions, group_comparisons, inference, probability, regression

__all__ = [
    "descriptive",
    "probability",
    "distributions",
    "inference",
    "regression",
    "group_comparisons",
]
