"""
group_comparisons.py

Funciones para el Módulo 6 — ANOVA, Chi-cuadrado y Pruebas No Paramétricas.
Incluye ANOVA de un factor (implementado manualmente, con p-valor vía
scipy.stats.f), test Chi-cuadrado de independencia sobre tablas de
contingencia, y el test de Mann-Whitney U como alternativa no paramétrica
al t-test de dos muestras.

Requiere scipy (scipy.stats) y pandas (ya usado desde el Módulo 1).
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# ANOVA de un factor
# ---------------------------------------------------------------------------

def anova_one_way(*groups) -> tuple[float, float]:
    """
    ANOVA de un factor sobre 2 o más grupos independientes.

    Cada argumento posicional es una secuencia de valores numéricos
    (un grupo). Devuelve (f_stat, p_value).
    """
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos para un ANOVA")

    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    n_total = sum(len(g) for g in groups)

    grand_mean = np.concatenate(groups).mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / ms_within
    p_value = stats.f.sf(f_stat, df_between, df_within)

    return float(f_stat), float(p_value)


# ---------------------------------------------------------------------------
# Chi-cuadrado
# ---------------------------------------------------------------------------

def build_contingency_table(categorical_x, categorical_y) -> pd.DataFrame:
    """Construye una tabla de contingencia (frecuencias cruzadas) a partir
    de dos secuencias de valores categóricos de la misma longitud."""
    x = pd.Series(categorical_x, name="X")
    y = pd.Series(categorical_y, name="Y")
    return pd.crosstab(x, y)


def chi2_test_independence(contingency_table) -> tuple[float, float, int, np.ndarray]:
    """
    Test Chi-cuadrado de independencia sobre una tabla de contingencia.

    `contingency_table` puede ser un DataFrame de pandas (como el que
    devuelve `build_contingency_table`) o un array 2D de frecuencias.

    Devuelve (chi2_stat, p_value, degrees_of_freedom, expected_frequencies).
    """
    table = np.asarray(contingency_table)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(table)
    return float(chi2_stat), float(p_value), int(dof), expected


def chi2_goodness_of_fit(observed, expected=None) -> tuple[float, float]:
    """
    Test Chi-cuadrado de bondad de ajuste: ¿las frecuencias observadas
    encajan con unas frecuencias esperadas (por defecto, uniformes)?

    Devuelve (chi2_stat, p_value).
    """
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(chi2_stat), float(p_value)


# ---------------------------------------------------------------------------
# Pruebas no paramétricas
# ---------------------------------------------------------------------------

def mann_whitney_u(x, y, alternative: str = "two-sided") -> tuple[float, float]:
    """
    Test de Mann-Whitney U: alternativa no paramétrica al t-test de dos
    muestras independientes, basada en rangos en vez de en medias.

    Devuelve (u_stat, p_value).
    """
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(u_stat), float(p_value)
