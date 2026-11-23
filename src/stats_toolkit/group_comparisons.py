"""
group_comparisons.py

Funciones para el Módulo 6 — ANOVA, Chi-cuadrado y Pruebas No Paramétricas.
Incluye ANOVA de un factor (implementado manualmente, con p-valor vía
scipy.stats.f), test Chi-cuadrado de independencia sobre tablas de
contingencia, y el test de Mann-Whitney U como alternativa no paramétrica
al t-test de dos muestras. Ampliación (fase 3): prueba de signos.

Requiere scipy (scipy.stats) y pandas (ya usado desde el Módulo 1).
"""

import numpy as np
import pandas as pd
from scipy import stats

from ._validation import as_sample, check_alternative


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

    groups = [as_sample(g, f"el grupo {i + 1}") for i, g in enumerate(groups)]
    k = len(groups)
    n_total = sum(len(g) for g in groups)

    grand_mean = np.concatenate(groups).mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0:
        raise ValueError("No hay variabilidad dentro de los grupos: el estadístico F no está definido.")

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


def chi2_test_independence(
    contingency_table, correction: bool = False
) -> tuple[float, float, int, np.ndarray]:
    """
    Test Chi-cuadrado de independencia sobre una tabla de contingencia.

    `contingency_table` puede ser un DataFrame de pandas (como el que
    devuelve `build_contingency_table`) o un array 2D de frecuencias.

    Por defecto calcula el estadístico de Pearson tal como se define en la
    teoría, sum((O - E)^2 / E), sin corrección de continuidad. Con
    ``correction=True`` aplica la corrección de Yates, que solo afecta a
    tablas 2×2 (resta 0,5 a cada |O - E| antes de elevar al cuadrado).
    Atención: ``scipy.stats.chi2_contingency`` la aplica por defecto en
    tablas 2×2, así que sus resultados difieren de los de esta función
    salvo que se le pase ``correction=False``.

    Devuelve (chi2_stat, p_value, degrees_of_freedom, expected_frequencies).
    """
    table = np.asarray(contingency_table)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(table, correction=correction)
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
    check_alternative(alternative)
    x, y = as_sample(x, "x", min_n=1), as_sample(y, "y", min_n=1)
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(u_stat), float(p_value)


# ---------------------------------------------------------------------------
# Tamaño del efecto (F2.2)
# ---------------------------------------------------------------------------

def eta_squared(*groups) -> float:
    """Tamaño del efecto del ANOVA: eta^2 = SS_entre / SS_total.

    Proporción de la variabilidad total de los datos que se asocia a las
    diferencias entre grupos (entre 0 y 1). Describe la muestra; con pocos
    datos tiende a sobrestimar el efecto poblacional.
    """
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos.")
    groups = [as_sample(g, f"el grupo {i + 1}", min_n=1) for i, g in enumerate(groups)]
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = np.sum((all_values - grand_mean) ** 2)
    if ss_total == 0:
        raise ValueError("Todos los valores son iguales: eta^2 no está definido.")
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def cramers_v(contingency_table) -> float:
    """V de Cramér: tamaño del efecto para una tabla de contingencia.

    V = sqrt(chi2 / (n · (min(filas, columnas) - 1))), con el chi2 de Pearson
    sin corrección. Vale 0 si hay independencia en la muestra y 1 si la
    asociación es perfecta.
    """
    table = np.asarray(contingency_table, dtype=float)
    if table.ndim != 2 or min(table.shape) < 2:
        raise ValueError("Se necesita una tabla de al menos 2×2.")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


# ---------------------------------------------------------------------------
# Comparaciones post-hoc (F2.4)
# ---------------------------------------------------------------------------

def tukey_hsd(*groups, labels=None, confidence: float = 0.95) -> pd.DataFrame:
    """Comparaciones por pares de Tukey (HSD) tras un ANOVA.

    Compara todas las parejas de medias controlando la probabilidad de cometer
    al menos un error de tipo I en el conjunto de comparaciones. Envuelve
    ``scipy.stats.tukey_hsd``.

    Devuelve un DataFrame con una fila por pareja: grupos, diferencia de medias
    (grupo_1 - grupo_2), p-valor ajustado e intervalo de confianza simultáneo.
    """
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos.")
    groups = [as_sample(g, f"el grupo {i + 1}") for i, g in enumerate(groups)]
    labels = list(labels) if labels is not None else [f"Grupo {i + 1}" for i in range(len(groups))]
    if len(labels) != len(groups):
        raise ValueError("labels debe tener un nombre por grupo.")
    res = stats.tukey_hsd(*groups)
    ci = res.confidence_interval(confidence_level=confidence)
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            rows.append({
                "grupo_1": labels[i],
                "grupo_2": labels[j],
                "diferencia": float(groups[i].mean() - groups[j].mean()),
                "p_valor": float(res.pvalue[i, j]),
                "ic_inferior": float(ci.low[i, j]),
                "ic_superior": float(ci.high[i, j]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Más pruebas no paramétricas (F2.4)
# ---------------------------------------------------------------------------

def kruskal_wallis(*groups) -> tuple[float, float]:
    """Test de Kruskal-Wallis: alternativa basada en rangos al ANOVA de un factor.

    H0: todos los grupos proceden de la misma distribución. Como Mann-Whitney,
    es sensible sobre todo a que unos grupos tiendan a tomar valores mayores
    que otros. Envuelve ``scipy.stats.kruskal`` (con corrección por empates).

    Devuelve (h_stat, p_value).
    """
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos.")
    groups = [as_sample(g, f"el grupo {i + 1}", min_n=1) for i, g in enumerate(groups)]
    h_stat, p_value = stats.kruskal(*groups)
    return float(h_stat), float(p_value)


def wilcoxon_signed_rank(x, y=None, alternative: str = "two-sided") -> tuple[float, float]:
    """Test de Wilcoxon de rangos con signo (una muestra o datos emparejados).

    Con ``y`` analiza las diferencias x - y; sin ``y``, los valores de x (frente
    a 0). Usa los rangos de las diferencias no nulas. Para interpretarlo como
    un contraste sobre la mediana hace falta suponer que las diferencias se
    distribuyen de forma simétrica. Envuelve ``scipy.stats.wilcoxon``.

    Devuelve (w_stat, p_value).
    """
    check_alternative(alternative)
    x = as_sample(x, "x", min_n=1)
    if y is not None:
        y = as_sample(y, "y", min_n=1)
        if x.size != y.size:
            raise ValueError("x e y deben tener la misma longitud (una pareja por observación).")
        d = x - y
    else:
        d = x
    if np.all(d == 0):
        raise ValueError("Todas las diferencias son 0: el contraste no está definido.")
    w_stat, p_value = stats.wilcoxon(d, alternative=alternative)
    return float(w_stat), float(p_value)


# ---------------------------------------------------------------------------
# Ampliación (F3.4): prueba de signos
# ---------------------------------------------------------------------------

def sign_test(x, y=None, alternative: str = "two-sided") -> tuple[int, float]:
    """Prueba de signos (una muestra frente a 0, o datos emparejados).

    Parameters
    ----------
    x : array_like
        Valores, o primera medida de cada pareja.
    y : array_like, optional
        Segunda medida de cada pareja; se analizan las diferencias ``x - y``.
    alternative : {"two-sided", "greater", "less"}
        ``"greater"``: las diferencias tienden a ser positivas.

    Returns
    -------
    tuple
        ``(k, p_value)``: ``k`` es el número de diferencias positivas entre
        las ``m`` no nulas (los ceros se descartan) y el p-valor es exacto,
        de la Binomial(m, 1/2) que sigue ``k`` bajo H0 (mediana de las
        diferencias igual a 0).
    """
    check_alternative(alternative)
    x = as_sample(x, "x", min_n=1)
    if y is not None:
        y = as_sample(y, "y", min_n=1)
        if x.size != y.size:
            raise ValueError("x e y deben tener la misma longitud (una pareja por observación).")
        d = x - y
    else:
        d = x
    d = d[d != 0]
    if d.size == 0:
        raise ValueError("Todas las diferencias son 0: el contraste no está definido.")
    k = int(np.sum(d > 0))
    p_value = stats.binomtest(k, d.size, 0.5, alternative=alternative).pvalue
    return k, float(p_value)
