"""
_pvalues.py

Utilidad interna (privada) de ``stats_toolkit``: cálculo del p-valor de un
contraste a partir de su estadístico y de la distribución de referencia bajo
H0. La usan ``inference`` (Módulo 4) y ``regression`` (Módulo 5).
"""

from __future__ import annotations


def p_value_from_statistic(statistic: float, dist, alternative: str = "two-sided") -> float:
    """
    p-valor de un contraste a partir de su estadístico.

    Parameters
    ----------
    statistic : float
        Valor observado del estadístico de contraste.
    dist : distribución congelada de ``scipy.stats``
        Distribución del estadístico bajo H0, p. ej. ``stats.norm()`` o
        ``stats.t(df)``. Debe ser continua y simétrica respecto a 0 para el
        contraste bilateral.
    alternative : {"two-sided", "greater", "less"}
        - ``"two-sided"``: 2 · P(T >= |t|).
        - ``"greater"`` (cola derecha): P(T >= t).
        - ``"less"`` (cola izquierda): P(T <= t).

    Returns
    -------
    float
        p-valor en [0, 1].

    Notes
    -----
    Las colas derechas se calculan con la función de supervivencia
    ``dist.sf(x) = P(T > x)`` en lugar de ``1 - dist.cdf(x)``: cuando
    ``cdf(x)`` está muy cerca de 1, la resta pierde las cifras significativas
    (con la Normal, ``1 - cdf(z)`` vale exactamente 0 desde z ≈ 8,3). La cola
    izquierda ya es precisa con ``cdf``. Aun así, un p-valor demasiado
    pequeño para representarse se devuelve como 0.0: es una limitación de la
    aritmética de punto flotante, no un p-valor realmente nulo. El límite
    depende de la precisión numérica y de la implementación de cada
    distribución, así que no hay un umbral universal garantizado. Como
    referencia orientativa, con SciPy 1.18 y la Normal ocurre a partir de
    |z| ≈ 37,7 (p del orden de 1e-310).
    """
    if alternative == "two-sided":
        p = 2 * dist.sf(abs(statistic))
    elif alternative == "greater":
        p = dist.sf(statistic)
    elif alternative == "less":
        p = dist.cdf(statistic)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")
    return float(p)
