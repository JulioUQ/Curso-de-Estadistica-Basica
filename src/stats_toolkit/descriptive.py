"""
descriptive.py
---------------
Funciones modulares de estadística descriptiva para el Módulo 1
del repositorio "Estadística Básica".

Incluye dos bloques:

1. Estadística descriptiva clásica sobre arrays numéricos (tendencia
   central, dispersión, forma, resumen de cinco números, outliers).
2. Utilidades de exploración y limpieza de DataFrames, adaptadas de
   ``tidy_functions.py``, pensadas para los notebooks de
   preprocesamiento (detección de duplicados, resumen de columnas,
   categorías únicas, resúmenes agrupados).

Nota: las funciones de ``tidy_functions.py`` que dependen de
``xlwings`` (``xlimport``, ``xlexport``) y las de importación/
exportación masiva de Excel se han dejado fuera deliberadamente.
``xlwings`` requiere una aplicación de Excel activa y no funciona en
Linux ni en el runner de GitHub Actions; importarlo aquí rompería el
despliegue del sitio.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from numbers import Number
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Iterable[float], np.ndarray]


# =======================================================================
# 1. Estadística descriptiva sobre arrays numéricos
# =======================================================================


def _to_array(data: ArrayLike) -> np.ndarray:
    """Convierte cualquier ArrayLike a un np.ndarray de floats, sin NaNs."""
    arr = np.asarray(data, dtype=float)
    return arr[~np.isnan(arr)]


def _require_n(arr: np.ndarray, minimum: int = 1) -> None:
    """Lanza ValueError si, tras quitar los NaN, quedan menos de ``minimum`` datos."""
    if arr.size < minimum:
        raise ValueError(
            f"Se necesitan al menos {minimum} valores no ausentes (hay {arr.size})."
        )


def modes(data: Iterable) -> list:
    """
    Devuelve las modas (valores más frecuentes) de un conjunto de datos
    numéricos o categóricos.

    Reglas:

    - Una moda: lista con ese valor.
    - Varias modas: si varios valores comparten la frecuencia máxima y esa
      frecuencia es mayor que la de algún otro valor, se devuelven todos.
    - Sin moda destacada: si todos los valores distintos aparecen el mismo
      número de veces (por ejemplo, todos son distintos, o 1, 1, 2, 2), se
      devuelve una lista vacía. Unos datos constantes sí tienen moda.
    - Sin observaciones válidas: lista vacía.

    Considerar que no hay moda cuando todos los valores tienen la misma
    frecuencia es una **convención didáctica del curso**, no una definición
    universal: ``pandas.Series.mode()`` y ``statistics.multimode()`` siguen
    otra convención y, en ese caso, devuelven todos los valores.

    Los valores se comparan por igualdad exacta de Python, sin agrupar,
    redondear ni convertir tipos antes de contar: ``1`` y ``1.0`` son el
    mismo valor, pero ``2**53`` y ``2**53 + 1``, o ``Fraction(1, 3)`` y
    ``1/3``, son distintos. Con datos continuos el resultado depende de la
    precisión con que se registraron (por ejemplo, salarios en céntimos
    suelen ser todos distintos). Para describir dónde se concentran unos
    datos continuos conviene agrupar en intervalos y buscar la clase modal.

    Parameters
    ----------
    data : Iterable
        Valores numéricos o categóricos (texto), homogéneos. Los valores
        ausentes (None, NaN, ``pd.NA``) se descartan.

    Returns
    -------
    list
        Modas en orden ascendente, con su tipo original (``int``,
        ``float``, ``Fraction``, ``str``…). Los escalares de NumPy se
        convierten al tipo equivalente de Python sin perder precisión
        (``np.int64`` → ``int``, ``np.float64`` → ``float``). Si se cuentan
        juntos valores iguales de tipos distintos (``1`` y ``1.0``), se
        devuelve el de tipo más exacto (``int``, luego ``Fraction``, luego
        ``float``), de modo que el resultado no depende del orden.

    Raises
    ------
    TypeError
        Si se mezclan valores numéricos y no numéricos.
    """
    valid = [v.item() if isinstance(v, np.generic) else v for v in data if not pd.isna(v)]
    if not valid:
        return []

    numeric = [isinstance(v, Number) and not isinstance(v, bool) for v in valid]
    if any(numeric) and not all(numeric):
        raise TypeError("modes() necesita datos homogéneos: todos numéricos o todos categóricos.")

    # Se cuenta con los valores originales (igualdad y hash de Python).
    counts = Counter(valid)

    # Representante de cada valor: si coinciden valores iguales de tipos
    # distintos (1 y 1.0), se elige el de tipo más exacto, sin depender del
    # orden de aparición.
    representative: dict = {}
    for v in valid:
        if v not in representative or _type_rank(v) < _type_rank(representative[v]):
            representative[v] = v

    if len(counts) == 1:
        return [representative[next(iter(counts))]]

    max_count = max(counts.values())
    if min(counts.values()) == max_count:
        return []  # todos los valores distintos tienen la misma frecuencia
    return sorted(representative[v] for v, c in counts.items() if c == max_count)


def _type_rank(value) -> int:
    """Prioridad de tipo para elegir representante entre valores iguales."""
    for rank, kind in enumerate((int, Fraction, float)):
        if type(value) is kind:
            return rank
    return 3


def central_tendency(data: ArrayLike) -> dict:
    """
    Calcula media, mediana y moda de un conjunto de datos numéricos.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos (lista, array o Serie de pandas). Los NaN se
        descartan.

    Returns
    -------
    dict
        {'mean': float, 'median': float, 'mode': float}

        ``mode`` es la moda cuando es única y NaN cuando no existe una moda
        única (varias modas o ninguna destacada; véase :func:`modes`, que
        devuelve todas las modas y explica la convención).
    """
    arr = _to_array(data)
    _require_n(arr)
    found = modes(arr)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "mode": float(found[0]) if len(found) == 1 else float("nan"),
    }


def dispersion(data: ArrayLike, ddof: int = 1) -> dict:
    """
    Calcula rango, varianza, desviación estándar y coeficiente de
    variación de un conjunto de datos numéricos.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos.
    ddof : int, default 1
        Grados de libertad. Usa 1 para varianza/desviación muestral
        (corrección de Bessel) o 0 para la versión poblacional.

    Returns
    -------
    dict
        {'range': float, 'variance': float, 'std': float, 'cv_pct': float}
    """
    arr = _to_array(data)
    _require_n(arr, ddof + 1)
    variance = float(np.var(arr, ddof=ddof))
    std = float(np.std(arr, ddof=ddof))
    mean = float(np.mean(arr))
    cv_pct = (std / mean) * 100 if mean != 0 else float("nan")

    return {
        "range": float(np.ptp(arr)),
        "variance": variance,
        "std": std,
        "cv_pct": cv_pct,
    }


def shape(data: ArrayLike) -> dict:
    """
    Calcula la asimetría (skewness) y la curtosis de exceso con los
    estimadores corregidos G1 y G2 (tipo 2 de Joanes y Gill, 1998).

    A partir de los momentos centrales muestrales
    ``m_k = (1/n) * sum((x_i - media)**k)`` se definen los coeficientes
    de momentos::

        g1 = m3 / m2**(3/2)
        g2 = m4 / m2**2 - 3

    y, sobre ellos, los estimadores corregidos que se devuelven::

        G1 = sqrt(n * (n - 1)) / (n - 2) * g1                 (n >= 3)
        G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)  (n >= 4)

    La corrección depende solo de n. En G1 es un factor multiplicativo; en
    G2 se multiplica y además se suma una constante, por lo que con
    muestras pequeñas G2 puede tener distinto signo que g2.

    Propiedades: si la población es normal, G2 es insesgado (g2 tiene media
    -6/(n+1)) y en la asimetría no hay sesgo que corregir (g1 y G1 tienen
    media 0). Fuera de la normalidad, las correcciones no garantizan
    eliminar el sesgo ni mejorar el estimador en todos los casos; además,
    G1 y G2 tienen más variabilidad que g1 y g2. La diferencia entre ambas
    versiones desaparece al crecer n, más despacio en la curtosis.

    Son los mismos valores que calculan ``pandas.Series.skew()`` /
    ``.kurt()``, Excel, SPSS y ``scipy.stats.skew/kurtosis`` con
    ``bias=False``.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos. Los NaN se descartan antes de calcular, y n es
        el número de observaciones válidas.

    Returns
    -------
    dict
        {'skewness': float, 'kurtosis_excess': float}

        Cada medida vale NaN cuando no está definida, de forma
        independiente: la asimetría si n < 3 y la curtosis si n < 4.
        Con datos constantes (dispersión nula) ninguna está definida.

    References
    ----------
    Joanes, D. N. y Gill, C. A. (1998). Comparing measures of sample
    skewness and kurtosis. *The Statistician*, 47(1), 183-189.
    """
    arr = _to_array(data)
    n = arr.size
    nan = float("nan")

    # Sin datos o con datos constantes, m2 = 0 y ambos cocientes son 0/0.
    if n == 0 or np.all(arr == arr[0]):
        return {"skewness": nan, "kurtosis_excess": nan}

    dev = arr - arr.mean()
    m2 = np.mean(dev**2)
    m3 = np.mean(dev**3)
    m4 = np.mean(dev**4)

    g1 = m3 / m2**1.5
    g2 = m4 / m2**2 - 3

    skewness = g1 * np.sqrt(n * (n - 1)) / (n - 2) if n >= 3 else nan
    kurtosis_excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6) if n >= 4 else nan

    return {
        "skewness": float(skewness),
        "kurtosis_excess": float(kurtosis_excess),
    }


def five_number_summary(data: ArrayLike) -> dict:
    """
    Calcula el resumen de cinco números de Tukey (mínimo, Q1, mediana,
    Q3, máximo) y el rango intercuartílico (IQR).

    Returns
    -------
    dict
        {'min': float, 'q1': float, 'median': float, 'q3': float,
         'max': float, 'iqr': float}
    """
    arr = _to_array(data)
    _require_n(arr)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(arr)),
        "iqr": float(q3 - q1),
    }


def detect_outliers(data: ArrayLike) -> np.ndarray:
    """
    Detecta outliers usando la regla de Tukey (1.5 * IQR).

    Returns
    -------
    np.ndarray
        Subconjunto de `data` considerado atípico.
    """
    arr = _to_array(data)
    _require_n(arr)
    summary = five_number_summary(arr)
    lower_bound = summary["q1"] - 1.5 * summary["iqr"]
    upper_bound = summary["q3"] + 1.5 * summary["iqr"]
    return arr[(arr < lower_bound) | (arr > upper_bound)]


def summary_statistics(data: ArrayLike, ddof: int = 1) -> dict:
    """
    Combina central_tendency, dispersion, shape y five_number_summary
    en un único diccionario de resumen completo.

    Parameters
    ----------
    data : ArrayLike
        Datos numéricos.
    ddof : int, default 1
        Grados de libertad para la varianza, la desviación estándar y el
        coeficiente de variación (``dispersion``). No afecta a las medidas
        de forma: ``shape`` usa siempre los estimadores corregidos G1 y G2.

    Returns
    -------
    dict
        Diccionario anidado con todas las métricas descriptivas.
    """
    arr = _to_array(data)
    _require_n(arr)
    return {
        "n": int(arr.size),
        "central_tendency": central_tendency(arr),
        "dispersion": dispersion(arr, ddof=ddof),
        "shape": shape(arr),
        "five_number_summary": five_number_summary(arr),
    }


# =======================================================================
# 2. Exploración y limpieza de DataFrames (adaptado de tidy_functions.py)
# =======================================================================


def describe_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen exploratorio de un DataFrame: tipo de dato, nulos, valores
    únicos, categorías más frecuentes y estadísticos básicos de las
    columnas numéricas.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Una fila por columna de `data`, con metadatos y estadísticos.
    """
    total = len(data)

    summary = pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str),
            "Non-null Count": data.count().values,
            "% Null Values": ((data.isnull().sum() / total) * 100).round(2).values,
            "Unique Values": data.nunique().values,
        }
    )

    def _top_counts(col: pd.Series) -> str:
        counts = col.value_counts(dropna=True).head(3)
        return ", ".join(f"{idx} ({cnt})" for idx, cnt in counts.items())

    categorical_cols = data.select_dtypes(include=["object", "category", "string"]).columns
    summary["TopCounts"] = None
    for col in categorical_cols:
        summary.loc[summary["Column"] == col, "TopCounts"] = _top_counts(data[col])

    numeric_cols = data.select_dtypes(include=["number"]).columns
    if not numeric_cols.empty:
        describe_stats = (
            data[numeric_cols]
            .describe()
            .T.rename(columns={"50%": "median"})[["mean", "median", "std", "min", "25%", "75%", "max"]]
            .reset_index()
            .rename(columns={"index": "Column"})
        )
        summary = pd.merge(summary, describe_stats, on="Column", how="left")

    return summary


def detect_duplicates(
    df: pd.DataFrame, subset: Sequence[str] | None = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Detecta filas duplicadas en un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str], opcional
        Columnas que definen un duplicado (por ejemplo, un identificador
        como ``['Person.ID']``). Si es None, se comparan todas las
        columnas (duplicado exacto de fila completa).
    verbose : bool, default True
        Si True, imprime un resumen legible.

    Returns
    -------
    pd.DataFrame
        Las filas consideradas duplicadas, ordenadas para que los
        duplicados queden consecutivos.
    """
    duplicates_df = df[df.duplicated(subset=subset, keep=False)].copy()
    n = len(duplicates_df)

    if verbose:
        if n == 0:
            print("No se encontraron duplicados en el DataFrame.")
        else:
            key = subset if subset else "todas las columnas"
            print(f"Se encontraron {n} filas duplicadas según: {key}.")

    if n == 0:
        return duplicates_df

    sort_cols = list(subset) if subset else list(df.columns)
    return duplicates_df.sort_values(by=sort_cols)


def unique_categories(
    df: pd.DataFrame, columns: Sequence[str] | None = None, verbose: bool = True
) -> dict:
    """
    Devuelve los valores únicos de las columnas categóricas de un
    DataFrame (todas por defecto), útil para detectar typos o
    categorías inconsistentes antes de normalizarlas.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], opcional
        Columnas a inspeccionar. Si es None, se usan todas las columnas
        de tipo `object`/`category`/`string`.
    verbose : bool, default True
        Si True, imprime cada columna con sus valores únicos.

    Returns
    -------
    dict
        {nombre_columna: array de valores únicos}
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    result = {}
    for col in columns:
        uniques = df[col].unique()
        result[col] = uniques
        if verbose:
            print(f"'{col}': {sorted(map(str, uniques))}\n")

    return result


def summarize_by_group(
    df: pd.DataFrame,
    levels: Sequence[str],
    metric_cols: Sequence[str],
    metrics,
    order: "str | Sequence[str] | None" = None,
    ascending=True,
) -> pd.DataFrame:
    """
    Agrupa `df` por `levels`, aplica `metrics` sobre `metric_cols` y
    opcionalmente ordena el resultado. Aplana las columnas MultiIndex
    que resultan de aplicar varias métricas a la vez.

    Parameters
    ----------
    df : pd.DataFrame
    levels : list[str]
        Columnas por las que agrupar (ej. ``['SD']``).
    metric_cols : list[str]
        Columnas numéricas sobre las que calcular las métricas.
    metrics : list | dict
        Métricas a aplicar, p. ej. ``['mean', 'std']`` o
        ``{'HR': ['mean', 'std'], 'Steps': 'median'}``.
    order : str | list[str], opcional
        Columna(s) resultantes por las que ordenar.
    ascending : bool | list[bool], default True

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> summarize_by_group(df, ['SD'], ['SH', 'SQ'], ['mean', 'std'])
    """
    summary = df.groupby(list(levels), dropna=False)[list(metric_cols)].agg(metrics).reset_index()

    if isinstance(summary.columns, pd.MultiIndex):
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    if order is not None:
        summary = summary.sort_values(by=order, ascending=ascending)

    return summary


if __name__ == "__main__":
    # Ejemplo de uso rápido con datos simulados de salarios
    np.random.seed(42)
    salarios_ejemplo = np.random.lognormal(mean=7.5, sigma=0.4, size=200)

    resumen = summary_statistics(salarios_ejemplo)

    print("== Resumen estadístico de ejemplo (salarios simulados) ==")
    for seccion, valores in resumen.items():
        print(f"\n{seccion}:")
        if isinstance(valores, dict):
            for k, v in valores.items():
                print(f"  {k}: {v:.2f}")
        else:
            print(f"  {valores}")

    # Ejemplo rápido de las utilidades de exploración de DataFrames
    df_ejemplo = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "categoria": ["A", "B", "B", "A"],
            "valor": [10.0, 20.0, 20.0, np.nan],
        }
    )
    print("\n== describe_df ==")
    print(describe_df(df_ejemplo))
    print("\n== detect_duplicates (subset='id') ==")
    print(detect_duplicates(df_ejemplo, subset=["id"]))
