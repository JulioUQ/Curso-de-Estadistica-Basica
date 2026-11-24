# CLAUDE.md — Curso de Estadística Básica

Guía de trabajo para Claude en este repositorio. Describe el proyecto **tal como
está** (inspeccionado el 2026-11-23, rama `test`, commit `8b62f8c`) y las normas
a seguir en futuras sesiones. Si algo de este documento contradice al código,
manda el código: verifica y actualiza este archivo.

---

## 1. Finalidad y objetivos

- Curso introductorio de estadística, en **español**, orientado a la práctica y
  publicado como **libro web Quarto** en GitHub Pages:
  https://juliouq.github.io/Curso-de-Estadistica-Basica/
- Cada módulo combina **teoría formal** (`teoria.qmd`) con **práctica en Python**
  (`practica.ipynb`), apoyadas en una librería propia `src/stats_toolkit/`.
- Progresión pedagógica lineal: descriptiva → probabilidad → distribuciones →
  inferencia → regresión → ANOVA/χ²/no paramétricas. Cada capítulo teórico
  cierra con un "puente" hacia el siguiente módulo; mantener esa continuidad.
- Autor y responsable de las decisiones de contenido: Julio Úbeda Quesada.
- Material de referencia declarado: apuntes UOC (Bernadó 2017; Vegas 2020) y
  manuales clásicos (Moore et al., Tukey…), ver `references.bib`.

## 2. Arquitectura y estructura

```
_quarto.yml            # Proyecto type: book; capítulos, tema cosmo, freeze: auto
index.qmd              # Portada ("Inicio"), rediseñada en F5.1 (ver §8, «Identidad visual»)
styles.css             # Estilos propios: §0 identidad visual (variables --eb-*,
                       # componentes .eb-*), §1 menú lateral (F5.2), tablas, citas…
assets/                # logo.svg (cabecera del menú) y navegacion.html (script del
                       # menú, incluido con format.html.include-after-body)
references.bib         # Bibliografía BibTeX citada con [@clave]
requirements.txt       # Dependencias Python de ejecución (sin versiones fijadas)
requirements-dev.txt   # Solo desarrollo: pytest
pytest.ini             # Configuración de pytest
tests/                 # conftest.py + test_<módulo>.py, uno por módulo del toolkit
notebooks/
  01_estadistica_descriptiva/  teoria.qmd, practica.ipynb,
                               practica_preprocesamiento.ipynb, data/sleephealth.xlsx,
                               data/sleephealth_processed.csv (generado por el
                               preprocesamiento y VERSIONADO desde F4: lo leen
                               M2, M4, M5, M6 y el proyecto sin depender de
                               ejecutar antes el preprocesamiento)
  02_probabilidad/             teoria.qmd, practica.ipynb
  03_distribuciones/           teoria.qmd, practica.ipynb
  04_inferencia/               teoria.qmd, practica.ipynb,
                               data/sleephealth_processed.xlsx (conservado; no se usa)
  05_regresion/                teoria.qmd, practica.ipynb
  06_anova_chi2/               teoria.qmd, practica.ipynb
  07_proyecto_integrador/      practica.ipynb (F3.5: proyecto final sobre sleephealth;
                               se llama practica.ipynb para que el CI lo ejecute)
apendices/             # F3.6: que_prueba_usar.qmd, notacion.qmd, glosario.qmd
                       # (registrados en `book.appendices` de _quarto.yml)
src/stats_toolkit/     # Librería del curso: un módulo .py por módulo del curso
  descriptive.py (M1) · probability.py (M2) · distributions.py (M3)
  inference.py (M4)   · regression.py (M5)  · group_comparisons.py (M6)
  _pvalues.py          # privado (F1.4): p-valor desde estadístico + distribución scipy;
                       # lo usan inference y regression
  _validation.py       # privado (F2.3): validación de entradas de M4–M6
  __init__.py          # docstring + importa y expone los seis módulos
.github/workflows/
  ci-check.yml         # PR → main: pytest + ejecución de las 8 prácticas + comprobación
                       # de que el CSV regenerado coincide con el versionado +
                       # `quarto render` sin publicar (check "render-check")
  deploy.yml           # push a main: render + publish a la rama gh-pages
```

Artefactos generados (ignorados por git): `_book/`, `.quarto/`, `_freeze/`,
`venv/`, `*.quarto_ipynb`. Desde F2.1 ningún archivo de `_freeze/` está
versionado (los 3 que lo estaban se retiraron del índice con `git rm --cached`;
siguen en disco). Ya no hace falta renderizar el M3 en una copia temporal.

### Cómo se importa `stats_toolkit`
No es un paquete instalable (no hay `pyproject.toml`). Cada `.qmd`/`.ipynb` hace
`sys.path.append("../../src")` (o `os.path.abspath("../../src")`) y luego
`from stats_toolkit import <módulo> as <alias>`. Alias en uso: `desc`, `prob`,
`dist`, `inf`, `reg`, `gc`. Funciona porque Quarto y Jupyter ejecutan con el
directorio del documento como cwd. Si se añade un capítulo nuevo, seguir el mismo
patrón (o proponer antes un cambio global; no mezclar mecanismos).

## 3. Tecnologías y dependencias

- **Quarto** (local: 1.10.18; CI: la última que instale `quarto-actions/setup@v2`).
- **Python**: CI usa **3.12**; el `venv/` local está en **3.13.7**. Escribir código
  compatible con ambas.
- `requirements.txt` (F1.8): numpy, pandas, scipy, statsmodels, scikit-learn,
  openpyxl, matplotlib, seaborn + jupyter, ipykernel, nbclient, nbformat.
  Mínimos = versiones más antiguas **verificadas** en Python 3.12 y 3.13 (pytest
  + comprobación de las APIs usadas): numpy 2.1.3, pandas 2.2.3, scipy 1.15.3,
  statsmodels 0.14.6, scikit-learn 1.5.2, openpyxl 3.1.5, matplotlib 3.9.4,
  seaborn 0.13.2. Sin máximos. Si se sube un mínimo, volver a verificarlo en un
  entorno temporal con esas versiones exactas.
- `requirements-dev.txt`: pytest (solo desarrollo).
- Compatibilidad a vigilar:
  - matplotlib ≥ 3.11 eliminó `boxplot(labels=)` → usar `tick_labels=` (≥ 3.9).
  - seaborn 0.13.2 (última) pasa `vert=` a matplotlib: aviso de obsolescencia en
    3.11 y **eliminación anunciada en matplotlib 3.13**. Desde F1.7 los
    diagramas de caja de las prácticas usan `ax.boxplot` de matplotlib, no
    `sns.boxplot`. Tampoco se pasa `center=` a `sns.heatmap` (emite un aviso
    `set_bad`) ni `palette=` sin `hue=`.
  - `scipy.stats.fisher_exact` con tablas RxC estima el p-valor por **Monte
    Carlo** (no es exacto): pasar `method=stats.MonteCarloMethod(n_resamples=…,
    rng=…)` con semilla e informarlo como estimación (`rng` existe desde 1.15).
  - `pd.read_csv`/`read_excel` leen el texto `"None"` como NaN por defecto: usar
    `keep_default_na=False` cuando sea una categoría (variable `SD`).
- Diagramas: bloques `{mermaid}` nativos de Quarto (no requieren dependencias).
- Widgets (F3.8): celdas `{ojs}` (Observable JS, integrado en Quarto; se
  ejecuta en el navegador, sin kernel ni dependencias de Python). Cargan
  `Inputs`/`Plot` desde la CDN de Quarto al abrir la página. Hay dos: M3
  (aproximación Normal a la Binomial) y M4 (potencia y tamaño muestral).
- No introducir dependencias nuevas sin justificarlo y sin añadirlas a
  `requirements.txt`.

## 4. Organización de módulos y contenidos

| Mód. | Teoría | Práctica | Toolkit | Datos / caso de uso real |
|---|---|---|---|---|
| 1 | Variables, tendencia central, dispersión, forma, outliers, missing, EDA vs. minería | `practica.ipynb` (datos simulados) + `practica_preprocesamiento.ipynb` (limpieza, imputación KNN, correlaciones) | `descriptive` | `sleephealth.xlsx` |
| 2 | Espacio muestral, Kolmogorov, condicionada, prob. total, Bayes, hacia Naive Bayes | Monte Carlo con dados, cartas, test médico, filtro de spam | `probability` | simulados |
| 3 | Bernoulli, Binomial, Poisson, Uniforme, Normal, Exponencial, TCL | Una sección por distribución + TCL | `distributions` | simulados |
| 4 | Estimación, IC, contrastes (z, t, dos proporciones), cobertura de IC | Actividad con 4 preguntas de investigación (PR1–PR4) sobre salud del sueño | `inference` (teoría y ejercicios de la práctica) | `sleephealth_processed.csv` (salida del M1) |
| 5 | Covarianza, Pearson, OLS, R², inferencia sobre la pendiente | Publicidad vs. ventas | `regression` | simulados |
| 6 | ANOVA 1 factor, χ² independencia/bondad de ajuste, Mann-Whitney U | Tratamientos, segmentos/producto, valoraciones de app | `group_comparisons` | simulados |
| P | Proyecto integrador (F3.5) | Calidad del sueño con las herramientas de M1–M6 | todos | `sleephealth_processed.csv` |

Estructura habitual de un `teoria.qmd`: introducción que enlaza con el módulo
anterior → definiciones con fórmulas LaTeX → ejemplos con código del toolkit →
mapa conceptual Mermaid → cierre/puente. Estructura habitual de `practica.ipynb`:
celda de imports + `sys.path` → secciones numeradas con simulación, cálculo y
gráfico → "Ejercicio(s) propuesto(s)" con celda vacía `# Escribe tu solución aquí`.

El orden de capítulos lo define **únicamente** `_quarto.yml` (`book.chapters`).
Un archivo nuevo no aparece en el libro hasta que se añade ahí.

## 5. Convenciones de programación (observadas)

- **Idioma**: identificadores de funciones/parámetros en **inglés**
  (`confidence_interval_mean_t`, `alternative`); docstrings, comentarios,
  mensajes de error y texto del curso en **español**.
- Docstrings estilo **NumPy** (`Parameters`, `Returns`) en `descriptive.py` y
  `probability.py`; en M3–M6 son más breves (una línea + `Devuelve (...)`).
  Para código nuevo, usar estilo NumPy.
- Type hints en firmas; `from __future__ import annotations` en M1–M2.
- Funciones puras que devuelven `float`, `tuple` o `dict` de Python (se convierte
  explícitamente con `float(...)`). Los contrastes devuelven `(estadístico, p_valor)`.
- Parámetro `alternative ∈ {"two-sided", "greater", "less"}` (convención scipy).
- Aleatoriedad: preferir `np.random.default_rng(seed)` y exponer `seed` como
  parámetro; los ejemplos usan `seed=42`. (`monte_carlo_estimate` y la práctica
  de M1 usan todavía `np.random.seed`, API heredada.)
- El toolkit mezcla implementaciones "manuales" (fórmulas explícitas: IC, z/t,
  OLS, ANOVA) con envoltorios de `scipy.stats` (χ², Mann-Whitney, cuantiles). El
  README afirma "funciones implementadas desde cero", lo cual solo es cierto en
  parte. Mantener el criterio: implementar a mano lo que tiene valor didáctico y
  contrastarlo con scipy/statsmodels.
- Bloque `if __name__ == "__main__":` con ejemplo rápido en M1–M2.
- Estilo de secciones con banners de comentarios `# ----`/`# ====`.
- No hay linter, formateador ni configuración de estilo en el repo.

## 6. Ejecución, validación y publicación

### Local (Windows, Git Bash o PowerShell)
```bash
python -m venv venv
source venv/Scripts/activate          # PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
quarto preview                        # servidor con recarga en vivo
quarto render                         # lo mismo que hace el CI
```

### Qué se ejecuta realmente al renderizar
- `.qmd`: los bloques `{python}` **se ejecutan** (con `freeze: auto`, solo si la
  fuente cambió respecto a `_freeze/`). En CI `_freeze/` está casi vacío, así que
  se ejecutan todos.
- `.ipynb`: Quarto **no los ejecuta** por defecto; muestra las salidas guardadas
  en el notebook. Desde F1.7 las siete prácticas tienen salidas guardadas,
  generadas con el `venv` sin errores ni avisos (stderr vacío).
- CI (F1.9, opción b): `ci-check.yml` ejecuta `pytest` y todas las prácticas
  con `jupyter nbconvert --execute` (copias fuera del repo, kernel `python3`)
  antes del render; una celda con error deja el check en rojo. Tras cambiar
  un notebook, guardar de nuevo sus salidas ejecutándolo con nbclient.

### Validación que debo hacer antes de dar algo por bueno
1. `quarto render` sin errores (como mínimo los capítulos tocados:
   `quarto render notebooks/0X_.../teoria.qmd`).
2. Si toco un `.ipynb`: editar solo las fuentes (con `nbformat`, nunca el JSON
   de salidas a mano) y ejecutarlo de principio a fin con el Python del `venv`,
   usando `nbclient.NotebookClient(..., record_timing=False)` con la carpeta del
   notebook como directorio de trabajo (así no se añaden marcas de tiempo al
   JSON). Revisar las salidas, no solo la ausencia de excepciones, y comparar
   con las anteriores. Ojo: los avisos (stderr) guardados incluyen la ruta
   local del kernel (`C:\Users\<usuario>\AppData\Local\Temp\ipykernel_…`).
   Para comprobar visualmente un capítulo renderizado, servir `_book/` en
   local (`python -m http.server`), porque abierto como archivo no carga
   MathJax ni el CSS.
3. Si toco `stats_toolkit`: contrastar cada resultado con una referencia
   independiente (scipy/statsmodels, fórmula cerrada o caso de libro) y probar
   casos límite (n pequeño, varianza cero, NaN, p en {0,1}). Ejecutar la suite:
   `pip install -r requirements-dev.txt` y `python -m pytest` desde la raíz
   (`tests/`, un archivo por módulo; `conftest.py` añade `src/` al path;
   `pytest.ini` fija `testpaths` y `xfail_strict`). Los defectos conocidos están
   marcados `xfail(strict=True)` con el ID de la tarea que los corrige: al
   corregirlos, retirar el marcador. Al comparar p-valores muy pequeños usar
   `pytest.approx(..., abs=0)` (la tolerancia absoluta por defecto, 1e-12,
   aceptaría p = 0).
4. Revisar el HTML generado en `_book/` cuando el cambio afecte a fórmulas,
   tablas, Mermaid o numeración.

### Flujo git y publicación (definido en README)
- Rama de trabajo permanente: `test`. `main` está protegida: solo vía PR con el
  check `render-check` en verde, fusión con *merge commit*.
- Merge a `main` → `deploy.yml` publica en `gh-pages`. **Nunca** tocar `gh-pages`
  a mano.
- Tras el merge: `git checkout main && git pull origin main && git checkout test
  && git merge origin/main && git push origin test`.
- Mensajes de commit observados: prefijos tipo Conventional Commits
  (`feat:`, `fix:`, `chore:`, `ci:`), normalmente en español
  (p. ej. `feat: Módulo 4 - Inferencia`).
- No hago commit, push ni PR sin que el usuario lo pida.

## 7. Criterios estadísticos

- **Corrección antes que ejecución**: que el código corra no valida el cálculo.
  Verificar fórmulas, grados de libertad, supuestos y la interpretación textual.
- Notación coherente con la teoría: parámetros poblacionales en griego
  ($\mu, \sigma, p, \beta_1$), estimadores con gorro o latinos
  ($\bar{x}, s, \hat{p}, \hat\beta_1$). Var./desv. típica **muestral con n−1** por
  defecto (`ddof=1`), y decirlo explícitamente cuando se use la poblacional.
- Explicitar siempre: hipótesis $H_0/H_1$, tipo de contraste (uni/bilateral),
  $\alpha$ (referencia 0,05), estadístico, distribución bajo $H_0$, p-valor,
  decisión e interpretación en contexto; distinguir significación estadística,
  tamaño del efecto/importancia práctica y causalidad.
- Interpretación frecuentista correcta del IC (cobertura a largo plazo), como ya
  hace `simulate_ci_coverage`; nunca "probabilidad de que μ esté en este intervalo".
- Comprobar supuestos (normalidad, independencia, homocedasticidad, frecuencias
  esperadas ≥ 5 en χ²) y mencionar alternativas (Welch, Fisher, no paramétricas).
- Detalles del toolkit actual a tener presentes (no son necesariamente errores,
  pero deben ser coherentes con lo que dice la teoría):
  - `shape()` (desde F1.2) calcula a mano, desde los momentos centrales, los
    estimadores corregidos **G₁ y G₂** (tipo 2), iguales a pandas y a SciPy con
    `bias=False`; NaN si n < 3 (asimetría), n < 4 (curtosis) o datos
    constantes, de forma independiente. La teoría del M1 presenta g₁/g₂ como
    definición y G₁/G₂ como lo que se calcula. Al hablar de ellos, no decir
    que la corrección "elimina el sesgo" ni que G₁/G₂ son "mejores": G₂ es
    insesgado solo bajo normalidad, y G₁/G₂ tienen más variabilidad que g₁/g₂.
    En LaTeX, escribir los decimales como `0{,}3` (coma decimal sin espacio).
  - Moda (desde F1.3): `modes(data)` devuelve todas las modas, ordenadas,
    ignorando ausentes y comparando por igualdad exacta de Python (sin
    agrupar, redondear ni convertir antes de contar: `1 == 1.0` cuentan
    juntos; `2**53` y `2**53+1`, o `Fraction(1,3)` y `1/3`, no). Conserva el
    tipo original (int, float, Fraction, str; NumPy → tipo Python
    equivalente); entre valores iguales de distinto tipo devuelve el más
    exacto (int > Fraction > float), sin depender del orden.
    **Convención del curso**: si todos los valores distintos tienen la misma
    frecuencia (incluido "todos distintos") no hay moda → `[]`; los datos
    constantes sí tienen moda. pandas y `statistics.multimode` siguen otra
    convención (devuelven todos). `central_tendency()['mode']` es float: la
    moda si es única, NaN si no (sin avisos ni elección arbitraria). Con datos
    continuos se enseña la clase modal (intervalos), indicando siempre la
    agrupación usada y comparando frecuencias solo entre intervalos de la
    misma anchura; una moda observada puede deberse a una coincidencia por
    redondeo y no ser descriptiva. No usar `scipy.stats.mode` (elige un
    valor arbitrario).
  - `five_number_summary()` usa cuartiles por interpolación lineal
    (`np.percentile`), que pueden diferir de los de Tukey (bisagras).
  - `confidence_interval_proportion()` es el IC de Wald recortado a [0, 1].
  - `chi2_test_independence(table, correction=False)` (desde F1.5): por
    defecto, estadístico de Pearson sin corrección, como la fórmula de la
    teoría; `correction=True` aplica Yates (solo afecta a tablas 2×2). Ojo:
    `scipy.stats.chi2_contingency` aplica Yates por defecto en 2×2.
  - p-valores (desde F1.4): `inference` y `regression` los calculan con
    `_pvalues.p_value_from_statistic(stat, dist, alternative)`, donde `dist`
    es una distribución congelada de SciPy: bilateral `2*sf(|s|)`, derecha
    `sf(s)`, izquierda `cdf(s)`. **Nunca `1 - cdf`** (se anula desde z ≈ 8,3).
    No se usa `logsf`. Un 0.0 solo debe aparecer por subdesbordamiento real:
    cuando el p-valor es menor de lo que se puede representar. Ese límite
    depende de la precisión numérica (`float64`) y de la implementación de
    cada función: no es una garantía universal. Como orientación, con SciPy
    1.18 y la Normal, `norm.sf` devuelve 0 por debajo de ~1e-310 (|z| ≳ 37,7),
    antes que el menor `float` subnormal (~5e-324). Los tests no deben fijar
    ese umbral exacto. Los contrastes devuelven
    estadístico y p-valor como `float` de Python. `normal_cdf` usa `erfc` y
    `exponential_cdf` usa `expm1` (precisión en colas y con x pequeño).
    Para mostrar p-valores en la teoría se usa `:.3g` (nunca `:.6f`, que
    imprime 0.000000) y se comunica `p < 0,001` cuando corresponde; las
    prácticas de M5–M6 aún usan `:.6f` (F1.7).
  - `confidence_interval_mean_t` recibe `sample_std` y asume que es la s con n−1.
  - `knn_impute(data, n_neighbors, ids)` (corrección del CI de F5.1): envuelve
    `KNNImputer` con una métrica que suma `1e-9·rango(ID)/n` a la distancia,
    de modo que entre vecinos equidistantes ganan siempre los de menor `ID`.
    Sin ella, los empates (registros idénticos salvo la edad) se resolvían
    según el orden interno de la selección, distinto entre máquinas. La usa el
    preprocesamiento; único cambio en el CSV: `Age` del `ID` 41, 30,33 → 30,0.
  - Fase 3: `power_z_test_mean` y `sample_size_z_test_mean` son del test z
    (σ conocida); para el test t se remite a `statsmodels` `TTestPower`. El
    tamaño muestral bilateral ignora la cola opuesta (potencia ≥ la pedida).
    `bootstrap_ci` es el método **percentil** (cobertura baja con n pequeño;
    scipy usa BCa por defecto); `statistic` debe aceptar `axis`.
    `spearman_test` usa la aproximación t con n−2 gl (como `spearmanr`);
    `sign_test` devuelve (k, p) con p exacto de la Binomial(m, 1/2),
    descartando las diferencias nulas.
- Criterios de redacción fijados en F1.1 (mantenerlos en contenido nuevo):
  - TCL: la media **tipificada** converge a N(0,1); $\bar X \approx N(\mu,\sigma^2/n)$
    es solo una aproximación para n grande. Condiciones suficientes (no únicas):
    i.i.d. y $0<\sigma^2<\infty$. El n necesario depende de la forma.
  - Las pruebas no paramétricas **tienen supuestos**. Mann-Whitney: H₀ igualdad
    de distribuciones, H₁ distribuciones distintas; es sensible sobre todo a
    P(X>Y) ≠ P(Y>X); solo es prueba de medianas con el supuesto de
    desplazamiento; p > 0,05 no demuestra igualdad.
  - Supuestos de modelos: en regresión, sobre los errores (Y condicionada a X);
    en ANOVA, normalidad dentro de cada grupo con varianza común. χ² es una
    aproximación (frecuencias esperadas ≥ 5 como regla habitual).
  - Imputación simple: conserva la media observada, no la poblacional; reduce
    la dispersión, puede distorsionar relaciones y subestimar errores estándar.
  - No remitir a recursos inexistentes (p. ej. un «repositorio de Machine
    Learning»); usar referencias genéricas a contenidos posteriores. El M5
    cierra como módulo con puente al M6, y el cierre del curso está en el M6.

## 8. Criterios pedagógicos

- Público: general, sin base estadística (ver §10). Introducir cada concepto con intuición,
  después definición formal, después ejemplo numérico/código y finalmente
  interpretación.
- Aprendizaje progresivo: no usar conceptos de módulos posteriores sin anunciarlo;
  reutilizar los de módulos previos y referenciarlos ("como vimos en el Módulo 2").
- Comparar cálculo manual ↔ `stats_toolkit` ↔ biblioteca estándar (patrón ya
  usado en M1 y M4) para que el alumno vea que coinciden y por qué difieren
  cuando difieren.
- Cada práctica termina con ejercicios propuestos y una sección «Soluciones»
  con un callout plegable por ejercicio (decisión del autor, §10; F2.5).
- Cada teoría empieza con «Objetivos del módulo» (callout) y termina con
  «Ideas clave» y «Autoevaluación» antes del cierre, que enlaza a su práctica
  (F2.8, F3.1). Autoevaluación: 5 preguntas, cada una con su respuesta en un
  callout plegable («Respuesta»).
- Cada práctica empieza (tras el título) con el callout «Ejecutar este
  cuaderno» (Colab, GitHub, zip) y una celda que solo actúa en Google Colab
  (clona el repositorio y hace `os.chdir` a la carpeta del cuaderno) (F3.7).
- Los p-valores en tablas (p. ej. Tukey) se formatean con `:.3g`; nunca
  `.round(4)` sobre la columna de p-valores (mostraría 0.0000).
- No numerar secciones a mano (Quarto ya numera); nombrar las secciones al
  referirse a ellas. Decimales con coma en el texto (`0{,}05` en LaTeX).
- Todas las figuras llevan `#| fig-alt:` (texto alternativo que describe qué
  se ve y el patrón principal, comprobado con la salida real).
- Gráficos (F2.7): título, ejes rotulados en español con unidades cuando se
  conozcan y leyenda cuando haya varias series o colores con significado.
  Estilo común de las prácticas en la celda de imports:
  `sns.set_theme(style="whitegrid")` y `plt.rcParams["figure.figsize"] = (8, 4.5)`;
  figuras de varios paneles con anchura ≤ 12. Leyendas que tapen datos, fuera
  del área de trazado. Reproducibilidad: semillas fijas.
- Citas con `[@clave]` contra `references.bib`; no inventar referencias ni
  añadir entradas sin fuente verificable.

### Identidad visual (F5.1)
- La guía de diseño vive al principio de `styles.css` («0) IDENTIDAD VISUAL»):
  variables `--eb-*` (colores, espaciado `--eb-space-1…8`, radios, sombras,
  tamaños `--eb-fs-hero`/`--eb-fs-h2`) y componentes `.eb-*` (`.eb-btn`
  primario/secundario/claro, `.eb-card`, `.eb-icon`, `.eb-kicker`, `.eb-grid`,
  `.eb-resource`). Cualquier pieza nueva de interfaz debe reutilizarlos; no
  escribir colores sueltos. Paleta: primario `#1f4fb8`, acento `#0d7a74`,
  tinta `#172033`, texto secundario `#525d71`; todo el texto ≥ 4,5:1 (AA).
- Modo oscuro futuro: redefinir las variables bajo el selector del tema oscuro.
- Iconos: Bootstrap Icons (`<i class="bi bi-…" aria-hidden="true">`), que
  Quarto ya sirve en local. Sin fuentes, frameworks ni recursos externos nuevos.
- Interacción: hover con sombra y elevación de 2 px; `:focus-visible` con
  contorno de 3 px; todo se desactiva con `prefers-reduced-motion`.
- Portada (`index.qmd`): `page-layout: full`, `toc: false`,
  `body-classes: eb-home`; el bloque de título del libro se oculta solo ahí
  (la cabecera tiene su propio `h1`). Secciones `##` con `{.unnumbered
  .unlisted}`. Enlaces internos siempre en Markdown a `.qmd`/`.ipynb` (Quarto
  los resuelve); el SVG de la cabecera es propio e inline (`aria-hidden`).
- Menú lateral (F5.2): etiquetas cortas con `href` + `text` en `book.chapters`
  («Teoría», «Práctica», «Preprocesamiento»; los títulos de las páginas no
  cambian); partes «N Título abreviado» con espacio de no separación tras el
  número (la primera letra se pinta como insignia con `::first-letter`);
  `book.sidebar.collapse-level: 1` (solo se despliega el grupo de la página
  actual) y `logo`. Los grupos se reconocen en CSS con `:has()` por sus
  enlaces (`notebooks/0X`, `07_proyecto_integrador`, `apendices/`): si cambian
  las rutas, revisar `styles.css` §1. `assets/navegacion.html` solo hace dos
  cosas: vuelve enfocables las cabeceras de grupo (Quarto las genera como
  `<a>` sin `href`; pasan a `role="button"`, Intro/Espacio) y antepone el grupo
  a las etiquetas del pie «anterior/siguiente» («4 Inferencia · Práctica»).
- Validar las páginas de interfaz a 1440, 768 y 375 px sin desplazamiento
  horizontal. Chrome headless no baja de ~500 px de ventana: para capturas
  móviles, usar un `iframe` de 375 px o el navegador con perfil móvil.
- No duplicar contenido entre teoría y práctica ni entre módulos: enlazar.

## 9. Incoherencias detectadas (estado de cada una)

1. ~~Doble numeración de secciones~~ — resuelto en F2.1 (no numerar a mano:
   Quarto numera; para referirse a una sección, nombrarla).
2. ~~Front matter heterogéneo~~ — resuelto en F2.1.
3. ~~`index.qmd` solo describe M1–M2~~ — resuelto en F2.1.
4. ~~README desactualizado~~ — resuelto en F2.1.
5. ~~Restos de adaptación en la práctica de M4~~ — resuelto en F1.10.
6. ~~Enlace M1→M4 no reproducible~~ — resuelto en F1.10: el preprocesamiento
   guarda `01_estadistica_descriptiva/data/sleephealth_processed.csv` y el M4
   lo lee (`04_inferencia/data/sleephealth_processed.xlsx` se conserva en el
   repositorio por decisión del autor, pero la práctica ya no lo usa).
7. ~~Dependencias no declaradas~~ — resuelto en F1.8.
8. ~~Prácticas M3–M6 sin salidas guardadas~~ — resuelto en F1.7.
9. ~~`_freeze/` ignorado pero con 3 archivos versionados~~ — resuelto en F2.1
   (`git rm --cached`, decisión del autor; la retirada queda en el índice
   hasta el próximo commit).
10. Tests y versiones mínimas resueltos (F1.0, F1.8); sigue sin licencia (F4.4).
11. `monte_carlo_estimate` usa el estado global `np.random.seed` mientras el
    resto usa `default_rng`: se mantiene por la API (`trial_fn()` sin
    argumentos); documentado y testeado en F2.3.

## 10. Decisiones del autor e información pendiente

### Decidido (2026-11-23)
- **Público**: general, sin conocimientos estadísticos previos; se presuponen
  matemáticas/álgebra básicas y cierta familiaridad con Python. Explicaciones
  accesibles **sin sacrificar rigor matemático**.
- **Soluciones de ejercicios**: desarrolladas y explicadas, en bloques plegables
  de Quarto (`::: {.callout-tip collapse="true"}`), colocadas después del
  enunciado y del espacio para intentarlo.
- **Material de terceros**: no publicar ni ampliar material ajeno sin verificar
  antes sus condiciones de uso. **Verificado por el autor**: la procedencia de
  `sleephealth.xlsx` y los materiales/PDF de la UOC adaptados en M1 y M4; sus
  condiciones de uso están resueltas y pueden usarse dentro del alcance del
  curso. No es necesario volver a investigarlo. Cualquier material externo
  *nuevo* sí debe verificarse antes.
- **Sleephealth**: caso práctico recurrente del curso cuando tenga sentido
  pedagógico. Sus datos son sintéticos/ilustrativos: no presentar conclusiones
  de salud como hallazgos reales.
- **Tests**: se incorpora pytest. Los tests comprueban la corrección estadística
  contra **referencias independientes** (scipy/statsmodels, fórmulas cerradas,
  valores de tablas o de libro, propiedades matemáticas, simulación); nunca se
  limitan a congelar la salida actual del toolkit.
- **Notebooks**: el CI debe validar que se ejecutan (estrategia concreta en F1.9).
- **Datos**: además de `sleephealth`, se mantienen simulaciones y otros datos
  cuando expliquen mejor un concepto.
- **Módulo 6**: módulo completo, al mismo nivel que los otros cinco (no una
  "ampliación").
- **Asimetría/curtosis**: aprobado el **tipo 2** de Joanes & Gill (1998),
  $G_1 = g_1\sqrt{n(n-1)}/(n-2)$ y
  $G_2 = \frac{n-1}{(n-2)(n-3)}[(n+1)g_2+6]$, coherente con pandas
  `.skew()/.kurt()`, Excel y SPSS. Se implementa en la tarea F1.2, no antes.

### Pendiente
- Licencia del curso y del código (F4.4).
- ¿Ejecutar `.ipynb` en el render o mantener salidas guardadas + ejecución en CI?
- Alcance de "desde cero" en `stats_toolkit`.

## 11. Hoja de ruta (resumen; tipo: E error · L limitación · D decisión · N nueva)

Se trabaja **una tarea cada vez**, con revisión del autor entre tareas. Los
`xfail` de `tests/` citan estos identificadores.

**Fase 1 — problemas importantes**
- F1.0 Tests de `stats_toolkit` con referencias independientes (L) — *hecho y
  aprobado* (124 pasan, 8 xfail).
- F1.1 Errores conceptuales en teoría M2, M3, M5, M6 y matices de M1 (E) —
  *hecho y aprobado*. Auditoría de las seis
  teorías: 11 hallazgos A y 16 B corregidos (M1, M2, M3, M5, M6; M4 sin
  cambios); 12 hallazgos C repartidos en las tareas indicadas abajo con
  «(de F1.1)». M3 validado con render en una copia temporal (sin tocar
  `_freeze/`).
- F1.2 Asimetría/curtosis al tipo 2 en código, teoría, práctica M1 y tests (E)
  — *hecho y aprobado*.
- F1.3 Moda: `modes()`, `central_tendency()`, teoría y práctica M1 (clase
  modal), tests (E) — *hecho y aprobado*, incluidas las correcciones de la
  revisión (recuento con valores originales, coincidencias por redondeo,
  anchura de intervalos) (182 pasan, 6 xfail al cerrarla).
- F1.4 p-valores con `sf` en vez de `1 - cdf` (E) — *hecho y aprobado*
  (348 pasan, 1 xfail de F2.3). Decisiones: módulo
  privado `_pvalues.py`, salidas `float`, `normal_cdf`/`exponential_cdf`
  estables, nota en la teoría del M4 y formato `:.3g` en teorías M5–M6.
- F1.5 Corrección de Yates explícita y coherente (E, D) — *hecho y aprobado*.
  `chi2_test_independence(..., correction=False)` por defecto
  (decisión: coherencia con la fórmula de la teoría y con la práctica del M4);
  tests 2×2 con y sin Yates; nota en las teorías de M4 y M6.
- F1.6 Rigor del preprocesamiento (E, D) — *hecho y aprobado*.
  Se elimina el duplicado exacto (ID 35, filas 33 y 35; quedan 373 filas) y se
  conserva el registro distinto con nuevo `ID`; KNN sobre variables
  estandarizadas; narración de atípicos corregida (73 se conserva como edad
  plausible); correlaciones actualizadas y aviso de datos sintéticos; salida en
  `data/sleephealth_processed.csv`. Pendiente (F2.7): gráfico de líneas para
  una variable nominal.
- F1.7 Notebooks ejecutables de principio a fin (E) — *hecho y aprobado*. Las 7 prácticas se ejecutan con nbclient sin errores ni stderr:
  `tick_labels=` en M6, diagramas de caja con matplotlib (M1, preprocesamiento,
  M4), sin `center=` en el heatmap. Además, en la práctica del M6 se corrigió
  el texto que atribuía a Mann-Whitney la detección de "la forma completa"
  (contradecía la teoría de F1.1 y sus propios resultados: p = 1) y el mensaje
  del ANOVA; formato `:.3g` en M5 y M6.
- F1.8 Dependencias completas en `requirements.txt` (E) — *hecho y aprobado*.
- F1.9 CI que ejecute pytest y los notebooks (L, D) — *implementado y
  validado en local; pendiente de comprobar en un PR real* (el comando del CI
  termina con código 1 ante un notebook roto y con 0 con los siete actuales).
- F1.10 Práctica M4 como actividad propia del curso (E) — *hecho y aprobado*. Sin restos de la conversación ni del R (la tabla R↔Python pasa a
  "Funciones utilizadas y precauciones"); fuente única con error explícito
  (CSV del M1, leído con `keep_default_na=False`); Fisher 3×2 por Monte Carlo
  con semilla e informado como cota/estimación; p-valores sin redondear a 0.

**Fase 2 — calidad y consistencia** — *completa* (F2.1–F2.9 hechas; tareas
aprobadas por el autor salvo el cierre de F2.1 y F2.7, pendiente de revisión)
- F2.1 Coherencia formal — *hecho*. Sin numeración manual en
  teorías M3–M6 ni en las prácticas (las referencias "sección N" pasan a
  nombrar la sección); front matter de M1–M2 sin `format`/`bibliography`/
  subtítulo «Repositorio 1» (todo está en `_quarto.yml`); notación RIC en M1;
  decimales con coma en el texto y las fórmulas de las teorías; `index.qmd`
  describe los 6 módulos y los datos; README corregido (sin `images/`, casos de
  uso reales, alcance de "desde cero", tests y CI). `_freeze/` deja de
  versionarse (§9.9). Queda fuera de alcance: la cita FUOC 2020 del M4 sigue
  como texto, sin entrada en `references.bib` (no inventar la referencia). Los decimales de
  las salidas de código siguen con punto (formato de Python).
- F2.2 Toolkit completo — *hecho*. `inference`: `confidence_interval_proportion_wilson`,
  `one_proportion_z_test`, `welch_t_test` (devuelve t, p, gl), `paired_t_test`,
  `chi2_test_variance`. `group_comparisons`: `eta_squared`, `cramers_v`,
  `tukey_hsd` (DataFrame), `kruskal_wallis`, `wilcoxon_signed_rank`. Todas con
  tests frente a scipy/statsmodels o fórmulas cerradas, y usadas en la teoría.
- F2.3 Robustez — *hecho*. `_validation.py` (privado): en M4–M6, NaN, n
  insuficiente, varianza nula o alternativas/confianzas inválidas lanzan
  `ValueError` con mensaje en español (antes daban NaN o avisos). `descriptive`
  sigue ignorando NaN, pero sin datos válidos lanza `ValueError`.
  `poisson_pmf` y `binomial_pmf` en escala logarítmica (sin desbordamiento).
  `__init__` documenta y expone los seis módulos. `monte_carlo_estimate`
  mantiene el estado global de `np.random` porque `trial_fn()` no recibe
  generador (API pública); su semilla es reproducible y está testeada.
  Teoría M1: método de cuartiles explicado.
- F2.4 Módulo 6 completo — *hecho*. Teoría M6: η², Tukey, supuestos y
  alternativas, 1 − 0,95³, frecuencias esperadas, V de Cramér, Fisher,
  Kruskal-Wallis, prueba de signos y Wilcoxon (trasladados desde el M4, que
  ahora solo los anuncia y enlaza), diagrama de decisión ampliado. Teoría M5:
  supuestos, diagnóstico de residuos (gráficos), intercepto y extrapolación.
  Práctica M6 con `sleephealth` (estrés por trastorno: ANOVA, η², Tukey,
  Kruskal-Wallis; género × trastorno: χ² y V).
- F2.5 Soluciones plegables — *hecho* en las 5 prácticas con ejercicios (M1,
  M2, M3, M5, M6): sección «Soluciones» al final, un callout plegable por
  ejercicio que envuelve celdas de código ejecutadas (el div se abre en una
  celda markdown y se cierra en otra). M4 y preprocesamiento recibieron
  ejercicios y soluciones en F3.1.
- F2.6 Huecos del M3 — *hecho*: función de distribución, cuantiles,
  tipificación, distribución muestral de la media, t/χ²/F con gráfico.
- F2.7 Visualizaciones — *hecho*. Normal superpuesta en el TCL,
  densidades t, región crítica, Q-Q, 100 IC de la simulación de cobertura,
  residuos en M5, mapa de calor en lugar del gráfico de líneas nominal del
  preprocesamiento, `fig-alt` en todas las figuras de las teorías y en las
  nuevas de las prácticas; CSS para que las figuras de los `.qmd` no
  desborden la columna. Cierre: `fig-alt` en las 25 figuras de las prácticas;
  estilo común (ver §8); rótulos que faltaban (frecuencia, tratamiento,
  mensaje, títulos generales en las cuadrículas), leyenda del filtro de spam
  y anchura máxima 12. Salidas de texto idénticas a las anteriores.
- F2.8 Navegación — *hecho*: objetivos al inicio e «Ideas clave» antes del
  cierre en las 6 teorías, enlaces teoría → práctica, puente M2 → M3, matiz
  media/mediana en M1.
- F2.9 sleephealth como hilo conductor — *hecho*: secciones nuevas en las
  prácticas de M2 (probabilidad condicionada), M5 (estrés y calidad del sueño)
  y M6, además de M1 y M4. Los ejemplos de juguete de las teorías de M5 y M6
  se mantienen como ejemplos mínimos; los datos reales están en las prácticas.

**Fase 3 — ampliaciones (N)** (ejecutada en una ronda; pendiente de revisión
del autor). El roadmap no fijaba identificadores ni criterios; se asignaron
F3.1–F3.8 en el orden de la lista original.
- F3.1 Autoevaluación — *hecho*: sección «Autoevaluación» (5 preguntas con
  respuesta plegable) en las 6 teorías; ejercicios y soluciones nuevos en la
  práctica del M4 (bootstrap, potencia, tamaño muestral) y en la de
  preprocesamiento (efecto de la imputación KNN, regla de Tukey en `HR`).
- F3.2 Potencia y tamaño muestral — *hecho*: `inference.power_z_test_mean`,
  `sample_size_z_test_mean`, `sample_size_mean_ci`, `sample_size_proportion_ci`
  (tests frente a statsmodels `NormalIndPower`, fórmulas y simulación); teoría
  M4 con fórmulas, curvas de potencia, `TTestPower` y la advertencia sobre la
  «potencia observada».
- F3.3 Bootstrap — *hecho*: `inference.bootstrap_distribution` y
  `bootstrap_ci` (percentil; tests frente a `scipy.stats.bootstrap` y el IC t);
  sección «Intervalos bootstrap» en la teoría M4 (mediana, figura, BCa,
  limitaciones).
- F3.4 Spearman y no paramétricas adicionales — *hecho*:
  `regression.spearman_correlation`, `spearman_test` (frente a `spearmanr`);
  `group_comparisons.sign_test` (frente a la Binomial calculada a mano).
  Teoría M5 (Spearman, atípico, mención de Kendall), teoría M6 (prueba de
  signos con código), práctica M5 (Spearman en sleephealth + ejercicio 4).
- F3.5 Proyecto integrador — *hecho*: `notebooks/07_proyecto_integrador/
  practica.ipynb` (parte propia en `_quarto.yml`): describir SQ, IC t y
  bootstrap, Welch/Mann-Whitney por actividad física, ANOVA/η²/Tukey/
  Kruskal-Wallis por trastorno, Pearson/Spearman/regresión con SH, tamaño
  muestral y síntesis con límites (observacional, sintético, comparaciones
  múltiples). Tareas con solución plegable; interpretaciones escritas a partir
  de las salidas reales.
- F3.6 Apéndices — *hecho*: «¿Qué prueba usar?» (diagrama y tabla con las
  funciones del toolkit), «Notación» y «Glosario».
- F3.7 Descarga/Colab — *hecho*: callout y celda de Colab en las 8 prácticas.
  **Sin verificar en Colab real**: los enlaces apuntan a `main` y solo
  funcionarán tras fusionar; M4 y el proyecto necesitan que
  `01_estadistica_descriptiva/data/sleephealth_processed.csv` esté versionado.
- F3.8 Widgets — *hecho* (2 widgets OJS en M3 y M4; Φ por Abramowitz-Stegun
  7.1.26 en el M4, comprobado frente a Python). Requieren conexión para cargar
  las bibliotecas de Observable.
- De paso: la tabla de Tukey de la teoría M6 mostraba un p-valor `0.0000`;
  ahora se formatea con `:.3g`. CSS: `main table code` con
  `overflow-wrap: anywhere` para que las tablas no desborden.

**Fase 4 — profesionalización**: empaquetado de `stats_toolkit` (D), CI
completo (lint, enlaces), reproducibilidad de versiones, licencias,
accesibilidad, documentación para colaboradores, PDF/EPUB (N).
El autor acotó la fase 4 a «dejar el curso listo para publicar en GitHub Pages
y usar en Colab» (2026-11-23). Estado:
- F4.1 Reproducibilidad y datos — *hecho*: `sleephealth_processed.csv` pasa a
  versionarse (pendiente de incluir en el commit); rutas relativas al
  cuaderno en local, CI y Colab; ninguna ruta absoluta ni del equipo local.
- F4.2 Colab — *hecho y simulado; sin probar en Colab real*. La celda de
  Colab clona el repo, instala las bibliotecas de cálculo de
  `requirements.txt` que falten (excluye jupyter/ipykernel/nbclient/nbformat),
  avisa si hay que reiniciar la sesión y hace `chdir` a la carpeta del
  cuaderno. Simulación: venv nuevo con Python 3.12, clon desde cero por
  cuaderno, módulo `google.colab` simulado.
- F4.3 CI y publicación — *revisado*: el CI ejecuta pytest, las 8 prácticas
  (glob `notebooks/*/practica*.ipynb`) y el render; paso nuevo que compara el
  CSV regenerado con el versionado (tolerancia 1e-9). `deploy.yml` sin
  cambios. Sin probar en GitHub (requiere push/PR del autor).
- F4.4 Widgets y experiencia de usuario — *hecho*: aviso
  `#aviso-widget-*` que una celda OJS retira solo si cargan `Plot`/`Inputs`;
  sin conexión, CSS oculta los errores técnicos de OJS. CSS para pantallas
  estrechas: fórmulas en bloque, código y tablas se desplazan dentro de su
  caja; código en títulos y tablas puede partirse. Verificado: las 18 páginas
  caben en 375 px sin desplazamiento horizontal.
- Validación F4 (2026-11-23): copia con solo los archivos versionables
  (`git ls-files -co --exclude-standard`), venv nuevo con Python 3.12 y las
  últimas versiones (pandas 3.1, scipy 1.18, numpy 2.5): pytest 448 pasan;
  simulación de Colab de las 8 prácticas (clon desde cero por cuaderno) sin
  errores ni stderr y con salidas de texto idénticas; camino «falta una
  biblioteca» probado (instala y pide reiniciar); `quarto render` completo
  aislado (QUARTO_PYTHON del venv, sin kernels de usuario): código 0, 18
  páginas, 37 figuras con texto alternativo, 0 enlaces o anclas rotos,
  widgets operativos. En Windows, `pip install jupyter` en una ruta muy larga
  falla por el límite de 260 caracteres (no afecta a Linux/CI/Colab).
- Sin hacer (decisión del autor o fuera del alcance acotado): empaquetado de
  `stats_toolkit`, lint y comprobación de enlaces en el CI, fijar versiones
  exactas (lockfile), licencia (sigue sin `LICENSE`), CONTRIBUTING, PDF/EPUB,
  MathJax sin CDN (sin conexión las fórmulas se ven en LaTeX).

**Fase 5 — frontend y experiencia de usuario** (fases independientes, cada
una con commit, PR y despliegue tras la revisión del autor)
- F5.1 Portada e identidad visual — *hecho, pendiente de revisión*: cabecera
  con ilustración SVG propia y dos acciones, catálogo de 6 tarjetas con
  enlaces reales a teoría y práctica, «Cómo funciona el curso», banda del
  proyecto integrador y «Requisitos y recursos». Validada a 1440/768/375 px,
  20 enlaces sin errores, contraste AA, foco por teclado y movimiento reducido.
  Sin cambios en el menú lateral, las teorías, los notebooks ni el toolkit.
  Publicada (merge 820e4e9). Corrección en el mismo PR (#11): el CI falló porque `KNNImputer` resolvía
  de forma distinta un empate de distancia 0; se añade `desc.knn_impute` con
  desempate por `ID` (ver §7), tests de reproducibilidad (incluida la
  regeneración del CSV desde el cuaderno) y se regenera el CSV.
- F5.2 Menú lateral — *hecho, pendiente de revisión*: cabecera con logotipo,
  «Inicio», rótulo «Módulos» con insignias numeradas, proyecto y apéndices
  separados y con icono, etiquetas cortas, solo el grupo actual desplegado,
  página actual resaltada (fondo + barra lateral), foco visible y grupos
  operables con teclado. Validado a 1440/1024/768/375 px (sin desbordamiento,
  objetivos táctiles de 47 px, panel móvil que se abre, se cierra y navega),
  búsqueda nativa y 951 referencias internas sin errores.

## 12. Normas para futuras sesiones

1. **Trabajar sobre lo existente**; no reestructurar ni reescribir módulos sin
   propuesta previa aprobada.
2. Antes de un cambio importante, exponer: problema, solución justificada,
   archivos afectados y efectos secundarios (render, numeración, enlaces,
   coherencia entre módulos, CI).
3. No modificar comportamiento de funciones existentes de `stats_toolkit` sin
   justificarlo: la teoría y los notebooks dependen de sus firmas y resultados.
   Buscar todos sus usos (`grep` en `notebooks/`) antes de cambiar una firma.
4. Verificar matemáticamente cada cálculo nuevo o modificado contra una
   referencia independiente; documentar supuestos y limitaciones.
5. Sin dependencias nuevas salvo justificación; si se añaden, a `requirements.txt`.
6. Evitar duplicación: una sola implementación por concepto en `stats_toolkit`;
   el contenido teórico se enlaza, no se copia.
7. Mantener idioma y convenciones de §5 y la estructura de capítulos de §4;
   registrar todo capítulo nuevo en `_quarto.yml`.
8. Validar según §6 e informar honestamente de lo que no se pudo comprobar.
9. Ser crítico: señalar incoherencias o mejores alternativas aunque no se pidan,
   pero no corregirlas sin acuerdo (lista viva en §9).
10. Si un dato no está en el repo ni lo ha dado el autor, preguntarlo (§10).
11. Mantener este `CLAUDE.md` actualizado cuando cambie la arquitectura, el flujo
    o se resuelva algún punto de §9/§10.
12. **Una tarea cada vez**: al terminar una tarea de la hoja de ruta, presentar
    resultados y esperar la revisión del autor; no encadenar la siguiente.
13. **Nunca** hacer commit ni push sin autorización explícita del autor.
14. **Ejecutar directamente**: realizar yo mismo los cambios en el repositorio y
    las operaciones en el entorno local (instalar en `venv/`, ejecutar pruebas,
    comprobaciones…) siempre que sea posible, en lugar de dar instrucciones al
    autor para que las ejecute. Siguen vigentes las demás restricciones (commits,
    `quarto render`, alcance de la tarea).
