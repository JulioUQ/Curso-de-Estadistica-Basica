# Estadística Básica

Curso interactivo de estadística básica orientado a la práctica, construido con [Quarto](https://quarto.org/) y publicado como libro web. Cada módulo combina teoría formal con un cuaderno de ejercicios en Python, y todo el contenido se apoya en una librería propia (`stats_toolkit`): implementa a mano los métodos con valor didáctico (intervalos, contrastes z y t, regresión, ANOVA…) y envuelve `scipy.stats` en los demás, comparando siempre sus resultados con `scipy` y `statsmodels`.

🔗 **Sitio en vivo:** https://juliouq.github.io/Curso-de-Estadistica-Basica/

![Estado del despliegue](https://github.com/JulioUQ/Curso-de-Estadistica-Basica/actions/workflows/deploy.yml/badge.svg)

## Contenido del curso

| Módulo | Título | Temas | Caso de uso |
|---|---|---|---|
| 1 | Fundamentos y Estadística Descriptiva | Tipos de variables, medidas de tendencia central, dispersión y forma | Salarios simulados por departamento; preprocesamiento del dataset `sleephealth` |
| 2 | Probabilidad Básica | Espacio muestral, probabilidad condicional, Teorema de Bayes, Naive Bayes intuitivo | Filtro de spam simplificado |
| 3 | Distribuciones de Probabilidad | Bernoulli, Binomial, Poisson, Uniforme, Normal, Exponencial, Teorema Central del Límite | Modelado de tiempos de espera / conversión |
| 4 | Inferencia Estadística | Estimación puntual, intervalos de confianza, contrastes de hipótesis (z, t, Welch, emparejados, proporciones), potencia y tamaño muestral, bootstrap | Cuatro preguntas de investigación sobre `sleephealth` |
| 5 | Correlación y Regresión Lineal | Covarianza, correlación de Pearson y de Spearman, regresión lineal simple (OLS), R², inferencia sobre la pendiente | Publicidad y ventas; sueño y estrés en `sleephealth` |
| 6 | ANOVA, Chi-cuadrado y Pruebas No Paramétricas | ANOVA de un factor, tamaño del efecto, Tukey, Chi-cuadrado, Mann-Whitney U, Kruskal-Wallis, Wilcoxon, prueba de signos | Tratamientos, segmentos de clientes y `sleephealth` |

> **Nota sobre la numeración:** los Módulos 4, 5 y 6 de este repositorio agrupan de forma distinta algunos temas respecto a otros temarios de referencia (por ejemplo, Muestreo/TLC, Estimación e IC, y Contraste de Hipótesis aparecen aquí unificados dentro del Módulo 4 — *Inferencia Estadística*). Es una decisión deliberada para mantener el curso compacto; el contenido cubierto es equivalente.

Cada módulo incluye:
- **`teoria.qmd`** — desarrollo teórico con fórmulas, ejemplos, preguntas de autoevaluación y, en varios casos, un diagrama Mermaid o un gráfico interactivo (Observable JS, integrado en Quarto).
- **`practica.ipynb`** — cuaderno Jupyter con simulaciones, visualizaciones, ejercicios propuestos y sus soluciones plegadas. Cada práctica enlaza a Google Colab y a su descarga.
- Un módulo Python correspondiente en `src/stats_toolkit/` con las funciones usadas en la teoría y, en la mayoría de los módulos, también en la práctica (la práctica del Módulo 4 trabaja sobre todo con `scipy` y `statsmodels`, y sus ejercicios usan `stats_toolkit`).

Además, el curso incluye un **proyecto integrador** (`notebooks/07_proyecto_integrador/`) que aplica todos los módulos a los datos de sueño, y tres **apéndices** (`apendices/`): qué prueba usar, notación y glosario.

## Estructura del repositorio

```
Estadistica-Basica/
├── _quarto.yml                # Configuración del libro (capítulos, tema, formato)
├── index.qmd                  # Portada del curso
├── styles.css                 # Estilos personalizados (sidebar, tablas, citas...)
├── references.bib             # Bibliografía citable desde los .qmd
├── requirements.txt           # Dependencias Python del proyecto
├── requirements-dev.txt       # Dependencias de desarrollo (pytest)
├── tests/                     # Tests de stats_toolkit (pytest)
├── notebooks/
│   ├── 01_estadistica_descriptiva/
│   ├── 02_probabilidad/
│   ├── 03_distribuciones/
│   ├── 04_inferencia/
│   ├── 05_regresion/
│   ├── 06_anova_chi2/
│   │   ├── teoria.qmd
│   │   └── practica.ipynb
│   └── 07_proyecto_integrador/  # Proyecto final (practica.ipynb)
├── apendices/                 # Qué prueba usar, notación y glosario
├── src/
│   └── stats_toolkit/         # Librería Python del curso
│       ├── __init__.py
│       ├── descriptive.py
│       ├── probability.py
│       ├── distributions.py
│       ├── inference.py
│       ├── regression.py
│       └── group_comparisons.py
└── .github/workflows/
    ├── deploy.yml              # Publica el libro en GitHub Pages (push a main)
    └── ci-check.yml            # En cada Pull Request: pytest, ejecución de las prácticas y render (sin publicar)
```

## Cómo ejecutarlo en local

**Requisitos previos:** [Quarto CLI](https://quarto.org/docs/get-started/) instalado y Python 3.12+.

```bash
# 1. Clonar el repositorio
git clone https://github.com/JulioUQ/Curso-de-Estadistica-Basica.git
cd Curso-de-Estadistica-Basica

# 2. Crear y activar un entorno virtual
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Levantar el libro con recarga en vivo
quarto preview
```

`quarto preview` abre el libro en el navegador y se actualiza automáticamente al guardar cambios. Para generar el sitio estático sin servirlo (equivalente a lo que hace el despliegue), usa `quarto render`.

## Uso en Google Colab

Cada práctica empieza con un enlace **Abrir en Google Colab**. Su primera celda de código solo actúa en Colab: clona este repositorio en `/content`, instala las bibliotecas de cálculo de `requirements.txt` que falten (si instala o actualiza alguna, pide reiniciar la sesión) y se sitúa en la carpeta del cuaderno, de modo que `stats_toolkit` y los datos se cargan con las mismas rutas relativas que en local. Los enlaces apuntan a la rama `main`.

Los datos procesados que usan varias prácticas (`notebooks/01_estadistica_descriptiva/data/sleephealth_processed.csv`) están versionados: ninguna práctica necesita ejecutar antes otra. La práctica de preprocesamiento muestra cómo se generan, y el CI comprueba que el archivo regenerado coincide con el versionado.

## Uso sin conexión

El libro publicado carga desde CDN MathJax (fórmulas) y las bibliotecas de los dos gráficos interactivos (Observable). Sin conexión, el texto, el código, las figuras y los diagramas se siguen viendo; las fórmulas aparecen en LaTeX sin formatear y, en lugar de los gráficos interactivos, se muestra un aviso explicativo.

## Flujo de trabajo del proyecto

El repositorio sigue un flujo de **rama de desarrollo → Pull Request → producción**, pensado para validar cada cambio antes de que llegue al sitio publicado. Esta es la rutina para cada módulo nuevo:

1º Trabajar en `test`, commit y `git push origin test`.
2º Abrir el PR a `main` en GitHub, esperar a que `render-check` salga en verde y hacer el merge (merge commit).
3º Sincronizar en local: `git checkout main`, `git pull origin main`, `git checkout test`, `git merge origin/main`, `git push origin test`.

```
test (rama de trabajo)
  │
  ├── commit + push
  │
  └── Pull Request hacia main
        │
        ├── ci-check.yml renderiza el sitio (sin publicar) para detectar errores
        │
        └── Merge → dispara deploy.yml → publica en GitHub Pages
```

- `main` está protegida: todo cambio debe pasar por una Pull Request, y el check `render-check` debe estar en verde antes de poder fusionar.
- `test` es la rama de trabajo permanente; tras cada merge se sincroniza con `git pull origin main`.

## Tecnologías

- **[Quarto](https://quarto.org/)** — motor de publicación del libro (Markdown + código ejecutable + Mermaid).
- **Python** — `numpy`, `pandas`, `matplotlib`, `scipy` para el cálculo y la visualización.
- **GitHub Actions** — integración continua (validación de PRs) y despliegue automático.
- **GitHub Pages** — hosting del sitio estático generado.

## Autor

Julio Úbeda Quesada
