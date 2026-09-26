# Estadística Básica

Curso interactivo de estadística básica orientado a la práctica, construido con [Quarto](https://quarto.org/) y publicado como libro web. Cada módulo combina teoría formal con un cuaderno de ejercicios en Python, y todo el contenido se apoya en una librería propia (`stats_toolkit`) con las funciones estadísticas implementadas desde cero.

🔗 **Sitio en vivo:** https://juliouq.github.io/Curso-de-Estadistica-Basica/

![Estado del despliegue](https://github.com/JulioUQ/Curso-de-Estadistica-Basica/actions/workflows/deploy.yml/badge.svg)

## Contenido del curso

| Módulo | Título | Temas | Caso de uso |
|---|---|---|---|
| 1 | Fundamentos y Estadística Descriptiva | Tipos de variables, medidas de tendencia central, dispersión y forma | Análisis exploratorio de un dataset de salarios/ventas |
| 2 | Probabilidad Básica | Espacio muestral, probabilidad condicional, Teorema de Bayes, Naive Bayes intuitivo | Filtro de spam simplificado |
| 3 | Distribuciones de Probabilidad | Bernoulli, Binomial, Poisson, Uniforme, Normal, Exponencial, Teorema Central del Límite | Modelado de tiempos de espera / conversión |
| 4 | Inferencia Estadística | Estimación puntual, intervalos de confianza, contraste de hipótesis (z-test, t-test, dos proporciones) | A/B testing de tasas de conversión |
| 5 | Correlación y Regresión Lineal | Covarianza, correlación de Pearson, regresión lineal simple (OLS), R², inferencia sobre la pendiente | Relación entre inversión en publicidad y ventas |
| 6 | ANOVA, Chi-cuadrado y Pruebas No Paramétricas | ANOVA de un factor, test Chi-cuadrado de independencia, Mann-Whitney U | Comparación de tratamientos médicos / segmentos de clientes |

> **Nota sobre la numeración:** los Módulos 4, 5 y 6 de este repositorio agrupan de forma distinta algunos temas respecto a otros temarios de referencia (por ejemplo, Muestreo/TLC, Estimación e IC, y Contraste de Hipótesis aparecen aquí unificados dentro del Módulo 4 — *Inferencia Estadística*). Es una decisión deliberada para mantener el curso compacto; el contenido cubierto es equivalente.

Cada módulo incluye:
- **`teoria.qmd`** — desarrollo teórico con fórmulas, ejemplos y, en varios casos, un diagrama Mermaid a modo de mapa conceptual.
- **`practica.ipynb`** — cuaderno Jupyter con simulaciones, visualizaciones y ejercicios propuestos.
- Un módulo Python correspondiente en `src/stats_toolkit/` con las funciones usadas en la teoría y la práctica.

## Estructura del repositorio

```
Estadistica-Basica/
├── _quarto.yml                # Configuración del libro (capítulos, tema, formato)
├── index.qmd                  # Portada del curso
├── styles.css                 # Estilos personalizados (sidebar, tablas, citas...)
├── references.bib             # Bibliografía citable desde los .qmd
├── requirements.txt           # Dependencias Python del proyecto
├── notebooks/
│   ├── 01_estadistica_descriptiva/
│   ├── 02_probabilidad/
│   ├── 03_distribuciones/
│   ├── 04_inferencia/
│   ├── 05_regresion/
│   └── 06_anova_chi2/
│       ├── teoria.qmd
│       └── practica.ipynb
├── src/
│   └── stats_toolkit/         # Librería Python del curso
│       ├── __init__.py
│       ├── descriptive.py
│       ├── probability.py
│       ├── distributions.py
│       ├── inference.py
│       ├── regression.py
│       └── group_comparisons.py
├── images/                    # Recursos gráficos
└── .github/workflows/
    ├── deploy.yml              # Publica el libro en GitHub Pages (push a main)
    └── ci-check.yml            # Valida el render en cada Pull Request (sin publicar)
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

## Flujo de trabajo del proyecto

El repositorio sigue un flujo de **rama de desarrollo → Pull Request → producción**, pensado para validar cada cambio antes de que llegue al sitio publicado:

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
