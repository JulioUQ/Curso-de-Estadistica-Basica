"""
Configuración común de la suite de tests de ``stats_toolkit``.

``stats_toolkit`` no es (todavía) un paquete instalable: los capítulos y
notebooks lo importan añadiendo ``src/`` a ``sys.path``. Aquí se replica ese
mismo mecanismo para los tests, de modo que no haya que instalar nada ni
adelantar la decisión de empaquetado (tarea F4.1 de la hoja de ruta).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
