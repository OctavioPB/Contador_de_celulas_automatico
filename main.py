"""Launcher principal de SmartCell AI Analysis Studio.

Ejecutar desde la raíz del proyecto:
    python main.py
"""

import os
import sys

# Asegurar que la raíz del proyecto está en sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Lanzar la UI
_APP_PATH = os.path.join(_ROOT, "ui", "app.py")

if not os.path.exists(_APP_PATH):
    print(f"Error: No se encontró app.py en {_APP_PATH}")
    sys.exit(1)

# Cambiar el directorio de trabajo a la raíz del proyecto para que las
# rutas relativas dentro de app.py sean consistentes
os.chdir(_ROOT)

# runpy.run_path ejecuta el archivo con __name__ == "__main__",
# lo que activa el bloque if __name__ == "__main__": de app.py
import runpy
runpy.run_path(_APP_PATH, run_name="__main__")
