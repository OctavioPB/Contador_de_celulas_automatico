"""Descarga automática de modelos desde Google Drive.

Uso:
    python models/download_models.py

O se invoca automáticamente al arrancar la app si los modelos no existen.
Edita MODELS_REGISTRY para añadir o cambiar modelos.
"""

import os
import sys

# Registro de modelos: { nombre_local: (drive_file_id, tamaño_aprox) }
MODELS_REGISTRY = {
    "model_base_3B": (
        "1rJzPz5gvGkDMWkkba7f81Y5hVr6yeqkd",
        "~1.2 GB",
    ),
    # "cnn_fiber_orientation.pth": (
    #     "REEMPLAZA_CON_EL_FILE_ID_ORIENTACION",
    #     "~50 MB",
    # ),
}

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def _file_id_from_url(url: str) -> str:
    """Extrae el file ID de una URL de Google Drive."""
    # Formatos soportados:
    #   https://drive.google.com/file/d/FILE_ID/view
    #   https://drive.google.com/open?id=FILE_ID
    #   https://drive.usercontent.google.com/download?id=FILE_ID
    import re
    m = re.search(r"(?:/d/|[?&]id=)([a-zA-Z0-9_-]{20,})", url)
    if m:
        return m.group(1)
    raise ValueError(f"No se pudo extraer el file ID de: {url}")


def ensure_models(registry: dict = None, models_dir: str = None, silent: bool = False) -> bool:
    """Descarga los modelos que falten. Retorna True si todos están disponibles.

    Args:
        registry:   Diccionario {nombre: (drive_id_o_url, tamaño)}. Por defecto MODELS_REGISTRY.
        models_dir: Directorio donde guardar. Por defecto la carpeta models/.
        silent:     Si True, no imprime progreso (útil cuando se llama desde la UI con diálogo propio).
    """
    try:
        import gdown
    except ImportError:
        print("[download_models] ERROR: gdown no está instalado. Ejecuta: pip install gdown>=5.1")
        return False

    registry   = registry or MODELS_REGISTRY
    models_dir = models_dir or _MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    all_ok = True
    for filename, (drive_ref, size_hint) in registry.items():
        dest = os.path.join(models_dir, filename)

        # Si ya existe (o existe con sufijo de fecha añadido por Cellpose), saltar
        import glob
        if os.path.isfile(dest) or glob.glob(dest + "*"):
            if not silent:
                print(f"  [ok] {filename} ya existe.")
            continue

        if drive_ref.startswith("REEMPLAZA"):
            print(f"  [skip] {filename}: file ID no configurado en MODELS_REGISTRY.")
            all_ok = False
            continue

        # Resolver ID si se pasó una URL completa
        file_id = _file_id_from_url(drive_ref) if drive_ref.startswith("http") else drive_ref

        if not silent:
            print(f"  Descargando {filename} ({size_hint}) desde Google Drive…")

        try:
            url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"
            result = gdown.download(url, output=dest, fuzzy=True, quiet=silent, resume=True)
            if result is None:
                raise RuntimeError(
                    "gdown retornó None — el archivo puede ser privado o el ID incorrecto."
                )
            if not silent:
                print(f"  [ok] {filename} descargado en: {result}")
        except Exception as e:
            print(f"  [error] No se pudo descargar {filename}: {e}")
            # Limpiar descarga parcial si existe y tiene menos de 1 KB (descarga fallida)
            if os.path.isfile(dest) and os.path.getsize(dest) < 1024:
                os.remove(dest)
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=== SmartCell AI — Descarga de modelos ===")
    ok = ensure_models()
    if ok:
        print("\nTodos los modelos están listos.")
    else:
        print("\nAlgunos modelos no pudieron descargarse. Revisa los file IDs en MODELS_REGISTRY.")
    sys.exit(0 if ok else 1)
