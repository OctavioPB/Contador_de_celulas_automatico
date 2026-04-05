"""Módulo de detección de fibras musculares usando Cellpose (HU3).

API pública:
    detect_fibers(image_path, model_path) -> List[np.ndarray]
"""

import glob
import os
from typing import List

import numpy as np
from cellpose import models
from skimage import io as skio


def run_cellpose(image_path: str, model_path: str):
    """Ejecuta Cellpose y retorna imagen, label map y flows completos.

    Returns:
        (img, label_map, flows) donde:
            img: array numpy de la imagen original
            label_map: array (H, W) int con etiquetas por célula
            flows: lista Cellpose [rgb_flow, dP_cellprob, style]
    """
    resolved = _resolve_model_path(model_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    img = skio.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    model = models.CellposeModel(gpu=True, pretrained_model=resolved)
    label_map, flows, styles = model.eval(img, diameter=None, channels=[0, 0])
    return img, label_map, flows


def detect_fibers(image_path: str, model_path: str) -> List[np.ndarray]:
    """Detecta fibras musculares y retorna una máscara binaria por fibra.

    Args:
        image_path: Ruta absoluta a la imagen de microscopía (.tif, .png, .jpg).
        model_path: Ruta al modelo Cellpose (puede omitir el sufijo de fecha).
                    Si es un directorio, se usa el archivo más reciente que
                    empiece con el basename de model_path.

    Returns:
        Lista de arrays numpy (H, W) uint8 — uno por fibra detectada.
        Cada array es binario: 255 en la región de la fibra, 0 en el resto.
        Lista vacía si no se detecta ninguna fibra.

    Raises:
        FileNotFoundError: Si image_path o model_path no existen.
        ValueError: Si la imagen no puede leerse.
    """
    resolved = _resolve_model_path(model_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    img = skio.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    model = models.CellposeModel(gpu=True, pretrained_model=resolved)
    label_map, _, _ = model.eval(img, diameter=None, channels=[0, 0])

    return _label_map_to_masks(label_map)


def _resolve_model_path(model_path: str) -> str:
    """Resuelve el modelo aunque tenga sufijo de fecha añadido por Cellpose."""
    if os.path.isfile(model_path):
        return model_path

    folder = os.path.dirname(model_path) or "."
    basename = os.path.basename(model_path)
    candidates = glob.glob(os.path.join(folder, f"{basename}*"))

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró modelo '{basename}' en '{folder}'. "
            "Verifica que el archivo exista en models/."
        )
    return max(candidates, key=os.path.getctime)


def _label_map_to_masks(label_map: np.ndarray) -> List[np.ndarray]:
    """Convierte el array de etiquetas Cellpose en máscaras binarias individuales."""
    n_cells = int(label_map.max())
    masks = []
    for i in range(1, n_cells + 1):
        binary = (label_map == i).astype(np.uint8) * 255
        masks.append(binary)
    return masks
