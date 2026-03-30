# CLAUDE.md — SmartCell AI Analysis Studio

Guía de implementación para Claude Code. Lee este archivo completo antes de tocar
cualquier archivo del proyecto.

---

## Estructura del monorepo

```
smartcell/
├── ui/                         ← GUI Tkinter (Adrián Trejo) — NO modificar layout
├── detection/
│   ├── detection3A.ipynb       ← notebook original (solo referencia)
│   └── detector.py             ← ⬅ CREAR: módulo de inferencia Cellpose
├── orientation/                ← CNN HU5-v2 (Octavio) — NO modificar
│   ├── main.py                 ← expone estimate_fiber_orientation()
│   ├── model/cnn.py
│   ├── utils/angle.py
│   └── utils/ellipse_fallback.py
├── core/
│   └── pipeline.py             ← ⬅ CREAR: capa de integración
├── models/
│   ├── model_base_3A           ← modelo Cellpose (sin extensión, Cellpose agrega fecha)
│   └── cnn_orientation.pth     ← pesos CNN de orientación
├── tests/
│   └── test_pipeline.py        ← ⬅ CREAR: tests de integración
├── requirements.txt            ← ⬅ CREAR
└── CLAUDE.md                   ← este archivo
```

---

## Reglas absolutas

- **NO modificar** nada dentro de `ui/` excepto el callback del botón Analyze.
- **NO modificar** nada dentro de `orientation/`.
- `detection/detection3A.ipynb` es solo referencia — no se ejecuta en producción.
- Todo código nuevo va en `detection/detector.py`, `core/pipeline.py` o `tests/`.

---

## Orden de implementación obligatorio

```
1. detection/detector.py
2. core/pipeline.py
3. Conectar ui/ con pipeline
4. tests/test_pipeline.py
5. requirements.txt
```

No saltes pasos. Cada uno es prerequisito del siguiente.

---

## Paso 1 — Crear `detection/detector.py`

### Qué hace el notebook original

El notebook usa **Cellpose** (no Mask R-CNN) con un modelo custom entrenado:

```python
# Lógica extraída de detection3A.ipynb — celda 8
model = models.CellposeModel(gpu=True, pretrained_model=path_to_model)
masks, flows, styles = model.eval(img, diameter=None, channels=[0,0])
# masks: array 2D (H, W) — cada entero diferente es una célula distinta
# masks == 0 es fondo
```

### Módulo a crear

```python
# detection/detector.py
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
    # --- Resolver ruta del modelo (Cellpose agrega sufijo de fecha) ---
    resolved = _resolve_model_path(model_path)

    # --- Cargar imagen ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    img = skio.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    # --- Inferencia Cellpose ---
    device = "gpu"  # cae a CPU automáticamente si no hay GPU
    model = models.CellposeModel(gpu=True, pretrained_model=resolved)
    label_map, _, _ = model.eval(img, diameter=None, channels=[0, 0])

    # --- Convertir label map → lista de máscaras binarias ---
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
    n_cells = label_map.max()
    masks = []
    for i in range(1, n_cells + 1):
        binary = (label_map == i).astype(np.uint8) * 255
        masks.append(binary)
    return masks
```

---

## Paso 2 — Crear `core/pipeline.py`

### Interfaz de orientación (YA EXISTE en orientation/main.py)

```python
from orientation.main import estimate_fiber_orientation

# Firma:
angle: float = estimate_fiber_orientation(
    mask=binary_mask_uint8,   # np.ndarray (H, W) uint8
    model_path="models/cnn_orientation.pth"
)
# Retorna ángulo en grados [0°, 180°)
```

### Fallback geométrico (YA EXISTE en orientation/utils/ellipse_fallback.py)

Usarlo cuando `estimate_fiber_orientation` lanza excepción o retorna NaN.

```python
from orientation.utils.ellipse_fallback import estimate_orientation_ellipse
angle = estimate_orientation_ellipse(binary_mask_uint8)
```

### Módulo a crear

```python
# core/pipeline.py
"""Capa de integración: detección + orientación → resultado unificado.

API pública:
    run_analysis(image_path, detection_model, orientation_model) -> AnalysisResult
"""

import logging
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    count: int                          # número de fibras detectadas
    masks: List[np.ndarray]             # máscara binaria por fibra (H, W) uint8
    angles: List[float]                 # ángulo [0°, 180°) por fibra
    overlay: np.ndarray                 # imagen BGR con contornos + ángulos dibujados
    used_fallback: List[bool] = field(default_factory=list)  # True si usó elipse


def run_analysis(
    image_path: str,
    detection_model: str,
    orientation_model: str,
) -> AnalysisResult:
    """Ejecuta el pipeline completo sobre una imagen de microscopía.

    1. Detecta fibras con Cellpose (detection/detector.py)
    2. Estima orientación con CNN (orientation/main.py)
    3. Fallback a elipse mínima si CNN falla para alguna fibra
    4. Genera imagen con overlays

    Args:
        image_path: Ruta a la imagen de microscopía.
        detection_model: Ruta al modelo Cellpose (ej. "models/model_base_3A").
        orientation_model: Ruta al .pth de la CNN (ej. "models/cnn_orientation.pth").

    Returns:
        AnalysisResult con count, masks, angles, overlay y flags de fallback.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from detection.detector import detect_fibers
    from orientation.main import estimate_fiber_orientation
    from orientation.utils.ellipse_fallback import estimate_orientation_ellipse

    # --- 1. Detección ---
    logger.info(f"Detectando fibras en: {image_path}")
    masks = detect_fibers(image_path, detection_model)
    logger.info(f"Fibras detectadas: {len(masks)}")

    if not masks:
        img_bgr = cv2.imread(image_path)
        return AnalysisResult(count=0, masks=[], angles=[], overlay=img_bgr)

    # --- 2. Orientación por fibra ---
    angles, used_fallback = [], []
    for idx, mask in enumerate(masks):
        try:
            angle = estimate_fiber_orientation(mask, orientation_model)
            if angle is None or np.isnan(angle):
                raise ValueError("CNN retornó NaN")
            angles.append(float(angle))
            used_fallback.append(False)
            logger.debug(f"Fibra {idx}: {angle:.1f}° (CNN)")
        except Exception as e:
            logger.warning(f"Fibra {idx}: CNN falló ({e}), usando fallback elipse")
            angle = estimate_orientation_ellipse(mask)
            angles.append(float(angle))
            used_fallback.append(True)

    # --- 3. Overlay ---
    overlay = _draw_overlay(image_path, masks, angles)

    return AnalysisResult(
        count=len(masks),
        masks=masks,
        angles=angles,
        overlay=overlay,
        used_fallback=used_fallback,
    )


def _draw_overlay(image_path: str, masks: List[np.ndarray], angles: List[float]) -> np.ndarray:
    """Dibuja contornos y vectores de orientación sobre la imagen original."""
    img = cv2.imread(image_path)
    if img is None:
        h, w = masks[0].shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

    for mask, angle in zip(masks, angles):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        # Vector de orientación centrado en el centroide
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            length = 20
            import math
            rad = math.radians(angle)
            dx = int(length * math.cos(rad))
            dy = int(length * math.sin(rad))
            cv2.arrowedLine(img, (cx - dx, cy - dy), (cx + dx, cy + dy),
                            (0, 165, 255), 1, tipLength=0.3)

    return img
```

---

## Paso 3 — Conectar la UI

### Dónde está el botón Analyze

Busca en `ui/` el callback del botón "Analyze" (o "Analizar"). Probablemente se llama
`on_analyze_click`, `run_analysis_callback`, o similar. Puede estar en `main.py` o en
un archivo de controlador.

**Antes de modificar**, ejecuta:
```
Describe la estructura de ui/main.py: ¿dónde está el callback del botón Analyze
y cómo actualiza el visor de imagen y la tabla de resultados?
```

### Reemplazo del callback

Reemplaza el stub `"Análisis pendiente de implementar"` con:

```python
import threading
from core.pipeline import run_analysis, AnalysisResult

DETECTION_MODEL  = "models/model_base_3A"
ORIENTATION_MODEL = "models/cnn_orientation.pth"

def on_analyze_click(self):
    """Ejecuta el pipeline en un thread separado para no bloquear la UI."""
    if not self.current_image_path:
        return

    self._set_status("Analizando…")
    self._set_controls_enabled(False)

    def worker():
        try:
            result: AnalysisResult = run_analysis(
                self.current_image_path,
                DETECTION_MODEL,
                ORIENTATION_MODEL,
            )
            # Actualizar UI desde el thread principal
            self.after(0, lambda: self._on_analysis_complete(result))
        except Exception as e:
            self.after(0, lambda: self._on_analysis_error(str(e)))

    threading.Thread(target=worker, daemon=True).start()


def _on_analysis_complete(self, result: AnalysisResult):
    self._update_image_viewer(result.overlay)   # mostrar overlay en el visor
    self._update_results_table(result)           # poblar tabla con count y ángulos
    self._set_status(f"Listo — {result.count} fibras detectadas")
    self._set_controls_enabled(True)


def _on_analysis_error(self, message: str):
    self._set_status(f"Error: {message}")
    self._set_controls_enabled(True)
```

**Adapta los nombres de método** (`_update_image_viewer`, `_update_results_table`, etc.)
a los que ya existen en la UI de Adrián. No inventes métodos nuevos — busca los existentes.

---

## Paso 4 — Crear `tests/test_pipeline.py`

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_empty_image_returns_zero_count(tmp_path):
    """Si Cellpose no detecta nada, count debe ser 0 sin lanzar excepción."""
    import cv2
    img_path = str(tmp_path / "empty.png")
    cv2.imwrite(img_path, np.zeros((128, 128, 3), dtype=np.uint8))

    with patch("detection.detector.models.CellposeModel") as mock_cp:
        mock_cp.return_value.eval.return_value = (np.zeros((128, 128), dtype=int), None, None)
        with patch("core.pipeline.cv2.imread", return_value=np.zeros((128,128,3), np.uint8)):
            from core.pipeline import run_analysis
            result = run_analysis(img_path, "models/mock", "models/mock.pth")

    assert result.count == 0
    assert result.masks == []
    assert result.angles == []


def test_single_fiber_detection():
    """Una fibra detectada → count == 1 y len(angles) == 1."""
    fake_label_map = np.zeros((128, 128), dtype=int)
    fake_label_map[30:90, 30:90] = 1  # un cuadrado como fibra

    with patch("detection.detector.models.CellposeModel") as mock_cp, \
         patch("detection.detector.skio.imread", return_value=np.zeros((128,128,3))), \
         patch("orientation.main.estimate_fiber_orientation", return_value=45.0), \
         patch("core.pipeline.cv2.imread", return_value=np.zeros((128,128,3), np.uint8)):

        mock_cp.return_value.eval.return_value = (fake_label_map, None, None)

        from core.pipeline import run_analysis
        result = run_analysis("fake.png", "models/mock", "models/mock.pth")

    assert result.count == 1
    assert len(result.angles) == 1
    assert result.used_fallback == [False]


def test_cnn_failure_triggers_ellipse_fallback():
    """Si CNN falla en una fibra, used_fallback[i] == True y no lanza excepción."""
    fake_label_map = np.zeros((128, 128), dtype=int)
    fake_label_map[30:90, 30:90] = 1

    with patch("detection.detector.models.CellposeModel") as mock_cp, \
         patch("detection.detector.skio.imread", return_value=np.zeros((128,128,3))), \
         patch("orientation.main.estimate_fiber_orientation", side_effect=RuntimeError("GPU error")), \
         patch("orientation.utils.ellipse_fallback.estimate_orientation_ellipse", return_value=30.0), \
         patch("core.pipeline.cv2.imread", return_value=np.zeros((128,128,3), np.uint8)):

        mock_cp.return_value.eval.return_value = (fake_label_map, None, None)

        from core.pipeline import run_analysis
        result = run_analysis("fake.png", "models/mock", "models/mock.pth")

    assert result.count == 1
    assert result.used_fallback == [True]
    assert result.angles[0] == pytest.approx(30.0)
```

---

## Paso 5 — Crear `requirements.txt`

```
cellpose>=3.0
torch>=2.0
torchvision
opencv-python
scikit-image
numpy
ttkbootstrap
Pillow
```

> Nota: `cellpose` instala sus propias dependencias de PyTorch. Si ya tienes PyTorch
> con CUDA instalado, instala `cellpose` con `--no-deps` para evitar sobreescribir.

---

## Contratos de datos entre módulos

```
imagen (ruta)
    │
    ▼
detect_fibers(image_path, model_path)
    │  Cellpose label map → List[np.ndarray (H,W) uint8]
    │  cada array: 255=fibra, 0=fondo
    ▼
estimate_fiber_orientation(mask, model_path)  [por cada máscara]
    │  CNN forward pass → float en [0°, 180°)
    │  fallback: estimate_orientation_ellipse(mask) → float
    ▼
AnalysisResult
    ├── count: int
    ├── masks: List[ndarray]
    ├── angles: List[float]
    ├── overlay: ndarray BGR
    └── used_fallback: List[bool]
    │
    ▼
UI (on_analyze_click → _on_analysis_complete)
    ├── visor de imagen ← overlay
    └── tabla de resultados ← count + angles
```

---

## Puntos de fallo conocidos

| Punto | Fallo probable | Diagnóstico |
|---|---|---|
| `_resolve_model_path` | Modelo no encontrado | Confirmar nombre exacto en `models/` con `ls models/` |
| `estimate_fiber_orientation` | GPU OOM con batch grande | Normal en CPU — el fallback lo cubre |
| `_on_analysis_complete` | Nombres de método incorrectos en la UI | Leer `ui/main.py` antes de conectar |
| Threading en Tkinter | `RuntimeError: main thread` | Siempre usar `self.after(0, callback)` desde el worker |

---

## Comandos de verificación rápida

```bash
# Verificar que los módulos importan correctamente
python -c "from detection.detector import detect_fibers; print('OK detection')"
python -c "from orientation.main import estimate_fiber_orientation; print('OK orientation')"
python -c "from core.pipeline import run_analysis; print('OK pipeline')"

# Ejecutar tests
pytest tests/test_pipeline.py -v

# Lanzar la UI
python ui/main.py
```
