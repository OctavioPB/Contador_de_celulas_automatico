"""Capa de integración: detección + orientación → resultado unificado.

API pública:
    run_analysis(image_path, detection_model, orientation_model) -> AnalysisResult
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Rutas de módulos
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ORIENT_DIR = os.path.join(_ROOT, "Orientation", "Orientador_De_Fibras_CNN")

for _p in (_ROOT, _ORIENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu

from Detection.detector import run_cellpose, _label_map_to_masks  # noqa: E402

_cnn_spec = _ilu.spec_from_file_location(
    "orient_cnn", os.path.join(_ORIENT_DIR, "model", "cnn.py")
)
_cnn_mod = _ilu.module_from_spec(_cnn_spec)
_cnn_spec.loader.exec_module(_cnn_mod)
FiberOrientationCNN = _cnn_mod.FiberOrientationCNN

_angle_spec = _ilu.spec_from_file_location(
    "orient_angle", os.path.join(_ORIENT_DIR, "utils", "angle.py")
)
_angle_mod = _ilu.module_from_spec(_angle_spec)
_angle_spec.loader.exec_module(_angle_mod)
target_to_theta = _angle_mod.target_to_theta

_ellipse_spec = _ilu.spec_from_file_location(
    "orient_ellipse", os.path.join(_ORIENT_DIR, "utils", "ellipse_fallback.py")
)
_ellipse_mod = _ilu.module_from_spec(_ellipse_spec)
_ellipse_spec.loader.exec_module(_ellipse_mod)
estimate_orientation_ellipse = _ellipse_mod.estimate_orientation_ellipse

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    count: int                            # número de células detectadas
    masks: List[np.ndarray]               # máscara binaria por célula (H, W) uint8
    angles: List[float]                   # ángulo CNN [0°, 180°) por célula
    areas: List[float]                    # área en píxeles por célula
    feature_angles: List[float]           # ángulo desde features [-90°, 90°) por célula
    overlay: np.ndarray                   # imagen BGR: overlay con contornos
    report_figure: np.ndarray             # figura BGR de 6 paneles para la UI
    used_fallback: List[bool] = field(default_factory=list)
    mean_area: float = 0.0
    std_area: float = 0.0
    cell_features: List[dict] = field(default_factory=list)  # regionprops por célula
    # Imágenes individuales para los tabs del visor
    img_original: Optional[np.ndarray] = None
    img_preprocessed: Optional[np.ndarray] = None
    img_segmentation: Optional[np.ndarray] = None
    img_cells: Optional[np.ndarray] = None
    img_area_hist: Optional[np.ndarray] = None
    img_orient_hist: Optional[np.ndarray] = None


def run_analysis(
    image_path: str,
    detection_model: str,
    orientation_model: str,
) -> AnalysisResult:
    """Ejecuta el pipeline completo y genera la figura de reporte de 6 paneles."""

    # 1. Cellpose: imagen + label map + flows
    logger.info(f"Ejecutando Cellpose en: {image_path}")
    img, label_map, flows = run_cellpose(image_path, detection_model)
    n_cells = int(label_map.max())
    logger.info(f"Células detectadas: {n_cells}")

    # 2. Preprocesado para visualización
    preprocessed = _preprocess(img)

    if n_cells == 0:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        report = _generate_report(img, preprocessed, flows, label_map, [], [], n_cells)
        return AnalysisResult(
            count=0, masks=[], angles=[], areas=[], feature_angles=[],
            overlay=img_bgr, report_figure=report,
        )

    # 3. Máscaras individuales
    masks = _label_map_to_masks(label_map)

    # 4. Áreas + features morfológicas (regionprops)
    from skimage.measure import regionprops
    props = regionprops(label_map)
    areas = [float(p.area) for p in props]
    cell_features = [
        {
            "label":       p.label,
            "area":        p.area,
            "perimeter":   round(p.perimeter, 3),
            "eccentricity": round(p.eccentricity, 4),
            "solidity":    round(p.solidity, 4),
            "major_axis":  round(p.major_axis_length, 4),
            "minor_axis":  round(p.minor_axis_length, 4),
            "orientation": round(math.degrees(p.orientation), 4),
            "centroid_x":  round(p.centroid[1], 4),
            "centroid_y":  round(p.centroid[0], 4),
        }
        for p in props
    ]

    # 5. Orientación geométrica (desde features) para el histograma → [-90, 90)
    feature_angles = [_ellipse_angle_signed(m) for m in masks]

    # 6. Orientación CNN (batch) para el overlay
    cnn_angles, used_fallback = _estimate_orientations_batch(masks, orientation_model)

    # 7. Overlay con contornos y vectores CNN
    overlay = _draw_overlay(img, masks, cnn_angles)

    # 8. Figura de 6 paneles
    report = _generate_report(img, preprocessed, flows, label_map, areas, feature_angles, n_cells)

    mean_area = float(np.mean(areas)) if areas else 0.0
    std_area = float(np.std(areas)) if areas else 0.0

    # Imágenes individuales para los tabs del visor
    img_original = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_prep = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    flow_rgb = flows[0] if flows and len(flows) > 0 else preprocessed
    img_seg = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR) if flow_rgb.ndim == 3 else cv2.cvtColor(flow_rgb, cv2.COLOR_GRAY2BGR)
    img_cells_bgr = cv2.cvtColor(_color_label_map(label_map), cv2.COLOR_RGB2BGR)
    img_area = _render_single_histogram(
        "Distribucion de Areas", "Area (pixeles)", areas, "#d9797a", bins=40
    )
    img_orient = _render_single_histogram(
        "Orientacion de Celulas (desde features)", "Angulo (grados)",
        feature_angles, "#d9797a", bins=36, range_=(-90, 90)
    )

    return AnalysisResult(
        count=n_cells,
        masks=masks,
        angles=cnn_angles,
        areas=areas,
        feature_angles=feature_angles,
        overlay=overlay,
        report_figure=report,
        used_fallback=used_fallback,
        mean_area=mean_area,
        std_area=std_area,
        cell_features=cell_features,
        img_original=img_original,
        img_preprocessed=img_prep,
        img_segmentation=img_seg,
        img_cells=img_cells_bgr,
        img_area_hist=img_area,
        img_orient_hist=img_orient,
    )


# ---------------------------------------------------------------------------
# Preprocesado
# ---------------------------------------------------------------------------

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises y aplica CLAHE para realzar contraste."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Orientación geométrica (desde features) → [-90, 90)
# ---------------------------------------------------------------------------

def _ellipse_angle_signed(mask: np.ndarray) -> float:
    """Ángulo del eje mayor de la elipse ajustada, en [-90°, 90°)."""
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    try:
        _, _, angle = cv2.fitEllipse(largest)
        # cv2.fitEllipse devuelve ángulo desde el eje vertical → convertir a horizontal
        angle = (angle - 90.0) % 180.0
        # Convertir [0, 180) → [-90, 90)
        return angle - 180.0 if angle > 90.0 else angle
    except cv2.error:
        return 0.0


# ---------------------------------------------------------------------------
# Orientación CNN en batch
# ---------------------------------------------------------------------------

def _estimate_orientations_batch(masks: List[np.ndarray], model_path: str):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cargando CNN de orientación en {device}…")

    cnn = FiberOrientationCNN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cnn.load_state_dict(checkpoint["model_state_dict"])
    else:
        cnn.load_state_dict(checkpoint)
    cnn.to(device)
    cnn.eval()

    tensors = []
    for mask in masks:
        resized = cv2.resize(mask.astype(np.uint8), (128, 128), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)
        tensors.append(t)

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = cnn(batch)

    angles, used_fallback = [], []
    for idx, (mask, out) in enumerate(zip(masks, outputs)):
        s, c = out[0].item(), out[1].item()
        norm = math.sqrt(s ** 2 + c ** 2)
        if norm > 1e-6:
            angle = target_to_theta(s / norm, c / norm)
            if math.isnan(angle):
                angle = estimate_orientation_ellipse(mask)
                used_fallback.append(True)
            else:
                used_fallback.append(False)
        else:
            angle = estimate_orientation_ellipse(mask)
            used_fallback.append(True)
        angles.append(float(angle))

    return angles, used_fallback


# ---------------------------------------------------------------------------
# Overlay de contornos + vectores CNN sobre la imagen original
# ---------------------------------------------------------------------------

def _draw_overlay(img: np.ndarray, masks: List[np.ndarray], angles: List[float]) -> np.ndarray:
    if img.ndim == 3:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for mask, angle in zip(masks, angles):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            length = 20
            rad = math.radians(angle)
            dx = int(length * math.cos(rad))
            dy = int(length * math.sin(rad))
            cv2.arrowedLine(out, (cx - dx, cy - dy), (cx + dx, cy + dy),
                            (0, 165, 255), 1, tipLength=0.3)
    return out


# ---------------------------------------------------------------------------
# Figura de 6 paneles
# ---------------------------------------------------------------------------

def _color_label_map(label_map: np.ndarray) -> np.ndarray:
    """Genera imagen RGB con cada célula en un color distinto sobre fondo azul."""
    import matplotlib.cm as cm

    n = int(label_map.max())
    H, W = label_map.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    colored[:, :] = [30, 60, 180]  # fondo azul (RGB)

    if n > 0:
        cmap = cm.get_cmap("gist_rainbow", n)
        for i in range(1, n + 1):
            r, g, b, _ = cmap(i / n)
            colored[label_map == i] = [int(r * 255), int(g * 255), int(b * 255)]

    return colored  # RGB


def _render_single_histogram(
    title: str, xlabel: str, data: List[float], color: str,
    bins: int = 40, range_: Optional[Tuple] = None
) -> np.ndarray:
    """Renderiza un único histograma como imagen BGR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor("#eeeeee")
    ax.set_facecolor("#e8e8e8")
    if data:
        kwargs: dict = dict(bins=bins, color=color, edgecolor="white", linewidth=0.5)
        if range_:
            kwargs["range"] = range_
        ax.hist(data, **kwargs)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.grid(axis="y", color="white", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def _generate_report(
    img: np.ndarray,
    preprocessed: np.ndarray,
    flows,
    label_map: np.ndarray,
    areas: List[float],
    feature_angles: List[float],
    n_cells: int,
) -> np.ndarray:
    """Renderiza la figura de 6 paneles a un array BGR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=120)
    fig.patch.set_facecolor("#eeeeee")

    bar_color = "#d9797a"
    grid_kw = dict(color="white", linewidth=0.8, alpha=0.7)

    # --- Panel 1: Imagen original ---
    if img.ndim == 3:
        axes[0, 0].imshow(img)
    else:
        axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Imagen Original", fontsize=13)
    axes[0, 0].axis("off")

    # --- Panel 2: Preprocesada (CLAHE) ---
    axes[0, 1].imshow(preprocessed, cmap="gray")
    axes[0, 1].set_title("Preprocesada", fontsize=13)
    axes[0, 1].axis("off")

    # --- Panel 3: Segmentación (flow RGB de Cellpose) ---
    flow_rgb = flows[0] if flows is not None and len(flows) > 0 else preprocessed
    axes[0, 2].imshow(flow_rgb, cmap="gray" if flow_rgb.ndim == 2 else None)
    axes[0, 2].set_title("Segmentación", fontsize=13)
    axes[0, 2].axis("off")

    # --- Panel 4: Células detectadas (colores por célula) ---
    colored = _color_label_map(label_map)
    axes[1, 0].imshow(colored)
    axes[1, 0].set_title(f"Células Detectadas: {n_cells}", fontsize=13)
    axes[1, 0].axis("off")

    # --- Panel 5: Histograma de áreas ---
    ax5 = axes[1, 1]
    ax5.set_facecolor("#e8e8e8")
    if areas:
        ax5.hist(areas, bins=40, color=bar_color, edgecolor="white", linewidth=0.5)
    ax5.set_title("Distribución de Áreas", fontsize=13)
    ax5.set_xlabel("Área (píxeles)", fontsize=10)
    ax5.set_ylabel("Frecuencia", fontsize=10)
    ax5.grid(axis="y", **grid_kw)
    ax5.spines[["top", "right"]].set_visible(False)

    # --- Panel 6: Histograma de orientaciones ---
    ax6 = axes[1, 2]
    ax6.set_facecolor("#e8e8e8")
    if feature_angles:
        ax6.hist(feature_angles, bins=36, range=(-90, 90),
                 color=bar_color, edgecolor="white", linewidth=0.5)
    ax6.set_title("Orientación de Células (desde features)", fontsize=13)
    ax6.set_xlabel("Ángulo (grados)", fontsize=10)
    ax6.set_ylabel("Frecuencia", fontsize=10)
    ax6.grid(axis="y", **grid_kw)
    ax6.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr_rgb = buf.reshape(h, w, 3)
    except AttributeError:
        # matplotlib >= 3.8: usar buffer_rgba (RGBA → recortar canal A)
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        arr_rgb = buf.reshape(h, w, 4)[:, :, :3]
    plt.close(fig)

    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
