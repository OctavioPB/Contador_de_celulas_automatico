"""Fallback geométrico para estimación de orientación mediante ajuste de elipse (Plan B)."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def estimate_orientation_ellipse(mask: np.ndarray) -> float:
    """Ajuste de elipse mínima a la máscara de instancia.

    Usa cv2.fitEllipse sobre los contornos de la máscara binaria para estimar
    el ángulo del eje mayor. Error esperado: ~12° en fibras regulares, hasta
    18° en irregulares.

    Args:
        mask: Máscara binaria (o escala de grises) de una fibra.
              Shape arbitrario, dtype uint8. Si es multicanal se convierte a gris.

    Returns:
        Ángulo del eje mayor de la elipse ajustada en [0°, 180°).
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Umbralizar para obtener máscara binaria robusta
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No se encontraron contornos. Retornando 0°.")
        return 0.0

    # Usar el contorno más grande
    largest = max(contours, key=cv2.contourArea)

    if len(largest) < 5:
        # fitEllipse requiere al menos 5 puntos; fallback a momentos
        logger.warning("Contorno con < 5 puntos. Usando momentos de imagen.")
        return _angle_from_moments(binary)

    try:
        _, _, angle = cv2.fitEllipse(largest)
        # cv2.fitEllipse devuelve ángulo en [0°, 180°) medido desde eje vertical
        # Convertir a convención desde eje horizontal
        angle = (angle - 90.0) % 180.0
    except cv2.error as exc:
        logger.warning("cv2.fitEllipse falló (%s). Usando momentos.", exc)
        angle = _angle_from_moments(binary)

    logger.debug("Ángulo estimado (ellipse): %.2f°", angle)
    return float(angle)


def _angle_from_moments(binary: np.ndarray) -> float:
    """Estima orientación usando momentos de imagen como fallback secundario.

    Args:
        binary: Imagen binaria uint8.

    Returns:
        Ángulo estimado en [0°, 180°).
    """
    moments = cv2.moments(binary)
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    if mu20 == mu02 and mu11 == 0:
        return 0.0

    angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    angle_deg = float(np.rad2deg(angle_rad)) % 180.0
    return angle_deg
