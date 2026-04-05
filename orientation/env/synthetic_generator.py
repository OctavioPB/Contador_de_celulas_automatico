"""Generador de imágenes sintéticas de fibras musculares con orientación angular controlada."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_fiber_image(
    theta: float,
    n_fibers: int = 12,
    noise_std: float = 8.0,
    size: int = 128,
) -> np.ndarray:
    """Genera imagen sintética de fibras con orientación theta (grados).

    Args:
        theta:      Ángulo de orientación dominante en grados, rango [0, 180).
        n_fibers:   Número de fibras a dibujar (5–20).
        noise_std:  Desviación estándar del ruido gaussiano (0–15).
        size:       Tamaño del lado de la imagen cuadrada en píxeles.

    Returns:
        Array uint8 de shape (size, size) con las fibras sintetizadas.
    """
    image = np.zeros((size, size), dtype=np.float32)

    rng = np.random.RandomState(int(theta * 1000) % 2147483648)

    fiber_length_range = (30, 60)
    fiber_width_range = (4, 8)

    angle_rad = np.deg2rad(theta)
    dx = np.cos(angle_rad)  # noqa: F841
    dy = np.sin(angle_rad)  # noqa: F841

    for _ in range(n_fibers):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        length = rng.randint(fiber_length_range[0], fiber_length_range[1] + 1)
        width = rng.randint(fiber_width_range[0], fiber_width_range[1] + 1)

        angle_jitter = rng.uniform(-5.0, 5.0)
        fiber_angle = theta + angle_jitter

        axes = (length // 2, width // 2)
        center = (cx, cy)
        cv2.ellipse(
            image,
            center=center,
            axes=axes,
            angle=fiber_angle,
            startAngle=0,
            endAngle=360,
            color=rng.uniform(180, 255),
            thickness=-1,
        )

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=image.shape).astype(np.float32)
        image = image + noise

    image = np.clip(image, 0, 255).astype(np.uint8)
    logger.debug("Imagen generada: theta=%.1f, n_fibers=%d, noise_std=%.1f", theta, n_fibers, noise_std)
    return image
