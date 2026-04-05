"""Histograma polar angular reutilizable para visualización de distribución de orientaciones."""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_angular_histogram(
    angles: Sequence[float],
    n_bins: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula histograma de ángulos en [0°, 180°).

    Args:
        angles: Secuencia de ángulos en grados, rango [0°, 180°).
        n_bins: Número de intervalos (default: 18 → bins de 10°).

    Returns:
        Tupla (counts, bin_centers_deg) donde:
            counts: Array de frecuencias por bin.
            bin_centers_deg: Array con el centro de cada bin en grados.
    """
    angles_arr = np.asarray(angles, dtype=np.float64) % 180.0
    counts, edges = np.histogram(angles_arr, bins=n_bins, range=(0.0, 180.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    logger.debug("Histograma calculado: %d ángulos, %d bins", len(angles_arr), n_bins)
    return counts, bin_centers
