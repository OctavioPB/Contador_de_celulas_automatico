"""Funciones de conversión angular para representación circular de orientación de fibras."""

import math


def theta_to_target(theta_deg: float) -> tuple[float, float]:
    """Convierte ángulo θ en grados a representación circular (sin(2θ), cos(2θ)).

    Las fibras tienen simetría de 180°: θ y θ+180° son visualmente idénticas.
    La codificación en 2θ mapea [0°, 180°) a una vuelta completa del círculo,
    eliminando la discontinuidad en 0°/180°.

    Args:
        theta_deg: Ángulo en grados, rango [0, 180).

    Returns:
        Tupla (sin_2theta, cos_2theta), ambos en [-1, 1].
    """
    theta_rad = math.radians(theta_deg)
    return math.sin(2 * theta_rad), math.cos(2 * theta_rad)


def target_to_theta(sin_2t: float, cos_2t: float) -> float:
    """Convierte representación circular (sin(2θ), cos(2θ)) de vuelta a grados.

    Args:
        sin_2t: sin(2θ), rango [-1, 1].
        cos_2t: cos(2θ), rango [-1, 1].

    Returns:
        Ángulo θ en grados, rango [0, 180).
    """
    angle_2t = math.atan2(sin_2t, cos_2t)  # en [-π, π]
    theta_rad = angle_2t / 2.0              # en [-π/2, π/2]
    theta_deg = math.degrees(theta_rad)
    # Normalizar a [0, 180)
    theta_deg = theta_deg % 180.0
    return theta_deg


def angular_distance(a: float, b: float) -> float:
    """Distancia angular mínima entre dos ángulos respetando la simetría de 180°.

    Una fibra a 0° es idéntica a una a 180°, por lo que la distancia máxima
    posible entre dos orientaciones es 90°.

    Args:
        a: Primer ángulo en grados.
        b: Segundo ángulo en grados.

    Returns:
        Distancia en grados, rango [0, 90].
    """
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)
