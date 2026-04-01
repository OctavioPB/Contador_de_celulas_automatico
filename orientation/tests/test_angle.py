"""Tests para utils/angle.py."""

import math
import pytest
from utils.angle import theta_to_target, target_to_theta, angular_distance


class TestThetaToTarget:
    def test_roundtrip(self):
        """target_to_theta(theta_to_target(θ)) ≈ θ para θ en [0, 180)."""
        for theta in range(0, 180, 5):
            s, c = theta_to_target(float(theta))
            recovered = target_to_theta(s, c)
            assert abs(recovered - theta) < 1e-9 or abs(recovered - theta) % 180 < 1e-9, (
                f"Roundtrip falló para θ={theta}: got {recovered}"
            )

    def test_symmetry_0_180(self):
        """theta_to_target(0) y theta_to_target(180) producen el mismo resultado."""
        s0, c0 = theta_to_target(0.0)
        s180, c180 = theta_to_target(180.0)
        assert abs(s0 - s180) < 1e-9
        assert abs(c0 - c180) < 1e-9

    def test_output_range(self):
        """Los valores de salida están en [-1, 1]."""
        for theta in range(0, 180, 10):
            s, c = theta_to_target(float(theta))
            assert -1.0 <= s <= 1.0
            assert -1.0 <= c <= 1.0

    def test_unit_norm(self):
        """La norma de (sin, cos) es siempre 1."""
        for theta in range(0, 180, 15):
            s, c = theta_to_target(float(theta))
            norm = math.sqrt(s**2 + c**2)
            assert abs(norm - 1.0) < 1e-9


class TestTargetToTheta:
    def test_output_range(self):
        """El ángulo recuperado está en [0, 180)."""
        for theta in range(0, 180, 5):
            s, c = theta_to_target(float(theta))
            recovered = target_to_theta(s, c)
            assert 0.0 <= recovered < 180.0, f"Fuera de rango: {recovered}"

    def test_known_values(self):
        """Verifica valores conocidos: 0°, 45°, 90°, 135°."""
        cases = [(0.0, 0.0, 1.0), (45.0, 1.0, 0.0), (90.0, 0.0, -1.0), (135.0, -1.0, 0.0)]
        for theta_expected, s, c in cases:
            result = target_to_theta(s, c)
            assert abs(result - theta_expected) < 1e-9, (
                f"Para (s={s}, c={c}) esperado {theta_expected}, got {result}"
            )


class TestAngularDistance:
    def test_zero_distance(self):
        """Distancia de un ángulo consigo mismo es 0."""
        for theta in [0.0, 45.0, 90.0, 135.0, 179.0]:
            assert angular_distance(theta, theta) == 0.0

    def test_symmetry_0_180(self):
        """angular_distance(0, 180) == 0 por simetría de fibras."""
        assert angular_distance(0.0, 180.0) == 0.0

    def test_orthogonal(self):
        """angular_distance(0, 90) == 90."""
        assert angular_distance(0.0, 90.0) == 90.0

    def test_symmetry_property(self):
        """La distancia es simétrica: d(a,b) == d(b,a)."""
        assert angular_distance(30.0, 120.0) == angular_distance(120.0, 30.0)

    def test_max_distance(self):
        """La distancia máxima nunca supera 90°."""
        for a in range(0, 180, 10):
            for b in range(0, 180, 10):
                d = angular_distance(float(a), float(b))
                assert 0.0 <= d <= 90.0, f"Fuera de rango: d({a},{b})={d}"

    def test_close_angles(self):
        """Ángulos cercanos con wrap: d(175, 5) debe ser 10."""
        assert abs(angular_distance(175.0, 5.0) - 10.0) < 1e-9
