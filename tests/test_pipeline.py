"""Tests de integración para el pipeline de detección + orientación."""

import os
import sys

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Asegurar que la raíz del proyecto está en sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_empty_image_returns_zero_count(tmp_path):
    """Si Cellpose no detecta nada, count debe ser 0 sin lanzar excepción."""
    import cv2

    img_path = str(tmp_path / "empty.png")
    cv2.imwrite(img_path, np.zeros((128, 128, 3), dtype=np.uint8))

    with patch("detection.detector.models") as mock_models:
        mock_cp = MagicMock()
        mock_cp.eval.return_value = (np.zeros((128, 128), dtype=int), None, None)
        mock_models.CellposeModel.return_value = mock_cp

        with patch("core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):
            from core.pipeline import run_analysis

            result = run_analysis(img_path, "models/mock", "models/mock.pth")

    assert result.count == 0
    assert result.masks == []
    assert result.angles == []


def test_single_fiber_detection(tmp_path):
    """Una fibra detectada → count == 1 y len(angles) == 1."""
    import cv2

    img_path = str(tmp_path / "single.png")
    cv2.imwrite(img_path, np.zeros((128, 128, 3), dtype=np.uint8))

    fake_label_map = np.zeros((128, 128), dtype=int)
    fake_label_map[30:90, 30:90] = 1

    with patch("detection.detector.models") as mock_models, \
         patch("detection.detector.skio.imread", return_value=np.zeros((128, 128, 3))), \
         patch("core.pipeline.estimate_fiber_orientation", return_value=45.0), \
         patch("core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from core.pipeline import run_analysis

        result = run_analysis(img_path, "models/mock", "models/mock.pth")

    assert result.count == 1
    assert len(result.angles) == 1
    assert result.used_fallback == [False]


def test_cnn_failure_triggers_ellipse_fallback(tmp_path):
    """Si CNN falla en una fibra, used_fallback[i] == True y no lanza excepción."""
    import cv2

    img_path = str(tmp_path / "fallback.png")
    cv2.imwrite(img_path, np.zeros((128, 128, 3), dtype=np.uint8))

    fake_label_map = np.zeros((128, 128), dtype=int)
    fake_label_map[30:90, 30:90] = 1

    with patch("detection.detector.models") as mock_models, \
         patch("detection.detector.skio.imread", return_value=np.zeros((128, 128, 3))), \
         patch("core.pipeline.estimate_fiber_orientation", side_effect=RuntimeError("GPU error")), \
         patch("core.pipeline.estimate_orientation_ellipse", return_value=30.0), \
         patch("core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from core.pipeline import run_analysis

        result = run_analysis(img_path, "models/mock", "models/mock.pth")

    assert result.count == 1
    assert result.used_fallback == [True]
    assert result.angles[0] == pytest.approx(30.0)


def test_multiple_fibers(tmp_path):
    """Múltiples fibras → count correcto y lista de ángulos con misma longitud."""
    import cv2

    img_path = str(tmp_path / "multi.png")
    cv2.imwrite(img_path, np.zeros((256, 256, 3), dtype=np.uint8))

    fake_label_map = np.zeros((256, 256), dtype=int)
    fake_label_map[10:50, 10:50] = 1
    fake_label_map[100:140, 100:140] = 2
    fake_label_map[180:220, 180:220] = 3

    with patch("detection.detector.models") as mock_models, \
         patch("detection.detector.skio.imread", return_value=np.zeros((256, 256, 3))), \
         patch("core.pipeline.estimate_fiber_orientation", side_effect=[10.0, 45.0, 90.0]), \
         patch("core.pipeline.cv2.imread", return_value=np.zeros((256, 256, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from core.pipeline import run_analysis

        result = run_analysis(img_path, "models/mock", "models/mock.pth")

    assert result.count == 3
    assert len(result.angles) == 3
    assert len(result.used_fallback) == 3
    assert result.used_fallback == [False, False, False]


# =============================================================================
# Tests de preprocesamiento (_preprocess)
# =============================================================================

def test_preprocess_normal_image():
    """Test 1 — imagen normal: resultado uint8, misma forma, std > 0."""
    from core.pipeline import _preprocess

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    result = _preprocess(img)

    assert result.dtype == np.uint8
    assert result.shape == (128, 128)
    assert np.std(result) > 0


def test_preprocess_dark_image():
    """Test 2 — imagen muy oscura: CLAHE aumenta el brillo medio."""
    from core.pipeline import _preprocess

    rng = np.random.default_rng(1)
    img = rng.integers(0, 21, (128, 128), dtype=np.uint8)
    result = _preprocess(img)

    assert np.mean(result) > np.mean(img)


def test_preprocess_bright_image():
    """Test 3 — imagen muy brillante: salida uint8 válida sin excepción."""
    from core.pipeline import _preprocess

    rng = np.random.default_rng(2)
    img = rng.integers(235, 256, (128, 128), dtype=np.uint8)
    result = _preprocess(img)

    assert result.dtype == np.uint8
    assert result.shape == (128, 128)


def test_preprocess_noisy_image():
    """Test 4 — imagen con ruido gaussiano alto: filtro reduce la desviación estándar."""
    from core.pipeline import _preprocess

    rng = np.random.default_rng(3)
    raw = rng.normal(128, 40, (128, 128))
    img = np.clip(raw, 0, 255).astype(np.uint8)
    result = _preprocess(img)

    assert np.std(result) < np.std(img)


def test_preprocess_rgb_image():
    """Test 5 — imagen RGB de 3 canales: _preprocess la convierte a gris (2D)."""
    from core.pipeline import _preprocess

    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    result = _preprocess(img)

    assert result.ndim == 2
    assert result.shape == (128, 128)
