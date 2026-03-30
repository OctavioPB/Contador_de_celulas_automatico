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

    with patch("Detection.detector.models") as mock_models:
        mock_cp = MagicMock()
        mock_cp.eval.return_value = (np.zeros((128, 128), dtype=int), None, None)
        mock_models.CellposeModel.return_value = mock_cp

        with patch("Core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):
            from Core.pipeline import run_analysis

            result = run_analysis(img_path, "Models/mock", "Models/mock.pth")

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

    with patch("Detection.detector.models") as mock_models, \
         patch("Detection.detector.skio.imread", return_value=np.zeros((128, 128, 3))), \
         patch("Core.pipeline.estimate_fiber_orientation", return_value=45.0), \
         patch("Core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from Core.pipeline import run_analysis

        result = run_analysis(img_path, "Models/mock", "Models/mock.pth")

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

    with patch("Detection.detector.models") as mock_models, \
         patch("Detection.detector.skio.imread", return_value=np.zeros((128, 128, 3))), \
         patch("Core.pipeline.estimate_fiber_orientation", side_effect=RuntimeError("GPU error")), \
         patch("Core.pipeline.estimate_orientation_ellipse", return_value=30.0), \
         patch("Core.pipeline.cv2.imread", return_value=np.zeros((128, 128, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from Core.pipeline import run_analysis

        result = run_analysis(img_path, "Models/mock", "Models/mock.pth")

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

    with patch("Detection.detector.models") as mock_models, \
         patch("Detection.detector.skio.imread", return_value=np.zeros((256, 256, 3))), \
         patch("Core.pipeline.estimate_fiber_orientation", side_effect=[10.0, 45.0, 90.0]), \
         patch("Core.pipeline.cv2.imread", return_value=np.zeros((256, 256, 3), np.uint8)):

        mock_cp = MagicMock()
        mock_cp.eval.return_value = (fake_label_map, None, None)
        mock_models.CellposeModel.return_value = mock_cp

        from Core.pipeline import run_analysis

        result = run_analysis(img_path, "Models/mock", "Models/mock.pth")

    assert result.count == 3
    assert len(result.angles) == 3
    assert len(result.used_fallback) == 3
    assert result.used_fallback == [False, False, False]
