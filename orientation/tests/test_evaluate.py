"""Tests para training/evaluate.py."""

import os
import tempfile
import math
import torch
import pytest
from model.cnn import FiberOrientationCNN
from training.evaluate import evaluate


@pytest.fixture
def saved_model(tmp_path):
    """Guarda un modelo CNN sin entrenar en un archivo temporal."""
    model = FiberOrientationCNN()
    model_path = str(tmp_path / "test_model.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "val_mae": 99.0, "train_loss": 1.0},
        model_path,
    )
    return model_path


class TestEvaluate:
    def test_returns_dict_with_metrics(self, saved_model, tmp_path):
        """evaluate() devuelve dict con mae, pct_lt3, pct_lt5, pct_lt10."""
        csv_path = str(tmp_path / "eval.csv")
        metrics = evaluate(model_path=saved_model, n_images=10, output_csv=csv_path)
        assert "mae" in metrics
        assert "pct_lt3" in metrics
        assert "pct_lt5" in metrics
        assert "pct_lt10" in metrics

    def test_mae_is_float(self, saved_model, tmp_path):
        """mae es un float."""
        csv_path = str(tmp_path / "eval.csv")
        metrics = evaluate(model_path=saved_model, n_images=10, output_csv=csv_path)
        assert isinstance(metrics["mae"], float)

    def test_mae_range(self, saved_model, tmp_path):
        """mae está en [0, 90]."""
        csv_path = str(tmp_path / "eval.csv")
        metrics = evaluate(model_path=saved_model, n_images=10, output_csv=csv_path)
        assert 0.0 <= metrics["mae"] <= 90.0

    def test_percentages_range(self, saved_model, tmp_path):
        """Porcentajes están en [0, 100]."""
        csv_path = str(tmp_path / "eval.csv")
        metrics = evaluate(model_path=saved_model, n_images=10, output_csv=csv_path)
        for key in ["pct_lt3", "pct_lt5", "pct_lt10"]:
            assert 0.0 <= metrics[key] <= 100.0

    def test_csv_created(self, saved_model, tmp_path):
        """El archivo CSV se crea correctamente."""
        csv_path = str(tmp_path / "eval.csv")
        evaluate(model_path=saved_model, n_images=10, output_csv=csv_path)
        assert os.path.exists(csv_path)

    def test_csv_row_count(self, saved_model, tmp_path):
        """El CSV tiene n_images filas de datos + 1 de cabecera."""
        csv_path = str(tmp_path / "eval.csv")
        evaluate(model_path=saved_model, n_images=15, output_csv=csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 16  # 1 header + 15 rows

    def test_csv_header(self, saved_model, tmp_path):
        """El CSV tiene la cabecera correcta."""
        csv_path = str(tmp_path / "eval.csv")
        evaluate(model_path=saved_model, n_images=5, output_csv=csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "theta_true,theta_predicted,error_deg"

    def test_pct_ordering(self, saved_model, tmp_path):
        """pct_lt10 >= pct_lt5 >= pct_lt3 siempre."""
        csv_path = str(tmp_path / "eval.csv")
        metrics = evaluate(model_path=saved_model, n_images=20, output_csv=csv_path)
        assert metrics["pct_lt10"] >= metrics["pct_lt5"]
        assert metrics["pct_lt5"] >= metrics["pct_lt3"]
