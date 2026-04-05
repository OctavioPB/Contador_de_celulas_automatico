"""Tests para model/cnn.py."""

import torch
import pytest
from model.cnn import FiberOrientationCNN


@pytest.fixture
def model():
    m = FiberOrientationCNN()
    m.eval()
    return m


class TestFiberOrientationCNN:
    def test_forward_output_shape(self, model):
        """Forward pass con batch (4, 1, 128, 128) produce output (4, 2)."""
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_dtype(self, model):
        """Output es float32."""
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_parameter_count(self, model):
        """El número de parámetros es < 2M."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 2_000_000, f"Modelo demasiado grande: {n_params:,} parámetros"

    def test_deterministic_eval(self, model):
        """model.eval() + torch.no_grad() produce output determinista."""
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_batch_size_1(self, model):
        """Funciona con batch size 1."""
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_batch_size_32(self, model):
        """Funciona con batch size 32."""
        x = torch.randn(32, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 2)

    def test_custom_dropout(self):
        """Acepta dropout personalizado."""
        model = FiberOrientationCNN(dropout=0.5)
        x = torch.randn(2, 1, 128, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_no_nan_in_output(self, model):
        """El output no contiene NaN."""
        x = torch.rand(4, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
