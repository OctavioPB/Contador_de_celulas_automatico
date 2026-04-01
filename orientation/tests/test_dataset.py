"""Tests para data/dataset.py."""

import torch
import pytest
from data.dataset import FiberDataset


@pytest.fixture
def dataset():
    return FiberDataset(n_samples=20, size=128, noise_std=8.0, seed=42)


class TestFiberDataset:
    def test_len(self, dataset):
        """len(dataset) == n_samples."""
        assert len(dataset) == 20

    def test_image_shape(self, dataset):
        """image.shape == (1, 128, 128)."""
        image, target, theta = dataset[0]
        assert image.shape == (1, 128, 128)

    def test_image_dtype(self, dataset):
        """image.dtype == torch.float32."""
        image, target, theta = dataset[0]
        assert image.dtype == torch.float32

    def test_image_range(self, dataset):
        """image está normalizada a [0, 1]."""
        image, target, theta = dataset[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_target_shape(self, dataset):
        """target.shape == (2,)."""
        image, target, theta = dataset[0]
        assert target.shape == (2,)

    def test_target_dtype(self, dataset):
        """target.dtype == torch.float32."""
        image, target, theta = dataset[0]
        assert target.dtype == torch.float32

    def test_target_range(self, dataset):
        """Valores de target en [-1, 1]."""
        for idx in range(len(dataset)):
            _, target, _ = dataset[idx]
            assert target.min() >= -1.0
            assert target.max() <= 1.0

    def test_reproducibility(self):
        """dataset[i] es reproducible con la misma semilla."""
        ds1 = FiberDataset(n_samples=10, seed=123)
        ds2 = FiberDataset(n_samples=10, seed=123)
        img1, tgt1, theta1 = ds1[3]
        img2, tgt2, theta2 = ds2[3]
        assert torch.allclose(img1, img2)
        assert torch.allclose(tgt1, tgt2)
        assert abs(theta1 - theta2) < 1e-9

    def test_theta_range(self, dataset):
        """El ángulo theta está en [0, 180)."""
        for idx in range(len(dataset)):
            _, _, theta = dataset[idx]
            assert 0.0 <= theta < 180.0

    def test_different_seeds_different_data(self):
        """Semillas diferentes producen datos distintos."""
        ds1 = FiberDataset(n_samples=5, seed=1)
        ds2 = FiberDataset(n_samples=5, seed=2)
        _, _, theta1 = ds1[0]
        _, _, theta2 = ds2[0]
        assert abs(theta1 - theta2) > 1e-3
