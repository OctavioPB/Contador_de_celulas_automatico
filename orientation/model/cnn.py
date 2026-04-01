"""Arquitectura CNN ligera para regresión de orientación de fibras musculares."""

import torch
import torch.nn as nn


class FiberOrientationCNN(nn.Module):
    """CNN de regresión para estimación de orientación de fibras musculares.

    Arquitectura:
        Conv2d(1→16, 5×5) → BatchNorm → ReLU → MaxPool(2)
        Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2)
        Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2)
        Conv2d(64→128, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(4×4)
        Flatten → Linear(2048→256) → ReLU → Dropout(0.3)
        Linear(256→2)  ← salida: (sin(2θ), cos(2θ))

    Input:  tensor (B, 1, 128, 128) float32, rango [0, 1]
    Output: tensor (B, 2) float32, sin normalizar (normalizar en inferencia)

    Args:
        dropout: Probabilidad de dropout (default: 0.3).
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: (B, 1, 128, 128) → (B, 16, 64, 64)
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 2: (B, 16, 64, 64) → (B, 32, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 3: (B, 32, 32, 32) → (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 4: (B, 64, 16, 16) → (B, 128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (B, 1, 128, 128), rango [0, 1].

        Returns:
            Tensor (B, 2) con (sin(2θ), cos(2θ)) sin normalizar.
        """
        x = self.features(x)
        return self.regressor(x)
