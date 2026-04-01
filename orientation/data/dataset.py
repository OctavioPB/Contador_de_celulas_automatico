"""Dataset PyTorch con generación on-the-fly de imágenes sintéticas de fibras musculares."""

import logging
import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Permite importar desde el directorio raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.synthetic_generator import generate_fiber_image
from utils.angle import theta_to_target

logger = logging.getLogger(__name__)


class FiberDataset(Dataset):
    """Dataset sintético de fibras musculares para regresión angular.

    Genera imágenes on-the-fly con ángulos muestreados uniformemente en [0°, 180°).
    Cada época el modelo ve imágenes diferentes con los mismos ángulos → mejor
    generalización que un dataset fijo.

    Args:
        n_samples: Número de muestras por época.
        size: Tamaño de imagen en píxeles (default: 128).
        noise_std: Desviación estándar del ruido gaussiano (default: 8.0).
        seed: Semilla base para reproducibilidad. Cada índice usa seed+idx.
    """

    def __init__(
        self,
        n_samples: int,
        size: int = 128,
        noise_std: float = 8.0,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.size = size
        self.noise_std = noise_std
        self.seed = seed

        # Generar ángulos uniformes deterministas para este dataset
        rng = np.random.RandomState(seed)
        self._thetas = rng.uniform(0.0, 180.0, size=n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Genera una imagen sintética para el índice dado.

        Args:
            idx: Índice de la muestra.

        Returns:
            image: Tensor float32 shape (1, size, size), rango [0, 1].
            target: Tensor float32 shape (2,) = [sin(2θ), cos(2θ)].
            theta: Ángulo real en grados (para métricas, no para loss).
        """
        theta = float(self._thetas[idx])

        # n_fibers varía por muestra para evitar que la red memorice la densidad.
        # Nota: generate_fiber_image usa su propia RNG interna (seed = int(theta*1000)),
        # por lo que el contenido de píxeles está determinado por theta, no por seed+idx.
        # El efecto de epoch-to-epoch es que los thetas cambian (seed distinto por época).
        rng = np.random.RandomState(self.seed + idx)
        n_fibers = rng.randint(8, 16)

        img = generate_fiber_image(
            theta=theta,
            n_fibers=int(n_fibers),
            noise_std=self.noise_std,
            size=self.size,
        )

        # Normalizar a [0, 1] y añadir dimensión de canal
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        sin_2t, cos_2t = theta_to_target(theta)
        target = torch.tensor([sin_2t, cos_2t], dtype=torch.float32)

        logger.debug("Dataset[%d]: theta=%.1f°", idx, theta)
        return img_tensor, target, theta
