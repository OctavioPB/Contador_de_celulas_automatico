"""Evaluación formal del modelo CNN de orientación de fibras musculares."""

import csv
import logging
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import FiberDataset
from model.cnn import FiberOrientationCNN
from utils.angle import target_to_theta, angular_distance

logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    n_images: int = 100,
    output_csv: str = "results/evaluation.csv",
) -> dict:
    """Evalúa el modelo CNN sobre imágenes sintéticas con ángulos conocidos.

    Args:
        model_path: Ruta al modelo guardado (.pth).
        n_images: Número de imágenes de evaluación.
        output_csv: CSV de salida con columnas [theta_true, theta_predicted, error_deg].

    Returns:
        Diccionario con métricas:
            - mae: Error absoluto medio en grados.
            - pct_lt3: Porcentaje con error < 3°.
            - pct_lt5: Porcentaje con error < 5°.
            - pct_lt10: Porcentaje con error < 10°.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FiberOrientationCNN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = FiberDataset(n_samples=n_images, seed=77777)

    results = []
    with torch.no_grad():
        for idx in range(n_images):
            image, _, theta_true = dataset[idx]
            image = image.unsqueeze(0).to(device)

            output = model(image)
            s, c = output[0, 0].item(), output[0, 1].item()
            norm = math.sqrt(s**2 + c**2)
            if norm > 0:
                s, c = s / norm, c / norm
            theta_pred = target_to_theta(s, c)
            error = angular_distance(theta_pred, float(theta_true))
            results.append((float(theta_true), theta_pred, error))

    errors = [r[2] for r in results]
    mae = float(np.mean(errors))
    pct_lt3 = 100.0 * sum(e < 3.0 for e in errors) / len(errors)
    pct_lt5 = 100.0 * sum(e < 5.0 for e in errors) / len(errors)
    pct_lt10 = 100.0 * sum(e < 10.0 for e in errors) / len(errors)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_true", "theta_predicted", "error_deg"])
        writer.writerows(results)

    metrics = {
        "mae": mae,
        "pct_lt3": pct_lt3,
        "pct_lt5": pct_lt5,
        "pct_lt10": pct_lt10,
    }

    logger.info("Evaluación completada: MAE=%.2f°", mae)
    logger.info("  < 3°: %.1f%%  < 5°: %.1f%%  < 10°: %.1f%%", pct_lt3, pct_lt5, pct_lt10)
    print(f"MAE: {mae:.2f}°  |  <3°: {pct_lt3:.1f}%  <5°: {pct_lt5:.1f}%  <10°: {pct_lt10:.1f}%")
    print(f"CSV guardado en: {output_csv}")

    return metrics
