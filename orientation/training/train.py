"""Loop de entrenamiento supervisado para la CNN de orientación de fibras."""

import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import FiberDataset
from model.cnn import FiberOrientationCNN
from utils.angle import target_to_theta, angular_distance

logger = logging.getLogger(__name__)


def _compute_val_mae(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Calcula el MAE angular en el conjunto de validación.

    Args:
        model: Modelo CNN en modo eval.
        val_loader: DataLoader de validación.
        device: Dispositivo de cómputo.

    Returns:
        MAE en grados.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for images, targets, thetas in val_loader:
            images = images.to(device)
            outputs = model(images)
            for i in range(len(thetas)):
                s, c = outputs[i, 0].item(), outputs[i, 1].item()
                norm = math.sqrt(s**2 + c**2)
                if norm > 0:
                    s, c = s / norm, c / norm
                pred_theta = target_to_theta(s, c)
                true_theta = float(thetas[i])
                errors.append(angular_distance(pred_theta, true_theta))
    return float(np.mean(errors))


def train(
    n_epochs: int = 50,
    n_samples_per_epoch: int = 10_000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    save_path: str = "models/cnn_fiber_orientation.pth",
    val_samples: int = 1_000,
) -> dict:
    """Entrena la CNN de regresión sobre datos sintéticos generados on-the-fly.

    Configuración:
        - Optimizador: Adam con lr=1e-3, weight_decay=1e-4
        - Loss: MSE entre (sin(2θ_pred), cos(2θ_pred)) y (sin(2θ_true), cos(2θ_true))
        - Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
        - Early stopping: si val_mae < 3° por 3 épocas consecutivas
        - Guarda checkpoint del mejor modelo (menor val_mae)

    Args:
        n_epochs: Número máximo de épocas.
        n_samples_per_epoch: Muestras sintéticas por época.
        batch_size: Tamaño de batch.
        learning_rate: Tasa de aprendizaje inicial.
        save_path: Ruta para guardar el modelo.
        val_samples: Muestras de validación (conjunto fijo por época).

    Returns:
        Diccionario con historial: {'train_loss': [...], 'val_mae': [...]}.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Usando dispositivo: %s", device)

    model = FiberOrientationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # Conjunto de validación fijo (misma semilla siempre)
    val_dataset = FiberDataset(n_samples=val_samples, seed=9999)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    history = {"train_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    early_stop_count = 0
    early_stop_patience = 3
    early_stop_threshold = 3.0  # grados

    for epoch in range(1, n_epochs + 1):
        model.train()
        # Nuevo dataset cada época → imágenes frescas con mismos ángulos (seed varía)
        train_dataset = FiberDataset(n_samples=n_samples_per_epoch, seed=epoch * 100)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        epoch_losses = []
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_mae = _compute_val_mae(model, val_loader, device)
        scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Época %2d/%d | Loss: %.4f | Val MAE: %5.1f° | LR: %.2e",
            epoch, n_epochs, train_loss, val_mae, current_lr,
        )
        print(
            f"Época {epoch:2d}/{n_epochs} | Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:5.1f}° | LR: {current_lr:.2e}"
        )

        # Guardar mejor modelo
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "train_loss": train_loss,
                },
                save_path,
            )
            logger.info("Checkpoint guardado en %s (MAE=%.2f°)", save_path, val_mae)

        # Early stopping: se requieren early_stop_patience épocas CONSECUTIVAS bajo
        # el umbral. El contador se reinicia si el MAE sube por encima del umbral,
        # evitando paradas prematuras por épocas buenas aisladas.
        if val_mae < early_stop_threshold:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                logger.info(
                    "Early stopping en época %d: val_mae=%.2f° < %.1f° por %d épocas",
                    epoch, val_mae, early_stop_threshold, early_stop_patience,
                )
                print(f"Early stopping: MAE={val_mae:.2f}° < {early_stop_threshold}° "
                      f"por {early_stop_patience} épocas consecutivas.")
                break
        else:
            early_stop_count = 0

    logger.info("Entrenamiento finalizado. Mejor val_mae=%.2f°", best_val_mae)

    # Guardar curvas de aprendizaje
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], color="#1f77b4", label="Train Loss")
    ax2.plot(epochs_range, history["val_mae"], color="#d62728", label="Val MAE (°)")
    ax2.axhline(y=5.0, color="#d62728", linestyle="--", linewidth=0.8,
                label="Umbral 5°")
    ax2.axhline(y=3.0, color="#ff7f0e", linestyle=":", linewidth=0.8,
                label="Umbral early stop 3°")

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Train Loss (MSE)", color="#1f77b4")
    ax2.set_ylabel("Val MAE (grados)", color="#d62728")
    ax1.set_title("Curvas de Aprendizaje — CNN Orientación de Fibras")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    curves_path = os.path.join(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        "learning_curves.png"
    )
    plt.savefig(curves_path, dpi=120)
    plt.close(fig)
    logger.info("Curvas de aprendizaje guardadas en: %s", curves_path)
    print(f"Curvas guardadas en: {curves_path}")

    return history
