"""CLI para el módulo CNN de orientación de fibras musculares (HU5-v2).

Uso:
    python main.py train  --epochs 50 --samples 10000 --save models/cnn_v1.pth
    python main.py eval   --model models/cnn_v1.pth --n 100 --output results/eval.csv
    python main.py infer  --model models/cnn_v1.pth --image ruta/imagen.png
"""

import argparse
import logging
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.cnn import FiberOrientationCNN
from utils.angle import target_to_theta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interfaz pública de integración con Mask R-CNN (compatible con HU5-v1)
# ---------------------------------------------------------------------------

def estimate_fiber_orientation(mask: np.ndarray, model_path: str) -> float:
    """Interfaz de integración: recibe máscara binaria, retorna ángulo en [0°, 180°).

    Compatible con la interfaz de HU5-v1. Recibe la máscara binaria de una fibra
    segmentada por Mask R-CNN y devuelve el ángulo de orientación dominante.

    Nota: carga el modelo desde disco en cada llamada para mantener la firma
    compatible con HU5-v1. En pipelines de alto rendimiento, cargar el modelo
    una sola vez externamente y llamar al forward pass directamente.

    Args:
        mask: Array numpy (H, W) binario o uint8 con la región de la fibra.
        model_path: Ruta al modelo CNN guardado (.pth).

    Returns:
        Ángulo estimado en grados, rango [0°, 180°).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FiberOrientationCNN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    img = cv2.resize(mask.astype(np.uint8), (128, 128), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
    s, c = output[0, 0].item(), output[0, 1].item()
    norm = math.sqrt(s**2 + c**2)
    if norm > 0:
        s, c = s / norm, c / norm
    return target_to_theta(s, c)


# ---------------------------------------------------------------------------
# Comandos CLI
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Ejecuta el loop de entrenamiento."""
    from training.train import train
    history = train(
        n_epochs=args.epochs,
        n_samples_per_epoch=args.samples,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save,
        val_samples=args.val_samples,
    )
    print(f"\nEntrenamiento completado. Mejor Val MAE: {min(history['val_mae']):.2f}°")
    print(f"Modelo guardado en: {args.save}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Ejecuta la evaluación formal."""
    from training.evaluate import evaluate
    metrics = evaluate(
        model_path=args.model,
        n_images=args.n,
        output_csv=args.output,
    )
    print(f"\nResultados de evaluación ({args.n} imágenes):")
    print(f"  MAE:    {metrics['mae']:.2f}°")
    print(f"  < 3°:   {metrics['pct_lt3']:.1f}%")
    print(f"  < 5°:   {metrics['pct_lt5']:.1f}%")
    print(f"  < 10°:  {metrics['pct_lt10']:.1f}%")

    if metrics["mae"] < 3.0:
        print("  Resultado: EXCELENTE (MAE < 3°)")
    elif metrics["mae"] < 5.0:
        print("  Resultado: PRODUCCION (MAE < 5°)")
    elif metrics["mae"] < 10.0:
        print("  Resultado: ACEPTABLE (MAE < 10°)")
    else:
        print("  Resultado: REVISAR ARQUITECTURA (MAE >= 10°)")


def cmd_infer(args: argparse.Namespace) -> None:
    """Inferencia sobre una imagen PNG con visualización del vector de orientación."""
    import matplotlib.pyplot as plt

    if not os.path.exists(args.image):
        print(f"Error: imagen no encontrada: {args.image}")
        sys.exit(1)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Error: no se pudo cargar la imagen: {args.image}")
        sys.exit(1)

    # Preprocesamiento
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_LINEAR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FiberOrientationCNN()
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    s, c = output[0, 0].item(), output[0, 1].item()
    norm = math.sqrt(s**2 + c**2)
    if norm > 0:
        s, c = s / norm, c / norm
    theta = target_to_theta(s, c)

    print(f"Orientación estimada: {theta:.1f}°")

    # Visualización con vector superpuesto
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_resized, cmap="gray", origin="upper")
    ax.set_title(f"Orientación: {theta:.1f}°")
    ax.axis("off")

    cx, cy = 64, 64
    length = 40
    theta_rad = math.radians(theta)
    dx = length * math.cos(theta_rad)
    dy = length * math.sin(theta_rad)
    ax.annotate(
        "",
        xy=(cx + dx, cy + dy),
        xytext=(cx - dx, cy - dy),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNN de orientación de fibras musculares (HU5-v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train
    p_train = subparsers.add_parser("train", help="Entrenar la CNN")
    p_train.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    p_train.add_argument("--samples", type=int, default=10_000, help="Muestras por época")
    p_train.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate inicial")
    p_train.add_argument("--val-samples", type=int, default=1_000, dest="val_samples")
    p_train.add_argument("--save", type=str, default="models/cnn_fiber_orientation.pth")

    # -- eval
    p_eval = subparsers.add_parser("eval", help="Evaluar el modelo")
    p_eval.add_argument("--model", type=str, required=True, help="Ruta al modelo .pth")
    p_eval.add_argument("--n", type=int, default=100, help="Número de imágenes")
    p_eval.add_argument("--output", type=str, default="results/evaluation.csv")

    # -- infer
    p_infer = subparsers.add_parser("infer", help="Inferencia sobre una imagen")
    p_infer.add_argument("--model", type=str, required=True, help="Ruta al modelo .pth")
    p_infer.add_argument("--image", type=str, required=True, help="Ruta a la imagen PNG")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "infer":
        cmd_infer(args)
