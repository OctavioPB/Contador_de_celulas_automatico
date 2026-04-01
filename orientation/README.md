# HU5-v2 — Orientador de Fibras Musculares (CNN)

Estimador de orientación angular de fibras musculares mediante CNN de regresión supervisada. Segunda versión del módulo HU5 de la tesis de maestría (UNIR), que reemplaza el enfoque de Reinforcement Learning (HU5-v1 / PPO).

---

## Por qué se abandonó el enfoque RL

La versión anterior (HU5-v1) entrenaba un agente PPO durante 500k pasos para que, mediante 200 acciones secuenciales, convergiera a la orientación correcta. Los resultados mostraron un problema estructural:

| Métrica | HU5-v1 (PPO) |
|---|---|
| MAE global | 26° |
| Casos correctos | 49% |
| Fallos completos | 28% |
| Distribución del error | Bimodal |

El análisis del agente reveló que el 28% de los episodios terminaba en un mínimo local del espacio de acciones. El problema raíz es que **estimar la orientación de una fibra es inherentemente una regresión, no un problema de control secuencial**: el ángulo puede determinarse en una sola pasada forward, sin necesidad de iterar 200 pasos. El RL añadía complejidad sin justificación algorítmica.

---

## Solución CNN (HU5-v2)

Una CNN ligera resuelve el problema directamente: recibe la imagen y produce el ángulo en ~5ms.

### Innovación clave: codificación angular en (sin 2θ, cos 2θ)

Las fibras tienen **simetría de 180°**: una fibra a 0° y una a 180° son visualmente idénticas. Si la red predijera θ directamente, el modelo vería 179° y 1° como valores muy distintos cuando en realidad están a 2° de distancia.

La codificación en 2θ resuelve esto mapeando [0°, 180°) a una vuelta completa del círculo unitario, eliminando la discontinuidad. En inferencia, la salida cruda (s, c) se normaliza antes de convertir a ángulo:

```
θ = ½ · atan2(s, c)  →  rango [0°, 180°)
```

### Entrenamiento completamente sintético

Al igual que HU5-v1, no se requieren datos reales. El generador de imágenes de HU5-v1 (`env/synthetic_generator.py`) se reutiliza sin modificaciones. Cada época genera 10 000 imágenes nuevas on-the-fly, lo que mejora la generalización respecto a un dataset fijo.

---

## Comparación directa

| Aspecto | HU5-v1 (PPO) | HU5-v2 (CNN) |
|---|---|---|
| Paradigma | Control secuencial | Regresión directa |
| Fundamento teórico | MDP + política óptima | Aprendizaje supervisado |
| Tiempo de inferencia | ~2 s (200 pasos) | ~5 ms (1 forward pass) |
| MAE | ~26° (distribución bimodal) | < 5° (objetivo producción) |
| Entrenamiento | 500k timesteps (~1 h) | 50 épocas × 10k muestras (~10 min CPU) |
| Fallos catastróficos | 28% de los casos | No (regresión continua) |
| Datos reales necesarios | 0 | 0 |
| Complejidad de implementación | Alta | Media |
| Parámetros del modelo | ~300k (actor PPO) | ~500k (CNN) |

---

## Estructura del proyecto

```
Orientador_De_Fibras_CNN/
├── main.py                     CLI: train / eval / infer
├── requirements.txt
│
├── env/
│   └── synthetic_generator.py  Generador sintético (reutilizado de HU5-v1)
│
├── data/
│   └── dataset.py              FiberDataset — generación on-the-fly
│
├── model/
│   └── cnn.py                  FiberOrientationCNN (~500k parámetros)
│
├── training/
│   ├── train.py                Loop supervisado + early stopping
│   └── evaluate.py             Evaluación formal con CSV
│
├── utils/
│   ├── angle.py                theta_to_target / target_to_theta / angular_distance
│   ├── histogram.py            Histograma polar angular
│   └── ellipse_fallback.py     Fallback geométrico (HU5-v1)
│
└── tests/                      38 tests — 100% pass
    ├── test_angle.py
    ├── test_dataset.py
    ├── test_model.py
    └── test_evaluate.py
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

```bash
# Entrenar
python main.py train --epochs 50 --samples 10000 --save models/cnn_v1.pth

# Evaluar (genera results/evaluation.csv)
python main.py eval --model models/cnn_v1.pth --n 100 --output results/eval.csv

# Inferencia sobre imagen PNG (muestra imagen con vector superpuesto)
python main.py infer --model models/cnn_v1.pth --image ruta/imagen.png
```

---

## Arquitectura CNN

```
Input (B, 1, 128, 128)
  → Conv(1→16, 5×5) + BN + ReLU + MaxPool(2)   →  (B, 16, 64, 64)
  → Conv(16→32, 3×3) + BN + ReLU + MaxPool(2)  →  (B, 32, 32, 32)
  → Conv(32→64, 3×3) + BN + ReLU + MaxPool(2)  →  (B, 64, 16, 16)
  → Conv(64→128, 3×3) + BN + ReLU + AvgPool(4) →  (B, 128, 4, 4)
  → Linear(2048→256) + ReLU + Dropout(0.3)
  → Linear(256→2)
Output (B, 2): [sin(2θ), cos(2θ)]
```

**Loss:** MSE sobre (sin 2θ, cos 2θ).
**Optimizador:** Adam (lr=1e-3, weight_decay=1e-4).
**Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5).
**Early stopping:** val MAE < 3° durante 3 épocas consecutivas.

---

## Criterios de aceptación

| Umbral | Significado |
|---|---|
| MAE < 3° | Excelencia |
| MAE < 5° | Producción |
| MAE < 10° | Aceptable |
| MAE ≥ 10° | Revisar arquitectura |

---

## Integración con el sistema mayor

La interfaz pública es compatible con HU5-v1:

```python
from main import estimate_fiber_orientation

# mask: np.ndarray binario de una fibra segmentada por Mask R-CNN
theta = estimate_fiber_orientation(mask, model_path="models/cnn_v1.pth")
# → float en [0°, 180°)
```

```
[Mask R-CNN HU2/HU3] → máscara binaria por fibra
        ↓
[HU5-v2 CNN]  → estimate_fiber_orientation(mask, model_path)
        ↓
[Histograma angular + CSV — HU6/GUI]
```
