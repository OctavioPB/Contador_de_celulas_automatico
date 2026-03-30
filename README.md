# SmartCell AI Analysis Studio

Aplicación de escritorio para el análisis automático de imágenes de microscopía de fibras musculares. Detecta y cuenta células de forma automática, estima su orientación angular y genera reportes visuales y tabulares exportables.

Desarrollado como parte de una tesis de maestría en la Universidad Internacional de La Rioja (UNIR).

---

## Características principales

- **Segmentación celular** con [Cellpose](https://github.com/MouseLand/cellpose) mediante un modelo personalizado entrenado sobre microscopía muscular
- **Estimación de orientación** con una CNN de regresión ligera (HU5-v2), con fallback geométrico por elipse mínima
- **Visualización en 7 paneles**: imagen original, preprocesada (CLAHE), flujo de segmentación, células coloreadas, histograma de áreas y de orientación
- **Métricas morfológicas** por célula: área, perímetro, excentricidad, solidez, ejes, orientación, centroide
- **Exportación** a PDF con tablas formateadas y a CSV por sección de análisis
- **GPU opcional**: detección automática de CUDA para acelerar la inferencia
- **Descarga automática del modelo** Cellpose desde Google Drive al primer arranque

---

## Arquitectura del sistema

```
smartcell/
├── main.py                        ← Punto de entrada principal
├── requirements.txt
│
├── Core/
│   └── pipeline.py                ← Integración: detección + orientación + métricas
│
├── Detection/
│   └── detector.py                ← Inferencia Cellpose → máscaras binarias
│
├── Orientation/
│   └── Orientador_De_Fibras_CNN/  ← CNN HU5-v2 (regresión angular)
│       ├── main.py                ← estimate_fiber_orientation()
│       ├── model/cnn.py           ← FiberOrientationCNN (~500k parámetros)
│       └── utils/
│           ├── angle.py           ← Codificación (sin2θ, cos2θ)
│           └── ellipse_fallback.py
│
├── Models/
│   ├── model_base_3B              ← Modelo Cellpose (descargado automáticamente)
│   └── download_models.py         ← Script de descarga desde Google Drive
│
├── UI/
│   └── .../app.py                 ← Interfaz gráfica (ttkbootstrap)
│
└── tests/
    └── test_pipeline.py
```

---

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| Almacenamiento | 3 GB libres | 5 GB |
| GPU (opcional) | — | NVIDIA con CUDA 11.8+ |
| SO | Windows 10 / Linux | Windows 11 / Ubuntu 22.04 |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/Contador_de_celulas_automatico.git
cd Contador_de_celulas_automatico
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** `cellpose` instala su propia versión de PyTorch. Si ya tienes PyTorch instalado con soporte CUDA, consulta la sección [Activar GPU](#activar-gpu-opcional) antes de este paso.

---

## Ejecución

```bash
python main.py
```

La aplicación se abrirá. Si el modelo Cellpose no está descargado, aparecerá un diálogo automático para descargarlo desde Google Drive (~1.2 GB).

---

## Modelo de detección Cellpose

El modelo `model_base_3B` tiene un peso de ~1.2 GB y **no está incluido en el repositorio** por sus dimensiones. Está alojado en Google Drive y se gestiona de dos formas:

### Opción A — Descarga automática (recomendada)

Al ejecutar `python main.py` por primera vez sin el modelo, la aplicación preguntará si deseas descargarlo. Al confirmar, se descargará automáticamente con barra de progreso.

### Opción B — Descarga manual desde terminal

```bash
python Models/download_models.py
```

### Opción C — Descarga manual directa

1. Descarga el archivo desde: https://drive.google.com/file/d/1rJzPz5gvGkDMWkkba7f81Y5hVr6yeqkd/view
2. Colócalo en la carpeta `Models/` con el nombre exacto `model_base_3B`

Una vez descargado, el modelo se reutiliza en todos los análisis posteriores sin volver a descargar.

---

## Activar GPU (opcional)

La GPU acelera significativamente el paso de segmentación Cellpose. La CNN de orientación es suficientemente ligera para CPU.

### Verificar si PyTorch detecta CUDA

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Instalar PyTorch con soporte CUDA

Si el resultado es `False` pero tienes una GPU NVIDIA, reinstala PyTorch con CUDA. Elige según tu versión de driver:

```bash
# CUDA 12.8 (driver >= 520, RTX 30xx / 40xx / 50xx)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8 (driver >= 450)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Consulta tu versión de driver en `nvidia-smi`. La versión del driver determina la versión máxima de CUDA compatible.

Verifica la versión de tu driver CUDA:
```bash
nvidia-smi
```

### Instalar Cellpose sin sobreescribir PyTorch

Si ya tienes PyTorch con CUDA instalado, instala Cellpose sin sus dependencias para evitar que sobreescriba la versión de PyTorch:

```bash
pip install cellpose --no-deps
pip install cellpose-omnipose roifile  # dependencias de cellpose que sí son necesarias
```

### Comportamiento sin GPU

Si no hay GPU disponible, tanto Cellpose como la CNN de orientación recaen automáticamente a CPU. El análisis funciona igualmente, pero puede tardar significativamente más (5-15 minutos por imagen de alta resolución vs. 30-60 segundos con GPU).

---

## Flujo de análisis

```
Imagen de microscopía (.tif / .png / .jpg)
        │
        ▼
  Cellpose (model_base_3B)
        │  label map → N máscaras binarias
        ▼
  Preprocesado CLAHE
        │
        ▼
  CNN de orientación (HU5-v2) — inferencia en batch
        │  fallback: elipse mínima si CNN falla
        ▼
  regionprops (scikit-image)
        │  área, perímetro, excentricidad, solidez, ejes, centroide
        ▼
  AnalysisResult
        ├── count, masks, angles, areas, cell_features
        ├── overlay (imagen con contornos y vectores)
        └── 6 imágenes de panel + figura combinada
        │
        ▼
  Interfaz gráfica
        ├── Visor con 7 tabs de imagen + zoom/pan
        ├── Tab Resumen (estadísticas globales)
        ├── Tab Orientación (CNN vs. feature por célula)
        ├── Tab Detalle (área + orientación)
        └── Tab Features (10 métricas morfológicas)
        │
        ▼
  Exportación
        ├── PDF con tablas formateadas (landscape, 2 páginas)
        └── CSV por tabla (Orientación, Detalle, Features)
```

---

## Módulo CNN de orientación (HU5-v2)

La CNN estima el ángulo de cada fibra en una sola pasada forward (~5 ms/fibra).

**Innovación clave — codificación angular en (sin 2θ, cos 2θ):**

Las fibras musculares tienen simetría de 180°: una fibra a 0° y a 180° son visualmente idénticas. Predecir θ directamente causaría discontinuidades en el loss. La codificación en 2θ mapea [0°, 180°) a una vuelta completa del círculo unitario, eliminando el problema.

**Comparación con la versión anterior (HU5-v1, PPO):**

| Aspecto | HU5-v1 (RL) | HU5-v2 (CNN) |
|---|---|---|
| Paradigma | Control secuencial (PPO) | Regresión directa |
| Tiempo de inferencia | ~2 s (200 pasos) | ~5 ms (1 forward pass) |
| MAE medio | ~26° (distribución bimodal) | < 5° |
| Fallos catastróficos | 28% de casos | No (regresión continua) |
| Datos reales necesarios | 0 | 0 |

---

## Ejecución de tests

```bash
pytest tests/ -v
```

---

## Solución de problemas frecuentes

| Problema | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: cellpose` | Dependencias no instaladas | `pip install -r requirements.txt` |
| `No se encontró modelo` | `Models/model_base_3B` no existe | Ejecutar `python Models/download_models.py` |
| Análisis muy lento | PyTorch en modo CPU | Ver sección [Activar GPU](#activar-gpu-opcional) |
| `CUDA out of memory` | GPU con poca VRAM | La app cae automáticamente a CPU |
| Error al exportar PDF | Sin análisis ejecutado | Ejecutar primero el análisis |
| La UI no abre | Error en app.py | Ejecutar `python main.py` desde la raíz del proyecto |

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `cellpose >= 3.0` | Segmentación celular |
| `torch >= 2.0` | CNN de orientación e inferencia GPU |
| `scikit-image` | regionprops para métricas morfológicas |
| `opencv-python` | Preprocesado, overlay, elipse fallback |
| `ttkbootstrap` | Interfaz gráfica moderna sobre Tkinter |
| `reportlab` | Generación de PDF con tablas |
| `gdown >= 5.1` | Descarga automática del modelo desde Drive |

---

## Licencia

Proyecto académico — Universidad Internacional de La Rioja (UNIR).
