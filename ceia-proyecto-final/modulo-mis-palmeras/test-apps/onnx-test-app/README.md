# ONNX Test App

<div align="center">
  <strong>Aplicación de diagnóstico para verificar ONNX Runtime, CUDA y capacidad de inferencia en producción</strong>
</div>

## 📋 Descripción

Esta es una aplicación FastAPI diseñada como herramienta de diagnóstico para verificar el funcionamiento de ONNX Runtime con soporte CUDA en entornos de producción con Docker. Incluye un modelo pre-cargado (ResNet50) para realizar inferencias reales.

A diferencia de las otras test-apps, esta herramienta se enfoca en validar la capacidad de inferencia con modelos ONNX, tanto en CPU como en GPU.

### Características

- ✅ Verifica disponibilidad de CUDA Execution Provider en ONNX Runtime
- ✅ Lista todos los Execution Providers disponibles
- ✅ Carga modelo ONNX (ResNet50-v1-7) y verifica providers utilizados
- ✅ Endpoint de inferencia real con upload de imágenes
- ✅ API REST con FastAPI y Uvicorn
- ✅ Imagen Docker optimizada con soporte CUDA
- ✅ Gestión de dependencias con `uv`
- ✅ Preprocesamiento de imágenes con OpenCV

### ⚠️ Notas Importantes

- Esta herramienta **valida ONNX Runtime con CUDA**, no PyTorch directamente
- Incluye modelo ResNet50 pre-cargado para inferencias reales
- Requiere CUDA 11.8 compatible con onnxruntime-gpu 1.19.2
- El modelo se carga al iniciar la aplicación

## 🔧 Requisitos Previos

- Docker instalado
- **Docker con soporte GPU**: Tener instalado `nvidia-docker` o la versión reciente de Docker con soporte NVIDIA
- GPU NVIDIA compatible con CUDA 11.8
- Validar compatibilidad de CUDA: La imagen usa `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`

### Verificar Requisitos

Para verificar que tu sistema tiene acceso a GPU:

```bash
# En Linux/Mac
nvidia-smi

# Con Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

## 🚀 Instalación y Ejecución

### Construcción Local (sin Docker)

```bash
# Clonar o descargar el proyecto
cd onnx-test-app

# Instalar dependencias (requiere uv)
uv sync

# Ejecutar aplicación
uv run fastapi dev app/main.py
```

La aplicación estará disponible en `http://localhost:8888`

### Construcción con Docker

```bash
# Construir la imagen Docker
docker build -t onnx-test-app .
```

### Ejecución del Contenedor

**Con GPU (recomendado para testing)**:
```bash
docker run -d \
  -p 8888:8888 \
  --gpus all \
  --name onnx-test-app \
  onnx-test-app
```

**Sin GPU (solo CPU)**:
```bash
docker run -d \
  -p 8888:8888 \
  --name onnx-test-app \
  onnx-test-app
```

**Con docker compose**:
```bash
docker compose up -d
```

## 📡 API Endpoints

### GET `/` - Health Check

Verifica que la aplicación está corriendo.

**Respuesta**:
```json
{
  "Hello": "ONNX World"
}
```

### GET `/gpu-info` - Información de CUDA

Obtiene información sobre disponibilidad de CUDA Execution Provider.

**Respuesta (con CUDA disponible)**:
```json
{
  "cuda_available": true,
  "available_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

**Respuesta (solo CPU)**:
```json
{
  "cuda_available": false,
  "available_providers": [
    "CPUExecutionProvider"
  ]
}
```

### GET `/gpu-test` - Test Funcional

Ejecuta una operación simple para validar disponibilidad de providers.

**Respuesta (CUDA disponible)**:
```json
{
  "gpu_test": "success",
  "result": [5.0, 7.0, 9.0],
  "providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

**Respuesta (solo CPU)**:
```json
{
  "gpu_test": "failed",
  "reason": "CUDAExecutionProvider not available",
  "providers": ["CPUExecutionProvider"]
}
```

### GET `/load-dummy-model` - Carga de Modelo

Carga el modelo ResNet50 y verifica qué providers se utilizan.

**Respuesta (exitosa con CUDA)**:
```json
{
  "model_loaded": true,
  "providers_used": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

**Respuesta (error)**:
```json
{
  "model_loaded": false,
  "error": "Error message details"
}
```

### POST `/predict-gpu` - Inferencia con Imagen (GPU)

Realiza inferencia real con el modelo ResNet50 sobre una imagen subida utilizando la GPU (CUDA).

**Request**:
```bash
curl -X 'POST' \
  'http://localhost:8888/predict-gpu' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test-image.jpeg;type=image/jpeg'
```

**Respuesta (exitosa)**:
```json
{
  "success": true,
  "prediction_shape": [1, 1000],
  "providers_used": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

**Respuesta (error)**:
```json
{
  "success": false,
  "error": "Error message details"
}
```

### POST `/predict-cpu` - Inferencia con Imagen (CPU)

Realiza inferencia real con el modelo ResNet50 sobre una imagen subida utilizando solo la CPU.

**Request**:
```bash
curl -X 'POST' \
  'http://localhost:8888/predict-cpu' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test-image.jpeg;type=image/jpeg'
```

**Respuesta (exitosa)**:
```json
{
  "success": true,
  "prediction_shape": [1, 1000],
  "providers_used": [
    "CPUExecutionProvider"
  ]
}
```

**Respuesta (error)**:
```json
{
  "success": false,
  "error": "Error message details"
}
```

## 🔍 Testing de la API

### Con cURL

```bash
# Health check
curl http://localhost:8888/

# Información de CUDA
curl http://localhost:8888/gpu-info

# Test funcional
curl http://localhost:8888/gpu-test

# Cargar modelo
curl http://localhost:8888/load-dummy-model

# Inferencia con imagen
curl -X POST \
  -F "file=@path/to/image.jpg" \
  http://localhost:8888/predict
```

### Con Python

```python
import requests

# Obtener información de providers
response = requests.get("http://localhost:8888/gpu-info")
print("GPU Info:", response.json())

# Test funcional
response = requests.get("http://localhost:8888/gpu-test")
print("GPU Test:", response.json())

# Cargar modelo
response = requests.get("http://localhost:8888/load-dummy-model")
print("Model Load:", response.json())

# Inferencia con imagen
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8888/predict", files=files)
    print("Prediction:", response.json())
```

## 📦 Dependencias

- **FastAPI >= 0.129.1**: Framework web ligero para APIs REST
- **onnxruntime-gpu == 1.19.2**: Runtime de ONNX con soporte CUDA 11.8
- **NumPy >= 2.4.2**: Librería numérica para operaciones con arrays
- **OpenCV (headless) >= 4.13.0**: Procesamiento de imágenes sin GUI
- **python-multipart >= 0.0.22**: Para manejar upload de archivos
- **CUDA 11.8.0**: Runtime de CUDA (base de imagen Docker)
- **cuDNN 8**: Librería de redes neuronales acelerada por GPU

Ver `pyproject.toml` para detalles completos.

## 🧠 Modelo Incluido

### ResNet50-v1-7

El proyecto incluye el modelo **ResNet50-v1-7** en formato ONNX:

- **Tipo**: Red Neuronal Convolucional para clasificación de imágenes
- **Input**: Imagen RGB de 224x224 pixels (formato CHW)
- **Output**: Vector de 1000 clases (ImageNet)
- **Tamaño**: ~97 MB
- **Ubicación**: `app/resources/resnet50-v1-7.onnx`

### Preprocesamiento de Imágenes

El endpoint `/predict` aplica el siguiente preprocesamiento:

1. Redimensionar a 224x224
2. Normalizar valores a [0, 1] (división por 255)
3. Transponer de HWC a CHW
4. Agregar dimensión de batch

## 🐳 Configuración de Docker

### Versión de CUDA

El Dockerfile especifica CUDA 11.8.0 que es **compatible con onnxruntime-gpu 1.19.2**:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

#### Compatibilidad ONNX Runtime-CUDA

| ONNX Runtime | CUDA Requerido |
|--------------|----------------|
| 1.19.2 (actual) | 11.8 |
| 1.18.x | 11.8, 12.x |
| 1.17.x | 11.8, 12.x |
| 1.16.x | 11.8 |

#### Cambiar Versión de CUDA

Si necesitas una versión diferente:

1. Verificar compatibilidad en https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
2. Actualizar Dockerfile con versión compatible
3. Actualizar `pyproject.toml` con versión de onnxruntime-gpu compatible
4. Reconstruir la imagen

**Ejemplo para CUDA 12.x**:
```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
```

Y en `pyproject.toml`:
```toml
dependencies = [
    "onnxruntime-gpu==1.18.0", # Compatible con CUDA 12.x
    # ... resto de dependencias
]
```

## 🔧 Configuración y Personalización

### Variables de Entorno

| Variable | Valor Actual | Descripción |
|----------|--------------|-------------|
| `PORT` | 8888 | Puerto en el que escucha la aplicación |
| `PYTHONPATH` | /app | Ruta de módulos Python en Docker |

Para cambiar el puerto:
```bash
docker run -e PORT=9000 -p 9000:9000 onnx-test-app
```

### Cambiar Modelo ONNX

Para usar un modelo diferente:

1. Reemplazar `app/resources/resnet50-v1-7.onnx` con tu modelo
2. Actualizar `MODEL_PATH` en `app/main.py`
3. Ajustar preprocesamiento según el modelo
4. Reconstruir la imagen

```python
MODEL_PATH = Path("resources/tu-modelo.onnx")
```

### Agregar Endpoints Personalizados

Modificar `app/main.py` para agregar nuevas rutas:

```python
@app.get("/model-info")
def model_info():
    try:
        return {
            "input_name": session.get_inputs()[0].name,
            "input_shape": session.get_inputs()[0].shape,
            "output_name": session.get_outputs()[0].name,
            "output_shape": session.get_outputs()[0].shape,
        }
    except Exception as e:
        return {"error": str(e)}
```

## 🚨 Solución de Problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| `CUDAExecutionProvider not available` | CUDA no instalado o incompatible | Verificar versión CUDA con `nvidia-smi` |
| `Model not found` | Archivo .onnx no existe | Verificar que existe `app/resources/resnet50-v1-7.onnx` |
| `ONNXRuntimeError` | Modelo corrupto o incompatible | Re-descargar modelo desde ONNX Model Zoo |
| `Shape mismatch` | Preprocesamiento incorrecto | Verificar dimensiones de entrada del modelo |
| `CUDA out of memory` | Imagen demasiado grande | Reducir tamaño de batch o imagen |
| `Import error: cv2` | OpenCV no instalado | Ejecutar `uv sync` |
| Puerto ocupado | Otro servicio usa el puerto | Cambiar puerto: `-e PORT=9000 -p 9000:9000` |

### Debug Avanzado

Para obtener información detallada de ONNX Runtime:

```python
import onnxruntime as ort

print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")
print(f"Device: {ort.get_device()}")

# Información de sesión
session_options = ort.SessionOptions()
session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
print(f"Providers Used: {session.get_providers()}")
```

## 📊 Casos de Uso

- ✅ Validar infraestructura ONNX Runtime con GPU en producción
- ✅ Testing de inferencia antes de desplegar modelos en producción
- ✅ Verificar compatibilidad CUDA con ONNX Runtime
- ✅ Benchmark de rendimiento CPU vs GPU
- ✅ Validación de pipeline de inferencia completo
- ✅ Testing en CI/CD para modelos ONNX
- ✅ Base para APIs de inferencia más complejas

## 🔗 Referencia Rápida

- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ResNet50 Model](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 📝 Licencia

Ver archivo LICENSE en el repositorio raíz.
