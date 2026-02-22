# PyTorch CUDA Test App

<div align="center">
  <strong>Aplicación de diagnóstico para verificar PyTorch, CUDA y compatibilidad de GPU en producción</strong>
</div>

## 📋 Descripción

Esta es una aplicación FastAPI diseñada como herramienta de diagnóstico exhaustiva para verificar la disponibilidad, compatibilidad y funcionamiento de GPU con PyTorch y CUDA en entornos de producción con Docker.

A diferencia de `simple-gpu-test-app`, esta herramienta ejecuta pruebas funcionales reales en la GPU, verifica la arquitectura y valida compatibilidad con PyTorch.

### Características

- ✅ Verifica disponibilidad de GPUs NVIDIA con PyTorch
- ✅ Ejecuta operaciones reales en GPU para confirmar funcionamiento
- ✅ Obtiene información de arquitectura y capacidad de cómputo (Compute Capability)
- ✅ Valida compatibilidad entre GPU y versiones de PyTorch
- ✅ Lista arquitecturas de GPU soportadas por PyTorch
- ✅ API REST con FastAPI y Uvicorn
- ✅ Imagen Docker optimizada con soporte CUDA
- ✅ Gestión de dependencias con `uv`
- ✅ Diseño minimalista pero funcional

### ⚠️ Notas Importantes

- Esta herramienta **valida PyTorch y CUDA compatible**, no solo drivers NVIDIA
- Ejecuta operaciones de tensores en GPU para confirmar funcionamiento real
- Verifica la arquitectura de GPU y su compatibilidad con PyTorch compilado
- Requiere más recursos que `simple-gpu-test-app` debido a PyTorch

## 🔧 Requisitos Previos

- Docker instalado
- **Docker con soporte GPU**: Tener instalado `nvidia-docker` o la versión reciente de Docker con soporte NVIDIA
- GPU NVIDIA compatible con PyTorch (típicamente arquitecturas Kepler o más nuevas)
- Validar compatibilidad de CUDA: La imagen usa `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` por defecto
- PyTorch 2.1.2 requiere CUDA >= 11.7

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
cd pytorch-cuda-test-app

# Instalar dependencias (requiere uv)
uv sync

# Ejecutar aplicación
uv run fastapi dev app/main.py
```

La aplicación estará disponible en `http://localhost:8888`

### Construcción con Docker

```bash
# Construir la imagen Docker
docker build -t pytorch-cuda-test-app .
```

### Ejecución del Contenedor

**Con GPU (recomendado para testing)**:
```bash
docker run -d \
  -p 8888:8888 \
  --gpus all \
  --name pytorch-cuda-test-app \
  pytorch-cuda-test-app
```

**Sin GPU (para testing sin hardware)**:
```bash
docker run -d \
  -p 8888:8888 \
  --name pytorch-cuda-test-app \
  pytorch-cuda-test-app
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
  "Hello": "World"
}
```

### GET `/gpu-info` - Información de GPUs

Obtiene lista de GPUs disponibles detectadas por PyTorch.

**Respuesta (con GPU disponible)**:
```json
{
  "gpu_available": true,
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA A100-SXM4-80GB"
    }
  ]
}
```

**Respuesta (sin GPU disponible)**:
```json
{
  "gpu_available": false,
  "gpus": []
}
```

### GET `/gpu-test` - Test Funcional de GPU

Ejecuta una operación simple de tensores en la GPU para validar funcionamiento real.

**Respuesta (GPU funcionando)**:
```json
{
  "gpu_test": "success",
  "result": [5.0, 7.0, 9.0]
}
```

**Respuesta (sin GPU)**:
```json
{
  "gpu_test": "failed",
  "reason": "No GPU available"
}
```

### GET `/gpu-architecture` - Información de Arquitectura

Obtiene información sobre la arquitectura y capacidad de cómputo de la GPU.

**Respuesta (GPU disponible)**:
```json
{
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "compute_capability": "8.0",
  "sm": "sm_80"
}
```

**Respuesta (sin GPU)**:
```json
{
  "gpu_available": false
}
```

### GET `/gpu-architecture-list` - Arquitecturas Soportadas

Lista todas las arquitecturas de GPU soportadas por la versión instalada de PyTorch.

**Respuesta**:
```json
{
  "supported_gpu_architectures": [
    "sm_50",
    "sm_60",
    "sm_61",
    "sm_70",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90"
  ]
}
```

### GET `/gpu-compatibility` - Validación de Compatibilidad

Verifica si la GPU actual es soportada por la versión de PyTorch instalada.

**Respuesta (compatible)**:
```json
{
  "gpu_available": true,
  "gpu_sm": "sm_80",
  "pytorch_supported_architectures": [
    "sm_50",
    "sm_60",
    "sm_61",
    "sm_70",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90"
  ],
  "is_supported": true
}
```

**Respuesta (incompatible)**:
```json
{
  "gpu_available": true,
  "gpu_sm": "sm_30",
  "pytorch_supported_architectures": ["sm_50", "sm_60", ...],
  "is_supported": false
}
```

**Respuesta (sin GPU)**:
```json
{
  "gpu_available": false
}
```

## 🔍 Testing de la API

### Con cURL

```bash
# Health check
curl http://localhost:8888/

# Información de GPUs
curl http://localhost:8888/gpu-info

# Test funcional
curl http://localhost:8888/gpu-test

# Arquitectura
curl http://localhost:8888/gpu-architecture

# Compatibilidad
curl http://localhost:8888/gpu-compatibility

# Arquitecturas soportadas
curl http://localhost:8888/gpu-architecture-list
```

### Con Python

```python
import requests

# Obtener información de GPU
response = requests.get("http://localhost:8888/gpu-info")
print("GPU Info:", response.json())

# Ejecutar test en GPU
response = requests.get("http://localhost:8888/gpu-test")
print("GPU Test:", response.json())

# Verificar compatible
response = requests.get("http://localhost:8888/gpu-compatibility")
print("GPU Compatibility:", response.json())
```

## 📦 Dependencias

- **FastAPI >= 0.122.0**: Framework web ligero para APIs REST
- **PyTorch == 2.1.2**: Framework de deep learning con soporte CUDA
- **NumPy >= 2.3.5**: Librería numérica utilizada por PyTorch
- **CUDA 11.8.0**: Runtime de CUDA (base de imagen Docker)
- **cuDNN 8**: Librería de redes neuronales acelerada por GPU

Ver `pyproject.toml` para detalles completos.

## 🐳 Configuración de Docker

### Versión de CUDA

El Dockerfile especifica una versión de CUDA que **debe ser compatible** con PyTorch 2.1.2:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

#### Compatibilidad PyTorch-CUDA

| PyTorch Version | CUDA Soportados |
|-----------------|-----------------|
| 2.1.2 (actual) | 11.8, 12.1 |
| 2.0.x | 11.7, 11.8, 12.1 |
| 1.13.x | 11.6, 11.7, 11.8 |

#### Cambiar Versión de CUDA

Si necesitas una versión diferente:

1. Verificar compatibilidad en https://pytorch.org/get-started/locally/
2. Actualizar Dockerfile con versión compatible
3. Actualizar `pyproject.toml` si es necesario (ajustar index de PyTorch)
4. Reconstruir la imagen

**Ejemplo para CUDA 12.1**:
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

Y en `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
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
docker run -e PORT=9000 -p 9000:9000 pytorch-cuda-test-app
```

### Agregar Endpoints Personalizados

Modificar `app/main.py` para agregar nuevas rutas de testing:

```python
@app.get("/gpu-memory")
def gpu_memory():
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {
                "allocated_mb": allocated / 1024 / 1024,
                "reserved_mb": reserved / 1024 / 1024
            }
        return {"gpu_available": False}
    except Exception as e:
        return {"error": str(e)}
```

## 🚨 Solución de Problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| `RuntimeError: CUDA out of memory` | Memoria GPU insuficiente | Usar GPU con más memoria o reducir operaciones |
| `torch.cuda.is_available() = False` | CUDA no inicializado o drivers no instalados | Verificar `nvidia-smi` y reinstalar drivers |
| `AssertionError: Torch not compiled with CUDA enabled` | PyTorch compilado sin soporte CUDA | Reinstalar PyTorch desde PyTorch.org (no pip) |
| `is_supported = false` | GPU con arquitectura antigua | Actualizar GPU o usar GPU compatible |
| `CudaRuntimeError: unspecified launch failure` | Operación incompatible con GPU | Verificar arquitectura and PyTorch version |
| `ImportError: cannot import torch` | PyTorch no instalado | Ejecutar `uv sync` |
| Puerto ocupado | Otro servicio usa el puerto | Cambiar puerto: `-e PORT=9000 -p 9000:9000` |

### Debug Avanzado

Para obtener información detallada de CUDA en PyTorch:

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
print(f"Supported Architectures: {torch.cuda.get_arch_list()}")
```

## 📊 Casos de Uso

- ✅ Validar infraestructura GPU con PyTorch en producción
- ✅ Verificar compatibilidad de GPU con versión específica de PyTorch
- ✅ Testing en CI/CD para pipelines que usan PyTorch
- ✅ Diagnóstico de problemas CUDA y compatibilidad
- ✅ Validación antes de desplegar modelos ML complejos
- ✅ Monitoreo de salud de GPU en cluster
- ✅ Base para aplicaciones ML más complejas

## 🔗 Referencia Rápida

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- [GPU Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)

## 📝 Licencia

Ver archivo LICENSE en el repositorio raíz.
