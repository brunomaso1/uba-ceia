# Simple GPU Test App

<div align="center">
  <strong>Aplicación minimalista para verificar la disponibilidad y funcionamiento de GPU en producción</strong>
</div>

## 📋 Descripción

Esta es una aplicación FastAPI simple diseñada como herramienta de diagnóstico para verificar la disponibilidad y estado de una GPU en entornos de producción con Docker. 

### Características

- ✅ Verifica disponibilidad de GPUs NVIDIA
- ✅ Recopila información detallada del hardware (nombre, memoria total/usada/libre)
- ✅ API REST con FastAPI y Uvicorn
- ✅ Imagen Docker optimizada con soporte CUDA
- ✅ Gestión de dependencias con `uv`
- ✅ Diseño minimalista y ligero

### ⚠️ Nota Importante

Esta herramienta **únicamente verifica** que los drivers NVIDIA estén correctamente instalados y que una GPU sea accesible. No valida versiones específicas de PyTorch o CUDA, ni ejecuta pruebas de rendimiento.

## 🔧 Requisitos Previos

- Docker instalado
- **Docker con soporte GPU**: Tener instalado `nvidia-docker` o la versión reciente de Docker con soporte NVIDIA
- GPU NVIDIA disponible en el host (con drivers actualizados)
- Validar compatibilidad de CUDA: La imagen usa `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`

### Verificar Requisitos

Para verificar que tu sistema tiene acceso a GPU:

```bash
# En Linux/Mac
nvidia-smi

# Con Docker
docker run --rm --gpus all nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 nvidia-smi
```

## 🚀 Instalación y Ejecución

### Construcción Local (sin Docker)

```bash
# Clonar o descargar el proyecto
cd simple-gpu-test-app

# Instalar dependencias (requiere uv)
uv sync

# Ejecutar aplicación
uv run fastapi dev app/main.py
```

La aplicación estará disponible en `http://localhost:8000`

### Construcción con Docker

```bash
# Construir la imagen Docker
docker build -t simple-gpu-test-app .
```

### Ejecución del Contenedor

**Con GPU (recomendado para testing)**:
```bash
docker run -d \
  -p 80:80 \
  --gpus all \
  --name simple-gpu-test-app \
  simple-gpu-test-app
```

**Sin GPU (para testing sin hardware)**:
```bash
docker run -d \
  -p 80:80 \
  --name simple-gpu-test-app \
  simple-gpu-test-app
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

### GET `/gpu` - Información de GPU

Obtiene información detallada de todas las GPUs disponibles.

**Respuesta (con GPU disponible)**:
```json
{
  "gpu_available": true,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24576,
      "memory_used_mb": 512,
      "memory_free_mb": 24064
    }
  ]
}
```

**Respuesta (sin GPU disponible)**:
```json
{
  "gpu_available": false,
  "error": "NVML Error Codes: GPU_NOT_FOUND : No NVIDIA GPU detected"
}
```

## 🔍 Testing de la API

### Con cURL

```bash
# Health check
curl http://localhost/

# Información de GPU
curl http://localhost/gpu
```

### Con Python

```python
import requests

response = requests.get("http://localhost/gpu")
print(response.json())
```

## 📦 Dependencias

- **FastAPI >= 0.122.0**: Framework web ligero para APIs REST
- **nvidia-ml-py >= 13.580.82**: Binding de Python para NVIDIA Management Library (NVML)

Ver `pyproject.toml` para detalles completos.

## 🐳 Configuración de Docker

### Versión de CUDA

El Dockerfile especifica una versión de CUDA que **debe ser compatible** con la GPU del servidor:

```dockerfile
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
```

#### Cambiar Versión de CUDA

Si necesitas una versión diferente:

1. Verificar versión de CUDA compatible con tu GPU: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
2. Reemplazar en el Dockerfile
3. Reconstruir la imagen

Imagen base recomendada:
- `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` (actual)
- `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04` (alternativa)

## 🔧 Configuración y Personalización

### Variables de Entorno

| Variable | Valor Actual | Descripción |
|----------|--------------|-------------|
| `PORT` | 80 | Puerto en el que escucha la aplicación |

Para cambiar el puerto, modificar en el Dockerfile o pasar como variable:

```bash
docker run -e PORT=8080 -p 8080:8080 ...
```

### Agregar Endpoints Personalizados

Modificar `app/main.py` para agregar nuevas rutas de testing:

```python
@app.get("/cuda-version")
def cuda_version():
    # Tu lógica aquí
    pass
```

## 🚨 Solución de Problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| `NVIDIA_SYCL: Did not detect GPU` | Drivers NVIDIA no instalados | Instalar drivers NVIDIA en el host |
| `NVMLError: GPU not found` | Container sin acceso a GPU | Verificar flag `--gpus all` en docker run |
| `RuntimeError: CUDA out of memory` | Memoria GPU insuficiente | Reducir tamaño de batch o usar GPU más grande |
| `No such file or directory: /python` | Problema con instalación de uv | Reconstruir imagen Docker |
| Puerto ocupado | Otro servicio usa el puerto | Cambiar puerto: `-p 8080:80` |

## 📊 Casos de Uso

- ✅ Validar infraestructura GPU en producción
- ✅ Monitoreo básico de disponibilidad de GPU
- ✅ Testing en CI/CD para pipelines que usan GPU
- ✅ Diagnóstico rápido de problemas de hardware
- ✅ Punto de entrada para aplicaciones más complejas

## 📝 Licencia

Ver archivo LICENSE en el repositorio raíz.