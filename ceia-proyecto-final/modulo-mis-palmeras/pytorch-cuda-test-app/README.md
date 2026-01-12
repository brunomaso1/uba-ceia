# Test app para verificar disponibilidad de GPU en entorno Docker con FastAPI y Uvicorn.

## Descripción

Esta aplicación FastAPI está diseñada para verificar la disponibilidad de una GPU en el entorno donde se ejecuta. Utiliza Uvicorn como servidor ASGI para manejar las solicitudes HTTP.
Es simple en el sentido que verifica si los si los drivers están correctamente instalados y si una GPU está accesible, pero no verifica la versión de PyTorch o CUDA.

## Comandos

- Construir la imagen Docker:
  ```bash
  docker build -t pytorch-cuda-test-app .
  ```
- Ejecutar el contenedor Docker:
  ```bash
  docker run -p 8888:8888 --gpus all --name pytorch-cuda-test-app-container pytorch-cuda-test-app
  ```