# Comandos

## UV

- Sincronizar dependencias (desarrollo):
```bash
# Sincronizar dependencias de desarrollo, esto implica que se instalarán dependencias de la sección [dependency-groups] de dev.
# En desarrollo se debe ejecutar este comando que permite también editar los módulos locales instalados.
uv sync --all-extras
```

- Sincronizar dependencias (producción):`
```bash
# Sincronizar dependencias para producción, esto implica que NO se instalarán dependencias de la sección [dependency-groups] de dev,
# sino que solo las necesarias para producción (dentro de la sección [dependencies]).
# Existen wheels precompilados que se toman en cuenta al ejecutar este comando.
uv sync --frozen --no-cache
```

- Ejecutar fastapi (en modo desarrollo) con uv:
```bash
uv run fastapi dev app/main.py
```

- O con uvicorn (en modo desarrollo):
```bash
uv run uvicorn app.main:app --reload
```

- Ejecutar fastapi (en modo producción) con fastapi:
```bash
uv run fastapi run app/main.py
```

- O con uvicorn (en modo producción) con uv:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Docker

- Run:
```bash
docker run --env-file .\.env --gpus all -p 8000:8000 --name fastapi-backend fastapi-backend
```