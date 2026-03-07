# Redis Read/Write Test

Aplicación mínima en Python para validar conectividad y operaciones básicas de escritura/lectura sobre Redis.

## Objetivo

- Probar que una instancia Redis (por ejemplo, deployada en Docker) responde correctamente.
- Escribir un valor con `SET` y leerlo con `GET`.

## Estructura

- `app/main.py`: funciones de prueba y ejecución de ejemplo.
- `pyproject.toml`: dependencias del proyecto.

## Requisitos

- Python 3.11+
- Redis accesible desde tu host

## Instalación

Con `uv` (recomendado en este proyecto):

```bash
uv sync
```

Alternativa con pip:

```bash
pip install redis
```

## Configuración

La app toma estos valores desde variables de entorno:

- `REDIS_USER` (opcional, default: vacío)
- `REDIS_PASSWORD` (default: `redispassword`)
- `REDIS_HOST` (default: `localhost`)
- `REDIS_PORT` (default: `6379`)
- `REDIS_DB` (default: `0`)

Ejemplo (PowerShell):

```powershell
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:REDIS_PASSWORD = "redispassword"
$env:REDIS_DB = "0"
```

## Ejecución

Antes de ejecutar el script, activa el entorno virtual:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
python app/main.py
```

Salida esperada (si todo funciona):

```text
Write OK: True
Read value: ok-from-python
```

## Qué hace internamente

En `app/main.py` se ejecuta:

- Escritura con la clave `redis:test:key` y valor `ok-from-python`
- Lectura de esa misma clave

Comportamiento de `SET`:

- Si la clave no existe, Redis la crea.
- Si existe, Redis sobrescribe el valor.

## Sobre claves, DB y JSON

- Redis maneja pares clave-valor por base lógica (`REDIS_DB`).
- En este proyecto se usa DB `0` por defecto.
- La clave `redis:test:key` es un string (los `:` son convención de nombres).
- El valor `ok-from-python` se guarda como string (bytes internamente), no como JSON.

Representación conceptual:

```text
DB 0
	redis:test:key -> "ok-from-python"
```

Si quieres guardar JSON, puedes serializarlo como texto con `json.dumps(...)` y luego parsearlo con `json.loads(...)` al leer.

## Manejo de errores

Las funciones capturan y reportan errores comunes:

- autenticación (`AuthenticationError`)
- conexión (`ConnectionError`)
- timeout (`TimeoutError`)
- errores genéricos de Redis (`RedisError`)

Además, se configuran timeouts de socket para evitar bloqueos largos.
