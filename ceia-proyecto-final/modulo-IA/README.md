# Comandos

- Sincronizar dependencias desarrollo:
```bash
uv sync --all-extras
```

## Gestión de Dependencias con `uv`

### Conceptos Clave

> **⚠️ Limitación de `uv`**: `uv` (igual que Poetry, PDM, etc.) **no propaga grupos de dependencias**. El grupo `dev` de un paquete remoto no se expande al instalarlo desde otro proyecto. Por eso diferenciamos entre "grupo dev" y "extra development" (dependencias opcionales).

### Instalación de `modulo-utilidades`

#### 📌 Desarrollo Local (Recomendado)

**Instalar con el grupo de desarrollo en modo editable:**
```bash
uv add "modulo-utilidades[development] @ ..\modulo-utilidades\" --optional development --editable
```

✅ **Ventajas:**
- Toma automáticamente el `.env` de la carpeta del módulo
- Permite editar el código en tiempo real
- Instala todas las dependencias de desarrollo

#### 🏭 Producción

**Paso 1: Crear la wheel en el módulo-utilidades**
```bash
cd ..\modulo-utilidades
uv build
```

**Paso 2: Instalar la wheel en modulo-IA**
```bash
uv add "modulo-utilidades @ .\dist\modulo_utilidades-0.2.0-py3-none-any.whl"
```

✅ **Ventajas:**
- Las variables de entorno se inyectan via Docker Compose (sin necesidad del `.env` local)
- El código de `modulo-utilidades` no se puede editar accidentalmente
- Utiliza el core estable de `modulo-utilidades` sin dependencias de desarrollo

### Configuración en `pyproject.toml`

Se utiliza `[project.optional-dependencies]` (no `[dependency-groups]`) para definir dependencias opcionales:

```toml
[project.optional-dependencies]
development = [
    # lista de dependencias...
]
```

**Para agregar nuevas dependencias de desarrollo:**
```bash
uv add <dependencia> --optional development
```

**Ejemplo:**
```bash
uv add "modulo-utilidades[development] @ ..\modulo-utilidades\" --optional development --editable
```

En `pyproject.toml`:
```toml
modulo-utilidades = { path = "dist/modulo_utilidades-0.2.0-py3-none-any.whl" }
```

### Resumen Rápido

| Contexto | Comando |
|----------|---------|
| **Desarrollo Local** | `uv add "modulo-utilidades[development] @ ..\modulo-utilidades\" --optional development --editable` |
| **Sincronizar Deps** | `uv sync --all-extras` |
| **Producción - Paso 1** | `uv build` (en modulo-utilidades) |
| **Producción - Paso 2** | `uv add "modulo-utilidades @ .\dist\modulo_utilidades-0.2.0-py3-none-any.whl"` |