# <div align="center"><b> Proyecto final CEIA </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Modulo de mini-apps                                                   |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Aplicaciones y scripts encargados de realizar tareas del procesamiento de datos |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

Este módulo se encarga de implementar las mini-apps que se utilizarán para el procesamiento de datos. Estas mini-apps son aplicaciones y scripts que realizan tareas específicas relacionadas con el procesamiento de datos, como la recolección, limpieza, etiquetado, transformación y análisis de datos. 

## Resolución

### Gestor de dependencias

Para gestionar las dependencias del proyecto, se utiliza Poetry.

#### Comandos útiles poetry

- Instalar proyecto:
```bash
poetry install
```

- Para instalar una nueva dependencia:
```bash
poetry add <nombre-dependencia>
poetry add <nombre-dependencia> --dev # para dependencias de desarrollo
```

- Listar las dependencias instaladas:
```bash
poetry show
```

- Listar intérpretes de python disponibles:
```bash
poetry python list
```

- Configurar entorno virtual para crearse en el directorio del proyecto:
```bash
poetry config virtualenvs.in-project true
```

- Activar el entorno virtual:
```powershell
Invoke-Expression (poetry env activate)
```

- Desactivar el entorno virtual:
```bash
deactivate
```

- Especificar el intérprete de python a utilizar:
```bash
poetry env use <ruta-al-intérprete-python>
```

### Aplicaciones

Dentro del módulo de mini-apps se encuentran las siguientes aplicaciones:
- `webscrapping-SIG`: Aplicación encargada de realizar el webscraping de los datos del SIG.

#### webscrapping-SIG