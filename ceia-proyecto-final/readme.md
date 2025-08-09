# <div align="center"><b> Proyecto final CEIA </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Sistema de monitoreo de Rhynchophorus ferrugineus en palmeras de montevideo                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Descrpción**  | Presentación del trabajo final de CEIA que consiste en un sistema de monitoreo de la plaga del picudo rojo |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                                                      |

</div>

## Consinga

En el marco de la Especialización en Inteligencia Artificial de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA), se propone la realización de un proyecto final que permita integrar los conocimientos adquiridos en las diferentes materias del programa.

El proyecto consiste en la implementación de un sistema de monitoreo de la plaga del picudo rojo en palmeras de Montevideo. El sistema incluye componentes de infraestructura, procesamiento de datos, análisis de imágenes, anotaciones de imágenes y visualización de resultados, entre otros. En resumen, pasa por varias de las temáticas que componen proyectos de visión por computadora.

## Resolución

### Gestión de proyecto

Inicialmente, se realizó una planificación del proyecto en la materia de Gestión de Proyectos, en la que se definieron los objetivos, alcance, entregables, riesgos y planificación temporal del proyecto. Se puede acceder a dicha planificación en el siguiente link: https://github.com/brunomaso1/uba-ceia/blob/ceia-gdp/ceia-gdp/charter.pdf

### Taller de trabajo final A

En el taller de trabajo final A se estructuró una memoria sobre el trabajo realizado. Adicionalmente, se escribieron los dos primeros capítulos de dicha memoria.

### Taller de trabajo final B

En el taller de trabajo final B se completó la memoria sobre el trabajo realizado (los capítulos 3, 4 y 5).

### Estructura de la rama

Esta rama del repositorio contiene la implementación y documentación del proyecto. Esta documentación incluye las herramientas utilizadas, que fueron desplegadas en los distintos ambientes utilizados. Se organiza de la siguiente forma: TODO.

### Memoria

Link al documento: TODO

### Gestor de dependencias

Para gestionar las dependencias del proyecto, se utiliza Poetry.

### Comandos útiles

#### Docker

- Detener y eliminar todos los contenedores:
```bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
```

- Prune:
  - Eliminar imágenes no utilizadas: `docker image prune`
  - Eliminar contenedores no utilizados: `docker container prune`
  - Eliminar volúmenes no utilizados: `docker volume prune`
  - Eliminar redes no utilizadas: `docker network prune`
  - Eliminar todo lo anterior: `docker system prune`
  
- Eliminar volumenes:
```bash
docker volume rm $(docker volume ls -q)
```

- Obtener logs de Traefik (parseados a windows - ejecutar dentro de la VM):
```bash
docker logs traefik-entrypoint | sed 's/\x1b\[[0-9;]*m//g' > traefik-entrypoint-20250610.log
```

#### Poetry

- Instalar proyecto:
```bash
poetry install # Producción
poetry install --with dev # Desarrollo, también instala módulos locales de forma editable
```

- Para instalar una nueva dependencia:
```bash
poetry add <nombre-dependencia>
poetry add <nombre-dependencia> --dev # para dependencias de desarrollo
```

- Agregar pytorch con cuda:
```bash
potery source add --priority explicit pytorch_gpu https://download.pytorch.org/whl/cu128
poetry add torch torchvision torchaudio --source pytorch_gpu
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

- Modo dependencias:
```toml
[tool.poetry]
package-mode = false
```

#### Vagrant

- Iniciar Vagrant:
```powershell
$env:ENVIRONMENT="dev"; vagrant up
```

- Detener Vagrant:
```powershell
vagrant halt
```

- Levantar vagrant y aprovisionar:
```powershell
$env:ENVIRONMENT="dev"; vagrant up --provision-with start-services
```

- Solamente Aprovisionar Vagrant:
```powershell
$env:ENVIRONMENT="dev"; vagrant provision --provision-with start-services
```

#### Windows
- Chequear si Hyper-v está habilitado:
```powershell
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V
```

- Deshabilitar Hyper-v [Tutorial](https://learn.microsoft.com/en-us/troubleshoot/windows-client/application-management/virtualization-apps-not-work-with-hyper-v):
```powershell
DISM /Online /Disable-Feature:Microsoft-Hyper-V
Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Hypervisor
bcdedit /set hypervisorlaunchtype off
```

- Habilitar "nested virtualization VirtualBox":
```powershell
VBoxManage modifyvm <YourVirtualMachineName> --nested-hw-virt on
VBoxManage modifyvm "ceia-proyecto-final-develop" --cpu-profile "Intel(R) Core(TM) i7-6700K"
```

- Acortar path de powershell:
```powershell
Function Prompt { "$( ( get-item $pwd ).Name )>" }
```

#### Linux

- Verificar espacio en disco:
```bash
df -h
```

- Verificar uso de memoria:
```bash
free -h
watch -n 5 free -m
```

- Verificar uso de CPU:
```bash
top
htop
ps aux
```

- Verificar si AEX está habilitado:
```bash
cat /proc/cpuinfo # Verifica si AVX está habilitado
grep -m1 -o 'avx[^ ]*' /proc/cpuinfo
grep -E 'avx' /proc/cpuinfo
egrep "svm|vmx" /proc/cpuinfo
```

#### MLFlow

- Limpiar experimentos:
```bash
mlflow gc --tracking-uri "http://localhost:5000" --backend-store-uri "sqlite:////mlruns/mlruns.db" --experiment-id 4
```

#### Fiftyone

- Cargar un dataset en formato YOLO:
```python
# Se carga con el split "val" por defecto. Si se desea otro split, se debe especificar
# con el parámetro `split`. Ej: split="full"
import fiftyone as fo
dataset = fo.Dataset.from_dir(
    dataset_dir="path/to/dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    name="my_yolo_dataset",
    overwrite=True
)
```

- Cargar con varios splits (train, val, test):
```python
import fiftyone as fo
dataset = fo.Dataset(name=dataset_name, overwrite=True)
for split in ["train", "val", "test", "full"]:
    try:
        dataset.add_dir(
            dataset_dir=dataset_path, dataset_type=fo.types.YOLOv5Dataset, split=split, tags=[split]
        )
    except Exception as e:
        LOGGER.warning(f'Advertencia: no se pudo agregar el split "{split}" al dataset. Error: {e}')
        pass
```

- Obtener los esquemas de un dataset:
```python
dataset.get_field_schema()
```

- Inspeccionar estructura de un ejemplo:
```python
sample = dataset.first()
print(sample)
```

- Obtener información sobre un campo:
```python
field_info = dataset.get_field("ground_truth")
print(field_info)
```

- Explorar las propiedades del objeto de detección:
```python
sample_with_detections = dataset.match(F("ground_truth.detections").length() > 0).first()
if sample_with_detections:
    detection = sample_with_detections.ground_truth.detections[0]

    # Common detection fields
    print(f"Label: {detection.label}")
    print(f"Confidence: {getattr(detection, 'confidence', 'N/A')}")
    print(f"Bounding box: {detection.bounding_box}")
    print(f"ID: {getattr(detection, 'id', 'N/A')}")

    # Get all attributes
    print("All detection attributes:")
    for attr in dir(detection):
        if not attr.startswith("_"):
            print(f"  {attr}: {getattr(detection, attr, 'N/A')}")
```