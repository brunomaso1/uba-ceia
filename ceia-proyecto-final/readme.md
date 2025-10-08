# <div align="center"><b> Proyecto final CEIA </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Sistema de monitoreo de Rhynchophorus ferrugineus en palmeras de montevideo                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Descrpción**  | Presentación del trabajo final de CEIA que consiste en un sistema de monitoreo de la plaga del picudo rojo |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                                                      |

</div>

## Consigna

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

### Entornos

El proyecto se desarrolla en dos entornos: desarrollo y producción. Cada entorno tiene su propia configuración y dependencias.

#### Producción

- Ubuntu 24.04
- Docker + Docker Compose
- SSH
- Samba
- VSCode Remote - SSH

### Colores del proyecto

```css
html {
	--mat-sys-background: light-dark(#f9faf3, #121410);
	--mat-sys-error: light-dark(#ba1a1a, #ffb4ab);
	--mat-sys-error-container: light-dark(#ffdad6, #93000a);
	--mat-sys-inverse-on-surface: light-dark(#f1f1eb, #2f312d);
	--mat-sys-inverse-primary: light-dark(#02e600, #026e00);
	--mat-sys-inverse-surface: light-dark(#2f312d, #e2e3dc);
	--mat-sys-on-background: light-dark(#1a1c18, #e2e3dc);
	--mat-sys-on-error: light-dark(#ffffff, #690005);
	--mat-sys-on-error-container: light-dark(#93000a, #ffdad6);
	--mat-sys-on-primary: light-dark(#ffffff, #013a00);
	--mat-sys-on-primary-container: light-dark(#015300, #77ff61);
	--mat-sys-on-primary-fixed: light-dark(#002200, #002200);
	--mat-sys-on-primary-fixed-variant: light-dark(#015300, #015300);
	--mat-sys-on-secondary: light-dark(#ffffff, #263422);
	--mat-sys-on-secondary-container: light-dark(#3c4b37, #d7e8cd);
	--mat-sys-on-secondary-fixed: light-dark(#121f0e, #121f0e);
	--mat-sys-on-secondary-fixed-variant: light-dark(#3c4b37, #3c4b37);
	--mat-sys-on-surface: light-dark(#1a1c18, #e2e3dc);
	--mat-sys-on-surface-variant: light-dark(#43483f, #dfe4d7);
	--mat-sys-on-tertiary: light-dark(#ffffff, #013a00);
	--mat-sys-on-tertiary-container: light-dark(#015300, #77ff61);
	--mat-sys-on-tertiary-fixed: light-dark(#002200, #002200);
	--mat-sys-on-tertiary-fixed-variant: light-dark(#015300, #015300);
	--mat-sys-outline: light-dark(#73796e, #8d9387);
	--mat-sys-outline-variant: light-dark(#c3c8bc, #43483f);
	--mat-sys-primary: light-dark(#026e00, #02e600);
	--mat-sys-primary-container: light-dark(#77ff61, #015300);
	--mat-sys-primary-fixed: light-dark(#77ff61, #77ff61);
	--mat-sys-primary-fixed-dim: light-dark(#02e600, #02e600);
	--mat-sys-scrim: light-dark(#000000, #000000);
	--mat-sys-secondary: light-dark(#54634d, #bbcbb2);
	--mat-sys-secondary-container: light-dark(#d7e8cd, #3c4b37);
	--mat-sys-secondary-fixed: light-dark(#d7e8cd, #d7e8cd);
	--mat-sys-secondary-fixed-dim: light-dark(#bbcbb2, #bbcbb2);
	--mat-sys-shadow: light-dark(#000000, #000000);
	--mat-sys-surface: light-dark(#f9faf3, #121410);
	--mat-sys-surface-bright: light-dark(#f9faf3, #383a35);
	--mat-sys-surface-container: light-dark(#eeeee7, #1e201c);
	--mat-sys-surface-container-high: light-dark(#e8e9e1, #282b26);
	--mat-sys-surface-container-highest: light-dark(#e2e3dc, #333531);
	--mat-sys-surface-container-low: light-dark(#f3f4ed, #1a1c18);
	--mat-sys-surface-container-lowest: light-dark(#ffffff, #0c0f0b);
	--mat-sys-surface-dim: light-dark(#dadbd3, #121410);
	--mat-sys-surface-tint: light-dark(#026e00, #02e600);
	--mat-sys-surface-variant: light-dark(#dfe4d7, #43483f);
	--mat-sys-tertiary: light-dark(#026e00, #02e600);
	--mat-sys-tertiary-container: light-dark(#77ff61, #015300);
	--mat-sys-tertiary-fixed: light-dark(#77ff61, #77ff61);
	--mat-sys-tertiary-fixed-dim: light-dark(#02e600, #02e600);
	--mat-sys-neutral-variant20: #2c3229;
	--mat-sys-neutral10: #1a1c18;
}
```

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

- Ingresar dentro de un contenedor:
```bash
docker exec -it <container_id> sh
```

- Ver las redes:
```bash
docker network ls
docker network inspect <network_id> # Te dice cuales los contenedores conectados
```

##### Comandos debugging

-  Ver todos los contenedores y sus redes
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Networks}}"
```

- Probar conectividad desde el contenedor traefik-entrypoint
```bash
docker exec traefik-entrypoint nslookup traefik
docker exec traefik-entrypoint wget -qO- http://traefik:8080 || echo "Connection failed"
```

- Más pruebas de conectividad:
```bash
docker exec traefik-entrypoint ping traefik
docker exec traefik-entrypoint telnet traefik 8080
```

- Ver logs del Traefik de CVAT con más detalle (modo seguimiento):
```bash
docker logs traefik -f
```

- Verificar nombres contenedor:
```bash
docker ps --filter "name=traefik"
```

- Ver el estado del contenedor Traefik de CVAT:
```bash
docker ps | grep traefik
```

- Ver los puertos que está exponiendo realmente:
```bash
docker port traefik
```

- Ver logs más detallados del Traefik de CVAT:
```bash
docker logs traefik --tail 50
```

- Verificar la configuración interna del contenedor:
```bash
docker exec traefik ps aux
```

- Ver qué puertos está escuchando dentro del contenedor (importante para ver si está escuchando en el puerto correcto):
```bash
docker exec traefik netstat -tlnp 2>/dev/null || docker exec traefik ss -tlnp
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

- Verificar los discos detectados:
```bash
lsblk
fdisk -l
```

- Revisar el grupo de volúmenes lógicos (LVM):
```bash
sudo vgs
```

- Agregar disco al grupo de volúmenes lógicos (LVM):
```bash
# Solo si el vgs mostró 0 en VFree
sudo pvcreate /dev/sdb # Inicializar el disco como un Physical Volume
sudo vgextend ubuntu-vg /dev/sdb # Agregar al grupo de volúmenes lógicos
```

- Extender el volúmen lógico raiz:
```bash
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
```

- Redimensionar el sistema de archivos:
```bash
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv # Para -> ext4
sudo xfs_growfs /dev/ubuntu-vg/ubuntu-lv # Para -> xfs
```

- Verificar uso de memoria (RAM):
```bash
free -h
watch -n 5 free -m
```

- Verificar uso de CPU:
```bash
top -b -n 1 |grep ^Cpu
ps -eo pcpu,pid,user,args | sort -r -k1 | less # Porcentaje de uso de CPU
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

- Listar experimentos:
```python
experiments = mlflow.search_experiments(view_type="ALL")
for exp in experiments:
    print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")
```

- Limpiar experimentos:
```bash
docker exec -it <nombre_contenedor_mlflow> mlflow gc \
	--tracking-uri "http://localhost:5000" \
    --backend-store-uri sqlite:////mlruns/mlruns.db \
    --experiment-ids 4

# Ejemplo:
docker exec -it 1e2a2d3b145f mlflow gc --tracking-uri "http://localhost:5000" --backend-store-uri sqlite:////mlruns/mlruns.db --experiment-ids 6
```

- Limpiar runs:
```bash
docker exec -it 1e2a2d3b145f mlflow gc --tracking-uri "http://localhost:5000" --backend-store-uri sqlite:////mlruns/mlruns.db
```

- Restaurar experimento borrado:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

# Restaurar experimento con ID 4
client.restore_experiment("2")
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