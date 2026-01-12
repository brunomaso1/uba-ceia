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

Para gestionar las dependencias del proyecto, se utiliza UV.

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

- Eliminar volumenes:
```bash
docker volume rm $(docker volume ls -q)
```

- Eliminar imágenes:
```bash
docker rmi $(docker images -q)
```

- Prune:
  - Eliminar imágenes no utilizadas: `docker image prune`
  - Eliminar contenedores no utilizados: `docker container prune`
  - Eliminar volúmenes no utilizados: `docker volume prune`
  - Eliminar redes no utilizadas: `docker network prune`
  - Eliminar todo lo anterior: `docker system prune`

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

#### Traefik

- Generación de certificados autofirmados (con un archivo de configuración personalizado):
```bash
openssl req -x509 -nodes -days 825 -newkey rsa:4096 -keyout dev.key -out dev.pem -config openssl.cnf
```

Nota archivo openssl.cnf:
```ini
[ req ]
default_bits       = 4096
prompt             = no
default_md         = sha256
req_extensions     = req_ext
distinguished_name = dn

[ dn ]
C=UY
ST=Montevideo
L=Montevideo
O=Picudo Rojo Dev
OU=Development
CN=picudo-rojo-desarrollo.org

[ req_ext ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = picudo-rojo-desarrollo.org
DNS.2 = *.picudo-rojo-desarrollo.org
```

#### Poetry (deprecado en favor de UV)

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

#### UV

- Chequeo de versiones:
```bash
uv --version
uv lock --check
```

- Crear entorno virtual:
```bash
uv sync --all-extras # para instalar dependencias opcionales
```
> [!NOTE]  
> Por defecto, cuando se crea el entorno virtual, UV instala las dependencias en modo desarrollo. Para instalar en modo producción, se debe usar el flag `--no-editable --no-dev`.
> Las dependencias en desarrollo se encuentran en `[dependency-groups].dev`. Se puede manejar explícitamente con `[project.optional-dependencies]` definiendo un grupo `dev` y luego instalando con `uv sync --all-extras`.
> El comando `uv sync --all-extras`, a diferencia de `uv sync`, que por defecto es `uv sync --dev`, es que con el primero se instalan las dependencias "optional-dependencies" y el segundo instala las dependencias "dependency-groups".


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
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
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

- Chequear GPU:
```bash
sudo lshw -C display
```

- Chequear NVIDIA dentro de un contenedor Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu24.04 nvidia-smi
```

- Chequear drivers de NVIDIA:
<!-- https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/ -->
```bash
cat /proc/driver/nvidia/version
nvidia-smi
nvcc --version
```

- Instalar drivers de NVIDIA:
```bash
sudo ubuntu-drivers list --gpgpu # Lista los drivers disponibles
sudo ubuntu-drivers install --gpgpu nvidia:580-server # Instala el driver recomendado
sudo apt install nvidia-utils-580-server # Instala las utilidades de nvidia
```

- Verificar sistema operativo:
```bash
cat /etc/os-release
```

- Desactivar firewall (ufw):
```bash
sudo ufw disable
```

- Copiar todo el contenido al servidor remoto:
```bash
# Estando en uba-ceia-proy-final
# Hay que ver el tema de los permisos de los archivos copiados. Se puede crear la carpeta en el servidor y hacer sudo chown -R maso:maso /opt/ceia-proyecto-final antes de copiar.
# O darle permisos directamente con sudo chown -R maso:maso /opt
scp -r ./ceia-proyecto-final/* maso@192.168.0.3:/opt/ceia-proyecto-final/
```

- Copiar un módulo al servidor remoto:
```bash
scp -r .\modulo-mis-palmeras\mis-palmeras-landing-page\ .\modulo-mis-palmeras\mis-palmeras-maintenance\ maso@192.168.0.3:/opt/ceia-proyecto-final/modulo-mis-palmeras/
```

- Copiar desde el servidor (mejor rsync) un backup al entorno local:
```bash
# Nota instalación: Cuando se instala con choco no funciona directamente en PowerShell porque hay un tema con el .ssh (SSH Hell). Tiene que usarse el de rsync pero utiliza otro.
# Para solucionar se puede usar: rsync -avz --progress -e "C:\ProgramData\chocolatey\lib\rsync\tools\bin\ssh.exe" maso@192.168.0.3:/opt/ceia-proyecto-final/modulo-respaldo/backup-20251213 ./
# O setear la variable de entorno: $env:RSYNC_RSH = "C:\ProgramData\chocolatey\lib\rsync\tools\bin\ssh.exe" (o donde esté instalado ssh.exe de rsync)
$env:RSYNC_RSH = "C:\ProgramData\chocolatey\lib\rsync\tools\bin\ssh.exe"
rsync -avz --progress maso@192.168.0.3:/opt/ceia-proyecto-final/modulo-respaldo/backup-20251213 ./
```

- Copiar a producción con rsync (con filtros):
```bash
$env:RSYNC_RSH = "C:\ProgramData\chocolatey\lib\rsync\tools\bin\ssh.exe"
rsync -avz --progress --filter="merge .rsync-filters" ceia-proyecto-final/ maso@192.168.0.3:/ruta/destino/
```

- Opción mirror (elimina archivos en destino que no estén en origen):
```bash
# Estando en la carpeta ceia-proyecto-final
$env:RSYNC_RSH = "C:\ProgramData\chocolatey\lib\rsync\tools\bin\ssh.exe"
# ¡EVITAR -a! Usar las opciones rsync que NO PRESERVAN PERMISOS ni DUEÑO (-p, -o, -g), después es un lio con Docker y permisos.
# --delete: permite "mirror", elimina archivos en destino que no estén en origen.
rsync -rltDv --progress --delete --filter="merge .rsync-filters" ./ maso@192.168.0.3:/opt/ceia-proyecto-final/

# Finalmente copiar el backup para restaurar los datos:
$folder = "backup-20251213"   # Cambiar por el nombre del backup a restaurar
ssh maso@192.168.0.3 "mkdir -p /opt/ceia-proyecto-final/modulo-respaldo/${folder}"
scp -r ".\modulo-respaldo\$folder" maso@192.168.0.3:/opt/ceia-proyecto-final/modulo-respaldo/
```

##### Servicios

NOTA: Los servicios personalizados se deben copiar a `/etc/systemd/system/` para luego habilitarlos e iniciarlos.
En `/lib/systemd/system/` se encuentran los servicios del sistema.


- Listar todos los servicios:
```bash
sudo systemctl list-unit-files --type=service
```

- Listar servicios cargados:
```bash
sudo systemctl list-units --type=service
```

- Listar servicios en ejecución:
```bash
sudo systemctl list-units --type=service --state=running
```

- Listar servicios hablitados (en el arranque):
```bash
sudo systemctl list-unit-files --type=service --state=enabled
```

- Listar servicios personalizados:
```bash
ls /etc/systemd/system/*.service
```

- Ver estado de un servicio:
```bash
systemctl status ceia-proyecto-final.service
```

- Recargar systemd:
```bash
# Es necesario cuando se modifica un archivo de servicio.
sudo systemctl daemon-reload
```

- Reiniciar un servicio:
```bash
sudo systemctl restart ceia-proyecto-final.service
```

- Logs de un servicio:
```bash
journalctl -u ceia-docker-start.service -n 50 --no-pager
```

- Borrar el servicio:
```bash
sudo systemctl disable ceia-proyecto-final.service
sudo rm /etc/systemd/system/ceia-proyecto-final.service
sudo systemctl daemon-reload
```

- Crear el servicio:
```bash
sudo cp /opt/ceia-proyecto-final/vagrant-scripts/ceia-proyecto-final.service /etc/systemd/system/
sudo systemctl enable ceia-proyecto-final.service
sudo systemctl start ceia-proyecto-final.service
sudo systemctl daemon-reload
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