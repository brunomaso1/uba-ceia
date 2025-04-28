# <div align="center"><b> Proyecto final CEIA </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Sistema de monitoreo de Rhynchophorus ferrugineus en palmeras de montevideo                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Presentación del trabajo final de CEIA que consiste en un sistema de monitoreo de la plaga del picudo rojo |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

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

#### Linux

- Verificar espacio en disco:
```bash
df -h
```

- Verificar uso de memoria:
```bash
free -h
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