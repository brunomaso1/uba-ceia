# <div align="center"><b> CVAT Deployment </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | CVAT Deployment                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment de la herramienta CVAT |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment de CVAT.

## Resolución

### Links auxiliares

- Github de CVAT: https://github.com/cvat-ai/cvat

### Local deployment

#### Pasos para levantar la instancia

1. Clonar este repositorio y el repositorio de CVAT:

```bash
git clone --depth 1 --branch v2.28.0 https://github.com/cvat-ai/cvat.git # Versión v2.28.0
```

2. Remplazar los archivos del repositorio de CVAT con los de este repositorio.
3. Completar las variables faltantes en los `.env`.
4. Hacer `docker compose up` según el ambiente que se desee levantar:

```bash
# Levantar el ambiente local (sin HTTPS)
docker compose --env-file .env.dev -f docker-compose.yml -f docker-compose.custom.yml up -d
```

```bash
# Levantar el ambiente de producción (con HTTPS)
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.custom.yml up -d
```

1. Luego, hay que agregar un super usuario (sino se utiliza LDAP):

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

> 📝<font color='Gray'>NOTA:</font> Se puede comprobar que esté levantada correctamente la herramienta con el siguiente healthcheck: `docker exec -t cvat_server python manage.py health_check`

#### Comandos útiles

- Chequear la configuración:

```bash
docker compose config
```

- Actualizar linux:

```bash
sudo apt update -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y
```

- Instalar docker y docker compose (https://docs.docker.com/engine/install/ubuntu/):
  - Agregar claves al registro:
		
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
	"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
	$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
	sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

  - Instalar docker y docker compose:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
```

- Clonar el repositorio de CVAT:

```bash
cd ../opt/
# git clone --depth 1 --branch <tag_name> <repo_url> # Clonar un tag específico
git clone --depth 1 --branch v2.28.0 https://github.com/cvat-ai/cvat.git # Versión v2.28.0 - LTS
```

- Copiar archivos:
	- Copiar todos los archivos en el directorio actual:
```bash
scp -i "C:\Users\maso\.ssh\id_rsa_scaleway" -r . root@51.15.236.232:../opt/cvat/ # Poner la IP de la instancia
```

- Ingresar a un contenedor:

```bash
docker compose exec app bash
```

### Respaldo

https://docs.cvat.ai/docs/administration/advanced/backup_guide/

Para respaldar los datos de CVAT, hay que respaldar varios volumenes de Docker. Para esto, se puede utilizar `docker-compose` y `docker` para realizar el backup de los datos.
Con `docker-compose` habría que modificar los volúmenes para que se monten en el fileserver local y ahí realizar la copia.

Con `docker` se puede realizar un backup de los volúmenes de la siguiente forma:

> 📝<font color='Gray'>NOTA:</font> Hay que detener los contenedores antes de realizar el backup: `docker compose stop` (incluir los flags ingresados al iniciar los contenedores)

> 🔮 <em><font color='violet'>Función auxiliar:</font></em>
> 
> Ejecución en bash:
> ```bash
> mkdir backup
>
> docker run --rm --name temp_backup --volumes-from cvat_db -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data
> docker run --rm --name temp_backup --volumes-from cvat_server -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data
> docker run --rm --name temp_backup --volumes-from cvat_clickhouse -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse
> ```

<details>
  <summary>Script de Respaldo</summary>
  <code>

	#!/bin/bash

	# Generar el nombre de la carpeta de respaldo con formato dinámico
	backupFolderName="backup-$(date +'%Y%m%d-%H%M%S')"

	# Crear la carpeta de respaldo
	mkdir -p "$backupFolderName"

	# Ejecutar los comandos de Docker
	docker run --rm --name temp_backup --volumes-from cvat_db -v "$(pwd)/$backupFolderName:/backup" ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data

	docker run --rm --name temp_backup --volumes-from cvat_server -v "$(pwd)/$backupFolderName:/backup" ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data

	docker run --rm --name temp_backup --volumes-from cvat_clickhouse -v "$(pwd)/$backupFolderName:/backup" ubuntu tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse

  </code>
</details>

Ejecución en PowerShell:
```bash
# Generar el nombre de la carpeta de respaldo con formato dinámico
$backupFolderName = "backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")

# Crear la carpeta de respaldo
New-Item -Path $backupFolderName -ItemType Directory

# Generar la ruta de la carpeta de respaldo
$backupPath = Join-Path -Path (pwd) -ChildPath $backupFolderName

# Ejecutar los comandos de Docker
docker run --rm --name temp_backup --volumes-from cvat_db -v "$($backupPath):/backup" ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data
docker run --rm --name temp_backup --volumes-from cvat_server -v "$($backupPath):/backup" ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data
docker run --rm --name temp_backup --volumes-from cvat_clickhouse -v "$($backupPath):/backup" ubuntu tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse
```

Restauración de los datos:
```bash
cd <path_to_backup_folder>
docker run --rm --name temp_backup --volumes-from cvat_db -v $(pwd):/backup ubuntu bash -c "cd /var/lib/postgresql/data && tar -xvf /backup/cvat_db.tar.gz --strip 4"
docker run --rm --name temp_backup --volumes-from cvat_server -v $(pwd):/backup ubuntu bash -c "cd /home/django/data && tar -xvf /backup/cvat_data.tar.gz --strip 3"
docker run --rm --name temp_backup --volumes-from cvat_clickhouse -v $(pwd):/backup ubuntu bash -c "cd /var/lib/clickhouse && tar -xvf /backup/cvat_events_db.tar.gz --strip 3"
```

#### Backup de la base de datos

Lo ideal es utilizar `pg_dump` para realizar el backup de la base de datos. Para esto, se puede utilizar el siguiente comando:

`pg_dump` $\rightarrow$
```bash
# Ejecutar un contenedor efímero para hacer el backup
docker run --rm --network picudo-rojo-project_main -v "($backupPath):/backup" postgres:15-alpine pg_dump -h cvat_db -U postgres -d cvat -F c -f /backup/backup.sql
```

`pg_restore` $\rightarrow$
```bash
# Solicitar la carpeta de respaldo a restaurar
$backupFolderName = Read-Host "Ingrese el nombre de la carpeta de backup a restaurar"

# Verificar si la carpeta existe
if (Test-Path "$backupFolderName\backup.sql") {
    # Ejecutar la restauración
    docker run --rm --network main -v "${PWD}\$backupFolderName:/backup" postgres pg_restore -h cvat_db -U postgres -d postgres -c /backup/backup.sql

    Write-Output "Restauración completada desde: $backupFolderName\backup.sql"
} else {
    Write-Output "No se encontró el archivo de backup en la carpeta especificada."
}
```

Sin embargo, si se detienen los contendores se puede realizar el backup copiando los archivos del volumen montando con el código anterior.