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

2. Remplazar los archivos del repositorio de CVAT con los de este repositorio:
3. Completar las variables faltantes de `.env`.
4. Hacer `docker compose up`.

> 📝<font color='Gray'>NOTA:</font> Para levantar la versión de DDNS se debe ejecutar lo siguiente: `docker compose -f docker-compose.yml -f docker-compose.local.yml -f docker-compose.local.minio.yml`

1. Luego, hay que agregar un super usuario:

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

> 📝<font color='Gray'>NOTA:</font> Se puede comprobar que esté levantada correctamente la herramienta con el siguiente healthcheck: `docker exec -t cvat_server python manage.py health_check`

#### Comandos útiles

- Chequear la configuración:

```bash
docker compose config
```

#### Sincronizar con bucket de Minio Local

- Para la sincronización con Minio Local, previamente se autorizó el host (se obtiene la IP de forma dinámica en el archivo `allow_minio.sh` y se la agrega como variable de entorno). Sino, no se conecta correctamente (tampoco resuelve `host.docker.internal`).

> 📝<font color='Gray'>NOTA:</font> El archivo `allow_minio.sh` está en la carpeta `test`, y lo que hace es obtener el IP de Minio para pasarle a Somkescreen, sino no funciona, dado que Smokescreen bloquea las peticiones de alguna forma (TODO: Averiguar más porque, dado que Smokescreen debe registrar las interacciones de CVAT con el exterior).

`allow_minio.sh` $\rightarrow$
```sh
# Same as allow_webhooks_receiver.sh, but for minio.
minio_ip_addr="$(getent hosts minio | head -1 | awk '{ print $1 }')"
export SMOKESCREEN_OPTS="$SMOKESCREEN_OPTS --allow-address=\"$minio_ip_addr\""
```

- Add Source Storage:

	```json
	{
			"Storage Type": "S3",
			"Storage Title": "picudo-rojo-minio-bucket",
			"Bucket Name": "picudo-rojo-bucket",
			"Bucket Prefix": "imagenes",
			"File Filter Regex": ".*(jpe?g|png|tiff)",
			"Region Name": "",
			"S3 Endpoint": "http://minio:9000", 
			"Access Key ID": "Clave acceso",
			"Secret Access Key": "Clave Secreta",
	}
	```

### Deploy via Ngrok

Para realizar el deploy via Ngrox, hay que tener el servicio (agente Ngrok) levantado, luego obtener la URL del frontend y colocarla en la variable `CVAT_HOST`. Luego, levantar la aplicación con suversión de Ngrok.

### Cloud deployment

#### Proveedor de nube

> ❗<font color='red'>IMPORTANTE:</font> El proveedor seleccionado es Scaleway, dado que brinda 75GB gratuitos en su object storage. Se seleccionó tanto para la instancia como para el storage.

#### Pasos para levantar la instancia

1. Lo primero es crear la instancia en <a href="https://www.scaleway.com/" target="_blank">Scaleway</a>. Para esto, se seleccionó una instancia con 8GB de RAM, que es lo recomendado.

> ❗<font color='red'>IMPORTANTE:</font> La instancia seleccionada tiene que ser AMD, dado que CVAT no soporta (no tiene imágenes de Docker) para la arquitectura ARM. En caso de que sea necesario ejecutar en dicha arquitectura, se puede emular con Quemu, o intentar construir las imágenes para dicha arquitectura (CVAT utiliza ffmpg y OpenCV, por lo que construir la imagen, con estas dependencias para esta arquitectura podría demorar mucho...)

<details>
  <summary>Detalles</summary>
  <ul>
    <li>
      <em>Generación de claves SSH:</em> Para la generación de claves SSH, primeramente hay que generar un par de claves (privada y
      pública) en la máquina actual. Esto se puede hacer con <code>ssh-keygen</code> o, en su defecto, con PuTTYgen
      también. Esto 
      Ejemplo: <code>ssh-keygen -t ed25519 -C "login" -Z aes256-gcm@openssh.com</code>. Luego, hay que subir estas
      claves a la "Organización" en <samp>Organization -> SSH Keys -> Add SSH key</samp>.
      Es de destacar, que esta clave se copia a la instancia una vez se crea la misma automáticamente.      
    </li>
    <li>
      <em>Generación de clave API Key:</em> Para crear una API Key es simplemente ir a <samp>Organization -> API Keys -> Generate APÏ Key</samp>. Hay que poner que se utilizará para acceder al bucket también.
    </li>
  </ul>
</details>

1. Una vez creada la instancia, hay que ingresar a la misma mediante SSH, instalar docker, clonar el repositorio, copiar los archivos "custom" necesarios y levantar docker-compose. Opcionalmente, anteriormente se puede utilizar CertBot para obtener los certificados.

> 📝<font color='Gray'>NOTA:</font> Todo este proceso se puede automatizar con https://docs.ansible.com/ Ansible. Inclusive, el ambiente (Linux) se puede simular utilizando Vagrant.

1. HTTPS: CVAT al utilizar Traefik, se puede configurar para que obtenga los certificados automáticamente. Para esto, simplemente hay que ejecutar docker con el override de https de Traefik, donde previamente hay que agregar la variable de entorno `ACME_EMAIL`:

```bash
docker compose -f docker-compose.yml -f docker-compose.https.yml up -d
```

#### Comandos útiles cloud

- Chequear la configuración:

```bash
docker compose config
```

- Ingresar en la instancia mediante SSH:
    - Tener la clave privada en .ssh (user)
    - Tener la configuración:

`config` $\rightarrow$
```bash
Host scaleway-instance
	HostName 51.15.236.232
	User root
	IdentityFile C:\Users\maso\.ssh\id_rsa_scaleway
```

`id_rsa_scaleway` $\rightarrow$

```txt
-----BEGIN OPENSSH PRIVATE KEY-----
tatatata
-----END OPENSSH PRIVATE KEY-----
```


- Ejecutar el comando: `ssh scaleway-instance`

- CORS bucket scaleway:
  - Obtener la configuración de CORS de un bucket:
  
```bash
aws s3api get-bucket-cors --bucket BUCKETNAME
```

- Setear la configuración CORS de un bucket:	

```bash
aws s3api put-bucket-cors --bucket picudo-rojo-bucket --cors-configuration file://cors.json --profile scaleway
```

<details>
<summary>Detalles</summary>

`cors.json` $\rightarrow$
```json
{
	"CORSRules": [
		{
			"AllowedOrigins": [
				"*"
			],
			"AllowedHeaders": [
				"*"
			],
			"AllowedMethods": [
				"GET",
				"HEAD",
				"POST",
				"PUT",
				"DELETE"
			],
			"MaxAgeSeconds": 3000,
			"ExposeHeaders": [
				"Etag"
			]
		}
	]
}
```

</details>
<br>

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

#### Sincronizar con bucket de Scaleway

- Add Source Storage:

	```json
	{
		"Storage Type": "S3",
		"Storage Title": "picudo-rojo-sw-bucket",
		"Bucket Name": "picudo-rojo-bucket",
		"Bucket Prefix": "imagenes",
		"Region Name": "",
		"S3 Endpoint": "https://s3.nl-ams.scw.cloud",
		"Access Key ID": "Clave acceso",
		"Secret Access Key": "Clave Secreta",
	}
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