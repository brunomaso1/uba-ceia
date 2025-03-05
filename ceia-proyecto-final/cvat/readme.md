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