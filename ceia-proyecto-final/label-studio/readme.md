# <div align="center"><b> Label Studio Cloud Deployment </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Label Studio Colud Deployment                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment en la nube de la herramienta Label Studio |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment de Label Studio en la nube.

## Resolución

### Proveedor de nube

> ❗<font color='red'>IMPORTANTE:</font> El proveedor seleccionado es Scaleway, dado que brinda 75GB gratuitos en su object storage. Se seleccionó tanto para la instancia como para el storage.

### Pasos para levantar la instancia

1. Lo primero es crear la instancia en <a href="https://www.scaleway.com/" target="_blank">Scaleway</a>. Para esto, se seleccionó una instancia con 8GB de RAM, que es lo recomendado.

<details>
  <summary>Detalles</summary>
  <ul>
    <li>
      <em>Generación de claves SSH:</em> Para la generación de claves SSH, primeramente hay que generar un par de claves (privada y
      pública) en la máquina actual. Esto se puede hacer con <code>ssh-keygen</code> o, en su defecto, con PuTTYgen
      también.
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

### Limitaciones de Label Studio

- La gestión de usuarios (User Management) no se encuentra en la versión "Comunity". Los usuarios son todos administradores.

### Links auxiliares

- Github de Label Studio: https://github.com/HumanSignal/label-studio

### Comandos útiles

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
	HostName XXXXXXXXX # Completar con IP del host
	User root
	IdentityFile C:\ruta\a\.ssh\id_rsa_scaleway
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

```json
# cors.json
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

- Clonar el repositorio de Label Studio:

```bash
cd ../opt/
# git clone https://github.com/HumanSignal/label-studio # Versión develop
# git clone --depth 1 --branch <tag_name> <repo_url> # Clonar un tag específico
git clone --depth 1 --branch 1.15.0 https://github.com/HumanSignal/label-studio.git # Versión v1.15.0 - LTS
```

- Copiar archivos:
  - Copiar un archivo en especial:

```bash
scp -i "C:\Users\maso\.ssh\id_rsa_scaleway" "E:\ruta\archivo\a\copiar\1" "E:\ruta\archivo\a\copiar\2" root@IP_INSTANCIA_CLOUD:../opt/label-studio/
```

  - Copiar todos los archivos en el directorio actual:
```bash
scp -i "C:\Users\maso\.ssh\id_rsa_scaleway" -r . root@IP_INSTANCIA_CLOUD:../opt/label-studio/
```

- Certificados:
  1. Levantar el `docker-compose.cert.yml`, que tiene un Nginx y CertBot, para firmar con Let's Encpryt.

		<details>
			<summary>Detalles</summary>
			<ol>
				<li>
					<p>Ejecutar CertBot en formato test:</p>
					<pre><code>docker compose -f docker-compose.cert.yml run --rm certbot certonly --webroot --webroot-path /var/www/certbot/ --dry-run -d DNS_INSTANCIA_CLOUD</code></pre>
				</li>
				<li>
					<p>Real (real):</p>
					<pre><code>docker compose -f docker-compose.cert.yml run --rm certbot certonly --webroot --webroot-path /var/www/certbot/ -d DNS_INSTANCIA_CLOUD</code></pre>
				</li>
				<li>
					<p>Copiar los certificados generados (también se puede montar la carpeta):</p>
					<pre><code># Desde tu host, copia los certs reales a la carpeta que montas en el contenedor
					cp /opt/label-studio/certbot/conf/live/DNS_INSTANCIA_CLOUD/fullchain.pem /opt/label-studio/deploy/nginx/certs/cert.pem
					cp /opt/label-studio/certbot/conf/live/DNS_INSTANCIA_CLOUD/privkey.pem /opt/label-studio/deploy/nginx/certs/cert.key
					</code></pre>
				</li>
			</ol>
		</details>

- Levantar label-studio con variables de entorno y el usuario del contenedor=usuarioLogeado:

```bash
UID_GID="$(id -u):$(id -g)" docker compose up
```

- Ingresar a un contenedor:

```bash
docker compose exec app bash
```

### Sincronizar con bucket de Scaleway

- Add Source Storage:

	```json
	{
		"Storage Type": "S3",
		"Storage Title": "picudo-rojo-sw-bucket",
		"Bucket Name": "picudo-rojo-bucket",
		"Bucket Prefix": "imagenes",
		"File Filter Regex": ".*(jpe?g|png|tiff)",
		"Region Name": "",
		"S3 Endpoint": "ENDPOINT_S3",
		"Access Key ID": "Clave acceso",
		"Secret Access Key": "Clave Secreta",
		"Session Token": "",
		"Treat every bucket": true,
		"Recursive Scan": false,
		"Use pre-signed URLs": false
	}
	```

- Add Target Storage:

	```json
	{
		"Storage Type": "S3",
		"Storage Title": "picudo-rojo-sw-bucket",
		"Bucket Name": "picudo-rojo-bucket",
		"Bucket Prefix": "anotaciones",
		"Region Name": "",
		"S3 Endpoint": "ENDPOINT_S3",
		"SSE KMS Key ID": "",
		"Access Key ID": "Clave acceso",
		"Secret Access Key": "Clave Secreta",
		"Session Token": "",
		"Can delete objects from storage": true
	}
	```