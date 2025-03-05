# <div align="center"><b> Minio Local Deployment </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Minio Local Deployment                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment en la local de la Minio |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment local de Minio.

## Resolución

### Pasos para levantar la instancia

1. Simplemente hacer `docker compose up`.

> 📝<font color='Gray'>NOTA:</font> Esto levanta un contenedor con Minio y otro contenedor con Prometheus.

### Estructura de directorios

<pre>
picudo-rojo-bucket/
├── anotaciones
├── imagenes
├── metadatos
├── patches
└── pre-anotaciones
</pre>

### Aplicaciones

> 📝<font color='Gray'>NOTA:</font> Se puede poner minio en el hosts para acceder desde su sinónimo: `127.0.0.1 minio` (en el hosts).

- Minio $\rightarrow$ localhost:9000 | minio:9000
- Prometheus $\rightarrow$ localhost:9090

### Comandos útiles

- Levantar el ambiente:

```bash
docker compose up
```

- Logs:

```bash
docker compose logs
```

### Minio TLS

Minio soporta nativamente TLS, simplemente hay que poner los certificados en una carpeta específica dentro del contenedor de Minio. Se puede ver más documentación en la siguiente URL: https://min.io/docs/minio/linux/operations/network-encryption.html

En la documentación se dice que los certificados hay que copiarlos a `${HOME}/.minio/certs`, lo que deriva en `/root/.minio/certs`.

Para generar los certificados en desarrollo, se puede utilizar `certgen`, del siguiente repositorio https://github.com/minio/certgen (simplemente copiar el exe y ejecutarlo):

```bash
certgen.exe -host "localhost,minio"
```

Luego, para levantar Minio con TLS:

```bash
docker compose -f docker-compose.yml -f docker-compose.https.yml up
```

Donde este archivo `docker-compose.https.yml` monta los certificados en el contenedor de Minio.

> 📝<font color='Gray'>NOTA:</font> Se puede automatizar este proceso con CertBot y utilizar los certificados de Let's Encrpyt.