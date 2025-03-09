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

### Backup y restore

Para realizar un backup y restore de la base de datos, se puede simplemente copiar la carpeta minio-data.
Sin embargo, lo recomendado es utilizar `mc mirror` para hacer un backup de los buckets.

Para realizar un backup:
```bash
# Generar el nombre de la carpeta de respaldo con formato dinámico
$backupFolderName = "backup-minio-" + (Get-Date -Format "yyyyMMdd-HHmmss")

# Crear la carpeta de respaldo
New-Item -Path $backupFolderName -ItemType Directory

# Ejecutar el backup con mc mirror
docker run --rm --network main -v "${PWD}\$backupFolderName:/backup" minio/mc \
  /bin/sh -c "
    mc alias set myminio http://minio:9000 $env:MINIO_ROOT_USER $env:MINIO_ROOT_PASSWORD &&
    mc mirror --overwrite --preserve myminio /backup
  "

Write-Output "Backup de MinIO guardado en: $backupFolderName"
```

Para restaurar:
```bash
# Solicitar la carpeta de respaldo a restaurar
$backupFolderName = Read-Host "Ingrese el nombre de la carpeta de backup de MinIO a restaurar"

# Verificar si la carpeta existe
if (Test-Path "$backupFolderName") {
    # Restaurar usando mc mirror
    docker run --rm --network main -v "${PWD}\$backupFolderName:/backup" minio/mc \
      /bin/sh -c "
        mc alias set myminio http://minio:9000 $env:MINIO_ROOT_USER $env:MINIO_ROOT_PASSWORD &&
        mc mirror --overwrite /backup myminio
      "

    Write-Output "Restauración completada desde: $backupFolderName"
} else {
    Write-Output "La carpeta especificada no existe."
}
```