# <div align="center"><b> Mongo Local Deployment </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Mongo Local Deployment                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment en la local de la base de datos Mongo |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment local de Mongo. Esta base de datos se utiliza para almacenar todos los metadatos de las imágenes.

## Resolución

### Estructura de la base de datos

<!-- TODO -->

### Pasos para levantar la instancia

1. Simplemente hacer `docker compose up`.

> 📝<font color='Gray'>NOTA:</font> Esto levanta un contenedor con Mongo y otro contenedor con Mongo Express.

### Colecciones

Hay una coleccion importante:
- `imagenes`: Tiene la información de todas las imágenes.

### Aplicaciones

- MongoExpress $\rightarrow$ localhost:8081

### Comandos útiles

- Levantar el ambiente:

```bash
docker compose up
```

- Logs:

```bash
docker compose logs
```

### Backup y restore

Para realizar un backup y restore de la base de datos, se puede simplemente copiar la carpeta mongodb_data:

```bash
# Generar el nombre de la carpeta de respaldo con formato dinámico
$backupFolderName = "backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")

# Crear la carpeta de respaldo
New-Item -Path $backupFolderName -ItemType Directory

# Copiar la carpeta de datos de MongoDB
Copy-Item -Path "mongodb_data" -Destination $backupFolderName -Recurse
```

> 📝<font color='Gray'>NOTA:</font> La forma sugerida es utilizar mongodump y mongorestore.

Con mongodump y mongorestore:

`mongodump` $\rightarrow$
```bash
# Generar el nombre de la carpeta de respaldo con formato dinámico
$backupFolderName = "backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")

# Crear la carpeta de respaldo
New-Item -Path $backupFolderName -ItemType Directory

# Ejecutar un contenedor efímero para hacer el backup
docker run --rm --network main -v "${PWD}\$backupFolderName:/backup" mongo mongodump --host mongodb --out /backup
```

`mongorestore` $\rightarrow$
```bash
# Solicitar la carpeta de respaldo a restaurar
$backupFolderName = Read-Host "Ingrese el nombre de la carpeta de backup a restaurar"

# Verificar si la carpeta existe
if (Test-Path $backupFolderName) {
    # Ejecutar la restauración
    docker run --rm --network main -v "${PWD}\$backupFolderName:/backup" mongo mongorestore --host mongodb /backup

    Write-Output "Restauración completada desde: $backupFolderName"
} else {
    Write-Output "La carpeta especificada no existe."
}
```