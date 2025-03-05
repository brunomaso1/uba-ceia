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
