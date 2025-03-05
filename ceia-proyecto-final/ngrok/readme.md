# <div align="center"><b> Ngrok Deployment </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Ngrok Deployment                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment local de la herramienta Label Studio |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment de Ngrok.

## Resolución

### Pasos para levantar la instancia

1. Simplemente hacer `docker compose up`. 

### Limitaciones de Label Studio

Label Studio tiene el problema que las URL deben ser pre-signed, y al exponer con Ngrok, queda todo por HTTPS. Esto puede dar errores de Mixed Content. Para solucionar esto, hay que exponer también el Minio con Ngrok. La siguiente configuración expone ambos servicios:

```yaml
version: 3
agent:
  authtoken: <TOKEN>
tunnels:
  label_studio_tunnel:
    proto: http
    domain: picudo-rojo-imm.ngrok.io
    addr: http://host.docker.internal:8080
  minio_tunnel:
    proto: http
    addr: http://host.docker.internal:9000
```

Donde la URL de Minio hay que buscarla en el Dashboard de Ngrok, y poner esta dirección en el attach storage de Label Studio. Todo esto porque ya se tiene un dominio reservado, pero en caso contrario, hay que cambiar la configuración de `domain` y buscar los dominios dinámicos en el Dashboard de Ngrok.

También es destacar que las aplicaciones (CVAT: traefik y Label Studio: nginx) deben estar en el puerto 80. Pero esto se puede adaptar en el archivo ngrok-config.yml. Si se resuelve el nombre de dominio (están dentro de la misma red) se puede acceder a CVAT con http://traefik:8080 y a Label Studio con http://nginx:8080.