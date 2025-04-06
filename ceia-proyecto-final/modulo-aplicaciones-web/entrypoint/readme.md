# <div align="center"><b> Entrypoint </b></div>

<div align="center">✨Datos del proyecto:✨</div>

<p></p>

<div align="center">

| Subtitulo       | Entrypoint                                         |
| --------------- | --------------------------------------------------------------------- |
| **Descrpción**  | Documentación de deployment del proxy reverso para exponer las herramientas a internet |
| **Integrantes** | Bruno Masoller (brunomaso1@gmail.com)                                 |

</div>

## Consinga

El objetivo es centralizar la documentación relacionada con el deployment del proxy reverso.

## Resolución

Se utiliza un proxy reverso para exponer las herramientas a internet. Se utiliza el servicio de [Cloudflare](https://www.cloudflare.com/) para manejar el tráfico de red mediante un dominio especifico. El objetivo es, dado un subdominio, redirigir el tráfico hacia un puerto específico de una máquina virtual (docker compose).

Estas configuraciones pueden verse en el archivo traefik_dynamic_conf.yml. Hay un ejemplo también del mismo caso de uso utilizando Nginx.

Simplemente hay que levanter el proxy con: `docker compose -f docker-compose.traefik.yml up`.

### Configuración de Traefik

Para que funcione el ambiente de desarrollo hay que configurar en el hosts el dominio picudo-rojo-desarrollo.org. Debe apuntar a la IP de la máquina de desarrollo. Este archivo se encuentra en windows en `C:\Windows\System32\drivers\etc\hosts`.

Ejemplo de configuración:

```bash
192.168.0.175  picudo-rojo-desarrollo.org cvat.picudo-rojo-desarrollo.org
```