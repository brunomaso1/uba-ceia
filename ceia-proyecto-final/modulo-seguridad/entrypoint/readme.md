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

## Certificados TLS

### Descripción General

El proxy reverso Traefik implementa soporte para HTTPS mediante certificados TLS (Transport Layer Security). Se utilizan certificados autofirmados para el ambiente de desarrollo y certificados válidos para producción. Los certificados se encuentran en la carpeta `certs/` y se montan como volúmenes de solo lectura en el contenedor de Traefik.

### Estructura de la Carpeta `certs/`

```
certs/
├── dev.key          # Clave privada para desarrollo (sin encriptar)
├── dev.pem          # Certificado autofirmado para desarrollo
├── prod.key         # Clave privada para producción
└── prod.pem         # Certificado válido para producción
```

- **dev.key / dev.pem**: Certificados autofirmados para el ambiente de desarrollo. Generados con OpenSSL usando la configuración de `openssl.cnf`.
- **prod.key / prod.pem**: Certificados válidos para el ambiente de producción. Deben ser adquiridos de una Autoridad Certificadora (CA) o generados por servicios como Let's Encrypt.

### Generación de Certificados Autofirmados (Desarrollo)

Para generar nuevos certificados autofirmados para desarrollo, se utiliza el archivo de configuración `openssl.cnf`:

```bash
# Generar certificado autofirmado válido por 365 días
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout certs/dev.key \
  -out certs/dev.pem \
  -config openssl.cnf
```

### Certificados de Cloudflare Origin CA

Para ambientes locales o servidores "home" se recomienda utilizar **Cloudflare Origin CA**, que proporciona certificados válidos de forma gratuita para servidores origen:

#### Ventajas

- ✅ Certificados válidos sin advertencias del navegador
- ✅ Válidos para servidores locales/home sin estar expuesto a internet público
- ✅ Emitidos por Cloudflare, confiables
- ✅ Gratuitos
- ✅ Fácil generación desde el panel de Cloudflare

#### Proceso

1. Ir al panel de Cloudflare → SSL/TLS → Origin Server
2. Crear un nuevo certificado Origin CA seleccionando:
   - Dominio: `picudo-rojo-desarrollo.org` (o el dominio usado)
   - Subdominios wildcard: `*.picudo-rojo-desarrollo.org`
   - Validez: 15 años (máximo)
3. Copiar la clave privada y el certificado generados
4. Reemplazar el contenido de `certs/dev.key` y `certs/dev.pem` con los valores de Cloudflare
5. Reiniciar Traefik: `docker compose restart traefik-entrypoint`

#### Configuración Requerida en Cloudflare

En el panel de Cloudflare hay que configurar:

- **SSL/TLS Encryption Mode**: Full (strict) - para garantizar conexión segura entre Cloudflare y el origen
- **Origin server dirección**: IP o dominio del servidor con Traefik

Con esta configuración, el navegador confía en el certificado aunque el servidor esté en la red local, y Cloudflare valida la conexión hacia el origen.

### Configuración de Traefik

#### Global (compose.yml)

El contenedor de Traefik se configura con dos entry points:

```yaml
command:
  - "--entrypoints.web.address=:80"           # HTTP
  - "--entrypoints.websecure.address=:443"    # HTTPS
```

Y monta los certificados como volumen:

```yaml
volumes:
  - ./certs:/certs:ro  # Montaje de solo lectura (read-only)
```

#### Configuración Dinámico (traefik_dynamic_conf_dev.yaml / traefik_dynamic_conf_prod.yaml)

La sección `tls` en los archivos de configuración dinámica especifica qué certificados usar:

```yaml
tls:
  stores:
    default:
      defaultCertificate:
        certFile: /certs/dev.pem    # O /certs/prod.pem para producción
        keyFile: /certs/dev.key     # O /certs/prod.key para producción
```

### Habilitación de TLS en Rutas

Para habilitar HTTPS en una ruta específica, se agrega la opción `tls: {}` en el router HTTPS correspondiente:

```yaml
# Router HTTP (redirige a HTTPS)
cvat-router-http:
  rule: "Host(`cvat.picudo-rojo-desarrollo.org`)"
  entryPoints:
    - web
  service: noop-service
  middlewares:
    - redirect-to-https

# Router HTTPS
cvat-router-https:
  rule: "Host(`cvat.picudo-rojo-desarrollo.org`)"
  entryPoints:
    - websecure
  tls: {}  # ← Habilita TLS para esta ruta
  service: cvat-service
```

### Pattern de Configuración Recomendado

Para cada aplicación que se exponga, se deben crear dos routers:

1. **Router HTTP**: Redirige automáticamente a HTTPS usando el middleware `redirect-to-https`
2. **Router HTTPS**: Usa `tls: {}` para habilitar HTTPS con los certificados configurados

Este patrón asegura que:
- Todas las peticiones HTTP se redirijen a HTTPS
- Se utiliza el certificado configurado en `tls.stores.default`
- Se proporciona una experiencia segura al usuario

### Cambio entre Ambientes

Para cambiar entre desarrollo y producción, se configura en el archivo `.env` o se ejecuta docker compose con:

```bash
# Usar configuración de desarrollo
docker compose up

# O especificar archivo de configuración dinámico
DYNAMIC_CONF_FILE=./traefik_dynamic_conf_prod.yaml docker compose up
```

### Certificados de Producción

Para producción, se recomienda:

1. **Adquirir certificados válidos** de una Autoridad Certificadora o servicio como Let's Encrypt
2. **Reemplazar los archivos** `prod.key` y `prod.pem` con los certificados reales
3. **Asegurar permisos** apropiados en los archivos (read-only para el contenedor)
4. **Usar configuración dinámica** con validación periódica de expiración

### Validación de Certificados

Para verificar la información de un certificado:

```bash
# Ver detalles del certificado
openssl x509 -in certs/dev.pem -text -noout

# Verificar que la clave privada coincida
openssl x509 -noout -modulus -in certs/dev.pem | openssl md5
openssl rsa -noout -modulus -in certs/dev.key | openssl md5
```

### Resolución de Problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| Error "certificate not found" | Certificados no generados | Ejecutar comando de generación arriba |
| Certificado expirado | Certificado autofirmado pasó 365 días | Regenerar certificado |
| Certificado no coincide | Clave privada y certificado no están relacionados | Regenerar ambos archivos juntos |
| Browser avisa certificado no válido | Certificado autofirmado en desarrollo | Es normal, agregar excepción en el browser |