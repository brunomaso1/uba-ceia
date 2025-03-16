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

### Comandos útiles

- Chequear dirección IP de máquina virtual:

```bash
ip addr show # Dentro de la máquina virtual
```

#### Firewall linux
https://www.digitalocean.com/community/tutorials/how-to-set-up-a-firewall-with-ufw-on-ubuntu#step-2-setting-up-default-policies

- Chequear si el firewall está activo:

```bash
sudo ufw status # Dentro de la máquina virtual
```

- Desactivar el firewall:

```bash
sudo ufw disable # Dentro de la máquina virtual
```
- Activar el firewall:

```bash
sudo ufw enable # Dentro de la máquina virtual
```

- Permitir el tráfico a través del puerto 80:

```bash
sudo ufw allow 80 # Dentro de la máquina virtual
```

#### Obtener IP pública

- IP pública de la máquina virtual:

```bash
curl ifconfig.me # Dentro de la máquina virtual
```

#### Redes de la máquina virtual

- Ver las redes de la máquina virtual:

```bash
ip addr which # Dentro de la máquina virtual
```

- Ver el ruteo de la máquina virtual:

```bash
ip route show # Dentro de la máquina virtual
```

> 📝<font color='Gray'>NOTA:</font> La configuración de la ruta debe tener a la interfaz que se le asigna a vagrant (eth1) como prioridad:
> `default via 192.168.0.1 dev eth1 proto dhcp metric 50`

#### Nginx

- Verificar si Nginx está activo:

```bash
sudo systemctl status nginx # Dentro de la máquina virtual
```

- Reiniciar Nginx:

```bash
sudo nginx -s reload # Dentro de la máquina virtual
```

- Verificar la configuración de Nginx:

```bash
sudo nginx -t # Dentro de la máquina virtual
```

- Ver los logs de Nginx:

```bash
cat /var/log/nginx/access.log # Dentro de la máquina virtual
cat /var/log/nginx/error.log # Dentro de la máquina virtual
```

- Cambiar configuración por defecto de Nginx:

```bash
sudo nano /etc/nginx/sites-available/default # Dentro de la máquina virtual
```

#### Debugging

- tcpdump:

```bash
sudo tcpdump -i any port 80 # Dentro de la máquina virtual
```

- tcpdump (solo interfaz eth1):

```bash
sudo tcpdump -i eth1 port 80 # Dentro de la máquina virtual
```

### Errores conocidos

Si aparece `vagrant@127.0.0.1: Permission denied (publickey)` al intentar acceder a la máquina virtual, ejecutar el siguiente comando:

```bash
$Env:VAGRANT_PREFER_SYSTEM_BIN += 0
```
