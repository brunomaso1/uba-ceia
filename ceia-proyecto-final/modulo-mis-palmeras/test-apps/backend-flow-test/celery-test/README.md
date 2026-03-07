# CELERY TEST

Proyecto de prueba de Celery.

## Requisitos

1. RabbitMQ: Asegúrate de tener RabbitMQ instalado y en funcionamiento en tu máquina. Puedes descargarlo desde [aquí](https://www.rabbitmq.com/download.html).
2. Celery: Instala Celery usando pip:

```bash
pip install celery
```

## Comandos útiles:

- Correr el worker de Celery:

```bash
celery -A app.main worker --loglevel=info
```

> [!NOTE]
> En Windows hay que usar `celery -A app.main worker --loglevel=info --pool=solo` para evitar problemas con los procesos. O se puede utilizar `gevent` como pool, pero eso requiere instalar `gevent (uv add gevent)`, en ese caso, el comando sería: `celery -A app.main worker --loglevel=info --pool=gevent`. En Linux y MacOS no es necesario especificar el pool, ya que el pool por defecto funciona correctamente.
