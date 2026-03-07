# BACKEND FLOW TEST

Prueba de flujo del backend. Apps:
- `FastAPI`
- `Celery`
- `Redis`
- `Flower`
- `RabbitMQ`

# Flujo:

┌─────────────┐         ┌─────────┐         ┌──────────┐         ┌──────────┐
│   FastAPI   │ ------→ │RabbitMQ │ ------→ │  Worker  │ ------→ │  Redis   │
│  (envía)    │ mensaje │ (broker)│ escucha │ (ejecuta)│ resultado│ (almacena│
└─────────────┘         └─────────┘         └──────────┘         └──────────┘

# Comandos útiles:

## RABBITMQ:

- Listar colas: `docker exec -it <RABBITMQ_CONTAINER_NAME> rabbitmqctl list_queues`
- Listar conexiones: `docker exec -it <RABBITMQ_CONTAINER_NAME> rabbitmqctl list_connections`
