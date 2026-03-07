# System imports
import os

# Local imports

# Third-party imports
from celery import Celery

RABBITMQ_VIRTUAL_HOST = os.getenv("RABBITMQ_DEFAULT_VHOST", "ai_vhost")
RABBITMQ_USER = os.getenv("RABBITMQ_DEFAULT_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_DEFAULT_PASS", "guest")
RABBITMQ_SERVER = os.getenv("RABBITMQ_SERVER", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    f"pyamqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_SERVER}:{RABBITMQ_PORT}/{RABBITMQ_VIRTUAL_HOST}",
)

REDIS_USER = os.getenv("REDIS_USER", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redispassword")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
)

CELERY_MAIN_MODULE_NAME = os.getenv("CELERY_MAIN_MODULE_NAME", "tasks")

print(f"Using broker URL: {CELERY_BROKER_URL}")
print(f"Using backend URL: {CELERY_RESULT_BACKEND}")

app = Celery(CELERY_MAIN_MODULE_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND, include=["app.tasks"])

app.conf.update(
    # result_expires=3600,  # Expirar resultados después de 1 hora
    # task_serializer="json", # Usar JSON para serializar las tareas, es más seguro que pickle
    # result_serializer="json", # Usar JSON para serializar los resultados, es más seguro que pickle
    # accept_content=["json"], # Aceptar solo contenido JSON, para evitar problemas de seguridad con pickle
    task_track_started=True,  # Para poder ver el estado "STARTED" de las tareas en el backend
)


if __name__ == "__main__":
    app.start()
